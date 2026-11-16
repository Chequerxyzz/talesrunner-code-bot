#!/usr/bin/env python3
"""
TalesRunner live-code watcher.

Watches a YouTube channel's live stream, grabs a frame every few seconds,
reads on-screen text with OCR, picks out item codes and posts them to a
Discord webhook.

Commands:
  python watcher.py should-run          -> prints "yes"/"no" (live now, or inside schedule window)
  python watcher.py watch               -> watch live streams until deadline
  python watcher.py rescan              -> rescan finished streams (VOD) for codes that were missed
  python watcher.py vod URL [--post]    -> scan one past video (test). Prints codes; --post sends to Discord
  python watcher.py image FILE          -> OCR one image (test)
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------- settings
CHANNEL = os.environ.get("CHANNEL_URL", "https://www.youtube.com/@thehof.talesrunner").rstrip("/")
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
COOKIES = os.environ.get("YT_COOKIES_FILE", "")
INTERVAL = int(os.environ.get("FRAME_INTERVAL", "15"))      # seconds between frames
CONFIRM = int(os.environ.get("CONFIRM_FRAMES", "2"))        # frames a code must appear in before posting
STATE_DIR = os.environ.get("STATE_DIR", "state")
GIT_PERSIST = os.environ.get("GIT_PERSIST") == "1"
# Schedule window in Thai time (streams 14:00-~24:00). Outside it the watcher exits when nothing is live.
WINDOW_START = os.environ.get("WINDOW_START", "13:30")
WINDOW_END = os.environ.get("WINDOW_END", "00:45")
ICT = timezone(timedelta(hours=7))

# ---------------------------------------------------------------- helpers


def log(*a):
    print(datetime.now(ICT).strftime("[%H:%M:%S]"), *a, flush=True)


def hms(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_window(now=None):
    now = now or datetime.now(ICT)
    m = now.hour * 60 + now.minute
    s, e = _mins(WINDOW_START), _mins(WINDOW_END)
    return s <= m <= e if s <= e else (m >= s or m <= e)


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # bundled static ffmpeg (works on Windows too)
    return imageio_ffmpeg.get_ffmpeg_exe()


def ytdlp_cmd(need_ffmpeg=False):
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-progress"]
    if need_ffmpeg:
        cmd += ["--ffmpeg-location", ffmpeg_exe()]
    if not shutil.which("deno") and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    if COOKIES and os.path.exists(COOKIES) and os.path.getsize(COOKIES) > 0:
        cmd += ["--cookies", COOKIES]
    return cmd


# ---------------------------------------------------------------- code detection
# Codes seen so far: VASE9R7K1J6M, ZEAL4C5Z9V2H, COMB8N1Q3M7T
#   = 4 letters, then digit/letter alternating 4 times (12 chars).
CODE_RE = re.compile(r"[A-Z]{4}(?:[0-9][A-Z]){4}")
TO_DIGIT = str.maketrans("OQDILZSBGTA", "00011258674")
TO_LETTER = str.maketrans("01524867", "OISZABGT")


def _fix(window):
    """Correct common OCR mix-ups using the known position pattern."""
    out = []
    for i, ch in enumerate(window):
        if i < 4 or i % 2 == 1:
            ch = ch.translate(TO_LETTER)
            if not ch.isalpha():
                return None
        else:
            ch = ch.translate(TO_DIGIT)
            if not ch.isdigit():
                return None
        out.append(ch)
    return "".join(out)


def extract_codes(lines):
    found = set()
    for line in lines:
        up = line.upper()
        # exact matches, also with spaces removed (OCR sometimes splits a code)
        for text in (up, re.sub(r"\s+", "", up)):
            for tok in re.split(r"[^A-Z0-9]+", text):
                found.update(CODE_RE.findall(tok))
        # fuzzy: tokens about the length of a code
        for tok in re.split(r"[^A-Z0-9]+", up):
            if 12 <= len(tok) <= 14 and not CODE_RE.search(tok):
                for i in range(len(tok) - 11):
                    c = _fix(tok[i:i + 12])
                    if c:
                        found.add(c)
    return found


_ocr = None


def ocr_lines(path):
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    result, _ = _ocr(path)
    return [r[1] for r in (result or [])]


# ---------------------------------------------------------------- state (one small file per item -> no git conflicts)


def _load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def code_file(code):
    return os.path.join(STATE_DIR, "codes", f"{code}.json")


def video_file(vid):
    return os.path.join(STATE_DIR, "videos", f"{vid}.json")


def persist(msg):
    if not GIT_PERSIST:
        return
    try:
        subprocess.run(["git", "add", STATE_DIR], check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            return
        subprocess.run(["git", "commit", "-q", "-m", msg], check=True)
        for _ in range(5):
            subprocess.run(["git", "pull", "-q", "--rebase", "-X", "theirs"])
            if subprocess.run(["git", "push", "-q"]).returncode == 0:
                return
            time.sleep(5)
        log("WARN: could not push state")
    except Exception as e:  # never let state saving kill the watcher
        log("WARN: persist failed:", e)


# ---------------------------------------------------------------- discord


def discord(content=None, embed=None, image=None):
    if not WEBHOOK:
        log("[discord-dry-run]", content or "", json.dumps(embed, ensure_ascii=False) if embed else "")
        return
    payload = {"content": content, "allowed_mentions": {"parse": []}}
    if embed:
        if image:
            embed["image"] = {"url": "attachment://frame.jpg"}
        payload["embeds"] = [embed]
    for _ in range(4):
        try:
            if image and os.path.exists(image):
                with open(image, "rb") as f:
                    r = requests.post(WEBHOOK, data={"payload_json": json.dumps(payload)},
                                      files={"files[0]": ("frame.jpg", f, "image/jpeg")}, timeout=30)
            else:
                r = requests.post(WEBHOOK, json=payload, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
                continue
            r.raise_for_status()
            return
        except Exception as e:
            log("discord error:", e)
            time.sleep(3)


_warned_block = False


def warn_blocked(err):
    global _warned_block
    if _warned_block:
        return
    _warned_block = True
    discord("⚠️ บอทเข้า YouTube ไม่ได้ (YouTube อาจบล็อกเซิร์ฟเวอร์ของ GitHub) — "
            "ดูวิธีแก้ในหัวข้อ \"ถ้า YouTube บล็อก\" ใน README\n```" + err[-300:] + "```")


# ---------------------------------------------------------------- youtube


def video_info(url):
    r = subprocess.run(ytdlp_cmd() + ["-J", "--skip-download", "--ignore-no-formats-error", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        err = r.stderr.strip()
        if "Sign in to confirm" in err or "HTTP Error 429" in err:
            warn_blocked(err)
        if "not currently live" not in err and "will begin" not in err:
            log("yt-dlp:", err[-300:])
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def get_live():
    info = video_info(CHANNEL + "/live")
    if info and info.get("live_status") == "is_live":
        return info
    return None


def start_pipeline(url, outdir):
    """yt-dlp -> ffmpeg, writing one JPEG every INTERVAL seconds of video."""
    y = subprocess.Popen(ytdlp_cmd(need_ffmpeg=True) + ["--quiet","-f", "bv*[height<=1080]/b", "-o", "-", url],
                         stdout=subprocess.PIPE)
    f = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0", "-vf", f"fps=1/{INTERVAL}",
                          "-q:v", "3", os.path.join(outdir, "f_%06d.jpg")], stdin=y.stdout)
    y.stdout.close()
    return y, f


def stop(*procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------------------------------------------------------------- scanning


class Scan:
    """Tracks code sightings for one video."""

    def __init__(self, info, mode, post=True):
        self.vid = info["id"]
        self.title = info.get("title") or self.vid
        self.start_ts = info.get("release_timestamp") or info.get("timestamp") or time.time()
        self.mode = mode            # "live" | "rescan" | "test"
        self.post = post
        self.counts = Counter()
        self.state = _load(video_file(self.vid)) or {
            "id": self.vid, "title": self.title, "start": self.start_ts,
            "codes": [], "ended": False, "rescanned": False}

    def frame(self, path, offset):
        try:
            lines = ocr_lines(path)
        except Exception as e:
            log("ocr error:", e)
            return
        for code in extract_codes(lines):
            self.counts[code] += 1
            if self.counts[code] == CONFIRM:
                self.confirmed(code, path, offset)

    def confirmed(self, code, path, offset):
        when = datetime.fromtimestamp(self.start_ts + offset, ICT)
        log(f"CODE {code}  at {when:%H:%M} (+{hms(offset)})")
        if self.mode == "test":
            if self.post:
                self.send(code, path, offset, when, "🧪 ทดสอบ")
            return
        if code not in self.state["codes"]:
            self.state["codes"].append(code)
            _save(video_file(self.vid), self.state)
        if os.path.exists(code_file(code)):
            return  # already posted (maybe from another stream)
        _save(code_file(code), {"code": code, "video": self.vid, "title": self.title,
                                "time": when.isoformat(), "offset": int(offset)})
        self.send(code, path, offset, when, "🎁 โค้ดใหม่" if self.mode == "live" else "🔁 โค้ดที่ตกหล่น (สแกนย้อนหลัง)")
        persist(f"code {code}")

    def send(self, code, path, offset, when, label):
        link = f"https://youtu.be/{self.vid}?t={int(offset)}"
        discord(content=f"{label}\n```\n{code}\n```",
                embed={"description": f"📺 **{self.title}**\n🕒 {when:%d/%m %H:%M} น. "
                                      f"(นาทีที่ {hms(offset)}) — [ดูช่วงนี้]({link})",
                       "color": 0xF5A623},
                image=path)


def run_frames(url, scan, offset_fn, deadline):
    """Run the pipeline until the video ends or deadline. Returns True if it ended by itself."""
    tmp = tempfile.mkdtemp(prefix="frames_")
    y, f = start_pipeline(url, tmp)
    ended = False
    try:
        while True:
            done = f.poll() is not None
            files = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
            if not done:
                files = files[:-1]  # newest may still be being written
            for p in files:
                n = int(os.path.basename(p)[2:8])
                scan.frame(p, offset_fn(n, p))
                os.remove(p)
            if done:
                ended = True
                break
            if time.time() > deadline:
                break
            time.sleep(2)
    finally:
        stop(f, y)
        shutil.rmtree(tmp, ignore_errors=True)
    return ended


def summary(scan):
    codes = scan.state["codes"]
    body = "\n".join(f"`{c}`" for c in codes) if codes else "ไม่พบโค้ดในไลฟ์นี้"
    discord(embed={"title": "📋 สรุปโค้ดจากไลฟ์", "color": 0x5865F2,
                   "description": f"**{scan.title}**\nhttps://youtu.be/{scan.vid}\n\n{body}"})


def watch_stream(info, deadline):
    url = f"https://www.youtube.com/watch?v={info['id']}"
    scan = Scan(info, "live")
    if not os.path.exists(video_file(scan.vid)):
        _save(video_file(scan.vid), scan.state)
        discord(embed={"title": "🔴 เริ่มไลฟ์แล้ว — บอทกำลังดูโค้ดให้", "color": 0xE02B2B,
                       "description": f"**{scan.title}**\nhttps://youtu.be/{scan.vid}"})
        persist(f"live {scan.vid}")
    log("watching", scan.vid, scan.title)
    live_offset = lambda n, p: os.path.getmtime(p) - scan.start_ts
    while time.time() < deadline:
        if not run_frames(url, scan, live_offset, deadline):
            return  # out of time; next run continues this stream
        time.sleep(5)
        again = get_live()
        if again and again["id"] == scan.vid:
            log("pipeline stopped but stream still live -> restarting")
            continue
        break
    else:
        return
    scan.state["ended"] = True
    _save(video_file(scan.vid), scan.state)
    summary(scan)
    persist(f"ended {scan.vid}")


def scan_vod(info, mode, post, deadline):
    url = f"https://www.youtube.com/watch?v={info['id']}"
    scan = Scan(info, mode, post)
    log("scanning VOD", scan.vid, scan.title, f"({hms(info.get('duration') or 0)})")
    ok = run_frames(url, scan, lambda n, p: (n - 1) * INTERVAL, deadline)
    return scan, ok


# ---------------------------------------------------------------- commands


def cmd_should_run():
    print("yes" if in_window() or get_live() else "no")


def cmd_watch(max_minutes):
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline - 60:
        info = get_live()
        if info:
            watch_stream(info, deadline - 60)
            continue
        if not in_window():
            log("nothing live and outside schedule window -> exit")
            break
        time.sleep(60)
    log("watch finished")


def cmd_rescan(max_minutes):
    deadline = time.time() + max_minutes * 60
    for path in sorted(glob.glob(os.path.join(STATE_DIR, "videos", "*.json"))):
        st = _load(path)
        if not st or st.get("rescanned") or not st.get("ended"):
            continue
        if time.time() > deadline - 600:
            break
        info = video_info(f"https://www.youtube.com/watch?v={st['id']}")
        if not info or info.get("live_status") not in ("was_live", "not_live", None):
            log("VOD not ready:", st["id"])
            continue
        scan, ok = scan_vod(info, "rescan", True, deadline)
        if ok:
            scan.state["rescanned"] = True
            _save(video_file(scan.vid), scan.state)
            persist(f"rescanned {scan.vid}")


def cmd_vod(url, post):
    info = video_info(url)
    if not info:
        sys.exit("cannot read video (see error above)")
    scan, _ = scan_vod(info, "test", post, time.time() + 5 * 3600)
    print("\n=== codes found ===")
    for c, n in scan.counts.most_common():
        print(f"{c}  (seen in {n} frames){'' if n >= CONFIRM else '  <- not confirmed'}")
    if not scan.counts:
        print("(none)")


def cmd_image(path):
    lines = ocr_lines(path)
    print("OCR lines:")
    for l in lines:
        print("  ", l)
    print("codes:", sorted(extract_codes(lines)) or "(none)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("should-run")
    w = sub.add_parser("watch")
    w.add_argument("--max-minutes", type=int, default=335)
    r = sub.add_parser("rescan")
    r.add_argument("--max-minutes", type=int, default=335)
    v = sub.add_parser("vod")
    v.add_argument("url")
    v.add_argument("--post", action="store_true")
    i = sub.add_parser("image")
    i.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "should-run":
        cmd_should_run()
    elif a.cmd == "watch":
        cmd_watch(a.max_minutes)
    elif a.cmd == "rescan":
        cmd_rescan(a.max_minutes)
    elif a.cmd == "vod":
        cmd_vod(a.url, a.post)
    elif a.cmd == "image":
        cmd_image(a.file)


if __name__ == "__main__":
    main()
