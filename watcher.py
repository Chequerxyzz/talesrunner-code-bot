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
# one or more webhook links (one per line, or separated by commas/spaces) → every room gets every post
WEBHOOKS = [w for w in re.split(r"[\s,]+", os.environ.get("DISCORD_WEBHOOK_URL", "")) if w.startswith("http")]
COOKIES = os.environ.get("YT_COOKIES_FILE", "")
INTERVAL = int(os.environ.get("FRAME_INTERVAL", "15"))      # seconds between frames
# evidence needed before posting: same spelling read cleanly in 2 frames (30 s), or ~8 frames of guessed reads
THRESHOLD = float(os.environ.get("CODE_THRESHOLD", "4"))
MIN_SCORE = float(os.environ.get("OCR_MIN_SCORE", "0.85"))  # OCR confidence needed for a full-weight read
# Text that looks like a code but isn't (placeholder in the stream overlay template etc.).
# More can be added one per line in state/blocked.txt
BLOCKED = {"TONT4N2A2I2I", "ROSA4L8R4Z9I"}
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
    """Return {code: (weight, source text)} for one frame.
    Weight: confident OCR, code standing on its own anywhere on screen (label or not) = 2;
    low-confidence or buried inside other text = 0.5;
    needed letter/digit correction = 1 next to the word CODE, 0.25 otherwise."""
    found = {}

    def add(c, w, src):
        if w > found.get(c, (0, ""))[0]:
            found[c] = (w, src)

    for item in lines:
        line, score = item if isinstance(item, tuple) else (item, 1.0)
        up = line.upper()
        sure = score >= MIN_SCORE   # OCR is confident about this text box (Thai text read as junk scores low)
        ctx = "CODE" in re.sub(r"[^A-Z]", "", up.replace("0", "O"))
        src = f"{line} ({score:.2f})"
        # exact matches, also with spaces removed (OCR sometimes splits a code)
        for text in (up, re.sub(r"\s+", "", up)):
            for tok in re.split(r"[^A-Z0-9]+", text):
                for c in CODE_RE.findall(tok):
                    # full weight only when the code stands on its own (or right after "CODE" / "CODE1")
                    clean = tok == c or re.fullmatch(r"C[O0]DE\d?" + c, tok)
                    add(c, 2 if clean and sure else 0.5, src)
        # fuzzy: tokens about the length of a code
        for tok in re.split(r"[^A-Z0-9]+", up):
            if 12 <= len(tok) <= 14 and not CODE_RE.search(tok):
                for i in range(len(tok) - 11):
                    c = _fix(tok[i:i + 12])
                    if c:
                        add(c, 1 if ctx and sure else 0.25, src)
    return found


_blocked_cache = None


def blocked():
    global _blocked_cache
    if _blocked_cache is None:
        _blocked_cache = set(BLOCKED)
        try:
            with open(os.path.join(STATE_DIR, "blocked.txt"), encoding="utf-8") as f:
                _blocked_cache |= {l.strip().upper() for l in f if l.strip() and not l.startswith("#")}
        except OSError:
            pass
    return _blocked_cache


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


_ocr = None


def _ocr_img(img):
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    result, _ = _ocr(img)
    return [(r[1], float(r[2])) for r in (result or [])]


def ocr_lines(path):
    """OCR the whole frame, plus 2x-enlarged top and bottom strips where codes are shown."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return []
    h = img.shape[0]
    lines = _ocr_img(img)
    for y0, y1 in ((0.0, 0.2), (0.88, 1.0)):
        crop = img[int(h * y0):int(h * y1), :]
        lines += _ocr_img(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    return lines


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
    if not WEBHOOKS:
        log("[discord-dry-run]", content or "", json.dumps(embed, ensure_ascii=False) if embed else "")
        return
    payload = {"content": content, "allowed_mentions": {"parse": []}}
    if embed:
        if image:
            embed["image"] = {"url": "attachment://frame.jpg"}
        payload["embeds"] = [embed]
    for n, hook in enumerate(WEBHOOKS, 1):
        for _ in range(4):
            try:
                if image and os.path.exists(image):
                    with open(image, "rb") as f:
                        r = requests.post(hook, data={"payload_json": json.dumps(payload)},
                                          files={"files[0]": ("frame.jpg", f, "image/jpeg")}, timeout=30)
                else:
                    r = requests.post(hook, json=payload, timeout=30)
                if r.status_code == 429:
                    time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
                    continue
                if r.status_code in (401, 403, 404):
                    log(f"discord: webhook #{n} was deleted or is wrong (HTTP {r.status_code}), skipping it")
                    break
                r.raise_for_status()
                break
            except Exception as e:
                log(f"discord error (webhook #{n}):", type(e).__name__)  # never log the URL itself
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


def start_pipeline(url, outdir, errpath):
    """yt-dlp -> ffmpeg, writing one JPEG every INTERVAL seconds of video."""
    errf = open(errpath, "wb")
    y = subprocess.Popen(ytdlp_cmd(need_ffmpeg=True) + ["-f", "bv*[height<=1080]/b", "-o", "-", url],
                         stdout=subprocess.PIPE, stderr=errf)
    f = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-i", "pipe:0", "-vf", f"fps=1/{INTERVAL}",
                          "-q:v", "3", os.path.join(outdir, "f_%06d.jpg")], stdin=y.stdout, stderr=errf)
    y.stdout.close()
    errf.close()
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
        self.counts = Counter()     # weighted sightings per exact spelling
        self.clusters = []          # [{"v": Counter, "posted": code|None}] similar spellings grouped
        self.src = {}               # last OCR text each code was read from (for debugging)
        self.frames = 0
        self.err = ""
        self.state =_load(video_file(self.vid)) or {
            "id": self.vid, "title": self.title, "start": self.start_ts,
            "codes": [], "ended": False, "rescanned": False}

    def frame(self, path, offset):
        try:
            lines = ocr_lines(path)
        except Exception as e:
            log("ocr error:", e)
            return
        for code, (w, src) in extract_codes(lines).items():
            if code in blocked():
                continue
            self.counts[code] += w
            self.src[code] = src
            cl = next((c for c in self.clusters if any(hamming(code, k) <= 2 for k in c["v"])), None)
            if cl is None:
                cl = {"v": Counter(), "posted": None}
                self.clusters.append(cl)
            cl["v"][code] += w
            self.decide(cl, path, offset)

    def decide(self, cl, path, offset):
        """Post the most-seen spelling once there is enough evidence; correct it if another spelling wins later."""
        best, best_n = cl["v"].most_common(1)[0]
        if best_n < THRESHOLD:
            return
        if cl["posted"] is None:
            cl["posted"] = best
            self.publish(best, path, offset)
        elif best != cl["posted"] and best_n >= cl["v"][cl["posted"]] + THRESHOLD:
            old, cl["posted"] = cl["posted"], best
            self.publish(best, path, offset, wrong=old)

    def _known(self, code):
        """Already posted before (same code or a near-identical misread of it)?"""
        if os.path.exists(code_file(code)):
            return True
        for p in glob.glob(os.path.join(STATE_DIR, "codes", "*.json")):
            if hamming(code, os.path.basename(p)[:-5]) <= 1:
                return True
        return False

    def publish(self, code, path, offset, wrong=None):
        when = datetime.fromtimestamp(self.start_ts + offset, ICT)
        log(f"CODE {code}  at {when:%H:%M} (+{hms(offset)})" + (f"  (correcting {wrong})" if wrong else ""))
        label = {"live": "🎁 โค้ดใหม่", "rescan": "🔁 โค้ดที่ตกหล่น (สแกนย้อนหลัง)", "test": "🧪 ทดสอบ"}[self.mode]
        if wrong:
            label = f"✏️ แก้ไข: โค้ดที่ส่งไปก่อนหน้า `{wrong}` อ่านผิด ที่ถูกคือ"
        if self.mode == "test":
            if self.post:
                self.send(code, path, offset, when, label + f"\n-# อ่านจาก: {self.src.get(code, '')[:120]}")
            return
        if wrong and wrong in self.state["codes"]:
            self.state["codes"].remove(wrong)
            try:
                os.remove(code_file(wrong))
            except OSError:
                pass
        if code not in self.state["codes"]:
            self.state["codes"].append(code)
            _save(video_file(self.vid), self.state)
        if not wrong and self._known(code):
            return  # already posted (maybe from another stream)
        _save(code_file(code), {"code": code, "video": self.vid, "title": self.title,
                                "time": when.isoformat(), "offset": int(offset)})
        self.send(code, path, offset, when, label)
        persist(f"code {code}")

    def send(self, code, path, offset, when, label):
        link = f"https://youtu.be/{self.vid}?t={int(offset)}"
        discord(content=label,
                embed={"description": f"📺 **{self.title}**\n🕒 {when:%d/%m %H:%M} น. "
                                      f"(นาทีที่ {hms(offset)}) — [ดูช่วงนี้]({link})",
                       "color": 0xF5A623},
                image=path)
        # the code goes alone in its own message, so on a phone "long-press → Copy Text" copies just the code
        discord(content=code)


def run_frames(url, scan, offset_fn, deadline):
    """Run the pipeline until the video ends or deadline. Returns True if it ended by itself."""
    tmp = tempfile.mkdtemp(prefix="frames_")
    errpath = tmp + "_err.log"
    y, f = start_pipeline(url, tmp, errpath)
    ended = False
    scan.frames = 0
    try:
        while True:
            done = f.poll() is not None
            files = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
            if not done:
                files = files[:-1]  # newest may still be being written
            for p in files:
                n = int(os.path.basename(p)[2:8])
                scan.frame(p, offset_fn(n, p))
                scan.frames += 1
                if scan.frames % 40 == 0:
                    log(f"{scan.frames} frames read, top readings: {scan.counts.most_common(6)}")
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
        try:
            with open(errpath, encoding="utf-8", errors="replace") as ef:
                scan.err = "\n".join(l for l in ef.read().splitlines() if "[download]" not in l)[-800:]
            os.remove(errpath)
        except OSError:
            scan.err = ""
    if scan.frames == 0:
        log("NO FRAMES. downloader output:\n" + scan.err)
        # live mode retries by itself and only warns after repeated failures (see watch_stream)
        if scan.mode != "live" and ("Sign in to confirm" in scan.err or "429" in scan.err):
            warn_blocked(scan.err)
    return ended


def summary(scan):
    # drop junk that was saved before it was blocked (e.g. by a run that started with older code)
    codes = [c for c in scan.state["codes"] if c not in blocked()]
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
    fails = 0
    while time.time() < deadline:
        if not run_frames(url, scan, live_offset, deadline):
            return  # out of time; next run continues this stream
        fails = fails + 1 if scan.frames == 0 else 0
        if fails == 3:  # a one-off hiccup (e.g. 429 Too Many Requests) is normal; 3 in a row is not
            warn_blocked(scan.err)
        time.sleep(5 if fails == 0 else min(30 * fails, 300))
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
    else:
        if in_window() or get_live():
            open(".continue", "w").close()  # tells the workflow to start the next run right away
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
    if post:
        discord("🧪 เริ่มทดสอบ — บอทกำลังดูไลฟ์เก่า อาจใช้เวลา 20–40 นาที")
    info = video_info(url)
    if not info:
        if post:
            discord("❌ ทดสอบไม่สำเร็จ: เปิดข้อมูลคลิปจาก YouTube ไม่ได้")
        sys.exit("cannot read video (see error above)")
    scan, _ = scan_vod(info, "test", post, time.time() + 5 * 3600)
    print(f"\n=== {scan.frames} frames read — codes found ===")
    lines = []
    for cl in sorted(scan.clusters, key=lambda c: -sum(c["v"].values())):
        print(dict(cl["v"]), "-> posted:", cl["posted"])
        if cl["posted"]:
            lines.append(f"✅ `{cl['posted']}`")
    ignored = sum(1 for cl in scan.clusters if not cl["posted"])
    if ignored:
        lines.append(f"(ตัดทิ้ง {ignored} อันที่น่าจะอ่านผิด/เห็นน้อยเกินไป)")
    if scan.frames == 0:
        if post:
            discord("❌ ทดสอบไม่สำเร็จ: โหลดวิดีโอไม่ได้\n```" + (scan.err or "no output")[-1500:] + "```")
        sys.exit("no frames were read — download failed (see output above)")
    if post:
        discord(f"🧪 ทดสอบเสร็จ อ่านไป {scan.frames} ภาพ\n" + ("\n".join(lines) if lines else "ไม่เจอโค้ดเลย"))


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
