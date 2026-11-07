"""Скачивает записи звонков набора в папку audio/ (для сверки времени реплик с записью).
Использование: py fetch_audio.py [data_DEV.json]"""
import html, json, os, re, sys, urllib.request

src = sys.argv[1] if len(sys.argv) > 1 else "data_DEV.json"
data = json.load(open(src, encoding="utf-8"))
records = (data.get("data") or data).get("records", [])
os.makedirs("audio", exist_ok=True)

for r in records:
    url = r.get("audio_url")
    if not url:
        continue
    cid = r["conversation_id"]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cqa-hackathon"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            ctype = resp.headers.get("Content-Type") or ""
            body = resp.read()
        if not ctype.startswith("audio/"):
            page = body.decode("utf-8", "replace")
            m = re.search(r'<source[^>]+src="(https?://[^"]+)"', page) or re.search(r'<audio[^>]+src="(https?://[^"]+)"', page)
            if not m:
                print(cid, "— на странице нет аудиофайла")
                continue
            req = urllib.request.Request(html.unescape(m.group(1)), headers={"User-Agent": "cqa-hackathon"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                ctype = resp.headers.get("Content-Type") or ""
                body = resp.read()
        ext = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/mp4": "m4a"}.get(ctype.split(";")[0].strip(), "mp3")
        path = os.path.join("audio", f"{cid}.{ext}")
        with open(path, "wb") as f:
            f.write(body)
        print(cid, "->", path, round(len(body) / 1e6, 1), "МБ")
    except Exception as e:
        print(cid, "ОШИБКА:", str(e)[:150])
