"""Сервер CQA: отдаёт интерфейс (static/index.html) и оценивает диалоги через evaluator.py
(Gemini + правила кода + голосование). Чек-листы, апелляции и подсчёт баллов живут в интерфейсе.

Запуск:  py server.py   →   http://localhost:8000
"""
import hashlib, html, ipaddress, json, os, re, socket, sys, threading, time, urllib.parse, urllib.request

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import evaluator as ev

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
LOCK = threading.Lock()  # оценки идут по одной: у бесплатного Gemini лимит запросов в минуту
TYPES = {"yesno": "binary", "binary": "binary", "scale": "scale", "flag": "flag"}

app = FastAPI(title="CQA — оценка качества диалогов")
MAX_BODY = 30_000_000       # самый большой запрос к серверу (общее хранилище с набором ~1–2 МБ)
MAX_DATASET = 50_000_000    # ответ API организаторов
MAX_AUDIO = 200_000_000     # запись звонка


@app.middleware("http")
async def limit_body(request, call_next):
    n = request.headers.get("content-length") or ""
    if n.isdigit() and int(n) > MAX_BODY:
        return JSONResponse({"detail": "Слишком большой запрос"}, status_code=413)
    return await call_next(request)


# ---------- исходящие запросы: только публичные адреса (защита от SSRF) ----------
ALLOW_LOCAL = os.getenv("ALLOW_LOCAL_URLS") == "1"  # для тестов с локальной заглушкой API


def check_url(u):
    """Сервер скачивает по ссылкам из интерфейса. Чтобы через него нельзя было достучаться до
    внутренних адресов (localhost, локальная сеть, метаданные облака), разрешаем только публичные."""
    p = urllib.parse.urlsplit(u or "")
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("нужна ссылка http(s)")
    if ALLOW_LOCAL:
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return  # имя не разрешилось здесь (например, работа через прокси) — пусть решает сам запрос
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("ссылки на локальные и внутренние адреса запрещены")


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)  # переадресация тоже не должна уводить во внутреннюю сеть
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirect)


def open_url(u, timeout, headers=None):
    check_url(u)
    return _OPENER.open(urllib.request.Request(u, headers={"User-Agent": "cqa-hackathon", **(headers or {})}), timeout=timeout)


def read_limited(r, limit):
    data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"файл больше {limit // 1_000_000} МБ")
    return data


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- приведение чужих данных к единому виду ----------
ROLES = [("system", ("system", "систем", "event", "событ", "жүйе")),
         ("bot", ("bot", "бот", "ivr", "assistant", "ассистент", "aim", "virtual", "робот")),
         ("operator", ("operator", "оператор", "agent", "агент", "support", "поддерж", "специалист", "менеджер",
                       "employee", "сотрудник", "consultant", "консультант", "маман")),
         ("client", ("client", "клиент", "customer", "user", "пользоват", "caller", "абонент", "клиент", "тұтын"))]


def norm_speaker(sp):
    t = str(sp or "").strip().lower()
    for role, keys in ROLES:
        if any(k in t for k in keys):
            return role
    return "client" if t in ("c", "к", "u") else "operator" if t in ("o", "о", "a", "а") else "system"


def detect_lang(text):
    letters = re.findall(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", text or "")
    if not letters:
        return None
    if sum(ch.isascii() for ch in letters) / len(letters) > 0.6:
        return "EN"
    return "KZ" if re.search(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]", text) else "RU"


def norm_lang(lang, transcript):
    t = str(lang or "").strip().lower()
    if t[:2] in ("kz", "kk", "ka", "қа") or "kaz" in t or "каз" in t:
        return "KZ"
    if t[:2] in ("ru", "ру") or "rus" in t:
        return "RU"
    if t[:2] in ("en", "ан", "an") or "eng" in t:
        return "EN"
    return detect_lang(" ".join(m["text"] for m in transcript if m["speaker"] == "client")) or "RU"


def norm_record(rec, n=0):
    """Любая разумная запись → формат организаторов (роли, язык, канал, время — строками)."""
    rec = dict(rec or {})
    tr = rec.get("transcript") or rec.get("messages") or rec.get("dialog") or []
    if isinstance(tr, str):
        try:
            tr = json.loads(tr)
        except ValueError:
            tr = [{"speaker": m.group(1), "text": m.group(2)} for m in re.finditer(r"^\s*([^:\n]{1,30}):\s*(.+)$", tr, re.M)]
    out = []
    for m in tr if isinstance(tr, list) else []:
        if not isinstance(m, dict):  # не пропускаем: номера реплик должны совпадать с исходным массивом
            m = {"text": "" if m is None else str(m)}
        out.append({"timestamp": str(m.get("timestamp") or m.get("time") or m.get("ts") or ""),
                    "speaker": norm_speaker(m.get("speaker") or m.get("role") or m.get("author") or m.get("from")),
                    "text": str(m.get("text") or m.get("message") or m.get("content") or "")})
    ch = str(rec.get("channel") or "").lower()
    voice = any(k in ch for k in ("voice", "call", "звон", "phone")) or (bool(rec.get("audio_url")) and "chat" not in ch)
    rec.update(conversation_id=str(rec.get("conversation_id") or rec.get("id") or f"record-{n + 1}"),
               channel="voice call" if voice else "agent chat", transcript=out,
               language=norm_lang(rec.get("language"), out), audio_url=rec.get("audio_url") or None)
    return rec


def to_checklist(criteria):
    """Критерии из интерфейса → формат evaluator.py / scoring.py."""
    out = []
    for c in criteria:
        out.append({
            "id": str(c["id"]), "section": c.get("section"), "name": c.get("name") or str(c["id"]),
            "type": TYPES.get(c.get("type"), "binary"), "weight": float(c.get("weight") or 0),
            "na_allowed": bool(c.get("allow_na")), "yes": c.get("question") or "",
            "na": c.get("na_condition") or None, "rules": c.get("rules") or "",
            "scale": [{"value": s["value"], "desc": s.get("label", "")} for s in c.get("scale") or []],
        })
    return {"id": "ui", "criteria": out}


def human_refs(text):
    """Модель ссылается на реплики с нуля ([9]), интерфейс нумерует с единицы (№10)."""
    return re.sub(r"\[(\d+)\]", lambda m: f"№{int(m.group(1)) + 1}", text or "")


def pack(res, cl):
    """Результат evaluator.py → ответ для интерфейса (yes/no/na или число по шкале)."""
    types = {c["id"]: c["type"] for c in cl["criteria"]}
    rules = {}
    for note in res.get("rule_overrides", []):
        cid, _, text = note.partition(":")
        rules[cid.strip()] = text.strip()
    out = []
    for r in res["criteria"]:
        if r["answer"] == "NA":
            ans = "na"
        elif types[r["id"]] == "scale":
            ans = r["score"] if r["score"] is not None else 0
        else:
            ans = "yes" if r["answer"] == "YES" else "no"
        out.append({"criterion_id": r["id"], "answer": ans, "explanation": human_refs(r.get("explanation")),
                    "evidence": r.get("evidence") or [], "checked_range": r.get("checked_range"),
                    "quote": human_refs(r.get("quote")), "agreement": res.get("agreement", {}).get(r["id"]),
                    "rule": rules.get(r["id"])})
    u = res.get("usage") or {}
    return {"results": out, "model": res.get("model"), "needs_review": res.get("needs_review"),
            "rule_overrides": res.get("rule_overrides", []), "total": res.get("total"),
            "usage": {"calls": u.get("calls") or 0, "input_tokens": u.get("input_tokens") or 0,
                      "output_tokens": u.get("output_tokens") or 0}}


def run(rec, criteria, extra="", votes=None):
    if not isinstance(rec, dict) or not isinstance(rec.get("transcript"), list) or not criteria:
        raise HTTPException(400, "Нужны record (с transcript) и criteria")
    rec = norm_record(rec)
    try:
        cl = to_checklist(criteria)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise HTTPException(400, f"Некорректный чек-лист: {e}")
    with LOCK:
        try:
            res = ev.evaluate(rec, cl, extra=extra, votes=votes)
        except Exception as e:
            raise HTTPException(502, f"Ошибка оценки моделью: {str(e)[:300]}")
    return pack(res, cl)


@app.get("/api/health")
def health():
    try:
        import sherpa_onnx, soundfile  # noqa: F401 — распознавание речи для привязки звонков
        asr = True
    except ImportError:
        asr = False
    return {"ok": True, "provider": "gemini", "model": ev.MODELS[0].strip(), "votes": ev.VOTES, "asr": asr,
            "llm": bool(ev.CLIENTS), "keys": len(ev.CLIENTS)}


@app.post("/api/llm/evaluate")
def llm_evaluate(body: dict = Body(...)):
    return run(body.get("record"), body.get("criteria") or [])


@app.post("/api/llm/second-opinion")
def second_opinion(body: dict = Body(...)):
    c = body.get("criterion") or {}
    comment = (body.get("comment") or "").strip()
    extra = (f"АПЕЛЛЯЦИЯ. Оценку по критерию «{c.get('name', '')}» оспаривают. Довод: «{comment}». "
             "Оцени этот критерий заново и независимо: учти довод, но проверь его по тексту диалога.")
    return run(body.get("record"), [c] if c else [], extra=extra, votes=1)


@app.post("/api/fetch")
def fetch_dataset(body: dict = Body(...)):
    """Загрузка набора по ссылке организаторов (GET ...?group_id=...) — через сервер, без CORS."""
    url = str(body.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "Нужна ссылка http(s)")
    try:
        with open_url(url, 90) as r:
            data = json.loads(read_limited(r, MAX_DATASET).decode("utf-8-sig"))
    except Exception as e:
        raise HTTPException(502, f"Не удалось загрузить набор: {str(e)[:200]}")
    if isinstance(data, dict) and data.get("isSuccess") is False:
        raise HTTPException(502, f"API организаторов вернул ошибку: {data.get('error')}")
    # копия набора на диск — её можно прогнать батчем: py batch.py data_<group>.json
    d = data.get("data") if isinstance(data, dict) else None
    gid = re.sub(r"[^\w-]", "_", str((d if isinstance(d, dict) else {}).get("dataset_group_id") or "dataset"))[:60]
    with open(os.path.join(BASE, f"data_{gid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def audio_file_url(u):
    """audio_url организаторов → прямая ссылка на файл записи (страница с плеером → src файла)."""
    with open_url(u, 60, {"Range": "bytes=0-0"}) as r:
        if (r.headers.get("Content-Type") or "").startswith("audio/"):
            return u
        page = r.read(200_000).decode("utf-8", "replace")
    m = re.search(r'<source[^>]+src="(https?://[^"]+)"', page) or re.search(r'<audio[^>]+src="(https?://[^"]+)"', page)
    if not m:
        raise ValueError("на странице записи не найден аудиофайл")
    src = html.unescape(m.group(1))
    check_url(src)
    return src


SYNC_LOCK = threading.Lock()
TOOLS = os.path.join(BASE, "tools")


@app.post("/api/sync")
def sync_audio(body: dict = Body(...)):
    """Привязка реплик звонка к записи: скачать запись, распознать речь, выровнять с транскриптом.
    Результат кешируется в sync_cache/ (повторное открытие — мгновенно)."""
    rec = norm_record(body.get("record") or {})
    if not rec.get("audio_url"):
        raise HTTPException(400, "У записи нет audio_url")
    cid = re.sub(r"[^\w-]", "_", rec["conversation_id"])[:80]
    key = hashlib.sha1(json.dumps([[m["speaker"], m["text"], m["timestamp"]] for m in rec["transcript"]],
                                  ensure_ascii=False).encode()).hexdigest()[:12]
    cache = os.path.join(BASE, "sync_cache", f"{cid}_{key}.json")
    if os.path.exists(cache):
        return read_json(cache)
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    try:
        import audio_sync
    except ImportError as e:
        raise HTTPException(501, f"Распознавание речи не установлено (pip install sherpa-onnx soundfile): {e}")
    with SYNC_LOCK:
        if os.path.exists(cache):
            return read_json(cache)
        try:
            src = audio_file_url(rec["audio_url"])
            with open_url(src, 180) as r:
                ctype, data = (r.headers.get("Content-Type") or "").split(";")[0].strip(), read_limited(r, MAX_AUDIO)
            ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/mp4": "m4a"}.get(ctype, "mp3")
            os.makedirs(os.path.join(BASE, "audio"), exist_ok=True)
            path = os.path.join(BASE, "audio", f"{cid}.{ext}")
            with open(path, "wb") as f:
                f.write(data)
            res = audio_sync.sync_record(rec, path)
        except Exception as e:
            raise HTTPException(502, f"Не удалось разметить запись: {str(e)[:200]}")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return res


@app.get("/api/audio")
def audio(u: str):
    """audio_url организаторов отдаёт HTML-страницу с плеером, а сам файл — по ссылке Dynamics,
    подписанной ~на час. Берём свежую ссылку на файл и перенаправляем плеер на неё (перемотка работает)."""
    if not re.match(r"^https?://", u or ""):
        raise HTTPException(400, "Нужна ссылка http(s)")
    try:
        return RedirectResponse(audio_file_url(u))
    except Exception as e:
        raise HTTPException(502, f"Не удалось открыть запись: {str(e)[:200]}")


# ---------- общее хранилище: чек-листы, оценки, апелляции, решения (data/state.json) ----------
STATE_FILE = os.path.join(BASE, "data", "state.json")
STATE_LOCK = threading.Lock()
_state = {"version": 0, "state": None}


def _load_state():
    if _state["state"] is None and os.path.exists(STATE_FILE):
        try:
            saved = read_json(STATE_FILE)
            _state.update(version=saved.get("version", 0), state=saved.get("state"))
        except Exception:
            pass  # битый файл — начнём заново, старый не трогаем
    return _state


def merge_state(a, b):
    """Слияние правок двух пользователей. a — то, что на сервере, b — пришло из браузера.
    Чек-листы — побеждает более новая версия; апелляции — объединение, решение важнее открытой;
    оценки по диалогу — побеждает более свежая отметка изменения; решения проверяющего — объединение."""
    if not a:
        return b
    if not b:
        return a
    out = dict(a)
    cls = dict(a.get("checklists") or {})
    for k, v in (b.get("checklists") or {}).items():
        if k not in cls or str(v.get("updated_at") or "") > str(cls[k].get("updated_at") or ""):
            cls[k] = v
    out["checklists"] = cls
    if (b.get("active_at") or 0) > (a.get("active_at") or 0) and b.get("active_checklist") in cls:
        out["active_checklist"], out["active_at"] = b["active_checklist"], b["active_at"]
    if out.get("active_checklist") not in cls and cls:
        out["active_checklist"] = next(iter(cls))
    sa, sb = a.get("stamp") or {}, b.get("stamp") or {}
    convs = set(sa) | set(sb)
    for f in ("judgments", "evals"):
        convs |= set(a.get(f) or {}) | set(b.get(f) or {})
    stamp = {}
    for f in ("judgments", "evals"):
        da, db, res = a.get(f) or {}, b.get(f) or {}, {}
        for c in convs:
            ta, tb = sa.get(c, 0), sb.get(c, 0)
            src = db if tb > ta else da if ta > tb else (da if c in da else db)
            if c in src:
                res[c] = src[c]
        out[f] = res
    for c in convs:
        stamp[c] = max(sa.get(c, 0), sb.get(c, 0))
    out["stamp"] = stamp
    imp = dict(a.get("imported") or {})
    for k, v in (b.get("imported") or {}).items():
        if k not in imp or (v.get("imported_at") or 0) > (imp[k].get("imported_at") or 0):
            imp[k] = v
    out["imported"] = imp
    ovr = {k: dict(v) for k, v in (a.get("overrides") or {}).items()}
    for c, items in (b.get("overrides") or {}).items():
        cur = ovr.setdefault(c, {})
        for h, o in items.items():
            if h not in cur or str(o.get("decided_at") or "") > str(cur[h].get("decided_at") or ""):
                cur[h] = o
    out["overrides"] = ovr
    aps = {x["id"]: x for x in a.get("appeals") or [] if x.get("id")}
    for x in b.get("appeals") or []:
        y = aps.get(x.get("id"))
        if y is None or (y.get("status") == "open" and x.get("status") != "open"):
            aps[x["id"]] = x
    out["appeals"] = sorted(aps.values(), key=lambda x: str(x.get("created_at") or ""))
    out["v"] = 1
    return out


@app.get("/api/state")
def get_state(v: int = -1):
    with STATE_LOCK:
        st = _load_state()
        if v == st["version"]:
            return {"version": st["version"]}  # ничего не менялось — без лишнего трафика
        return {"version": st["version"], "state": st["state"]}


@app.put("/api/state")
def put_state(body: dict = Body(...)):
    incoming = body.get("state")
    if not isinstance(incoming, dict) or not isinstance(incoming.get("checklists"), dict):
        raise HTTPException(400, "Нужен state с checklists")
    with STATE_LOCK:
        st = _load_state()
        try:
            merged = merge_state(st["state"], incoming)
        except (AttributeError, TypeError, KeyError, ValueError):
            raise HTTPException(400, "Некорректный формат состояния")
        if merged != st["state"]:
            st["state"], st["version"] = merged, st["version"] + 1
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"version": st["version"], "state": merged}, f, ensure_ascii=False)
            os.replace(tmp, STATE_FILE)  # атомарно: при сбое не останется полузаписанного файла
        return {"version": st["version"], "state": st["state"]}


CODES = ["contact", "discovery", "identification", "consultation", "empathy", "dialogue", "closure", "critical"]


@app.post("/api/results")
def save_results(body: dict = Body(...)):
    """Файл для сдачи по формату организаторов: results/dev/results.json или results/prod/results.json."""
    kind, data = body.get("set"), body.get("data")
    if kind not in ("dev", "prod") or not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise HTTPException(400, "Нужны set (dev или prod) и data.results")
    if not all(isinstance(r, dict) for r in data["results"]):
        raise HTTPException(400, "Каждая запись results — объект")
    ids = [r.get("conversation_id") for r in data["results"]]
    if len(ids) != len(set(ids)) or not all(ids):
        raise HTTPException(400, "Пустые или повторяющиеся conversation_id")
    for r in data["results"]:
        if r.get("status") == "ok" and sorted(c.get("criterion_id") for c in r.get("criteria") or []) != sorted(CODES):
            raise HTTPException(400, f"{r['conversation_id']}: нужны ровно 8 критериев {CODES}")
    folder = os.path.join(BASE, "results", kind)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "results.json")
    if os.path.exists(path):  # прежний файл не теряем: копия в data/backup (data/ не попадает в Git)
        bak = os.path.join(BASE, "data", "backup")
        os.makedirs(bak, exist_ok=True)
        os.replace(path, os.path.join(bak, f"results_{kind}_{time.strftime('%Y%m%d_%H%M%S')}.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"path": f"results/{kind}/results.json", "count": len(ids),
            "ok": sum(1 for r in data["results"] if r.get("status") == "ok")}


@app.get("/api/precomputed")
def precomputed():
    """Готовые оценки пакетного прогона (py batch.py → results/<чек-лист>/*.json).
    Интерфейс подставляет их в ещё не оценённые диалоги, чтобы не тратить запросы к Gemini повторно."""
    folder = os.path.join(BASE, "results", os.getenv("RESULTS_DIR", "chat_v1"))
    out = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            try:
                r = read_json(os.path.join(folder, name))
            except Exception:
                continue
            keep = ("id", "name", "answer", "score", "explanation", "quote", "evidence", "checked_range")
            out[r.get("conversation_id") or name[:-5]] = {
                "mtime": int(os.path.getmtime(os.path.join(folder, name))), "model": r.get("model"), "agreement": r.get("agreement") or {},
                "rule_overrides": r.get("rule_overrides") or [], "needs_review": r.get("needs_review"),
                "criteria": [{k: c.get(k) for k in keep} for c in r.get("criteria") or []]}
    return out


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")  # HOST=0.0.0.0 — открыть сайт с других устройств в той же сети
    print(f"CQA: откройте http://localhost:{port}")
    if host == "0.0.0.0":
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
            print(f"     с других устройств в этой сети: http://{s.getsockname()[0]}:{port}")
        except OSError:
            pass
    uvicorn.run(app, host=host, port=port)
