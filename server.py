"""Сервер CQA: отдаёт интерфейс (static/index.html) и оценивает диалоги через evaluator.py
(Gemini + правила кода + голосование). Чек-листы, апелляции и подсчёт баллов живут в интерфейсе.

Запуск:  py server.py   →   http://localhost:8000
"""
import html, json, os, re, threading, urllib.request

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import evaluator as ev

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
LOCK = threading.Lock()  # оценки идут по одной: у бесплатного Gemini лимит запросов в минуту
TYPES = {"yesno": "binary", "binary": "binary", "scale": "scale", "flag": "flag"}

app = FastAPI(title="CQA — оценка качества диалогов")


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
    rec = dict(rec, channel=rec.get("channel") or "agent chat", language=rec.get("language") or "RU")
    cl = to_checklist(criteria)
    with LOCK:
        try:
            res = ev.evaluate(rec, cl, extra=extra, votes=votes)
        except Exception as e:
            raise HTTPException(502, f"Ошибка оценки моделью: {str(e)[:300]}")
    return pack(res, cl)


@app.get("/api/health")
def health():
    return {"ok": True, "provider": "gemini", "model": ev.MODELS[0].strip(), "votes": ev.VOTES}


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
    url = (body.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "Нужна ссылка http(s)")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cqa-hackathon"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8-sig"))
    except Exception as e:
        raise HTTPException(502, f"Не удалось загрузить набор: {str(e)[:200]}")
    if isinstance(data, dict) and data.get("isSuccess") is False:
        raise HTTPException(502, f"API организаторов вернул ошибку: {data.get('error')}")
    # копия набора на диск — её можно прогнать батчем: py batch.py data_<group>.json
    gid = re.sub(r"[^\w-]", "_", str(((data or {}).get("data") or {}).get("dataset_group_id") or "dataset"))
    with open(os.path.join(BASE, f"data_{gid}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


@app.get("/api/audio")
def audio(u: str):
    """audio_url организаторов отдаёт HTML-страницу с плеером, а сам файл — по ссылке Dynamics,
    подписанной ~на час. Берём свежую ссылку на файл и перенаправляем плеер на неё (перемотка работает)."""
    if not re.match(r"^https?://", u or ""):
        raise HTTPException(400, "Нужна ссылка http(s)")
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "cqa-hackathon", "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if (r.headers.get("Content-Type") or "").startswith("audio/"):
                return RedirectResponse(u)  # ссылка уже на файл
            page = r.read(200_000).decode("utf-8", "replace")
    except Exception as e:
        raise HTTPException(502, f"Не удалось открыть запись: {str(e)[:200]}")
    m = re.search(r'<source[^>]+src="(https?://[^"]+)"', page) or re.search(r'<audio[^>]+src="(https?://[^"]+)"', page)
    if not m:
        raise HTTPException(502, "На странице записи не найден аудиофайл")
    return RedirectResponse(html.unescape(m.group(1)))


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"CQA: откройте http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
