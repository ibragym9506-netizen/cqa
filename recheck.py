"""Перепроверка диалогов DEV, где ручная проверка разошлась с сохранёнными оценками модели.
Оценивает их заново текущим evaluator.py (Gemini + правила кода + голосование), сохраняет
в results/chat_v1/ и печатает, совпало ли с ручной проверкой.

Использование:  py recheck.py            (по умолчанию 13 диалогов × VOTES запросов)
                py recheck.py p1-ru-005  (только указанные)
"""
import json, os, sys, time

from evaluator import evaluate

# ручная проверка по чек-листу и памятке организаторов: что должно быть (только спорные критерии)
EXPECTED = {
    "p1-ru-003": {"c06": 10, "c07": "NO", "c04": 10},
    "p1-ru-005": {"c06": 4},
    "p1-ru-007": {"c06": 10, "c07": "NO", "c04": 10},
    "p1-ru-009": {"c03": "NO", "c08": "YES", "c05": "NO"},
    "p1-ru-016": {"c02": "NO", "c06": 4},
    "p1-ru-018": {"c04": 0, "c07": "NO"},
    "p1-ru-023": {"c05": "YES", "c08": "YES", "c04": 20},
    "p1-kz-005": {"c06": 4, "c07": "NO"},
    "p1-kz-015": {"c05": "NO", "c03": "NO", "c08": "YES"},
    "p1-en-007": {"c06": 10, "c07": "NO", "c04": 10},
    "p1-en-013": {"c05": "YES", "c08": "YES"},
    "p1-en-014": {"c05": "NO", "c03": "NO", "c08": "YES", "c07": "NO"},
    "p4-ru-001": {"c04": 10},
}


def value(r):
    return r["score"] if r["type"] == "scale" and r["answer"] != "NA" else r["answer"]


def main():
    ids = sys.argv[1:] or list(EXPECTED)
    cl = json.load(open("checklist.json", encoding="utf-8"))
    recs = {r["conversation_id"]: r for r in json.load(open("dev.json", encoding="utf-8"))["data"]["records"]}
    out_dir = os.path.join("results", cl["id"])
    ok = total = 0
    for k, cid in enumerate(ids, 1):
        path = os.path.join(out_dir, f"{cid}.json")
        old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
        try:
            res = evaluate(recs[cid], cl)
        except Exception as e:
            print(f"[{k}/{len(ids)}] {cid} ОШИБКА: {str(e)[:120]}", flush=True)
            continue
        json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        got = {r["id"]: value(r) for r in res["criteria"]}
        was = {r["id"]: value(r) for r in old["criteria"]} if old else {}
        names = {r["id"]: r["name"] for r in res["criteria"]}
        print(f"[{k}/{len(ids)}] {cid}: {old['total'] if old else '—'} -> {res['total']}", flush=True)
        for c, exp in EXPECTED.get(cid, {}).items():
            total += 1
            hit = got.get(c) == exp
            ok += hit
            print(f"    {'OK ' if hit else 'НЕТ'} {names[c]}: было {was.get(c)}, стало {got.get(c)}, по проверке {exp}", flush=True)
        time.sleep(15)  # лимит бесплатного тарифа: 15 запросов в минуту (3 прогона на диалог)
    if total:
        print(f"\nСовпало с ручной проверкой: {ok} из {total}")


if __name__ == "__main__":
    main()
