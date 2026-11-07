"""Пробный прогон другой модели на нескольких диалогах DEV — без сохранения результатов.
Показывает ответ модели ДО правил кода и после, рядом с ручной проверкой.

Использование:  py try_model.py                      (gemini-3.8-flash, 4 диалога, 1 прогон)
                py try_model.py gemini-3.6-flash p1-ru-005 p1-ru-023
"""
import json, sys, time

import evaluator as ev
from recheck import EXPECTED, value

DEFAULT_IDS = ["p1-ru-005", "p1-ru-023", "p1-kz-005", "p1-ru-007"]


def resolve(name):
    """Точное имя модели из списка доступных (например, '3.8' → 'gemini-3.8-flash')."""
    names = [m.name.split("/")[-1] for m in ev.client.models.list()]
    if name in names:
        return name
    bad = ("lite", "tts", "live", "image", "audio", "embedding", "transcribe")
    cand = [n for n in names if name in n and not any(b in n for b in bad)]
    if not cand:
        sys.exit(f"Модель «{name}» не найдена. Доступные: {', '.join(n for n in names if 'gemini' in n)}")
    return sorted(cand, key=len)[0]


def main():
    args = sys.argv[1:]
    model = resolve(args[0] if args and not args[0].startswith("p") else "3.8-flash")
    ids = [a for a in args if a.startswith("p")] or DEFAULT_IDS
    ev.MODELS = [model]  # только эта модель, без перехода на запасную
    print(f"Модель: {model}, диалогов: {len(ids)}, прогонов на диалог: 1\n", flush=True)
    cl = json.load(open("checklist.json", encoding="utf-8"))
    recs = {r["conversation_id"]: r for r in json.load(open("dev.json", encoding="utf-8"))["data"]["records"]}
    raw_ok = fin_ok = total = 0
    for cid in ids:
        t0 = time.time()
        try:
            res = ev.evaluate(recs[cid], cl, votes=1)
        except Exception as e:
            print(f"{cid}: ОШИБКА {str(e)[:150]}"); continue
        old = json.load(open(f"results/{cl['id']}/{cid}.json", encoding="utf-8"))
        got = {r["id"]: value(r) for r in res["criteria"]}
        prev = {r["id"]: value(r) for r in old["criteria"]}
        names = {r["id"]: r["name"] for r in res["criteria"]}
        fixed = {n.split(":")[0]: n for n in res.get("rule_overrides", [])}
        print(f"{cid}: итог {res['total']} (3.5 + правила: {old['total']}), {time.time() - t0:.0f} с, модель {res['model']}")
        for c, exp in EXPECTED.get(cid, {}).items():
            raw = fixed[c].split("модель ")[1].split(" →")[0] if c in fixed else got[c]
            raw = int(raw) if str(raw).isdigit() else raw
            total += 1; raw_ok += raw == exp; fin_ok += got[c] == exp
            print(f"   {names[c]}: сама модель {raw}, после правил {got[c]}, 3.5 {prev[c]}, по проверке {exp}"
                  f"{'  ✓' if got[c] == exp else '  ✗'}")
        time.sleep(13)  # у 3.8 Flash лимит 5 запросов в минуту
    if total:
        print(f"\nСама модель угадала {raw_ok} из {total}, с правилами кода {fin_ok} из {total}")


if __name__ == "__main__":
    main()
