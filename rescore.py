"""Применяет текущие правила кода к уже готовым результатам — без запросов к модели.
Использование: py rescore.py [файл_данных] [файл_чеклиста]"""
import json, glob, os, sys

from evaluator import apply_rules
from scoring import compute

src = sys.argv[1] if len(sys.argv) > 1 else "dev.json"
cl = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "checklist.json", encoding="utf-8"))
convs = {c["conversation_id"]: c for c in json.load(open(src, encoding="utf-8"))["data"]["records"]}
KEEP = ("explanation", "reasoning", "quote", "evidence", "checked_range")

changed = 0
for path in sorted(glob.glob(os.path.join("results", cl["id"], "*.json"))):
    res = json.load(open(path, encoding="utf-8"))
    conv = convs.get(res.get("conversation_id"))
    if not conv:
        continue
    answers = {r["id"]: {k: r.get(k) for k in ("answer", "score") + KEEP} for r in res["criteria"]}
    notes = apply_rules(conv, cl, answers)
    if not notes:
        continue
    new = compute(cl, answers)
    for r in new["criteria"]:
        for k in KEEP:
            r[k] = answers.get(r["id"], {}).get(k)
    old = res.get("total")
    res.update({k: new[k] for k in ("criteria", "sections", "total", "gross_error", "applicable_weight")})
    res["rule_overrides"] = res.get("rule_overrides", []) + notes
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    changed += 1
    print(f'{res["conversation_id"]}: {old} -> {res["total"]} | {"; ".join(notes)}')
print(f"Пересчитано: {changed}")
