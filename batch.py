import json, os, sys, time
from evaluator import evaluate

# Использование: py batch.py [файл_данных] [файл_чеклиста]
src = sys.argv[1] if len(sys.argv) > 1 else "dev.json"
cl_path = sys.argv[2] if len(sys.argv) > 2 else "checklist.json"

cl = json.load(open(cl_path, encoding="utf-8"))
records = json.load(open(src, encoding="utf-8"))["data"]["records"]
out_dir = os.path.join("results", cl["id"])
os.makedirs(out_dir, exist_ok=True)

for i, conv in enumerate(records, 1):
    cid = conv["conversation_id"]
    path = os.path.join(out_dir, f"{cid}.json")
    if os.path.exists(path):
        print(f"[{i}/{len(records)}] {cid} — уже есть, пропуск")
        continue
    try:
        res = evaluate(conv, cl)
        json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{i}/{len(records)}] {cid} -> {res['total']}  грубая={res['gross_error']}", flush=True)
    except Exception as e:
        print(f"[{i}/{len(records)}] {cid} ОШИБКА: {str(e)[:100]}", flush=True)
    time.sleep(10)  # чтобы не упереться в лимит бесплатного тарифа