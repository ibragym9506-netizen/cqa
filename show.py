import glob, json

M = {"YES": "Y", "NO": "N", "NA": "-"}
for p in sorted(glob.glob("results/chat_v1/*.json")):
    r = json.load(open(p, encoding="utf-8"))
    cells = []
    for c in r["criteria"]:
        v = c["score"] if c["score"] is not None else M.get(c["answer"], "?")
        cells.append(f'{c["id"][1:]}:{v}')
    g = "G" if r["gross_error"] else " "
    print(f'{r["conversation_id"]:10} {str(r["total"]):>6} {g} ' + " ".join(cells))