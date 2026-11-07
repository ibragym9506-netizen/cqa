def _max_scale(c):
    return max(s["value"] for s in c["scale"])


def _snap(c, value):
    """Приводим балл к ближайшей допустимой ступени шкалы."""
    steps = [s["value"] for s in c["scale"]]
    return min(steps, key=lambda s: abs(s - value))


def compute(checklist, answers):
    """
    answers: {crit_id: {"answer": "YES"|"NO"|"NA", "score": int|None, ...}}
    Возвращает подробный результат с баллами, секциями, итогом и флагом грубой ошибки.
    """
    rows = []
    applicable_weight = 0.0

    for c in checklist["criteria"]:
        a = answers.get(c["id"], {})
        ans = (a.get("answer") or "NO").upper()
        if ans == "NA" and not c.get("na_allowed"):
            ans = "NO"  # N/A запрещён правилами — считаем как невыполненный

        row = {"id": c["id"], "name": c["name"], "section": c.get("section"),
               "type": c["type"], "weight": c["weight"], "answer": ans,
               "score": None, "raw_points": None, "points": None}

        if c["type"] == "flag" or c["weight"] == 0:
            rows.append(row)
            continue
        if ans == "NA":
            rows.append(row)
            continue

        if c["type"] == "scale":
            score = _snap(c, a.get("score") or 0)
            row["score"] = score
            row["raw_points"] = score / _max_scale(c) * c["weight"]
        else:
            row["raw_points"] = c["weight"] if ans == "YES" else 0.0

        applicable_weight += c["weight"]
        rows.append(row)

    total = None
    if applicable_weight > 0:
        k = 100.0 / applicable_weight  # перераспределение весов N/A
        for r in rows:
            if r["raw_points"] is not None:
                r["points"] = round(r["raw_points"] * k, 2)
                r["max_points"] = round(r["weight"] * k, 2)
        total = round(sum(r["raw_points"] for r in rows if r["raw_points"] is not None) * k, 2)

    sections = {}
    for r in rows:
        if r["points"] is None:
            continue
        s = sections.setdefault(r["section"], {"points": 0.0, "max": 0.0})
        s["points"] += r["points"]
        s["max"] += r["max_points"]
    for s in sections.values():
        s["points"] = round(s["points"], 2)
        s["max"] = round(s["max"], 2)

    gross = any(r["type"] == "flag" and r["answer"] == "YES" for r in rows)

    return {"criteria": rows, "sections": sections, "total": total,
            "gross_error": gross, "applicable_weight": applicable_weight}


if __name__ == "__main__":
    import json
    cl = json.load(open("checklist.json", encoding="utf-8"))
    test = {"c01": {"answer": "YES"}, "c02": {"answer": "NA"}, "c03": {"answer": "NA"},
            "c04": {"answer": "YES", "score": 15}, "c05": {"answer": "YES"},
            "c06": {"answer": "YES", "score": 7}, "c07": {"answer": "NO"},
            "c08": {"answer": "NO"}}
    print(json.dumps(compute(cl, test), ensure_ascii=False, indent=1)["total"] if False else compute(cl, test)["total"])