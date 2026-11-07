from decimal import Decimal, ROUND_HALF_UP


def r2(x):
    """Округление до двух знаков «половина вверх» (63.125 → 63.13), как в интерфейсе.
    Встроенный round() округляет к чётному (63.125 → 63.12)."""
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


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
                r["points"] = r2(r["raw_points"] * k)
                r["max_points"] = r2(r["weight"] * k)
        total = r2(sum(r["raw_points"] for r in rows if r["raw_points"] is not None) * k)

    sections = {}
    for r in rows:
        if r["points"] is None:
            continue
        s = sections.setdefault(r["section"], {"points": 0.0, "max": 0.0})
        s["points"] += r["points"]
        s["max"] += r["max_points"]
    for s in sections.values():
        s["points"] = r2(s["points"])
        s["max"] = r2(s["max"])

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