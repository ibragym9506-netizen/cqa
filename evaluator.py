import json, os, sys, time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scoring import compute

load_dotenv()
MODELS = os.getenv("GEMINI_MODELS", "gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite").split(",")
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1)),
)

SYSTEM = """Ты — эксперт по контролю качества клиентской поддержки брокерской компании Freedom Broker.
Оцени работу ОПЕРАТОРА по чек-листу.

Правила:
- Оценивается только оператор (speaker = operator). Реплики bot и system — контекст: приветствие бота не считается приветствием оператора; системное сообщение «Личность клиента подтверждена» подтверждает идентификацию.
- Диалог может быть на русском, казахском или английском. Обоснования пиши на русском.
- binary и flag: answer = "YES" или "NO" (или "NA", если N/A разрешён и условие выполнено), score = null.
- scale: score = строго одна из ступеней шкалы, answer = "YES"; если N/A разрешён и условие выполнено — answer = "NA", score = null.
- flag (грубые ошибки): "YES" значит, что нарушение ЕСТЬ.
- «NO» — действие было нужно, но оператор его пропустил или выполнил неправильно. N/A — только по условию неприменимости.
- evidence — номера реплик [i], на которых основан вывод. Если действие пропущено, в checked_range укажи [начало, конец] участка, где оно ожидалось.
- В звонках реплики оператора, пока клиент на удержании (между «Клиент находится на удержании» и «Клиент больше не находится на удержании»), — внутренняя консультация с коллегой, клиент их не слышит.
- Транскрипт звонка распознан автоматически, возможны ошибки распознавания — не снижай за них оценку грамотности.
- explanation: 1–2 предложения, например: «Завершение — Нет. После ответа оператор закрыл чат без прощания. Основание — [12], [13].»

Ответ — строго JSON, по объекту на КАЖДЫЙ критерий чек-листа:
{"criteria":[{"id":"c01","answer":"YES","score":null,"explanation":"...","evidence":[9],"checked_range":[9,11]}]}"""


def checklist_text(cl):
    out = []
    for c in cl["criteria"]:
        s = f'- {c["id"]} «{c["name"]}» (тип: {c["type"]}, вес: {c["weight"]}, N/A разрешён: {"да" if c.get("na_allowed") else "нет"})\n'
        if c.get("description"):
            s += f'  Что оценивать: {c["description"]}\n'
        if c["type"] == "scale":
            s += "  Шкала:\n" + "".join(f'    {x["value"]} — {x["desc"]}\n' for x in c["scale"])
        else:
            s += f'  «Да», если: {c.get("yes")}\n'
        if c.get("na_allowed") and c.get("na"):
            s += f'  N/A, если: {c["na"]}\n'
        out.append(s)
    return "\n".join(out)


def transcript_text(conv):
    return "\n".join(f'[{i}] ({m["timestamp"]}) {m["speaker"]}: {m["text"]}'
                     for i, m in enumerate(conv["transcript"]))


def call_llm(prompt):
    last = None
    for m in MODELS:
        m = m.strip()
        for _ in range(2):
            try:
                r = client.models.generate_content(
                    model=m, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM, temperature=0,
                        response_mime_type="application/json"))
                return json.loads(r.text), m
            except Exception as e:
                last = e
                print(f"  {m}: {str(e)[:80]}", flush=True)
                time.sleep(2)
    raise RuntimeError(f"Все модели недоступны: {last}")


def evaluate(conv, cl):
    prompt = (f"ЧЕК-ЛИСТ:\n{checklist_text(cl)}\n\n"
              f"ДИАЛОГ {conv['conversation_id']} (канал: {conv['channel']}, язык: {conv['language']}):\n"
              f"{transcript_text(conv)}\n\nВерни JSON.")
    data, model = call_llm(prompt)
    items = data.get("criteria", []) if isinstance(data, dict) else data
    answers = {it["id"]: it for it in items}

    result = compute(cl, answers)
    for r in result["criteria"]:
        a = answers.get(r["id"], {})
        r["explanation"] = a.get("explanation", "")
        r["evidence"] = a.get("evidence", [])
        r["checked_range"] = a.get("checked_range")
    result.update({"conversation_id": conv["conversation_id"],
                   "checklist_id": cl["id"], "model": model})
    return result


if __name__ == "__main__":
    cl = json.load(open("checklist.json", encoding="utf-8"))
    ds = json.load(open("dev.json", encoding="utf-8"))["data"]["records"]
    cid = sys.argv[1] if len(sys.argv) > 1 else ds[0]["conversation_id"]
    conv = next(c for c in ds if c["conversation_id"] == cid)

    res = evaluate(conv, cl)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{cid}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for r in res["criteria"]:
        print(r["id"], r["answer"], r["score"], r["points"], "|", r["explanation"][:100], "| ev:", r["evidence"])
    print("ИТОГ:", res["total"], "| грубая ошибка:", res["gross_error"], "| модель:", res["model"])