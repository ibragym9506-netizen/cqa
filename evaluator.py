import json, os, re, sys, time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scoring import compute
from typing import Optional
from pydantic import BaseModel

load_dotenv()
MODELS = os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1)),
)

SYSTEM = """Ты — строгий эксперт по контролю качества клиентской поддержки брокерской компании Freedom Broker.
Оцени работу ОПЕРАТОРА по чек-листу. Будь строгим: «Да» и высший балл — только если требование выполнено полностью.

Общие правила:
- Оценивается только оператор (speaker = operator). Реплики bot и system — контекст. Приветствие бота НЕ считается приветствием оператора. Системное сообщение «Личность клиента подтверждена» (или аналог на KZ/EN) — подтверждение идентификации.
- Диалог может быть на русском, казахском или английском. Обоснования пиши на русском.
- binary и flag: answer = "YES" или "NO" (или "NA", если N/A разрешён и условие выполнено), score = null.
- scale: score = строго одна из ступеней шкалы, answer = "YES"; если N/A разрешён и условие выполнено — answer = "NA", score = null.
- flag (грубые ошибки): "YES" значит, что нарушение ЕСТЬ.
- «NO» — действие было нужно, но оператор его пропустил или выполнил неправильно. N/A — только по условию неприменимости.

Жёсткие правила для спорных случаев:
- Установление контакта: нужны И приветствие, И имя оператора в его ПЕРВОЙ реплике. «Привет» без имени, или имя без приветствия — NO.
- Идентификация: если оператор сообщил сведения о конкретном счёте клиента (ограничения, позиции, статус поручения) ДО подтверждения личности — NO. Фраза оператора «подтверждение не нужно» при раскрытии данных счёта — NO.
- Раскрытие сведений о счёте без идентификации — это ТАКЖЕ грубая ошибка (раскрытие конфиденциальной информации): флаг YES.
- Грубость, сарказм, унижение клиента («вы читать умеете?») — оскорбление: флаг YES, и эмпатия — NO.
- Предложение перевести деньги оператору лично или дать личный номер для платной консультации — флаг YES (достаточно самого предложения).
- Консультация: пройди по КАЖДОМУ вопросу клиента. Если хотя бы один существенный вопрос остался без ответа или оператор ответил не на тот вопрос — не выше 10. Ошибочная ключевая рекомендация или «не знаю» без следующего шага — 0.
- Эмпатия: если клиент выразил недовольство, беспокойство или неудобство, а оператор не отреагировал на это — NO.
- Управление диалогом: если клиенту пришлось переспрашивать или повторять уже сказанное (оператор не прочитал историю чата) — не выше 4. Если ожидание было явным (оператор ушёл проверять) и он не предупредил — снизь балл.
- Завершение: нужно прощание И (итог ИЛИ предложение дальнейшей помощи). Одно «До свидания»/«Bye» после неотвеченного вопроса клиента — NO. Если оператор закрыл чат без прощания — NO. N/A — только если чат оборвался не по вине оператора (клиент ушёл, связь прервалась).

Звонки:
- Реплики оператора, пока клиент на удержании (между «Клиент находится на удержании» и «Клиент больше не находится на удержании»), — внутренняя консультация с коллегой, клиент их не слышит. Не оценивай их как общение с клиентом, но используй как контекст.
- Транскрипт распознан автоматически, ошибки распознавания не снижают оценку грамотности. [Персональные данные скрыты] — маскировка, это нормально.

Поля ответа:
- reasoning: 1–2 коротких предложения — что именно ты проверил в диалоге. Пиши его ДО вывода.
- evidence: номера реплик [i], на которых основан вывод.
- checked_range: [начало, конец] — участок диалога, который ты проверял (особенно важно, если действие пропущено).
- explanation: итог для человека, например: «Завершение — Нет. После ответа оператор закрыл чат без прощания. Основание — [12], [13].»

Ответ — строго JSON без текста вокруг, по объекту на КАЖДЫЙ критерий чек-листа:
{"criteria":[{"id":"c01","reasoning":"...","answer":"YES","score":null,"evidence":[9],"checked_range":[9,9],"explanation":"..."}]}"""


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


def parse_json(text):
    """Достаёт JSON: снимает обёртку ```json и собирает несколько объектов подряд."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    dec = json.JSONDecoder()
    objs, i = [], 0
    while i < len(text):
        starts = [x for x in (text.find("{", i), text.find("[", i)) if x != -1]
        if not starts:
            break
        i = min(starts)
        try:
            obj, end = dec.raw_decode(text, i)
            objs.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    for o in objs:
        if isinstance(o, dict) and "criteria" in o:
            return o
    items = []
    for o in objs:
        if isinstance(o, list):
            items += [x for x in o if isinstance(x, dict)]
        elif isinstance(o, dict) and "id" in o:
            items.append(o)
    if items:
        return {"criteria": items}
    raise ValueError("Не удалось разобрать JSON модели")


class CritOut(BaseModel):
    id: str
    reasoning: str
    answer: str
    score: Optional[int] = None
    evidence: list[int] = []
    checked_range: list[int] = []
    explanation: str


class EvalOut(BaseModel):
    criteria: list[CritOut]


def call_llm(prompt, need_ids):
    last = None
    for m in MODELS:
        m = m.strip()
        for _ in range(2):
            try:
                r = client.models.generate_content(
                    model=m, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM, temperature=0,
                        response_mime_type="application/json",
                        response_schema=EvalOut))
                data = parse_json(r.text)
                got = {it.get("id") for it in data.get("criteria", [])}
                missing = [c for c in need_ids if c not in got]
                if missing:
                    raise ValueError(f"нет оценок для {missing}")
                u = r.usage_metadata
                usage = {"input_tokens": getattr(u, "prompt_token_count", None),
                         "output_tokens": getattr(u, "candidates_token_count", None)}
                return data, m, usage
            except Exception as e:
                last = e
                print(f"  {m}: {str(e)[:80]}", flush=True)
                time.sleep(3)
    raise RuntimeError(f"Все модели недоступны: {last}")




def evaluate(conv, cl):
    prompt = (f"ЧЕК-ЛИСТ:\n{checklist_text(cl)}\n\n"
              f"ДИАЛОГ {conv['conversation_id']} (канал: {conv['channel']}, язык: {conv['language']}):\n"
              f"{transcript_text(conv)}\n\nВерни JSON.")
    data, model, usage = call_llm(prompt, [c["id"] for c in cl["criteria"]])
    items = data.get("criteria", []) if isinstance(data, dict) else data
    answers = {it["id"]: it for it in items}

    result = compute(cl, answers)
    for r in result["criteria"]:
        a = answers.get(r["id"], {})
        r["explanation"] = a.get("explanation", "")
        r["reasoning"] = a.get("reasoning", "")
        r["evidence"] = a.get("evidence", [])
        r["checked_range"] = a.get("checked_range")
    result.update({"conversation_id": conv["conversation_id"], "checklist_id": cl["id"],
                   "model": model, "usage": usage})
    return result

if __name__ == "__main__":
    cl = json.load(open("checklist.json", encoding="utf-8"))
    ds = json.load(open("dev.json", encoding="utf-8"))["data"]["records"]
    cid = sys.argv[1] if len(sys.argv) > 1 else ds[0]["conversation_id"]
    conv = next(c for c in ds if c["conversation_id"] == cid)

    res = evaluate(conv, cl)
    for r in res["criteria"]:
        print(r["id"], r["answer"], r["score"], r["points"], "|", r["explanation"][:100], "| ev:", r["evidence"])
    print("ИТОГ:", res["total"], "| грубая:", res["gross_error"], "| модель:", res["model"], "| токены:", res["usage"])