import json, os, re, sys, time
from collections import Counter
from statistics import median_low
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from scoring import compute

load_dotenv()
MODELS = os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
VOTES = int(os.getenv("VOTES", "3"))  # VOTES=1 — быстрый прогон без голосования
# в .env можно GEMINI_API_KEY=ключ или несколько через запятую (GEMINI_API_KEY или GEMINI_API_KEYS)
_KEY0 = (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or "").split(",")[0].strip() or None
client = genai.Client(
    api_key=_KEY0,
    http_options=types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1)),
)

SYSTEM = """Ты — строгий эксперт по контролю качества клиентской поддержки брокерской компании Freedom Broker.
Оцени работу ОПЕРАТОРА по чек-листу. Будь строгим: «Да» и высший балл — только если требование выполнено полностью.

Общие правила:
- Оценивай КАЖДЫЙ критерий строго по его тексту из чек-листа. Если у критерия есть «Уточнения эксперта» — они обязательны. Новый или изменённый критерий оценивай только по его тексту.
- Оценивается только оператор (speaker = operator). Реплики bot и system — контекст. Приветствие бота НЕ считается приветствием оператора. Системное сообщение «Личность клиента подтверждена» (или аналог на KZ/EN) — подтверждение идентификации.
- Один и тот же чек-лист применяется к диалогам на русском, казахском и английском — правила одинаковы для всех языков. Обоснования пиши на русском.
- Перед диалогом даны ФАКТЫ, извлечённые кодом (первая реплика оператора, язык клиента, паузы, конец диалога). Они точны — опирайся на них.
- binary и flag: answer = "YES" или "NO" (или "NA", если N/A разрешён и условие выполнено), score = null.
- scale: score = строго одна из ступеней шкалы, answer = "YES"; если N/A разрешён и условие выполнено — answer = "NA", score = null.
- flag: "YES" значит, что нарушение ЕСТЬ.
- «NO» — действие было нужно, но оператор его пропустил или выполнил неправильно. «NA» — критерий неприменим по своему условию N/A; в explanation назови, какое условие выполнено.
- Не смешивай критерии: консультация — правильно и полно ли решён запрос; ведение диалога — насколько понятно и последовательно общался оператор; эмпатия — тон и реакция на проблему клиента.
- По условию задания НЕ оцениваются: время первой реакции оператора, теги, вложения.
- Для пропущенного действия укажи в checked_range участок, где оно ожидалось.
Звонки:
- Реплики с пометкой «удержание» — внутренняя консультация с коллегой, клиент их не слышит. Не оценивай их как общение с клиентом, но используй как контекст.
- Транскрипт распознан автоматически: ошибки распознавания не снижают оценку грамотности. [Персональные данные скрыты] — маскировка, это нормально.
- Звонок длинный: проверь его целиком, до последней реплики, особенно поздние вопросы и жалобы клиента.

Поля ответа:
- quote: дословная цитата (до 20 слов) самой важной реплики для этого критерия, с номером, например «[9] Сәлем. Қай сайтқа кірдіңіз?». Пиши её ПЕРВОЙ.
- reasoning: 1–3 коротких предложения — что именно ты проверил. Пиши ДО вывода.
- evidence: номера реплик [i], на которых основан вывод.
- checked_range: [начало, конец] — участок диалога, который ты проверял.
- explanation: итог для человека, например: «Завершение — Нет. После ответа оператор закрыл чат без прощания. Основание — [12], [13].»
- В reasoning и explanation ссылайся на другие критерии по НАЗВАНИЮ («см. Консультация»), а не по id.

Ответ — строго JSON без текста вокруг, по объекту на КАЖДЫЙ критерий чек-листа:
{"criteria":[{"id":"<id критерия из чек-листа>","quote":"[9] ...","reasoning":"...","answer":"YES","score":null,"evidence":[9],"checked_range":[9,9],"explanation":"..."}]}
В поле id пиши ТОЧНО тот id, который указан у критерия в чек-листе."""


# ---------- уточнения к стандартным критериям ----------
# Подключаются только если критерий в чек-листе по смыслу остался прежним:
# ключ в названии + опорное слово в тексте. Заменённый или переписанный критерий
# оценивается только по своему тексту (и полю "rules", если его задали в интерфейсе).

HINTS = [
    ("контакт", "представ",
     "В ПЕРВОЙ реплике оператора нужны И приветствие на языке клиента, И имя оператора. «Привет»/«Сәлем»/«Hi» без имени "
     "или имя без приветствия — NO. Имя под маской [Имя скрыто] или искажённое распознаванием в звонке считается названным."),
    ("потребност", "уточн",
     "NO — оператор переспросил то, что клиент уже написал (клиент отвечает «я уже писал», «жоғарыда жаздым», «I already said»), "
     "или дал инструкцию наугад, не уточнив важную неизвестную деталь (приложение или сайт, устройство, какая бумага), и она "
     "не подошла — даже если потом исправился. YES — задал уточняющий вопрос и учёл ответ. N/A — уточнять было нечего. "
     "Неверная инструкция при полностью известных деталях — это ошибка консультации или ведения диалога, а не этого критерия."),
    ("идентификац", "личност",
     "Общий вопрос (тарифы, где найти раздел, как пользоваться приложением) не требует идентификации — N/A. Если требовалась "
     "проверка личности перед защищённым действием и она не подтверждена — NO. Если оператор сообщил сведения о конкретном "
     "счёте (ограничения, позиции, статус поручения) ДО подтверждения личности — NO; если клиент возразил, а оператор "
     "отмахнулся («подтверждение не нужно», «We can skip that», «Қазір растаудың қажеті жоқ») — тоже NO. YES — подтверждение "
     "(системное сообщение или проверка данных в звонке) прошло до раскрытия данных или защищённого действия. "
     "Раскрытие — это сведения, взятые из счёта клиента («вижу ограничение на вашем счёте», «ваше поручение в обработке»). "
     "Догадка оператора без просмотра счёта («наверное, биржа отменяет») — не раскрытие, это ошибка консультации."),
    ("консультац", "существенн",
     "В reasoning перечисли ВСЕ вопросы клиента к оператору и отметь, ответил ли оператор по существу на каждый. Реплика, "
     "которая повторяет уже сказанное или говорит о другом («Путь к обновлению я указал», «Название тарифа смотрите там же»), — "
     "не ответ. 20 — все вопросы закрыты или дан корректный следующий шаг (в т.ч. передача специалистам с понятным каналом "
     "ответа). 15 — основной вопрос решён, второстепенный вопрос закрыт не полностью. 10 — существенный вопрос остался без "
     "ответа. 5 — только общая подсказка. 0 — ничего по существу, «не знаю» без шага или ошибочная ключевая рекомендация; если "
     "хоть часть запроса решена — не 0. Отказ без альтернативы («мы этого не делаем», «торгуйте сами») — не выше 10. Ответ "
     "с нарушением регламента (данные счёта без идентификации) — не выше 10. Клиент сам отложил вопрос, а оператор дал "
     "корректный следующий шаг — 20, а не N/A."),
    ("эмпати", "недовольств",
     "Грубость, сарказм, пренебрежение («Вы вообще читать умеете?», «Всё есть в приложении») — NO. Если клиент выразил "
     "недовольство, беспокойство или описал неудобство («надоело», «запутался», «отфутболили», «навязываюсь», «не могу звонить "
     "при коллегах», «nervous», возражение о безопасности своих данных), оператор должен отреагировать на ЭТУ проблему: "
     "признать её или предложить решение именно для неё. Дежурное «понимаю, но…», перекладывание вины («вас неправильно "
     "проинформировали»), сухой отказ, отмахивание — NO. Если поводов не было, достаточно вежливости. Предложение платной "
     "услуги и раскрытие данных сами по себе эмпатию не снижают — это грубые ошибки: реплику с таким предложением при оценке "
     "эмпатии пропусти и оцени остальное общение."),
    ("управлени", "переспраш",
     "Оценивай ясность, связность, порядок ответов и предупреждение об ожидании — НЕ полноту ответа (её оценивает "
     "консультация: ясный ответ на один из двух вопросов может получить высокий балл здесь и частичный в консультации). "
     "Неотвеченный вопрос, ответ не на тот вопрос или повтор уже сказанного здесь НЕ снижают балл — за это уже снижена "
     "консультация; снижай, только если клиенту пришлось переспросить из-за неясной формулировки оператора. "
     "4 — клиенту пришлось переспросить или поправить оператора из-за его формулировки или ошибки («В каком меню?», «Какое "
     "приложение?», «Settings where?», «Профильде ондай бөлім жоқ», «Не, я имею в виду…»). 7 — один заметный сбой, который "
     "клиенту не пришлось исправлять (неточность, долгая пауза без предупреждения — см. ФАКТЫ). 0 — противоречивые реплики "
     "(сначала «точно», потом «не знаю»), ход диалога непонятен. Время первой реакции не оценивается."),
    ("завершени", "попрощ",
     "YES — оператор попрощался («До свидания», «Хорошего дня», «Сау болыңыз», «Take care», «Bye») после того, как вопросы "
     "клиента решены или переданы дальше; отдельный итог не обязателен. NO — прощания нет, оператор закрыл чат без прощания, "
     "или попрощался, оставив последний вопрос клиента без ответа. N/A — клиент ушёл, связь прервалась или клиент сам "
     "завершил звонок до того, как оператор мог попрощаться."),
    ("груб", "оскорб",
     "Флаг YES: оскорбление или сарказм; предложение перевести деньги оператору лично или просьба дать личный номер для "
     "платной консультации в частном порядке (достаточно самого предложения, согласие клиента не нужно); раскрытие сведений "
     "о счёте без подтверждённой идентификации."),
]


def _ctext(c):
    parts = [c.get(k) or "" for k in ("name", "yes", "na", "description")]
    parts += [s.get("desc", "") for s in c.get("scale") or []]
    return " ".join(str(p) for p in parts).lower()


def _matches(c, key, must):
    return key in (c.get("name") or "").lower() and must in _ctext(c)


def hint_for(c):
    if c.get("rules"):  # уточнения, заданные экспертом в интерфейсе, важнее встроенных
        return c["rules"]
    for key, must, text in HINTS:
        if _matches(c, key, must):
            return text
    return ""


def find_crit(cl, key, must):
    return next((c for c in cl["criteria"] if _matches(c, key, must)), None)


# ---------- подготовка транскрипта ----------

HOLD_ON = ("клиент находится на удержании", "on hold", "ұстауда")
HOLD_OFF = ("больше не находится на удержании", "no longer on hold", "ұстаудан шықты")
ID_OK = ("личность клиента подтверждена", "identity verified", "жеке басы расталды")
OPERATOR_CLOSED = ("оператор закрыл чат", "agent closed the chat", "оператор чатты жапты")
CLIENT_GONE = ("клиент покинул чат", "соединение с клиентом прервано", "клиент завершил разговор",
               "клиентпен байланыс үзілді", "connection to customer lost", "customer left")
WAIT_WORDS = ("провер", "минут", "секунд", "подожд", "ожида", "уточн", "check", "moment", "minute", "wait",
              "hold", "күте", "тексер")
PAUSE_SEC = 120  # пауза внутри диалога, о которой сообщаем модели


def _low(m):
    return (m.get("text") or "").lower()


def _has(text, words):
    return any(w in text for w in words)


def _short_ts(ts):
    ts = ts or ""
    return ts[11:19] if "T" in ts else ts


def _secs(ts):
    """Секунды от начала суток (звонки, ISO) или от начала чата (MM:SS)."""
    parts = _short_ts(ts).split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return None
    total = 0
    for n in nums:
        total = total * 60 + n
    return total


def _bot_text(text):
    """Длинные шаблоны бота заменяем короткой меткой — номера реплик сохраняются."""
    t = text.lower()
    if "aim" in t and len(text) > 150:
        return "(стандартное представление бота AIM)"
    if _has(t, ("политикой конфиденциальности", "privacy policy", "құпиялылық саясат")):
        return "(согласие на обработку персональных данных)"
    return text


def hold_flags(conv):
    """True для реплик, сказанных, пока клиент на удержании."""
    flags, on = [], False
    for m in conv["transcript"]:
        t = _low(m)
        if m["speaker"] == "system" and _has(t, HOLD_OFF):
            on = False
        elif m["speaker"] == "system" and _has(t, HOLD_ON):
            on = True
        flags.append(on)
    return flags


def transcript_text(conv):
    hold = hold_flags(conv)
    out = []
    for i, m in enumerate(conv["transcript"]):
        sp, text = m["speaker"], m.get("text") or ""
        if sp == "bot":
            text = _bot_text(text)
        if sp == "operator" and hold[i]:
            sp = "operator (удержание, клиент не слышит)"
        out.append(f'[{i}] ({_short_ts(m.get("timestamp"))}) {sp}: {text}')
    return "\n".join(out)


def operator_lines(conv):
    hold = hold_flags(conv)
    return [(i, m["text"]) for i, m in enumerate(conv["transcript"])
            if m["speaker"] == "operator" and not hold[i]]


def detect_lang(text):
    letters = re.findall(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", text or "")
    if not letters:
        return None
    if sum(ch.isascii() for ch in letters) / len(letters) > 0.6:
        return "EN"
    if re.search(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]", text):
        return "KZ"
    return "RU"


def client_lang(conv):
    text = " ".join(m.get("text") or "" for m in conv["transcript"] if m["speaker"] == "client")
    return detect_lang(text) or (conv.get("language") or "").upper() or None


def pauses(conv):
    """Паузы перед ответом оператора внутри диалога (без первой реакции и удержаний)."""
    tr, hold = conv["transcript"], hold_flags(conv)
    res, last_t, last_op, seen_op, held = [], None, None, False, False
    for i, m in enumerate(tr):
        sp = m["speaker"]
        if sp == "system":
            if _has(_low(m), HOLD_ON):
                held = True
            continue
        if sp == "operator" and hold[i]:
            continue
        if sp not in ("client", "operator"):
            continue
        t = _secs(m.get("timestamp"))
        if sp == "operator":
            if seen_op and last_t is not None and t is not None and not held and t - last_t >= PAUSE_SEC:
                warned = last_op is not None and _has(last_op[1].lower(), WAIT_WORDS)
                res.append((i, t - last_t, last_op if warned else None))
            seen_op, last_op, held = True, (i, m.get("text") or ""), False
        if t is not None:
            last_t = t
    return res


def facts(conv):
    tr = conv["transcript"]
    ops = operator_lines(conv)
    f = []
    lang = client_lang(conv)
    if lang:
        f.append(f"- Язык клиента: {lang}")
    if ops:
        f.append(f"- Первая реплика оператора: [{ops[0][0]}] {ops[0][1]}")
        f.append(f"- Последняя реплика оператора: [{ops[-1][0]}] {ops[-1][1]}")
    else:
        f.append("- Оператор ничего не сказал клиенту.")
    cl = [(i, m["text"]) for i, m in enumerate(tr) if m["speaker"] == "client"]
    if cl:
        f.append(f"- Последняя реплика клиента: [{cl[-1][0]}] {cl[-1][1]}")
    idok = [i for i, m in enumerate(tr) if m["speaker"] == "system" and _has(_low(m), ID_OK)]
    f.append(f"- Системное подтверждение личности: {', '.join(f'[{i}]' for i in idok) if idok else 'нет'}")
    ends = [(i, m["text"]) for i, m in enumerate(tr)
            if m["speaker"] == "system" and _has(_low(m), OPERATOR_CLOSED + CLIENT_GONE)]
    if ends:
        f.append(f"- Событие завершения: [{ends[-1][0]}] {ends[-1][1]}")
    hold = hold_flags(conv)
    if any(hold):
        f.append(f"- Клиент на удержании в репликах [{hold.index(True)}]–[{len(hold) - 1 - hold[::-1].index(True)}]")
    for i, sec, warn in pauses(conv):
        w = f"оператор предупредил в [{warn[0]}] «{warn[1][:60]}»" if warn else "предупреждения не было"
        f.append(f"- Пауза перед репликой оператора [{i}]: {sec} с, {w}")
    return "\n".join(f)


# ---------- детерминированные правила ----------

GREET = {
    "RU": re.compile(r"здравствуй|добр(ый|ое|ого)\s+(день|утр|вечер|времени)|приветствую|привет", re.I),
    "KZ": re.compile(r"сәлем|қайырлы", re.I),
    "EN": re.compile(r"\bhi\b|\bhello\b|\bhey\b|good (morning|afternoon|evening|day)", re.I),
}
BYE = re.compile(r"до свидания|всего (доброго|хорошего)|хорошего (дня|вечера)|сау бол|күніңіз сәтті|"
                 r"(благодар\w*|спасибо)( вас)? за обращение|\bbye\b|goodbye|take care|have a (good|nice|great)|"
                 r"thanks for (reaching out|contacting)", re.I)
LANG_NAME = {"RU": "русском", "KZ": "казахском", "EN": "английском"}


def _name_found(text):
    if re.search(r"\[имя скрыто\]|меня зовут|менің атым|my name is", text, re.I):
        return True
    # «я Лера», «мен Дана», «I'm Sam» — имя с заглавной буквы
    return bool(re.search(r"(?:^|[\s,.!])(?:[яЯ]|[мМ]ен|I'm|I am|[тТ]his is|[эЭ]то)\s+"
                          r"[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][a-zа-яёәғқңөұүһі]+", text))


def code_rules(conv, cl):
    """{crit_id: (answer, пояснение)} — решения, которые код принимает сам.
    Каждое правило работает, только если соответствующий критерий в чек-листе остался по смыслу прежним."""
    rules = {}
    ops = operator_lines(conv)
    tr = conv["transcript"]

    # Установление контакта: приветствие (на языке клиента) + имя в первой реплике оператора
    c = find_crit(cl, "контакт", "представ")
    if c and ops:
        i, text = ops[0]
        langs = [lg for lg, rx in GREET.items() if rx.search(text)]
        name = _name_found(text)
        lang = client_lang(conv)
        need_lang = "на языке" in _ctext(c)
        if langs and name and (not need_lang or not lang or lang in langs):
            rules[c["id"]] = ("YES", f"{c['name']} — Да. В первой реплике есть приветствие и имя оператора. Основание — [{i}].")
        elif langs and name:
            rules[c["id"]] = ("NO", f"{c['name']} — Нет. Оператор поздоровался не на языке клиента "
                                    f"({LANG_NAME.get(lang, lang)}). Основание — [{i}].")
        else:
            miss = " и ".join(x for x, ok in (("приветствия", bool(langs)), ("имени оператора", name)) if not ok)
            rules[c["id"]] = ("NO", f"{c['name']} — Нет. В первой реплике оператора нет {miss}. Основание — [{i}].")

    # Завершение
    c = find_crit(cl, "завершени", "попрощ")
    if c and ops:
        ends = [(i, _low(m)) for i, m in enumerate(tr) if m["speaker"] == "system"
                and _has(_low(m), OPERATOR_CLOSED + CLIENT_GONE)]
        if ends:
            ei, et = ends[-1]
            before = [(i, t) for i, t in ops if i < ei]
            said_bye = bool(before) and bool(BYE.search(before[-1][1]))
            # диалог оборвался «на ходу»: после последней реплики клиента/оператора прошло не больше 90 с
            last_msg = [m for j, m in enumerate(tr) if m["speaker"] in ("client", "operator") and j < ei]
            t_end = _secs(tr[ei].get("timestamp"))
            t_msg = _secs(last_msg[-1].get("timestamp")) if last_msg else None
            quick = t_end is None or t_msg is None or t_end - t_msg <= 90
            if _has(et, CLIENT_GONE) and not said_bye and quick and c.get("na_allowed"):
                rules[c["id"]] = ("NA", f"{c['name']} — N/A. Клиент ушёл или связь прервалась до того, как оператор мог "
                                        f"попрощаться. Основание — [{ei}].")
            elif _has(et, OPERATOR_CLOSED) and not said_bye:
                rules[c["id"]] = ("NO", f"{c['name']} — Нет. Оператор закрыл чат без прощания. Основание — последняя "
                                        f"реплика оператора [{before[-1][0] if before else ei}] и сообщение о закрытии [{ei}].")
        elif not BYE.search(ops[-1][1]):
            # чат никто не обрывал, а прощания от оператора так и не было
            rules[c["id"]] = ("NO", f"{c['name']} — Нет. Оператор не попрощался: последняя реплика оператора "
                                    f"[{ops[-1][0]}] без прощания, обрыва связи не было.")
        if c["id"] not in rules:
            oi, ot = ops[-1]
            prev = [(i, m["text"]) for i, m in enumerate(tr) if m["speaker"] == "client" and i < oi]
            bare_bye = BYE.search(ot) and len(BYE.sub("", ot).strip(" .!,")) <= 3
            if prev and prev[-1][1].strip().endswith("?") and bare_bye:
                rules[c["id"]] = ("NO", f"{c['name']} — Нет. Оператор попрощался, не ответив на вопрос клиента "
                                        f"[{prev[-1][0]}]. Основание — [{oi}].")
    return rules


DISCLOSED = re.compile(r"раскры|сообщил|озвучил|disclos", re.I)

# клиент переспрашивает / поправляет оператора сразу после его реплики (ступень «клиенту приходится переспрашивать»)
REASK = re.compile(
    r"в как(ом|ой) (меню|разделе|месте|вкладке)|как(ое|ой|ую) (приложение|сайт|меню|раздел|кнопк)|"
    r"что (вы )?имеете в виду|не понял|не понимаю,? (где|что|куда|на как|о ч)|там (нет|такого нет)|"
    r"такого (раздела|пункта|меню|нет)|нет такого|кнопки[^.?!]{0,20}нет|я на компьютере|не, я имею в виду|"
    r"ондай[^.?!]{0,20}жоқ|бөлім жоқ|түсінбедім|қай (жерде|мәзір|бөлім)|"
    r"where exactly|settings where|what do you mean|there'?s no (login|such|button|sign)|which (menu|app|page|section)|"
    r"i'?m on the (company )?website|i don'?t see (it|that|such)|not sure what you mean", re.I)
# модель снизила ведение именно за неотвеченный вопрос (по памятке это снижает только консультацию)
UNANSWERED = re.compile(r"не ответил|без ответа|игнорир|пропустил[^.]{0,30}вопрос|не на тот вопрос|"
                        r"оставил[^.]{0,30}вопрос|вопрос[^.]{0,30}(остал|не закрыт|не получил)|отказал[^.]{0,20}поясн", re.I)
UNCLEAR = re.compile(r"переспр|неясн|непонятн|расплывч|путан|формулировк|поправ", re.I)
PAID = re.compile(r"скиньте деньги|мне на карту|личный номер|за плату|частн\w* консультац|жеке нөмір|ақылы|"
                  r"personal number|i charge|private advice|private consult|send me (the )?money", re.I)
OFFER_WORDS = re.compile(r"платн|обход|личн|частн|карт|деньг|ақылы|paid|private|personal", re.I)
INDIGNANT = re.compile(r"серь[её]зно|шутите|так говорить не|seriously|are you kidding|шын ба", re.I)


def _reask(conv):
    """Номер реплики клиента, которой он переспросил/поправил оператора сразу после его ответа."""
    tr, hold, last_op = conv["transcript"], hold_flags(conv), None
    for i, m in enumerate(tr):
        if m["speaker"] == "operator" and not hold[i]:
            last_op = i
        elif m["speaker"] == "client" and last_op is not None:
            if REASK.search(m.get("text") or ""):
                return i
            last_op = None  # смотрим только первую реплику клиента после ответа оператора
    return None


def _step(c, word):
    """Значение ступени шкалы, в описании которой есть слово (например, «переспраш»)."""
    return next((s["value"] for s in c.get("scale") or [] if word in (s.get("desc") or "").lower()), None)


def apply_rules(conv, cl, answers):
    notes = []
    rules = code_rules(conv, cl)

    # «Попрощался, оставив последний вопрос клиента без ответа»: клиент закончил вопросом,
    # оператор ответил прощанием, а консультация признала существенный вопрос открытым (≤ 10)
    comp, cons = find_crit(cl, "завершени", "попрощ"), find_crit(cl, "консультац", "существенн")
    ops = operator_lines(conv)
    if comp and cons and ops and comp["id"] not in rules:
        sc = answers.get(cons["id"], {})
        oi, ot = ops[-1]
        prev = [(i, m["text"]) for i, m in enumerate(conv["transcript"]) if m["speaker"] == "client" and i < oi]
        passed_on = re.search(r"переда|специалист|отдел|жолда|жібер|тексеруге|send it|pass|escalat|team", ot, re.I)
        if sc.get("answer") == "YES" and (sc.get("score") or 0) <= 10 and BYE.search(ot) and not passed_on \
                and prev and prev[-1][1].strip().endswith("?"):
            rules[comp["id"]] = ("NO", f"{comp['name']} — Нет. Оператор попрощался, оставив вопрос клиента "
                                       f"[{prev[-1][0]}] без ответа (см. «{cons['name']}»). Основание — [{oi}].")

    for cid, (ans, expl) in rules.items():
        if cid in answers and answers[cid].get("answer") != ans:
            notes.append(f"{cid}: модель {answers[cid].get('answer')} → правило {ans}")
            answers[cid] = dict(answers[cid], answer=ans, score=None, explanation=expl)

    # Ведение диалога: клиент переспросил из-за формулировки оператора → ступень «переспрашивать»;
    # снижено только за неотвеченный вопрос (это уже учтено в консультации) → верхняя ступень
    dlg = find_crit(cl, "управлени", "переспраш")
    if dlg and dlg["type"] == "scale" and dlg["id"] in answers and answers[dlg["id"]].get("answer") == "YES":
        a, low, top = answers[dlg["id"]], _step(dlg, "переспраш"), max(s["value"] for s in dlg["scale"])
        ri, sc = _reask(conv), a.get("score")
        if ri is not None and low is not None and sc is not None and sc > low:
            notes.append(f"{dlg['id']}: модель {sc} → правило {low}")
            answers[dlg["id"]] = dict(a, score=low, explanation=f"{dlg['name']} — {low}. Клиенту пришлось переспросить или "
                                      f"поправить оператора после его ответа. Основание — [{ri}].")
        elif ri is None and sc is not None and top > sc > (low or 0):
            why = " ".join(str(a.get(k) or "") for k in ("explanation", "reasoning"))
            if UNANSWERED.search(why) and not UNCLEAR.search(a.get("explanation") or ""):
                notes.append(f"{dlg['id']}: модель {sc} → правило {top}")
                answers[dlg["id"]] = dict(a, score=top, explanation=f"{dlg['name']} — {top}. Ответы ясные; неотвеченный вопрос "
                                          f"учтён в консультации, а не здесь (памятка организаторов). " + (a.get("explanation") or ""))

    # Эмпатия: предложение платной услуги само по себе эмпатию не снижает (это грубая ошибка)
    emp = find_crit(cl, "эмпати", "недовольств")
    if emp and emp["id"] in answers and answers[emp["id"]].get("answer") == "NO":
        tr = conv["transcript"]
        offer = next((i for i, t in ops if PAID.search(t)), None)
        why = " ".join(str(answers[emp["id"]].get(k) or "") for k in ("explanation", "reasoning"))
        later = [m.get("text") or "" for m in tr[offer + 1:] if m["speaker"] == "client"] if offer is not None else []
        if offer is not None and OFFER_WORDS.search(why) and not any(INDIGNANT.search(t) for t in later):
            notes.append(f"{emp['id']}: модель NO → правило YES")
            answers[emp["id"]] = dict(answers[emp["id"]], answer="YES", explanation=f"{emp['name']} — Да. Оператор вежлив; "
                                      f"предложение платной услуги [{offer}] — это грубая ошибка, эмпатию само по себе "
                                      f"не снижает (памятка организаторов).")

    # раскрытие данных до идентификации — всегда грубая ошибка
    # (только если «Нет» по идентификации поставлено именно за раскрытие сведений)
    idc = find_crit(cl, "идентификац", "личност")
    flag = next((c for c in cl["criteria"] if c["type"] == "flag" and "раскрыт" in _ctext(c)), None)
    ida = answers.get(idc["id"], {}) if idc else {}
    why = " ".join(str(ida.get(k) or "") for k in ("explanation", "reasoning"))
    if idc and flag and ida.get("answer") == "NO" and DISCLOSED.search(why) \
            and answers.get(flag["id"], {}).get("answer") != "YES":
        notes.append(f"{flag['id']}: модель NO → правило YES ({idc['id']} = NO)")
        answers[flag["id"]] = dict(answers[flag["id"]], answer="YES",
                                   evidence=answers[idc["id"]].get("evidence", []),
                                   explanation=f"{flag['name']} — Да. Сведения о счёте раскрыты до идентификации "
                                               f"(см. «{idc['name']}»).")
    return notes


# ---------- LLM ----------

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
        h = hint_for(c)
        if h:
            s += f'  Уточнения эксперта: {h}\n'
        out.append(s)
    return "\n".join(out)


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
    quote: str = ""
    reasoning: str
    answer: str
    score: Optional[int] = None
    evidence: list[int] = []
    checked_range: list[int] = []
    explanation: str


class EvalOut(BaseModel):
    criteria: list[CritOut]


# Ключи: GEMINI_API_KEYS=ключ1,ключ2,… (у каждого ключа своя квота, если он из отдельного проекта Google).
KEYS = [k.strip() for k in (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or "").split(",") if k.strip()]
KEYS = list(dict.fromkeys(KEYS))  # без повторов
CLIENTS = [genai.Client(api_key=k, http_options=types.HttpOptions(
    timeout=90000, retry_options=types.HttpRetryOptions(attempts=1))) for k in KEYS[1:]]
CLIENTS = [client] + CLIENTS
DAY_DONE = set()  # (модель, номер ключа), у которых кончился дневной лимит


def _limit_kind(e):
    """'key' — ключ не принят, 'day' — дневной лимит, 'minute' — минутный, None — другая ошибка."""
    s = str(e)
    if "401" in s or "UNAUTHENTICATED" in s or "API key not valid" in s or "PERMISSION_DENIED" in s:
        return "key"
    if "429" not in s and "RESOURCE_EXHAUSTED" not in s and "quota" not in s.lower():
        return None
    return "day" if ("PerDay" in s or "per day" in s.lower()) else "minute"


def call_llm(prompt, need_ids):
    """Порядок: основная модель на всех ключах, потом запасная. Минутный лимит — ждём и повторяем
    ту же модель (не уходим на более слабую); дневной — следующий ключ, затем следующая модель."""
    last = None
    for m in MODELS:
        m = m.strip()
        for ki, cli in enumerate(CLIENTS):
            if (m, ki) in DAY_DONE:
                continue
            waits = fails = 0
            while True:
                try:
                    return _ask(cli, m, prompt, need_ids)
                except Exception as e:
                    last, kind = e, _limit_kind(e)
                    print(f"  {m} (ключ {ki + 1}): {str(e)[:80]}", flush=True)
                    if kind == "day":
                        DAY_DONE.add((m, ki)); break
                    if kind == "key":  # неверный ключ — больше его не трогаем ни для какой модели
                        print(f"  ключ {ki + 1} не принят Google — пропускаю его", flush=True)
                        DAY_DONE.update((x.strip(), ki) for x in MODELS); break
                    if kind == "minute" and waits < 4:
                        waits += 1
                        time.sleep(15 * waits)  # 15, 30, 45, 60 с — минутное окно освобождается
                        continue
                    fails += 1
                    if fails >= 2 or kind == "minute":
                        break
                    time.sleep(3)
    raise RuntimeError(f"Все модели недоступны: {last}")


def _ask(cli, m, prompt, need_ids):
    """Один запрос к модели; ошибка формата ответа — исключение."""
    r = cli.models.generate_content(
        model=m, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0,
            response_mime_type="application/json",
            response_schema=EvalOut))
    data = parse_json(r.text)
    items = data.get("criteria", [])
    got = {it.get("id") for it in items}
    # модель перепутала id (например, c01 вместо greeting), но вернула все критерии по порядку
    if not (got & set(need_ids)) and len(items) == len(need_ids):
        for it, cid in zip(items, need_ids):
            it["id"] = cid
        got = set(need_ids)
    missing = [c for c in need_ids if c not in got]
    if missing:
        raise ValueError(f"нет оценок для {missing}")
    u = r.usage_metadata
    usage = {"input_tokens": getattr(u, "prompt_token_count", None),
             "output_tokens": getattr(u, "candidates_token_count", None)}
    return data, m, usage


def vote(prompt, ids, llm=None, votes=None):
    llm = llm or call_llm
    runs, model = [], None
    usage = {"input_tokens": 0, "output_tokens": 0}
    for _ in range(votes or VOTES):
        data, model, u = llm(prompt, ids)
        items = data.get("criteria", []) if isinstance(data, dict) else data
        for it in items:
            it["answer"] = str(it.get("answer", "")).upper()
            if it.get("score") is not None and it["answer"] not in ("YES", "NO", "NA"):
                it["answer"] = "YES"
        runs.append({it["id"]: it for it in items})
        for k in usage:
            usage[k] += u.get(k) or 0
    answers, agreement = {}, {}
    key = lambda it: (it["answer"], it.get("score"))
    for cid in ids:
        its = [r[cid] for r in runs if cid in r]
        top = Counter(it["answer"] for it in its).most_common(1)[0][0]
        same = [it for it in its if it["answer"] == top]
        sc = [it["score"] for it in same if it.get("score") is not None]
        pick = next(it for it in same if it.get("score") == median_low(sc)) if sc else same[0]
        answers[cid] = dict(pick)
        agreement[cid] = round(sum(key(it) == key(pick) for it in its) / len(its), 2)
    return answers, model, usage, agreement


def build_prompt(conv, cl, extra=""):
    ids = ", ".join(c["id"] for c in cl["criteria"])
    return (f"ЧЕК-ЛИСТ (id критериев: {ids}):\n{checklist_text(cl)}\n\n"
            f"ФАКТЫ (извлечены кодом):\n{facts(conv)}\n\n"
            f"ДИАЛОГ {conv['conversation_id']} (канал: {conv['channel']}, язык: {conv['language']}):\n"
            f"{transcript_text(conv)}\n\n" + (f"{extra}\n\n" if extra else "") + "Верни JSON.")


def evaluate(conv, cl, llm=None, extra="", votes=None):
    ids = [c["id"] for c in cl["criteria"]]
    answers, model, usage, agreement = vote(build_prompt(conv, cl, extra), ids, llm, votes)
    rule_notes = apply_rules(conv, cl, answers)

    result = compute(cl, answers)
    # служебные id критериев (consultation, c04…) в текстах для человека → названия
    names = {c["id"]: c.get("name") or c["id"] for c in cl["criteria"]}
    id_re = re.compile(r"(?<![\w«])(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?![\w»])")
    human = lambda s: id_re.sub(lambda m: f"«{names[m.group(1)]}»", s or "")
    for r in result["criteria"]:
        a = answers.get(r["id"], {})
        r["explanation"] = human(a.get("explanation", ""))
        r["reasoning"] = human(a.get("reasoning", ""))
        r["quote"] = a.get("quote", "")
        r["evidence"] = a.get("evidence", [])
        r["checked_range"] = a.get("checked_range")
    disputed = [c for c, v in agreement.items() if v < 1]
    flag_ids = [c["id"] for c in cl["criteria"] if c["type"] == "flag"]
    result.update({"conversation_id": conv["conversation_id"], "checklist_id": cl["id"],
                   "model": model, "usage": dict(usage, calls=votes or VOTES), "agreement": agreement,
                   "rule_overrides": rule_notes,
                   "needs_review": len(disputed) >= 2 or any(f in disputed for f in flag_ids)})
    return result


if __name__ == "__main__":
    cl = json.load(open("checklist.json", encoding="utf-8"))
    ds = json.load(open("dev.json", encoding="utf-8"))["data"]["records"]
    cid = sys.argv[1] if len(sys.argv) > 1 else ds[0]["conversation_id"]
    conv = next(c for c in ds if c["conversation_id"] == cid)

    res = evaluate(conv, cl)
    for r in res["criteria"]:
        print(r["id"], r["answer"], r["score"], r["points"], res["agreement"][r["id"]], "|", r["explanation"][:100], "| ev:", r["evidence"])
    if res["rule_overrides"]:
        print("правила кода:", "; ".join(res["rule_overrides"]))
    print("ИТОГ:", res["total"], "| грубая:", res["gross_error"], "| модель:", res["model"], "| токены:", res["usage"], "| review:", res["needs_review"])
