"""Точная привязка реплик звонка к записи (плеер в карточке звонка).

Почему нужно: время у реплик в транскрипте не совпадает со временем в файле записи — запись
ставится на паузу («Запись и транскрибирование приостановлены»), начинается не с первой реплики,
а метка времени ставится ближе к концу фразы. Поэтому каждую реплику ищем в самой записи:
  1) VAD (silero) режет запись на фрагменты речи, офлайн-модель sherpa-onnx (zipformer, русский)
     распознаёт их со временем каждого слова;
  2) слова транскрипта выравниваются со словами распознавания (динамическое программирование);
  3) начало реплики = время её первого совпавшего слова; реплики без совпадений («Да.», «2»,
     скрытые данные) — интерполяция между соседями с прилипанием к ближайшему началу речи;
  4) если слов почти не нашлось (казахский/английский звонок, шум) — опора на метки транскрипта
     с учётом начала и пауз записи плюс прилипание к началам речи по VAD.
Всё работает локально, без интернета (кроме однократной загрузки моделей ~110 МБ).

Использование:
  py tools/audio_sync.py --setup            # один раз: скачать модели в models/
  py tools/audio_sync.py data_DEV.json      # разметить звонки из audio/ → sync.json
Сервер (server.py) вызывает sync_record() сам для новых звонков: POST /api/sync.
"""
import datetime, difflib, json, os, re, statistics, subprocess, sys, tarfile, urllib.request

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.getenv("ASR_MODELS", os.path.join(BASE, "models"))
ASR_NAME = "sherpa-onnx-small-zipformer-ru-2024-09-18"
ASR = os.path.join(MODELS, ASR_NAME)
REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
NEED = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
SR = 16000

REC_START = re.compile(r"запись и транскрибирование запущены|жазу және транскрипция басталды|recording (and transcription )?(has |have )?started", re.I)
REC_PAUSE = re.compile(r"приостановлен|тоқтатыл|paused", re.I)
REC_RESUME = re.compile(r"возобновлен|жалғастырыл|resumed", re.I)


# ---------------- модели ----------------

def models_ready():
    return os.path.exists(os.path.join(MODELS, "silero_vad.onnx")) and all(os.path.exists(os.path.join(ASR, f)) for f in NEED)


def setup_models():
    """Скачивает VAD и русскую модель распознавания с GitHub (только нужные файлы)."""
    os.makedirs(ASR, exist_ok=True)
    vad = os.path.join(MODELS, "silero_vad.onnx")
    if not os.path.exists(vad):
        print("Скачиваю silero_vad.onnx…", flush=True)
        urllib.request.urlretrieve(REL + "silero_vad.onnx", vad)
    if not all(os.path.exists(os.path.join(ASR, f)) for f in NEED):
        print(f"Скачиваю {ASR_NAME} (~110 МБ, пара минут)…", flush=True)
        tmp = os.path.join(MODELS, ASR_NAME + ".tar.bz2")
        urllib.request.urlretrieve(REL + ASR_NAME + ".tar.bz2", tmp)
        with tarfile.open(tmp, mode="r:bz2") as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if name in NEED and m.isfile():
                    with open(os.path.join(ASR, name), "wb") as f:
                        f.write(tar.extractfile(m).read())
        os.remove(tmp)
    print("Модели готовы:", MODELS, flush=True)


_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        import sherpa_onnx
        if not models_ready():
            setup_models()
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(ASR, "encoder.int8.onnx"), decoder=os.path.join(ASR, "decoder.int8.onnx"),
            joiner=os.path.join(ASR, "joiner.int8.onnx"), tokens=os.path.join(ASR, "tokens.txt"), num_threads=4)
        vcfg = sherpa_onnx.VadModelConfig()
        vcfg.silero_vad.model = os.path.join(MODELS, "silero_vad.onnx")
        vcfg.silero_vad.min_silence_duration, vcfg.silero_vad.min_speech_duration = 0.25, 0.2
        vcfg.silero_vad.max_speech_duration, vcfg.sample_rate = 15, SR
        _ENGINE = (sherpa_onnx, rec, vcfg)
    return _ENGINE


# ---------------- звук ----------------

def load_audio(path):
    """Моно 16 кГц float32. Сначала soundfile (mp3/wav/ogg без внешних программ), иначе ffmpeg."""
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
    except Exception:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(pcm, dtype=np.float32)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


def recognize(path):
    """Начала фрагментов речи и слова со временем (секунды от начала файла), длительность файла."""
    so, rec, vcfg = engine()
    a = load_audio(path)
    vad = so.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60)
    segs = []
    for i in range(0, len(a), 512):
        vad.accept_waveform(a[i:i + 512])
        while not vad.empty():
            segs.append((vad.front.start, np.array(vad.front.samples))); vad.pop()
    vad.flush()
    while not vad.empty():
        segs.append((vad.front.start, np.array(vad.front.samples))); vad.pop()
    words = []
    for st, s in segs:
        stream = rec.create_stream(); stream.accept_waveform(SR, s); rec.decode_stream(stream)
        r, t0, cur = stream.result, st / SR, None
        for tok, ts in zip(r.tokens, r.timestamps):
            if tok.startswith((" ", "▁")) or cur is None:  # начало нового слова
                if cur:
                    words.append(cur)
                cur = {"w": tok.replace("▁", "").strip(), "t": round(t0 + ts, 2)}
            else:
                cur["w"] += tok
        if cur:
            words.append(cur)
    return [round(st / SR, 2) for st, _ in segs], words, len(a) / SR


# ---------------- выравнивание ----------------

def secs(t):
    t = str(t or "").strip()
    if "T" in t:
        try:
            return datetime.datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", t.replace("Z", "+00:00"))).timestamp()
        except ValueError:
            pass
    try:
        total = 0.0
        for p in t.split(":"):
            total = total * 60 + float(p)
        return total
    except ValueError:
        return None


def norm(t):
    t = re.sub(r"\[[^\]]*\]", " ", str(t or "").lower().replace("ё", "е"))
    return re.findall(r"[a-zа-яәіңғүұқөһ0-9]+", t)


def sim(a, b):
    if a == b:
        return 2.0
    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
        return 1.4
    return 1.0 if min(len(a), len(b)) >= 3 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.72 else -0.6


def baseline(tr):
    """Время в записи по меткам транскрипта: от «запись запущена», минус паузы записи."""
    raw = [secs(m.get("timestamp")) for m in tr]
    known = [x for x in raw if x is not None]
    first = known[0] if known else 0.0
    raw = [x if x is not None else None for x in raw]
    for i in range(len(raw)):  # дыры в метках — тянем предыдущую
        if raw[i] is None:
            raw[i] = raw[i - 1] if i else first
    t0 = raw[0]
    for i, m in enumerate(tr[:8]):
        if m.get("speaker") == "system" and REC_START.search(m.get("text") or ""):
            t0 = raw[i]
    out, gap, paused = [], 0.0, None
    for i, m in enumerate(tr):
        txt = m.get("text") or ""
        if m.get("speaker") == "system" and REC_PAUSE.search(txt) and paused is None:
            paused = raw[i]
        if m.get("speaker") == "system" and REC_RESUME.search(txt) and paused is not None:
            gap += raw[i] - paused; paused = None
        out.append(max(0.0, (paused if paused is not None else raw[i]) - t0 - gap))
    return raw, out


def align(tr, seg_starts, asr_words, dur):
    """Секунда начала в записи для каждой реплики и число реплик, найденных по словам."""
    if not tr:
        return [], 0
    raw, base = baseline(tr)
    T = [(i, k, w) for i, m in enumerate(tr) if m.get("speaker") != "system" for k, w in enumerate(norm(m.get("text")))]
    A = [(w["w"].lower().replace("ё", "е"), w["t"]) for w in asr_words if w["w"]]
    est = [None] * len(tr)
    if T and A:
        n, mm, G = len(T), len(A), -0.35
        S = [[0.0] * (mm + 1) for _ in range(n + 1)]
        P = [[0] * (mm + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            S[i][0], P[i][0] = i * G, 1
        for j in range(1, mm + 1):
            S[0][j], P[0][j] = j * G, 2
        for i in range(1, n + 1):
            ti, Si, Sp, Pi = T[i - 1][2], S[i], S[i - 1], P[i]
            for j in range(1, mm + 1):
                d, u, l = Sp[j - 1] + sim(ti, A[j - 1][0]), Sp[j] + G, Si[j - 1] + G
                if d >= u and d >= l:
                    Si[j], Pi[j] = d, 0
                elif u >= l:
                    Si[j], Pi[j] = u, 1
                else:
                    Si[j], Pi[j] = l, 2
        i, j, matches = n, mm, {}
        while i > 0 and j > 0:
            p = P[i][j]
            if p == 0:
                if sim(T[i - 1][2], A[j - 1][0]) >= 1.0:
                    matches.setdefault(T[i - 1][0], []).append((T[i - 1][1], A[j - 1][1]))
                i, j = i - 1, j - 1
            elif p == 1:
                i -= 1
            else:
                j -= 1
        for li, ms in matches.items():
            ms.sort()
            if len(ms) >= 2 or len(norm(tr[li].get("text"))) <= 2:
                k, t = ms[0]
                est[li] = max(0.0, t - k * 0.32)  # ~0.32 с на слово до первого совпавшего
        # сдвиг «транскрипт − запись» меняется плавно: резкие выбросы — ложные совпадения
        good = [i for i, e in enumerate(est) if e is not None]
        off = {i: raw[i] - est[i] for i in good}
        for x, i in enumerate(good):
            nb = [off[g] for g in good[max(0, x - 3):x + 4] if g != i]
            if len(nb) >= 2 and abs(off[i] - statistics.median(nb)) > 12:
                est[i] = None
        good = [i for i, e in enumerate(est) if e is not None]
        for a_, b_ in zip(good, good[1:]):
            if est[b_] < est[a_] - 1.0 and raw[b_] > raw[a_] + 0.5:
                est[b_] = None
    good = [i for i, e in enumerate(est) if e is not None]

    # мало совпадений (не русский язык, шум): опора на метки транскрипта
    if len(good) < 3:
        est, good = [None] * len(tr), []
        anchors = None
    else:
        anchors = good
    used, out = set(round(e, 1) for e in est if e is not None), []
    for i in range(len(tr)):
        if est[i] is not None:
            out.append(round(est[i], 1)); continue
        if anchors is None:
            v, lo, hi = base[i], -1e9, 1e9
        else:
            prev = max([g for g in good if g < i], default=None)
            nxt = min([g for g in good if g > i], default=None)
            if prev is None:
                v, lo, hi = max(0.0, est[nxt] - (raw[nxt] - raw[i])), -1e9, est[nxt] - 0.2
            elif nxt is None:
                v, lo, hi = est[prev] + (raw[i] - raw[prev]), est[prev] + 0.4, 1e9
            else:
                span = raw[nxt] - raw[prev]
                v = est[prev] + ((raw[i] - raw[prev]) / span if span > 0 else 0.0) * (est[nxt] - est[prev])
                lo, hi = est[prev] + 0.4, est[nxt] - 0.2
        if tr[i].get("speaker") != "system":  # прилипаем к ближайшему началу речи
            cand = [a for a in seg_starts if lo < a < hi and abs(a - v) <= 4 and a not in used]
            if cand:
                v = min(cand, key=lambda a: abs(a - v)); used.add(v)
        out.append(round(min(max(v, 0.0), dur), 1))
    for i in range(1, len(out)):
        if out[i] < out[i - 1] and est[i] is None:
            out[i] = out[i - 1]
    return out, len(good)


def sync_record(record, audio_path):
    seg_starts, words, dur = recognize(audio_path)
    offs, matched = align(record["transcript"], seg_starts, words, dur)
    speech = sum(1 for m in record["transcript"] if m.get("speaker") != "system")
    return {"offsets": offs, "matched": matched, "speech": speech, "duration": round(dur, 1),
            "method": "words" if matched >= 3 else "timestamps+vad"}


def main():
    if "--setup" in sys.argv:
        setup_models(); return
    src = sys.argv[1] if len(sys.argv) > 1 else "data_DEV.json"
    data = json.load(open(src, encoding="utf-8"))
    records = (data.get("data") or data)["records"]
    sync = {}
    for r in records:
        cid = r["conversation_id"]
        path = next((os.path.join("audio", f"{cid}.{e}") for e in ("mp3", "wav", "ogg", "m4a")
                     if os.path.exists(os.path.join("audio", f"{cid}.{e}"))), None)
        if r.get("channel") != "voice call" or not path:
            continue
        res = sync_record(r, path)
        sync[cid] = res["offsets"]
        print(f"{cid}: запись {res['duration']:.0f} с, реплик {len(r['transcript'])}, найдено по словам {res['matched']} ({res['method']})")
    json.dump(sync, open("sync.json", "w", encoding="utf-8"))
    print("sync.json готов")


if __name__ == "__main__":
    main()
