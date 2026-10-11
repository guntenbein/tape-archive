"""Стадия 5 — transcribe: расшифровка там, где она что-то даёт.

На ASR отдаются только сегменты, где есть речь или разборчивый вокал: гонять
Whisper по инструменталу и шуму ленты бессмысленно и дорого — он начинает
придумывать текст. По той же причине condition_on_previous_text выключен:
на шумной записи он затягивает модель в петлю повторов, когда одна
галлюцинация становится контекстом для следующей.

Язык: по умолчанию русский, но на каждом сегменте отдельно запускается
определение языка, и если уверенно выходит белорусский — сегмент
перерасшифровывается на нём. Результат с распознанным языком и мерами
уверенности пишется как есть, ненадёжное помечается, а не выбрасывается.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from . import audio
from .config import ASR_BEAM, ASR_COMPUTE, ASR_DEFAULT_LANG, ASR_SR, MODELS_ROOT
from .runlog import log

MODELS_DIR = MODELS_ROOT / "whisper"
TRANSCRIBE_CATEGORIES = {"conversation", "rehearsal", "music", "music_complete",
                         "music_partial", "music_fragment", "instrument_setup"}
# меры, по которым расшифровка объявляется ненадёжной
MAX_NO_SPEECH = 0.75
MIN_AVG_LOGPROB = -1.05
MAX_COMPRESSION = 2.4
BE_SWITCH_PROB = 0.5

# Whisper на неречевом звуке выдаёт не шум, а заученные фразы из обучающих
# данных — титры с ютуба и подобное, причём с высокой уверенностью. Поэтому
# статистические фильтры (logprob, no_speech, сжатие) их не ловят: модель
# действительно «уверена». Отсекается это только списком. Проверено живьём:
# на синтетическом неречевом сегменте модель уверенно написала
# «Субтитры создавал DimaTorzok» с logprob -0.21 и no_speech 0.00.
HALLUCINATION_MARKERS = (
    "субтитры", "subtitles", "amara.org", "dimatorzok", "продолжение следует",
    "спасибо за просмотр", "подписывайтесь на канал", "редактор субтитров",
    "корректор", "перевод и озвучивание", "музыка", "аплодисменты",
    "thanks for watching", "subscribe", "корректор а.кулакова",
)


class Asr:
    """Обёртка над faster-whisper: модель грузится один раз на прогон."""

    def __init__(self, model_name: str, threads: int = 8):
        from faster_whisper import WhisperModel
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        repo = {"large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2"}.get(
            model_name, model_name)
        log.info("asr: поднимаю %s (%s, %d потоков)", model_name, ASR_COMPUTE, threads)
        self.name = model_name
        self.model = WhisperModel(repo, device="cpu", compute_type=ASR_COMPUTE,
                                  cpu_threads=threads, download_root=str(MODELS_DIR))

    def detect(self, y: np.ndarray) -> list[tuple[str, float]]:
        """Топ-3 языка участка. faster-whisper отдаёт список пар, а не словарь."""
        try:
            lang, prob, all_probs = self.model.detect_language(
                audio=y[: 30 * ASR_SR], vad_filter=True)
            pairs = list(all_probs.items()) if isinstance(all_probs, dict) else list(all_probs or [])
            top = sorted(pairs, key=lambda kv: -kv[1])[:3]
            return [(str(k), round(float(v), 3)) for k, v in top] or [(lang, round(float(prob), 3))]
        except Exception as exc:                        # noqa: BLE001
            log.debug("asr: определение языка не удалось: %s", exc)
            return []

    def transcribe(self, y: np.ndarray, language: str) -> dict:
        segs, info = self.model.transcribe(
            y, language=language, beam_size=ASR_BEAM, vad_filter=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=MAX_COMPRESSION,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        items = []
        for s in segs:
            items.append({
                "start": round(float(s.start), 2), "end": round(float(s.end), 2),
                "text": s.text.strip(),
                "avg_logprob": round(float(s.avg_logprob), 3),
                "no_speech_prob": round(float(s.no_speech_prob), 3),
                "compression_ratio": round(float(s.compression_ratio), 3),
            })
        return {"language": language, "language_probability": round(float(info.language_probability), 3),
                "items": items}


def _is_hallucination(text: str) -> bool:
    t = text.lower().replace("ё", "е")
    return any(mark in t for mark in HALLUCINATION_MARKERS)


def _filter_hallucinations(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Выбросить заученные фразы и зацикленные повторы. Что выбросили — вернуть."""
    kept, dropped = [], []
    for it in items:
        if _is_hallucination(it["text"]):
            dropped.append(it["text"])
            continue
        kept.append(it)
    # один и тот же текст три раза и больше — модель зациклилась
    counts: dict[str, int] = {}
    for it in kept:
        key = it["text"].strip().lower()
        counts[key] = counts.get(key, 0) + 1
    loops = {k for k, v in counts.items() if v >= 3 and len(k) > 4}
    if loops:
        dropped += [f"повтор ×{counts[k]}: {k[:60]}" for k in loops]
        kept = [it for it in kept if it["text"].strip().lower() not in loops]
    return kept, dropped


def _quality(items: list[dict], dropped: list[str]) -> dict:
    if not items:
        return {"usable": False,
                "note": ("вся расшифровка оказалась заученной фразой Whisper, "
                         "а не речью: " + "; ".join(dropped[:2])) if dropped
                        else "речи не найдено",
                "dropped": dropped}
    avg = float(np.mean([i["avg_logprob"] for i in items]))
    nsp = float(np.mean([i["no_speech_prob"] for i in items]))
    comp = float(np.max([i["compression_ratio"] for i in items]))
    reasons = []
    if avg < MIN_AVG_LOGPROB:
        reasons.append(f"низкая уверенность модели (avg_logprob {avg:.2f})")
    if nsp > MAX_NO_SPEECH:
        reasons.append(f"модель сама сомневается, что это речь (no_speech {nsp:.2f})")
    if comp > MAX_COMPRESSION:
        reasons.append(f"текст похож на зацикленную галлюцинацию (сжатие {comp:.2f})")
    # Выброшенная заученная фраза сама по себе не делает расшифровку негодной:
    # на пилоте сегмент с настоящим разбором («нажми соль», «чуть тише соло»)
    # помечался непригодным из-за одной лишней строки титров.
    note_drop = f"выброшено как галлюцинация Whisper: {len(dropped)} фраз" if dropped else None
    note = "; ".join(reasons) or "похоже на разборчивую речь"
    if note_drop:
        note += f" ({note_drop})"
    return {"usable": not reasons, "avg_logprob": round(avg, 3),
            "no_speech_prob": round(nsp, 3), "max_compression_ratio": round(comp, 3),
            "note": note, "dropped": dropped}


def _targets(segments: dict, classify: dict, music: dict | None) -> list[dict]:
    """Какие сегменты вообще отдавать на ASR."""
    by_id = {c["id"]: c for c in classify["segments"]}
    out = []
    for seg in segments["segments"]:
        cls = by_id[seg["id"]]
        cat = cls["category"]
        if cat not in TRANSCRIBE_CATEGORIES:
            continue
        if cat not in ("conversation", "rehearsal") and not cls["attributes"].get("has_vocal"):
            continue                               # инструментал на ASR не отдаём
        out.append({"id": seg["id"], "start": seg["start"], "end": seg["end"], "category": cat})
    return out


def _merge(regions: list[dict], gap: float = 1.0) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for r in sorted(regions, key=lambda x: x["start"]):
        if spans and r["start"] - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], r["end"])
        else:
            spans.append([r["start"], r["end"]])
    return [(a, b) for a, b in spans]


def run(wdir: Path, src: Path, segments: dict, classify: dict, asr: Asr,
        music: dict | None = None) -> dict:
    """Один проход ASR на весь файл вместо вызова на каждый сегмент.

    Whisper всегда считает окно в 30 секунд, поэтому вызов на двухсекундном
    куске стоит столько же, сколько на тридцатисекундном. На ленте 1981 года с
    её 55 мелкими сегментами это давало 2.6x от длительности записи, из которых
    почти всё — накладные расходы. Поэтому речевые участки склеиваются в одну
    дорожку (с полусекундными паузами на стыках), прогоняются разом, а потом
    таймкоды возвращаются на исходную шкалу по таблице смещений.
    """
    all_targets = _targets(segments, classify, music)
    # Речь гоним одним потоком — там важна скорость. Вокальные музыкальные
    # сегменты гоним по отдельности: в общем потоке VAD подрезает пение, а
    # текст песни — это то, чем вещь опознаётся. Их мало и они длинные,
    # поэтому накладные расходы на отдельный вызов тут не важны.
    # Отдельный вызов оправдан только для ДЛИННОГО вокального куска: цена
    # вызова ~30 с независимо от длины, а терять пару слов на четырёхсекундном
    # обрывке не жалко. На ленте 1984 года 13 из 15 отдельных вызовов пришлись
    # на куски по 3-11 секунд и съели 6 минут впустую.
    SPEECHY = {"conversation", "rehearsal", "instrument_setup"}
    SOLO_MIN_S = 25.0
    targets = [t for t in all_targets
               if t["category"] in SPEECHY or t["end"] - t["start"] < SOLO_MIN_S]
    solo = [t for t in all_targets
            if t["category"] not in SPEECHY and t["end"] - t["start"] >= SOLO_MIN_S]
    if not all_targets:
        result = {"model": asr.name, "segments": [], "n_usable": 0}
        (wdir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        log.info("transcribe: речевых сегментов нет, пропускаю")
        return result

    spans = _merge(targets) if targets else []
    gap = np.zeros(int(0.5 * ASR_SR), dtype=np.float32)
    pieces, offsets, cursor = [], [], 0.0
    for a, b in spans:
        y = audio.decode_window(src, a, b - a, ASR_SR, 1)
        if y.size < ASR_SR // 4:
            continue
        pieces.append(y)
        offsets.append({"concat_start": cursor, "concat_end": cursor + y.size / ASR_SR,
                        "orig_start": a})
        cursor += y.size / ASR_SR
        pieces.append(gap)
        cursor += gap.size / ASR_SR
    joined = np.concatenate(pieces) if pieces else np.zeros(ASR_SR, dtype=np.float32)
    speech_s = sum(o["concat_end"] - o["concat_start"] for o in offsets)
    log.info("transcribe: один проход по %.1f мин речи из %.1f мин ленты (%d участков)",
             speech_s / 60, segments["segments"][-1]["end"] / 60, len(offsets))

    langs = asr.detect(joined)                      # язык определяем один раз на файл
    lang = ASR_DEFAULT_LANG
    for code, prob in langs:
        if code == "be" and prob >= BE_SWITCH_PROB:
            lang = "be"
            break
    res = asr.transcribe(joined, lang)

    def to_orig(t: float) -> float | None:
        for o in offsets:
            if o["concat_start"] <= t <= o["concat_end"]:
                return o["orig_start"] + (t - o["concat_start"])
        return None                                  # попало в стык — выбрасываем

    phrases = []
    for it in res["items"]:
        a0, b0 = to_orig(it["start"]), to_orig(it["end"])
        if a0 is None or b0 is None or b0 < a0:
            continue
        phrases.append({**it, "abs_start": round(a0, 2), "abs_end": round(b0, 2)})

    # вокальные сегменты — отдельными вызовами, чтобы не терять слова песен
    for t in solo:
        y = audio.decode_window(src, t["start"], t["end"] - t["start"], ASR_SR, 1)
        if y.size < ASR_SR:
            continue
        r = asr.transcribe(y, lang)
        for it in r["items"]:
            phrases.append({**it, "abs_start": round(t["start"] + it["start"], 2),
                            "abs_end": round(t["start"] + it["end"], 2)})
    if solo:
        log.info("transcribe: плюс %d вокальных сегментов отдельными вызовами", len(solo))

    out = []
    for t in all_targets:
        mine = [p for p in phrases if t["start"] <= (p["abs_start"] + p["abs_end"]) / 2 < t["end"]]
        mine, dropped = _filter_hallucinations(mine)
        qual = _quality(mine, dropped)
        out.append({
            "id": t["id"], "start": t["start"], "end": t["end"], "category": t["category"],
            "model": asr.name, "language": res["language"],
            "language_probability": res["language_probability"],
            "language_candidates": langs, "quality": qual,
            "text": " ".join(p["text"] for p in mine).strip(),
            "items": mine,
        })

    result = {"model": asr.name, "single_pass": True, "speech_minutes": round(speech_s / 60, 1),
              "segments": out, "n_usable": sum(1 for s in out if s["quality"]["usable"])}
    (wdir / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info("transcribe: %d сегментов, пригодных %d, слов всего %d",
             len(out), result["n_usable"], sum(len(s["text"].split()) for s in out))
    return result
