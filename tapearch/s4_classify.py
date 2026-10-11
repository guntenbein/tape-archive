"""Стадия 4 — classify: категория и признаки каждого сегмента.

Таксономия разделена на две оси, и это сделано нарочно. Категория отвечает на
вопрос «что это»: музыка, разговор, настройка, аплодисменты, шум, тишина.
Признаки (вокал, барабаны, обрыв, провалы, язык) — независимые поля. Иначе
пришлось бы плодить классы на каждое сочетание: «законченная песня с вокалом»,
«оборванный инструментал» и так далее, и любой из них было бы нечем заполнить.

Законченность исполнения здесь не решается: это задача стадии music, у которой
есть анализ концовки. Поле `category` для музыкальных сегментов она уточняет.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .runlog import log

CATEGORIES = [
    "music",             # исполнение; music уточнит до complete/partial/fragment
    "rehearsal",         # играют и тут же обсуждают: разбор, подсказки, «давай сначала»
    "instrument_setup",  # настройка, перебор струн, проба звука
    "conversation",      # разговор
    "applause",          # хлопки, реакция зала
    "noise",             # технический мусор, шум ленты, щелчки
    "silence",
    "unknown",
]


def _agg(f: dict[str, np.ndarray], a: int, b: int, floor_db: float) -> dict:
    sl = slice(a, b)

    def med(k: str) -> float:
        v = f[k][sl]
        return float(np.median(v)) if v.size else 0.0

    lvl = f["rms_db"][sl] - floor_db
    return {
        "level_over_floor_db": float(np.median(lvl)) if lvl.size else 0.0,
        "level_p95_over_floor_db": float(np.percentile(lvl, 95)) if lvl.size else 0.0,
        "level_std_db": float(np.std(lvl)) if lvl.size else 0.0,
        "mod_syllabic": med("mod_syllabic"),
        "chroma_entropy": med("chroma_entropy"),
        "chroma_peak": med("chroma_peak"),
        "pulse_clarity": med("pulse_clarity"),
        "harmonic_ratio": med("harmonic_ratio"),
        "flatness": med("flatness"),
        "hf_ratio": med("hf_ratio"),
        "centroid_hz": med("centroid"),
        "rolloff95_hz": med("rolloff95"),
    }


def _tag_agg(tags: dict[str, np.ndarray], start: float, end: float) -> dict:
    if not tags or "times" not in tags:
        return {}
    tt = tags["times"]
    m = (tt >= start) & (tt < end)
    if not m.any():
        m = np.abs(tt - (start + end) / 2) < 2.0
    if not m.any():
        return {}
    keys = [k for k in tags if not k.startswith("_") and k != "times"]
    out = {f"tag_{k}_mean": float(np.mean(tags[k][m])) for k in keys}
    out.update({f"tag_{k}_p90": float(np.percentile(tags[k][m], 90)) for k in keys})
    return out


def _onset_density(f: dict[str, np.ndarray], a: int, b: int, fps: float) -> float:
    o = f["onset"][a:b]
    if o.size < 8:
        return 0.0
    thr = np.percentile(f["onset"], 70)
    peaks = (o > thr) & (o > np.roll(o, 1)) & (o > np.roll(o, -1))
    return float(peaks.sum() / (o.size / fps))


def _vocal_evidence(ag: dict, tg: dict, dens: float) -> tuple[bool, float, str]:
    """Есть ли в музыкальном сегменте голос.

    Основной источник — теггер: Singing он отличает от Speech, а эвристики на
    спектре принципиально не могут (у пения и тональность, и форманты). Без
    теггера остаётся слабый признак: слоговая модуляция поверх тонального
    материала. Уверенность возвращается отдельно, чтобы отчёт не врал.
    """
    sing = tg.get("tag_singing_p90")
    speech = tg.get("tag_speech_p90", 0.0)
    if sing is not None:
        if sing >= 0.15:
            return True, min(1.0, 0.4 + sing), f"теггер: пение {sing:.2f}"
        if sing >= 0.05 or speech >= 0.2:
            return True, 0.35, f"теггер: слабые признаки голоса (пение {sing:.2f}, речь {speech:.2f})"
        return False, 0.7, f"теггер: пения не слышит ({sing:.2f})"
    syl = ag["mod_syllabic"]
    if syl >= 0.3:
        return True, 0.3, f"слоговая модуляция {syl:.2f} поверх тонального материала (слабый признак)"
    return False, 0.25, f"слоговая модуляция низкая ({syl:.2f}), но без теггера это ненадёжно"


def classify_one(seg: dict, f: dict[str, np.ndarray], fps: float, floor_db: float,
                 tags: dict[str, np.ndarray]) -> dict:
    a, b = seg["frame_start"], min(seg["frame_end"], len(f["rms_db"]))
    ag = _agg(f, a, b, floor_db)
    tg = _tag_agg(tags, seg["start"], seg["end"])
    dens = _onset_density(f, a, b, fps)
    ag["onset_density_per_s"] = dens
    dur = seg["duration"]
    state = seg["state"]
    reasons: list[str] = []

    if state == "silence":
        cat = "silence"
        reasons.append(f"уровень у шумовой полки ({ag['level_over_floor_db']:.1f} дБ над ней)")
    elif state == "applause":
        cat = "applause"
        reasons.append(f"негармонический широкополосный материал (гармоничность {ag['harmonic_ratio']:.2f})")
    elif state == "speech":
        cat = "conversation"
        reasons.append(f"слоговая модуляция {ag['mod_syllabic']:.2f}, "
                       f"высота тона размазана (энтропия хромы {ag['chroma_entropy']:.2f})")
    elif state == "noise":
        cat = "noise"
        reasons.append(f"спектр плоский ({ag['flatness']:.3f}), тональной структуры нет")
    elif state == "music":
        setup_score = sum((
            1.0 if ag["chroma_entropy"] < 0.65 else 0.0,     # одиночные ноты, не аккорды
            1.0 if ag["pulse_clarity"] < 0.45 else 0.0,      # ритма нет
            1.0 if dens < 1.6 else 0.0,                      # атаки редкие
            0.5 if dur < 120 else 0.0,
            0.5 if ag["level_over_floor_db"] < 20 else 0.0,
        ))
        # Репетиция — это не «музыка, в которой кто-то говорит», а отдельное
        # состояние: играют и тут же обсуждают. На пилоте два куска по 5 и 11
        # минут были размечены как законченные песни, хотя расшифровка показала
        # сплошной разбор: «нажми соль», «чуть тише соло», «давай сначала».
        speech_ev = max(tg.get("tag_speech_p90", 0.0), tg.get("tag_speech_mean", 0.0) * 2)
        if speech_ev >= 0.35 and tg.get("tag_singing_p90", 0.0) < 0.25:
            cat = "rehearsal"
            reasons.append(
                f"играют и говорят одновременно: теггер слышит речь {speech_ev:.2f} "
                f"при пении {tg.get('tag_singing_p90', 0.0):.2f} — это разбор, а не исполнение")
        elif setup_score >= 2.5:
            cat = "instrument_setup"
            reasons.append(
                f"тональный материал без ритма: энтропия хромы {ag['chroma_entropy']:.2f}, "
                f"бит {ag['pulse_clarity']:.2f}, атак {dens:.1f}/с — похоже на настройку/перебор")
        else:
            cat = "music"
            reasons.append(f"тональный материал с ритмом: бит {ag['pulse_clarity']:.2f}, "
                           f"атак {dens:.1f}/с, энтропия хромы {ag['chroma_entropy']:.2f}")
    else:
        cat = "unknown"
        reasons.append("признаки не сложились ни в одну категорию")

    attrs: dict = {}
    if cat in ("music", "instrument_setup", "rehearsal"):
        has_v, conf, why = _vocal_evidence(ag, tg, dens)
        attrs["has_vocal"] = has_v
        attrs["vocal_confidence"] = round(conf, 2)
        attrs["vocal_reason"] = why
        attrs["has_percussion"] = bool(
            tg.get("tag_music_p90", 0) > 0 and dens >= 2.5 and ag["pulse_clarity"] >= 0.5)
        attrs["is_instrumental"] = not has_v

    return {
        "id": seg["id"],
        "category": cat,
        "state": state,
        "reasons": reasons,
        "attributes": attrs,
        "acoustics": {k: round(v, 4) for k, v in ag.items()},
        "tagger": {k: round(v, 3) for k, v in tg.items()} or None,
    }


def run(wdir: Path, segments: dict, feats: dict[str, np.ndarray], fmeta: dict,
        tags: dict[str, np.ndarray] | None) -> dict:
    fps = fmeta["fps"]
    floor_db = fmeta["noise_floor_db"]
    tags = tags or {}
    out = [classify_one(s, feats, fps, floor_db, tags) for s in segments["segments"]]
    result = {"categories": CATEGORIES, "used_tagger": bool(tags), "segments": out}
    (wdir / "classify.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    hist: dict[str, int] = {}
    for c in out:
        hist[c["category"]] = hist.get(c["category"], 0) + 1
    log.info("classify: %s", ", ".join(f"{k} {v}" for k, v in sorted(hist.items(), key=lambda x: -x[1])))
    return result
