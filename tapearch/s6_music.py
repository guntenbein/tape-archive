"""Стадия 6 — music: сохранность и пригодность музыкальных сегментов.

Здесь не оценивается, «хорошая ли музыка». Оценивается другое: доиграно ли
исполнение до конца, не оборвано ли оно, сколько внутри провалов, насколько
запись технически цела и что из этого можно вытащить реставрацией.

Законченность решается по контексту, а не по самому сегменту: песня может
кончиться затуханием, хлопками или просто паузой перед разговором — все три
случая законные. Оборвано — это когда уровень до последнего кадра высокий,
паузы нет и сразу начинается что-то другое.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import signal

from . import audio
from .config import (CLIP_RUN_SAMPLES, CLIP_THRESHOLD, DROPOUT_MIN_S,
                     DROPOUT_OVER_FLOOR_DB, HISS_BAND_HZ, HUM_FREQS)
from .runlog import log

COMPLETENESS = ["complete", "likely_complete", "partial", "fragment"]


def _runs_where(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[::2].tolist(), idx[1::2].tolist()))


def _edges(seg: dict, nxt: dict | None, prev: dict | None, lvl: np.ndarray,
           pause: np.ndarray, fps: float, duration: float) -> dict:
    """Что происходит на краях сегмента: от паузы ли начали и паузой ли кончили.

    Пауза берётся из той же относительной маски, что и в сегментации. Абсолютный
    порог здесь уже один раз всё испортил: фон между дублями на этой ленте стоит
    на 27-37 дБ выше шумовой полки файла, тест «после концовки пауза» не
    срабатывал почти никогда, и девять исполнений из одиннадцати объявлялись
    оборванными.
    """
    a, b = seg["frame_start"], min(seg["frame_end"], len(lvl))
    w = lambda s: max(1, int(s * fps))                                   # noqa: E731

    def med(x: np.ndarray) -> float:
        return float(np.median(x)) if x.size else 0.0

    tail_last = med(lvl[max(a, b - w(0.8)): b])
    tail_before = med(lvl[max(a, b - w(5.0)): max(a, b - w(2.5))])
    # Уровень за секунду ДО конца, и общий уровень тела сегмента. Именно по ним
    #判 решается затухание: последние полсекунды смотреть нельзя, туда попадает
    # начало следующего куска, и обрубленная на полном ходу музыка выглядит
    # аккуратно затухающей. Настоящая концовка затихает заранее.
    body = float(np.percentile(lvl[a:b], 75)) if b > a else 0.0
    end_run_up = med(lvl[max(a, b - w(2.0)): max(a + 1, b - w(1.0))])
    head_first = med(lvl[a: min(b, a + w(0.8))])
    pause_before = med(lvl[max(0, a - w(1.5)): a]) if a > 0 else 0.0
    pause_after = med(lvl[b: min(len(lvl), b + w(1.5))]) if b < len(lvl) else 0.0
    before_win = pause[max(0, a - w(1.5)): a]
    after_win = pause[b: min(len(pause), b + w(1.5))]
    frac_before = float(before_win.mean()) if before_win.size else 1.0
    frac_after = float(after_win.mean()) if after_win.size else 1.0

    decay_end = end_run_up < body - 8.0
    pause_after_ok = frac_after >= 0.5
    applause_after = bool(nxt and nxt.get("category") == "applause")
    at_file_end = seg["end"] >= duration - 1.5
    at_file_start = seg["start"] <= 1.5

    clean_end = bool(decay_end or applause_after or pause_after_ok)
    clean_start = bool(frac_before >= 0.5 or (prev and prev.get("category") in
                                              ("conversation", "silence", "applause",
                                               "rehearsal", "noise")))

    truncated_end = bool(not clean_end or (at_file_end and tail_last > 12.0))
    truncated_start = bool((at_file_start and head_first > 12.0)
                           or (not clean_start and prev and prev.get("category") == "music"))
    return {
        "level_tail_db": round(tail_last, 1), "level_before_tail_db": round(tail_before, 1),
        "level_body_db": round(body, 1), "level_1s_before_end_db": round(end_run_up, 1),
        "level_head_db": round(head_first, 1),
        "pause_before_db": round(pause_before, 1), "pause_after_db": round(pause_after, 1),
        "pause_fraction_before": round(frac_before, 2), "pause_fraction_after": round(frac_after, 2),
        "ends_with_decay": decay_end, "followed_by_applause": applause_after,
        "followed_by_pause": pause_after_ok,
        "at_file_start": at_file_start, "at_file_end": at_file_end,
        "truncated_start": truncated_start, "truncated_end": truncated_end,
    }


def _completeness(dur: float, e: dict) -> tuple[str, str]:
    if dur < 25.0:
        return "fragment", f"слишком коротко для законченного исполнения ({dur:.0f} с)"
    if e["truncated_end"]:
        return "partial", ("исполнение обрывается: уровень до последнего кадра высокий "
                           f"({e['level_1s_before_end_db']:.0f} дБ над полкой за секунду до конца "
                           f"при уровне вещи {e['level_body_db']:.0f} дБ), паузы и затухания нет")
    if e["truncated_start"]:
        return "partial", "начало срезано: сегмент стартует уже на полном уровне"
    why = []
    if e["ends_with_decay"]:
        why.append("концовка затухает")
    if e["followed_by_applause"]:
        why.append("сразу после — хлопки")
    if e["followed_by_pause"]:
        why.append("после концовки пауза")
    if e["truncated_start"] is False and e["pause_fraction_before"] >= 0.5:
        return "complete", "начало от паузы, " + ", ".join(why)
    return "likely_complete", ", ".join(why) + "; начало под вопросом"


def _prominence_db(f: np.ndarray, p: np.ndarray, target: float, half: float = 2.0) -> float | None:
    peak = p[(f >= target - half) & (f <= target + half)]
    ref = p[((f >= target - 25) & (f <= target - 5)) | ((f >= target + 5) & (f <= target + 25))]
    if peak.size == 0 or ref.size == 0:
        return None
    return float(10 * np.log10(peak.max() / np.median(ref)))


def _technical(y: np.ndarray, sr: int) -> dict:
    """Техническое состояние участка на родной частоте: клиппинг, гул, шипение, ровность."""
    if y.size < sr // 2:
        return {}
    peak = float(np.abs(y).max())
    rms = float(np.sqrt((y.astype(np.float64) ** 2).mean()))
    hot = np.abs(y) >= CLIP_THRESHOLD
    clipped = 0
    for i0, i1 in _runs_where(hot):
        if i1 - i0 >= CLIP_RUN_SAMPLES:
            clipped += i1 - i0

    nper = min(1 << 15, 1 << int(np.log2(max(1024, y.size // 4))))
    f, p = signal.welch(y, fs=sr, nperseg=nper, scaling="density")
    p = np.maximum(p, 1e-20)
    hi = min(HISS_BAND_HZ[1], sr / 2 - 200)
    band = (f >= HISS_BAND_HZ[0]) & (f <= hi)
    full = (f >= 50) & (f <= min(sr / 2 - 100, 16000))

    win = int(sr)
    nw = max(1, y.size // win)
    seg_rms = np.sqrt((y[: nw * win].reshape(nw, win).astype(np.float64) ** 2).mean(axis=1) + 1e-20)
    lvl_db = 20 * np.log10(seg_rms)

    ref = p[(f >= 500) & (f <= 2000)]
    cutoff = None
    if ref.size:
        thr = np.median(ref) / 10 ** 4.0
        k = 9
        sm = np.convolve(p, np.ones(k) / k, mode="same")
        above = np.where((sm > thr) & (f > 2000))[0]
        cutoff = float(f[above[-1]]) if above.size else None

    return {
        "peak_dbfs": round(float(20 * np.log10(max(peak, 1e-9))), 2),
        "rms_dbfs": round(float(20 * np.log10(max(rms, 1e-9))), 2),
        "crest_db": round(float(20 * np.log10(max(peak, 1e-9) / max(rms, 1e-9))), 2),
        "clip_ratio": round(clipped / y.size, 6),
        "clipped_samples": int(clipped),
        "hum_prominence_db": {str(int(t)): (round(v, 1) if (v := _prominence_db(f, p, t)) is not None else None)
                              for t in HUM_FREQS},
        "hiss_band_rel_db": round(float(10 * np.log10(p[band].mean() / p[full].mean())), 1) if band.any() else None,
        "cutoff_hz": round(cutoff) if cutoff else None,
        "level_stability_db_std": round(float(np.std(lvl_db)), 2) if nw > 2 else None,
        "level_drift_db": round(float(lvl_db[-max(1, nw // 5):].mean()
                                      - lvl_db[:max(1, nw // 5)].mean()), 2) if nw > 4 else None,
    }


def _dropouts(lvl: np.ndarray, a: int, b: int, fps: float) -> dict:
    inner = lvl[a:b]
    quiet = inner < DROPOUT_OVER_FLOOR_DB
    runs = [(i0, i1) for i0, i1 in _runs_where(quiet)
            if (i1 - i0) / fps >= DROPOUT_MIN_S and i0 > fps and i1 < len(inner) - fps]
    return {
        "count": len(runs),
        "total_s": round(sum((i1 - i0) / fps for i0, i1 in runs), 2),
        "spans": [[round(i0 / fps, 2), round(i1 / fps, 2)] for i0, i1 in runs[:20]],
    }


def _restoration(tech: dict, drops: dict, dur: float) -> dict:
    """Что даст реставрация и что уже не вернуть. Чистые рецепты, без правки оригинала."""
    plan, blockers = [], []
    score = 60
    hum = max((v for v in (tech.get("hum_prominence_db") or {}).values() if v is not None),
              default=0.0)
    if hum >= 10:
        plan.append(f"снять сетевой гул (пик {hum:.0f} дБ над фоном): узкие notch на 50/100/150 Гц")
        score += 8
    hiss = tech.get("hiss_band_rel_db")
    if hiss is not None and hiss > -14:
        plan.append("шипение ленты заметное: спектральный denoise по профилю из тихого участка")
        score += 8
    clip = tech.get("clip_ratio") or 0.0
    if clip > 0.001:
        plan.append(f"declip: срезано {100*clip:.2f}% сэмплов")
        if clip > 0.02:
            blockers.append(f"клиппинг тяжёлый ({100*clip:.1f}%) — вершины восстановятся лишь частично")
            score -= 25
        else:
            score -= 5
    crest = tech.get("crest_db")
    if crest is not None and crest < 8:
        blockers.append(f"низкий crest ({crest} дБ): материал уже пережат или перегружен")
        score -= 12
    cutoff = tech.get("cutoff_hz")
    if cutoff is not None and cutoff < 8000:
        blockers.append(f"полоса обрезана на {cutoff} Гц — высоких в источнике нет, вернуть нечем")
        score -= 10
    if drops["count"] > 0:
        blockers.append(f"провалов уровня: {drops['count']} на {drops['total_s']:.1f} с — это потеря сигнала, не шум")
        score -= min(20, 5 * drops["count"])
    stab = tech.get("level_stability_db_std")
    if stab is not None and stab > 6:
        plan.append(f"выровнять уровень (разброс {stab} дБ): медленная автоматика, не компрессор")
        score += 4
    if abs(tech.get("level_drift_db") or 0) > 6:
        plan.append("уровень уползает по ходу — похоже на подсевшие батарейки или плавающую АРУ")
    plan.append("нормализация в конце, после всего остального")
    return {"score": int(np.clip(score, 0, 100)), "plan": plan, "blockers": blockers}


def run(wdir: Path, src: Path, segments: dict, classify: dict,
        feats: dict[str, np.ndarray], fmeta: dict, probe: dict) -> dict:
    from .s3_segment import pause_mask
    fps = fmeta["fps"]
    floor_db = fmeta["noise_floor_db"]
    lvl = feats["rms_db"] - floor_db
    pause = pause_mask(feats, floor_db)
    duration = fmeta["duration_s"]
    native_sr = probe["format"]["sample_rate"]
    by_id = {c["id"]: c for c in classify["segments"]}
    segs = segments["segments"]

    out = []
    for i, seg in enumerate(segs):
        cls = by_id[seg["id"]]
        if cls["category"] not in ("music", "instrument_setup", "rehearsal"):
            continue
        prev = by_id.get(segs[i - 1]["id"]) if i > 0 else None
        nxt = by_id.get(segs[i + 1]["id"]) if i + 1 < len(segs) else None
        e = _edges(seg, nxt, prev, lvl, pause, fps, duration)
        # законченность — свойство исполнения; у настройки её просто нет
        # законченность — свойство исполнения; у настройки и репетиционного
        # разбора её просто нет
        comp, comp_why = ((None, "законченность к этой категории не применима")
                          if cls["category"] in ("instrument_setup", "rehearsal")
                          else _completeness(seg["duration"], e))
        y = audio.decode_window(src, seg["start"], seg["duration"], native_sr, 1)
        tech = _technical(y, native_sr)
        drops = _dropouts(lvl, seg["frame_start"], min(seg["frame_end"], len(lvl)), fps)
        rest = _restoration(tech, drops, seg["duration"])

        category = cls["category"]
        if category == "music" and comp is not None:
            category = {"complete": "music_complete", "likely_complete": "music_complete",
                        "partial": "music_partial", "fragment": "music_fragment"}[comp]
        out.append({
            "id": seg["id"], "start": seg["start"], "end": seg["end"],
            "duration": seg["duration"],
            "category": category,
            "completeness": comp, "completeness_reason": comp_why,
            "edges": e, "dropouts": drops, "technical": tech, "restoration": rest,
            "has_vocal": cls["attributes"].get("has_vocal"),
            "vocal_confidence": cls["attributes"].get("vocal_confidence"),
        })

    result = {"completeness_levels": COMPLETENESS, "segments": out}
    (wdir / "music.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info("music: проанализировано %d сегментов; целиком %d, оборвано %d, обрывки %d",
             len(out), sum(1 for s in out if s["completeness"] in ("complete", "likely_complete")),
             sum(1 for s in out if s["completeness"] == "partial"),
             sum(1 for s in out if s["completeness"] == "fragment"))
    return result
