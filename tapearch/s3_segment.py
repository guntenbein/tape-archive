"""Стадия 3 — segment: естественные границы вместо нарезки по N минут.

Схема: на каждый кадр считается правдоподобие шести состояний (тишина, шум,
речь, музыка, настройка, аплодисменты), отдельно по эвристикам на признаках и
отдельно по нейросетевому теггеру; оба мнения складываются; по полученной
матрице гоняется Виterbi с «липкой» диагональю, который сам даёт длинные
связные куски вместо дрожания метки. Готовые границы притягиваются к ближайшей
паузе — чтобы песня не начиналась посреди такта.

Пороги вынесены в config и заданы в абсолютных величинах: слоговая модуляция и
энтропия хромы — безразмерные доли, они переносятся между кассетами. Единственное,
что нормируется на файл, — уровень: он считается в дБ над шумовой полкой
конкретной ленты, а она у кассет гуляет на десятки дБ.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .config import (DIP_BELOW_LOCAL_DB, MIN_DIP_S, MIN_SEGMENT_S,
                     MIN_SILENCE_SEGMENT_S, MIN_SILENCE_SPLIT_S, SILENCE_OVER_FLOOR_DB,
                     SMOOTH_MEDIAN_S, TAKE_SPLIT_MIN_LEN_S, VITERBI_STICKINESS)
from .runlog import log

STATES = ["silence", "noise", "speech", "music", "applause"]


def _sig(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / max(scale, 1e-6)))


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k < 3:
        return x
    k = k + 1 if k % 2 == 0 else k
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(strides, axis=-1).astype(np.float32)


def _onset_density(onset: np.ndarray, fps: float) -> np.ndarray:
    """Атак в секунду: у плотной музыки много, у одиночных щипков настройки мало."""
    thr = np.percentile(onset, 70)
    peaks = (onset > thr) & (onset > np.roll(onset, 1)) & (onset > np.roll(onset, -1))
    win = int(round(4.0 * fps))
    kernel = np.ones(win, dtype=np.float32)
    dens = np.convolve(peaks.astype(np.float32), kernel, mode="same") / (win / fps)
    return dens.astype(np.float32)


def heuristic_scores(f: dict[str, np.ndarray], fps: float, floor_db: float) -> np.ndarray:
    """Логиты шести состояний по акустическим признакам. Каждый член — названный признак."""
    lvl = f["rms_db"] - floor_db                  # дБ над шумовой полкой этой ленты
    syl = f["mod_syllabic"]                       # 3–6.5 Гц: слоговой ритм речи
    ent = f["chroma_entropy"]                     # низкая -> есть чёткая высота тона
    peak = f["chroma_peak"]
    beat = f["pulse_clarity"]
    harm = f["harmonic_ratio"]
    flat = f["flatness"]
    dens = _onset_density(f["onset"], fps)

    present = _sig(lvl, SILENCE_OVER_FLOOR_DB + 2.0, 1.5)
    # Центр 0.80, а не 0.86: на двух лентах музыка лежит в 0.63-0.81, речь в
    # 0.85-0.93. Прежний центр выдавал речи половину музыкального балла.
    tonal = (1.0 - _sig(ent, 0.80, 0.04)) * 0.5 + _sig(peak, 0.22, 0.05) * 0.5

    s = np.zeros((len(lvl), len(STATES)), dtype=np.float32)
    s[:, 0] = 5.0 * (1.0 - present)                                     # silence
    s[:, 1] = (1.2 + 1.0 * _sig(flat, 0.08, 0.04)                       # noise
               + 1.0 * _sig(ent, 0.9, 0.04) + 0.8 * (1.0 - _sig(lvl, 12.0, 4.0))) * present
    s[:, 2] = (1.4 * _sig(syl, 0.25, 0.08)                              # speech
               + 2.0 * _sig(ent, 0.84, 0.04)
               + 0.7 * (1.0 - _sig(beat, 0.45, 0.12))) * present
    s[:, 3] = (2.2 * tonal + 1.6 * _sig(beat, 0.4, 0.1)                 # music
               + 1.1 * (1.0 - _sig(syl, 0.32, 0.07))
               + 0.6 * _sig(dens, 1.5, 0.6)) * present
    s[:, 4] = (2.0 * (1.0 - _sig(harm, 0.85, 0.05))                     # applause
               + 1.2 * _sig(flat, 0.06, 0.03)
               + 1.0 * _sig(ent, 0.92, 0.03)
               + 0.8 * (1.0 - _sig(beat, 0.3, 0.08))) * present
    return s


def tagger_scores(tags: dict[str, np.ndarray], times: np.ndarray,
                  lvl: np.ndarray) -> np.ndarray | None:
    """Мнение теггера, разложенное на те же шесть состояний и натянутое на сетку кадров."""
    if not tags or "times" not in tags:
        return None
    tt = tags["times"]

    def on_grid(key: str) -> np.ndarray:
        return np.interp(times, tt, tags[key]).astype(np.float32)

    speech, singing = on_grid("speech"), on_grid("singing")
    music, applause = on_grid("music"), on_grid("applause")
    noise, silence = on_grid("noise"), on_grid("silence")
    s = np.zeros((len(times), len(STATES)), dtype=np.float32)
    s[:, 0] = 2.0 * silence
    s[:, 1] = 2.0 * noise
    s[:, 2] = 3.0 * np.clip(speech - singing, 0, 1)      # пение в речь не записываем
    s[:, 3] = 3.0 * np.maximum(music, singing)
    s[:, 4] = 3.0 * applause
    # на уровне у самой полки теггер фантазирует (шум ленты уходит в Music) — глушим
    gate = _sig(lvl, SILENCE_OVER_FLOOR_DB + 2.0, 1.5)[:, None]
    # Насколько теггер решителен: разрыв между «речь» и «музыка». Когда он
    # велик, теггеру надо верить. На ленте 1981 года он давал речь 0.76 при
    # музыке 0.07 и был прав, а эвристика всё равно перевешивала его в музыку.
    margin = np.abs(np.maximum(speech, 0) - np.maximum(music, singing))
    s = np.concatenate([s, np.clip(margin / 0.5, 0.0, 1.0)[:, None]], axis=1)
    return s * np.concatenate([gate] * len(STATES) + [np.ones_like(gate)], axis=1)


def _viterbi(logp: np.ndarray, stickiness: float) -> np.ndarray:
    n, k = logp.shape
    stay = np.log(stickiness)
    move = np.log((1.0 - stickiness) / max(1, k - 1))
    trans = np.full((k, k), move, dtype=np.float64)
    np.fill_diagonal(trans, stay)
    dp = logp[0].astype(np.float64).copy()
    back = np.zeros((n, k), dtype=np.int8)
    for i in range(1, n):
        cand = dp[:, None] + trans
        best = cand.argmax(axis=0)
        back[i] = best
        dp = cand[best, np.arange(k)] + logp[i]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(dp.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _runs(path: np.ndarray) -> list[tuple[int, int, int]]:
    edges = np.flatnonzero(np.diff(path)) + 1
    bounds = np.concatenate([[0], edges, [len(path)]])
    return [(int(bounds[i]), int(bounds[i + 1]), int(path[bounds[i]]))
            for i in range(len(bounds) - 1)]


def pause_mask(feats: dict[str, np.ndarray], floor_db: float) -> np.ndarray:
    """Где на ленте пауза.

    Два независимых признака, и достаточно любого:
      * настоящая тишина — уровень у шумовой полки всего файла;
      * провал — заметно тише, чем звучало вокруг в эти полминуты.

    Второй признак и есть главный. Между дублями на кассете не тишина: там
    комнатный шум, гудящий усилитель, иногда вытянутый автоматикой магнитофона
    почти до уровня сигнала. Абсолютного порога такая пауза не достигает
    никогда, а ухом слышна сразу.
    """
    rms = feats["rms_db"]
    ref = feats.get("local_ref_db")
    quiet_abs = rms < floor_db + SILENCE_OVER_FLOOR_DB
    if ref is None:
        return quiet_abs
    return quiet_abs | (rms < ref - DIP_BELOW_LOCAL_DB)


def _snap(idx: int, pause: np.ndarray) -> int:
    """Выровнять границу по началу паузы, в которую она попала.

    Правило нарочно консервативное: двигаем границу, только если она УЖЕ внутри
    паузы, и только назад, к моменту, когда звук прекратился. Искать ближайшую
    паузу в окне нельзя — граница перепрыгивает через громкий кусок в паузу
    внутри следующей речи, музыкальный сегмент заглатывает её начало, хвост
    сегмента становится тихим, и оборванное на полуслове исполнение выглядит
    аккуратно затухающей концовкой. Проверено дважды: оба раза именно так и
    ломалось определение обрыва.

    Нет паузы на границе — значит переход и правда резкий, и его надо сохранить.
    """
    if idx <= 0 or idx >= len(pause) or not pause[idx]:
        return idx
    j = idx
    while j > 0 and pause[j - 1]:
        j -= 1
    return j


def _split_takes(seg: dict, pause: np.ndarray, fps: float) -> list[dict]:
    """Разрезать длинную музыку на дубли по внутренним паузам.

    Репетиция — это один и тот же кусок несколько раз подряд; Виterbi видит их
    как одну музыку, а пользователю нужны отдельные дубли.
    """
    a, b = seg["frame_start"], seg["frame_end"]
    if (b - a) / fps < 2 * TAKE_SPLIT_MIN_LEN_S:
        return [seg]
    quiet = pause[a:b]
    gaps = []
    i = 0
    min_gap = int(MIN_DIP_S * fps)
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if j - i >= min_gap:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    if not gaps:
        return [seg]
    cuts = []
    min_len = int(TAKE_SPLIT_MIN_LEN_S * fps)
    last = 0
    for gi, gj in gaps:
        mid = (gi + gj) // 2
        if mid - last >= min_len and (b - a) - mid >= min_len:
            cuts.append(mid)
            last = mid
    if not cuts:
        return [seg]
    out = []
    edges = [0] + cuts + [b - a]
    for k in range(len(edges) - 1):
        out.append({**seg, "frame_start": a + edges[k], "frame_end": a + edges[k + 1],
                    "split_reason": "пауза внутри музыки: отдельный дубль"})
    return out


def _consolidate(segs: list[dict], fps: float) -> list[dict]:
    """Склейка одинаковых соседей и поглощение коротышей — по кругу до стабилизации.

    Один проход не годится: поглотив короткую паузу, мы оставляем рядом два
    куска с одной меткой, и без второго круга они остаются раздельными.
    """
    cur = [dict(s) for s in segs]
    for _ in range(50):
        changed = False

        out: list[dict] = []
        for s in cur:
            if out and out[-1]["state"] == s["state"]:
                out[-1]["frame_end"] = s["frame_end"]
                changed = True
                continue
            out.append(s)
        cur = out

        for i, s in enumerate(cur):
            dur = (s["frame_end"] - s["frame_start"]) / fps
            keep_floor = (MIN_SILENCE_SEGMENT_S if s["state"] == "silence" else MIN_SEGMENT_S)
            if dur >= keep_floor or len(cur) == 1:
                continue
            left = cur[i - 1] if i > 0 else None
            right = cur[i + 1] if i + 1 < len(cur) else None
            if left is None and right is None:
                continue
            if left is None:
                host = right
            elif right is None:
                host = left
            else:
                host = left if (left["frame_end"] - left["frame_start"]) >= \
                    (right["frame_end"] - right["frame_start"]) else right
            host["frame_start"] = min(host["frame_start"], s["frame_start"])
            host["frame_end"] = max(host["frame_end"], s["frame_end"])
            cur.pop(i)
            changed = True
            break

        if not changed:
            break
    return cur


def run(wdir: Path, feats: dict[str, np.ndarray], fmeta: dict,
        tags: dict[str, np.ndarray] | None) -> dict:
    fps = fmeta["fps"]
    times = feats["times"]
    floor_db = fmeta["noise_floor_db"]
    lvl = feats["rms_db"] - floor_db

    pause = pause_mask(feats, floor_db)

    h = heuristic_scores(feats, fps, floor_db)
    t = tagger_scores(tags or {}, times, lvl)
    if t is None:
        combined = h
    else:
        # Вес теггера плавает от 0.45 до 0.85 по его же решительности: там, где
        # он сомневается, правят акустические признаки; там, где уверен, он.
        decisive = t[:, -1:]
        t = t[:, :-1]
        w = 0.45 + 0.40 * decisive
        combined = (1.0 - w) * h + w * t

    k = int(round(SMOOTH_MEDIAN_S * fps))
    smoothed = np.stack([_median_filter(combined[:, i], k) for i in range(combined.shape[1])], axis=1)
    logp = smoothed - smoothed.max(axis=1, keepdims=True)
    logp = np.log(np.exp(logp).sum(axis=1, keepdims=True) ** -1 * np.exp(logp) + 1e-12)

    path = _viterbi(logp, VITERBI_STICKINESS)

    # длинная пауза — всегда граница, даже если Виterbi её не заметил
    long_sil = int(MIN_SILENCE_SPLIT_S * fps)
    is_quiet = feats["rms_db"] < floor_db + SILENCE_OVER_FLOOR_DB
    i = 0
    while i < len(is_quiet):
        if is_quiet[i]:
            j = i
            while j < len(is_quiet) and is_quiet[j]:
                j += 1
            if j - i >= long_sil:
                path[i:j] = STATES.index("silence")
            i = j
        else:
            i += 1

    runs = _runs(path)
    segs: list[dict] = []
    for a, b, st in runs:
        segs.append({"frame_start": a, "frame_end": b, "state": STATES[st]})

    merged = _consolidate(segs, fps)

    # границы к ближайшей настоящей паузе
    for i in range(1, len(merged)):
        snapped = _snap(merged[i]["frame_start"], pause)
        lo = merged[i - 1]["frame_start"] + int(MIN_SEGMENT_S * fps / 2)
        hi = merged[i]["frame_end"] - int(MIN_SEGMENT_S * fps / 2)
        snapped = int(np.clip(snapped, lo, max(lo, hi)))
        merged[i - 1]["frame_end"] = merged[i]["frame_start"] = snapped

    # притягивание границ к паузе съедает короткие паузы с обеих сторон, после чего
    # от них остаются огрызки в 1 с — второй проход склейки их убирает
    merged = _consolidate(merged, fps)

    final: list[dict] = []
    for s in merged:
        final.extend(_split_takes(s, pause, fps) if s["state"] == "music" else [s])

    out_segs = []
    for n, s in enumerate(final, 1):
        a, b = s["frame_start"], s["frame_end"]
        out_segs.append({
            "id": f"seg{n:04d}",
            "start": round(float(times[min(a, len(times) - 1)]), 3),
            "end": round(float(times[min(b, len(times) - 1)]), 3)
            if b < len(times) else round(float(fmeta["duration_s"]), 3),
            "frame_start": a, "frame_end": b,
            "state": s["state"],
            "split_reason": s.get("split_reason"),
            "state_support": {
                STATES[i]: round(float(np.mean(smoothed[a:b, i])), 3)
                for i in range(len(STATES))
            },
        })
        out_segs[-1]["duration"] = round(out_segs[-1]["end"] - out_segs[-1]["start"], 3)

    result = {
        "states": STATES,
        "pause_total_s": round(float(pause.sum()) / fps, 1),
        "used_tagger": t is not None,
        "noise_floor_db": floor_db,
        "fps": fps,
        "segments": out_segs,
    }
    (wdir / "segments.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    hist: dict[str, float] = {}
    for s in out_segs:
        hist[s["state"]] = hist.get(s["state"], 0.0) + s["duration"]
    log.info("segment: %d сегментов; по времени %s", len(out_segs),
             ", ".join(f"{k} {v/60:.1f} мин" for k, v in sorted(hist.items(), key=lambda x: -x[1])))
    return result
