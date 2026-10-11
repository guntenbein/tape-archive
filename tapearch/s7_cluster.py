"""Стадия 7 — cluster: одна и та же песня в разных местах архива.

Задача не сводится к фингерпринту файла: один мотив в репетиции играется в
другой тональности, в другом темпе и с ошибками, так что побитовое сравнение
бесполезно. Работает это так:

  * материал описывается CENS-хромой — профилем звучащих нот, без тембра;
  * тональность: хрома сдвигается циклически по всем 12 полутонам, берётся
    лучший сдвиг, так что «та же песня на два тона выше» совпадает;
  * темп: ось времени второго фрагмента растягивается по сетке отношений
    ±33%, берётся лучшее;
  * совпадение ищется как длинная диагональ в матрице похожести, то есть как
    «долгий участок, где обе записи идут нота в ноту», а не как общая похожесть.

Отдельно, дёшево и точно, chromaprint ловит буквально одну и ту же запись,
попавшую в архив дважды, — это другая задача, и путать её с версиями нельзя.
"""
from __future__ import annotations
import json
from pathlib import Path

import networkx as nx
import numpy as np

from . import audio
from .config import (CHROMAPRINT_DUP_BER, CLUSTER_MIN_MATCH_S, CLUSTER_MIN_MUSIC_S,
                     CLUSTER_MIN_RUN_S, CLUSTER_MIN_SIMILARITY, CLUSTER_PREFILTER_TOPK)
from .runlog import log

TARGET_FPS = 1.0            # хрома приводится к ~1 кадру в секунду
RATIOS = np.geomspace(0.75, 1.0 / 0.75, 11)
# Абсолютный порог косинуса здесь не работает: векторы хромы неотрицательны,
# лежат в одном октанте, и косинус между любыми двумя аккордами выходит 0.7–0.9.
# Поэтому матрица похожести бинаризуется по квантилю: совпавшими считаются
# только те кадры, что попали в верхние KAPPA процентов именно этой пары.
# У настоящей версии совпадения выстраиваются в длинную диагональ, у разной
# музыки рассыпаны случайно, и длина диагонали их разделяет.
KAPPA = 0.08                # доля кадров, объявляемых совпавшими
SEARCH_MIN_S = 12.0         # короче этого выравнивание даже не рассматривается
MAX_GAP_FRAMES = 3          # провал внутри совпадения: репетиция — это игра с ошибками


def _seg_chroma(feats: dict[str, np.ndarray], start: float, end: float) -> np.ndarray | None:
    """Хрома сегмента, приведённая к ~1 Гц и нормированная по столбцам."""
    ct = feats["chroma_times"]
    m = (ct >= start) & (ct < end)
    if m.sum() < 8:
        return None
    c = feats["chroma"][:, m]
    hop = float(ct[1] - ct[0]) if len(ct) > 1 else 0.093
    factor = max(1, int(round((1.0 / TARGET_FPS) / hop)))
    n = c.shape[1] // factor
    if n < 8:
        return None
    c = c[:, : n * factor].reshape(12, n, factor).mean(axis=2)
    return (c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)).astype(np.float32)


def _key_invariant(c: np.ndarray) -> np.ndarray:
    """Дескриптор, не зависящий от тональности: модуль ДПФ по 12 бинам хромы.

    Циклический сдвиг хромы (транспонирование) меняет только фазу ДПФ, не модуль,
    поэтому дескриптор одинаков для одной песни в любой тональности.
    """
    mean = c.mean(axis=1)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    mag = np.abs(np.fft.rfft(mean))
    return (mag / (np.linalg.norm(mag) + 1e-9)).astype(np.float32)


def _resample_time(c: np.ndarray, ratio: float) -> np.ndarray:
    n = c.shape[1]
    m = max(4, int(round(n * ratio)))
    idx = np.linspace(0, n - 1, m)
    out = np.empty((12, m), dtype=np.float32)
    for k in range(12):
        out[k] = np.interp(idx, np.arange(n), c[k])
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)


def _longest_match(hot: np.ndarray, sim: np.ndarray) -> tuple[int, int, float]:
    """Длиннейший связный участок совпадения: (совпавших кадров, длина участка, похожесть).

    Считаются именно совпавшие кадры, а не длина отрезка. Репетиция — это игра
    с ошибками и сбивками: требовать непрерывного совпадения значит терять
    настоящие дубли там, где музыканты на пару секунд разъехались.
    """
    best = (0, 0, 0.0)
    i = 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j, gap, matched = i, 0, 0
        k = i
        while k < len(hot):
            if hot[k]:
                j, gap = k, 0
                matched += 1
            else:
                gap += 1
                if gap > MAX_GAP_FRAMES:
                    break
            k += 1
        if matched > best[0]:
            best = (matched, j - i + 1, float(sim[i:j + 1].mean()))
        i = k
    return best


def compare(ca: np.ndarray, cb: np.ndarray, fps: float = TARGET_FPS) -> dict:
    """Лучшее выравнивание двух фрагментов по всем тональностям и темпам.

    Мерой служит АБСОЛЮТНАЯ длительность совпадения и похожесть на совпавших
    кадрах, а не доля сегмента. Доля даёт перевёрнутый ответ: пятнадцать секунд
    случайного созвучия внутри сорокасекундного куска выглядят убедительнее,
    чем сорок секунд настоящего совпадения внутри трёхминутной песни. При этом
    вероятность случайно совпасть падает именно с длиной, а не с долей.
    """
    search_min = int(SEARCH_MIN_S * fps)
    best = {"score": 0.0, "semitones": 0, "tempo_ratio": 1.0, "match_s": 0.0,
            "run_s": 0.0, "mean_similarity": 0.0, "coverage": 0.0, "offset_s": 0.0}
    best_matched = -1
    for ratio in RATIOS:
        cbr = _resample_time(cb, ratio)
        n, m = ca.shape[1], cbr.shape[1]
        if min(n, m) < search_min:
            continue
        for shift in range(12):
            S = ca.T @ np.roll(cbr, shift, axis=0)
            thr = float(np.quantile(S, 1.0 - KAPPA))
            B = S >= thr
            for k in range(-(n - search_min), m - search_min + 1):
                diag_b, diag_s = B.diagonal(k), S.diagonal(k)
                if diag_b.size < search_min:
                    continue
                matched = int(diag_b.sum())
                if matched <= best_matched:
                    continue
                run, _, _ = _longest_match(diag_b, diag_s)
                sim = float(diag_s[diag_b].mean()) if matched else 0.0
                best_matched = matched
                best = {
                    "score": round(min(1.0, matched / (2.0 * CLUSTER_MIN_MATCH_S * fps)) * sim, 4),
                    "semitones": shift if shift <= 6 else shift - 12,
                    "tempo_ratio": round(float(ratio), 3),
                    "match_s": round(matched / fps, 1),
                    "run_s": round(run / fps, 1),
                    "mean_similarity": round(sim, 3),
                    "coverage": round(matched / min(n, m), 3),
                    "offset_s": round(float(k) / fps, 1),
                }
    return best


def is_same_song(res: dict) -> tuple[bool, str]:
    """Решение по трём независимым признакам сразу — так устойчивее, чем по одному."""
    fails = []
    if res["match_s"] < CLUSTER_MIN_MATCH_S:
        fails.append(f"совпало всего {res['match_s']} с (нужно {CLUSTER_MIN_MATCH_S:.0f})")
    if res["run_s"] < CLUSTER_MIN_RUN_S:
        fails.append(f"подряд только {res['run_s']} с (нужно {CLUSTER_MIN_RUN_S:.0f})")
    if res["mean_similarity"] < CLUSTER_MIN_SIMILARITY:
        fails.append(f"похожесть {res['mean_similarity']} (нужно {CLUSTER_MIN_SIMILARITY})")
    if fails:
        return False, "; ".join(fails)
    return True, (f"совпало {res['match_s']} с (подряд {res['run_s']} с) при похожести "
                  f"{res['mean_similarity']}, сдвиг {res['semitones']:+d} полутона, "
                  f"темп ×{res['tempo_ratio']}")


def _fp_distance(fa: list[int], fb: list[int]) -> float | None:
    """Доля несовпавших бит chromaprint при лучшем сдвиге: та же запись или нет."""
    if not fa or not fb:
        return None
    a, b = np.array(fa, dtype=np.uint32), np.array(fb, dtype=np.uint32)
    best = 1.0
    span = min(len(a), len(b))
    for off in range(-min(60, len(a) - 1), min(60, len(b) - 1) + 1):
        if off >= 0:
            x, y = a[: span - off], b[off: off + span - off]
        else:
            x, y = a[-off: -off + span + off], b[: span + off]
        n = min(len(x), len(y))
        if n < 20:
            continue
        xor = np.bitwise_xor(x[:n], y[:n])
        bits = np.unpackbits(xor.astype(">u4").view(np.uint8)).sum()
        best = min(best, bits / (n * 32))
    return float(best)


def run(items: list[dict], out_dir: Path) -> dict:
    """items: по одному на музыкальный сегмент: slug, src, feats, start, end, duration, meta."""
    cand = [it for it in items if it["duration"] >= CLUSTER_MIN_MUSIC_S]
    log.info("cluster: сравниваю %d музыкальных сегментов из %d (короче %.0f с не берём)",
             len(cand), len(items), CLUSTER_MIN_MUSIC_S)
    for it in cand:
        it["chroma"] = _seg_chroma(it["feats"], it["start"], it["end"])
    cand = [it for it in cand if it["chroma"] is not None]
    for it in cand:
        it["descriptor"] = _key_invariant(it["chroma"])

    n = len(cand)
    pairs: set[tuple[int, int]] = set()
    if n > 1:
        D = np.stack([it["descriptor"] for it in cand])
        sim = D @ D.T
        np.fill_diagonal(sim, -1)
        topk = min(CLUSTER_PREFILTER_TOPK, n - 1)
        for i in range(n):
            for j in np.argsort(-sim[i])[:topk]:
                pairs.add((min(i, int(j)), max(i, int(j))))

    edges, rejected = [], []
    for i, j in sorted(pairs):
        res = compare(cand[i]["chroma"], cand[j]["chroma"])
        ok, why = is_same_song(res)
        if not ok:
            if res["match_s"] >= SEARCH_MIN_S:
                rejected.append({**res, "a": cand[i]["key"], "b": cand[j]["key"], "why": why})
            continue
        if True:
            fp = _fp_distance(cand[i].get("fingerprint") or [], cand[j].get("fingerprint") or [])
            res.update({
                "a": cand[i]["key"], "b": cand[j]["key"], "why": why,
                "fingerprint_ber": round(fp, 3) if fp is not None else None,
                "same_recording": bool(fp is not None and fp < CHROMAPRINT_DUP_BER),
            })
            edges.append(res)

    g = nx.Graph()
    for it in cand:
        g.add_node(it["key"])
    for e in edges:
        g.add_edge(e["a"], e["b"], weight=e["score"])

    clusters = []
    for ci, comp in enumerate(nx.connected_components(g), 1):
        members = sorted(comp)
        if len(members) < 2:
            continue
        inner = [e for e in edges if e["a"] in comp and e["b"] in comp]
        by_key = {it["key"]: it for it in cand}
        clusters.append({
            "id": f"cluster{ci:03d}",
            "members": [{
                "key": k, "slug": by_key[k]["slug"], "segment_id": by_key[k]["segment_id"],
                "start": by_key[k]["start"], "end": by_key[k]["end"],
                "duration": by_key[k]["duration"],
                "completeness": by_key[k]["meta"].get("completeness"),
                "restoration_score": by_key[k]["meta"].get("restoration", {}).get("score"),
                "clip_ratio": by_key[k]["meta"].get("technical", {}).get("clip_ratio"),
                "has_vocal": by_key[k]["meta"].get("has_vocal"),
            } for k in members],
            "links": [{kk: e[kk] for kk in ("a", "b", "score", "semitones", "tempo_ratio",
                                            "match_s", "run_s", "mean_similarity", "why",
                                            "same_recording", "fingerprint_ber")} for e in inner],
            "n_members": len(members),
        })

    result = {
        "params": {"kappa": KAPPA,
                   "min_match_s": CLUSTER_MIN_MATCH_S, "min_run_s": CLUSTER_MIN_RUN_S,
                   "min_similarity": CLUSTER_MIN_SIMILARITY, "chroma_fps": TARGET_FPS,
                   "tempo_ratios": [round(float(r), 3) for r in RATIOS],
                   "min_segment_s": CLUSTER_MIN_MUSIC_S},
        "n_compared": n, "n_pairs_checked": len(pairs),
        "all_edges": edges,
        "rejected_pairs": sorted(rejected, key=lambda r: -r["match_s"])[:40],
        "clusters": clusters,
        "singletons": [it["key"] for it in cand if g.degree(it["key"]) == 0],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clusters.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info("cluster: проверено пар %d, рёбер %d, кластеров %d, одиночек %d",
             len(pairs), len(edges), len(clusters), len(result["singletons"]))
    return result
