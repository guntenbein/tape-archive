#!/usr/bin/env python3
"""Реставрация A-сегментов всего архива. Оригиналы не трогаются.

Параметры НЕ общие: у каждого сегмента свои, и берутся они из измерений.

  * срез верха — частота, выше которой сигнал перестаёт возвышаться над шумом.
    На проверенном архиве срез разошёлся от 4 до 16 кГц; общая настройка
    либо срезала бы музыку, либо оставляла шипение;
  * профиль шума — с ближайшей паузы, а не из начала ленты: фон по кассете
    гуляет на десятки децибел, чужой профиль сделает хуже;
  * notch и declip — по фактическому гулу и клиппингу этого куска.

Порядок: declip ДО denoise, иначе шумодав примет срезанные вершины за сигнал.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal as sg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tapearch import audio, s2_features, s3_segment      # noqa: E402
from tapearch.paths import ORIGINALS, WORK               # noqa: E402

from tapearch.config import DATA_ROOT  # noqa: E402
OUT_ROOT = DATA_ROOT / "restored"
SR = 44100


def _run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def find_noise_window(feats, fmeta, start: float, end: float,
                      search: float = 90.0, want: float = 2.0) -> tuple[float, float] | None:
    """Ближайшая к сегменту пауза длиной хотя бы `want` секунд."""
    pause = s3_segment.pause_mask(feats, fmeta["noise_floor_db"])
    t = feats["times"]
    near = (t > start - search) & (t < end + search) & ~((t >= start) & (t < end))
    idx = np.flatnonzero(near & pause)
    if idx.size == 0:
        return None
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    fps = fmeta["fps"]
    good = [r for r in runs if len(r) / fps >= want]
    if not good:
        good = [max(runs, key=len)]
        if len(good[0]) / fps < 0.8:
            return None
    mid = (start + end) / 2
    best = min(good, key=lambda r: abs(float(t[r[len(r) // 2]]) - mid))
    return float(t[best[0]]), min(float(t[best[-1]]) - float(t[best[0]]), 6.0)


def _band_db(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(max(p[m].mean(), 1e-20)) if m.any() else None


def _flatten_hz(f, ps) -> int | None:
    """Где спектр самого сигнала перестаёт падать и выходит на ровную полку.

    Второй, независимый признак конца музыки. Нужен потому, что сравнение с
    паузой обманывается: если пауза тише, чем шум, едущий вместе с музыкой,
    «сигнал выше шума» тянется до потолка. На одной из записей спектр падал до
    -51 дБ на 8 кГц и дальше идёт ровно (-48, -46, -44) — музыки там уже нет,
    а по сравнению с паузой выходило 16 кГц.

    Пороги подобраны по четырём кускам, где ответ известен из прямых замеров:
    полка начинается не там, где спад замедлился, а там, где он практически
    прекратился (меньше 1 дБ на 3 кГц) И уровень уже на 40 дБ ниже середины.
    На одной из проверенных записей спад держался -1.8 дБ на 3 кГц вплоть до
    13 кГц, и порог в 2 дБ срезал бы музыку на восьми.
    Более мягкое условие срезало настоящую музыку.
    """
    ref = _band_db(f, ps, 300, 1000)
    if ref is None:
        return None
    levels = {}
    for lo in range(2000, 20000, 1000):
        v = _band_db(f, ps, lo, lo + 1000)
        if v is not None:
            levels[lo] = v - ref
    for lo in sorted(levels):
        if levels[lo] > -40.0:
            continue                         # ещё слишком громко: это музыка
        ahead = [levels[lo + k] for k in (1000, 2000, 3000) if lo + k in levels]
        if len(ahead) < 3:
            break
        if max(ahead) - levels[lo] > -1.0:   # спад практически прекратился
            return lo
    return None


def crossover_hz(src: Path, a: float, d: float, na: float, nd: float) -> int:
    """Выше какой частоты музыки уже нет.

    Берётся минимум из двух независимых оценок: по превышению над шумом
    ближайшей паузы и по выходу спектра на ровную полку. Каждая по отдельности
    ошибается в свою сторону, вместе дают правдоподобный ответ.
    """
    ys = audio.decode_window(src, a, min(d, 60.0), SR, 1)
    yn = audio.decode_window(src, na, nd, SR, 1)
    if ys.size < SR:
        return 11000
    nper = 1 << 14
    f, ps = sg.welch(ys, fs=SR, nperseg=nper)
    by_snr = None
    if yn.size >= SR // 2:
        _, pn = sg.welch(yn, fs=SR, nperseg=nper)
        edges = np.arange(1500, 18001, 250)
        for lo, hi in zip(edges, edges[1:]):
            m = (f >= lo) & (f < hi)
            if not m.any():
                continue
            snr = 10 * np.log10(max(ps[m].mean(), 1e-20) / max(pn[m].mean(), 1e-20))
            if snr >= 6.0:
                by_snr = int(hi)
    by_shape = _flatten_hz(f, ps)
    cands = [c for c in (by_snr, by_shape) if c]
    if not cands:
        return 6000
    return int(np.clip(min(cands), 4000, 16000))


def restore_one(src: Path, out_dir: Path, name: str, seg: dict, mus: dict,
                feats, fmeta) -> dict | None:
    a, d = seg["start"], seg["duration"]
    nw = find_noise_window(feats, fmeta, seg["start"], seg["end"])
    tech = mus.get("technical", {})
    clip = tech.get("clip_ratio") or 0.0
    hums = {k: v for k, v in (tech.get("hum_prominence_db") or {}).items() if v is not None}
    hum50 = hums.get("50", 0.0)

    lp = crossover_hz(src, a, d, nw[0], nw[1]) if nw else 11000
    pre = ["highpass=f=25"]
    if clip > 0.0005:
        pre.append("adeclip=window=55:overlap=75:arorder=8:threshold=10")
    if hum50 >= 8:
        pre.append(f"equalizer=f=50:t=q:w={0.6 if hum50 >= 18 else 1.0}:g=-{min(28, int(hum50) + 3)}")
        pre.append("equalizer=f=100:t=q:w=1.2:g=-14")
        pre.append("equalizer=f=150:t=q:w=1.5:g=-8")
    pre.append(f"lowpass=f={lp}")
    chain = ",".join(pre)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if not _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}",
                     "-t", f"{d:.3f}", "-i", str(src), "-af", chain,
                     "-c:a", "pcm_s24le", str(td / "sig.wav")]):
            return None
        clean = td / "sig.wav"
        if nw:
            if _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{nw[0]:.3f}",
                     "-t", f"{nw[1]:.3f}", "-i", str(src), "-af", chain,
                     "-c:a", "pcm_s24le", str(td / "noise.wav")]) and \
               _run(["sox", str(td / "noise.wav"), "-n", "noiseprof", str(td / "noise.prof")]):
                if _run(["sox", str(td / "sig.wav"), str(td / "clean.wav"), "gain", "-6",
                         "noisered", str(td / "noise.prof"), "0.26", "gain", "-n", "-1"]):
                    clean = td / "clean.wav"
        if clean == td / "sig.wav":
            _run(["sox", str(td / "sig.wav"), str(td / "clean.wav"), "gain", "-n", "-1"])
            clean = td / "clean.wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(clean),
              "-c:a", "flac", str(out_dir / f"{name}.flac")])
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(clean),
              "-c:a", "libmp3lame", "-b:a", "256k", str(out_dir / f"{name}.mp3")])
    return {"file": f"{name}.flac", "start": a, "end": seg["end"], "duration": round(d, 1),
            "lowpass_hz": lp, "declip": clip > 0.0005, "clip_ratio_before": round(clip, 5),
            "hum50_db": round(hum50, 1),
            "noise_profile_at": round(nw[0], 1) if nw else None,
            "denoised": nw is not None}


def main() -> None:
    manifest: dict = {}
    for wdir in sorted(WORK.iterdir()):
        if not (wdir / "rank.json").exists():
            continue
        slug = wdir.name
        man = json.loads((DATA_ROOT / "manifest.json").read_text())
        rec = man["files"].get(slug)
        if not rec:
            continue
        src = ORIGINALS / rec["name"]
        rk = {r["id"]: r for r in json.loads((wdir / "rank.json").read_text())["segments"]}
        mus = {m["id"]: m for m in json.loads((wdir / "music.json").read_text())["segments"]}
        segs = json.loads((wdir / "segments.json").read_text())["segments"]
        picks = [s for s in segs
                 if rk[s["id"]]["grade"] == "A"
                 and (mus.get(s["id"], {}).get("category") or "").startswith("music")]
        if not picks:
            continue
        feats = s2_features.load(wdir)
        fmeta = json.loads((wdir / "features.meta.json").read_text())
        out_dir = OUT_ROOT / slug
        done = []
        for n, s in enumerate(picks, 1):
            nm = (f"A{n:02d}_{int(s['start'])//60:02d}m{int(s['start'])%60:02d}s"
                  f"-{int(s['end'])//60:02d}m{int(s['end'])%60:02d}s")
            info = restore_one(src, out_dir, nm, s, mus[s["id"]], feats, fmeta)
            if info:
                done.append(info)
                print(f"  {slug[:38]:40s} {nm}  срез {info['lowpass_hz']} Гц"
                      f"{' declip' if info['declip'] else ''}"
                      f"{' denoise' if info['denoised'] else ''}", flush=True)
        if done:
            manifest[slug] = {"source": rec["name"], "segments": done}
            (out_dir / "README.txt").write_text(
                f"Реставрированные A-сегменты из «{rec['name']}».\n"
                f"Оригинал не изменялся. Параметры у каждого куска свои, из измерений.\n"
                f"Подробности — restored/manifest.json\n")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    n = sum(len(v["segments"]) for v in manifest.values())
    print(f"\nготово: {n} кусков в {len(manifest)} папках")


if __name__ == "__main__":
    main()
