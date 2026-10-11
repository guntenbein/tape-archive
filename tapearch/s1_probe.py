"""Стадия 1 — probe: технический паспорт файла.

Один потоковый проход по файлу даёт громкость, уровни, точный скан клиппинга,
стерео-метрики и средний спектр. Вторым, крошечным, проходом добираем спектр
самых тихих кусков: подпись шума ленты (гул 50 Гц, шипение) надо снимать там,
где нет музыки, иначе бас и тарелки выдают себя за сетевой гул и за шум.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

from . import audio
from .config import ANALYSIS_SR, HISS_BAND_HZ, HUM_FREQS
from .runlog import log


def _spectrum(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(65536, 1 << int(math.log2(max(1024, len(x) // 4))))
    f, p = signal.welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, scaling="density")
    return f, np.maximum(p, 1e-20)


def _prominence_db(f: np.ndarray, p: np.ndarray, target: float, half: float = 1.5,
                   ref_lo: float = 5.0, ref_hi: float = 25.0) -> float | None:
    """На сколько дБ пик на target выше окружающего фона — так ловим гул 50 Гц."""
    peak = p[(f >= target - half) & (f <= target + half)]
    ref = p[((f >= target - ref_hi) & (f <= target - ref_lo)) |
            ((f >= target + ref_lo) & (f <= target + ref_hi))]
    if peak.size == 0 or ref.size == 0:
        return None
    return float(10.0 * np.log10(peak.max() / np.median(ref)))


def _cutoff_hz(f: np.ndarray, p: np.ndarray, drop_db: float = 40.0) -> float | None:
    """Верхняя граница полосы: у mp3 с кассеты срез говорит и о кодеке, и о ленте."""
    ref_band = p[(f >= 500) & (f <= 2000)]
    if ref_band.size == 0:
        return None
    thr = np.median(ref_band) / (10 ** (drop_db / 10.0))
    k = 9
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    above = np.where((sm > thr) & (f > 2000))[0]
    return float(f[above[-1]]) if above.size else None


def _noise_signature(src: Path, chunks: list[dict], chunk_s: float, sr: int,
                     n_quiet: int = 12) -> dict:
    """Спектр самых тихих кусков: гул, шипение, его стационарность."""
    if not chunks:
        return {}
    order = sorted(range(len(chunks)), key=lambda i: chunks[i]["rms"])[:max(3, n_quiet)]
    pieces, band_db = [], []
    hi = min(HISS_BAND_HZ[1], sr / 2 - 200)
    sos = signal.butter(4, [HISS_BAND_HZ[0], hi], btype="bandpass", fs=sr, output="sos")
    for i in order:
        y = audio.decode_window(src, chunks[i]["start"], chunk_s, sr, channels=1)
        if y.size < sr // 2:
            continue
        pieces.append(y)
        band_db.append(float(10 * np.log10(max((signal.sosfilt(sos, y) ** 2).mean(), 1e-20))))
    if not pieces:
        return {}
    quiet = np.concatenate(pieces)
    f, p = _spectrum(quiet, sr)
    band = (f >= HISS_BAND_HZ[0]) & (f <= hi)
    full = (f >= 50) & (f <= min(sr / 2 - 100, 16000))
    return {
        "measured_on_s": float(len(quiet) / sr),
        "hum_prominence_db": {str(int(t)): _prominence_db(f, p, t) for t in HUM_FREQS},
        "hiss_band_rel_db": float(10 * np.log10(p[band].mean() / p[full].mean())) if band.any() else None,
        "hiss_level_db": float(np.mean(band_db)) if band_db else None,
        "hiss_stationarity_db_std": float(np.std(band_db)) if len(band_db) > 2 else None,
    }


def run(src: Path, wdir: Path, source_sha: str) -> dict:
    probe = audio.ffprobe_json(src)
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = probe.get("format", {})

    m = audio.stream_measure(src, chunk_s=5.0)
    st = m.get("astats", {})
    native_sr = m["sample_rate"]
    duration = m.get("duration_decoded_s") or float(fmt.get("duration") or 0.0)

    spec: dict = {}
    if m.get("psd_mean") is not None:
        f, p = m["psd_freqs"], m["psd_mean"]
        spec["cutoff_hz"] = _cutoff_hz(f, p)
        lo = p[(f >= 200) & (f <= 1000)].mean()
        hi = p[(f >= 4000) & (f <= 8000)].mean()
        spec["spectral_tilt_db"] = float(10 * np.log10(hi / lo))
    noise = _noise_signature(src, m.get("chunks", []), m.get("chunk_s", 5.0), native_sr)

    peak_db, rms_db = st.get("peak_level_db"), st.get("rms_level_db")
    noise_floor = st.get("noise_floor_db")

    out = {
        "file": {"name": src.name, "bytes": src.stat().st_size, "sha256": source_sha},
        "format": {
            "codec": stream.get("codec_name"),
            "codec_long": stream.get("codec_long_name"),
            "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
            "stream_bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
            "sample_rate": native_sr,
            "channels": m.get("channels"),
            "channel_layout": stream.get("channel_layout"),
            "duration_s": duration,
            "duration_ffprobe_s": float(fmt["duration"]) if fmt.get("duration") else None,
            "tags": {k: v for k, v in (fmt.get("tags") or {}).items()
                     if k.lower() in {"encoder", "title", "artist", "date", "comment"}},
        },
        "loudness": m.get("loudness", {}),
        "levels": {
            "peak_level_db": peak_db,
            "rms_level_db": rms_db,
            "rms_peak_db": st.get("rms_peak_db"),
            "rms_trough_db": st.get("rms_trough_db"),
            "crest_factor_db": round(peak_db - rms_db, 2) if None not in (peak_db, rms_db) else None,
            "dynamic_range_db": (round(peak_db - noise_floor, 2)
                                 if None not in (peak_db, noise_floor) else None),
            "dc_offset": st.get("dc_offset"),
            "flat_factor": st.get("flat_factor"),
            "noise_floor_db": noise_floor,
            "entropy": st.get("entropy"),
        },
        "clipping": {
            "clipped_samples": m.get("clipped_samples"),
            "clip_ratio": m.get("clip_ratio"),
            "astats_peak_count": st.get("peak_count"),
            "scan": "полный проход по всем каналам",
        },
        "spectrum": spec,
        "noise": noise,
        "stereo": {
            "lr_correlation": m.get("lr_correlation"),
            "side_to_mid_db": m.get("side_to_mid_db"),
            "effectively_mono": bool(
                m.get("channels") == 1
                or (m.get("lr_correlation") is not None and m["lr_correlation"] > 0.995)),
        },
        "level_profile": {
            "chunk_s": m.get("chunk_s"),
            "rms_db": [round(float(20 * np.log10(max(c["rms"], 1e-9))), 2) for c in m.get("chunks", [])],
        },
    }
    (wdir / "probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    audio.decode_mono(src, wdir / f"analysis_{ANALYSIS_SR}.wav", ANALYSIS_SR)
    log.info("probe: %.1f мин, %s %s Гц %s кбит/с, noise floor %s дБ, клиппинг %.4f%%",
             duration / 60, out["format"]["codec"], native_sr,
             (out["format"]["bit_rate"] or 0) // 1000, noise_floor,
             100 * (m.get("clip_ratio") or 0))
    return out
