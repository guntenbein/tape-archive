"""Стадия 2 — features: покадровые признаки всей ленты.

Две сетки кадров:
  * мелкая, 23 мс (hop 512) — всё, что решает границы сегментов и качество;
  * крупная, 93 мс (hop 2048) — хрома CQT для сравнения версий песен.

Длинный файл идёт блоками с перекрытием: память не растёт с длиной ленты,
а сетка кадров остаётся глобальной и непрерывной. HPSS считается на
mel-спектре, а не на полном STFT: то же разделение гармоники и перкуссии
выходит в 13 раз дешевле, а на двухчасовой ленте это разница в часы.
"""
from __future__ import annotations
import json
from pathlib import Path

import librosa
import numpy as np

from . import audio
from .config import ANALYSIS_SR, LOCAL_REF_WINDOW_S
from .runlog import log

N_FFT = 2048
HOP = 512
CHROMA_HOP = 2048
N_MFCC = 13
N_MELS = 96
BLOCK_S = 240.0
OVERLAP_S = 8.0

FINE_SCALARS = ["rms_db", "centroid", "rolloff85", "rolloff95", "flatness", "bandwidth",
                "zcr", "harmonic_ratio", "onset", "hf_ratio", "lf_ratio",
                "speech_band_ratio", "chroma_peak", "chroma_entropy"]


def _blocks(n: int, per: int, pad: int) -> list[tuple[int, int, int, int]]:
    out, k = [], 0
    while k < n:
        k1 = min(n, k + per)
        out.append((k, k1, max(0, k - pad), min(n, k1 + pad)))
        k = k1
    return out


def _windowed(x: np.ndarray, win: int, fn) -> np.ndarray:
    """fn по скользящему окну, с шагом win/4 и линейной интерполяцией между центрами."""
    n = len(x)
    if n == 0:
        return x
    half, step = win // 2, max(1, win // 4)
    centers = np.arange(0, n, step)
    vals = np.array([fn(x[max(0, c - half): min(n, c + half + 1)]) for c in centers],
                    dtype=np.float32)
    if len(centers) == 1:
        return np.full(n, vals[0], dtype=np.float32)
    return np.interp(np.arange(n), centers, vals).astype(np.float32)


def _mod_syllabic(rms_db: np.ndarray, fps: float) -> np.ndarray:
    """Доля энергии огибающей в слоговом диапазоне 3–6.5 Гц.

    Речь модулируется на слоговой частоте ~4 Гц; у музыки энергия огибающей
    сидит на доле такта (<3 Гц) и на атаках (>7 Гц). Самый дешёвый честный
    признак «речь или музыка», не требующий никакой модели.
    """
    env = rms_db.astype(np.float64)

    def ratio(seg: np.ndarray) -> float:
        if len(seg) < 24:
            return 0.0
        s = seg - seg.mean()
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), d=1.0 / fps)
        syl = spec[(fr >= 3.0) & (fr <= 6.5)].sum()
        slow = spec[(fr >= 0.5) & (fr < 3.0)].sum()
        fast = spec[(fr > 6.5) & (fr <= 16.0)].sum()
        return float(syl / (syl + slow + fast + 1e-12))

    return _windowed(env.astype(np.float32), int(round(3.0 * fps)), ratio)


def _pulse_clarity(onset: np.ndarray, fps: float) -> np.ndarray:
    """Выраженность регулярного бита: насколько пик автокорреляции атак торчит над фоном.

    Берётся именно превышение над медианой по диапазону лагов, а не абсолютный
    максимум: у медленно плывущей огибающей автокорреляция высока на всех лагах,
    и абсолютный максимум давал бы «бит» даже в разговоре.
    """
    win = int(round(6.0 * fps))
    lo, hi = max(1, int(0.25 * fps)), int(1.6 * fps)

    def clarity(seg: np.ndarray) -> float:
        if len(seg) < hi * 2:
            return 0.0
        s = seg.astype(np.float64)
        s = s - s.mean()
        e = float((s * s).sum())
        if e <= 1e-9:
            return 0.0
        ac = np.correlate(s, s, mode="full")[len(s) - 1:] / e
        band = ac[lo:hi]
        if band.size < 4:
            return 0.0
        return float(np.clip(band.max() - np.median(band), 0.0, 1.0))

    return _windowed(onset, win, clarity)


def run(wdir: Path, probe: dict) -> dict:
    wav = wdir / f"analysis_{ANALYSIS_SR}.wav"
    y, sr = audio.read_wav(wav)
    fps = sr / HOP
    n_fine = 1 + len(y) // HOP
    yp = np.pad(y, (N_FFT // 2, N_FFT // 2))      # эмуляция center=True для стыка блоков

    acc: dict[str, list] = {k: [] for k in FINE_SCALARS}
    acc["mfcc"] = []

    mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    b_hf, b_lf = freqs >= 8000, freqs <= 120
    b_sp = (freqs >= 300) & (freqs <= 3400)

    fine_blocks = _blocks(n_fine, int(BLOCK_S * fps), int(OVERLAP_S * fps))
    for bi, (k0, k1, p0, p1) in enumerate(fine_blocks):
        seg = yp[p0 * HOP: min(len(yp), (p1 - 1) * HOP + N_FFT)]
        if seg.size < N_FFT:
            seg = np.pad(seg, (0, N_FFT - seg.size))
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, center=False)).astype(np.float32)
        P = S ** 2
        tot = P.sum(axis=0) + 1e-12
        mel = mel_fb @ P
        mel_db = librosa.power_to_db(mel)
        H, _ = librosa.decompose.hpss(mel, margin=1.0)
        chroma_fast = librosa.feature.chroma_stft(S=P, sr=sr)
        cn = chroma_fast / (chroma_fast.sum(axis=0, keepdims=True) + 1e-12)

        vals = {
            "rms_db": 20 * np.log10(librosa.feature.rms(
                y=seg, frame_length=N_FFT, hop_length=HOP, center=False)[0] + 1e-10),
            "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
            "rolloff85": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
            "rolloff95": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0],
            "flatness": librosa.feature.spectral_flatness(S=S)[0],
            "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
            "zcr": librosa.feature.zero_crossing_rate(
                seg, frame_length=N_FFT, hop_length=HOP, center=False)[0],
            "harmonic_ratio": H.sum(axis=0) / (mel.sum(axis=0) + 1e-12),
            "onset": librosa.onset.onset_strength(S=mel_db, sr=sr),
            "hf_ratio": P[b_hf].sum(axis=0) / tot,
            "lf_ratio": P[b_lf].sum(axis=0) / tot,
            "speech_band_ratio": P[b_sp].sum(axis=0) / tot,
            # высота пика хромы и её энтропия: у аккорда 3 ярких бина, у речи и шума размазано
            "chroma_peak": cn.max(axis=0),
            "chroma_entropy": (-(cn * np.log(cn + 1e-12)).sum(axis=0) / np.log(12.0)),
        }
        take = slice(k0 - p0, k0 - p0 + (k1 - k0))
        for k in FINE_SCALARS:
            v = librosa.util.fix_length(np.asarray(vals[k], dtype=np.float32),
                                        size=S.shape[1], axis=0)
            acc[k].append(v[take])
        acc["mfcc"].append(librosa.util.fix_length(
            librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC), size=S.shape[1], axis=1)[:, take])
        if bi % 5 == 0:
            log.debug("features: мелкая сетка, блок %d/%d", bi + 1, len(fine_blocks))

    feats: dict[str, np.ndarray] = {k: np.concatenate(acc[k]).astype(np.float32)
                                   for k in FINE_SCALARS}
    feats["mfcc"] = np.concatenate(acc["mfcc"], axis=1).astype(np.float32)
    n_fine = len(feats["rms_db"])

    # крупная сетка: CENS-хрома поверх CQT — именно она переживает смену тональности
    n_coarse = 1 + len(y) // CHROMA_HOP
    pad_m = int(round(OVERLAP_S * sr / CHROMA_HOP))
    chroma_parts = []
    for m0, m1, q0, q1 in _blocks(n_coarse, int(BLOCK_S * sr / CHROMA_HOP), pad_m):
        a, b = q0 * CHROMA_HOP, min(len(y), q1 * CHROMA_HOP + CHROMA_HOP)
        c = librosa.feature.chroma_cens(y=y[a:b], sr=sr, hop_length=CHROMA_HOP)
        off = m0 - q0
        chroma_parts.append(c[:, off: off + (m1 - m0)])
    feats["chroma"] = np.concatenate(chroma_parts, axis=1).astype(np.float32)

    cf = feats["chroma"]
    lag = 6                                   # ~0.55 с: смена аккорда видна, дрожание нет
    if cf.shape[1] > lag:
        a = cf[:, :-lag] / (np.linalg.norm(cf[:, :-lag], axis=0, keepdims=True) + 1e-9)
        b = cf[:, lag:] / (np.linalg.norm(cf[:, lag:], axis=0, keepdims=True) + 1e-9)
        ch = np.concatenate([np.zeros(lag, dtype=np.float32), 1.0 - (a * b).sum(axis=0)])
    else:
        ch = np.zeros(cf.shape[1], dtype=np.float32)
    feats["chroma_change"] = ch.astype(np.float32)

    # скользящий опорный уровень: «насколько громко вокруг в этот момент».
    # Паузы ищутся как провал ниже него, а не ниже полки всего файла.
    feats["local_ref_db"] = _windowed(
        feats["rms_db"], int(round(LOCAL_REF_WINDOW_S * fps)),
        lambda seg: float(np.median(seg)) if seg.size else 0.0)

    feats["mod_syllabic"] = _mod_syllabic(feats["rms_db"], fps)
    feats["pulse_clarity"] = _pulse_clarity(feats["onset"], fps)
    feats["flux"] = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(feats["mfcc"], axis=1), axis=0)]).astype(np.float32)
    feats["times"] = (np.arange(n_fine) * HOP / sr).astype(np.float32)
    feats["chroma_times"] = (np.arange(cf.shape[1]) * CHROMA_HOP / sr).astype(np.float32)

    meta = {
        "sr": sr, "hop": HOP, "chroma_hop": CHROMA_HOP, "n_fft": N_FFT, "fps": fps,
        "n_frames": int(n_fine), "n_chroma_frames": int(cf.shape[1]),
        "duration_s": float(len(y) / sr),
        "noise_floor_db": float(np.percentile(feats["rms_db"], 5)),
        "rms_p50_db": float(np.percentile(feats["rms_db"], 50)),
        "rms_p95_db": float(np.percentile(feats["rms_db"], 95)),
        "keys": sorted(feats.keys()),
    }
    np.savez_compressed(wdir / "features.npz", **feats)
    (wdir / "features.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("features: %d кадров (%.1f мин), полка %.1f дБ, медиана %.1f дБ",
             n_fine, meta["duration_s"] / 60, meta["noise_floor_db"], meta["rms_p50_db"])
    return meta


def load(wdir: Path) -> dict[str, np.ndarray]:
    with np.load(wdir / "features.npz") as z:
        return {k: z[k] for k in z.files}
