"""Обёртки над ffmpeg/ffprobe и утилиты чтения звука.

Оригиналы открываются только на чтение; всё декодированное кладётся в work/.
"""
from __future__ import annotations
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .config import CLIP_RUN_SAMPLES, CLIP_THRESHOLD
from .runlog import log


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ffprobe_json(path: Path) -> dict:
    cp = _run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)])
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe упал на {path.name}: {cp.stderr.strip()[:300]}")
    return json.loads(cp.stdout)


def decode_mono(src: Path, dst: Path, sr: int, force: bool = False) -> Path:
    """Декодировать в моно WAV PCM16 указанной частоты. Кэшируется по наличию файла."""
    if dst.exists() and not force and dst.stat().st_size > 1024:
        return dst
    tmp = dst.with_suffix(".part.wav")
    cp = _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
               "-f", "wav", str(tmp)])
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg decode упал: {cp.stderr.strip()[:300]}")
    tmp.replace(dst)
    return dst


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


_ASTATS_KEYS = {
    "DC offset": "dc_offset", "Min level": "min_level", "Max level": "max_level",
    "Peak level dB": "peak_level_db", "RMS level dB": "rms_level_db",
    "RMS peak dB": "rms_peak_db", "RMS trough dB": "rms_trough_db",
    "Crest factor": "crest_factor", "Flat factor": "flat_factor",
    "Peak count": "peak_count", "Noise floor dB": "noise_floor_db",
    "Noise floor count": "noise_floor_count", "Entropy": "entropy",
    "Dynamic range": "dynamic_range", "Zero crossings": "zero_crossings",
    "Zero crossings rate": "zero_crossings_rate", "Number of samples": "n_samples",
    "Bit depth": "bit_depth", "Abs Peak count": "abs_peak_count",
}


def astats(src: Path) -> dict:
    """ffmpeg astats: пики, RMS, шумовая полка, flat factor, peak count — по каналам и суммарно."""
    cp = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(src),
               "-af", "astats=measure_perchannel=all:measure_overall=all",
               "-f", "null", "-"])
    out: dict = {"channels": [], "overall": {}}
    cur: dict | None = None
    for line in cp.stderr.splitlines():
        line = line.strip()
        m = re.match(r"^\[Parsed_astats.*?\]\s*(Channel|Overall)\b.*?(\d+)?\s*$", line)
        if m:
            cur = {}
            if m.group(1) == "Channel":
                out["channels"].append(cur)
            else:
                out["overall"] = cur
            continue
        m = re.match(r"^\[Parsed_astats.*?\]\s*([A-Za-z][A-Za-z ]+?):\s*(-?[\d.einf]+|-?nan)\s*$", line)
        if m and cur is not None:
            key = _ASTATS_KEYS.get(m.group(1).strip())
            if key:
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    cur[key] = None
    return out


def ebur128(src: Path) -> dict:
    """Интегральная громкость (LUFS), LRA и true peak."""
    cp = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(src),
               "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    txt = cp.stderr
    res: dict = {}
    tail = txt[txt.rfind("Summary"):] if "Summary" in txt else txt
    for key, pat in (("lufs_integrated", r"I:\s*(-?[\d.]+)\s*LUFS"),
                     ("lra", r"LRA:\s*(-?[\d.]+)\s*LU"),
                     ("true_peak_dbtp", r"Peak:\s*(-?[\d.inf]+)\s*dBFS")):
        m = re.search(pat, tail)
        if m:
            try:
                res[key] = float(m.group(1))
            except ValueError:
                res[key] = None
    return res


def silencedetect(src: Path, threshold_db: float, min_dur: float = 0.4) -> list[tuple[float, float]]:
    """Список интервалов тишины по порогу ffmpeg silencedetect."""
    cp = _run(["ffmpeg", "-nostdin", "-v", "info", "-i", str(src),
               "-af", f"silencedetect=noise={threshold_db:.1f}dB:d={min_dur}",
               "-f", "null", "-"])
    starts, spans = [], []
    for line in cp.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            starts.append(float(m.group(1)))
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and starts:
            spans.append((max(0.0, starts.pop()), float(m.group(1))))
    return sorted(spans)


def stream_measure(src: Path, chunk_s: float = 5.0) -> dict:
    """Один декод, из которого выжимается всё сразу.

    ffmpeg одновременно пишет PCM в stdout и отчёты ebur128/astats в stderr,
    поэтому громкость, уровни, точный скан клиппинга и покусочные RMS/спектр
    берутся за один проход по файлу вместо четырёх.
    """
    probe = ffprobe_json(src)
    stream = next((st for st in probe.get("streams", []) if st.get("codec_type") == "audio"), {})
    sr = int(stream.get("sample_rate") or 44100)
    src_ch = int(stream.get("channels") or 2)
    ch = 2 if src_ch >= 2 else 1

    with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as errf:
        err_path = Path(errf.name)
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(src),
         "-af", "ebur128=peak=true:framelog=quiet,astats=measure_perchannel=none:measure_overall=all",
         "-vn", "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=err_path.open("wb"))

    frames = chunk_s * sr
    want = int(frames) * ch * 4
    acc = _StreamAcc(sr, ch, chunk_s)
    assert proc.stdout is not None
    buf = b""
    while True:
        block = proc.stdout.read(want - len(buf))
        if not block:
            break
        buf += block
        if len(buf) >= want:
            acc.push(np.frombuffer(buf, dtype="<f4"))
            buf = b""
    if buf:
        usable = (len(buf) // (ch * 4)) * ch * 4
        if usable:
            acc.push(np.frombuffer(buf[:usable], dtype="<f4"))
    proc.stdout.close()
    proc.wait()
    err_text = err_path.read_text(errors="replace")
    err_path.unlink(missing_ok=True)

    out = acc.result()
    out["sample_rate"] = sr
    out["channels"] = src_ch
    out.update(_parse_astats(err_text))
    out["loudness"] = _parse_ebur128(err_text)
    return out


class _StreamAcc:
    """Накопитель по кускам: клиппинг, стерео-метрики, покусочные RMS и спектры."""

    def __init__(self, sr: int, ch: int, chunk_s: float):
        self.sr, self.ch, self.chunk_s = sr, ch, chunk_s
        self.nperseg = 1 << 14
        self.n = 0
        self.clipped = 0
        self.sum_ll = self.sum_rr = self.sum_lr = self.sum_l = self.sum_r = 0.0
        self.mid_p = self.side_p = 0.0
        self.psd_sum: np.ndarray | None = None
        self.psd_n = 0
        self.chunks: list[dict] = []
        self.t = 0.0

    def push(self, flat: np.ndarray) -> None:
        x = flat.reshape(-1, self.ch) if self.ch > 1 else flat.reshape(-1, 1)
        mono = x.mean(axis=1)
        self.n += x.shape[0]
        self.clipped += _clip_count(x)
        if self.ch > 1:
            l, r = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
            self.sum_ll += float((l * l).sum()); self.sum_rr += float((r * r).sum())
            self.sum_lr += float((l * r).sum())
            self.sum_l += float(l.sum()); self.sum_r += float(r.sum())
            self.mid_p += float((((l + r) / 2) ** 2).sum())
            self.side_p += float((((l - r) / 2) ** 2).sum())
        rms = float(np.sqrt((mono.astype(np.float64) ** 2).mean() + 1e-20))
        if mono.size >= self.nperseg:
            f, p = signal.welch(mono, fs=self.sr, nperseg=self.nperseg, scaling="density")
            p = np.maximum(p, 1e-20)
            self.psd_sum = p if self.psd_sum is None else self.psd_sum + p
            self.psd_n += 1
            self.freqs = f
        self.chunks.append({"start": round(self.t, 3), "rms": rms,
                            "peak": float(np.abs(mono).max()) if mono.size else 0.0})
        self.t += x.shape[0] / self.sr

    def result(self) -> dict:
        res = {
            "n_samples_per_channel": self.n,
            "duration_decoded_s": self.n / self.sr if self.sr else None,
            "clipped_samples": self.clipped,
            "clip_ratio": self.clipped / max(1, self.n * self.ch),
            "chunks": self.chunks,
            "chunk_s": self.chunk_s,
        }
        if self.ch > 1 and self.n:
            n = self.n
            cov = self.sum_lr / n - (self.sum_l / n) * (self.sum_r / n)
            vl = self.sum_ll / n - (self.sum_l / n) ** 2
            vr = self.sum_rr / n - (self.sum_r / n) ** 2
            res["lr_correlation"] = float(cov / np.sqrt(vl * vr)) if vl > 0 and vr > 0 else None
            res["side_to_mid_db"] = float(
                10 * np.log10(max(self.side_p, 1e-20) / max(self.mid_p, 1e-20)))
        if self.psd_sum is not None:
            res["psd_freqs"] = self.freqs
            res["psd_mean"] = self.psd_sum / self.psd_n
        return res


def _clip_count(x: np.ndarray) -> int:
    """Сэмплы в «плоских» вершинах (подряд на пределе) — прямой признак клиппинга."""
    total = 0
    for c in range(x.shape[1]):
        hot = np.abs(x[:, c]) >= CLIP_THRESHOLD
        if not hot.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], hot.view(np.int8), [0]))))
        lens = idx[1::2] - idx[::2]
        total += int(lens[lens >= CLIP_RUN_SAMPLES].sum())
    return total


def _parse_astats(text: str) -> dict:
    cur: dict = {}
    seen_overall = False
    for line in text.splitlines():
        line = line.strip()
        if re.search(r"\]\s*Overall\s*$", line):
            seen_overall = True
            continue
        if not seen_overall:
            continue
        m = re.match(r"^\[Parsed_astats.*?\]\s*([A-Za-z][A-Za-z ]+?):\s*(-?[\d.e+]+|-?inf|-?nan)\s*$", line)
        if m:
            key = _ASTATS_KEYS.get(m.group(1).strip())
            if key:
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    cur[key] = None
    return {"astats": cur}


def _parse_ebur128(text: str) -> dict:
    tail = text[text.rfind("Summary"):] if "Summary" in text else text
    res: dict = {}
    for key, pat in (("lufs_integrated", r"I:\s*(-?[\d.]+)\s*LUFS"),
                     ("lra", r"LRA:\s*(-?[\d.]+)\s*LU"),
                     ("true_peak_dbtp", r"Peak:\s*(-?[\d.inf]+)\s*dBFS")):
        m = re.search(pat, tail)
        if m:
            try:
                res[key] = float(m.group(1))
            except ValueError:
                res[key] = None
    return res


def decode_window(src: Path, start: float, dur: float, sr: int, channels: int = 1) -> np.ndarray:
    """Короткий участок в numpy — для адресного добора (например, тихих кусков)."""
    cp = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(src), "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=False)
    if cp.returncode != 0 or not cp.stdout:
        return np.zeros(0, dtype=np.float32)
    a = np.frombuffer(cp.stdout, dtype="<f4")
    if channels > 1:
        a = a[: (a.size // channels) * channels].reshape(-1, channels)
    return a


def make_preview(src: Path, dst: Path, bitrate: str = "64k", force: bool = False) -> Path:
    """Моно AAC для плеера в отчёте. Оригинал не трогаем, это отдельный производный файл."""
    if dst.exists() and not force and dst.stat().st_size > 1024:
        return dst
    tmp = dst.with_suffix(".part.m4a")
    cp = _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-vn", "-ac", "1", "-c:a", "aac", "-b:a", bitrate,
               "-movflags", "+faststart", str(tmp)])
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg preview упал: {cp.stderr.strip()[:300]}")
    tmp.replace(dst)
    return dst


def waveform_peaks(y: np.ndarray, sr: int, per_second: int = 10) -> dict:
    """min/max в бакетах для отрисовки формы волны на canvas (int8, компактно)."""
    bucket = max(1, int(round(sr / per_second)))
    n = len(y) // bucket
    if n == 0:
        return {"per_second": per_second, "min": [], "max": []}
    trimmed = y[: n * bucket].reshape(n, bucket)
    lo = np.clip(trimmed.min(axis=1) * 127, -127, 127).astype(np.int8)
    hi = np.clip(trimmed.max(axis=1) * 127, -127, 127).astype(np.int8)
    return {"per_second": per_second, "min": lo.tolist(), "max": hi.tolist()}


def cut_wav(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * sr))
    b = min(len(y), int(end * sr))
    return y[a:b] if b > a else np.zeros(1, dtype=np.float32)


def chromaprint(src: Path, start: float, length: float) -> list[int] | None:
    """Chromaprint-фингерпринт участка: ловит буквально одну и ту же запись в двух файлах.

    fpcalc не умеет перематывать внутрь файла, поэтому участок сначала
    вырезается ffmpeg-ом во временный wav.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        cut = _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}",
                    "-t", f"{max(5.0, length):.3f}", "-i", str(src),
                    "-ac", "1", "-ar", "22050", str(tmp)])
        if cut.returncode != 0:
            return None
        cp = _run(["fpcalc", "-raw", "-json", str(tmp)])
        if cp.returncode != 0:
            log.debug("fpcalc не смог %s@%.1f: %s", src.name, start, cp.stderr.strip()[:160])
            return None
        return json.loads(cp.stdout).get("fingerprint")
    except ValueError:
        return None
    finally:
        tmp.unlink(missing_ok=True)
