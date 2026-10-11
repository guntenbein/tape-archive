#!/usr/bin/env python3
"""Сравнение моделей faster-whisper на CPU: скорость и качество на русской речи.

Запускается на публичных тестовых записях, а не на личном архиве.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    ("medium", "medium"),
    ("large-v3", "large-v3"),
    ("large-v3-turbo", "deepdml/faster-whisper-large-v3-turbo-ct2"),
]
ROOT = Path.home() / ".cache" / "tapearch-models" / "whisper"


def wav_of(src: Path, seconds: float) -> Path:
    dst = Path("/tmp") / f"bench-{src.stem[:20]}-{int(seconds)}.wav"
    if not dst.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-t", str(seconds),
                        "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def main() -> None:
    from faster_whisper import WhisperModel
    ROOT.mkdir(parents=True, exist_ok=True)
    clips = [(Path(p), 90.0) for p in sys.argv[1:]]
    for label, repo in MODELS:
        print(f"\n{'='*78}\n### {label}  ({repo})", flush=True)
        t0 = time.perf_counter()
        try:
            model = WhisperModel(repo, device="cpu", compute_type="int8",
                                 cpu_threads=8, download_root=str(ROOT))
        except Exception as exc:                        # noqa: BLE001
            print(f"  НЕ ЗАГРУЗИЛАСЬ: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"  загрузка модели: {time.perf_counter()-t0:.1f} с", flush=True)
        for src, secs in clips:
            wav = wav_of(src, secs)
            import soundfile as sf
            y, _ = sf.read(str(wav), dtype="float32")
            t0 = time.perf_counter()
            segs, info = model.transcribe(
                y, language="ru", beam_size=5, vad_filter=True,
                condition_on_previous_text=False,
                vad_parameters={"min_silence_duration_ms": 500})
            segs = list(segs)
            dt = time.perf_counter() - t0
            text = " ".join(s.text.strip() for s in segs)
            speech = sum(s.end - s.start for s in segs)
            print(f"\n  {src.name}: {dt:.1f} с на {secs:.0f} с аудио "
                  f"(RTF {dt/secs:.2f}x), речи найдено {speech:.0f} с, "
                  f"сегментов {len(segs)}", flush=True)
            print(f"  текст: {text[:420]}", flush=True)
            if segs:
                avg = sum(s.avg_logprob for s in segs) / len(segs)
                nsp = sum(s.no_speech_prob for s in segs) / len(segs)
                print(f"  avg_logprob {avg:.2f}, no_speech_prob {nsp:.2f}", flush=True)
        del model


if __name__ == "__main__":
    main()
