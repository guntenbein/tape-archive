#!/usr/bin/env python3
"""Синтетическая «кассета» для проверки pipeline без настоящих записей.

Внутри нарочно уложены все случаи, которые должен различать анализ:
шум ленты, речеподобные фрагменты, настройка инструмента, оборванный дубль,
тот же мотив в другой тональности и темпе (проверка кластеризации версий),
аплодисменты, клиппинг, провал уровня. Детерминировано: seed фиксирован.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100
RNG = np.random.default_rng(20260107)
# две РАЗНЫЕ прогрессии: X — минорная i-VI-III-VII, Y — мажорная I-IV-V-I.
# Нужны обе: без второй в тесте нет отрицательного примера, и порог
# кластеризации нечем проверить — она просто склеит всё со всем.
PROG_X = [(0, "min"), (-4, "maj"), (3, "maj"), (-2, "maj")]
PROG_Y = [(3, "maj"), (8, "maj"), (10, "maj"), (3, "maj")]
TRIADS = {"min": (0, 3, 7), "maj": (0, 4, 7)}


def t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def note(freq: float, dur: float, kind: str = "pluck", amp: float = 0.2) -> np.ndarray:
    n = int(dur * SR)
    tt = t(n)
    if kind == "pluck":
        env = np.exp(-tt * 3.2)
        harm = sum(np.sin(2 * np.pi * freq * k * tt) / (k ** 1.6) for k in range(1, 9))
    else:
        env = np.minimum(1.0, tt * 40) * np.exp(-tt * 1.1)
        harm = sum(np.sin(2 * np.pi * freq * k * tt) / (k ** 2.2) for k in range(1, 6))
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.1 * tt)
    return (amp * env * harm * vib).astype(np.float32)


def add_at(buf: np.ndarray, seg: np.ndarray, start_samples: int) -> None:
    """Сложить фрагмент в буфер с обрезкой по границам — без этого хвост вылезает за массив."""
    a = max(0, start_samples)
    b = min(len(buf), a + len(seg))
    if b > a:
        buf[a:b] += seg[: b - a]


def midi_hz(semitones: float) -> float:
    return 220.0 * 2 ** (semitones / 12.0)


def drum(dur: float, amp: float = 0.25) -> np.ndarray:
    n = int(dur * SR)
    env = np.exp(-t(n) * 28)
    return (amp * env * RNG.normal(0, 1, n)).astype(np.float32)


def music(dur: float, bpm: float, transpose: float, vocal: bool, seed_mix: float = 1.0,
          with_drums: bool = True, prog: list | None = None) -> np.ndarray:
    """Аккордовая прогрессия + бас + (опционально) бит и «вокал»."""
    PROG = prog if prog is not None else PROG_X
    out = np.zeros(int(dur * SR), dtype=np.float32)
    beat = 60.0 / bpm
    bar = beat * 4
    pos = 0.0
    i = 0
    while pos < dur:
        root, quality = PROG[i % len(PROG)]
        i += 1
        for iv in TRIADS[quality]:
            s = note(midi_hz(root + iv + 12 + transpose), min(bar, dur - pos), "pad", 0.13 * seed_mix)
            add_at(out, s, int(pos * SR))
        b = note(midi_hz(root + transpose - 12), min(bar * 0.9, dur - pos), "pluck", 0.22 * seed_mix)
        add_at(out, b, int(pos * SR))
        if with_drums:
            for k in range(4):
                bt = pos + k * beat
                if bt >= dur:
                    break
                add_at(out, drum(0.12, 0.2 if k % 2 == 0 else 0.1), int(bt * SR))
        if vocal:
            mel = [0, 3, 5, 3, 7, 5, 3, 0]
            for k, st in enumerate(mel):
                bt = pos + k * beat / 2
                if bt >= dur:
                    break
                v = note(midi_hz(root + st + 12 + transpose), beat * 0.45, "pad", 0.12)
                v *= 1 + 0.06 * np.sin(2 * np.pi * 6.0 * t(len(v)))  # вибрато «голоса»
                add_at(out, v, int(bt * SR))
        pos += bar
    return out[: int(dur * SR)]


def speechlike(dur: float, amp: float = 0.14, syl_hz: float = 4.2) -> np.ndarray:
    """Речь имитируем формантным шумом с слоговой модуляцией ~4 Гц."""
    n = int(dur * SR)
    tt = t(n)
    base = RNG.normal(0, 1, n)
    from scipy import signal as sg
    voiced = np.zeros(n, dtype=np.float64)
    for f0, bw, g in ((520, 110, 1.0), (1350, 190, 0.6), (2600, 320, 0.3)):
        sos = sg.butter(2, [max(60, f0 - bw), f0 + bw], btype="bandpass", fs=SR, output="sos")
        voiced += g * sg.sosfilt(sos, base)
    syl = 0.5 + 0.5 * np.sin(2 * np.pi * syl_hz * tt + 0.7 * np.sin(2 * np.pi * 0.3 * tt))
    pauses = (np.sin(2 * np.pi * 0.17 * tt) > -0.75).astype(float)
    env = syl ** 2 * pauses
    return (amp * env * voiced / (np.abs(voiced).max() + 1e-9) * 4).astype(np.float32)


def applause(dur: float, amp: float = 0.18) -> np.ndarray:
    n = int(dur * SR)
    from scipy import signal as sg
    claps = np.zeros(n)
    for _ in range(int(dur * 90)):
        p = RNG.integers(0, n - 500)
        claps[p: p + 400] += RNG.normal(0, 1, 400) * np.exp(-np.arange(400) / 90)
    sos = sg.butter(2, [900, 7000], btype="bandpass", fs=SR, output="sos")
    out = sg.sosfilt(sos, claps)
    env = np.minimum(1, t(n) * 6) * np.minimum(1, (dur - t(n)) * 2)
    return (amp * env * out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def tuning(dur: float) -> np.ndarray:
    """Настройка: одиночные щипки с уползающей высотой, без бита."""
    out = np.zeros(int(dur * SR), dtype=np.float32)
    pos = 0.0
    k = 0
    while pos < dur - 0.8:
        detune = (-18 + 6 * k + RNG.normal(0, 2)) / 100.0
        add_at(out, note(midi_hz(7 + detune), min(1.1, dur - pos), "pluck", 0.17), int(pos * SR))
        pos += RNG.uniform(0.9, 1.6)
        k += 1
    return out


def place(track: np.ndarray, seg: np.ndarray, start: float) -> None:
    add_at(track, seg, int(start * SR))


def build() -> tuple[np.ndarray, list[dict]]:
    total = 9 * 60 + 15
    track = np.zeros(int(total * SR), dtype=np.float32)
    truth: list[dict] = []

    def mark(start, end, label, note_=""):
        truth.append({"start": round(start, 2), "end": round(end, 2),
                      "label": label, "note": note_})

    place(track, speechlike(34, 0.16), 8.0);            mark(8, 42, "conversation", "разговор перед репетицией")
    place(track, tuning(22), 44.0);                      mark(44, 66, "instrument_setup", "подстройка гитары")
    place(track, speechlike(9, 0.15), 68.0);             mark(68, 77, "conversation", "«давай с начала»")
    # дубль 1 мотива X: настоящий обрыв — рубим посреди такта, без затухания,
    # и сразу, без паузы, поверх начинается разговор
    take1 = music(70, 96, 0, vocal=True)[: int(62 * SR)]
    place(track, take1, 80.0);                           mark(80, 142, "partial_song", "мотив X, дубль 1, обрыв")
    place(track, speechlike(11, 0.15), 142.2);           mark(142.2, 153, "conversation", "обсуждение ошибки")
    # дубль 2 мотива X: та же прогрессия, +2 полутона, быстрее, доигран, аплодисменты
    place(track, music(168, 104, 2, vocal=True), 158.0); mark(158, 326, "complete_song", "мотив X, дубль 2, целиком")
    place(track, applause(6), 327.0);                    mark(327, 333, "applause", "хлопки")
    place(track, speechlike(14, 0.14), 336.0);           mark(336, 350, "conversation", "разговор после")
    # другой мотив: ДРУГАЯ прогрессия, инструментал. Это отрицательный пример
    # для кластеризации — он не должен попасть в кластер мотива X
    place(track, music(74, 88, -5, vocal=False, prog=PROG_Y), 354.0)
    mark(354, 428, "instrumental", "мотив Y (другая прогрессия), без вокала")
    # перегруженный кусок с клиппингом и провалом уровня
    loud = (music(34, 112, 0, vocal=True, seed_mix=2.9) * 2.6)[: int(28 * SR)]
    loud[int(12 * SR):int(13.4 * SR)] *= 0.01           # провал уровня (обрыв записи)
    place(track, loud, 440.0);                            mark(440, 468, "partial_song",
                                                               "мотив X перегруженный, клиппинг + провал")
    place(track, speechlike(9, 0.15), 468.2);             mark(468.2, 477, "conversation", "разговор в конце")
    # дословная копия куска дубля 2: проверка, что chromaprint отличает
    # «та же песня в другом дубле» от «та же самая запись дважды»
    take2 = music(168, 104, 2, vocal=True)
    place(track, take2[int(40 * SR): int(84 * SR)], 482.0)
    mark(482, 526, "complete_song", "ДОСЛОВНАЯ копия куска дубля 2")

    # фон всей ленты: шипение + сетевой гул 50 Гц + лёгкий дрейф уровня
    n = len(track)
    from scipy import signal as sg
    hiss = RNG.normal(0, 1, n)
    sos = sg.butter(2, [2500, 14000], btype="bandpass", fs=SR, output="sos")
    hiss = sg.sosfilt(sos, hiss)
    track += (0.0045 * hiss / (np.abs(hiss).max() + 1e-9) * 6).astype(np.float32)
    tt = np.arange(n) / SR
    track += (0.0035 * np.sin(2 * np.pi * 50 * tt)
              + 0.0018 * np.sin(2 * np.pi * 100 * tt)
              + 0.0009 * np.sin(2 * np.pi * 150 * tt)).astype(np.float32)
    track *= (1 + 0.06 * np.sin(2 * np.pi * tt / 190)).astype(np.float32)
    track = np.clip(track, -1.0, 1.0)

    stereo = np.stack([track, np.roll(track, 12) * 0.96], axis=1)   # почти моно, как с кассеты
    return stereo, truth


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic-tape.mp3")
    stereo, truth = build()
    import json
    import soundfile as sf
    raw = out.with_suffix(".wav")
    sf.write(str(raw), stereo, SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
                    "-c:a", "libmp3lame", "-b:a", "128k", "-metadata",
                    "encoder=synthetic-test", str(out)], check=True)
    raw.unlink()
    out.with_suffix(".truth.json").write_text(
        json.dumps({"segments": truth}, ensure_ascii=False, indent=2))
    print(f"{out} готов, {len(stereo)/SR/60:.1f} мин, разметка в {out.with_suffix('.truth.json').name}")


if __name__ == "__main__":
    main()
