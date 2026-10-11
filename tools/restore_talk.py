#!/usr/bin/env python3
"""Реставрация РАЗГОВОРОВ: семейные записи, а не музыка.

Отдельный инструмент, потому что задача другая во всём.

Нарезка: речь нельзя отдавать кусками по три секунды. Соседние разговорные
сегменты склеиваются в непрерывные блоки через паузы до 20 секунд — слушать
надо сцену целиком, а не реплику.

Отбор: по оценке A/B/C/D отбирать нельзя. Рейтинг заточен на музыку, и
короткая реплика набирает мало баллов по определению, хотя для семейного
архива она может быть самым ценным, что есть на ленте. Берутся все блоки
длиннее порога.

Обработка: срез верха на 7 кГц (разборчивость речи держится до 4 кГц, выше
только шипение), шумоподавление сильнее музыкального, declip по факту.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tapearch import s2_features, s3_segment                # noqa: E402
from tapearch.paths import ORIGINALS, WORK                  # noqa: E402

from tapearch.config import DATA_ROOT  # noqa: E402
OUT_ROOT = DATA_ROOT / "restored"
MERGE_GAP_S = 20.0      # склеивать разговоры через паузы короче этой
MIN_BLOCK_S = 25.0      # блоки короче не выделять
TALK_LOWPASS = 7000
TALK_CATEGORIES = {"conversation", "rehearsal"}


def _run(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def blocks_for(wdir: Path) -> list[dict]:
    cls = {c["id"]: c for c in json.loads((wdir / "classify.json").read_text())["segments"]}
    mus = {m["id"]: m for m in json.loads((wdir / "music.json").read_text())["segments"]}
    segs = json.loads((wdir / "segments.json").read_text())["segments"]
    tr = {t["id"]: t for t in json.loads((wdir / "transcript.json").read_text())["segments"]}

    out: list[dict] = []
    for s in segs:
        cat = (mus.get(s["id"]) or {}).get("category") or cls[s["id"]]["category"]
        if cat not in TALK_CATEGORIES:
            continue
        txt = (tr.get(s["id"]) or {}).get("text", "")
        if out and s["start"] - out[-1]["end"] <= MERGE_GAP_S:
            out[-1]["end"] = s["end"]
            out[-1]["text"] = (out[-1]["text"] + " " + txt).strip()
            out[-1]["parts"] += 1
        else:
            out.append({"start": s["start"], "end": s["end"], "text": txt, "parts": 1})
    return [b for b in out if b["end"] - b["start"] >= MIN_BLOCK_S]


def restore(src: Path, out_dir: Path, name: str, b: dict, feats, fmeta,
            clip_ratio: float, hum50: float) -> bool:
    a, d = b["start"], b["end"] - b["start"]
    nw = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from restore_a import find_noise_window
        nw = find_noise_window(feats, fmeta, b["start"], b["end"])
    except Exception:                                   # noqa: BLE001
        nw = None
    pre = ["highpass=f=70"]                             # речи ниже 70 Гц нет
    if clip_ratio > 0.0005:
        pre.append("adeclip=window=55:overlap=75:arorder=8:threshold=10")
    if hum50 >= 8:
        pre.append(f"equalizer=f=50:t=q:w={0.6 if hum50 >= 18 else 1.0}:g=-{min(28, int(hum50) + 3)}")
        pre.append("equalizer=f=100:t=q:w=1.2:g=-14")
    pre.append(f"lowpass=f={TALK_LOWPASS}")
    chain = ",".join(pre)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if not _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}",
                     "-t", f"{d:.3f}", "-i", str(src), "-af", chain,
                     "-c:a", "pcm_s24le", str(td / "sig.wav")]):
            return False
        clean = td / "sig.wav"
        if nw and _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{nw[0]:.3f}",
                        "-t", f"{nw[1]:.3f}", "-i", str(src), "-af", chain,
                        "-c:a", "pcm_s24le", str(td / "n.wav")]) and \
           _run(["sox", str(td / "n.wav"), "-n", "noiseprof", str(td / "n.prof")]) and \
           _run(["sox", str(td / "sig.wav"), str(td / "clean.wav"), "gain", "-6",
                 "noisered", str(td / "n.prof"), "0.32", "gain", "-n", "-1"]):
            clean = td / "clean.wav"
        else:
            _run(["sox", str(td / "sig.wav"), str(td / "clean.wav"), "gain", "-n", "-1"])
            clean = td / "clean.wav"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(clean),
              "-c:a", "flac", str(out_dir / f"{name}.flac")])
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(clean),
              "-c:a", "libmp3lame", "-b:a", "192k", str(out_dir / f"{name}.mp3")])
    return True


def main() -> None:
    man = json.loads((DATA_ROOT / "manifest.json").read_text())["files"]
    total = 0
    index: dict = {}
    for wdir in sorted(WORK.iterdir()):
        if not (wdir / "rank.json").exists() or wdir.name not in man:
            continue
        bl = blocks_for(wdir)
        if not bl:
            continue
        rec = man[wdir.name]
        src = ORIGINALS / rec["name"]
        probe = json.loads((wdir / "probe.json").read_text())
        clip = probe["clipping"].get("clip_ratio") or 0.0
        hum = (probe.get("noise", {}).get("hum_prominence_db") or {}).get("50") or 0.0
        feats = s2_features.load(wdir)
        fmeta = json.loads((wdir / "features.meta.json").read_text())
        out_dir = OUT_ROOT / wdir.name
        done = []
        for n, b in enumerate(bl, 1):
            nm = (f"TALK{n:02d}_{int(b['start'])//60:02d}m{int(b['start'])%60:02d}s"
                  f"-{int(b['end'])//60:02d}m{int(b['end'])%60:02d}s")
            if restore(src, out_dir, nm, b, feats, fmeta, clip, hum):
                done.append({"file": f"{nm}.flac", "start": b["start"], "end": b["end"],
                             "duration": round(b["end"] - b["start"], 1),
                             "merged_from": b["parts"], "text": b["text"][:400]})
                total += 1
                print(f"  {wdir.name[:40]:42s} {nm}  {b['end']-b['start']:5.0f} с "
                      f"(склеено из {b['parts']})", flush=True)
        if done:
            index[wdir.name] = {"source": rec["name"], "blocks": done}
            (out_dir / "RAZGOVORY.txt").write_text(
                "\n\n".join(f"{d['file']}  ({d['duration']:.0f} с)\n{d['text']}" for d in done))
    (OUT_ROOT / "manifest_talk.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"\nготово: {total} разговорных блоков")


if __name__ == "__main__":
    main()
