"""Нейросетевое второе мнение: PANNs CNN14, обученный на AudioSet.

Эвристики на признаках хорошо ловят «речь или музыка», но принципиально не
различают пение и речь (у пения и тональность, и голосовые форманты) и не
знают про настройку инструмента и аплодисменты. CNN14 выдаёт 527 тегов
AudioSet, среди которых есть ровно нужные: Singing, Speech, Tuning, Applause.

Модель локальная, CPU, ~320 МБ чекпойнт. Если его нет и скачать нельзя —
pipeline работает на одних эвристиках, просто с меньшей уверенностью.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .config import MODELS_ROOT, TAGGER_SR
from .runlog import log

MODELS_DIR = MODELS_ROOT
CHECKPOINT = MODELS_DIR / "Cnn14_mAP=0.431.pth"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

WIN_S = 2.0
HOP_S = 1.0
BATCH = 24

# группы тегов AudioSet -> макроклассы, которыми оперирует сегментация
GROUPS: dict[str, tuple[str, ...]] = {
    "speech": ("Speech", "Male speech, man speaking", "Female speech, woman speaking",
               "Child speech, kid speaking", "Conversation", "Narration, monologue",
               "Babbling", "Whispering", "Shout", "Yell", "Children shouting",
               "Speech synthesizer"),
    "singing": ("Singing", "Male singing", "Female singing", "Child singing",
                "Choir", "Chant", "Humming", "Yodeling", "Rapping", "Vocal music",
                "A capella"),
    "music": ("Music", "Musical instrument", "Guitar", "Acoustic guitar",
              "Electric guitar", "Bass guitar", "Plucked string instrument",
              "Piano", "Electric piano", "Keyboard (musical)", "Drum kit", "Drum",
              "Snare drum", "Bass drum", "Cymbal", "Hi-hat", "Percussion",
              "Rock music", "Pop music", "Folk music", "Accordion", "Violin, fiddle",
              "Brass instrument", "Wind instrument, woodwind instrument", "Synthesizer",
              "Harmonica", "Flute", "Trumpet", "Saxophone", "Organ", "Strum",
              "Bowed string instrument", "Orchestra", "Song"),
    "tuning": ("Tuning fork", "Tuning", "Pizzicato", "Scratching (performance technique)"),
    "applause": ("Applause", "Clapping", "Cheering", "Crowd", "Hubbub, speech noise, speech babble"),
    "laughter": ("Laughter", "Giggle", "Chuckle, chortle", "Snicker", "Belly laugh"),
    "noise": ("Noise", "White noise", "Pink noise", "Static", "Hum", "Mains hum",
              "Hiss", "Rustling leaves", "Rustle", "Crackle", "Clicking", "Tick",
              "Tick-tock", "Microphone", "Distortion", "Sound effect", "Vibration",
              "Cacophony", "Echo", "Noise burst"),
    "silence": ("Silence",),
}


def ensure_checkpoint() -> Path | None:
    """Скачать чекпойнт, если его нет. Возвращает None, если модель недоступна."""
    if CHECKPOINT.exists() and CHECKPOINT.stat().st_size > 100 * 1024 * 1024:
        return CHECKPOINT
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import subprocess
    tmp = CHECKPOINT.with_suffix(".part")
    log.info("tagger: качаю чекпойнт PANNs CNN14 (~320 МБ)")
    cp = subprocess.run(["curl", "-sSL", "--max-time", "900", "-o", str(tmp), CHECKPOINT_URL],
                        capture_output=True, text=True, check=False)
    if cp.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 100 * 1024 * 1024:
        log.warning("tagger: чекпойнт скачать не удалось (%s), работаю без теггера",
                    cp.stderr.strip()[:200])
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(CHECKPOINT)
    return CHECKPOINT


class Tagger:
    def __init__(self) -> None:
        self.ok = False
        ckpt = ensure_checkpoint()
        if ckpt is None:
            return
        try:
            import torch
            from panns_inference import AudioTagging
            from panns_inference.config import labels
            torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
            self.torch = torch
            self.model = AudioTagging(checkpoint_path=str(ckpt), device="cpu")
            self.labels = list(labels)
            self.index = {name: i for i, name in enumerate(self.labels)}
            self.group_idx = {
                g: [self.index[n] for n in names if n in self.index]
                for g, names in GROUPS.items()
            }
            missing = {g: [n for n in names if n not in self.index]
                       for g, names in GROUPS.items()}
            missing = {g: v for g, v in missing.items() if v}
            if missing:
                log.debug("tagger: теги не найдены в AudioSet и пропущены: %s", missing)
            self.ok = True
        except Exception as exc:                        # noqa: BLE001
            log.warning("tagger: модель не поднялась (%s), работаю без теггера", exc)

    def tag(self, y32: np.ndarray, sr: int = TAGGER_SR) -> dict[str, np.ndarray]:
        """Прогнать ленту окнами 2 с с шагом 1 с. Возвращает вероятности по макроклассам."""
        if not self.ok:
            return {}
        win, hop = int(WIN_S * sr), int(HOP_S * sr)
        if len(y32) < win:
            y32 = np.pad(y32, (0, win - len(y32)))
        starts = list(range(0, max(1, len(y32) - win + 1), hop))
        probs = np.zeros((len(starts), len(self.labels)), dtype=np.float32)
        for b in range(0, len(starts), BATCH):
            chunk = starts[b: b + BATCH]
            batch = np.stack([y32[s: s + win] for s in chunk]).astype(np.float32)
            with self.torch.no_grad():
                clip, _ = self.model.inference(batch)
            probs[b: b + len(chunk)] = np.asarray(clip, dtype=np.float32)
        out = {g: probs[:, idx].max(axis=1) if idx else np.zeros(len(starts), dtype=np.float32)
               for g, idx in self.group_idx.items()}
        out["times"] = (np.array(starts, dtype=np.float32) / sr + WIN_S / 2).astype(np.float32)
        # топ-теги на окно пригодятся в отчёте для объяснения решения
        out["_top_idx"] = probs.argsort(axis=1)[:, -5:][:, ::-1].astype(np.int16)
        out["_top_prob"] = np.take_along_axis(probs, out["_top_idx"].astype(np.int64), axis=1)
        return out

    def label_names(self, idx: np.ndarray) -> list[str]:
        return [self.labels[int(i)] for i in idx]
