"""Раскладка данных. Оригиналы — только чтение, всё производное отдельно."""
from __future__ import annotations
import hashlib
import re
import unicodedata
from pathlib import Path

from .config import DATA_ROOT

ORIGINALS = DATA_ROOT / "originals"
WORK = DATA_ROOT / "work"
DERIVED = DATA_ROOT / "derived"
REPORT = DATA_ROOT / "report"
LOGS = DATA_ROOT / "logs"
COLLECTION = DATA_ROOT / "collection"
MANIFEST = DATA_ROOT / "manifest.json"

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "і": "i", "ў": "w",
    "'": "", "’": "",
}


def slugify(name: str) -> str:
    """Имя файла -> стабильный ascii-слаг. Кириллица транслитерируется."""
    stem = Path(name).stem.lower()
    stem = unicodedata.normalize("NFKC", stem)
    out = []
    for ch in stem:
        out.append(_TRANSLIT.get(ch, ch))
    s = "".join(out)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s or "tape"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def work_dir(slug: str) -> Path:
    d = WORK / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def derived_dir(slug: str) -> Path:
    d = DERIVED / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_layout() -> None:
    for d in (ORIGINALS, WORK, DERIVED, REPORT, LOGS, COLLECTION):
        d.mkdir(parents=True, exist_ok=True)
