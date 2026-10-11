"""Даты, названные в записи вслух.

Люди часто проговаривают дату перед записью: «восьмое декабря такого-то года,
проверка магнитофона» — а через несколько минут, уже на другой сессии той же
кассеты, называют следующую. Это точная датировка внутри самого звука, и она
надёжнее имени файла: имена обычно дают при оцифровке, по памяти, спустя годы.
На проверенном архиве так нашлись три даты на двух лентах из тринадцати.

Разбирается расшифровка, а не звук, так что стадия почти бесплатна.
"""
from __future__ import annotations
import re

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTH_RU = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля",
            "августа", "сентября", "октября", "ноября", "декабря"]

# «8 декабря 1981 года», «31 декабря 1987», «в декабре 96-го»
_FULL = re.compile(
    r"(?:(\d{1,2})\s*[-–]?\s*)?"
    r"(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]\w*|июн\w*|июл\w*|август\w*|"
    r"сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)"
    r"\s*(\d{4}|\d{2})?\s*(?:год\w*)?", re.IGNORECASE)
# отдельно стоящий год: «записано в 1996 году»
_YEAR = re.compile(r"\b(19[4-9]\d|20[0-1]\d)\b\s*(?:год\w*)?")

MIN_YEAR, MAX_YEAR = 1940, 2015


def _norm_year(raw: str | None) -> int | None:
    if not raw:
        return None
    y = int(raw)
    if y < 100:                      # «96-го» -> 1996
        y += 1900 if y >= 40 else 2000
    return y if MIN_YEAR <= y <= MAX_YEAR else None


def _month_of(word: str) -> int | None:
    w = word.lower()
    for stem, num in MONTHS.items():
        if w.startswith(stem):
            # «ма» совпало бы с «март», поэтому его проверяем последним и строго
            if stem == "ма" and not w.startswith(("мая", "мае", "май")):
                continue
            return num
    return None


def extract(transcript: dict) -> list[dict]:
    """Список дат с таймкодом и фразой, в которой они прозвучали."""
    found: list[dict] = []
    for seg in transcript.get("segments", []):
        for ph in seg.get("items", []):
            text = ph.get("text", "")
            at = ph.get("abs_start", seg.get("start", 0.0))
            seen_spans = []
            for m in _FULL.finditer(text):
                month = _month_of(m.group(2))
                if month is None:
                    continue
                year = _norm_year(m.group(3))
                day = int(m.group(1)) if m.group(1) and 1 <= int(m.group(1)) <= 31 else None
                if year is None and day is None:
                    continue                      # один месяц без года бесполезен
                seen_spans.append(m.span())
                found.append({
                    "at": round(float(at), 1), "day": day, "month": month, "year": year,
                    "confidence": "высокая" if (day and year) else "средняя",
                    "display": (f"{day} {MONTH_RU[month]} {year}" if day and year
                                else f"{MONTH_RU[month]} {year}" if year
                                else f"{day} {MONTH_RU[month]}"),
                    "quote": text.strip()[:160],
                    "segment_id": seg["id"],
                })
            for m in _YEAR.finditer(text):
                if any(a <= m.start() < b for a, b in seen_spans):
                    continue                      # уже учтён как часть полной даты
                year = _norm_year(m.group(1))
                if year is None:
                    continue
                found.append({
                    "at": round(float(at), 1), "day": None, "month": None, "year": year,
                    "confidence": "низкая", "display": f"{year} год",
                    "quote": text.strip()[:160], "segment_id": seg["id"],
                })
    found.sort(key=lambda d: d["at"])
    return found


def summarize(dates: list[dict]) -> dict:
    """Короткий вывод для заголовка отчёта: какие годы звучат на ленте."""
    years = sorted({d["year"] for d in dates if d["year"]})
    best = [d for d in dates if d["confidence"] == "высокая"]
    return {
        "years": years,
        "span": (f"{years[0]}–{years[-1]}" if len(years) > 1 else
                 (str(years[0]) if years else None)),
        "precise": [d["display"] for d in best],
        "count": len(dates),
    }
