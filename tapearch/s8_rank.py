"""Стадия 8 — rank: понятная оценка с объяснением, откуда она взялась.

Балл складывается из названных слагаемых, и каждое слагаемое попадает в отчёт
вместе со своей величиной. Это сделано нарочно: оценка, которую нельзя
проверить глазами, бесполезна — ей придётся либо верить, либо игнорировать.

Что оценивается: сохранность и полезность материала, а не музыкальные
достоинства. «Законченная песня с разборчивым вокалом и чистым уровнем» —
это A. «Шум ленты» — это D. Вопрос «хорошая ли это музыка» не ставится.
"""
from __future__ import annotations
import json
from pathlib import Path

from .config import GRADE_CUTOFFS
from .runlog import log


def _grade(score: float) -> str:
    if score >= GRADE_CUTOFFS["A"]:
        return "A"
    if score >= GRADE_CUTOFFS["B"]:
        return "B"
    if score >= GRADE_CUTOFFS["C"]:
        return "C"
    return "D"


GRADE_MEANING = {
    "A": "обязательно послушать, хороший кандидат на реставрацию",
    "B": "потенциально интересно",
    "C": "архивный материал",
    "D": "технический мусор",
}


def score_segment(seg: dict, cls: dict, mus: dict | None, tr: dict | None,
                  cluster_info: dict | None) -> dict:
    """Вернуть балл, оценку и список слагаемых с объяснениями."""
    parts: list[dict] = []

    def add(name: str, value: float, why: str) -> None:
        parts.append({"name": name, "value": round(value, 1), "why": why})

    cat = (mus or {}).get("category") or cls["category"]
    dur = seg["duration"]

    if cat in ("silence", "noise"):
        add("категория", 0, f"{'тишина' if cat == 'silence' else 'шум без содержания'} — смотреть нечего")
        return {"id": seg["id"], "score": 0, "grade": "D",
                "grade_meaning": GRADE_MEANING["D"], "parts": parts,
                "headline": f"D — {'тишина' if cat == 'silence' else 'технический мусор'}, {dur:.0f} с"}

    if cat == "conversation":
        words = len((tr or {}).get("text", "").split())
        base = (34 if words >= 120 else 28 if words >= 60 else 22 if words >= 25
                else 14 if words >= 8 else 8)
        add("разговор", base, f"речь, расшифровано слов: {words}")
        if tr and tr.get("quality", {}).get("usable"):
            add("разборчивость", 8, "расшифровка выглядит осмысленной")
        elif tr:
            add("разборчивость", -6, "расшифровка ненадёжна: " + tr["quality"]["note"])
        if dur >= 180:
            add("длительность", 10, f"{dur/60:.1f} мин разговора — вероятно содержательный кусок")
        elif dur >= 60:
            add("длительность", 6, f"{dur:.0f} с разговора")
        wpm = words / max(dur / 60, 0.1)
        if words and wpm < 25:
            add("плотность речи", -4, f"слов мало на такую длину ({wpm:.0f} в минуту) — "
                                      "возможно, пауз больше, чем речи")
        score = max(0.0, sum(p["value"] for p in parts))
        if words:
            tail = f"разговор, {dur:.0f} с, слов {words}"
        elif tr and tr.get("quality", {}).get("dropped"):
            tail = f"разговор, {dur:.0f} с, речь не разобрана (Whisper выдал заученную фразу)"
        else:
            tail = f"разговор, {dur:.0f} с, речь не разобрана"
        return {"id": seg["id"], "score": round(score), "grade": _grade(score),
                "grade_meaning": GRADE_MEANING[_grade(score)], "parts": parts,
                "headline": f"{_grade(score)} — {tail}"}

    if cat == "applause":
        add("реакция зала", 14, "хлопки — метка конца исполнения, слушать отдельно незачем")
        score = 14.0
        return {"id": seg["id"], "score": round(score), "grade": _grade(score),
                "grade_meaning": GRADE_MEANING[_grade(score)], "parts": parts,
                "headline": f"{_grade(score)} — хлопки, {dur:.0f} с"}

    if cat == "rehearsal":
        words = len((tr or {}).get("text", "").split())
        add("репетиция", 24 if words >= 60 else 20 if words >= 25 else 15 if words >= 8 else 11,
            f"играют и разбирают; расшифровано слов: {words}")
        if dur >= 180:
            add("длительность", 8, f"{dur/60:.1f} мин разбора — вероятно, есть что послушать")
        elif dur >= 60:
            add("длительность", 4, f"{dur:.0f} с разбора")
        if tr and tr.get("quality", {}).get("usable"):
            add("разборчивость", 5, "расшифровка выглядит осмысленной")
        score = max(0.0, sum(p["value"] for p in parts))
        return {"id": seg["id"], "score": round(score), "grade": _grade(score),
                "grade_meaning": GRADE_MEANING[_grade(score)], "parts": parts,
                "headline": f"{_grade(score)} — репетиция, {dur/60:.1f} мин, слов {words}"}

    if cat == "instrument_setup":
        add("настройка", 10, "подстройка инструмента или перебор — не исполнение")
        score = 10.0
        return {"id": seg["id"], "score": round(score), "grade": _grade(score),
                "grade_meaning": GRADE_MEANING[_grade(score)], "parts": parts,
                "headline": f"{_grade(score)} — настройка/перебор, {dur:.0f} с"}

    # дальше только музыка
    comp = (mus or {}).get("completeness")
    comp_pts = {"complete": 32, "likely_complete": 25, "partial": 11, "fragment": 3}.get(comp, 8)
    add("законченность", comp_pts, (mus or {}).get("completeness_reason", "не определена"))

    if 90 <= dur <= 420:
        add("длительность", 14, f"{dur/60:.1f} мин — похоже на целую вещь")
    elif 45 <= dur < 90:
        add("длительность", 8, f"{dur:.0f} с — короткая вещь или крупный фрагмент")
    elif dur > 420:
        add("длительность", 6, f"{dur/60:.1f} мин — длиннее песни, вероятно слитые куски")
    else:
        add("длительность", 2, f"{dur:.0f} с — слишком коротко")

    tech = (mus or {}).get("technical", {})
    clip = tech.get("clip_ratio") or 0.0
    if clip < 0.0005:
        add("клиппинг", 10, "клиппинга практически нет")
    elif clip < 0.005:
        add("клиппинг", 5, f"клиппинг заметен, но терпимый ({100*clip:.2f}%)")
    elif clip < 0.03:
        add("клиппинг", -6, f"клиппинг существенный ({100*clip:.1f}%)")
    else:
        add("клиппинг", -16, f"клиппинг тяжёлый ({100*clip:.1f}%), вершины срезаны")

    lvl = (mus or {}).get("edges", {}).get("level_tail_db")
    rms = tech.get("rms_dbfs")
    if rms is not None:
        if -26 <= rms <= -9:
            add("уровень записи", 8, f"уровень в норме ({rms:.0f} dBFS)")
        elif rms < -36:
            add("уровень записи", -5, f"записано очень тихо ({rms:.0f} dBFS) — шум полезет вперёд")
        else:
            add("уровень записи", 3, f"уровень на краю ({rms:.0f} dBFS)")

    stab = tech.get("level_stability_db_std")
    if stab is not None and stab > 7:
        add("ровность уровня", -4, f"уровень скачет (разброс {stab} дБ)")

    drops = (mus or {}).get("dropouts", {})
    if drops.get("count"):
        add("провалы", -min(14, 4 * drops["count"]),
            f"провалов уровня {drops['count']} на {drops['total_s']:.1f} с — потеря сигнала")

    if (mus or {}).get("has_vocal"):
        conf = cls["attributes"].get("vocal_confidence") or 0.3
        add("вокал", 8 * conf + 2, f"голос слышен ({cls['attributes'].get('vocal_reason','')})")
        if tr and tr.get("text"):
            words = len(tr["text"].split())
            if tr.get("quality", {}).get("usable") and words >= 10:
                add("разборчивость текста", 6, f"слова разбираются, расшифровано {words}")
    else:
        add("вокал", 0, "голоса не слышно — инструментал")

    rest = (mus or {}).get("restoration", {})
    if rest.get("score") is not None:
        add("реставрируемость", (rest["score"] - 50) / 5.0,
            f"оценка восстановимости {rest['score']}/100"
            + ("; стопоры: " + "; ".join(rest.get("blockers", [])) if rest.get("blockers") else ""))

    if cluster_info:
        n = cluster_info["n_members"]
        if cluster_info.get("is_best"):
            add("лучший дубль", 7, f"лучший из {n} исполнений этой вещи в архиве")
        else:
            add("есть дубль получше", -5,
                f"эту вещь в архиве играют {n} раз, и другой дубль сохранился лучше")

    score = max(0.0, min(100.0, sum(p["value"] for p in parts)))
    grade = _grade(score)
    bits = [{"music_complete": "полная песня", "music_partial": "оборванное исполнение",
             "music_fragment": "фрагмент"}.get(cat, "музыка"),
            f"{dur/60:.1f} мин" if dur >= 60 else f"{dur:.0f} с"]
    if (mus or {}).get("has_vocal"):
        bits.append("вокал слышен")
    else:
        bits.append("инструментал")
    bits.append("клиппинга почти нет" if clip < 0.0005 else f"клиппинг {100*clip:.2f}%")
    if drops.get("count"):
        bits.append(f"провалов {drops['count']}")
    return {"id": seg["id"], "score": round(score), "grade": grade,
            "grade_meaning": GRADE_MEANING[grade], "parts": parts,
            "headline": f"{grade} — " + ", ".join(bits)}


def run(wdir: Path, segments: dict, classify: dict, music: dict,
        transcript: dict, clusters: dict | None, slug: str) -> dict:
    by_cls = {c["id"]: c for c in classify["segments"]}
    by_mus = {m["id"]: m for m in music["segments"]}
    by_tr = {t["id"]: t for t in transcript["segments"]}

    cl_lookup: dict[str, dict] = {}
    for c in (clusters or {}).get("clusters", []):
        ranked = sorted(c["members"],
                        key=lambda m: (-(m.get("restoration_score") or 0),
                                       {"complete": 0, "likely_complete": 1,
                                        "partial": 2, "fragment": 3}.get(m.get("completeness"), 4),
                                       -m["duration"]))
        best_key = ranked[0]["key"] if ranked else None
        for m in c["members"]:
            if m["slug"] != slug:
                continue
            cl_lookup[m["segment_id"]] = {
                "cluster_id": c["id"], "n_members": c["n_members"],
                "is_best": m["key"] == best_key,
            }

    out = [score_segment(s, by_cls[s["id"]], by_mus.get(s["id"]), by_tr.get(s["id"]),
                         cl_lookup.get(s["id"]))
           for s in segments["segments"]]
    result = {"grade_meaning": GRADE_MEANING, "cutoffs": GRADE_CUTOFFS, "segments": out}
    (wdir / "rank.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    hist: dict[str, int] = {}
    for r in out:
        hist[r["grade"]] = hist.get(r["grade"], 0) + 1
    log.info("rank: %s", ", ".join(f"{g} {hist.get(g,0)}" for g in "ABCD"))
    return result
