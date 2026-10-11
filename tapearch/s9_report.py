"""Стадия 9 — report: HTML, который можно открыть и слушать.

Плеер устроен так: на ленту ровно один производный аудиофайл (моно AAC 64
кбит/с), а кнопка на сегменте делает seek и останавливает воспроизведение на
его конце. Поэтому не нужно нарезать сотни клипов на диск, переход между
сегментами мгновенный, а отчёт остаётся переносимым каталогом.

Форма волны рисуется на canvas из заранее посчитанных пиков. Пики кладутся
.js-файлом, а не .json: иначе отчёт, открытый как file://, не смог бы их
загрузить — браузер запрещает fetch с локального диска.
"""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import audio, dates as dates_mod
from .config import PEAKS_PER_SECOND, PREVIEW_BITRATE
from .paths import REPORT, derived_dir
from .runlog import log

CATEGORY_RU = {
    "music": "музыка", "music_complete": "песня целиком",
    "music_partial": "исполнение оборвано", "music_fragment": "музыкальный обрывок",
    "instrument_setup": "настройка / перебор", "conversation": "разговор",
    "rehearsal": "репетиция: играют и разбирают",
    "applause": "хлопки", "noise": "шум, технический мусор", "silence": "тишина",
    "unknown": "не опознано",
}
COMPLETENESS_RU = {
    "complete": "доиграно целиком", "likely_complete": "похоже на целое, начало под вопросом",
    "partial": "оборвано", "fragment": "обрывок", None: "—",
}
COLORS = {
    "music": "#d9a84e", "music_complete": "#d9a84e", "music_partial": "#a8803a",
    "music_fragment": "#8c6c31", "instrument_setup": "#b58bc8", "conversation": "#6fa8c8",
    "applause": "#c8826f", "noise": "#7d7467", "silence": "#4a443b", "unknown": "#7d7467",
}


def hms(t: float) -> str:
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    frac = int((t - int(t)) * 10)
    return (f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}") + (f".{frac}" if not h else "")


def dur_h(t: float) -> str:
    return f"{t/60:.1f} мин" if t >= 60 else f"{t:.0f} с"


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
                      autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters["plural"] = _plural
    return env


def _fmt(v, nd: int = 1, suffix: str = "") -> str:
    return "—" if v is None else f"{float(v):.{nd}f}{suffix}"


def _tech_rows(probe: dict, silence_share: float | None = None) -> list[tuple[str, str]]:
    f, lv, cl = probe["format"], probe["levels"], probe["clipping"]
    sp, nz, st, ld = probe["spectrum"], probe.get("noise", {}), probe["stereo"], probe["loudness"]
    hum = max((v for v in (nz.get("hum_prominence_db") or {}).values() if v is not None), default=None)
    rows = [
        ("длительность", hms(f["duration_s"])),
        ("кодек", f"{f['codec']} {(f['bit_rate'] or 0)//1000} кбит/с"),
        ("частота / каналы", f"{f['sample_rate']} Гц, {f['channels']}"
                             + (" (фактически моно)" if st.get("effectively_mono") else "")),
        ("громкость", f"{ld.get('lufs_integrated')} LUFS, разброс {ld.get('lra')} LU"),
        ("пик", f"{_fmt(lv.get('peak_level_db'))} дБ "
                f"(true peak {_fmt(ld.get('true_peak_dbtp'))} dBTP)"),
        ("RMS / crest", f"{_fmt(lv.get('rms_level_db'))} дБ / {_fmt(lv.get('crest_factor_db'))} дБ"),
        ("шумовая полка", f"{_fmt(lv.get('noise_floor_db'))} дБ, "
                          f"динамика {_fmt(lv.get('dynamic_range_db'))} дБ"),
        ("клиппинг", f"{100*(cl.get('clip_ratio') or 0):.4f}% ({cl.get('clipped_samples')} сэмплов)"),
        ("полоса сверху", f"{round(sp['cutoff_hz']) if sp.get('cutoff_hz') else '—'} Гц"),
        ("сетевой гул", f"пик {hum:.0f} дБ над фоном" if hum is not None else "не найден"),
        ("шипение ленты", f"{nz.get('hiss_level_db'):.0f} дБ, ровность ±{nz.get('hiss_stationarity_db_std'):.1f} дБ"
                          if nz.get("hiss_level_db") is not None else "—"),
        ("корреляция L/R", _fmt(st.get("lr_correlation"), 3)),
        ("тишина и пустоты", f"{100*silence_share:.0f}% ленты"
                             if silence_share is not None else "—"),
        ("sha256 оригинала", probe["file"]["sha256"][:16] + "…"),
    ]
    return [(k, v) for k, v in rows if v not in (None, "None")]


def _seg_tech_rows(m: dict | None) -> list[tuple[str, str]]:
    if not m:
        return []
    t = m.get("technical", {})
    e = m.get("edges", {})
    d = m.get("dropouts", {})
    hum = max((v for v in (t.get("hum_prominence_db") or {}).values() if v is not None), default=None)
    rows = [
        ("пик / RMS / crest", f"{t.get('peak_dbfs')} / {t.get('rms_dbfs')} / {t.get('crest_db')} дБ"),
        ("клиппинг", f"{100*(t.get('clip_ratio') or 0):.3f}% ({t.get('clipped_samples')} сэмплов)"),
        ("сетевой гул", f"{hum:.0f} дБ над фоном" if hum is not None else "не найден"),
        ("шипение (8–11 кГц)", f"{t.get('hiss_band_rel_db')} дБ относительно полосы"),
        ("полоса сверху", f"{t.get('cutoff_hz')} Гц"),
        ("ровность уровня", f"±{t.get('level_stability_db_std')} дБ, дрейф {t.get('level_drift_db')} дБ"),
        ("провалы уровня", f"{d.get('count', 0)} шт на {d.get('total_s', 0)} с" if d.get("count") else "нет"),
        ("края", f"начало {'срезано' if e.get('truncated_start') else 'целое'}, "
                 f"конец {'оборван' if e.get('truncated_end') else 'целый'}; "
                 f"уровень в хвосте {e.get('level_tail_db')} дБ над полкой"),
    ]
    return [(k, v) for k, v in rows if "None" not in str(v)]


def prepare_assets() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "assets").mkdir(exist_ok=True)
    src = Path(__file__).parent / "templates" / "assets"
    for p in src.iterdir():
        shutil.copy2(p, REPORT / "assets" / p.name)


def render_file(slug: str, src: Path, wdir: Path, probe: dict, segments: dict, classify: dict,
                music: dict, transcript: dict, rank: dict, clusters: dict | None,
                feats_wav: Path) -> dict:
    prepare_assets()
    ddir = derived_dir(slug)
    preview = audio.make_preview(src, ddir / "preview.m4a", PREVIEW_BITRATE)
    (REPORT / "audio").mkdir(exist_ok=True)
    link = REPORT / "audio" / f"{slug}.m4a"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / ".." / "derived" / slug / "preview.m4a")

    y, sr = audio.read_wav(feats_wav)
    peaks = audio.waveform_peaks(y, sr, PEAKS_PER_SECOND)
    (REPORT / f"{slug}.peaks.js").write_text(
        "window.TA_PEAKS=" + json.dumps(peaks, separators=(",", ":")) + ";")

    by_cls = {c["id"]: c for c in classify["segments"]}
    by_mus = {m["id"]: m for m in music["segments"]}
    by_tr = {t["id"]: t for t in transcript["segments"]}
    by_rank = {r["id"]: r for r in rank["segments"]}

    cl_lookup: dict[str, dict] = {}
    for c in (clusters or {}).get("clusters", []):
        ranked = _rank_members(c)
        for m in c["members"]:
            if m["slug"] == slug:
                cl_lookup[m["segment_id"]] = {
                    "cluster_id": c["id"], "n_members": c["n_members"],
                    "is_best": m["key"] == (ranked[0]["key"] if ranked else None)}

    rows, bands = [], []
    counts = {g: 0 for g in "ABCD"}
    for s in segments["segments"]:
        cls, mus = by_cls[s["id"]], by_mus.get(s["id"])
        rk = by_rank[s["id"]]
        cat = (mus or {}).get("category") or cls["category"]
        tr = by_tr.get(s["id"])
        counts[rk["grade"]] += 1
        attrs = []
        if mus and mus.get("completeness"):
            attrs.append(COMPLETENESS_RU[mus["completeness"]])
        if mus:
            if mus.get("has_vocal") is True:
                attrs.append(f"вокал слышен (уверенность {mus.get('vocal_confidence')})")
            elif mus.get("has_vocal") is False:
                attrs.append("инструментал")
        if tr:
            attrs.append(f"язык {tr['language']}")
        rows.append({
            "key": f"{slug}:{s['id']}", "anchor": s["id"], "start": s["start"], "end": s["end"],
            "start_hms": hms(s["start"]), "end_hms": hms(s["end"]), "dur_h": dur_h(s["duration"]),
            "category": cat, "category_ru": CATEGORY_RU.get(cat, cat),
            "color": COLORS.get(cat, "#8a8375"),
            "grade": rk["grade"], "headline_tail": rk["headline"].split("— ", 1)[-1],
            "why": "; ".join(cls["reasons"]), "parts": rk["parts"],
            "tech_rows": _seg_tech_rows(mus), "restoration": (mus or {}).get("restoration"),
            "attrs": " · ".join(attrs),
            "cluster": cl_lookup.get(s["id"]),
            # ключ называется phrases, а не items: Jinja разрешил бы .items
            # как метод словаря и молча отдал бы функцию вместо списка
            "transcript": ({**tr, "phrases": [{**i, "hms": hms(i["abs_start"])} for i in tr["items"]]}
                           if tr and tr["items"] else None),
        })
        bands.append({"start": s["start"], "end": s["end"], "color": COLORS.get(cat, "#8a8375")})

    spoken = dates_mod.extract(transcript)
    (wdir / "dates.json").write_text(json.dumps(
        {"dates": spoken, "summary": dates_mod.summarize(spoken)}, ensure_ascii=False, indent=2))
    date_sum = dates_mod.summarize(spoken)
    if spoken:
        log.info("даты, названные вслух: %s", ", ".join(d["display"] for d in spoken[:4]))

    dur_total = probe["format"]["duration_s"] or 1.0
    silence_share = sum(s["duration"] for s in segments["segments"]
                        if by_cls[s["id"]]["category"] in ("silence", "noise")) / dur_total

    html = _env().get_template("file.html").render(
        title=f"{probe['file']['name']} — анализ",
        generated=time.strftime("%Y-%m-%d %H:%M"),
        f={"name": probe["file"]["name"], "duration": probe["format"]["duration_s"],
           "duration_hms": hms(probe["format"]["duration_s"]),
           "preview": f"audio/{slug}.m4a", "peaks_js": f"{slug}.peaks.js"},
        tech=_tech_rows(probe, silence_share), segs=rows, counts=counts,
        spoken_dates=spoken,
        bands=bands)
    (REPORT / f"{slug}.html").write_text(html)
    log.info("report: %s.html — %d сегментов, A %d / B %d / C %d / D %d",
             slug, len(rows), counts["A"], counts["B"], counts["C"], counts["D"])

    music_s = sum(s["duration"] for s in segments["segments"]
                  if ((by_mus.get(s["id"]) or {}).get("category") or by_cls[s["id"]]["category"]
                      ).startswith("music"))
    speech_s = sum(s["duration"] for s in segments["segments"]
                   if by_cls[s["id"]]["category"] == "conversation")
    return {
        "slug": slug, "name": probe["file"]["name"],
        "duration": probe["format"]["duration_s"], "duration_hms": hms(probe["format"]["duration_s"]),
        "n_segments": len(rows), "counts": counts,
        "music_hms": hms(music_s), "speech_hms": hms(speech_s),
        "n_complete": sum(1 for m in music["segments"] if m["category"] == "music_complete"),
        "n_partial": sum(1 for m in music["segments"] if m["category"] == "music_partial"),
        "dates": date_sum,
        "tech_short": (f"{probe['format']['codec']} {(probe['format']['bit_rate'] or 0)//1000}к, "
                       f"{probe['loudness'].get('lufs_integrated')} LUFS, "
                       f"клиппинг {100*(probe['clipping'].get('clip_ratio') or 0):.3f}%"),
        "rows": rows,
    }


def _rank_members(c: dict) -> list[dict]:
    """Лучший источник: сохранность, потом законченность, потом клиппинг, потом длина.

    Клиппинг стоит выше длины нарочно: срезанные вершины не возвращаются, а
    лишние полминуты записи — дело наживное.
    """
    return sorted(c["members"],
                  key=lambda m: (-(m.get("restoration_score") or 0),
                                 {"complete": 0, "likely_complete": 1,
                                  "partial": 2, "fragment": 3}.get(m.get("completeness"), 4),
                                 m.get("clip_ratio") if m.get("clip_ratio") is not None else 1.0,
                                 -m["duration"]))


def render_collection(files: list[dict], clusters: dict | None) -> None:
    prepare_assets()
    env = _env()
    total = sum(f["duration"] for f in files)
    by_slug = {f["slug"]: f for f in files}
    row_by_key = {r["key"]: r for f in files for r in f["rows"]}

    top = []
    for f in files:
        for r in f["rows"]:
            if r["grade"] in ("A", "B") and r["category"].startswith("music"):
                top.append({**r, "slug": f["slug"], "name": f["name"]})
    top.sort(key=lambda r: ("AB".index(r["grade"]), -r["end"] + r["start"]))

    cl_out = []
    for c in (clusters or {}).get("clusters", []):
        ranked = _rank_members(c)
        best = ranked[0]["key"] if ranked else None
        members = []
        for m in ranked:
            row = row_by_key.get(m["key"])
            members.append({
                "slug": m["slug"], "name": by_slug.get(m["slug"], {}).get("name", m["slug"]),
                "anchor": m["segment_id"],
                "start_hms": hms(m["start"]), "end_hms": hms(m["end"]),
                "dur_h": dur_h(m["duration"]),
                "completeness_ru": COMPLETENESS_RU.get(m.get("completeness"), "—"),
                "has_vocal": (row or {}).get("attrs", "").find("вокал слышен") >= 0,
                "grade": (row or {}).get("grade"),
                "headline_tail": (row or {}).get("headline_tail", ""),
                "is_best": m["key"] == best,
            })
        cl_out.append({
            "id": c["id"], "n_members": c["n_members"], "title": None, "members": members,
            "links": [{"pair": f"{l['a'].split(':')[-1]} ~ {l['b'].split(':')[-1]}",
                       "score": l["score"], "match_s": l["match_s"],
                       "semitones": f"{l['semitones']:+d}", "tempo_ratio": l["tempo_ratio"],
                       "same_recording": l["same_recording"],
                       "fingerprint_ber": l["fingerprint_ber"]} for l in c["links"]],
            "best_why": "по убыванию реставрируемости, затем по законченности, затем по длине",
        })

    (REPORT / "index.html").write_text(env.get_template("index.html").render(
        title="Архив кассетных записей", generated=time.strftime("%Y-%m-%d %H:%M"),
        files=files, total_hms=hms(total), n_clusters=len(cl_out),
        totals={"music": sum(f["n_complete"] + f["n_partial"] for f in files),
                "complete": sum(f["n_complete"] for f in files)},
        top=top[:40]))
    # Одиночные исполнения — полноправные вещи каталога, а не остаток от
    # кластеризации: цельная композиция, сыгранная один раз, часто и есть
    # самое ценное на ленте.
    in_cluster = {m["key"] for c in (clusters or {}).get("clusters", []) for m in c["members"]}
    standalone = []
    for f in files:
        for row in f["rows"]:
            if row["key"] in in_cluster or not row["category"].startswith("music"):
                continue
            standalone.append({
                "slug": f["slug"], "name": f["name"], "anchor": row["anchor"],
                "start_hms": row["start_hms"], "end_hms": row["end_hms"],
                "dur_h": dur_h(row["end"] - row["start"]),
                "completeness_ru": next((v for k, v in COMPLETENESS_RU.items()
                                         if k and v in row["attrs"]), "—"),
                "has_vocal": "вокал слышен" in row["attrs"],
                "grade": row["grade"], "headline_tail": row["headline_tail"],
                "_sort": ("ABCD".index(row["grade"]), -(row["end"] - row["start"])),
            })
    standalone.sort(key=lambda m: m["_sort"])

    # Пары, не дотянувшие до порога, выводим отдельно: машина тут честно не
    # знает ответа, а человек решает на слух за десять секунд.
    near = []
    for rj in (clusters or {}).get("rejected_pairs", [])[:25]:
        a_slug, a_id = rj["a"].split(":", 1)
        b_slug, b_id = rj["b"].split(":", 1)
        ra, rb = row_by_key.get(rj["a"]), row_by_key.get(rj["b"])
        if not ra or not rb:
            continue
        near.append({
            "a_slug": a_slug, "a_anchor": a_id, "a_time": f"{ra['start_hms']}–{ra['end_hms']}",
            "b_slug": b_slug, "b_anchor": b_id, "b_time": f"{rb['start_hms']}–{rb['end_hms']}",
            "match_s": rj["match_s"], "run_s": rj["run_s"],
            "mean_similarity": rj["mean_similarity"], "why": rj["why"],
        })

    (REPORT / "songs.html").write_text(env.get_template("songs.html").render(
        title="Песни и дубли", generated=time.strftime("%Y-%m-%d %H:%M"),
        clusters=cl_out, near=near, standalone=standalone,
        n_singletons=len((clusters or {}).get("singletons", []))))
    log.info("report: index.html и songs.html готовы (%d лент, %d кластеров)", len(files), len(cl_out))
