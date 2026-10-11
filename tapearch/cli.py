"""Командная строка pipeline.

Прогон идемпотентный и возобновляемый: рядом с результатом каждой стадии лежит
маркер с хэшем её входа (sha256 оригинала + версия параметров стадии + маркеры
стадий, от которых она зависит). Совпал — стадия пропускается. Поменял порог в
config и бампнул версию стадии — пересчитается только она и то, что от неё
зависит. Оригиналы кладутся в режим только-чтение и больше не трогаются.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import cache, paths, runlog
from .config import (ANALYSIS_SR, ASR_MODEL, COLLECTION_STAGES, DATA_ROOT, STAGE_ORDER,
                     TAGGER_SR)
from .runlog import event, log, stage


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _manifest() -> dict:
    return _load(paths.MANIFEST) if paths.MANIFEST.exists() else {"files": {}}


def _save_manifest(m: dict) -> None:
    paths.MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2))


def cmd_ingest(args) -> int:
    """Положить оригиналы в архив: скопировать, снять право записи, посчитать sha256."""
    paths.ensure_layout()
    m = _manifest()
    srcs: list[Path] = []
    if args.rclone_remote:
        dst = paths.ORIGINALS
        cmd = ["rclone", "copy", "--progress", "--include", "*.mp3",
               args.rclone_remote, str(dst)]
        log.info("ingest: %s", " ".join(cmd))
        if os.system(" ".join(f"'{c}'" if " " in c else c for c in cmd)) != 0:
            log.error("ingest: rclone вернул ошибку")
            return 2
    for p in args.paths or []:
        q = Path(p)
        srcs += sorted(q.glob("*.mp3")) if q.is_dir() else [q]
    for s in srcs:
        dst = paths.ORIGINALS / s.name
        if not dst.exists():
            shutil.copy2(s, dst)
            log.info("ingest: скопирован %s", s.name)
    n_new = 0
    for p in sorted(paths.ORIGINALS.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
            continue
        slug = paths.slugify(p.name)
        rec = m["files"].get(slug)
        if rec and rec.get("bytes") == p.stat().st_size and rec.get("sha256"):
            continue
        sha = paths.sha256_file(p)
        m["files"][slug] = {"name": p.name, "bytes": p.stat().st_size, "sha256": sha,
                            "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        n_new += 1
        log.info("ingest: %s -> %s (%.1f МБ, sha %s…)", p.name, slug,
                 p.stat().st_size / 1e6, sha[:12])
    for p in paths.ORIGINALS.iterdir():
        if p.is_file():
            p.chmod(0o444)                 # оригиналы только на чтение, и это навсегда
    _save_manifest(m)
    log.info("ingest: в архиве %d файлов, новых %d; оригиналы переведены в режим 0444",
             len(m["files"]), n_new)
    return 0


def _file_stages(slug: str, rec: dict, args, asr_holder: dict) -> dict:
    """Стадии, считающиеся по одному файлу. Возвращает загруженные результаты."""
    from . import (s1_probe, s2_features, s3_segment, s4_classify, s5_transcribe,
                   s6_music, tagger)
    from . import audio as au

    src = paths.ORIGINALS / rec["name"]
    wdir = paths.work_dir(slug)
    sha = rec["sha256"]
    res: dict = {}

    key = cache.input_key(wdir, "probe", sha)
    outs = [wdir / "probe.json", wdir / f"analysis_{ANALYSIS_SR}.wav"]
    if cache.is_fresh(wdir, "probe", key, outs) and not args.force:
        log.info("[%s] probe: уже посчитано, пропускаю", slug)
    else:
        with stage("probe", slug):
            s1_probe.run(src, wdir, sha)
        cache.commit(wdir, "probe", key, outs)
    res["probe"] = _load(wdir / "probe.json")

    key = cache.input_key(wdir, "features", sha)
    outs = [wdir / "features.npz", wdir / "features.meta.json"]
    if cache.is_fresh(wdir, "features", key, outs) and not args.force:
        log.info("[%s] features: уже посчитано, пропускаю", slug)
    else:
        with stage("features", slug):
            s2_features.run(wdir, res["probe"])
        cache.commit(wdir, "features", key, outs)
    res["fmeta"] = _load(wdir / "features.meta.json")
    res["feats"] = s2_features.load(wdir)

    tags: dict = {}
    if not args.no_tagger:
        key = cache.input_key(wdir, "tags", sha, {"model": "panns-cnn14"})
        outs = [wdir / "tags.npz"]
        if cache.is_fresh(wdir, "tags", key, outs) and not args.force:
            log.info("[%s] tags: уже посчитано, пропускаю", slug)
        else:
            with stage("tags", slug):
                tg = asr_holder.get("tagger")
                if tg is None:
                    tg = tagger.Tagger()
                    asr_holder["tagger"] = tg
                if tg.ok:
                    y32 = au.decode_window(src, 0, res["fmeta"]["duration_s"] + 1, TAGGER_SR, 1)
                    np.savez_compressed(wdir / "tags.npz", **tg.tag(y32))
                else:
                    np.savez_compressed(wdir / "tags.npz", unavailable=np.zeros(1))
            cache.commit(wdir, "tags", key, outs)
        with np.load(wdir / "tags.npz") as z:
            tags = {k: z[k] for k in z.files}
        if "unavailable" in tags:
            tags = {}
    res["tags"] = tags

    key = cache.input_key(wdir, "segment", sha)
    outs = [wdir / "segments.json"]
    if cache.is_fresh(wdir, "segment", key, outs) and not args.force:
        log.info("[%s] segment: уже посчитано, пропускаю", slug)
    else:
        with stage("segment", slug):
            s3_segment.run(wdir, res["feats"], res["fmeta"], tags)
        cache.commit(wdir, "segment", key, outs)
    res["segments"] = _load(wdir / "segments.json")

    key = cache.input_key(wdir, "classify", sha)
    outs = [wdir / "classify.json"]
    if cache.is_fresh(wdir, "classify", key, outs) and not args.force:
        log.info("[%s] classify: уже посчитано, пропускаю", slug)
    else:
        with stage("classify", slug):
            s4_classify.run(wdir, res["segments"], res["feats"], res["fmeta"], tags)
        cache.commit(wdir, "classify", key, outs)
    res["classify"] = _load(wdir / "classify.json")

    key = cache.input_key(wdir, "music", sha)
    outs = [wdir / "music.json"]
    if cache.is_fresh(wdir, "music", key, outs) and not args.force:
        log.info("[%s] music: уже посчитано, пропускаю", slug)
    else:
        with stage("music", slug):
            s6_music.run(wdir, src, res["segments"], res["classify"],
                         res["feats"], res["fmeta"], res["probe"])
        cache.commit(wdir, "music", key, outs)
    res["music"] = _load(wdir / "music.json")

    key = cache.input_key(wdir, "transcribe", sha, {"model": args.asr_model, "off": args.no_asr})
    outs = [wdir / "transcript.json"]
    if cache.is_fresh(wdir, "transcribe", key, outs) and not args.force:
        log.info("[%s] transcribe: уже посчитано, пропускаю", slug)
    elif args.no_asr:
        (wdir / "transcript.json").write_text(json.dumps(
            {"model": None, "segments": [], "n_usable": 0, "skipped": "--no-asr"},
            ensure_ascii=False, indent=2))
        cache.commit(wdir, "transcribe", key, outs)
        log.info("[%s] transcribe: выключено флагом --no-asr", slug)
    else:
        with stage("transcribe", slug):
            asr = asr_holder.get("asr")
            if asr is None:
                asr = s5_transcribe.Asr(args.asr_model, threads=args.threads)
                asr_holder["asr"] = asr
            s5_transcribe.run(wdir, src, res["segments"], res["classify"], asr, res["music"])
        cache.commit(wdir, "transcribe", key, outs)
    res["transcript"] = _load(wdir / "transcript.json")
    return res


def cmd_analyze(args) -> int:
    from . import s7_cluster, s8_rank, s9_report
    from . import audio as au

    paths.ensure_layout()
    m = _manifest()
    if not m["files"]:
        log.error("в архиве нет файлов — сначала `tapearch ingest`")
        return 2
    slugs = [s for s in sorted(m["files"]) if not args.only or s in args.only]
    if not slugs:
        log.error("ни один файл не подошёл под --only %s", args.only)
        return 2
    if args.from_stage:
        for s in slugs:
            cache.invalidate(paths.work_dir(s), args.from_stage)
        log.info("сброшены маркеры со стадии %s и дальше", args.from_stage)

    holder: dict = {}
    per_file: dict[str, dict] = {}
    t_all = time.perf_counter()
    for slug in slugs:
        log.info("=== %s (%s) ===", slug, m["files"][slug]["name"])
        per_file[slug] = _file_stages(slug, m["files"][slug], args, holder)

    # ---- уровень коллекции: одна и та же песня в разных местах архива
    cdir = paths.COLLECTION
    cdir.mkdir(parents=True, exist_ok=True)
    items = []
    for slug, r in per_file.items():
        src = paths.ORIGINALS / m["files"][slug]["name"]
        for mu in r["music"]["segments"]:
            if not mu["category"].startswith("music"):
                continue
            items.append({
                "key": f"{slug}:{mu['id']}", "slug": slug, "src": src,
                "segment_id": mu["id"], "feats": r["feats"],
                "start": mu["start"], "end": mu["end"], "duration": mu["duration"],
                "meta": mu,
                "fingerprint": au.chromaprint(src, mu["start"], mu["duration"]),
            })
    # В clusters.json копируются поля из music.json (законченность, реставрируемость),
    # поэтому ключ кэша обязан зависеть и от маркера music: иначе после правки
    # музыкальной стадии в отчёте остаются устаревшие значения рядом со свежими.
    music_keys = {}
    for slug in slugs:
        mk = cache.marker_path(paths.work_dir(slug), "music")
        music_keys[slug] = json.loads(mk.read_text())["key"] if mk.exists() else "missing"
    ckey = cache._digest({"stage": "cluster", "music": music_keys, "items": sorted(
        (i["key"], round(i["start"], 2), round(i["duration"], 2)) for i in items)})
    if cache.is_fresh(cdir, "cluster", ckey, [cdir / "clusters.json"]) and not args.force:
        log.info("cluster: уже посчитано, пропускаю")
    else:
        with stage("cluster", "коллекция", n_segments=len(items)):
            s7_cluster.run(items, cdir)
        cache.commit(cdir, "cluster", ckey, [cdir / "clusters.json"])
    clusters = _load(cdir / "clusters.json")

    # ---- обратно по файлам: рейтинг и отчёт
    summaries = []
    for slug in slugs:
        r = per_file[slug]
        wdir = paths.work_dir(slug)
        sha = m["files"][slug]["sha256"]
        key = cache.input_key(wdir, "rank", sha, {"clusters": ckey})
        if cache.is_fresh(wdir, "rank", key, [wdir / "rank.json"]) and not args.force:
            log.info("[%s] rank: уже посчитано, пропускаю", slug)
        else:
            with stage("rank", slug):
                s8_rank.run(wdir, r["segments"], r["classify"], r["music"],
                            r["transcript"], clusters, slug)
            cache.commit(wdir, "rank", key, [wdir / "rank.json"])
        r["rank"] = _load(wdir / "rank.json")

        with stage("report", slug):
            summaries.append(s9_report.render_file(
                slug, paths.ORIGINALS / m["files"][slug]["name"], wdir, r["probe"],
                r["segments"], r["classify"], r["music"], r["transcript"], r["rank"],
                clusters, wdir / f"analysis_{ANALYSIS_SR}.wav"))

    with stage("report", "коллекция"):
        s9_report.render_collection(summaries, clusters)
    (paths.COLLECTION / "index.json").write_text(json.dumps(
        {"files": [{k: v for k, v in s.items() if k != "rows"} for s in summaries],
         "n_clusters": len(clusters.get("clusters", []))}, ensure_ascii=False, indent=2))

    dt = time.perf_counter() - t_all
    total_audio = sum(s["duration"] for s in summaries)
    log.info("ГОТОВО: %d лент, %.1f мин звука, обработка %.1f мин (%.2f× от длины звука)",
             len(summaries), total_audio / 60, dt / 60, dt / max(total_audio, 1))
    event("run_done", files=len(summaries), audio_s=total_audio, wall_s=dt)
    log.info("отчёт: %s/index.html", paths.REPORT)
    return 0


def cmd_extract(args) -> int:
    """Вырезать сегменты в отдельные файлы — чтобы послушать или отдать дальше.

    Режется потоковым копированием, без перекодирования: звук тот же самый,
    что в оригинале, просто короче. Оригинал, разумеется, не трогается.
    """
    import subprocess
    m = _manifest()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = 0
    for slug, rec in sorted(m["files"].items()):
        if args.only and slug not in args.only:
            continue
        wdir = paths.WORK / slug
        if not (wdir / "rank.json").exists():
            continue
        segs = {s["id"]: s for s in _load(wdir / "segments.json")["segments"]}
        cls = {c["id"]: c for c in _load(wdir / "classify.json")["segments"]}
        mus = {x["id"]: x for x in _load(wdir / "music.json")["segments"]}
        src = paths.ORIGINALS / rec["name"]
        for r in _load(wdir / "rank.json")["segments"]:
            seg = segs[r["id"]]
            cat = (mus.get(r["id"]) or {}).get("category") or cls[r["id"]]["category"]
            if args.ids and r["id"] not in args.ids:
                continue
            if not args.ids:
                if r["grade"] not in args.grades:
                    continue
                if not cat.startswith("music"):
                    continue
            name = (f"{slug}__{r['id']}__{int(seg['start'])//60:02d}m{int(seg['start'])%60:02d}s"
                    f"-{int(seg['end'])//60:02d}m{int(seg['end'])%60:02d}s__{r['grade']}.mp3")
            dst = out_dir / name
            cp = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{seg['start']:.3f}",
                 "-to", f"{seg['end']:.3f}", "-i", str(src), "-c:a", "copy", str(dst)],
                capture_output=True, text=True, check=False)
            if cp.returncode != 0:
                log.error("не смог вырезать %s: %s", name, cp.stderr.strip()[:160])
                continue
            picked += 1
            log.info("%s  (%.0f с, %s) -> %s", r["grade"], seg["duration"],
                     r["headline"][:60], name)
    log.info("вырезано %d кусков в %s", picked, out_dir)
    return 0


def cmd_status(args) -> int:
    m = _manifest()
    if not m["files"]:
        print("архив пуст")
        return 0
    print(f"{'лента':44s} {'длина':>8s}  стадии")
    for slug, rec in sorted(m["files"].items()):
        wdir = paths.WORK / slug
        done = [s for s in STAGE_ORDER if s not in COLLECTION_STAGES
                and cache.marker_path(wdir, s).exists()]
        dur = "?"
        if (wdir / "probe.json").exists():
            dur = f"{_load(wdir / 'probe.json')['format']['duration_s']/60:.1f} мин"
        print(f"{rec['name'][:44]:44s} {dur:>8s}  {', '.join(done) or '—'}")
    cdir = paths.COLLECTION
    if (cdir / "clusters.json").exists():
        c = _load(cdir / "clusters.json")
        print(f"\nкластеры песен: {len(c.get('clusters', []))}, "
              f"одиночек {len(c.get('singletons', []))}")
    return 0


class _RangeHandler:
    """Отдача файлов с поддержкой Range.

    Без неё браузер не умеет перематывать <audio>: он тянет файл целиком и
    только потом даёт двигать ползунок. На получасовой ленте это 20+ МБ на
    каждый клик по сегменту, а весь смысл отчёта — быстро прыгать по кускам.
    SimpleHTTPRequestHandler Range не умеет, поэтому добавляем сами.
    """

    def __init__(self, directory: str):
        self.directory = directory

    def build(self):
        import http.server
        import os
        directory = self.directory

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=directory, **kw)

            def log_message(self, fmt, *args):      # тише в консоли
                pass

            def send_head(self):
                rng = self.headers.get("Range")
                if not rng or not rng.startswith("bytes="):
                    return super().send_head()
                path = self.translate_path(self.path)
                if os.path.isdir(path) or not os.path.exists(path):
                    return super().send_head()
                size = os.path.getsize(path)
                try:
                    first, _, last = rng[6:].partition("-")
                    start = int(first) if first else 0
                    end = int(last) if last else size - 1
                except ValueError:
                    return super().send_head()
                start = max(0, start)
                end = min(end, size - 1)
                if start > end:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return None
                f = open(path, "rb")
                f.seek(start)
                self.send_response(206)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self._range_remaining = end - start + 1
                return f

            def copyfile(self, source, outputfile):
                remaining = getattr(self, "_range_remaining", None)
                if remaining is None:
                    return super().copyfile(source, outputfile)
                while remaining > 0:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    outputfile.write(chunk)
                    remaining -= len(chunk)
                self._range_remaining = None

            def end_headers(self):
                if "Accept-Ranges" not in self._headers_buffer_str():
                    self.send_header("Accept-Ranges", "bytes")
                super().end_headers()

            def _headers_buffer_str(self):
                return b"".join(getattr(self, "_headers_buffer", [])).decode("latin-1", "ignore")

        return Handler


def cmd_serve(args) -> int:
    import socketserver
    handler = _RangeHandler(str(paths.REPORT)).build()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"отчёт на http://127.0.0.1:{args.port}/ (только localhost)")
        print(f"с другой машины:  ssh -N -L {args.port}:127.0.0.1:{args.port} <хост>")
        httpd.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tapearch", description="Анализ архива кассетных записей")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="корень данных")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="положить оригиналы в архив (только чтение) и посчитать sha256")
    p.add_argument("paths", nargs="*", help="файлы или каталоги")
    p.add_argument("--rclone-remote", help="например gdrive-tapes: (копируется read-only)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("analyze", help="прогнать стадии анализа и собрать отчёт")
    p.add_argument("--only", nargs="*", help="слаги файлов")
    p.add_argument("--from", dest="from_stage", choices=STAGE_ORDER,
                   help="сбросить маркеры с этой стадии и пересчитать её и зависимые")
    p.add_argument("--force", action="store_true", help="пересчитать всё, игнорируя кэш")
    p.add_argument("--asr-model", default=ASR_MODEL,
                   help="medium | large-v3 | large-v3-turbo (по умолчанию %(default)s)")
    p.add_argument("--no-asr", action="store_true", help="без расшифровки")
    p.add_argument("--no-tagger", action="store_true", help="без нейросетевого теггера")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("extract", help="вырезать сегменты в отдельные файлы (без перекодирования)")
    p.add_argument("--out", default=str(DATA_ROOT / "clips"))
    p.add_argument("--grades", nargs="*", default=["A"], help="какие оценки брать")
    p.add_argument("--ids", nargs="*", help="конкретные сегменты по id")
    p.add_argument("--only", nargs="*", help="слаги лент")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("status", help="что уже посчитано")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("serve", help="отдать отчёт по http на localhost")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    if args.data_root != str(DATA_ROOT):
        import tapearch.config as cfg
        cfg.DATA_ROOT = Path(args.data_root)
        paths.DATA_ROOT = Path(args.data_root)
        for name in ("ORIGINALS", "WORK", "DERIVED", "REPORT", "LOGS", "COLLECTION"):
            setattr(paths, name, Path(args.data_root) / name.lower())
        paths.MANIFEST = Path(args.data_root) / "manifest.json"
    runlog.setup()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        log.warning("прервано пользователем; посчитанные стадии сохранены, "
                    "повторный запуск продолжит с места остановки")
        return 130


if __name__ == "__main__":
    sys.exit(main())
