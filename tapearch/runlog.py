"""Лог обработки: человекочитаемый в stderr + JSONL для машин."""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import paths

_JSONL: Path | None = None
log = logging.getLogger("tapearch")


def setup(run_id: str | None = None, verbose: bool = True) -> str:
    global _JSONL
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _JSONL = paths.LOGS / f"run-{run_id}.jsonl"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))
    log.handlers = [handler]
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    event("run_start", run_id=run_id, pid=os.getpid(), argv=sys.argv[1:])
    return run_id


def event(kind: str, **fields) -> None:
    rec = {"ts": time.time(), "kind": kind, **fields}
    if _JSONL is not None:
        with _JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@contextmanager
def stage(name: str, slug: str = "-", **fields):
    """Таймер стадии: пишет длительность и в лог, и в JSONL."""
    t0 = time.perf_counter()
    log.info("[%s] %s: старт", slug, name)
    event("stage_start", stage=name, slug=slug, **fields)
    status, err = "ok", None
    try:
        yield
    except BaseException as exc:          # noqa: BLE001 - лог и прокинуть дальше
        status, err = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dt = time.perf_counter() - t0
        event("stage_end", stage=name, slug=slug, seconds=round(dt, 3),
              status=status, error=err, **fields)
        lvl = log.info if status == "ok" else log.error
        lvl("[%s] %s: %s за %.1f с%s", slug, name, status, dt,
            "" if err is None else f" ({err})")
