"""Идемпотентность стадий.

На каждую стадию рядом с её результатом лежит <stage>.done.json с хэшем входа
(sha256 исходника + версия параметров + хэши результатов стадий-зависимостей).
Совпал — стадия пропускается. Это и даёт resumable-прогон без лишней работы.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .config import STAGE_DEPS, STAGE_PARAMS_VERSION


def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def marker_path(wdir: Path, stage: str) -> Path:
    return wdir / f"{stage}.done.json"


def input_key(wdir: Path, stage: str, source_sha: str, extra: dict | None = None) -> str:
    deps = {}
    for dep in STAGE_DEPS.get(stage, []):
        m = marker_path(wdir, dep)
        deps[dep] = json.loads(m.read_text())["key"] if m.exists() else "missing"
    return _digest({
        "stage": stage,
        "params_version": STAGE_PARAMS_VERSION[stage],
        "source": source_sha,
        "deps": deps,
        "extra": extra or {},
    })


def is_fresh(wdir: Path, stage: str, key: str, outputs: list[Path]) -> bool:
    m = marker_path(wdir, stage)
    if not m.exists():
        return False
    try:
        rec = json.loads(m.read_text())
    except (ValueError, OSError):
        return False
    if rec.get("key") != key:
        return False
    return all(p.exists() for p in outputs)


def commit(wdir: Path, stage: str, key: str, outputs: list[Path], meta: dict | None = None) -> None:
    marker_path(wdir, stage).write_text(json.dumps({
        "stage": stage, "key": key,
        "outputs": [str(p.relative_to(wdir)) if p.is_relative_to(wdir) else str(p) for p in outputs],
        "meta": meta or {},
    }, ensure_ascii=False, indent=2))


def invalidate(wdir: Path, stage: str) -> None:
    """Снести маркер стадии и всех, кто от неё зависит (транзитивно)."""
    todo = [stage]
    seen = set()
    while todo:
        s = todo.pop()
        if s in seen:
            continue
        seen.add(s)
        marker_path(wdir, s).unlink(missing_ok=True)
        todo += [d for d, deps in STAGE_DEPS.items() if s in deps]
