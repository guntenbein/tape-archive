#!/usr/bin/env bash
# Разворачивает окружение с нуля. Идемпотентен.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== системные пакеты =="
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ffmpeg sox libsox-fmt-mp3 libchromaprint-tools libsndfile1 \
  python3-venv python3-dev build-essential pkg-config

echo "== python-окружение =="
[ -d "$ROOT/.venv" ] || python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q --upgrade pip wheel setuptools
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"

echo "== проверка =="
"$ROOT/.venv/bin/python" - <<'PY'
import librosa, faster_whisper, soundfile, networkx, jinja2
print("librosa", librosa.__version__, "faster-whisper", faster_whisper.__version__)
try:
    import torch; print("torch", torch.__version__, "(теггер доступен)")
except ImportError:
    print("torch нет — pipeline пойдёт на одних эвристиках")
PY
echo "готово. дальше: ./bin/tapearch ingest <файлы>  и  ./bin/tapearch analyze"
