#!/usr/bin/env bash
# Проверка read-only доступа к ОДНОЙ папке Google Drive и выкачивание её на VPS.
#
# Скрипт ничего не создаёт и никаких учётных данных не придумывает: он только
# проверяет уже настроенный remote и копирует файлы. Настройка remote — шаг
# пользователя, см. README и вывод `--help`.
set -euo pipefail

REMOTE="${REMOTE:-gdrive-tapes:}"
FOLDER_ID="${FOLDER_ID:?укажи FOLDER_ID — id папки Google Drive}"
DEST="${DEST:-$HOME/audio-archive/originals}"

usage() {
  cat <<'TXT'
Использование: tools/setup_gdrive.sh [check|list|pull]

  check  — проверить, что remote настроен и папка видна (ничего не качает)
  list   — показать, что лежит в папке, с размерами
  pull   — выкачать mp3 в архив и прогнать ingest

Переменные: REMOTE (по умолчанию gdrive-tapes:), FOLDER_ID (обязательна), DEST.

FOLDER_ID — идентификатор папки из адреса в браузере:
https://drive.google.com/drive/folders/<FOLDER_ID>

Remote должен быть настроен ТОЛЬКО на чтение. Два способа — в README.
Проверить, что доступ read-only: `rclone config show` — у remote должно быть
либо scope = drive.readonly, либо service_account_file, и обязательно
root_folder_id именно этой папки.
TXT
}

check() {
  command -v rclone >/dev/null || { echo "rclone не установлен: sudo apt install rclone"; exit 2; }
  if ! rclone listremotes | grep -qx "$REMOTE"; then
    echo "remote '$REMOTE' не настроен. Настрой его (см. README), потом запусти снова."
    echo "Настроенные сейчас: $(rclone listremotes | tr '\n' ' ')"
    exit 2
  fi
  echo "== конфигурация remote (секреты rclone не печатает) =="
  rclone config show "${REMOTE%:}" | grep -vE 'token|client_secret' || true
  echo "== доступ к папке =="
  rclone lsd "$REMOTE" --drive-root-folder-id "$FOLDER_ID" >/dev/null 2>&1 || true
  rclone size "$REMOTE" --drive-root-folder-id "$FOLDER_ID"
}

list() {
  rclone lsl "$REMOTE" --drive-root-folder-id "$FOLDER_ID" | sort -k1 -n -r
}

pull() {
  mkdir -p "$DEST"
  rclone copy "$REMOTE" "$DEST" \
    --drive-root-folder-id "$FOLDER_ID" \
    --include "*.mp3" --include "*.MP3" --include "*.wav" --include "*.flac" \
    --transfers 3 --checkers 6 --progress
  "$(dirname "${BASH_SOURCE[0]}")/../bin/tapearch" ingest
}

case "${1:-check}" in
  check) check ;;
  list)  list ;;
  pull)  pull ;;
  *)     usage ;;
esac
