# Доступ к Google Drive только на чтение

Конвейер сам учётных данных не создаёт и не хранит: он работает с уже
настроенным remote rclone. Настройка — шаг владельца аккаунта.

`FOLDER_ID` — идентификатор папки из адреса в браузере:
`https://drive.google.com/drive/folders/<FOLDER_ID>`

## Способ 1 — сервисный аккаунт (точнее всех по правам)

Сервисный аккаунт видит ровно то, что ему расшарили, и больше ничего.

```bash
gcloud iam service-accounts create tape-archive-reader \
    --display-name "tape-archive read-only" --project <PROJECT>
gcloud iam service-accounts keys create ~/tape-reader.json \
    --iam-account tape-archive-reader@<PROJECT>.iam.gserviceaccount.com
# в Google Drive: папку -> Поделиться -> добавить этот e-mail как «Читатель»
scp ~/tape-reader.json <хост>:~/.config/tape-reader.json
```

На машине с конвейером (в конфиге rclone только путь к ключу, не сам ключ):

```bash
rclone config create gdrive-tapes drive \
    scope=drive.readonly \
    service_account_file=$HOME/.config/tape-reader.json \
    root_folder_id=<FOLDER_ID>
```

## Способ 2 — OAuth-токен rclone, read-only

Быстрее, но токен даёт чтение всего Drive, а не одной папки: `root_folder_id`
ограничивает только то, что видит конвейер, а не сам токен.

На машине с браузером:

```bash
rclone authorize "drive" --drive-scope=drive.readonly
```

Напечатанный блок токена вставить там, где стоит конвейер:

```bash
rclone config create gdrive-tapes drive \
    scope=drive.readonly \
    root_folder_id=<FOLDER_ID> \
    token '<блок из rclone authorize>'
```

## Проверка и выкачивание

```bash
FOLDER_ID=<...> tools/setup_gdrive.sh check   # видна ли папка
FOLDER_ID=<...> tools/setup_gdrive.sh list    # что в ней лежит
FOLDER_ID=<...> tools/setup_gdrive.sh pull    # выкачать и сделать ingest
```

Делать папку доступной «всем, у кого есть ссылка» не надо: это публикует
записи на всё время, пока ссылка жива.
