VENV := .venv/bin
DATA ?= $(HOME)/audio-archive

.PHONY: setup selftest pilot analyze report serve status clean-selftest

setup:            ## развернуть окружение
	./setup.sh

selftest:         ## прогнать весь pipeline на синтетической ленте (без личных записей)
	$(VENV)/python tools/make_synthetic_tape.py $(DATA)/testdata/synthetic-tape.mp3
	rm -rf $(DATA)/selftest && mkdir -p $(DATA)/selftest
	./bin/tapearch --data-root $(DATA)/selftest ingest $(DATA)/testdata/synthetic-tape.mp3
	./bin/tapearch --data-root $(DATA)/selftest analyze
	$(VENV)/python tools/check_selftest.py $(DATA)/selftest $(DATA)/testdata/synthetic-tape.truth.json

incoming:         ## забрать всё, что положили в audio-archive/incoming, и прогнать
	@ls -1 $(DATA)/incoming/* >/dev/null 2>&1 || { echo "в $(DATA)/incoming пусто"; exit 1; }
	./bin/tapearch ingest $(DATA)/incoming
	./bin/tapearch analyze

pilot:            ## прогон одной ленты: make pilot FILE=<слаг>
	./bin/tapearch analyze --only $(FILE)

analyze:          ## прогон всего архива
	./bin/tapearch analyze

report:           ## пересобрать только отчёт
	./bin/tapearch analyze --from report

serve:            ## отдать отчёт на localhost:8765
	./bin/tapearch serve

status:
	./bin/tapearch status

clean-selftest:
	rm -rf $(DATA)/selftest
