#!/usr/bin/env python3
"""Сверка результата pipeline с эталонной разметкой синтетической ленты.

Это регрессионный тест всего конвейера: он падает, если правка порогов или
кода ломает то, что раньше работало. Проверяются ровно те свойства, ради
которых стадии и писались, а не внутренние числа.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

EXPECT = {
    "conversation": "conversation",
    "instrument_setup": "instrument_setup",
    "applause": "applause",
    "partial_song": "music_partial",
    "complete_song": "music_complete",
    "instrumental": "music_complete",
}

fails: list[str] = []
oks: list[str] = []


def check(cond: bool, msg: str) -> None:
    (oks if cond else fails).append(msg)


def main() -> int:
    root = Path(sys.argv[1])
    truth = json.loads(Path(sys.argv[2]).read_text())["segments"]
    slug = next(iter(json.loads((root / "manifest.json").read_text())["files"]))
    w = root / "work" / slug
    segs = json.loads((w / "segments.json").read_text())["segments"]
    cls = {c["id"]: c for c in json.loads((w / "classify.json").read_text())["segments"]}
    mus = {m["id"]: m for m in json.loads((w / "music.json").read_text())["segments"]}
    tr = json.loads((w / "transcript.json").read_text())
    clusters = json.loads((root / "collection" / "clusters.json").read_text())

    def cat(s: dict) -> str:
        return (mus.get(s["id"], {}).get("category") or cls[s["id"]]["category"])

    # 1. каждый эталонный участок опознан правильно
    for t in truth:
        tl = t["end"] - t["start"]
        best, best_ov = None, 0.0
        for s in segs:
            ov = min(t["end"], s["end"]) - max(t["start"], s["start"])
            if ov > best_ov:
                best, best_ov = s, ov
        if best is None or best_ov < 0.5 * tl:
            check(False, f"участок {t['start']:.0f}-{t['end']:.0f} ({t['label']}) не покрыт сегментом")
            continue
        want = EXPECT[t["label"]]
        check(cat(best) == want,
              f"{t['start']:.0f}-{t['end']:.0f} {t['note'][:30]}: ждали {want}, вышло {cat(best)}")

    # 2. клиппинг виден там, где он есть, и не виден там, где его нет
    def clip_of(m: dict) -> float | None:
        # именно так, а не через `or`: честный ноль ложен в Python и превратился
        # бы в подставленное значение по умолчанию
        v = m["technical"].get("clip_ratio")
        return None if v is None else float(v)

    clipped = [m for m in mus.values() if 430 <= m["start"] <= 450]
    check(bool(clipped) and (clip_of(clipped[0]) or 0.0) > 0.01,
          "перегруженный кусок опознан как клиппинг >1%")
    clean = [m for m in mus.values() if 150 <= m["start"] <= 170]
    check(bool(clean) and clip_of(clean[0]) is not None and clip_of(clean[0]) < 0.001,
          "чистый дубль не помечен клиппингом")

    # 3. сетевой гул найден (он в ленту заложен на 50/100/150 Гц)
    probe = json.loads((w / "probe.json").read_text())
    hums = probe["noise"]["hum_prominence_db"]
    check((hums.get("50") or 0) > 12, f"гул 50 Гц найден ({hums.get('50')} дБ)")
    check((hums.get("200") or 0) < 12, f"ложного гула на 200 Гц нет ({hums.get('200')} дБ)")

    # 4. кластеризация: мотив X собран, мотив Y отдельно
    members: list[str] = []
    for c in clusters["clusters"]:
        members += [m["segment_id"] for m in c["members"]]
    y_seg = [s["id"] for s in segs if 354 <= s["start"] <= 380 and cat(s).startswith("music")]
    check(len(clusters["clusters"]) == 1, f"ровно один кластер (вышло {len(clusters['clusters'])})")
    check(len(members) >= 3, f"в кластер попало ≥3 исполнений мотива X (вышло {len(members)})")
    check(all(y not in members for y in y_seg),
          "мотив Y с другой прогрессией в кластер мотива X НЕ попал")

    # 5. chromaprint отличает дословную копию от другого дубля
    dup = [l for l in clusters["all_edges"] if l["same_recording"]]
    other = [l for l in clusters["all_edges"] if not l["same_recording"]]
    check(len(dup) >= 1, "дословная копия опознана как та же самая запись")
    check(all((l["fingerprint_ber"] or 1) > 0.25 for l in other),
          "разные дубли не приняты за одну запись")

    # 6. фильтр галлюцинаций: в ленте нет настоящей речи, значит пригодных расшифровок быть не должно
    check(tr["n_usable"] == 0,
          f"галлюцинации Whisper на неречевом звуке отсеяны (пригодных {tr['n_usable']}, ждали 0)")

    for m in oks:
        print(f"  ok    {m}")
    for m in fails:
        print(f"  ПЛОХО {m}")
    print(f"\n{len(oks)} проверок прошло, {len(fails)} упало")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
