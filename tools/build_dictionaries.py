"""Build the bundled frequency dictionaries from FrequencyWords lists.

Source: https://github.com/hermitdave/FrequencyWords (2018, OpenSubtitles),
content licensed CC-BY-SA-4.0. Download {lang}_50k.txt for en/es/ca, then:

    python tools/build_dictionaries.py path/to/en_50k.txt path/to/es_50k.txt path/to/ca_50k.txt

Writes looksay/resources/dictionaries/{lang}_frequency.txt (top 20k usable
words). The hand-curated {lang}_words.txt lists (AAC vocabulary such as
"pain", "nurse") are kept separately and merged at load time.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looksay.prediction.engine import is_word  # noqa: E402

LIMIT = 20_000
SINGLE_LETTERS = {"en": {"a", "i"}, "es": {"a", "e", "o", "u", "y"}, "ca": {"a", "i", "o", "e"}}
# OpenSubtitles tokenisation splits English contractions ("wasn't" -> "wasn" + "t").
EN_CONTRACTIONS = {w: w[:-1] + "'t" for w in (
    "don", "didn", "doesn", "isn", "wasn", "weren", "haven", "hasn", "hadn", "couldn", "wouldn", "shouldn",
    "aren", "ain", "mustn", "needn")}
EN_FRAGMENTS = {"ll", "ve", "re", "s", "t", "m", "d"}
HEADER = ("# {lang} word frequencies (word<TAB>count), top {n} usable words.\n"
          "# Derived from FrequencyWords 2018 (OpenSubtitles) by Hermit Dave,\n"
          "# https://github.com/hermitdave/FrequencyWords - licensed CC-BY-SA-4.0.\n")


def build(source: Path, lang: str, out_dir: Path) -> int:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        word = parts[0].strip().lower()
        if lang == "en":
            if word in EN_FRAGMENTS:
                continue
            word = EN_CONTRACTIONS.get(word, word)
        if not is_word(word) or (len(word) == 1 and word not in SINGLE_LETTERS[lang]):
            continue
        if re.search(r"(.)\1\1", word):  # "aaah", "mmm"
            continue
        rows.append((word, int(parts[1])))
        if len(rows) >= LIMIT:
            break
    out = out_dir / f"{lang}_frequency.txt"
    out.write_text(HEADER.format(lang=lang, n=len(rows)) + "".join(f"{w}\t{c}\n" for w, c in rows), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "looksay" / "resources" / "dictionaries"
    for arg in sys.argv[1:]:
        path = Path(arg)
        lang = path.name.split("_")[0]
        print(lang, build(path, lang, out_dir))
