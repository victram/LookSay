"""Letter keyboard layouts per language.

"qwerty" keeps a familiar ordering; "frequency" puts the most frequent letters
of the language in the top row, closest to the suggestion bar, which reduces
gaze travel. The symbols page holds accented letters, digits and punctuation.
"""

from __future__ import annotations

_QWERTY = {
    "en": ["qwertyuiop", "asdfghjkl'", "zxcvbnm,.?"],
    "es": ["qwertyuiop", "asdfghjklñ", "zxcvbnm,.?"],
    "ca": ["qwertyuiop", "asdfghjklç", "zxcvbnm',.?"],
}

# Letters ordered by approximate frequency in running text.
_FREQUENCY_ORDER = {
    "en": "etaoinshrdlcumwfgypbvkjxqz",
    "es": "eaosrnidlctumpbgvqyhfzjñxkw",
    "ca": "easirltnocudmpgvbqfhjxçzykw",
}

_EXTRA_PUNCT = {"en": "',.?", "es": ",.?", "ca": "',.?"}

_SYMBOLS = {
    "en": ["1234567890", "!?¿¡-:;\"()", "@#&%€/+=*_"],
    "es": ["áéíóúü¿¡", "1234567890", "!?-:;\"()@€"],
    "ca": ["àèéíïòóúü·", "1234567890", "!?-:;\"()@€"],
}


def letter_rows(lang: str, layout: str = "qwerty") -> list[list[str]]:
    lang = lang if lang in _QWERTY else "en"
    if layout == "frequency":
        letters = _FREQUENCY_ORDER[lang] + _EXTRA_PUNCT[lang]
        rows, size = [], 10
        for i in range(0, len(letters), size):
            rows.append(list(letters[i:i + size]))
        if len(rows) > 3:  # fold any overflow into the last row
            rows[2].extend(k for row in rows[3:] for k in row)
            rows = rows[:3]
        return rows
    return [list(row) for row in _QWERTY[lang]]


def symbol_rows(lang: str) -> list[list[str]]:
    return [list(row) for row in _SYMBOLS.get(lang, _SYMBOLS["en"])]
