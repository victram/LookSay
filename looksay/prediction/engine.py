"""Deterministic on-device word prediction.

Candidates come from a sorted prefix index (accent-insensitive) over the
bundled baseline dictionary plus the user's personal vocabulary. Each
candidate gets a weighted score:

    score = W_PREFIX * prefix_match
          + W_CONTEXT * P(word | previous word)
          + W_PERSONAL * personal_frequency   (bounded, log-scaled)
          + W_RECENCY * recency               (exponential decay)
          + W_BASE * baseline_frequency

Ties break alphabetically, so the same input always yields the same output.
No network, no neural models; a language loads in a few milliseconds.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import time
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

W_PREFIX = 1.0
W_CONTEXT = 2.0
W_PERSONAL = 1.0
W_RECENCY = 0.5
W_BASE = 1.0

PERSONAL_SATURATION = 50  # uses after which personal frequency stops growing
RECENCY_HALF_LIFE_S = 3 * 24 * 3600
MAX_PREFIX_SCAN = 6000

WORD_RE = re.compile(r"^[^\W\d_]+(?:['’·][^\W\d_]+)*$", re.UNICODE)
TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’·][^\W\d_]+)*", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase and strip diacritics so 'cafe' matches 'café' and 'nino' matches 'niño'."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("’", "'")


def tokenize(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOKEN_RE.findall(text)]


def is_word(token: str) -> bool:
    return bool(WORD_RE.match(token)) and len(token) <= 30


class LearningStore(Protocol):
    def record_word(self, lang: str, word: str, prev: str | None) -> None: ...
    def personal_vocabulary(self, lang: str) -> dict[str, tuple[int, float]]: ...
    def personal_bigrams(self, lang: str) -> dict[tuple[str, str], tuple[int, float]]: ...


@dataclass
class Baseline:
    """Read-only per-language resources."""

    frequency: dict[str, float]
    bigrams: dict[str, dict[str, int]]
    top_words: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path, lang: str) -> "Baseline":
        frequency: dict[str, float] = {}
        # Large corpus-derived list (word<TAB>count), scaled so the top word = 1e6.
        freq_path = directory / f"{lang}_frequency.txt"
        if freq_path.exists():
            rows = []
            for line in freq_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or "\t" not in line:
                    continue
                word, count = line.split("\t", 1)
                if is_word(word) and count.strip().isdigit():
                    rows.append((word, int(count)))
            top = max((c for _, c in rows), default=1)
            frequency = {w: 1_000_000 * c / top for w, c in rows}
        # Small curated AAC list (rank order): guarantees everyday care
        # vocabulary ranks well even if rare in the corpus.
        curated: list[str] = []
        for line in (directory / f"{lang}_words.txt").read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word and not word.startswith("#") and is_word(word) and word not in curated:
                curated.append(word)
        for rank, word in enumerate(curated):
            frequency[word] = max(frequency.get(word, 0.0), 1_000_000 / (rank + 1))
        bigrams: dict[str, dict[str, int]] = defaultdict(dict)
        corpus = directory / f"{lang}_corpus.txt"
        if corpus.exists():
            for line in corpus.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                tokens = tokenize(line)
                for prev, word in zip(tokens, tokens[1:]):
                    bigrams[prev][word] = bigrams[prev].get(word, 0) + 1
                for token in tokens:
                    frequency.setdefault(token, 1_000_000 / (len(curated) + 1))
        top_words = [w for w, _ in sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
        return cls(frequency=frequency, bigrams=dict(bigrams), top_words=top_words)


@dataclass
class _State:
    lang: str
    baseline: Baseline
    personal: dict[str, tuple[int, float]]
    personal_bigrams: dict[str, dict[str, tuple[int, float]]]
    index: list[tuple[str, str]] = field(default_factory=list)  # (normalized, word), sorted


class Predictor:
    def __init__(self, resources: Path, store: LearningStore | None = None, clock=time.time):
        self._resources = resources
        self._store = store
        self._clock = clock
        self._baselines: dict[str, Baseline] = {}
        self._state: _State | None = None
        self.session_recent: deque[str] = deque(maxlen=50)
        self.learning_enabled = True
        self._phrases: dict[str, list[tuple[str, bool]]] | None = None

    # ------------------------------------------------------------ languages
    @property
    def language(self) -> str:
        return self._state.lang if self._state else "en"

    def set_language(self, lang: str) -> None:
        if lang not in self._baselines:
            self._baselines[lang] = Baseline.load(self._resources, lang)
        baseline = self._baselines[lang]
        personal = self._store.personal_vocabulary(lang) if self._store else {}
        pb: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
        if self._store:
            for (prev, word), value in self._store.personal_bigrams(lang).items():
                pb[prev][word] = value
        state = _State(lang, baseline, personal, dict(pb))
        words = set(baseline.frequency) | set(personal)
        state.index = sorted((normalize(w), w) for w in words)
        self._state = state

    def default_phrases(self, lang: str) -> list[tuple[str, bool]]:
        if self._phrases is None:
            raw = json.loads((self._resources / "phrases.json").read_text(encoding="utf-8"))
            self._phrases = {k: [(t, bool(e)) for t, e in v] for k, v in raw.items()}
        return self._phrases.get(lang, [])

    # ----------------------------------------------------------- prediction
    def _prefix_candidates(self, prefix: str) -> list[str]:
        assert self._state is not None
        key = normalize(prefix)
        index = self._state.index
        start = bisect.bisect_left(index, (key, ""))
        out = []
        for norm, word in index[start:start + MAX_PREFIX_SCAN]:
            if not norm.startswith(key):
                break
            out.append(word)
        return out

    def _context_candidates(self, prev: str | None) -> set[str]:
        assert self._state is not None
        cands: set[str] = set()
        if prev:
            cands |= set(self._state.baseline.bigrams.get(prev, {}))
            cands |= set(self._state.personal_bigrams.get(prev, {}))
        cands |= set(self._state.baseline.top_words)
        top_personal = sorted(self._state.personal.items(), key=lambda kv: (-kv[1][0], kv[0]))[:40]
        cands |= {w for w, _ in top_personal}
        cands |= set(self.session_recent)
        return cands

    def _context_prob(self, prev: str | None, word: str) -> float:
        assert self._state is not None
        if not prev:
            return 0.0
        base = self._state.baseline.bigrams.get(prev, {})
        pers = self._state.personal_bigrams.get(prev, {})
        # Personal bigrams count 3x so the model adapts to the user, but the
        # per-pair cap in storage keeps that adaptation bounded.
        total = sum(base.values()) + 3 * sum(c for c, _ in pers.values())
        if total == 0:
            return 0.0
        count = base.get(word, 0) + 3 * pers.get(word, (0, 0.0))[0]
        return count / (total + 1)

    def score(self, word: str, prefix: str, prev: str | None, prefix_checked: bool = False) -> float:
        assert self._state is not None
        state = self._state
        now = self._clock()
        if prefix:
            if not prefix_checked and not normalize(word).startswith(normalize(prefix)):
                return -math.inf
            prefix_match = 1.0 if word.startswith(prefix.lower()) else 0.9
        else:
            prefix_match = 0.0
        base_f = state.baseline.frequency.get(word, 0.0)
        base = math.log1p(base_f) / math.log1p(1_000_000)
        count, last_used = state.personal.get(word, (0, 0.0))
        personal = min(1.0, math.log1p(count) / math.log1p(PERSONAL_SATURATION))
        recency = 0.0
        if last_used:
            recency = 0.5 ** (max(0.0, now - last_used) / RECENCY_HALF_LIFE_S)
        if word in self.session_recent:
            recency = max(recency, 1.0)
        return (W_PREFIX * prefix_match + W_CONTEXT * self._context_prob(prev, word)
                + W_PERSONAL * personal + W_RECENCY * recency + W_BASE * base)

    def suggest(self, prefix: str, prev: str | None = None, n: int = 5) -> list[str]:
        """Return up to n completions for `prefix` (or next-word predictions if empty)."""
        if self._state is None:
            self.set_language("en")
        prefix = prefix.lower().replace("’", "'")
        prev = prev.lower() if prev else None
        candidates = self._prefix_candidates(prefix) if prefix else self._context_candidates(prev)
        scored = []
        for word in candidates:
            if prefix and word == prefix:
                continue  # already fully typed; offering it again wastes a slot
            s = self.score(word, prefix, prev, prefix_checked=True)
            if s > -math.inf:
                scored.append((-s, word))
        scored.sort()
        return [self.display_form(w) for _, w in scored[:n]]

    def display_form(self, word: str) -> str:
        if self.language == "en" and (word == "i" or word.startswith("i'")):
            return "I" + word[1:]
        return word

    # ------------------------------------------------------------- learning
    def learn(self, word: str, prev: str | None = None) -> None:
        if self._state is None:
            return
        word = word.lower().replace("’", "'")
        prev = prev.lower().replace("’", "'") if prev else None
        if not is_word(word):
            return
        self.session_recent.append(word)
        if not self.learning_enabled:
            return
        state = self._state
        now = self._clock()
        count, _ = state.personal.get(word, (0, 0.0))
        if word not in state.personal and word not in state.baseline.frequency:
            bisect.insort(state.index, (normalize(word), word))
        state.personal[word] = (min(count + 1, 500), now)
        if prev and is_word(prev):
            row = state.personal_bigrams.setdefault(prev, {})
            c, _ = row.get(word, (0, 0.0))
            row[word] = (min(c + 1, 500), now)
        if self._store:
            self._store.record_word(state.lang, word, prev if prev and is_word(prev) else None)
