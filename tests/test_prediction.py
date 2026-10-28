from looksay.prediction.engine import Predictor, normalize, tokenize
from tests.conftest import DICTS


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(lang="en", store=None, clock=None):
    p = Predictor(DICTS, store=store, clock=clock or FakeClock())
    p.set_language(lang)
    return p


def test_prefix_completion_returns_matching_words():
    p = make()
    out = p.suggest("wa", n=5)
    assert out and all(w.lower().startswith("wa") for w in out)
    assert "want" in out
    assert "water" in p.suggest("wat", n=3)


def test_suggestions_are_deterministic():
    p1, p2 = make(), make()
    assert p1.suggest("th", n=6) == p2.suggest("th", n=6)
    assert p1.suggest("", prev="i", n=6) == p2.suggest("", prev="i", n=6)


def test_suggestion_count_is_respected():
    p = make()
    for n in (3, 4, 5, 6):
        assert len(p.suggest("", n=n)) == n


def test_next_word_prediction_uses_bigrams():
    p = make()
    assert "help" in p.suggest("", prev="need", n=5)
    assert "you" in p.suggest("", prev="thank", n=3)


def test_accent_insensitive_matching_spanish_and_catalan():
    es = make("es")
    assert "médico" in es.suggest("medi", n=6)
    ca = make("ca")
    assert "gràcies" in ca.suggest("gracie", n=3)


def test_english_i_is_capitalised():
    p = make()
    assert "I" in p.suggest("", n=6)


def test_personal_frequency_promotes_learned_words():
    p = make()
    assert "wheelchair" not in p.suggest("wh", n=5)
    for _ in range(5):
        p.learn("wheelchair", None)
    assert p.suggest("wh", n=5)[0] == "wheelchair"


def test_personal_bigram_adapts_next_word():
    p = make()
    for _ in range(4):
        p.learn("pizza", "want")
    assert p.suggest("", prev="want", n=3)[0] == "pizza"


def test_personalization_is_bounded():
    clock = FakeClock()
    p = make(clock=clock)
    for _ in range(2000):
        p.learn("zebra", None)
    state = p._state
    assert state.personal["zebra"][0] <= 500
    # Saturated personal score cannot exceed its weight.
    assert p.score("zebra", "z", None) < 1.0 + 2.0 + 1.0 + 0.5 + 1.0


def test_learning_disabled_does_not_store(db):
    p = make(store=db)
    p.learning_enabled = False
    p.learn("zebra", "a")
    assert db.personal_vocabulary("en") == {}


def test_learning_persists_through_store(db):
    p = make(store=db)
    p.learn("wheelchair", "my")
    p2 = make(store=db)
    assert "wheelchair" in p2.suggest("whe", n=5)
    assert db.personal_bigrams("en")[("my", "wheelchair")][0] == 1


def test_non_words_are_not_learned(db):
    p = make(store=db)
    p.learn("1234", None)
    p.learn("", None)
    assert db.personal_vocabulary("en") == {}


def test_all_languages_load_with_phrases():
    p = Predictor(DICTS)
    for lang in ("en", "es", "ca"):
        p.set_language(lang)
        assert len(p._state.baseline.frequency) > 200
        assert any(e for _, e in p.default_phrases(lang)), "each language needs an emergency phrase"


def test_contraction_fragments_are_not_suggested():
    p = make()
    assert "wasn" not in p.suggest("wa", n=6)
    assert "don't" in p.suggest("do", n=6)


def test_suggest_latency_is_interactive():
    import time

    p = make("es")
    start = time.perf_counter()
    for prefix in "abcdefghijklmnopqrstuvwxyz":
        p.suggest(prefix, None, 6)
    assert (time.perf_counter() - start) / 26 < 0.05


def test_helpers():
    assert normalize("Niño Çà") == "nino ca"
    assert tokenize("I'm fine, thanks!") == ["i'm", "fine", "thanks"]
