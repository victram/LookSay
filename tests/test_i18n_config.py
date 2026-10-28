from looksay.config import SETTING_SPECS, SPECS_BY_KEY
from looksay.i18n.layouts import letter_rows, symbol_rows
from looksay.i18n.strings import LANGUAGES, STRINGS, tr


def test_every_string_is_translated():
    for key, values in STRINGS.items():
        assert len(values) == len(LANGUAGES), key
        assert all(v.strip() for v in values), key


def test_every_setting_has_a_label():
    for spec in SETTING_SPECS:
        assert f"set_{spec.key}" in STRINGS, spec.key


def test_tr_formats_and_falls_back():
    assert tr("exported", "es", path="x") == "Exportado a x"
    assert tr("missing-key", "ca") == "missing-key"
    assert tr("yes", "xx") == "Yes"


def test_layouts_contain_language_letters():
    flat = lambda rows: {k for r in rows for k in r}
    assert "ñ" in flat(letter_rows("es"))
    assert "ç" in flat(letter_rows("ca"))
    assert "·" in flat(symbol_rows("ca"))
    for lang in LANGUAGES:
        for layout in ("qwerty", "frequency"):
            rows = letter_rows(lang, layout)
            assert len(rows) == 3
            assert set("abcdefghijklmnopqrstuvwxyz") <= flat(rows)


def test_setting_spec_clamp_and_step():
    dwell = SPECS_BY_KEY["dwell_ms"]
    assert dwell.clamp(50) == 300 and dwell.clamp("bad") == dwell.default
    assert dwell.step_value(3000, 1) == 3000
    lang = SPECS_BY_KEY["language"]
    assert lang.step_value("ca", 1) == "en"
    assert SPECS_BY_KEY["blink_select"].step_value(False, 1) is True
    rate = SPECS_BY_KEY["speech_rate"]
    assert rate.step_value(0.9, 1) == 1.0
