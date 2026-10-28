"""Integration tests for the speech adapters (no audio device required)."""

import pytest

from looksay.speech.tts import NullSpeech, QtSpeech, create_speech


def test_null_speech_records_text():
    s = NullSpeech()
    s.speak("hello")
    s.stop()
    assert s.spoken == ["hello"] and s.voices() == [] and not s.is_speaking()


def test_qt_mock_engine_speaks_and_stops(qapp):
    try:
        speech = QtSpeech("mock")
    except Exception as exc:  # pragma: no cover - plugin not shipped
        pytest.skip(f"Qt mock TTS engine unavailable: {exc}")
    for lang in ("en", "es", "ca"):
        speech.set_language(lang)
    speech.set_rate(0.5)
    speech.set_volume(0.7)
    speech.set_voice(1)
    speech.speak("Hello there")
    speech.stop()
    assert not speech.is_speaking()
    speech.speak("   ")  # blank text is ignored


def test_create_speech_always_returns_an_engine(qapp):
    speech = create_speech()
    for method in ("speak", "stop", "voices", "set_language", "set_rate", "set_volume", "set_voice"):
        assert callable(getattr(speech, method))
    speech.set_language("es")
    assert isinstance(speech.voices(), list)
    speech.stop()
