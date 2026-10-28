from looksay.input.text_buffer import TextBuffer


def type_text(buf, text):
    for ch in text:
        buf.space() if ch == " " else buf.insert_char(ch)


def test_auto_capitalisation_and_punctuation_spacing():
    buf = TextBuffer()
    type_text(buf, "hi there")
    buf.insert_char(".")
    type_text(buf, "ok")
    assert buf.text == "Hi there. Ok"


def test_accept_word_replaces_prefix_and_adds_space():
    buf = TextBuffer()
    type_text(buf, "i need wa")
    buf.accept_word("water")
    assert buf.text == "I need water "
    assert buf.current_prefix() == ""
    assert buf.previous_word() == "water"


def test_accept_word_capitalises_at_sentence_start():
    buf = TextBuffer()
    buf.accept_word("hello")
    assert buf.text == "Hello "


def test_word_commit_callback_reports_previous_word():
    committed = []
    buf = TextBuffer(on_word_committed=lambda w, p: committed.append((w, p)))
    type_text(buf, "i want ")
    buf.accept_word("water")
    assert committed == [("I", None), ("want", "i"), ("water", "want")]


def test_backspace_delete_word_undo_and_clear():
    buf = TextBuffer()
    type_text(buf, "hello world")
    buf.backspace()
    assert buf.text == "Hello worl"
    buf.delete_word()
    assert buf.text == "Hello "
    buf.undo()
    assert buf.text == "Hello worl"
    buf.clear()
    assert buf.text == ""
    buf.undo()
    assert buf.text == "Hello worl"


def test_shift_is_one_shot():
    buf = TextBuffer(auto_capitalize=False)
    buf.toggle_shift()
    type_text(buf, "ab")
    assert buf.text == "Ab"


def test_last_sentence():
    buf = TextBuffer(text="Hello. How are you? I am fine")
    assert buf.last_sentence() == "I am fine"


def test_previous_word_stops_at_sentence_boundary():
    buf = TextBuffer(text="I am tired. ")
    assert buf.previous_word() is None
    buf = TextBuffer(text="I am ")
    assert buf.previous_word() == "am"


def test_catalan_apostrophe_word():
    buf = TextBuffer(text="Vull ")
    buf.accept_word("l'")
    assert buf.text == "Vull l'"
