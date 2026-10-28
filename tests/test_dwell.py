from looksay.input.dwell import DwellSelector


def run(sel, target, start, end, step=0.033, **kw):
    selected = []
    t = start
    while t < end:
        s = sel.update(target, t, **kw)
        if s.selected is not None:
            selected.append((s.selected, round(t, 2)))
        t += step
    return selected, t


def test_selects_after_dwell_time():
    sel = DwellSelector(dwell_s=1.0, cooldown_s=0.5)
    selected, _ = run(sel, "a", 0.0, 0.9)
    assert selected == []
    selected, _ = run(sel, "a", 0.9, 1.1)
    assert [s for s, _ in selected] == ["a"]


def test_progress_increases_monotonically():
    sel = DwellSelector(dwell_s=1.0)
    last = 0.0
    t = 0.0
    while t < 0.9:
        p = sel.update("k", t).progress
        assert p >= last
        last = p
        t += 0.05
    assert 0.8 < last < 1.0


def test_cooldown_and_repeat_requires_full_dwell():
    sel = DwellSelector(dwell_s=0.5, cooldown_s=0.5)
    selected, _ = run(sel, "a", 0.0, 2.0)
    times = [t for _, t in selected]
    assert len(times) == 2
    # second selection only after cooldown (0.5) + full dwell (0.5)
    assert times[1] - times[0] >= 0.95


def test_brief_glance_away_does_not_reset():
    sel = DwellSelector(dwell_s=1.0, grace_s=0.15)
    run(sel, "a", 0.0, 0.6)
    sel.update(None, 0.63)
    sel.update("b", 0.66)  # 60 ms excursion
    selected, _ = run(sel, "a", 0.70, 1.25)
    assert [s for s, _ in selected] == ["a"]


def test_switching_targets_resets_progress():
    sel = DwellSelector(dwell_s=1.0, grace_s=0.15)
    run(sel, "a", 0.0, 0.8)
    selected, _ = run(sel, "b", 0.8, 1.6)
    assert selected == []
    assert sel.target == "b"


def test_inactive_tracking_pauses_and_resets():
    sel = DwellSelector(dwell_s=1.0)
    run(sel, "a", 0.0, 0.8)
    state = sel.update("a", 0.85, active=False)
    assert state.target is None and state.progress == 0.0
    selected, _ = run(sel, "a", 0.9, 1.5)
    assert selected == []


def test_hold_freezes_without_reset():
    sel = DwellSelector(dwell_s=1.0)
    run(sel, "a", 0.0, 0.5)
    before = sel.progress
    run(sel, "a", 0.5, 2.0, hold=True)
    assert sel.progress == before


def test_stalled_frame_cannot_complete_dwell():
    sel = DwellSelector(dwell_s=1.0)
    sel.update("a", 0.0)
    state = sel.update("a", 5.0)  # 5 s gap
    assert state.selected is None


def test_blink_trigger_needs_minimum_hold():
    sel = DwellSelector(dwell_s=2.0, blink_min_hold_s=0.3)
    run(sel, "a", 0.0, 0.1)
    assert sel.trigger(0.1) is None
    run(sel, "a", 0.1, 0.5)
    assert sel.trigger(0.5) == "a"
    assert sel.trigger(0.55) is None  # cooldown
