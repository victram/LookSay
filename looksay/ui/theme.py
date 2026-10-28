"""Day/night themes with normal or high contrast, rendered to a Qt stylesheet.

Calm, flat colours; no animation or flashing anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    window: str
    canvas: str
    canvas_text: str
    text: str
    key: str
    key_text: str
    key_border: str
    key_hover: str
    special: str  # control keys
    suggestion: str
    accent: str  # dwell progress, gaze cursor, caret
    emergency: str
    emergency_text: str
    muted: str
    border_width: int = 1


PALETTES = {
    ("day", "normal"): Palette(
        window="#eef0f2", canvas="#ffffff", canvas_text="#111111", text="#1d1d1f", key="#ffffff",
        key_text="#111111", key_border="#c9ced6", key_hover="#e3ecf8", special="#dfe4ea", suggestion="#e8f0fb",
        accent="#1458b8", emergency="#b3261e", emergency_text="#ffffff", muted="#5f6670",
    ),
    ("night", "normal"): Palette(
        window="#15171a", canvas="#202327", canvas_text="#f2f2f2", text="#e8e8e8", key="#2a2e33",
        key_text="#f2f2f2", key_border="#3b4148", key_hover="#33404f", special="#23272c", suggestion="#1e2a38",
        accent="#7fb2ff", emergency="#d0453c", emergency_text="#ffffff", muted="#a3a9b1",
    ),
    ("day", "high"): Palette(
        window="#ffffff", canvas="#ffffff", canvas_text="#000000", text="#000000", key="#ffffff",
        key_text="#000000", key_border="#000000", key_hover="#ffffff", special="#ffffff", suggestion="#ffffff",
        accent="#0000cc", emergency="#a00000", emergency_text="#ffffff", muted="#000000", border_width=3,
    ),
    ("night", "high"): Palette(
        window="#000000", canvas="#000000", canvas_text="#ffff00", text="#ffff00", key="#000000",
        key_text="#ffff00", key_border="#ffff00", key_hover="#000000", special="#000000", suggestion="#000000",
        accent="#00ffff", emergency="#ff3030", emergency_text="#000000", muted="#ffff00", border_width=3,
    ),
}


def palette(theme: str, contrast: str) -> Palette:
    return PALETTES.get((theme, contrast), PALETTES[("day", "normal")])


def stylesheet(p: Palette, key_scale: float) -> str:
    key_px = int(26 * key_scale)
    small_px = int(19 * key_scale)
    bw = p.border_width
    return f"""
    QWidget#Root, QWidget#Page {{ background: {p.window}; color: {p.text}; }}
    QLabel {{ color: {p.text}; font-size: {small_px}px; }}
    QLabel#Title {{ font-size: {int(small_px * 1.3)}px; font-weight: 600; }}
    QLabel#Status {{ color: {p.muted}; font-size: {small_px}px; }}
    QLabel#Value {{ font-size: {small_px}px; font-weight: 600; }}
    QLabel#Message {{ font-size: {int(small_px * 1.4)}px; }}
    GazeButton {{
        background: {p.key}; color: {p.key_text}; border: {bw}px solid {p.key_border};
        border-radius: 10px; font-size: {key_px}px; padding: 2px;
    }}
    GazeButton:hover {{ background: {p.key_hover}; }}
    GazeButton[role="special"] {{ background: {p.special}; font-size: {small_px}px; }}
    GazeButton[role="suggestion"] {{ background: {p.suggestion}; font-size: {int(key_px * 0.9)}px; }}
    GazeButton[role="emergency"] {{
        background: {p.emergency}; color: {p.emergency_text}; font-weight: 700; font-size: {small_px}px;
    }}
    GazeButton[role="active"] {{ background: {p.special}; border: {bw + 2}px solid {p.accent}; font-size: {small_px}px; }}
    GazeButton:disabled {{ color: {p.muted}; }}
    QWidget#Overlay {{ background: {p.window}; }}
    """
