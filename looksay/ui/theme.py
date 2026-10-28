"""Themes rendered to a Qt stylesheet.

Day = "Papel": white page, black type, light borderless keys, one black
primary action (Speak) and a red-outlined HELP. Night = the same layout in
dark with an amber accent. Both have a high-contrast variant. Calm, flat
colours; no animation or flashing anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

FONT_FAMILY = "Inter"


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
    accent: str  # dwell outline/progress, gaze cursor, caret
    primary: str  # the one strong action (Speak)
    primary_text: str
    emergency: str  # HELP: outline + text colour
    emergency_bg: str
    muted: str
    pill_border: str
    border_width: int = 1


PALETTES = {
    ("day", "normal"): Palette(
        window="#FFFFFF", canvas="#FFFFFF", canvas_text="#111111", text="#111111", key="#FAFAFA",
        key_text="#111111", key_border="#E6E6E6", key_hover="#F2F2F2", special="#FAFAFA", suggestion="#F4F4F4",
        accent="#111111", primary="#111111", primary_text="#FFFFFF", emergency="#B3261E", emergency_bg="#FFFFFF",
        muted="#595959", pill_border="#D4D4D4",
    ),
    ("night", "normal"): Palette(
        window="#0E0F11", canvas="#0E0F11", canvas_text="#F2F2F2", text="#F2F2F2", key="#1B1D21",
        key_text="#F2F2F2", key_border="#1B1D21", key_hover="#24272C", special="#1B1D21", suggestion="#16181B",
        accent="#F5B83D", primary="#F5B83D", primary_text="#0E0F11", emergency="#FF6B5E", emergency_bg="#0E0F11",
        muted="#A6A9AF", pill_border="#33363C",
    ),
    ("day", "high"): Palette(
        window="#FFFFFF", canvas="#FFFFFF", canvas_text="#000000", text="#000000", key="#FFFFFF",
        key_text="#000000", key_border="#000000", key_hover="#FFFFFF", special="#FFFFFF", suggestion="#FFFFFF",
        accent="#0000CC", primary="#000000", primary_text="#FFFFFF", emergency="#A00000", emergency_bg="#FFFFFF",
        muted="#000000", pill_border="#000000", border_width=3,
    ),
    ("night", "high"): Palette(
        window="#000000", canvas="#000000", canvas_text="#FFFF00", text="#FFFF00", key="#000000",
        key_text="#FFFF00", key_border="#FFFF00", key_hover="#000000", special="#000000", suggestion="#000000",
        accent="#00FFFF", primary="#FFFF00", primary_text="#000000", emergency="#FF3030", emergency_bg="#000000",
        muted="#FFFF00", pill_border="#FFFF00", border_width=3,
    ),
}


def palette(theme: str, contrast: str) -> Palette:
    return PALETTES.get((theme, contrast), PALETTES[("day", "normal")])


def stylesheet(p: Palette, key_scale: float) -> str:
    key_px = int(30 * key_scale)
    small_px = int(20 * key_scale)
    label_px = int(17 * key_scale)
    bw = p.border_width
    sugg_border = f"{bw}px solid {p.key_border}" if bw > 1 else "none"
    return f"""
    QWidget#Root, QWidget#Page {{ background: {p.window}; color: {p.text}; font-family: "{FONT_FAMILY}"; }}
    QLabel {{ color: {p.text}; font-size: {label_px}px; }}
    QLabel#Title {{ font-size: {int(label_px * 1.2)}px; font-weight: 600; }}
    QLabel#Status {{ color: {p.muted}; font-size: {label_px}px; }}
    QLabel#Value {{ font-size: {label_px}px; font-weight: 600; }}
    QLabel#Message {{ font-size: {int(small_px * 1.4)}px; }}
    GazeButton {{
        background: {p.key}; color: {p.key_text}; border: {bw}px solid {p.key_border};
        border-radius: 14px; font-size: {key_px}px; padding: 2px;
    }}
    GazeButton:hover {{ background: {p.key_hover}; }}
    GazeButton[role="special"] {{ background: {p.special}; font-size: {small_px}px; }}
    GazeButton[role="suggestion"] {{
        background: {p.suggestion}; border: {sugg_border}; border-radius: 16px;
        font-size: {int(key_px * 0.87)}px; font-weight: 500;
    }}
    GazeButton[role="primary"] {{
        background: {p.primary}; color: {p.primary_text}; border: none; font-size: {int(small_px * 1.1)}px;
        font-weight: 600;
    }}
    GazeButton[role="emergency"] {{
        background: {p.emergency_bg}; color: {p.emergency}; border: {max(2, bw)}px solid {p.emergency};
        font-weight: 600; font-size: {small_px}px;
    }}
    GazeButton[role="active"] {{
        background: {p.window}; border: {bw + 2}px solid {p.accent}; font-size: {small_px}px;
    }}
    GazeButton[role="small"] {{ background: {p.special}; font-size: {label_px}px; }}
    GazeButton[role="small_active"] {{
        background: {p.window}; border: {bw + 2}px solid {p.accent}; font-size: {label_px}px;
    }}
    GazeButton[role="small_danger"] {{
        background: {p.emergency_bg}; color: {p.emergency}; border: {max(2, bw)}px solid {p.emergency};
        font-size: {label_px}px; font-weight: 600;
    }}
    GazeButton[role="urgent"] {{
        background: {p.emergency_bg}; color: {p.emergency}; border: {max(2, bw)}px solid {p.emergency};
        border-radius: 16px; font-size: {int(key_px * 0.87)}px; font-weight: 600;
    }}
    GazeButton[role="pill"] {{
        background: transparent; border: {bw}px solid {p.pill_border}; border-radius: 22px;
        font-size: {label_px}px; padding: 0 20px;
    }}
    GazeButton:disabled {{ color: {p.muted}; }}
    QWidget#Overlay {{ background: {p.window}; }}
    """
