"""Setting definitions: defaults, ranges and choices.

The settings UI is generated from SETTING_SPECS, so every option here is
automatically gaze-operable (stepper or cycle control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SettingSpec:
    key: str
    default: Any
    group: str
    choices: tuple | None = None  # cycle through these values
    minimum: float | None = None  # numeric stepper
    maximum: float | None = None
    step: float | None = None

    @property
    def is_bool(self) -> bool:
        return isinstance(self.default, bool)

    def clamp(self, value: Any) -> Any:
        """Coerce a stored/imported value into a valid value, falling back to the default."""
        try:
            if self.is_bool:
                return bool(value)
            if self.choices is not None:
                return value if value in self.choices else self.default
            if self.minimum is not None:
                num = type(self.default)(value)
                num = min(max(num, type(self.default)(self.minimum)), type(self.default)(self.maximum))
                return round(num, 3) if isinstance(num, float) else num
            return type(self.default)(value)
        except (TypeError, ValueError):
            return self.default

    def step_value(self, value: Any, direction: int) -> Any:
        if self.is_bool:
            return not value
        if self.choices is not None:
            idx = self.choices.index(value) if value in self.choices else 0
            return self.choices[(idx + direction) % len(self.choices)]
        if self.minimum is not None:
            return self.clamp(value + direction * self.step)
        return value


SETTING_SPECS: tuple[SettingSpec, ...] = (
    # Language & display
    SettingSpec("language", "en", "display", choices=("en", "es", "ca")),
    SettingSpec("theme", "day", "display", choices=("day", "night")),
    SettingSpec("contrast", "normal", "display", choices=("normal", "high")),
    SettingSpec("text_size", 36, "display", minimum=16, maximum=96, step=4),
    SettingSpec("key_scale", 1.0, "display", minimum=0.7, maximum=1.8, step=0.1),
    SettingSpec("sound_feedback", False, "display"),
    # Keyboard
    SettingSpec("keyboard_mode", "letters", "keyboard", choices=("letters", "words")),
    SettingSpec("keyboard_layout", "qwerty", "keyboard", choices=("qwerty", "frequency")),
    SettingSpec("suggestion_count", 5, "keyboard", minimum=3, maximum=6, step=1),
    SettingSpec("auto_capitalize", True, "keyboard"),
    SettingSpec("learn_words", True, "keyboard"),
    # Selection
    SettingSpec("input_source", "both", "selection", choices=("both", "gaze", "mouse")),
    SettingSpec("dwell_ms", 1000, "selection", minimum=300, maximum=3000, step=100),
    SettingSpec("cooldown_ms", 400, "selection", minimum=100, maximum=2000, step=100),
    SettingSpec("blink_select", False, "selection"),
    # Tracking
    SettingSpec("tracking_eye", "both", "tracking", choices=("both", "left", "right")),
    SettingSpec("calibration_points", 9, "tracking", choices=(5, 9, 13)),
    SettingSpec("sensitivity", 1.0, "tracking", minimum=0.5, maximum=2.0, step=0.1),
    SettingSpec("smoothing", 5, "tracking", minimum=0, maximum=10, step=1),
    SettingSpec("confidence_threshold", 0.5, "tracking", minimum=0.2, maximum=0.9, step=0.05),
    SettingSpec("auto_drift", True, "tracking"),
    SettingSpec("fixation_stabilizer", True, "tracking"),
    SettingSpec("iris_refinement", False, "tracking"),
    SettingSpec("gaze_cursor", True, "tracking"),
    SettingSpec("gaze_cursor_size", 24, "tracking", minimum=8, maximum=64, step=4),
    # Camera
    SettingSpec("camera_index", 0, "camera", minimum=0, maximum=9, step=1),
    SettingSpec("camera_resolution", "1280x720", "camera", choices=("640x480", "1280x720", "1920x1080")),
    SettingSpec("camera_fps", 30, "camera", choices=(15, 30)),
    SettingSpec("inference_every", 1, "camera", minimum=1, maximum=4, step=1),
    SettingSpec("camera_preview", False, "camera"),
    SettingSpec("show_landmarks", False, "camera"),
    # Speech
    SettingSpec("speech_voice", 0, "speech", minimum=0, maximum=50, step=1),  # 0 = automatic
    SettingSpec("speech_rate", 0.0, "speech", minimum=-1.0, maximum=1.0, step=0.1),
    SettingSpec("speech_volume", 0.9, "speech", minimum=0.0, maximum=1.0, step=0.1),
    SettingSpec("speak_mode", "all", "speech", choices=("all", "sentence")),
)

SPECS_BY_KEY = {spec.key: spec for spec in SETTING_SPECS}
DEFAULTS = {spec.key: spec.default for spec in SETTING_SPECS}
SETTING_GROUPS = ("display", "keyboard", "selection", "tracking", "camera", "speech")
