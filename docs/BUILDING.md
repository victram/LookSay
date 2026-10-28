# Setup, testing and packaging

## Run from source

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # pinned, tested versions
python -m looksay                      # --mouse (no camera) · --fullscreen · --data-dir DIR · --debug
```

After installation everything runs offline. The face-landmark model
(`looksay/resources/models/face_landmarker.task`) and the dictionaries are
bundled with the app.

Linux also needs the system Qt/GL libraries (`libegl1 libgl1 libgles2
libxkbcommon0`) and, for speech, `speech-dispatcher` or `espeak-ng`.
Windows and macOS use their built-in voices. To add more offline voices, use
Windows *Settings → Time & language → Speech*, or macOS *System Settings →
Accessibility → Spoken content*.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q                     # unit + integration + offscreen UI tests
python tools/benchmark.py     # performance on this machine
```

The UI tests run headless (`QT_QPA_PLATFORM=offscreen`) and type a sentence
using simulated gaze only. `tests/test_tracking.py` also runs the real
landmark model on a portrait bundled with matplotlib.

## Windows executable and installer

```bat
pip install -r requirements-dev.txt
pyinstaller packaging\looksay.spec --noconfirm     :: -> dist\LookSay\LookSay.exe
iscc packaging\installer.iss                       :: -> dist\LookSay-0.1.0-setup.exe (Inno Setup 6)
```

The same spec builds a folder bundle on Linux and macOS. Build on the target
OS, because PyInstaller does not cross-compile. Use a clean virtual
environment: any extra package you have installed (for example a second
OpenCV wheel) ends up in the bundle.

## Offline dependencies

| Component | Used for | Size | License |
|---|---|---|---|
| PySide6 (Qt 6) Essentials + Addons | UI, offline TTS (QtTextToSpeech), click sound | ~130 MB | LGPLv3 |
| MediaPipe (Tasks) + `face_landmarker.task` | Face + iris landmarks on the CPU | ~125 MB | Apache-2.0 |
| OpenCV | Camera capture, resizing | ~90–200 MB | Apache-2.0 |
| NumPy | Calibration maths | ~40 MB | BSD |
| pyttsx3 | Speech fallback (SAPI5 / NSSpeech / eSpeak) | small | MPL-2.0 |
| SQLite (Python stdlib) | Local storage | — | Public domain |

## Updating the dictionaries

`tools/build_dictionaries.py` rebuilds `*_frequency.txt` from the
[FrequencyWords](https://github.com/hermitdave/FrequencyWords) 2018 lists
(CC-BY-SA-4.0). The curated `*_words.txt`, `*_corpus.txt` (bigram seed
sentences) and `phrases.json` (default quick phrases) are plain text you can
edit.
