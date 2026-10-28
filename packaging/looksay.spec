# PyInstaller spec: builds a self-contained, offline LookSay folder (onedir).
#   pyinstaller packaging/looksay.spec --noconfirm
# Output: dist/LookSay/ (run LookSay.exe on Windows). Wrap it with
# packaging/installer.iss (Inno Setup) to produce a Windows installer.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent
datas = [(str(root / "looksay" / "resources"), "looksay/resources")]
datas += collect_data_files("mediapipe", includes=["**/*.so", "**/*.dll", "**/*.dylib", "**/*.binarypb", "**/*.tflite"])
binaries = collect_dynamic_libs("mediapipe")
hidden = ["pyttsx3.drivers", "pyttsx3.drivers.sapi5", "pyttsx3.drivers.espeak", "pyttsx3.drivers.nsss"]

a = Analysis(
    [str(root / "looksay" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtPdf",
              "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtCharts", "PySide6.QtDataVisualization",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtOpenGL",
              # Pulled in by mediapipe's optional helpers, unused by LookSay at runtime.
              "pandas", "lxml", "scipy", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="LookSay", console=False,
    icon=str(root / "assets" / ("logo.ico" if sys.platform == "win32" else "logo.png")),
)
coll = COLLECT(exe, a.binaries, a.datas, name="LookSay")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="LookSay.app", icon=str(root / "assets" / "logo.png"), bundle_identifier="org.looksay.app",
        info_plist={
            "NSCameraUsageDescription": "LookSay uses the camera to follow your eyes. Images never leave this computer.",
            "NSHighResolutionCapable": True,
        },
    )
