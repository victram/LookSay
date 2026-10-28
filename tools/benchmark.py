"""Measure LookSay's hot paths on this machine and print a Markdown table.

    python tools/benchmark.py

Uses a real portrait bundled with matplotlib (if installed) for landmark
inference; no camera needed. Numbers are machine-specific - rerun on the
target hardware.
"""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def timeit(fn, n: int) -> tuple[float, float]:
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.95) - 1]


def main() -> None:
    import numpy as np

    from looksay.calibration.model import CalibrationSession, calibration_targets
    from looksay.input.dwell import DwellSelector
    from looksay.paths import resources_dir
    from looksay.prediction.engine import Predictor
    from looksay.tracking.features import compute_features

    rows = []
    t0 = time.perf_counter()
    predictor = Predictor(resources_dir() / "dictionaries")
    predictor.set_language("es")
    rows.append(("Load language resources (es)", (time.perf_counter() - t0) * 1000, None))
    rows.append(("Prefix suggestions ('ne', 6)", *timeit(lambda: predictor.suggest("ne", "yo", 6), 500)))
    rows.append(("Next-word suggestions (6)", *timeit(lambda: predictor.suggest("", "necesito", 6), 500)))
    rows.append(("Word grid (21)", *timeit(lambda: predictor.suggest("", "quiero", 21), 300)))

    dwell = DwellSelector()
    t = [0.0]

    def tick():
        t[0] += 0.033
        dwell.update("k", t[0])
    rows.append(("Dwell state update", *timeit(tick, 5000)))

    rng = np.random.default_rng(0)
    session = CalibrationSession(calibration_targets(9))
    for i, (tx, ty) in enumerate(session.targets):
        for _ in range(30):
            session.add_sample(i, np.array([0.4 + 0.2 * tx, -0.1 + 0.1 * ty, 0, 0]) + rng.normal(0, 0.003, 4))
    rows.append(("Fit 9-point calibration (270 samples)", *timeit(session.fit, 50)))
    model = session.fit()
    rows.append(("Gaze prediction", *timeit(lambda: model.predict(np.array([0.5, -0.05, 0, 0])), 2000)))

    try:
        import cv2
        import matplotlib

        from looksay.tracking.landmarks import LandmarkDetector

        image = cv2.imread(str(Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"))
        detector = LandmarkDetector()
        for size in ((320, 240), (640, 480)):
            frame = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
            ts = [0.0]

            def detect():
                ts[0] += 0.033
                return detector.detect(frame, ts[0])
            for _ in range(10):
                detect()
            med, p95 = timeit(detect, 100)
            rows.append((f"Face+iris landmarks {size[0]}x{size[1]} (MediaPipe)", med, p95))
        pts, _ = detector.detect(frame, 1000.0)
        rows.append(("Gaze features from landmarks", *timeit(lambda: compute_features(pts, 640, 480, 0.0), 2000)))
        detector.close()
    except Exception as exc:  # pragma: no cover
        print(f"(landmark benchmark skipped: {exc})")

    try:
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        rss_mb = float("nan")

    print(f"Machine: {platform.processor() or platform.machine()} · {os.cpu_count()} CPUs · "
          f"Python {platform.python_version()} · {platform.system()}\n")
    print("| Operation | median ms | p95 ms |")
    print("|---|---:|---:|")
    for name, med, p95 in rows:
        print(f"| {name} | {med:.3f} | {'' if p95 is None else f'{p95:.3f}'} |")
    print(f"\nPeak memory of this benchmark process: {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
