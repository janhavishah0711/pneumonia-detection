
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent

LOG_FILE = BASE_DIR / "logs" / "predictions.csv"
BASELINE_FILE = BASE_DIR / "reports" / "training_baseline.json"
REPORT_FILE = BASE_DIR / "reports" / "drift_report.json"

def detect_drift():

    if not LOG_FILE.exists():
        print("No prediction logs available.")
        return

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    logs = pd.read_csv(LOG_FILE)

    if len(logs) < 30:
        print("At least 30 predictions required for drift analysis.")
        return

    recent = logs.tail(100)

    baseline_mean = baseline["mean_intensity"]
    baseline_std = baseline["std_intensity"]

    incoming_mean = recent["mean_intensity"].mean()

    difference = abs(incoming_mean - baseline_mean)

    threshold = max(0.02, 2 * baseline_std)

    drift_detected = bool(difference > threshold)

    pneumonia_rate = float(
        (recent["prediction"] == 1).mean()
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples_analyzed": len(recent),
        "training_mean_intensity": baseline_mean,
        "incoming_mean_intensity": float(incoming_mean),
        "absolute_difference": float(difference),
        "threshold": float(threshold),
        "drift_detected": drift_detected,
        "incoming_pneumonia_prediction_rate": pneumonia_rate
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("Drift report generated.")
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    detect_drift()
