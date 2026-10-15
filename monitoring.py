
import csv
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "predictions.csv"

MODEL_VERSION = "v1"

def log_prediction(prediction, probability, mean_intensity):

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "pneumonia_probability",
                "mean_intensity",
                "model_version"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            round(float(probability), 4),
            round(float(mean_intensity), 4),
            MODEL_VERSION
        ])
