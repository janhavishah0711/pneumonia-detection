import io
from pathlib import Path
from monitoring import log_prediction
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


MODEL_PATH = Path(__file__).with_name("logistic_regression.pkl")
IMAGE_SIZE = (64, 64)
EXPECTED_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1]

model = joblib.load(MODEL_PATH)

if getattr(model, "n_features_in_", None) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Model expects {getattr(model, 'n_features_in_', 'an unknown number of')} "
        f"features; {EXPECTED_FEATURES} features are required."
    )

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Classifies chest X-ray images as NORMAL or PNEUMONIA.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _preprocess_image(contents: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image_array = np.asarray(
                image.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid JPEG or PNG image.",
        ) from exc

    return image_array.reshape(1, EXPECTED_FEATURES)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    features = _preprocess_image(contents)
    probabilities = model.predict_proba(features)[0]
    classes = np.asarray(model.classes_)
    pneumonia_indices = np.flatnonzero(classes == 1)
    if len(pneumonia_indices) != 1:
        raise RuntimeError("The loaded model does not define a class labeled 1.")

    pneumonia_probability = float(probabilities[pneumonia_indices[0]])
    predicted_class = int(model.predict(features)[0])

    log_prediction(
    prediction=predicted_class,
    probability=pneumonia_probability,
    mean_intensity=float(features.mean())
    )

    return {
        "prediction": "PNEUMONIA" if predicted_class == 1 else "NORMAL",
        "class": predicted_class,
        "confidence": pneumonia_probability,
    }
