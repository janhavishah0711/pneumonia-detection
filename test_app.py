import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import EXPECTED_FEATURES, _preprocess_image, app


client = TestClient(app)


def test_preprocessing_matches_training_pixel_range() -> None:
    pixels = np.tile(np.array([0, 128, 255], dtype=np.uint8), (64, 1))
    pixels = np.resize(pixels, (64, 64))
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")

    features = _preprocess_image(output.getvalue())

    assert features.shape == (1, EXPECTED_FEATURES)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features, pixels.reshape(1, -1) / 255.0)


def _jpeg_bytes() -> bytes:
    image = Image.new("L", (80, 80), color=128)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_predict_accepts_jpeg_and_returns_expected_schema() -> None:
    response = client.post(
        "/predict",
        files={"file": ("chest_xray.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"NORMAL", "PNEUMONIA"}
    assert body["class"] in {0, 1}
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_unsupported_media_type() -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
