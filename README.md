# Chest X-Ray Pneumonia Detection API

This project exposes the trained `logistic_regression.pkl` pipeline through
FastAPI. The pipeline is verified to accept 4,096 features, so uploaded images
are converted to grayscale, resized to 64 x 64 pixels, and flattened before
inference.

## Run the API

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the development server from this directory:

```powershell
python -m uvicorn app:app --reload
```

Open the interactive API documentation at
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Predict

Send a JPEG or PNG image as the `file` multipart/form-data field:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -H "accept: application/json" `
  -F "file=@chest_xray.jpg"
```

The response contains the predicted class and the model's probability for the
PNEUMONIA class:

```json
{
  "prediction": "PNEUMONIA",
  "class": 1,
  "confidence": 0.94
}
```

The confidence value above is illustrative; it is calculated from the uploaded
image at request time.

## Deploy on Render

This repository includes a Render Blueprint in [`render.yaml`](render.yaml).

1. Push the project to a GitHub or GitLab repository.
2. In the Render dashboard, select **New > Blueprint** and connect the repository.
3. Select `render.yaml` and create the service.
4. Wait for the health check at `/health` to pass, then use the generated
   `https://...onrender.com/docs` URL to access the API documentation.

The service uses Render's `PORT` environment variable and binds to
`0.0.0.0`, as required for a Render web service. Render's free service
filesystem is ephemeral, so prediction logs written by the monitoring module
should not be treated as durable storage.
