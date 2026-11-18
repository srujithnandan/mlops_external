# 3D Printer Warping Prediction

This project predicts the `warping_defect` outcome for a 3D-printing job.
The notebook generated V1/V2 datasets, engineered temperature-difference and
cooling-stress features, compared Logistic Regression with Random Forest, and
registered the selected model as `WarpingPredictionModel` in MLflow.

## Run locally

Install dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The API expects the MLflow registry/database and model artifacts produced by
the notebook to be available locally. Set the tracking URI and model version
when they are in a different location:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MODEL_URI = "models:/WarpingPredictionModel/2"
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the API. The health endpoint is
`GET /health`; predictions use `POST /predict`.

## Validate

```powershell
pytest -q
```

## Recorded experiment result

The notebook's 30-sample stratified test set produced the following best
Random Forest report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.91 | 0.80 | 0.85 | 25 |
| 1 | 0.38 | 0.60 | 0.46 | 5 |

Overall accuracy was 0.77 and weighted F1 was 0.79. The model was registered
under `WarpingPredictionModel` as version 2.
