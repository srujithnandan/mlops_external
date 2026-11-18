# 3D Printer Warping Prediction — Final Report

## Executive summary

The project creates material-aware 3D-printing datasets, transforms process
variables into a model-ready feature set, compares two classifiers, and
registers the selected model in MLflow for API serving.

## Dataset versions

Both versions contain 250 rows and 9 columns, including the binary
`warping_defect` target.

| Dataset | Class 0 | Class 1 | Change |
|---|---:|---:|---|
| V1 | 83.6% | 16.4% | Baseline process parameters |
| V2 | 46.8% | 53.2% | Lower bed temperatures and higher speeds |

## Pipeline

The preprocessing pipeline imputes numeric values with the median, scales
numeric features, imputes categorical values with the most frequent category,
and one-hot encodes `material`. Feature engineering adds:

- `nozzle_bed_temperature_difference`
- `cooling_stress_index`

The transformed V1 feature matrix has 12 columns.

## Model selection

Logistic Regression and three Random Forest configurations were evaluated on
a dedicated stratified 30-sample test set. The selected Random Forest achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.77 |
| Positive-class precision | 0.38 |
| Positive-class recall | 0.60 |
| Positive-class F1 | 0.46 |
| Weighted F1 | 0.79 |

The model was registered as `WarpingPredictionModel`, version 2.

## Deployment and CI/CD

`app.py` provides `/health` and `/predict` endpoints and loads the registered
MLflow model through `MODEL_URI`. `.github/workflows/validate.yml` installs the
requirements and runs the test suite on every push and pull request.

## Limitations and next steps

The positive class has only five test examples, so recall and F1 have high
uncertainty. Before production use, add more real print outcomes, validate
against V2 as a drift set, capture MLflow screenshots, and add an authenticated
deployment workflow.
