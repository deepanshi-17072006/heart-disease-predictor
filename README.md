# Heart Disease Risk Predictor

CSL236 end-term project: a production-style ML system (MLflow tracking, validated FastAPI service, pytest, Docker, GitHub Actions CI, cloud deployment with input logging).

**Live URL:** `<paste deployment URL here>`  |  **API docs:** `<URL>/docs`

## 1. Problem statement and dataset

Given 13 clinical measurements, estimate whether a patient is at higher risk of heart disease (binary classification). This continues the mid-term model: same data, same preprocessing, now served as a system.

- **Dataset:** Heart Disease (Cleveland) dataset from Kaggle/UCI: `<paste Kaggle or UCI link here>`. 13 input features and 1 binary target. The Kaggle CSV has 1,025 rows, but only 302 are unique patients (the rest are repeats), so `train.py` drops the duplicates before splitting. Public and anonymised.
- **Size note:** after removing duplicate rows the dataset has 302 unique patients, below the 1,000-row guideline. Training on the raw 1,025 rows would meet the number on paper but leaks repeated patients into the test set (we measured 100% test accuracy that way, which is not real). We kept the mid-term dataset; every other requirement is met. Because the test split is only about 61 rows, models are selected with 5-fold cross-validation on the training split, and test metrics are treated as indicative.
- **Label direction:** `train.py` prints which target value it treats as "heart disease" (detected from how the target correlates with `oldpeak`, `exang`, `ca` and `thalach`) and stores it in `model/meta.json`. The API uses it, so the reported risk always means P(heart disease).

## 2. Data contract

Every rule is enforced in the Pydantic schema in `app.py`. Out-of-range, wrongly typed, missing, extra or NaN values return HTTP 422.

| Feature | Type | Allowed range / values | Why it matters |
|---|---|---|---|
| age | int | 29 – 77 | Risk rises with age; also used for the age-group feature |
| sex | int | {0, 1} (0 = female, 1 = male) | Risk differs by sex |
| cp | int | {0, 1, 2, 3} | Chest pain type; strong signal |
| trestbps | int | 80 – 220 mm Hg | Resting blood pressure |
| chol | int | 100 – 600 mg/dl | Serum cholesterol |
| fbs | int | {0, 1} | Fasting blood sugar > 120 mg/dl |
| restecg | int | {0, 1, 2} | Resting ECG result |
| thalach | int | 60 – 220 bpm | Maximum heart rate; lower values go with disease |
| exang | int | {0, 1} | Exercise-induced angina |
| oldpeak | float | 0.0 – 7.0 | ST depression after exercise |
| slope | int | {0, 1, 2} | Slope of the peak exercise ST segment |
| ca | int | 0 – 4 | Major vessels coloured by fluoroscopy; strong signal |
| thal | int | {0, 1, 2, 3} | Thalassemia test result |
| target | int | {0, 1} | Label (heart disease present / absent) |

## 3. Setup and run

```bash
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# put the dataset at data/heart.csv
python train.py --n_estimators 100 --max_depth 6      # single run (random forest)
python train.py --sweep                               # 9 runs, 3 algorithms
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
uvicorn app:app --reload --port 8000                  # UI at /, docs at /docs
pytest -v --tb=short
```

Docker:

```bash
docker build -t heart-disease-predictor .
docker run -d -p 8000:8000 --restart unless-stopped heart-disease-predictor
docker images heart-disease-predictor                 # note the image size below
```

Image size: `<fill in from docker images>`

Drift check (bonus): `python drift_check.py` compares the logged inputs in `logs/predictions.jsonl` with the training data.

## 4. Experiments (MLflow)

Run `python train.py --sweep` and paste the table it prints here. Add the MLflow UI screenshot below.

| Run | Algorithm | Hyperparameters | CV F1 | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| | | | | | | |

## 5. Model choice

`<Fill in after the sweep: which model won on cross-validated F1, and why (simplicity, stability across folds, recall on the disease class).>`

The final model is a single scikit-learn `Pipeline` (one-hot encoding, age bins, log1p on skewed columns, RobustScaler, SelectKBest k=15, classifier) saved as `model/model.joblib`, so the API applies exactly the preprocessing used in training. It is loaded once at startup in `app.py`.

## 6. Why two requirements files?

`requirements.txt` is the full training/dev environment (MLflow, pytest). `requirements-serve.txt` has only what the API needs, which keeps the Docker image small.

## 7. Screenshots

UI, `/docs`, MLflow UI (3+ runs), `pytest -v` output, green CI check on the latest commit, sample prediction log (`logs/predictions.jsonl`). `<add images here>`

## 8. Team

| Name | Roll no. | Contribution |
|---|---|---|
| Deepanshi | 24csu328 | `<fill in>` |
| Baani | 24csu349 | `<fill in>` |

Section: `<fill in>`
