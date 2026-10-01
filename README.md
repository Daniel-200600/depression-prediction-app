# Student Depression Risk Predictor

A Streamlit app that estimates a **depression risk score (0–100 %)** for a student
from lifestyle and academic factors, using a Random Forest model.

> Educational project built for a Machine Learning course. It is **not** a medical
> tool and must not be used for diagnosis.

## What it does

- Sidebar form: age, gender, department, CGPA, study hours, sleep duration,
  social-media time, physical activity and stress level.
- **Prediction** tab: risk gauge (low < 35 %, moderate 35–65 %, high > 65 %).
- **Profile analysis** tab: radar chart comparing the entered profile with dataset averages.
- **Model information** tab: model type, input variables, target and main hyperparameters.

## Model

- Random Forest regressor in a scikit-learn pipeline, saved as `depression_rf_model.pkl`.
- Target: depression (0/1); the model outputs a probability-like score in [0, 1].
- Main hyperparameters: `n_estimators=300`, `max_features="sqrt"`, `min_samples_leaf=5`, `random_state=42`.
- Training data: `student_lifestyle_100k.csv` (not included in this repository).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_depression.py
```

Dependencies are pinned (notably `scikit-learn==1.4.2`) because the model is a
pickle file, which must be loaded with a compatible scikit-learn version.

## Files

| File | Purpose |
|---|---|
| `app_depression.py` | Streamlit application |
| `depression_rf_model.pkl` | Trained model |
| `requirements.txt` | Pinned dependencies |

## Stack

Python, Streamlit, scikit-learn, pandas, NumPy, matplotlib, seaborn.

## Limitations

- Trained on a survey-style student dataset; results do not generalise to clinical use.
- The training notebook/script is not part of this repository.
