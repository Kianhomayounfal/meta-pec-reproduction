# Meta-Pec Reproduction

Reproduction and analysis of the Meta-Pec framework from the paper:

**"A Preference-aware Meta-optimization Framework for Personalized Vehicle Energy Consumption Estimation"**
Lai et al., KDD 2023 — https://doi.org/10.1145/3580305.3599767

## What this repo contains

- Reproduction of the original Meta-Pec model (`model.py`, `model_utils.py`)
- Two additional baselines implemented from scratch: Multiple Linear Regression (MLR) and XGBoost (`baselines.py`)
- Full training and evaluation pipeline (`main.py`, `utils.py`)
- Experimental report comparing all three models on the VED and ETTD datasets (`Practical_Work_Report.pdf`)

## How to run

```bash
# Run Meta-Pec (default)
python main.py --dataset VED

# Run MLR baseline
python main.py --dataset VED --baseline_type mlr

# Run XGBoost baseline
python main.py --dataset VED --baseline_type xgboost
```

## Results summary

| Dataset | Model | MSE | MAE | MAPE |
|---------|-------|-----|-----|------|
| VED | MLR | 0.0668 | 0.1043 | 123.49% |
| VED | XGBoost | 0.0199 | 0.0208 | 12.66% |
| VED | Meta-Pec | 0.0727 | 0.0607 | 32.86% |
| ETTD | MLR | 0.3870 | 0.3996 | 34.58% |
| ETTD | XGBoost | 0.2203 | 0.2905 | 23.48% |
| ETTD | Meta-Pec | 1.1183 | 0.6744 | 55.13% |

In my reproduction setup (CPU only, 6 epochs), XGBoost outperformed Meta-Pec on both datasets. See the report for full analysis.

## Datasets

Datasets are not included. Download them here:
- VED: https://github.com/gsoh/VED
- ETTD: http://guangwang.me/#/data

## Requirements

```bash
pip install torch numpy scikit-learn xgboost tqdm
```
