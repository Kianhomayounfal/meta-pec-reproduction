# Meta-Pec Reproduction

Reproduction and analysis of the Meta-Pec framework from the paper:
**"A Preference-aware Meta-optimization Framework for Personalized Vehicle Energy Consumption Estimation"** (Lai et al., KDD 2023)

## What this repo contains
- Reproduction of the original Meta-Pec model
- Two additional baselines: Multiple Linear Regression (MLR) and XGBoost
- Full experimental report comparing all three models on the VED and ETTD datasets

## Datasets
The VED and ETTD datasets used in this project are not included in this repo.
- VED: https://github.com/gsoh/VED
- ETTD: http://guangwang.me/#/data

## Results summary
In my reproduction setup, XGBoost achieved the strongest results on both datasets, while Meta-Pec was harder to train under limited compute (CPU only, 6 epochs).

## Reference
Lai et al. (2023). A Preference-aware Meta-optimization Framework for Personalized Vehicle Energy Consumption Estimation. KDD 2023.
https://doi.org/10.1145/3580305.3599767
