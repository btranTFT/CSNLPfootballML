# Replication Specification

## Objective
Replicate key findings from:
- Muhammad Gustya Armanda and Eka Miranda, "Forecasting Football Results Using Machine Learning Techniques" (IEEE ICAISD 2025).

## Scope
- Re-implement the baseline classification pipeline with:
  - Random Forest
  - XGBoost
  - AdaBoost
- Evaluate both:
  - Imbalanced setting
  - SMOTE setting

## Data and Split
- Dataset path configured in `configs/replication.yaml`.
- Target column: `target`.
- Temporal split enabled when date column is present (`match_date`).

## Metrics
- Accuracy
- Macro Precision / Recall / F1
- One-vs-Rest macro ROC-AUC (when probabilities are available)

## Outputs
- Per-run artifacts in `results/<experiment_name>_<timestamp>/`
- Per-model:
  - `metrics.json`
  - `confusion_matrix.csv`
  - `confusion_matrix.png` (if generated)
## Replication Spec: Football Match Outcome Prediction

### Source Paper
- `Forecasting_Football_Results_using_Machine_Learning_Techniques.pdf`
- Core task: 3-class football outcome classification (`home_win`, `draw`, `away_win`)

### Reproduction Objective
Recreate the paper's core supervised learning pipeline with both:
1. imbalanced training data (baseline), and
2. balanced training data via SMOTE.

Primary models:
- Random Forest
- XGBoost
- AdaBoost

### Implementation Scope
- Build a reproducible tabular ML pipeline in Python.
- Support configurable preprocessing and model hyperparameters.
- Produce standard metrics and confusion matrices for direct comparison.
- Persist all run artifacts (metrics JSON, confusion matrix CSV/PNG, config snapshot).

### Data Assumptions
Because the paper does not provide one exact reconstruction script, this project uses the following assumptions:
- Input format: single flat CSV where each row is a match.
- Required target column: `target` with values `home_win`, `draw`, `away_win`.
- Optional date column: `match_date` for temporal splits.
- Categorical and numerical columns are inferred from dtypes unless explicitly configured.
- If no date column is provided, use stratified random split.

### Preprocessing Policy
- Missing values:
  - numerical -> median imputation
  - categorical -> most frequent imputation
- Outliers:
  - numerical columns winsorized to configured quantile cap (default `0.95`)
- Encoding:
  - one-hot encoding for categorical columns
- Scaling:
  - min-max scaling for numerical features
- Feature selection:
  - RandomForest-based `SelectFromModel` to keep top-k features (default 20)

### Split & Reproducibility
- Default split: 80/20 train/test
- Seeded runs (`seed=42` default)
- Support 5-fold cross-validation on training set for stability estimates

### Evaluation
Metrics:
- Accuracy
- Macro precision, recall, F1
- Per-class precision/recall/F1
- Multi-class ROC-AUC (one-vs-rest, macro)
- Confusion matrix (counts)

### Replication Acceptance Criteria
Given likely differences in exact data versions and undocumented details, "successful replication" is defined as:
- Directional consistency: same top-performing model family in at least one major setting.
- Absolute gap tolerance:
  - accuracy within +/- 7 points of paper-reported values, OR
  - macro F1 within +/- 0.08.
- If outside tolerance, provide a documented gap analysis with at least 3 plausible causes.

### Paper-Reported Targets (for comparison)
- Imbalanced setting (table in paper):
  - RF: 54% accuracy
  - XGBoost: 58% accuracy
  - AdaBoost: 57% accuracy
- Balanced setting (SMOTE):
  - RF: 59% accuracy, 59% precision, 59% recall, 59% F1, ROC-AUC 61%
  - XGBoost: 55% accuracy
  - AdaBoost: 52% accuracy

### Known Ambiguities to Track in Analysis
- Inconsistent references to exact season ranges in the paper text.
- Potential mismatch in model lists between sections (mentions additional models in narrative).
- Unclear exact parameter grids and final selected hyperparameters for all models.
- Partial ambiguity on whether scaling occurs before/after certain feature selection steps.
