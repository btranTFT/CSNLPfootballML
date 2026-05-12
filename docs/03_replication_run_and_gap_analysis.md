# Replication Run and Gap Analysis

## Run Summary
- Experiment entrypoint: `scripts/run_experiment.py`
- Config: `configs/replication.yaml`
- Sample data generation: `scripts/generate_sample_data.py`

## Current Status
- End-to-end replication pipeline is runnable.
- Result artifacts are generated in `results/` with per-model metrics and confusion matrices.

## Gap Notes
- Current committed runs use synthetic/sample data for pipeline validation.
- Final fidelity comparison against the original paper requires running on the intended real dataset with matching feature/label definitions.

## Planned Validation
- Re-run with final dataset.
- Compare model-level metrics against reported paper values.
- Document any reproducibility gap and likely causes (data differences, preprocessing assumptions, random seed effects).
## Replication Run Report and Gap Analysis

### Run Metadata
- Run ID: `replication_v1_20260509_142543`
- Config: `configs/replication.yaml`
- Command: `python scripts/run_experiment.py --config configs/replication.yaml`
- Dataset used for this run: `data/football_matches.csv` (generated via `scripts/generate_sample_data.py`)

### Observed Results (This Repository)

#### Imbalanced Setting (no SMOTE)
- Random Forest: accuracy `0.7688`, macro F1 `0.6992`, ROC-AUC `0.8923`
- XGBoost: accuracy `0.7896`, macro F1 `0.7569`, ROC-AUC `0.9162`
- AdaBoost: accuracy `0.7458`, macro F1 `0.7135`, ROC-AUC `0.8315`

#### SMOTE Setting
- Random Forest: accuracy `0.7792`, macro F1 `0.7579`, ROC-AUC `0.8924`
- XGBoost: accuracy `0.8167`, macro F1 `0.8020`, ROC-AUC `0.9229`
- AdaBoost: accuracy `0.6979`, macro F1 `0.6962`, ROC-AUC `0.8393`

### Paper vs Replication Snapshot

Paper-reported best:
- Imbalanced: XGBoost accuracy `0.58`
- SMOTE: Random Forest accuracy `0.59`

Current run best:
- Imbalanced: XGBoost accuracy `0.7896` (same model family ranking at top)
- SMOTE: XGBoost accuracy `0.8167` (different top model than paper)

### Replication Fidelity Assessment
- Directional match:
  - Imbalanced setting: aligned (XGBoost strongest in both paper and this run).
  - SMOTE setting: not aligned (paper favors RF, this run favors XGBoost).
- Absolute gap:
  - All observed accuracies are substantially higher than paper values.
  - Because this run uses synthetic data (not the paper's original dataset), this is expected and does not represent true numeric replication fidelity.

### Most Plausible Causes of Gaps
1. **Dataset mismatch**: this run uses synthetic sample data, while the paper uses a specific curated Kaggle + domain pipeline.
2. **Feature-space mismatch**: the paper references larger feature sets and additional contextual factors that are not reconstructed exactly.
3. **Split/protocol ambiguity**: paper text contains ambiguities around temporal scope and season partitioning.
4. **Hyperparameter ambiguity**: full search space and final selected settings are only partially specified.
5. **Toolchain drift**: differences in library versions and defaults can alter optimization behavior.

### Draw-Class Error Analysis (Key Insight)
- As in the paper narrative, draw-class performance remains weaker than home/away classes in several settings.
- SMOTE improved draw recall most clearly for AdaBoost but caused precision collapse, showing class-balancing trade-offs.
- XGBoost + SMOTE provided the best overall draw-class balance in this run.

### Next Step for True Replication
To complete fidelity-level replication against the paper claims:
- replace synthetic data with the actual target dataset,
- lock exact schema/feature engineering assumptions in `docs/02_replication_spec.md`,
- rerun the same pipeline and regenerate this report.
