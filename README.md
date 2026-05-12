# CSNLP Final Project: Replication and Extension

Replication target:
`Forecasting_Football_Results_using_Machine_Learning_Techniques.pdf`

## Team Members
- Benjamin Tran
- Nicholas Ramirez-Ornelas

## Paper Reference
- Muhammad Gustya Armanda and Eka Miranda. "Forecasting Football Results Using Machine Learning Techniques." IEEE ICAISD 2025.

## Project Structure
- `configs/`: experiment configuration files
- `data/`: input datasets (local; not committed if large/private)
- `scripts/`: runnable scripts for data generation and experiments
- `src/`: modular preprocessing, modeling, and evaluation code
- `results/`: metrics, confusion matrices, and artifacts from full runs
- `docs/`: replication spec, risk memo, gap analysis, extension plan, paper draft, demo script

## Setup
```bash
python -m pip install -r requirements.txt
```

## Quickstart
1. Generate sample data:
```bash
python scripts/generate_sample_data.py
```

2. Run replication experiments (imbalanced + SMOTE):
```bash
python scripts/run_experiment.py --config configs/replication.yaml
```

3. Check outputs in:
- `results/<experiment_name>_<timestamp>/imbalanced/...`
- `results/<experiment_name>_<timestamp>/smote/...`

4. Export draft paper PDF:
```bash
python scripts/export_paper_pdf.py
```

## Config Notes
Default config is `configs/replication.yaml`:
- target column: `target`
- optional date column for temporal split: `match_date`
- models: RandomForest, XGBoost, AdaBoost
- settings: imbalanced and SMOTE both enabled

## Results Snapshot (Sample-Data Run)
Run ID: `replication_v1_20260509_142543`

| Setting | Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---:|---:|---:|
| Imbalanced | RandomForest | 0.7688 | 0.6992 | 0.8923 |
| Imbalanced | XGBoost | 0.7896 | 0.7569 | 0.9162 |
| Imbalanced | AdaBoost | 0.7458 | 0.7135 | 0.8315 |
| SMOTE | RandomForest | 0.7792 | 0.7579 | 0.8924 |
| SMOTE | XGBoost | 0.8167 | 0.8020 | 0.9229 |
| SMOTE | AdaBoost | 0.6979 | 0.6962 | 0.8393 |

## Reproducibility Checklist
- Dependency lock: `requirements.txt`
- Deterministic seed in config
- Config snapshot saved per run
- Per-model metrics in JSON
- Confusion matrices exported to CSV and PNG

## Deliverables in this Repo
- Replication spec: `docs/02_replication_spec.md`
- Run analysis: `docs/03_replication_run_and_gap_analysis.md`
- Extension design: `docs/04_extension_plan.md`
- EMNLP-style draft: `docs/05_emnlp_paper_draft.md`
- Exported PDF draft: `paper/emnlp_paper_draft.pdf`
- Demo script: `docs/06_demo_video_script.md`

## Checkpoints
- No model checkpoint files are committed to this repository by default.
- To reproduce results, run:
```bash
python scripts/generate_sample_data.py
python scripts/run_experiment.py --config configs/replication.yaml
```
- If checkpoints are later added, store them under `checkpoints/` and document exact filenames and loading commands here.

## Notes
- Current run uses synthetic sample data (`scripts/generate_sample_data.py`) to validate the end-to-end pipeline.
- For final course submission fidelity, replace with the intended real dataset and rerun.
