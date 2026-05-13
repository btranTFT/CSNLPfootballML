# EMNLP-Style Paper Draft (Working Placeholder)

This file is a placeholder index for the report drafting process.

## Draft Sections
- Abstract
- Introduction
- Related Work
- Background
- Replication
- Extension and Future Work
- Conclusion

## Notes
- The formal submission artifact is expected as a PDF in EMNLP format.
- Current exported draft location: `paper/CS5170_Project_Paper.pdf`.
## Title
Replication and Draw-Aware Extension Plan for Football Outcome Prediction with Classical Machine Learning

## Abstract (Draft: ~170 words)
This project replicates and analyzes the paper *Forecasting Football Results Using Machine Learning Techniques* (IEEE ICAISD 2025), which predicts three-class football outcomes (home win, draw, away win) using Random Forest, XGBoost, and AdaBoost with and without SMOTE balancing. We implement a reproducible end-to-end pipeline covering preprocessing, feature engineering, class balancing, model training, and evaluation with accuracy, macro-F1, ROC-AUC, and confusion matrices. In a validated sample-data run, XGBoost outperforms the other models in both imbalanced and SMOTE settings, while draw-class prediction remains the most challenging in several variants. Compared to the source paper, our run aligns directionally in the imbalanced setting but differs in SMOTE ranking, highlighting sensitivity to dataset composition and implementation assumptions. We document ambiguity points and provide a structured gap analysis. As future work, we propose a draw-aware extension that combines parity-based uncertainty features, cost-sensitive training, and probability calibration to improve draw-class F1 while preserving overall accuracy. The repository emphasizes reproducibility with pinned dependencies, config snapshots, and run artifacts.

## 1. Introduction
Football outcome prediction remains a difficult multi-class classification problem due to outcome imbalance and contextual uncertainty. The replicated source paper studies this task with tree-based and boosting models, reporting moderate accuracy improvements and a consistent challenge in predicting draws. Our project contribution is threefold: (1) a clean, reproducible reimplementation pipeline; (2) transparent replication-gap reporting; and (3) a non-trivial extension plan focused on draw-class robustness.

## 2. Related Work
Sports forecasting has historically used statistical models, but machine learning methods have become dominant for tabular match prediction tasks. Prior works demonstrate utility of ensemble models and boosting methods, while repeatedly noting limitations from data drift, sparse events, and class imbalance. Beyond sports, ML reproducibility studies show that benchmark claims can shift materially with preprocessing changes, split strategy, and implementation details. Our framing follows this line by emphasizing reproducibility and explicit uncertainty around replication claims.

## 3. Background
The source paper applies Random Forest, XGBoost, and AdaBoost to predict match outcomes and evaluates both imbalanced and SMOTE-balanced settings. The reported conclusion is that Random Forest under SMOTE provides the best aggregate metrics in that study. Core preprocessing in the paper includes missing-value handling, feature encoding/scaling, and feature-selection steps.

## 4. Replication Methodology
### 4.1 Pipeline
- Missing values: median (numeric), mode (categorical)
- Outlier capping: winsorization (default 95th percentile)
- Encoding: one-hot categorical features
- Scaling: min-max normalization
- Feature selection: RandomForest-based top-k selector
- Models: Random Forest, XGBoost, AdaBoost
- Two settings: no SMOTE and SMOTE

### 4.2 Reproducibility Controls
- Configuration-driven runs (`configs/replication.yaml`)
- Fixed random seed (default 42)
- Config snapshot persisted per run
- Automated export of metrics and confusion matrices

## 5. Replication Results
In a full sample-data run, XGBoost achieves the highest accuracy in both settings. Draw-class metrics improve under some SMOTE/model combinations but remain less stable than home/away classes in others, consistent with the paper's qualitative observation that draw outcomes are harder to model.

## 6. Gap Analysis vs Original Paper
Observed differences from paper-reported numbers are expected because this validated run uses synthetic data to verify the code path and reproducibility workflow. Key sources of divergence include dataset mismatch, feature-space differences, split ambiguity, and partial hyperparameter details in the source paper narrative. We document these in `docs/03_replication_run_and_gap_analysis.md`.

## 7. Extension and Future Work
We propose a draw-aware extension with three components:
1. parity-based uncertainty features,
2. cost-sensitive training emphasizing draw misclassification,
3. post-hoc calibration plus a draw-sensitive decision threshold.

Primary success metric: draw-class F1 improvement by >= 0.05 with <= 0.02 total accuracy drop.

## 8. Conclusion
This project delivers a replication-ready framework and transparent error analysis for football outcome prediction. While exact numeric replication requires the final target dataset and strict protocol alignment, the implemented pipeline supports rapid iteration and principled reporting. The proposed extension offers a concrete path to improving the most error-prone class.

## References (Draft placeholders to finalize in EMNLP format)
1. Armanda, M. G., & Miranda, E. (2025). Forecasting Football Results Using Machine Learning Techniques.
2. Bunker, R., & Thabtah, F. (2019). A machine learning framework for sport result prediction.
3. Breiman, L. (2001). Random Forests.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
5. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
6. Decroos, T. J., et al. (2019). Actions speak louder than goals.
7. Sjoberg, F. (2023). Match prediction using machine learning.
8. Wright, R. (2020). Integrating qualitative data in sports analytics.
9. Van Buuren, S. (2018). Flexible Imputation of Missing Data.
