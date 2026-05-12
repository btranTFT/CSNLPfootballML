# Extension Plan

## Proposed Extension
Introduce temporal deep learning for sequential match prediction:
- LSTM-based sequence model
- Transformer-based sequence model

## Rationale
Football outcomes depend on temporal context (form streaks, injuries, lineup shifts, momentum), which sequence models can capture better than static classifiers.

## High-Level Implementation Plan
- Build per-team rolling sequences from prior matches (e.g., last 5-10 games).
- Train sequence models to predict match outcome probabilities (win/draw/loss).
- Evaluate with chronological splits to avoid temporal leakage.

## Comparison Plan
- Baselines: Random Forest, XGBoost, AdaBoost pipeline.
- Compare on identical time-based splits using accuracy, macro-F1, and calibration-aware metrics.

## Risks
- Overfitting due to limited data.
- Increased training complexity.
- Sensitivity to sequence length and feature schema.
## Extension Proposal: Draw-Aware Robustness Enhancement

### Motivation
The replication and paper narrative both indicate that the `draw` class is consistently harder to predict than `home_win` and `away_win`. This suggests a modeling blind spot around balanced matches.

### Extension Hypothesis
Adding **draw-aware confidence calibration and uncertainty features** will improve draw-class F1 without significantly harming overall accuracy.

### Proposed Method (Future Work)
1. **Uncertainty Features**
   - Add parity indicators derived from existing inputs (for example, absolute differences in attack/defense/form: `|home_form-away_form|`, `|home_attack-away_attack|`).
   - Rationale: draw outcomes are more likely when team-strength signals are near equilibrium.

2. **Cost-Sensitive Training**
   - Apply class weighting to penalize draw misclassification more than dominant classes.
   - For model families that support sample weighting, assign higher weight to `draw` samples.

3. **Probability Calibration**
   - Use post-hoc calibration (`isotonic` or `sigmoid`) on the best base model.
   - Evaluate whether calibrated probabilities improve draw thresholding and decision quality.

4. **Decision Rule Variant**
   - Instead of plain argmax, use a draw-sensitive decision rule:
     - predict `draw` when calibrated draw probability exceeds threshold `tau_draw`;
     - otherwise use standard class argmax.

### Experimental Plan
- Baseline: best current model from replication pipeline (currently XGBoost + SMOTE on sample data).
- Extension variants:
  - V1: baseline + uncertainty features
  - V2: V1 + cost-sensitive weighting
  - V3: V2 + calibration + draw-threshold decision rule
- Data protocol:
  - keep same split strategy and seeds for fair comparison.
- Primary target metric:
  - `draw` class F1
- Secondary metrics:
  - macro F1, overall accuracy, ROC-AUC, confusion matrix shift.

### Success Criteria
- Minimum acceptable gain:
  - `draw` F1 improves by >= 0.05 absolute over baseline.
- Guardrail:
  - overall accuracy drop <= 0.02.
- If gains fail:
  - report negative result with diagnostic error slices (by league, weather, parity bins).

### Risks and Mitigations
- **Risk**: draw overprediction after weighting/calibration.
  - **Mitigation**: tune `tau_draw` on validation set and track precision-recall balance.
- **Risk**: extension gains are data-specific.
  - **Mitigation**: repeat with alternate seed and temporal split to test robustness.

### Paper Integration
Use this content in the EMNLP section:
- `Extension and Future Work`
- include hypothesis, rationale, protocol, and explicit success/failure criteria.
