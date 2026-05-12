# Demo Video Script (Draft)

## Duration Target
5-10 minutes

## Suggested Flow
1. Project goal and selected paper
2. Replication pipeline overview (data, features, models)
3. Key replication results from `results/`
4. Extension proposal (temporal LSTM/Transformer)
5. Next steps and limitations

## Speaker Split
- Benjamin Tran: sections 1-3
- Nicholas Ramirez-Ornelas: sections 4-5
## Demo Video Script (5–10 minutes)

### Segment 1 (0:00–0:45): Problem and Paper
- Introduce team and project title.
- State source paper and task: predict `home_win`, `draw`, `away_win`.
- Mention replication + extension objective.

### Segment 2 (0:45–2:00): Pipeline Walkthrough
- Show repository layout quickly.
- Explain config-driven pipeline (`configs/replication.yaml`).
- Explain preprocessing, feature selection, and model set.

### Segment 3 (2:00–3:30): Running the Project
- Show setup and run commands:
  - `python -m pip install -r requirements.txt`
  - `python scripts/generate_sample_data.py`
  - `python scripts/run_experiment.py --config configs/replication.yaml`
- Show generated outputs in `results/`.

### Segment 4 (3:30–5:30): Results and Analysis
- Present imbalanced vs SMOTE results.
- Highlight confusion matrices and draw-class behavior.
- Explain key replication gap reasons from `docs/03_replication_run_and_gap_analysis.md`.

### Segment 5 (5:30–7:00): Extension Proposal
- Present draw-aware extension hypothesis.
- Explain uncertainty features + cost-sensitive training + calibration.
- State measurable success criteria.

### Segment 6 (7:00–8:30): Wrap-up
- Summarize what was replicated successfully.
- Mention limitations and next steps using real target dataset.
- Confirm where all deliverables are located.

## Recording Checklist
- Ensure terminal commands are visible and readable.
- Show at least one metrics file and one confusion matrix image.
- Split narration across team members.
- Keep pace so each segment stays within its time window.
