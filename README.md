# Lending Club Loan Default Prediction

## Overview
Predicts loan defaults using Lending Club data (2007–2015, ~16% defaults). Compared neural network and gradient-boosting tree (GBT) models to maximize AUC and sensitivity.

## Data Preprocessing
- **Source**: `loan_data.csv` (14 columns, 9578 rows).
- **Feature Engineering**:
  - Log-transformed `revol.bal` and `days.with.cr.line`.
  - Created `cr_line_years`, `pub_rec_flag`, `inquiry_rate`.
  - One-hot encoded `purpose` (6 categories).
  - Standardized numeric features.
- **Variants**:
  - `loan_data_ready_raw.csv`: All features, scaled.
  - `loan_data_ready_log.csv`: Includes log transforms.
  - `loan_data_ready_minimal.csv`: Top 3 features + purpose flags.
  - `loan_data_ready_fico_ir.csv`: FICO + interest rate.
  - `loan_data_ready_fico_ir_inq.csv`: FICO + interest rate + inquiries.

## Modeling
- **Neural Network**:
  - Architecture: 2-layer Dense (128→32), ReLU, Dropout (0.2/0.2), L2=1e-4.
  - Imbalance: Class weights (outperformed SMOTE).
  - Hyperparameters: Adam optimizer, batch size 256, up to 40 epochs, early stopping on sensitivity.
  - Best Results: AUC 0.6874, Sensitivity 0.6938, Specificity ~0.64 at threshold 0.15 (on `loan_data_ready_raw.csv`).
  - Note: Retraining attempts yielded lower AUC (~0.6722, Sensitivity 1.0000, Specificity ~0.0062) due to training instability.
- **Gradient-Boosting Tree**:
  - Model: HistGradientBoostingClassifier, tuned (`learning_rate=0.05`, `max_depth=3`, `max_iter=100`, `class_weight='balanced'`).
  - Best Results: AUC 0.6676, Sensitivity 0.9967, Specificity ~0.0329 at threshold 0.15.
- **Final Model**: Neural net chosen for higher AUC and balanced sensitivity/specificity, using best reported results (AUC 0.6874).

## Key Metrics
- AUC: 0.6874
- Sensitivity: 0.6938 at threshold 0.15
- Specificity: ~0.64
- Precision: ~0.25–0.35 (varies)
- F1: ~0.45 (varies)

## Usage
1. **Load Data**:
   ```python
   import pandas as pd
   data = pd.read_csv('../data/loan_data_ready_raw.csv')
   X = data.drop('not.fully.paid', axis=1)