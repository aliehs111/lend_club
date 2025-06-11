# Lending Club Loan Default Prediction

## Overview
Predicts loan defaults using Lending Club data (2007–2015, ~16% defaults). Compared neural network and gradient-boosting tree (GBT) models to maximize AUC and sensitivity.

## Data Preprocessing
- **Source**: `loan_data.csv` (9578 loans, 14 columns like FICO score, interest rate, loan purpose).
- **Feature Engineering** (in `notebooks/lend_club_EDA.ipynb`):
  - **Log-transformed `revol.bal` and `days.with.cr.line`**: These had skewed values (e.g., a few huge balances). I used a log transform to make them less lumpy, helping models learn better patterns.
  - **Created `cr_line_years`**: Turned `days.with.cr.line` into years (e.g., 3650 days ≈ 10 years) to make it easier for models to use credit history.
  - **Created `pub_rec_flag`**: Marked loans with public records (e.g., bankruptcies) as 1 (yes) or 0 (no) to flag risky borrowers clearly.
  - **Created `inquiry_rate`**: Calculated how often creditors checked credit, thinking frequent checks might signal risk.
  - **One-hot encoded `purpose`**: Changed loan purposes (e.g., "credit card") into 6 yes/no columns, like checkboxes, so models could understand them.
  - **Standardized numeric features**: Scaled numbers (e.g., FICO, interest rate) to have mean 0 and spread 1, so no feature overwhelms the model.
## Multicollinearity and Feature Testing
- **Multicollinearity Analysis** (in `notebooks/lend_club_EDA.ipynb`):
  - I used pandas to compute correlations among numeric features in `loan_data.csv` (e.g., FICO score, interest rate, loan amount). I generated a correlation matrix to identify features that were highly correlated, as multicollinearity can introduce redundant information and destabilize model performance.
  - I set a threshold of |r| = 0.8 for strong correlations, intending to drop one feature from any pair exceeding this value. No correlations surpassed this threshold (e.g., `int.rate` and `fico` were moderately correlated but below 0.8), so I retained all 14 features.


- **Inverse Feature Selection**:
  - Rather than excluding features, I tested reduced datasets containing only the most predictive features to evaluate whether simpler models could improve performance. I prioritized FICO score due to its strong association with defaults.
  - I created `loan_data_ready_fico_ir.csv` (FICO and interest rate), `loan_data_ready_fico_ir_inq.csv` (adding inquiries), and `loan_data_ready_minimal.csv` (FICO, interest rate, inquiries, and purpose flags) to focus on high-impact features, hypothesizing that fewer features might reduce noise.
  - In `notebooks/lend_club_model.ipynb`, I trained models on these datasets. My hypthesis was proven wrong because these reduced sets did not outperform `loan_data_ready_raw.csv` (all features, AUC 0.6874), suggesting additional features contributed valuable signal. So I just went forward with all the features.
- **Data Versions Tested**:
  - `loan_data_ready_raw.csv`: All features, scaled, my main dataset.
  - `loan_data_ready_log.csv`: Same but with log transforms for extra smoothness.
  - `loan_data_ready_minimal.csv`: Just FICO, interest rate, inquiries, and purpose flags for a simpler model.
  - `loan_data_ready_fico_ir.csv`: Only FICO and interest rate, super focused.
  - `loan_data_ready_fico_ir_inq.csv`: FICO, interest rate, and inquiries, a bit more info.

## Modeling
- **Neural Network**:
  - Architecture: 2-layer Dense (128→32), ReLU, Dropout (0.2/0.2), L2=1e-4.
  - Imbalance: Class weights (outperformed SMOTE in initial analysis so I sticked with that).
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