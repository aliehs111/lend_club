# Project Notes: Lending Club Loan Data Analysis

## 2025-05-23
### Initial Scaffold
- Created project structure:
lending_club/
├── data/ # raw CSVs
├── notebooks/ # Jupyter notebooks
│ └── 01_EDA.ipynb
├── src/ # (future) Python scripts
├── .gitignore
└── README.md

- Initialized Git repository and pushed initial commit (`initial scaffold`).

### Initial Data Analysis (`adds initialdata analysis`)
- In `01_EDA.ipynb`:
1. Loaded `loan_data.csv`.
2. Ran `df.info()` and `df.describe()` to inspect dtypes, non-null counts, summary statistics.
3. Checked target balance:  
   ```
   not.fully.paid 
   0    0.839946 
   1    0.160054 
   ```
4. Examined `purpose.value_counts()` to confirm one-hot encoding would be feasible.
5. Calculated skew for `revol.bal` (~11.16) and `days.with.cr.line` (~1.16), identified them as heavily skewed.
6. Created histograms to visualize distributions.

---

## 2025-05-28
### Baseline Model Running Without Errors (`baseline model running without errors. ready for tuning. saved transformed data in new csv ready for model.`)
- Completed EDA transformations in `01_EDA.ipynb`:
1. **Log-transforms**  
   - `revol_bal_log = log1p(revol.bal)` (skew dropped from ~11 to ~–2.21)  
   - `cr_line_years = days.with.cr.line / 365`  
   - `cr_line_years_log = log1p(cr_line_years)` (skew dropped from ~1.16 to ~–0.49)  
2. **Engineered features**  
   - `pub_rec_flag = (pub.rec > 0) ? 1 : 0`  
   - `inquiry_rate = inq.last.6mths / cr_line_years`  
3. **One-hot encoding** for `purpose` (6 dummy columns).  
4. **Standard scaling** for numeric columns:  
   ```
   ['revol_bal_log', 'cr_line_years_log', 'int.rate', 'installment',
    'log.annual.inc', 'dti', 'fico', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'inquiry_rate']
   ```  
   Verified means ≈ 0, stds ≈ 1.
5. Dropped raw `revol.bal` and `days.with.cr.line`.
6. Saved processed DataFrame as `loan_data_ready_log.csv`.

- Built baseline in `02_Modeling.ipynb`:
1. **Loaded** `loan_data_ready_log.csv`.
2. **Defined features** (all columns except `not.fully.paid`) and **target**.
3. **Train/test split** (80/20 stratified).
4. **Applied SMOTE** on training data → resampled to 50/50.
5. **Casted** all arrays to `float32` for TensorFlow.
6. **Model architecture**:  
   ```
   Input → Dense(64, relu, L2=1e-4) → Dropout(0.3)
         → Dense(32, relu, L2=1e-4) → Dropout(0.2)
         → Dense(1, sigmoid)
   ```
   - Loss: `binary_crossentropy`  
   - Optimizer: Adam  
   - Metrics: `AUC(name='auc')`, `Recall(name='sensitivity')`
7. **Trained** for 20 epochs, batch size 256 (validation_split=0.2).
8. **Baseline results (0.5 cutoff)**:  
   - Test loss: 0.5340  
   - Test AUC: 0.6528  
   - Test Sensitivity: 0.3127  
   - Test Accuracy: 0.7620  
9. Saved modeling notebook; ready for tuning.

### Environment & Reproducibility (`adds environment.yml and requirements.txt for full reproducibility of environment`)
- Exported Conda environment to `environment.yml`.  
- Created `requirements.txt` via `pip freeze`.  
- Both files committed for reproducibility.

### Gitignore Updates (`updates gitignore with editor and OS stuff`)
- Expanded `.gitignore` to include:
pycache/
*.py[cod]
.ipynb_checkpoints/
.env
.vscode/
.DS_Store
venv/
.venv/
---

*2025-06-02 Next Steps:*  
- Threshold analysis and plotting ROC/PR curves.  
- Compare SMOTE vs. class weights.  
- Hyperparameter tuning.  

**Threshold Analysis (2025-05-28)**  
I ran a sweep of probability thresholds on the test set to understand the trade-off between catching defaulters (sensitivity) and correctly approving non-defaulters (specificity). Here are the results:

| Threshold | Sensitivity | Specificity | Accuracy |
|-----------|-------------|-------------|----------|
| 0.10      | 0.993       | 0.060       | 0.209    |
| 0.15      | 0.964       | 0.121       | 0.256    |
| 0.20      | 0.951       | 0.193       | 0.314    |
| 0.25      | 0.896       | 0.270       | 0.371    |
| 0.30      | 0.818       | 0.366       | 0.438    |
| 0.35      | 0.723       | 0.482       | 0.520    |
| 0.40      | 0.603       | 0.617       | 0.615    |
| 0.45      | 0.427       | 0.745       | 0.694    |
| 0.50      | 0.313       | 0.848       | 0.762    |

- **Threshold = 0.10**  
  - Sensitivity = 0.993 → almost all defaulters caught  
  - Specificity = 0.060 → almost all non-defaulters flagged as default  
  - Accuracy = 0.209 → poor overall because too many false positives  

- **Threshold = 0.25**  
  - Sensitivity = 0.896 → catch ~90% of defaulters  
  - Specificity = 0.270 → approve only ~27% of safe loans  
  - Accuracy = 0.371 → still low due to many false positives  

- **Threshold = 0.35**  
  - Sensitivity = 0.723 → catch ~72% of defaulters  
  - Specificity = 0.482 → approve ~48% of safe loans  
  - Accuracy = 0.520 → modest overall performance  

- **Threshold = 0.40**  
  - Sensitivity = 0.603 → catch ~60% of defaulters  
  - Specificity = 0.617 → approve ~62% of safe loans  
  - Accuracy = 0.615 → balanced trade-off  

- **Threshold = 0.45**  
  - Sensitivity = 0.427 → catch ~43% of defaulters  
  - Specificity = 0.745 → approve ~75% of safe loans  
  - Accuracy = 0.694 → higher overall accuracy but fewer defaulters caught  

- **Threshold = 0.50**  
  - Sensitivity = 0.313 → catch ~31% of defaulters  
  - Specificity = 0.848 → approve ~85% of safe loans  
  - Accuracy = 0.762 → highest accuracy but misses most defaulters  

**Chosen cutoff: 0.40**  
At threshold = 0.40, I catch about 60% of defaulters while still approving ~62% of safe loans (accuracy = 0.615). This feels like the best balance between flagging true defaults and minimizing false alarms.  

**ROC & PR Analysis (2025-05-28)**  
- ROC AUC ≈ 0.653 (moderate ranking ability).  
- At recall ≈ 0.60, precision ≈ 0.25 on the PR curve (i.e., 1 in 4 flagged loans truly default).  
- This reaffirms that threshold = 0.40 trades sensitivity for a tolerable level of false positives.

## [2025-06-03] Hyperparameter Tuning on Class-Weights Pipeline

- **Pipeline chosen:** Raw-No-Log + Class Weights (AUC = 0.6874).
- **Tuning goal:** Push AUC ≥ 0.70 while keeping specificity ≥ 0.50 at threshold 0.40.

### Trial 1: Larger First Layer (128→64→1)
- **Architecture changes:**  
  - Dense(128, relu, L2=1e-4) → Dropout(0.3)  
  - Dense(64,  relu, L2=1e-4) → Dropout(0.2)  
  - Dense(1,   sigmoid)
- **Results (0.5 cutoff):**  
  - Test AUC       = 0.6921  
  - Test Sensitivity = 0.6904  
  - Test Accuracy   = 0.6050
- **Threshold 0.40:**  
  - Sensitivity = 0.865  
  - Specificity = 0.345  
  - Accuracy    = 0.429
- **Notes:**  
  - Slight AUC bump, but specificity remains low.

### Trial 2: Increased L2 (1e-3) on Same Architecture
- **Architecture:**  
  - Same 64→32→1 layers, but L2 regularization = 1e-3 instead of 1e-4.
- **Results (0.5 cutoff):**  
  - Test AUC       = 0.6965  
  - Test Sensitivity = 0.6782  
  - Test Accuracy   = 0.6123
- **Threshold 0.40:**  
  - Sensitivity = 0.842  
  - Specificity = 0.360  
  - Accuracy    = 0.444  
- **Notes:**  
  - AUC improved again; specificity slightly better.

