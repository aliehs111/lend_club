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

**Multicollinearity analysis (2025-06-07)**  

- Added a cell and imported the raw csv again and inspects correlations
- Since no feature pair exceeded |r| > 0.8 on the raw variables, there is no need to drop any columns for multicollinearity
- made a new cell to do the datacleaning and save the cleaned csv file as READy_FILE variable so I can get it in my model notebook

- ran model again without SMOTE and just using class weights.  after researching what all this means, I'm understanding the following:
- - AUC ≈ 0.679 (better than SMOTE’s ~0.668, but still room above 0.70)

- - Sensitivity ≈ 0.596 & Specificity ≈ 0.635 at the 0.5 cutoff (accuracy 0.629)

- - At “working cutoff” 0.40 I get ~86% of defaulters but only approve ~37% of safe loans (accuracy 0.451)
 
  - - next steps to try and tune the hyperparameters sticking with class weights instead of smote just because it was slightly better.

**Hyperparameter tuning (2025-06-07)**  

- raw‐no‐log + class‐weights pipeline gives AUC = 0.6789. To push AUC higher (≥ 0.70) without losing too much specificity, I need to find a better set of hyperparameters.
  
- - As I strategize on what hyperparameters to tune to what values...I need to research this better on tensorflow documentation
 

**Which hyperparameters to try first:**  
1. **Layer sizes**  
   - First layer units: try 128, 64 (current), 32  
   - Second layer units: try 64, 32 (current), 16  
2. **Dropout rates**  
   - Pairs to test: (0.2, 0.2), (0.3, 0.2) (current), (0.4, 0.3)  
3. **L2 regularization strength**  
   - Test values: 1e-3, 1e-4 (current), 1e-5  
4. **Learning rate for Adam optimizer**  
   - Try: 1e-2, 1e-3 (default), 1e-4  

**Tuning approach:**  
- **Manual grid search:**  
  1. Copy `make_model(...)` to `make_model_v2(...)`, parameterized for these values.  
  2. In separate cells (or a simple loop), instantiate and train each variant with `class_weight` fixed.  
  3. Record for each trial:  
     - Test AUC  
     - Sensitivity & specificity at threshold = 0.40  
- **Documentation:**  
  - Add a “Hyperparameter Trials” section below this plan.  
  - For each trial, note the hyperparameters and the resulting metrics in a small table.  

**Next actions:**  
1. Modify `make_model` to use 128 units in the first layer (keep everything else constant).  
2. Run training & evaluation.  
3. Record the new AUC/sensitivity/specificity results.  
4. Repeat for the next hyperparameter.  
5. After a few trials, identify the best combination and lock it in as the final model.  


### Hyperparameter Trials Results

| first_layer | second_layer | dropout1 | dropout2 | test_auc | test_sensitivity |
|-------------|--------------|----------|----------|----------|------------------|
| 128         | 32           | 0.2      | 0.2      | 0.6838   | 0.5537           |
| 128         | 32           | 0.3      | 0.2      | 0.6816   | 0.5700           |
| 128         | 64           | 0.2      | 0.2      | 0.6812   | 0.5635           |
| 64          | 32           | 0.3      | 0.2      | 0.6802   | 0.5798           |
| 128         | 64           | 0.3      | 0.2      | 0.6801   | 0.5668           |
| 64          | 32           | 0.2      | 0.2      | 0.6772   | 0.5668           |
| 64          | 64           | 0.2      | 0.2      | 0.6768   | 0.5831           |
| 64          | 64           | 0.3      | 0.2      | 0.6760   | 0.6091           |

**Observations:**  
- Highest AUC = 0.6838 at 128→32 layers with (0.2, 0.2) dropout, but sensitivity only 55 %.  
- If we want higher sensitivity (e.g. ≥ 60 %), 64→64 with (0.3, 0.2) gives sens≈ 61 % at AUC ≈ 0.676.  
- The drop in AUC from 0.684 → 0.676 may be an acceptable trade-off for better recall.

**Next steps:**  
1. **Decide on the priority:** Is **maximizing AUC** or **raising sensitivity** more critical?  
2. Based on that, choose one configuration to refine further—e.g.  
   - If you want **max AUC**, stick with (128→32, 0.2/0.2).  
   - If you need **higher recall**, try (64→64, 0.3/0.2).  
3. **Tune additional hyperparameters** around that config:  
   - Try different **L2 strengths** (1e-3, 1e-5).  
   - Adjust **learning rate** (1e-2, 1e-4).  
4. **Re-run** the same training + evaluation loop and record results.

By capturing both the numerical outcomes and your decision logic, you’ll have clear documentation of how each trial influenced the model—and why you chose your final architecture.




### [2025-06-04] Final Model: 64→64, Dropout (0.3,0.2), Class Weights, Early Stopping

**Configuration:**  
- **Layers:** Dense(64) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(1)  
- **Regularization:** L2=1e-4 on both Dense layers  
- **Optimizer & LR:** Adam, lr=1e-3  
- **Imbalance handling:** `class_weight = {0: w0, 1: w1}`  
- **Training:** up to 40 epochs, EarlyStopping on `val_sensitivity` (patience=5)

**Training Dynamics (first 10 epochs):**  
Epoch 1: val_auc=0.6704, val_sensitivity=0.5191
Epoch 2: val_auc=0.6708, val_sensitivity=0.5660
Epoch 3: val_auc=0.6761, val_sensitivity=0.6298
Epoch 4: val_auc=0.6752, val_sensitivity=0.5787
Epoch 5: val_auc=0.6804, val_sensitivity=0.6468
Epoch 6: val_auc=0.6782, val_sensitivity=0.5787
Epoch 7: val_auc=0.6770, val_sensitivity=0.6000
Epoch 8: val_auc=0.6828, val_sensitivity=0.5787
Epoch 9: val_auc=0.6764, val_sensitivity=0.6213
Epoch10: val_auc=0.6783, val_sensitivity=0.5830


**Final Test Results (0.5 cutoff):**  
- **Test loss** = 0.6619  
- **Test AUC** = 0.6756  
- **Test Sensitivity** = 0.5993  
- **Test Accuracy** = 0.6289  (from previous accuracy cell)

**Threshold = 0.40 performance:**  
- **Sensitivity** = 0.857  
- **Specificity** = 0.374  
- **Accuracy** = 0.451

**Interpretation:**  
- Prioritized recall: final sensitivity ≈ 60% at the 0.5 cutoff, and ≈ 86% at the 0.40 cutoff.  
- AUC settled at ≈ 0.676, slightly below the 0.68–0.69 range seen in earlier grid trials—early stopping likely prevented overfitting but also capped peak performance.  
- Overall accuracy at ≈ 63%, reflecting the trade-off toward catching defaulters.


### [2025-06-07] Two-Feature Model: FICO + Interest Rate

**Features:**  
- `fico` (scaled)  
- `int.rate` (scaled)  
- Target: `not.fully.paid`

**Model pipeline:**  
- Class weights  
- Same 64→64 network with dropout (0.3,0.2), L2=1e-4  
- 20 epochs, batch=256

**Results:**  
- Test AUC = _\<fill in\>_  
- Test Sensitivity = _\<fill in\>_  
- At cutoff 0.40: sensitivity = _\<fill in\>_, specificity = _\<fill in\>_, accuracy = _\<fill in\>_

**Takeaway:**  
This shows how much joint signal FICO + interest rate capture. Based on these metrics, decide whether to add a third feature (e.g. `inq.last.6mths`) or pivot to richer engineered features.


### [2025-06-07] Three-Feature Model: FICO + Interest Rate + Inquiries

**Features:**  
- `fico`  
- `int.rate`  
- `inq.last.6mths`  
- Target: `not.fully.paid`

**Pipeline:**  
- Class weights  
- Same 64→64 network, Dropout(0.3,0.2), L2=1e-4  
- 20 epochs, batch=256

**Results:**  
- Test AUC = 
- Test Sensitivity = _ 
- At cutoff 0.40: sensitivity =

**Interpretation:**  
This reveals how much incremental gain “number of inquiries” provides. If AUC rises substantially (closer to 0.68+), we’ll continue adding the next top feature; if not, we’ll reconsider more advanced feature engineering (ratios, flags, interactions).



