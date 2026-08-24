# Fraud Analysis AI — Model Development Notes

> Internal quick-reference for ML development decisions, experiments, and findings.

## 1. Dataset

**Dataset:** Kaggle IEEE-CIS Fraud Detection

Files:

* `train_transaction.csv`
* `train_identity.csv`

Initial dataset:

| Item                               |   Value |
| ---------------------------------- | ------: |
| Transaction rows                   | 590,540 |
| Transaction columns                |     394 |
| Identity rows                      | 144,233 |
| Identity columns                   |      41 |
| Fraud transactions                 |  20,663 |
| Legitimate transactions            | 569,877 |
| Fraud rate                         |  ~3.50% |
| Transactions without identity data | 446,307 |

After preprocessing/merging:

* Total model features: **432**
* Train: **413,378**
* Validation: **88,581**
* Test: **88,581**

Training class distribution:

* Negative: ~399K
* Positive: ~14K
* Significant class imbalance

---

## 2. Initial Model — XGBoost

### Why XGBoost

Selected as the first ML model because:

* Strong performance on tabular data
* Handles nonlinear feature interactions
* Works well with mixed numerical/categorical features
* Handles missing values effectively
* Supports class weighting
* Provides feature importance and integrates well with SHAP
* Suitable foundation for later fraud-investigation/explanation workflow

### Initial configuration

```text
objective              = binary:logistic
eval_metric            = aucpr
scale_pos_weight       = negative_count / positive_count
tree_method            = hist
n_estimators           = 4000
early_stopping_rounds  = 100
random_state            = 42
```

`scale_pos_weight` was calculated from the training-set class distribution to compensate for fraud/legitimate imbalance.

### Training result

Best iteration:

```text
3269
```

Best validation AUC-PR:

```text
0.864192
```

The 4,000-tree limit was intentionally used as an upper bound with early stopping rather than forcing the model to train all 4,000 trees.

---

## 3. Baseline Evaluation

Baseline model:

```text
Features: 432
Threshold: 0.5
```

Test-set results:

| Metric    | Result |
| --------- | -----: |
| AUC-ROC   | 0.9663 |
| AUC-PR    | 0.8657 |
| Precision | 0.9316 |
| Recall    | 0.7561 |
| F1        | 0.8347 |

Confusion matrix:

```text
TN = 85,310
FP = 172
FN = 756
TP = 2,343
```

AUC-ROC and AUC-PR are threshold-independent. Precision, recall, F1 and the confusion matrix depend on the classification threshold.

---

## 4. Threshold Analysis

Baseline XGBoost was evaluated across thresholds:

```text
0.1 → 0.9
```

General finding:

* Lower threshold → higher recall, more false positives
* Higher threshold → higher precision, more false negatives
* F1 was strongest around **0.4–0.5** in the baseline experiments
* Threshold selection will ultimately depend on the fraud-investigation workflow and business cost of false positives vs false negatives

Current threshold experiments are exploratory. The final operating threshold will be selected as part of the investigation workflow rather than treated as a permanent model property.

---

## 5. SHAP Analysis

SHAP was used for two different purposes:

### Global SHAP

Used to understand which features contribute most strongly across the dataset and to support feature-selection experiments.

Current global SHAP analysis was performed on the validation data.

Top influential features included:

```text
TransactionDT
TransactionAmt
card1
card2
P_emaildomain
addr1
C13
C1
id_31
card6
C14
D15
dist1
M4
C11
...
```

### Local SHAP

Will be used later during the investigation workflow to explain an individual transaction.

Conceptually:

```text
Global SHAP → Which features matter across the model?
Local SHAP  → Why did this particular transaction receive this prediction?
```

Local SHAP is therefore part of the **final fraud explanation**, not the feature-selection process.

---

## 6. SHAP Feature-Importance Distribution

Initial global SHAP analysis across 432 features:

| SHAP importance | Number of features |
| --------------- | -----------------: |
| > 1.0           |                  3 |
| > 0.5           |                  9 |
| > 0.1           |                 57 |
| > 0.01          |                202 |
| > 0.001         |                350 |
| <= 0.001        |                 82 |

This distribution was used to define feature-selection experiments rather than arbitrarily selecting a fixed number of features.

---

## 7. SHAP-Based Feature Selection Experiments

Baseline:

```text
432 features
AUC-ROC = 0.9663
AUC-PR  = 0.8657
```

### Experiment 1 — Remove zero-SHAP features

```text
432 → 403 features
```

Result was effectively unchanged from baseline, indicating that the zero-contribution features could be removed without meaningful performance loss.

### Experiment 2 — Remove features with SHAP <= 0.001

```text
432 → 350 features
```

Results:

```text
AUC-ROC = 0.9673
AUC-PR  = 0.8665
Precision = 0.9731
Recall    = 0.6896
F1        = 0.8072
```

This produced a small improvement in ranking metrics while reducing the feature count by ~19%.

### Experiment 3 — Remove features with SHAP <= 0.01

```text
432 → 202 features
```

Results:

```text
AUC-ROC = 0.9671
AUC-PR  = 0.8673
Precision = 0.9754
Recall    = 0.6922
F1        = 0.8097
```

This produced the strongest AUC-PR observed so far while reducing the feature set by more than half.

### Experiment 4 — Remove features with SHAP <= 0.1

```text
432 → 57 features
```

Results:

```text
AUC-ROC = 0.9653
AUC-PR  = 0.8610
Precision = 0.9735
Recall    = 0.6880
F1        = 0.8062
```

Performance dropped, suggesting that the lower-importance features collectively contain useful information.

### Current observation

The **202-feature configuration** is currently the most interesting candidate:

```text
432 → 202 features
AUC-PR: 0.8657 → 0.8673
```

````text
Exploratory SHAP analysis: performed during initial XGBoost feature investigation. Existing results are retained and reused for the current XGBoost implementation. Future model-specific feature selection will be performed using development data rather than the final test set.
````

---

## 8. Current Conclusions

1. XGBoost is a strong initial model for this tabular fraud dataset.
2. Class imbalance requires explicit handling using `scale_pos_weight`.
3. AUC-PR is particularly useful because of the ~3.5% fraud rate.
4. Threshold selection is a separate decision from model ranking performance.
5. Removing very low/zero-SHAP features can reduce dimensionality without necessarily hurting performance.
6. Removing too many features eventually reduces performance.
7. The **202-feature model** currently provides the best AUC-PR among the tested feature sets.
8. Global SHAP is useful for feature analysis; local SHAP will later support transaction-level explanations.
9. Final threshold and final feature set should be selected after the remaining model-tuning/evaluation work.
10. The final XGBoost model will become the first ML component of the Fraud Analysis & Investigation workflow.

---

## 9. Next ML Steps

Current priority:

```text
Feature selection
        ↓
Hyperparameter tuning
        ↓
Final XGBoost training
        ↓
Final test evaluation
        ↓
Threshold selection
        ↓
Local SHAP investigation explanation
        ↓
Rule engine + investigation signals
        ↓
Commit stable ML foundation
```

The test set should remain reserved for final evaluation and should not be repeatedly used to drive feature-selection decisions.
## XGBoost Hyperparameter Tuning

Hyperparameter tuning was performed on the 202-feature dataset using validation AUC-PR as the primary metric.

* `max_depth` was tested from 4 to 12. Performance improved through depth 10, while depth 12 reduced performance.
* `max_depth=10` is currently selected.
* `n_estimators=4000` was sufficient. Increasing to 6000 produced no improvement; early stopping reached the same best iteration.
* `learning_rate` was tested at 0.1, 0.2, and 0.3. The current best result was obtained with **0.3**.
* Current best validation AUC-PR: **0.872673**
* Current best configuration:

    * `max_depth=10`
    * `learning_rate=0.3`
    * `n_estimators=4000`
    * `early_stopping_rounds=100`
  
### min_child_weight Experiment

* Tested `min_child_weight=5` against the default value of `1` while keeping other parameters fixed.
* `min_child_weight=1` → Validation AUC-PR: **0.872673**
* `min_child_weight=5` → Validation AUC-PR: **0.864146**
* Higher `min_child_weight` reduced performance, so the default **`min_child_weight=1` was retained**.
* An earlier run combined `learning_rate=0.2` with `min_child_weight=5`; it was not used for the parameter decision because it did not isolate `min_child_weight`.


Detailed experiment results are maintained in `xgboost_hyperparameter_experiments.csv`.

**Status:** Provisional best XGBoost configuration. Further tuning will focus on regularization/tree-growth parameters.

### Final XGBoost Threshold Selection

After finalizing the XGBoost hyperparameters, threshold selection was performed on the **validation set** using the final 202-feature model.

Thresholds from **0.1–0.9** were evaluated using Precision, Recall, and F1. The best validation F1 was achieved at **threshold 0.4**:

* Precision: **0.9280**
* Recall: **0.7652**
* F1: **0.8388**

**Decision:** Lock **0.4** as the final operating threshold.

The untouched test set was then evaluated once using this locked threshold. Detailed threshold results are stored in `xgboost_model_evaluation.csv`.

### Final XGBoost Result

XGBoost development is complete. The final model uses **202 features** after SHAP-based feature selection (`importance <= 0.01` removed).

**Final configuration**

* `max_depth = 10`
* `learning_rate = 0.3`
* `min_child_weight = 1`
* `n_estimators = 4000`
* `early_stopping_rounds = 100`
* Best iteration = `1709`
* Validation PR-AUC = `0.872673`
* Final threshold = `0.4`, selected using validation-set F1

**Final test results**

* ROC-AUC: **0.96849**
* PR-AUC: **0.87424**
* Precision: **0.93895**
* Recall: **0.75928**
* F1: **0.83961**

**Confusion matrix**

* TN = `85,329`
* FP = `153`
* FN = `746`
* TP = `2,353`

The test set was used only for the final evaluation after feature selection, hyperparameter tuning, early stopping, and threshold selection were completed. No further XGBoost tuning is planned.

Detailed experiment history remains in the CSV files under `analysis/`.
