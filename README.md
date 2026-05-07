# Customer Churn Prediction Model

## 1. Project Objective
The primary goal of this project is to develop a machine learning model to predict bank customer churn (a binary classification task where Exited = 1). 

The business objective is to proactively identify customers at risk of leaving to implement targeted retention strategies. Consequently, the model is evaluated primarily on **Recall (Sensitivity)** for the churn class, aiming to minimize False Negatives (missed churners).

## 2. Initial Data Brainstorming & Hypotheses
Before modeling, an analysis of the dataset's features was conducted to hypothesize potential churn drivers.

**Key Feature Hypotheses:**
*   **Customer Engagement:** `IsActiveMember` is strongly predictive (inactive members have a high probability of leaving). `Complain` and `Satisfaction Score` were initially considered strong but later identified as data leakage.
*   **Banking Relationship:** `NumOfProducts` is highly predictive but non-linear (1 product = low stickiness, 2 = stable, 3-4 = higher churn rate). `Balance`, `Tenure`, and `Age` are also key indicators of loyalty and investment in the bank.
*   **Demographics & Financials:** `Geography` (e.g., Germany) and extreme `CreditScore` values are potentially predictive. `Point Earned` was flagged as potential data leakage.

**Proposed Feature Engineering:**
*   `TenureToAgeRatio`: To capture the "life-long customer" persona.
*   `BalanceToSalaryRatio`: To measure financial health.
*   `IsTransactionalCustomer`: A flag for credit card-only users (zero balance, has credit card).
*   `CreditScoreTier`: Binning scores to capture trends more easily.

## 3. Model V1: Initial Build & Leakage Discovery
The first model utilized an XGBoost classifier within a pipeline featuring GridSearchCV for hyperparameter tuning, optimized for Recall.

*   **Results:** The model achieved near-perfect scores (99.8% cross-validation recall and 99.5% test recall).
*   **Diagnosis:** These results pointed to severe data leakage. 
*   **Root Cause:** The `Complain` feature had >66% importance, and related features contributed another ~23%. The model learned an unrealistic rule (IF Complain == 1 THEN Exited = 1), which fails to predict churners *before* they complain.

## 4. Model V1.1: The Realistic Baseline
To address the leakage, a revised baseline model was created.

*   **Actions Taken:** The leaky columns (`Complain`, `Satisfaction Score`, and `Point Earned`) were completely removed. The exact same XGBoost pipeline was re-run.

**V1.1 Performance Summary:**
*   **Best Cross-Validation Recall:** 79.14%
*   **Final Test Set Recall:** 80.88%
*   **Final Test Set Precision (Churners):** 47.00%
*   **Final Test Set Accuracy:** 77.60%

**Overfitting Analysis:**
The model demonstrated excellent generalization. The gap between training recall (82.3%) and CV recall (79.1%) was minimal. Learning and validation curves confirmed healthy convergence and optimal `max_depth`.

**Top 5 Predictive Features (V1.1):**
1.  `NumOfProducts` (19.0%)
2.  `Age` (17.7%)
3.  `IsActiveMember` (12.0%)
4.  `Geography_Germany` (8.8%)
5.  `Gender_Male` (6.1%)

*Conclusion: This trustworthy baseline successfully identifies ~81 out of every 100 churners.*

## 5. Future Work: Roadmap for V2 and V3
While V1.1 is a solid start, future iterations will focus on addressing class imbalance and extracting more signal from the data.

### Model V2: Advanced Imbalance Handling
*   **Objective:** Improve Recall and Precision by training on a better-quality, balanced dataset.
*   **Proposed Technique:** Implement a hybrid resampling strategy using **SMOTEENN** (SMOTE to create synthetic minority examples, ENN to clean noisy samples near the class boundary).
*   **Implementation:** This will be executed within an `ImbPipeline` to prevent data leakage during cross-validation. The XGBoost `scale_pos_weight` parameter will be removed.

### Model V3: Deeper Feature Engineering & Alternative Models
*   **Objective:** Squeeze additional performance through complex feature interactions and alternative algorithms.
*   **Proposed Feature Engineering:** Create interaction terms (e.g., `Age * NumOfProducts`, `Balance * IsActiveMember`) and explore polynomial features for non-linear transformations.
*   **Alternative Models:** Test LightGBM (for speed/efficiency) or TabNet (deep learning) to see if different architectures better capture data patterns.
*   **Ensembling:** Combine predictions of the best models using stacking or voting to create a robust final classifier.
