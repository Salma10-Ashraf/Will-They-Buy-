# 🛒 Will They Buy? Online Shoppers Purchase Intention

> Predicting whether an online store visitor will make a purchase, using classical Machine Learning and a small Neural Network.

**🔴 Live demo:** https://salma-will-they-buy.streamlit.app

**Author:** Salma Ashraf | **Course:** AI for Business – Final Project

---

## 1. Business Problem

An online store receives thousands of visits every day, but only about **15% of visitors actually buy something**. The store owner cannot tell in advance who is likely to buy, so marketing budget (ads, discounts, emails) is spread over everyone, including visitors who will never convert.

**Goal:** build a model that looks at a visitor's browsing behavior during the session and predicts whether they will make a purchase (`Revenue = 1`) or not (`Revenue = 0`).

**Business value:**
- Show offers and discounts only to visitors who are likely to buy, or who are close to buying.
- Spend less on visitors who are unlikely to convert.
- Understand which behaviors (such as high page value or many product pages) lead to a sale.

This is a **binary classification** problem with **imbalanced classes**.

## 2. Dataset

- **Source:** [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI Machine Learning Repository), also available on Kaggle.
- **Size:** 12,330 sessions (12,205 after removing duplicates), 17 features + 1 target.
- **Target:** `Revenue` (True = purchase, False = no purchase), roughly 15% positive.
- **File in this repo:** `online_shoppers_intention.csv`

| Feature group | Columns |
|---|---|
| Pages visited | `Administrative`, `Informational`, `ProductRelated` |
| Time spent (seconds) | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` |
| Google Analytics metrics | `BounceRates`, `ExitRates`, `PageValues` |
| Time context | `SpecialDay`, `Month`, `Weekend` |
| Visitor / technical | `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |

## 3. Project Pipeline

The full work is in the notebook **`Will_They_Buy.ipynb`**.

1. **Data cleaning:** missing values check, removal of duplicate rows, outlier detection with the IQR method.
2. **Exploratory Data Analysis:** summary statistics, histograms, boxplots, class distribution, correlation heatmap, purchase rate by month and visitor type.
3. **Preprocessing:**
   - One-hot encoding for `Month` and `VisitorType`.
   - Train/test split **before** scaling (stratified) to avoid data leakage.
   - Outlier capping at the 99th percentile (limits learned from the training set only).
   - `StandardScaler` fitted on the training set only.
   - Class imbalance handled with class weights (and oversampling for the neural network).
4. **Feature engineering:** `TotalPages`, `TotalDuration`, `AvgTimePerPage`. Feature selection tested with `SelectKBest` (ANOVA F-test).
5. **Models trained and compared:**
   - Logistic Regression
   - Gradient Descent (Logistic Regression trained with `SGDClassifier`)
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Naive Bayes
   - Artificial Neural Network (MLP), as a deep learning bonus
6. **Hyperparameter tuning:** `GridSearchCV` (5-fold cross-validation, scored by F1) for Decision Tree and KNN.
7. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices and ROC curves.
8. **Deployment:** best model saved with `joblib`, served through a FastAPI API and a Streamlit web app.

## 4. Results

Because only about 15% of visitors buy, **Accuracy alone is misleading** (a model that always predicts "no purchase" would already reach about 84%). Recall, F1 and ROC-AUC are the metrics that matter here.

### Final model: Decision Tree (Tuned)

| Metric | Score |
|---|---|
| Accuracy | 0.857 |
| Recall (buyers) | 0.853 |
| F1-score (buyers) | 0.651 |
| ROC-AUC | **0.929** |

The model finds about **85% of the visitors who actually buy**, and ranks a random buyer above a random non-buyer 93% of the time.

### ROC-AUC of all models

| Model | ROC-AUC |
|---|---|
| **Decision Tree (Tuned)** | **0.929** |
| Decision Tree | 0.922 |
| Logistic Regression | 0.912 |
| Artificial Neural Network | 0.899 |
| Gradient Descent | 0.891 |
| Naive Bayes | 0.834 |
| KNN | 0.820 |
| KNN (Tuned) | 0.810 |

### Why this model?
- Highest ROC-AUC and a high recall, which matters most to a store owner who does not want to miss buyers.
- Simpler than the untuned tree (`max_depth=5`, `min_samples_leaf=20`), so it is less likely to overfit.
- KNN has a high accuracy but a low recall (it misses most buyers), which is a good example of why accuracy is not enough.
- The ANN performed well, but did not beat the tree. This is common on small tabular datasets (about 12k rows).

### Key findings
- `PageValues` is the strongest indicator of a purchase.
- The engineered features `TotalPages` and `TotalDuration` are more correlated with purchase than any single page-count column.
- Feature selection did not change performance, so all features were kept.
- Most outliers are real behavior (for example, visitors with a high page value), so they were capped rather than removed.

## 5. Project Structure

```
.
├── app.py                          # Streamlit web app
├── api.py                          # FastAPI service
├── predictor.py                    # Shared preprocessing and prediction logic
├── requirements.txt
├── online_shoppers_intention.csv   # Dataset
├── Will_They_Buy.ipynb             # Full analysis, modeling and evaluation
└── model/
    ├── model.joblib                # Trained model
    ├── scaler.joblib               # Fitted StandardScaler
    ├── columns.joblib              # Feature names and order
    └── cap_limits.joblib           # Outlier capping limits
```

## 6. Installation and Usage

**Requirements:** Python 3.12 (tested) and the packages in `requirements.txt`. The model was trained with `scikit-learn==1.6.1`, so use the same version.

```bash
git clone https://github.com/Salma10-Ashraf/Will-They-Buy-.git
cd Will-They-Buy-
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
python -m streamlit run app.py
```
Then open http://localhost:8501. Use the **Quick examples** buttons in the sidebar to try a likely buyer and a casual browser.

### Run the API
```bash
python -m uvicorn api:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Administrative": 3, "Administrative_Duration": 90, "Informational": 1,
       "Informational_Duration": 40, "ProductRelated": 45, "ProductRelated_Duration": 2000,
       "BounceRates": 0.0, "ExitRates": 0.015, "PageValues": 25.0, "SpecialDay": 0.0,
       "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1,
       "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": false}'
```

Example response:
```json
{"will_buy": true, "probability": 0.97, "label": "Likely to buy"}
```

### Re-run the notebook
Open `Will_They_Buy.ipynb` in Jupyter or Google Colab, place `online_shoppers_intention.csv` next to it, and run all cells.

## 7. Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, FastAPI, Streamlit, joblib.

## 8. Future Improvements

- Try ensemble models such as Random Forest and Gradient Boosting.
- Tune the decision threshold according to the store's cost of a missed buyer versus a wasted offer.
- Add explanations for each prediction (for example, SHAP values).
