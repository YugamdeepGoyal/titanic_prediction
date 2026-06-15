# Titanic Survival Prediction

## Overview
A machine learning project to predict passenger survival on the Titanic using Logistic Regression and Support Vector Machine (SVM). The project focuses on building a clean, modular, and leak-free ML pipeline with proper preprocessing, feature engineering, and hyperparameter tuning.

---

## Project Structure
```
titanic_prediction/
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── svm_model.pkl
│
├── images/                        
├── pipelines.py                   
├── eda.ipynb                      
├── logistic_regression.ipynb      
├── svm.ipynb                      
└── README.md
```

---

## Dataset
- **Source:** [Titanic3 Dataset on Kaggle](https://www.kaggle.com/datasets/vinicius150987/titanic3) --> available in public domain
- **Size:** 1309 passengers, 14 features
- **Target:** `survived` (0 = not survived, 1 = survived)
- **Class distribution:** ~62% not survived, ~38% survived

---

## Installation

```bash
git clone https://github.com/YugamdeepGoyal/titanic_prediction.git
cd titanic_prediction
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Approach

### 1. Exploratory Data Analysis (`eda.ipynb`)
- Analyzed missing values, distributions, and survival rates across all features
- Key findings:
  - **Sex** — strongest predictor, female survival rate ~74% vs male ~19%
  - **Pclass** — clear survival hierarchy, 1st class ~63%, 2nd ~47%, 3rd ~24%
  - **Age** — children survived more, elderly survived less
  - **Fare** — highly skewed with extreme outliers, strong correlation with pclass and survival
  - **Family size** — solo travelers and very large families survived less, small families survived more
- VIF analysis confirmed multicollinearity between `sibsp` and `parch` — combined into single `family` feature
- `ticket_shared` (passengers sharing same ticket) tested — VIF of 8.7 and no accuracy improvement confirmed it adds noise, dropped

### 2. Feature Engineering (`pipelines.py`)
- Extracted **titles** from passenger names
- Created **family** = sibsp + parch
- Created **family_binned** (alone / small / large) from family size
- Added **age bins** alongside raw age (child / teen / adult / middle / senior). The values of age bins are taken from internet.
- Dropped columns upfront: `boat`, `body`, `home.dest`
- Dropped inside pipeline: `cabin` (77% missing), `name`, `ticket`

### 3. Preprocessing Pipeline (`pipelines.py`)
All preprocessing happens **inside cross-validation folds** — zero data leakage.


### 4. Class Imbalance
- Handled with **SMOTE** inside the IMBPipeline after preprocessing
- SMOTE runs inside each CV fold — no synthetic samples leak from validation into training

### 5. Hyperparameter Tuning
- Used GridSearchCV
- CV = 3 folds for both Logistic Regression and SVM
- Scoring metric: accuracy

---

## Models and Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~83% |
| SVM (linear kernel) | ~82% |

---

## Confusion Matrix (Logistic Regression, threshold=0.5)
```
                  Predicted 0    Predicted 1
Actual 0              136           26
Actual 1               19            81

Accuracy:  ~83%
Precision: ~75%
Recall:    ~81%
F1 Score:  ~78%
ROC-AUC:   ~0.82
```

---

---

## Future Work
- Implement RandomForest and XGBoost
- More aggressive feature engineering
- Deploy as a web app with Flask or Streamlit
---

This is an educational project.