# Titanic Survival Prediction

## Overview
A machine learning project to predict passenger survival on the Titanic using Logistic Regression and Support Vector Machine (SVM). The project focuses on building a clean, modular, and leak-free ML pipeline with proper preprocessing, feature engineering, and hyperparameter tuning.

---

## Project Structure

```text
titanic_prediction/
│
├── models/
│   ├── artifacts/
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   └── y_test.pkl
│   ├── logistic_regression_model.pkl
│   └── svm_model.pkl
│
├── images/                        # EDA and evaluation plots
├── pipelines.py                   # Shared preprocessing pipelines
├── eda.ipynb                      # Exploratory data analysis
├── logistic_regression.ipynb      # Logistic Regression training
├── svm.ipynb                      # SVM training
├── testing.ipynb                  # File in which .pkl files are loaded and models are tested
└── README.md
```

---

## Dataset

- **Source:** Titanic3 Dataset on Kaggle
- **Size:** 1,309 passengers, 14 features
- **Target:** `survived`
  - `0` = Did Not Survive
  - `1` = Survived
- **Class Distribution:**
  - ~62% Not Survived
  - ~38% Survived

---

## Installation

```bash
git clone https://github.com/YugamdeepGoyal/titanic_prediction.git
cd titanic_prediction

python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
kagglehub
joblib
openpyxl
```

---

## Approach

### 1. Exploratory Data Analysis (`eda.ipynb`)

Performed analysis of:

- Missing values
- Feature distributions
- Correlations
- Survival patterns

#### Key Findings

| Feature | Observation |
|----------|------------|
| Sex | Strongest predictor. Female survival rate ≈ 74%, Male ≈ 19% |
| Pclass | 1st class ≈ 63%, 2nd ≈ 47%, 3rd ≈ 24% survival |
| Age | Children survived more often, elderly passengers less |
| Fare | Highly skewed and positively correlated with survival |
| Family Size | Solo travelers and very large families survived less |

#### Multicollinearity Analysis

Variance Inflation Factor (VIF) analysis revealed multicollinearity between:

- `sibsp`
- `parch`

These were combined into a single feature:

```python
family = sibsp + parch
```

---

### 2. Feature Engineering (`pipelines.py`)

Created additional features:

#### Title Extraction

Passenger titles extracted from names:

- Mr
- Mrs
- Miss
- Master
- Rare

#### Family Features

```python
family = sibsp + parch
```

Created:

- `family`
- `family_binned`

Categories:

- Alone
- Small Family
- Large Family

#### Age Features

Retained:

- Raw Age

Created:

- Child
- Teen
- Adult
- Middle Age
- Senior

#### Features Dropped

| Feature | Reason |
|----------|----------|
| cabin | 77% missing values |
| name | Information captured via title extraction |
| ticket | High cardinality |
| boat | Data leakage |
| body | Data leakage |
| home.dest | Sparse and noisy |

#### Tested and Removed

- `ticket_shared`

Reason:

- High VIF (~8.7)
- No measurable performance improvement

---

### 3. Preprocessing Pipeline (`pipelines.py`)

All preprocessing is performed **inside cross-validation folds** to avoid data leakage.

| Feature | Treatment |
|----------|----------|
| age | Median Imputation → Age Binning → StandardScaler |
| fare | Median Imputation → Log Transform (`log1p`) → StandardScaler |
| embarked | Most Frequent Imputation → OneHotEncoder |
| titles | Most Frequent Imputation → OneHotEncoder |
| family_binned | Most Frequent Imputation → OneHotEncoder |
| sex | Most Frequent Imputation → OrdinalEncoder |
| pclass | Most Frequent Imputation |

---

### 4. Class Imbalance Handling

Used:

```python
SMOTE()
```

Characteristics:

- Applied after preprocessing
- Runs inside each CV fold
- Prevents information leakage

---

### 5. Hyperparameter Tuning

Used:

```python
RandomizedSearchCV
```

Configuration:

- CV = 5 folds
- Scoring = ROC-AUC

#### Why RandomizedSearchCV?

Compared with GridSearchCV:

- Faster
- Better coverage of large search spaces
- More practical when SMOTE and SVM are included inside the pipeline

---

## Models and Results

| Model | CV ROC-AUC | Test Accuracy |
|---------|------------|------------|
| Logistic Regression | 0.84 | 79% |
| Linear SVM | 0.85 | 79% |

---

## Key Decisions

### RandomizedSearchCV over GridSearchCV

Chosen because:

- SVM is computationally expensive
- SMOTE increases training cost
- Random search explores larger parameter spaces efficiently

### Scaling Reverted for Certain Features

Testing showed that scaling:

- `pclass`
- `family`

reduced model performance.

Therefore, scaling was removed for those features.

### Dropped `ticket_shared`

Reasons:

- VIF ≈ 8.7
- No accuracy gain
- Added unnecessary complexity

### Default Threshold Retained

Threshold:

```python
0.5
```

Reason:

- Balanced False Positives and False Negatives
- No asymmetric business cost exists in this problem

---

## Logistic Regression Evaluation

### Confusion Matrix

```text
                  Predicted
                0          1
Actual 0      122         22
Actual 1       33         85
```

### Metrics

| Metric | Value |
|----------|----------|
| Accuracy | 79% |
| Precision | 79% |
| Recall | 72% |
| F1 Score | 75% |

---

## What I Learned

Through this project, I learned:

- Building leak-free machine learning pipelines
- Proper placement of preprocessing inside cross-validation folds
- Using RandomizedSearchCV effectively
- Detecting multicollinearity using VIF
- Feature engineering based on domain knowledge
- Correct usage of SMOTE inside pipelines
- Importance of validating preprocessing decisions empirically
- Organizing ML projects using a modular structure

---

## Future Work

Planned improvements:

- Implement Random Forest
- Implement XGBoost
- Engineer deck information from cabin
- Create fare-per-person features
- Experiment with stacking and ensemble models
- Deploy using Flask or Streamlit

---

## Pipeline structure

Raw Data
   ↓
Train-Test Split
   ↓
Feature Engineering
   ↓
ColumnTransformer
   ↓
SMOTE
   ↓
Model
   ↓
RandomizedSearchCV
   ↓
Evaluation

>> This project is intended for educational and portfolio purposes.