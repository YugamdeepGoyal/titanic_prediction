import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

def feature_engineer(dataset):
    dataset = dataset.copy()
    dataset["family"] = dataset["sibsp"] + dataset["parch"]
    dataset["titles"] = dataset["name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    dataset.loc[dataset["titles"] == "Mlle", "titles"] = "Miss"
    dataset.loc[(dataset["titles"] == "Mme") | (dataset["titles"] == "Ms") | (dataset["titles"] == "Lady") | (dataset["titles"] == "Dona"), "titles"] = "Mrs"
    dataset.loc[(dataset["titles"] == "Dr") | (dataset["titles"] == "Rev") | (dataset["titles"] == "Col") | (dataset["titles"] == "Major") | (dataset["titles"] == "Capt") | (dataset["titles"] == "Don") | (dataset["titles"] == "Jonkheer") | (dataset["titles"] == "the Countess") | (dataset["titles"] == "Sir"), "titles"] = "Rare"
    dataset["family_binned"] = pd.cut(
        dataset["family"],
        bins=[-1, 0, 4, float("inf")],
        labels=["alone", "small", "large"],
        include_lowest=True
    )
    dataset = dataset.drop(columns=["sibsp", "parch", "name", "ticket", "cabin", "family"])
    return dataset

def bin_age(X):
    age_bins = pd.cut(
        X.ravel(),
        bins=[-1, 12, 18, 35, 60, float("inf")],
        labels=[0, 1, 2, 3, 4]
    ).codes
    return age_bins.reshape(-1, 1)  # return just bins, not stacked with age


categorical_encoding = ["embarked", "titles", "family_binned"]
label_encoding_cols = ["sex"]
numerical_cols = ["age"]
categorical_cols_non_encoding = ["pclass"]
family_col = ["family"]
fare_col = ["fare"]

def get_preprocessor():
    numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

    age_bin_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # impute first
    ("binner", FunctionTransformer(bin_age))         # then bin, no scaler
    ])

    encoding_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    label_encoder_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    passthrough_cols_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    fare_col_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", FunctionTransformer(np.log1p, inverse_func=np.expm1)),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("age_bin", age_bin_pipeline, ["age"]),
        ("numerical", numerical_pipeline, numerical_cols),
        ("encoder", encoding_pipeline, categorical_encoding),
        ("passthrough", passthrough_cols_pipeline, categorical_cols_non_encoding),
        ("fare_cols", fare_col_pipe, fare_col),
        ("label_encoder", label_encoder_pipeline, label_encoding_cols)
    ])
    return preprocessor