import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset():
    path = kagglehub.dataset_download("vinicius150987/titanic3")
    excel_path = os.path.join(path, "titanic3.xls")
    os.makedirs("images", exist_ok=True)
    os.makedirs("models/", exist_ok=True)
    os.makedirs("models/artifacts", exist_ok=True)
    df = pd.read_excel(excel_path)
    df = df.drop(columns=["boat", "body", "home.dest"])

    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    return (X_train, X_test, y_train, y_test)
