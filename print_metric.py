from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, f1_score

def print_results(y_true, y_pred, y_pred_proba):
    print(f"Accuracy Score {accuracy_score(y_true, y_pred)}")
    print(f"Recall Score {recall_score(y_true, y_pred)}")
    print(f"Precision Score {precision_score(y_true, y_pred)}")
    print(f"ROC AUC Score {roc_auc_score(y_true, y_pred_proba)}")
    print(f"F1 Score {f1_score(y_true, y_pred)}")
    print(f"Confusion Matrix\n{confusion_matrix(y_true, y_pred)}")
    print(f"Classification Report\n{classification_report(y_true, y_pred)}")