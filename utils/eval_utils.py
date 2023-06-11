import os
import json_fix
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
json.fallback_table[np.ndarray] = lambda array: array.tolist()


def save_results(results, cm, exp_dir):
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f)
    cm[0].to_csv(os.path.join(exp_dir, "confusion_matrix.csv"))
    cm[1].to_csv(os.path.join(exp_dir, "cm_metrics.csv"))
    file = open(os.path.join(exp_dir, "cm_report.log"), "a").write(cm[2])


def get_confusion_matrix(y_true, y_pred, class_names):
    """Generates the confusion matrix given the predictions
    and ground truth values.

    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.
        class_names (list): A list of string labels or class names.

    Returns:
        pandas DataFrame: The confusion matrix.
        pandas DataFrame: A dataframe containing the precision,
            recall, and F1 score per class.
    """

    y_pred = pd.Series(y_pred, name="Predicted")
    y_true = pd.Series(y_true, name="Actual")

    labels = class_names
    if isinstance(list(y_true)[0], int):
        labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_metrics = _get_cm_metrics(cm, list(cm.columns))
    cm_report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return cm, cm_metrics, cm_report


def _get_cm_metrics(cm, class_names):
    """Return the precision, recall, and F1 score per class.

    Args:
        cm (pandas DataFrame or numpy array): The confusion matrix.
        class_names (list): A list of string labels or class names.

    Returns:
        pandas DataFrame: A dataframe containing the precision,
            recall, and F1 score per class.
    """

    metrics = {}
    for i in class_names:
        tp = cm.loc[i, i]
        fn = cm.loc[i, :].drop(i).sum()
        fp = cm.loc[:, i].drop(i).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        omission_error = fn / (fn + tp) if (fn + tp) > 0 else 0
        commission_error = fp / (fp + tp) if (fp + tp) > 0 else 0
        f1 = 2 / (recall**-1 + precision**-1) if precision + recall > 0 else 0

        scores = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100,
        }

        metrics[i] = scores
    metrics = pd.DataFrame(metrics).T

    return metrics


def evaluate(y_true, y_pred):
    """Returns a dictionary of performance metrics.

    Args:
        y_true (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.

    Returns:
        dict: A dictionary of performance metrics.
    """

    return {
        "overall_accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "f1_score_micro": f1_score(y_true, y_pred, average="micro", zero_division=0) * 100,
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "precision_score": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall_score": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0) * 100,
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0) * 100,
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0) * 100,
    }


def get_scoring():
    """Returns the dictionary of scorer objects."""
    return {"f1_score": make_scorer(f1_score, average='macro')}
