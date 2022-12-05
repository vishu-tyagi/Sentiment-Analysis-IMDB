from typing import (List, Optional)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score
)

from sentiment_analysis.config import SentimentAnalysisConfig
from sentiment_analysis.utils.constants import (
    ACCURACY,
    PRECISION,
    RECALL,
    F1_SCORE
)


class CustomEvaluation():
    def __init__(self, config: SentimentAnalysisConfig = SentimentAnalysisConfig):
        self.thresholds = np.arange(0, 1, 0.005)

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray):
        return accuracy_score(y_true, y_pred)

    def compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray):
        return precision_score(
            y_true, y_pred, zero_division=0
        )

    def compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray):
        return recall_score(
            y_true, y_pred, zero_division=0
        )

    def compute_fbeta(self, y_true: np.ndarray, y_pred: np.ndarray, beta: np.float64):
        return fbeta_score(
            y_true, y_pred, beta=beta, zero_division=0
        )

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        accuracy = self.compute_accuracy(y_true, y_pred)
        precision_score = self.compute_precision(y_true, y_pred)
        recall_score = self.compute_recall(y_true, y_pred)
        f1_score = self.compute_fbeta(y_true, y_pred, beta=1)
        scores = [accuracy, precision_score, recall_score, f1_score]
        metrics = [ACCURACY, PRECISION, RECALL, F1_SCORE]
        return pd.Series(data=scores, index=metrics)

    def threshold_discovery(self, y_true: np.ndarray, y_pred_probab: np.ndarray) -> float:
        """
        Evaluate accuracy score at various thresholds

        Args:
            y_score (np.ndarray): Prediction scores
            y_true (np.ndarray): True labels

        Returns:
            float: Threshold which observes maximum accuracy score
        """
        max_score = self.compute_accuracy(y_true, np.where(y_pred_probab > .5, 1, 0))
        optimal_threshold = .5
        for th in self.thresholds:
            pred = np.where(y_pred_probab > th, 1, 0)
            score = self.compute_accuracy(y_true, pred)
            if score > max_score or (score == max_score and abs(optimal_threshold-0.5) > abs(th-0.5)):
                max_score = score
                optimal_threshold = th
        return optimal_threshold

    def binary_logistic(
        self,
        predt: np.ndarray,
        dmatrix: xgb.DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Implements XGBoost objective for binary logistic loss
        -[yln(p) + (1-y)ln(1-p)] where p = sigmoid(predt)

        Args:
            predt (np.ndarray): Predictions (before sigmoid)
            dmatrix (xgb.DMatrix): Samples DMatrix

        Returns:
            tuple[np.ndarray, np.ndarray]: Graident and Hessian of objective
        """
        y_true = dmatrix.get_label()
        y_pred_score = 1.0 / (1.0 + np.exp(-predt, dtype=np.float32))
        grad = y_pred_score - y_true
        hess = y_pred_score * (1 - y_pred_score)
        return grad, hess

    def accuracy_eval(
        self,
        y_score: np.ndarray,
        dmatrix: xgb.DMatrix
    ) -> tuple[str, float]:
        """
        Implements XGBoost accuracy evaluation metric

        Args:
            y_score (np.ndarray): Prediction scores
            dmatrix (xgb.DMatrix): Samples DMatrix

        Returns:
            tuple[str, float]: Tuple consisting of metric name and score
        """
        y_true = dmatrix.get_label()
        threshold = self.threshold_discovery(y_true=y_true, y_pred_probab=y_score)
        y_pred = np.where(y_score > threshold, 1, 0)
        score = self.compute_accuracy(y_true=y_true, y_pred=y_pred)
        return ("accuracy", score)
