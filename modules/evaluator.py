import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BenchmarkEvaluator:
    """
    Evaluation & Benchmark Module for eKYC Presentation Attack Detection.
    Calculates:
    - APCER (Attack Presentation Classification Error Rate / False Accept Rate)
    - BPCER (Bona Fide Presentation Classification Error Rate / False Reject Rate)
    - EER (Equal Error Rate)
    - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    """

    def evaluate_predictions(self, y_true, y_pred_scores, threshold=50.0):
        """
        y_true: list of ints (0 = Bona Fide / Real, 1 = Attack / Spoof)
        y_pred_scores: list of floats (Risk Score 0.0 - 100.0)
        threshold: risk threshold above which sample is classified as Attack (1)
        """
        y_true = np.array(y_true)
        y_pred_scores = np.array(y_pred_scores)

        # Binary predictions based on threshold
        y_pred = (y_pred_scores >= threshold).astype(int)

        # Confusion Matrix: TN, FP, FN, TP
        # Ground Truth: 0 = Genuine, 1 = Attack
        # Predicted: 0 = Genuine, 1 = Attack
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # APCER: Attack Presentation Classification Error Rate (False Accept Rate of Attack)
        # APCER = FN / (TP + FN) -> Attacks misclassified as Genuine
        apcer = fn / float(tp + fn) if (tp + fn) > 0 else 0.0

        # BPCER: Bona Fide Presentation Classification Error Rate (False Reject Rate of Genuine)
        # BPCER = FP / (TN + FP) -> Genuine misclassified as Attack
        bpcer = fp / float(tn + fp) if (tn + fp) > 0 else 0.0

        # Overall Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # EER & ROC curve threshold sweep
        eer, eer_threshold = self._compute_eer(y_true, y_pred_scores)

        return {
            "threshold_used": threshold,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "apcer": round(float(apcer), 4),
            "bpcer": round(float(bpcer), 4),
            "eer": round(float(eer), 4),
            "eer_threshold": round(float(eer_threshold), 2),
            "confusion_matrix": {
                "TN_Genuine_Correct": int(tn),
                "FP_Genuine_as_Attack": int(fp),
                "FN_Attack_as_Genuine": int(fn),
                "TP_Attack_Correct": int(tp)
            }
        }

    def _compute_eer(self, y_true, y_scores):
        """
        Sweeps thresholds from 0 to 100 to find point where APCER ≈ BPCER.
        """
        thresholds = np.linspace(0, 100, 201)
        apcers = []
        bpcers = []

        for th in thresholds:
            y_pred = (y_scores >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            apcer = fn / float(tp + fn) if (tp + fn) > 0 else 0.0
            bpcer = fp / float(tn + fp) if (tn + fp) > 0 else 0.0

            apcers.append(apcer)
            bpcers.append(bpcer)

        apcers = np.array(apcers)
        bpcers = np.array(bpcers)

        # Minimum absolute difference between APCER and BPCER
        diffs = np.abs(apcers - bpcers)
        min_idx = np.argmin(diffs)

        eer = (apcers[min_idx] + bpcers[min_idx]) / 2.0
        eer_th = thresholds[min_idx]

        return eer, eer_th
