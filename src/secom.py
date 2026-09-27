"""SECOM 데이터 로드, 전처리 파이프라인, 평가 지표."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def load():
    """X: 센서 DataFrame, y: 1=fail 0=pass, t: 측정 시각 (파일 순서 그대로)."""
    X = pd.read_csv(DATA / "secom.data", sep=" ", header=None, na_values="NaN")
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(DATA / "secom_labels.data", sep=" ", header=None, names=["y", "ts"])
    y = (lab["y"] == 1).astype(int).to_numpy()
    t = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    return X, y, t


class ColumnFilter(BaseEstimator, TransformerMixin):
    """학습 데이터 기준으로 결측 비율이 높은 센서와 값이 하나뿐인 센서를 제거한다."""

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        missing = X.isna().mean()
        constant = X.nunique(dropna=True) <= 1
        self.keep_ = ((missing <= self.max_missing) & ~constant).to_numpy()
        self.n_missing_ = int((missing > self.max_missing).sum())
        self.n_constant_ = int(constant.sum())
        return self

    def transform(self, X):
        return np.asarray(pd.DataFrame(X).loc[:, self.keep_], dtype=float)


def make_pipeline(model):
    """모든 전처리를 학습 fold 안에서만 fit 하는 파이프라인."""
    return Pipeline([
        ("filter", ColumnFilter()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def rf(balanced=False):
    return RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1,
        class_weight="balanced_subsample" if balanced else None,
    )


def l1_logreg(C=0.05):
    return LogisticRegression(penalty="l1", solver="liblinear", C=C,
                              class_weight="balanced", max_iter=2000)


def metrics(y_true, score, threshold=0.5):
    """정확도, 불량 검출률(fail recall), 오탐률(pass를 fail로 예측한 비율), PR-AUC."""
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / len(y_true),
        "fail_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "pr_auc": average_precision_score(y_true, score),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "n": int(len(y_true)), "n_fail": int(tp + fn),
    }
