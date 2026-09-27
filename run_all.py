"""SECOM 불량 예측 데모: 1차 결과 -> 검증 -> 후보 센서. 모든 수치와 그림을 다시 만든다.

실행: uv run python run_all.py
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.figures import plot_pr_curves, plot_stability, plot_steps
from src.secom import SEED, ColumnFilter, l1_logreg, load, make_pipeline, metrics, rf

warnings.filterwarnings("ignore")
T0 = time.time()
R = {}
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)


def cv_scores(X, y, build):
    """build(X_tr, y_tr, X_te) -> 점수. 25개 fold의 지표 목록."""
    out = []
    for tr, te in CV.split(X, y):
        s = build(X[tr], y[tr], X[te])
        out.append(metrics(y[te], s))
    return out


def summarize(rows):
    keys = ["accuracy", "fail_recall", "false_alarm_rate", "pr_auc"]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])),
                "std": float(np.nanstd([r[k] for r in rows]))} for k in keys}


# ---------------------------------------------------------------- 데이터
Xdf, y, t = load()
X = Xdf.to_numpy()
cf = ColumnFilter().fit(Xdf)
R["data"] = {
    "n_wafers": int(len(y)), "n_sensors": int(X.shape[1]), "n_fail": int(y.sum()),
    "fail_rate": float(y.mean()), "time_min": str(t.min()), "time_max": str(t.max()),
    "time_sorted": bool(t.is_monotonic_increasing),
    "missing_cell_ratio": float(np.isnan(X).mean()),
    "constant_sensors": cf.n_constant_, "missing_over_50pct_sensors": cf.n_missing_,
    "sensors_kept": int(cf.keep_.sum()),
    "wafers_with_any_missing": int(np.isnan(X).any(axis=1).sum()),
    "fail_rate_by_month": {str(k.date()): round(float(v), 4) for k, v in
                           pd.Series(y, index=t).resample("ME").mean().items()},
}
print("data", R["data"])

# ---------------------------------------------------------------- Step 1: 1차 파이프라인
# 빠르게 만든 첫 버전이 흔히 하는 순서: 전체 데이터로 대체/스케일/특징선택 후 랜덤 분할.
X1 = SimpleImputer(strategy="mean").fit_transform(X)
X1 = VarianceThreshold(0.0).fit_transform(X1)
X1 = StandardScaler().fit_transform(X1)
X1 = SelectKBest(f_classif, k=40).fit_transform(X1, y)  # 라벨을 본 특징선택 (전체 데이터)
Xtr, Xte, ytr, yte = train_test_split(X1, y, test_size=0.2, random_state=SEED)
m1 = rf().fit(Xtr, ytr)
s1 = m1.predict_proba(Xte)[:, 1]
step1 = metrics(yte, s1)
step1["majority_accuracy"] = float(accuracy_score(yte, np.zeros_like(yte)))
R["step1"] = step1
print("step1", step1)

# ---------------------------------------------------------------- Step 2: 검증
checks = {}
# C1, C2는 Step 1과 같은 테스트 셋에서 본다.
checks["C1_majority_baseline"] = {
    "model_accuracy": step1["accuracy"], "all_pass_accuracy": step1["majority_accuracy"]}
checks["C2_metrics"] = {k: step1[k] for k in
                        ["fail_recall", "false_alarm_rate", "pr_auc", "tp", "fn", "fp", "n_fail"]}


# C3 누수: 같은 RF, 같은 25 fold. 전처리+특징선택을 전체에서 fit vs fold 안에서 fit.
def leaky_select(Xall, yall):
    Z = SimpleImputer(strategy="mean").fit_transform(Xall)
    Z = VarianceThreshold(0.0).fit_transform(Z)
    Z = StandardScaler().fit_transform(Z)
    return SelectKBest(f_classif, k=40).fit_transform(Z, yall)


Xleak = leaky_select(X, y)
leak_rows = cv_scores(Xleak, y, lambda a, b, c: rf().fit(a, b).predict_proba(c)[:, 1])
pipe_sel = Pipeline(make_pipeline(rf()).steps[:-1]
                    + [("select", SelectKBest(f_classif, k=40)), ("model", rf())])
clean_rows = cv_scores(X, y, lambda a, b, c: clone(pipe_sel).fit(a, b).predict_proba(c)[:, 1])
checks["C3_leakage"] = {"leaky_cv": summarize(leak_rows), "pipeline_cv": summarize(clean_rows)}
print("C3", checks["C3_leakage"])

# 후보 모델 비교 (파이프라인, 랜덤 층화 CV, AP 기준). 테스트 셋은 보지 않는다.
cands = {"rf_balanced": rf(balanced=True), "l1_logreg": l1_logreg()}
cand_cv = {}
for name, mdl in cands.items():
    rows = cv_scores(X, y, lambda a, b, c, m=mdl: clone(make_pipeline(m)).fit(a, b).predict_proba(c)[:, 1])
    cand_cv[name] = summarize(rows)
best = max(cand_cv, key=lambda k: cand_cv[k]["pr_auc"]["mean"])
checks["model_choice"] = {"cv": cand_cv, "chosen": best}
print("model choice", checks["model_choice"])
final = make_pipeline(cands[best])

# C4 분할 현실성: 시간순 60/20/20 (학습/검증/테스트) vs 같은 비율의 랜덤 층화 분할 20회.
order = np.argsort(t.to_numpy(), kind="stable")
n = len(y)
i_tr, i_va, i_te = order[: int(.6 * n)], order[int(.6 * n): int(.8 * n)], order[int(.8 * n):]
tm = clone(final).fit(X[i_tr], y[i_tr])
s_va, s_te = tm.predict_proba(X[i_va])[:, 1], tm.predict_proba(X[i_te])[:, 1]
time_test = metrics(y[i_te], s_te)
rand = []
for seed in range(20):
    a, b = train_test_split(np.arange(n), test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(b, test_size=0.5, stratify=y[b], random_state=seed)
    m = clone(final).fit(X[a], y[a])
    r = metrics(y[te], m.predict_proba(X[te])[:, 1])
    r["prevalence"] = float(y[te].mean())
    rand.append(r)
last_rand = (y[te], m.predict_proba(X[te])[:, 1])
# 참고: 선택되지 않은 L1 로지스틱 회귀의 시간순 테스트 결과 (모델 선택에 쓰지 않음)
lm = make_pipeline(l1_logreg()).fit(X[i_tr], y[i_tr])
l1_time = metrics(y[i_te], lm.predict_proba(X[i_te])[:, 1])
checks["C4_split"] = {
    "l1_logreg_time_test_reference": l1_time,
    "random_20x": summarize(rand),
    "random_pr_auc_min_max": [float(min(r["pr_auc"] for r in rand)), float(max(r["pr_auc"] for r in rand))],
    "random_test_prevalence": float(np.mean([r["prevalence"] for r in rand])),
    "time_test": time_test,
    "time_prevalence": {"train": float(y[i_tr].mean()), "val": float(y[i_va].mean()),
                        "test": float(y[i_te].mean())},
    "time_ranges": {k: [str(t.iloc[idx].min()), str(t.iloc[idx].max())]
                    for k, idx in [("train", i_tr), ("val", i_va), ("test", i_te)]},
}
print("C4", checks["C4_split"])

# C5 임계값: 시간순 검증 구간에서 (검출률 - 오탐률)이 최대인 점을 고르고 테스트에 그대로 적용.
grid = np.unique(s_va)
youden = [metrics(y[i_va], s_va, th) for th in grid]
j = int(np.argmax([m_["fail_recall"] - m_["false_alarm_rate"] for m_ in youden]))
th = float(grid[j])
checks["C5_threshold"] = {
    "threshold": th, "val_at_threshold": youden[j],
    "test_at_default_0.5": metrics(y[i_te], s_te, 0.5),
    "test_at_threshold": metrics(y[i_te], s_te, th),
    "test_tradeoff": [{"threshold": float(q), **{k: v for k, v in metrics(y[i_te], s_te, float(q)).items()
                                                  if k in ("fail_recall", "false_alarm_rate", "tp", "fp")}}
                      for q in np.quantile(s_va, [0.5, 0.7, 0.8, 0.9, 0.95])],
}
print("C5", checks["C5_threshold"])
R["step2"] = checks

# ---------------------------------------------------------------- Step 3: 후보 센서 안정성
TOPK = 10
names = Xdf.columns.to_numpy()
hits, signs = {}, {}
n_folds = 0
for tr, _ in CV.split(X, y):
    p = make_pipeline(l1_logreg()).fit(X[tr], y[tr])
    kept = names[p.named_steps["filter"].keep_]
    coef = p.named_steps["model"].coef_.ravel()
    for i in np.argsort(-np.abs(coef))[:TOPK]:
        if coef[i] != 0:
            hits[kept[i]] = hits.get(kept[i], 0) + 1
            signs.setdefault(kept[i], []).append(np.sign(coef[i]))
    n_folds += 1
rank = sorted(hits.items(), key=lambda kv: -kv[1])
sensors = []
for s, c in rank[:15]:
    col = Xdf[s]
    sensors.append({
        "sensor": s, "top10_folds": c, "n_folds": n_folds,
        "sign_consistent": bool(abs(np.mean(signs[s])) == 1.0),
        "coef_sign": int(np.sign(np.mean(signs[s]))),  # +: 값이 클수록 fail 쪽 점수 증가
        "median_pass": float(col[y == 0].median()), "median_fail": float(col[y == 1].median()),
        "missing_ratio": float(col.isna().mean()),
    })
R["step3"] = {"method": f"L1 로지스틱 회귀, 25 fold 각각 |계수| 상위 {TOPK}", "candidates": sensors,
              "n_distinct_sensors_in_any_top10": len(hits)}
print("step3", R["step3"])

# ---------------------------------------------------------------- 그림
plot_steps(R)
plot_pr_curves(last_rand, (y[i_te], s_te), R)
plot_stability(sensors, n_folds)

R["runtime_sec"] = round(time.time() - T0, 1)
with open("report/results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)
print(f"done in {R['runtime_sec']} s")
