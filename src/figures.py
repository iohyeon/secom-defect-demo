"""report/ 아래 PNG 3장."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, precision_recall_curve

OUT = Path(__file__).resolve().parent.parent / "report"
BLUE, ORANGE, AQUA = "#b9bec7", "#2b6cb0", "#4b5260"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# 한글 폰트 (macOS 기본 AppleGothic, 없으면 기본 폰트)
for name in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "font.size": 10,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def plot_steps(R):
    s1, c = R["step1"], R["step2"]
    pipe = c["C3_leakage"]["pipeline_cv"]
    rows = [
        ("1차: 전체 데이터 전처리\n+ 랜덤 분할 1회", s1["accuracy"], s1["fail_recall"], s1["false_alarm_rate"]),
        ("누수 제거: fold 안 전처리\n(RF, 층화 CV 25회 평균)", pipe["accuracy"]["mean"],
         pipe["fail_recall"]["mean"], pipe["false_alarm_rate"]["mean"]),
        ("최종 모델, 시간순 테스트\n(임계값 0.5)", c["C5_threshold"]["test_at_default_0.5"]["accuracy"],
         c["C5_threshold"]["test_at_default_0.5"]["fail_recall"],
         c["C5_threshold"]["test_at_default_0.5"]["false_alarm_rate"]),
        ("최종 모델, 시간순 테스트\n(검증 구간에서 고른 임계값)", c["C5_threshold"]["test_at_threshold"]["accuracy"],
         c["C5_threshold"]["test_at_threshold"]["fail_recall"],
         c["C5_threshold"]["test_at_threshold"]["false_alarm_rate"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(rows))
    w = 0.26
    for k, (label, color) in enumerate([("정확도", BLUE), ("불량 검출률 (fail recall)", ORANGE),
                                        ("오탐률 (pass를 fail로)", AQUA)]):
        vals = [r[k + 1] for r in rows]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    fontsize=8.5, color=INK)
    ax.axhline(s1["majority_accuracy"], color=INK2, lw=1, ls="--")
    ax.text(len(rows) - 0.5, s1["majority_accuracy"] + 0.02, f"전부 pass로 예측한 정확도 {s1['majority_accuracy']:.2f}",
            ha="right", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("비율")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    ax.set_title("1차 정확도 0.92는 전부 pass로 찍은 정확도와 같습니다", pad=30, color=INK)
    _save(fig, "fig1_steps_accuracy_vs_recall.png")


def plot_pr_curves(rand, timed, R):
    c4 = R["step2"]["C4_split"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for (yt, sc), color, label in [
        (rand, BLUE, f"랜덤 층화 분할 예시 1회, AP {average_precision_score(*rand):.3f}\n"
                     f"(20회 평균 {c4['random_20x']['pr_auc']['mean']:.3f})"),
        (timed, ORANGE, f"시간순 분할 (마지막 20%), AP {c4['time_test']['pr_auc']:.3f}"),
    ]:
        p, r, _ = precision_recall_curve(yt, sc)
        ax.step(r[:-1], p[:-1], where="post", color=color, lw=2, label=label)
        ax.axhline(yt.mean(), color=color, lw=1, ls=":")
    ax.plot([], [], color=INK2, lw=1, ls=":", label="점선: 각 테스트 셋의 불량 비율 (무작위 기준선)")
    ax.set_xlabel("불량 검출률 (recall)")
    ax.set_ylabel("정밀도 (precision)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("PR 곡선: 랜덤 분할 vs 시간순 분할", color=INK)
    _save(fig, "fig2_pr_random_vs_time.png")


def plot_stability(sensors, n_folds):
    top = sensors[:12][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    vals = [s["top10_folds"] for s in top]
    ax.barh([s["sensor"] for s in top], vals, color=BLUE, height=0.7)
    for i, (v, s) in enumerate(zip(vals, top)):
        ax.text(v + 0.3, i, f"{v}/{n_folds}  계수 {'+' if s['coef_sign'] > 0 else '-'}", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, n_folds + 8)
    ax.set_xlabel(f"|계수| 상위 10에 든 fold 수 (전체 {n_folds} fold)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("불량 후보 센서의 fold 간 안정성 (원인 아님, 검토 대상)", color=INK)
    _save(fig, "fig3_sensor_stability.png")
