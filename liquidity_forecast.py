"""
LiquidityForecastScore لسهم TADAWUL:4140 — يتنبأ باحتمال السيولة المرتفعة (Level 2:
RelativeVolume>1.5) للجلسة القادمة، باستخدام انحدار لوجستي مبني من الصفر (NumPy فقط)
ومُدرَّب ومُختبر عبر Walk-Forward Validation حقيقي (بدون أي تسرّب معلومات مستقبلية).

يعتمد على liquidity_cycle_analysis.py (نفس المتغيرات ونفس مصدر البيانات
data/4140_180_daily.csv، قراءة فقط).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from liquidity_cycle_analysis import (
    DATA_DIR, DAILY_180_CSV, DAYS_PER_CYCLE, CANDIDATE_PERIODS,
    load_daily, build_liquidity_features, periodicity_analysis,
)

FORECAST_OUT_CSV = os.path.join(DATA_DIR, "4140_liquidity_forecast.csv")

WALK_FORWARD_START = 90
MIN_TRAIN_ROWS = 30
TARGET_COL = "high_liquidity_L2"
FEATURE_COLS = [
    "day_prob_causal", "day_avg_relvol_causal", "gap_ratio_causal",
    "lag1_relvol", "phase_cos", "phase_sin",
    "recent5_avg_relvol_causal", "trend5_causal",
]
N_FORECAST_SESSIONS = 15

pd.set_option("display.width", 175)


def section(title):
    print()
    print(title)
    print("-" * len(title))


# ============================================================
# 1) بناء المتغيرات السببية (Causal) — كل قيمة تعتمد فقط على بيانات سابقة لصفّها
# ============================================================

def build_causal_features(df, best_period):
    out = df.copy()

    flag = out[TARGET_COL].astype(float)
    relvol = out["relative_volume"]

    # 1) احتمال السيولة تاريخيًا لنفس موقع اليوم داخل الدورة (سببي: shift قبل expanding)
    out["day_prob_causal"] = out.groupby("day_in_cycle")[TARGET_COL].apply(
        lambda s: s.astype(float).shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # 2) متوسط RelativeVolume التاريخي لنفس موقع اليوم (سببي)
    out["day_avg_relvol_causal"] = out.groupby("day_in_cycle")["relative_volume"].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True)

    # 3) المسافة منذ آخر يوم سيولة مرتفعة، نسبةً إلى متوسط الفجوة التاريخي (سببي)
    event_idx = pd.Series(np.where(flag.shift(1) == 1, np.arange(len(out)), np.nan), index=out.index)
    last_event_idx = event_idx.ffill()
    current_idx = pd.Series(np.arange(len(out)), index=out.index)
    gap_since = current_idx - last_event_idx

    mean_gap_causal = flag.shift(1).expanding().mean()  # p التراكمية -> 1/p تقريب لمتوسط الفجوة
    approx_mean_gap = 1.0 / mean_gap_causal.replace(0, np.nan)
    out["gap_ratio_causal"] = gap_since / approx_mean_gap

    # 4) الزخم المرتبط بالارتباط الذاتي: نستخدم القيمة المتأخرة خطوة واحدة مباشرة
    #    (يترك الانحدار اللوجستي يقدّر وزنها = تقدير عملي لتأثير ACF(1))
    out["lag1_relvol"] = relvol.shift(1)

    # 5) الطور الدوري (Harmonic phase) وفق أقوى دورة مكتشفة من بيانات معروفة فقط
    position = np.arange(len(out))
    out["phase_cos"] = np.cos(2 * np.pi * position / best_period)
    out["phase_sin"] = np.sin(2 * np.pi * position / best_period)

    # 6) متوسط آخر 5 جلسات (سببي)
    out["recent5_avg_relvol_causal"] = relvol.shift(1).rolling(5).mean()

    # 7) اتجاه RelativeVolume الحالي (فرق بين متوسطي 5 جلسات متتاليين، سببي)
    older5 = relvol.shift(6).rolling(5).mean()
    out["trend5_causal"] = out["recent5_avg_relvol_causal"] - older5

    return out


# ============================================================
# 2) انحدار لوجستي من الصفر (IRLS / Newton-Raphson + Ridge بسيط للاستقرار)
# ============================================================

def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def fit_logistic_regression(X, y, l2=1.0, max_iter=100, tol=1e-8):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(p + 1)
    reg = np.eye(p + 1) * l2
    reg[0, 0] = 0.0

    for _ in range(max_iter):
        z = np.clip(Xb @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(pr * (1 - pr), 1e-6, None)
        grad = Xb.T @ (y - pr) - reg @ beta
        hess = (Xb * w[:, None]).T @ Xb + reg
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta_new = beta + delta
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_logistic(beta, X):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.clip(Xb @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# 3) مقاييس التقييم (Precision / Recall / F1 / ROC-AUC / Brier) — تطبيق يدوي
# ============================================================

def confusion_counts(y_true, y_pred_bin):
    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    return tp, fp, fn, tn


def evaluation_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred_bin = (y_prob >= threshold).astype(int)

    tp, fp, fn, tn = confusion_counts(y_true, y_pred_bin)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else np.nan

    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos > 0 and n_neg > 0:
        ranks = stats.rankdata(y_prob)
        auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = np.nan

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "n": len(y_true), "n_positive": n_pos, "n_negative": n_neg,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": auc, "brier_score": brier,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_auc_ci(y_true, y_prob, n_boot=3000, seed=42):
    """فاصل ثقة 95% لـ ROC-AUC عبر إعادة أخذ العينات (Bootstrap) — لتقييم موثوقية النتيجة
    مع عيّنة صغيرة بدل إعلان تفوّق مؤكد من رقم واحد فقط."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        n_pos, n_neg = yt.sum(), (1 - yt).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = stats.rankdata(yp)
        auc = (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        aucs.append(auc)

    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ============================================================
# 4) Walk-Forward Validation
# ============================================================

def run_walk_forward(df):
    results = []

    for t in range(WALK_FORWARD_START, len(df)):
        known = df.iloc[:t]
        valid_relvol = known["relative_volume"].dropna()

        if len(valid_relvol) >= 40:
            periodicity = periodicity_analysis(known["relative_volume"], max_lag=40)
            best_period = int(periodicity["candidates"].sort_values("power_rank").iloc[0]["period"])
        else:
            best_period = DAYS_PER_CYCLE

        feat_df = build_causal_features(df.iloc[:t + 1], best_period)

        train_X_raw = feat_df.loc[feat_df.index < t, FEATURE_COLS].dropna()
        if len(train_X_raw) < MIN_TRAIN_ROWS:
            continue
        train_y = df.loc[train_X_raw.index, TARGET_COL].astype(int).to_numpy()

        target_feat = feat_df.loc[[t], FEATURE_COLS]
        if target_feat.isna().any(axis=1).iloc[0]:
            continue

        mean, std = standardize_fit(train_X_raw.to_numpy(dtype=float))
        train_X = standardize_apply(train_X_raw.to_numpy(dtype=float), mean, std)
        target_X = standardize_apply(target_feat.to_numpy(dtype=float), mean, std)

        beta = fit_logistic_regression(train_X, train_y, l2=1.0)
        predicted_prob = float(predict_logistic(beta, target_X)[0])

        baseline_prob = float(train_y.mean())

        actual_relvol = float(df.loc[t, "relative_volume"])
        actual_high = bool(df.loc[t, TARGET_COL])

        results.append({
            "phase": "walk_forward_eval",
            "session_number": int(df.loc[t, "session_number"]),
            "date": df.loc[t, "date"],
            "day_in_cycle": int(df.loc[t, "day_in_cycle"]),
            "best_period_used": best_period,
            "predicted_liquidity_probability": predicted_prob,
            "baseline_probability": baseline_prob,
            "actual_relative_volume": actual_relvol,
            "actual_high_liquidity": "Yes" if actual_high else "No",
            "actual_high_liquidity_bin": int(actual_high),
        })

    return pd.DataFrame(results)


# ============================================================
# 5) توقّع الـ15 جلسة القادمة (بدون أي بيانات مستقبلية حقيقية)
# ============================================================

def forecast_next_sessions(df, n_sessions=N_FORECAST_SESSIONS):
    periodicity = periodicity_analysis(df["relative_volume"], max_lag=40)
    best_period = int(periodicity["candidates"].sort_values("power_rank").iloc[0]["period"])

    feat_df = build_causal_features(df, best_period)
    train_X_raw = feat_df[FEATURE_COLS].dropna()
    train_y = df.loc[train_X_raw.index, TARGET_COL].astype(int).to_numpy()

    mean, std = standardize_fit(train_X_raw.to_numpy(dtype=float))
    train_X = standardize_apply(train_X_raw.to_numpy(dtype=float), mean, std)
    beta = fit_logistic_regression(train_X, train_y, l2=1.0)

    last_idx = len(df) - 1
    last_row = feat_df.iloc[last_idx]

    rows = []
    for step in range(1, n_sessions + 1):
        future_idx = last_idx + step
        day_in_cycle = (future_idx % DAYS_PER_CYCLE) + 1

        day_group = feat_df[feat_df["day_in_cycle"] == day_in_cycle]
        day_prob = df.loc[day_group.index, TARGET_COL].astype(float).mean() if len(day_group) else np.nan
        day_avg_relvol = df.loc[day_group.index, "relative_volume"].mean() if len(day_group) else np.nan

        gap_since = future_idx - (df[df[TARGET_COL]]["session_number"].max() - 1) if df[TARGET_COL].any() else np.nan
        mean_gap = 1.0 / df[TARGET_COL].astype(float).mean() if df[TARGET_COL].astype(float).mean() > 0 else np.nan
        gap_ratio = gap_since / mean_gap if mean_gap else np.nan

        phase_cos = np.cos(2 * np.pi * future_idx / best_period)
        phase_sin = np.sin(2 * np.pi * future_idx / best_period)

        # الميزات المعتمدة على الزخم الفعلي (lag1 / recent5 / trend5) تُثبَّت عند آخر قيمة معروفة
        # فعليًا (بدون بيانات مستقبلية مصطنعة) — لذلك تتراجع الثقة كلما ابتعدنا عن الجلسة القادمة مباشرة.
        lag1 = last_row["recent5_avg_relvol_causal"] if step > 1 else df["relative_volume"].iloc[-1]
        recent5 = last_row["recent5_avg_relvol_causal"]
        trend5 = last_row["trend5_causal"]

        feature_vector = np.array([[day_prob, day_avg_relvol, gap_ratio, lag1, phase_cos, phase_sin, recent5, trend5]], dtype=float)

        if np.isnan(feature_vector).any():
            predicted_prob = np.nan
        else:
            scaled = standardize_apply(feature_vector, mean, std)
            predicted_prob = float(predict_logistic(beta, scaled)[0])

        expected_relvol = day_avg_relvol if not np.isnan(day_avg_relvol) else df["relative_volume"].mean()

        if step == 1:
            confidence = "High"
        elif step <= 5:
            confidence = "Medium"
        else:
            confidence = "Low"

        if pd.isna(predicted_prob):
            liquidity_label = "غير محدد"
        elif predicted_prob >= 0.5:
            liquidity_label = "مرتفع"
        else:
            liquidity_label = "طبيعي"

        rows.append({
            "phase": "future_forecast",
            "session_offset": step,
            "day_in_cycle": day_in_cycle,
            "forecast_probability": predicted_prob,
            "expected_relative_volume": expected_relvol,
            "high_liquidity_probability_label": liquidity_label,
            "confidence_level": confidence,
        })

    return pd.DataFrame(rows), best_period


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 70)
    print("TADAWUL:4140 - LIQUIDITY FORECAST (Walk-Forward Validation)")
    print("=" * 70)

    raw = load_daily()
    df = build_liquidity_features(raw)
    print(f"\nعدد الجلسات: {len(df)}  |  بداية Walk-Forward: بعد أول {WALK_FORWARD_START} جلسة")

    section("[1] Walk-Forward Validation (Expanding Window, بدون أي تسرّب معلومات)")
    wf_results = run_walk_forward(df)
    print(f"عدد التوقعات المُنجزة فعليًا: {len(wf_results)} من أصل {len(df) - WALK_FORWARD_START} جلسة مستهدفة")

    if len(wf_results) == 0:
        print("لا توجد توقعات كافية للتقييم.")
        model_metrics = baseline_metrics = None
    else:
        model_metrics = evaluation_metrics(wf_results["actual_high_liquidity_bin"], wf_results["predicted_liquidity_probability"])
        baseline_metrics = evaluation_metrics(wf_results["actual_high_liquidity_bin"], wf_results["baseline_probability"])

        section("[2] Model vs Baseline (احتمال ثابت = المعدل التاريخي وقت كل توقع)")
        print(f"{'المقياس':<15}{'النموذج':>15}{'Baseline':>15}")
        for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score"),
                            ("roc_auc", "ROC-AUC"), ("brier_score", "Brier Score")]:
            mv = model_metrics[key]
            bv = baseline_metrics[key]
            mv_s = f"{mv:.4f}" if not pd.isna(mv) else "N/A"
            bv_s = f"{bv:.4f}" if not pd.isna(bv) else "N/A"
            print(f"{label:<15}{mv_s:>15}{bv_s:>15}")

        print(f"\nn={model_metrics['n']}  (High Liquidity فعليًا: {model_metrics['n_positive']}  |  عادي: {model_metrics['n_negative']})")

        auc_lo, auc_hi = bootstrap_auc_ci(wf_results["actual_high_liquidity_bin"], wf_results["predicted_liquidity_probability"])
        print(f"\nفاصل ثقة 95% لـ ROC-AUC (Bootstrap, n_boot=3000): [{auc_lo:.3f} , {auc_hi:.3f}]")
        ci_excludes_half = auc_lo > 0.5
        if not ci_excludes_half:
            print("⚠️ الفاصل يشمل 0.5 (أداء عشوائي) — أي أن التفوّق الظاهري على Baseline قد يكون نتيجة تذبذب العينة الصغيرة (n=90، 18 حالة إيجابية فقط) وليس مهارة تنبؤية مؤكدة.")

        beats_baseline = (
            (not pd.isna(model_metrics["roc_auc"]) and not pd.isna(baseline_metrics["roc_auc"]) and model_metrics["roc_auc"] > baseline_metrics["roc_auc"])
            and (model_metrics["brier_score"] < baseline_metrics["brier_score"])
        )
        if beats_baseline and ci_excludes_half:
            print("\n=> النموذج تفوّق على الـ Baseline في ROC-AUC و Brier Score معًا، وفاصل ثقة AUC لا يشمل 0.5 — دليل معقول (لكن على عينة صغيرة) على قدرة تنبؤية خارج العينة.")
        elif beats_baseline and not ci_excludes_half:
            print("\n=> النموذج تفوّق رقميًا على الـ Baseline لكن فاصل الثقة الإحصائي لا يستبعد أن يكون ذلك محض صدفة عيّنة صغيرة — النتيجة واعدة لكن غير مؤكدة بعد.")
        else:
            print("\n=> النموذج لم يتفوّق بوضوح على الـ Baseline (احتمال ثابت) خارج العينة.")
            print('   "No statistically useful periodic liquidity pattern detected."')

    # ------------------------------------------------------------
    # [3] توقع الـ15 جلسة القادمة
    # ------------------------------------------------------------
    section("[3] Next 15 Trading Sessions Forecast")
    forecast_df, best_period_final = forecast_next_sessions(df, N_FORECAST_SESSIONS)
    print(f"أقوى دورة مكتشفة من كامل البيانات المتاحة (180 جلسة): {best_period_final} جلسة (تُستخدم أساسًا للطور الدوري فقط، وليست حقيقة مؤكدة).")
    print(forecast_df.to_string(index=False))

    # ------------------------------------------------------------
    # حفظ النتائج
    # ------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    combined = pd.concat([wf_results, forecast_df], ignore_index=True)
    combined.to_csv(FORECAST_OUT_CSV, index=False)
    print(f"\nتم حفظ: {FORECAST_OUT_CSV}")

    return df, wf_results, model_metrics, baseline_metrics, forecast_df


if __name__ == "__main__":
    main()
