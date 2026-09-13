"""
ContextualLiquidityModel لسهم TADAWUL:4140 — يحلّل السيولة ضمن سياقها السعري
(موقع الإغلاق، الاتجاه، الموقع من القمة/القاع، الاختراق) بدلًا من تحليلها منفردة.

يقرأ فقط data/4140_180_daily.csv (بيانات حقيقية مجلوبة مسبقًا عبر historical_180.py)
ولا يعدّلها ولا يتصل بأي مصدر جديد. لا علاقة له بـ tv_test.py.

يحتوي على:
  - Event Study حقيقي لأربع حالات سياقية (A/B/C/D)
  - نموذج مركب ContextualLiquidityScore بأربع نسخ (Model A..D) بعد Z-score
  - Out-of-sample validation: تدريب على أول 70% واختبار على آخر 30%
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "data"
DAILY_180_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")  # قراءة فقط، بدون أي تعديل

FEATURES_OUT_CSV = os.path.join(DATA_DIR, "4140_contextual_features.csv")
EVENTS_OUT_CSV = os.path.join(DATA_DIR, "4140_contextual_events.csv")
MODELS_OUT_CSV = os.path.join(DATA_DIR, "4140_contextual_models.csv")

TRAIN_FRACTION = 0.70
MIN_GROUP_SAMPLE = 8
MIN_CORR_SAMPLE = 20

HORIZONS = {
    "NextDayReturn": 1,
    "Return3Days": 3,
    "Return5Days": 5,
    "Return7Days": 7,
    "Return10Days": 10,
}

pd.set_option("display.width", 175)


def section(title):
    print()
    print(title)
    print("-" * len(title))


def fmt(df, decimals=4, int_cols=(), pct_cols=()):
    view = df.copy()
    for col in view.columns:
        if col in int_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
        elif col in pct_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}%")
        elif pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
        elif pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda x: "YES" if x else "")
    return view


# ============================================================
# 0) تحميل البيانات الخام (قراءة فقط)
# ============================================================

def load_daily():
    if not os.path.exists(DAILY_180_CSV):
        raise FileNotFoundError(f"لم يتم العثور على {DAILY_180_CSV}. شغّل historical_180.py أولًا.")
    df = pd.read_csv(DAILY_180_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# 1) المتغيرات السياقية
# ============================================================

def build_features(raw):
    df = raw.copy()

    df["prev_close"] = df["close"].shift(1)
    df["daily_return"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    rng = df["high"] - df["low"]
    df["close_strength"] = np.where(rng == 0, 0.5, (df["close"] - df["low"]) / rng)

    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["relative_volume20"] = df["volume"] / df["volume_sma20"]

    df["close_sma20"] = df["close"].rolling(20).mean()
    df["close_sma50"] = df["close"].rolling(50).mean()
    df["trend20"] = (df["close"] - df["close_sma20"]) / df["close_sma20"]
    df["trend50"] = (df["close"] - df["close_sma50"]) / df["close_sma50"]

    lowest_low20 = df["low"].rolling(20).min()
    highest_high20 = df["high"].rolling(20).max()
    pp_range = highest_high20 - lowest_low20
    df["price_position20"] = np.where(pp_range == 0, 0.5, (df["close"] - lowest_low20) / pp_range)

    # أعلى إغلاق خلال الـ20 جلسة "السابقة" (بدون تضمين اليوم الحالي لتفادي أي تسرّب معلومات)
    prior_highest_close20 = df["close"].shift(1).rolling(20).max()
    df["breakout20"] = (df["close"] > prior_highest_close20).astype(float)
    df.loc[prior_highest_close20.isna(), "breakout20"] = np.nan

    df["previous_return5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["previous_return10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10)

    for label, n in HORIZONS.items():
        df[label] = (df["close"].shift(-n) - df["close"]) / df["close"]

    return df


# ============================================================
# 2) تعريف الحالات السياقية A / B / C / D
# ============================================================

def define_cases(df):
    p75_prev_return10 = df["previous_return10"].quantile(0.75)

    case_a = (
        (df["relative_volume20"] > 1.5)
        & (df["close_strength"] >= 0.75)
        & (df["close"] > df["close_sma20"])
        & (df["trend20"] > 0)
    )
    case_b = (
        (df["relative_volume20"] > 1.5)
        & (df["close_strength"] >= 0.75)
        & (df["breakout20"] == 1)
    )
    case_c = (
        (df["relative_volume20"] > 1.5)
        & (df["close_strength"] >= 0.75)
        & (df["previous_return10"] > p75_prev_return10)
    )
    case_d = (
        (df["relative_volume20"] > 1.5)
        & (df["close_strength"] < 0.50)
    )

    df["case_a_uptrend"] = case_a.fillna(False)
    df["case_b_breakout"] = case_b.fillna(False)
    df["case_c_extended"] = case_c.fillna(False)
    df["case_d_weak_close"] = case_d.fillna(False)

    df["normal_day"] = ~(df["case_a_uptrend"] | df["case_b_breakout"] | df["case_c_extended"] | df["case_d_weak_close"])

    return df, p75_prev_return10


# ============================================================
# أدوات إحصائية عامة (Event Study)
# ============================================================

def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def event_stats(series_pct):
    s = pd.Series(series_pct).dropna()
    if len(s) == 0:
        return None
    std = s.std(ddof=1) if len(s) > 1 else np.nan
    return {
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "win_rate": (s > 0).mean() * 100,
        "std": std,
        "sharpe_like": (s.mean() / std) if (std and std > 0) else np.nan,
        "max": s.max(),
        "min": s.min(),
    }


def event_vs_normal_test(event_returns, normal_returns):
    e = pd.Series(event_returns).dropna()
    n = pd.Series(normal_returns).dropna()

    result = {
        "event": event_stats(e),
        "normal": event_stats(n),
        "ttest_p": np.nan,
        "mannwhitney_p": np.nan,
        "cohens_d": np.nan,
        "reliable": False,
    }

    if len(e) >= MIN_GROUP_SAMPLE and len(n) >= MIN_GROUP_SAMPLE:
        tt = stats.ttest_ind(e, n, equal_var=False)
        result["ttest_p"] = tt.pvalue
        try:
            mw = stats.mannwhitneyu(e, n, alternative="two-sided")
            result["mannwhitney_p"] = mw.pvalue
        except ValueError:
            pass
        result["cohens_d"] = cohens_d(e, n)
        result["reliable"] = True

    return result


def run_event_study(df):
    cases = {
        "CaseA_HighVol_StrongClose_Uptrend": "case_a_uptrend",
        "CaseB_HighVol_StrongClose_Breakout": "case_b_breakout",
        "CaseC_HighVol_StrongClose_AlreadyExtended": "case_c_extended",
        "CaseD_HighVol_WeakClose": "case_d_weak_close",
    }

    normal_mask = df["normal_day"]
    results = {}

    for case_name, flag_col in cases.items():
        flag = df[flag_col]
        n_events = int(flag.sum())
        horizon_results = {}
        for horizon in HORIZONS:
            event_returns = df.loc[flag, horizon] * 100
            normal_returns = df.loc[normal_mask, horizon] * 100
            horizon_results[horizon] = event_vs_normal_test(event_returns, normal_returns)
        results[case_name] = {"n_events": n_events, "horizons": horizon_results}

    return results


def print_event_study(results):
    for case_name, data in results.items():
        section(f"Event Study: {case_name}  (N events = {data['n_events']})")
        if data["n_events"] == 0:
            print("لا توجد أي جلسة مطابقة لهذه الحالة ضمن 180 جلسة.")
            continue
        for horizon, res in data["horizons"].items():
            if res["event"] is None:
                print(f"  {horizon}: لا توجد قيم عائد مستقبلي متاحة.")
                continue
            e = res["event"]
            print(f"\n  {horizon}:")
            print(f"    Event  : N={e['n']:>3}  mean={e['mean']:+.2f}%  median={e['median']:+.2f}%  "
                  f"win_rate={e['win_rate']:.0f}%  std={e['std']:.2f}%  sharpe~={e['sharpe_like']:.2f}  "
                  f"max={e['max']:+.2f}%  min={e['min']:+.2f}%")
            if res["normal"] is not None:
                nrm = res["normal"]
                print(f"    Normal : N={nrm['n']:>3}  mean={nrm['mean']:+.2f}%  median={nrm['median']:+.2f}%  "
                      f"win_rate={nrm['win_rate']:.0f}%  std={nrm['std']:.2f}%")
            if res["reliable"]:
                sig = "✅ دلالة إحصائية" if (res["ttest_p"] < 0.05 or res["mannwhitney_p"] < 0.05) else "❌ بدون دلالة واضحة"
                print(f"    t-test p={res['ttest_p']:.4f}  |  Mann-Whitney p={res['mannwhitney_p']:.4f}  |  "
                      f"Cohen's d={res['cohens_d']:+.3f}  |  {sig}")
            else:
                print(f"    ⚠️ N={e['n']} أقل من الحد الأدنى ({MIN_GROUP_SAMPLE}) — اختبار غير موثوق، أرقام وصفية فقط.")


def save_events_csv(results):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for case_name, data in results.items():
        for horizon, res in data["horizons"].items():
            if res["event"] is None:
                continue
            e = res["event"]
            rows.append({
                "case": case_name, "horizon": horizon, "n": e["n"],
                "mean_pct": e["mean"], "median_pct": e["median"], "win_rate_pct": e["win_rate"],
                "std_pct": e["std"], "sharpe_like": e["sharpe_like"], "max_pct": e["max"], "min_pct": e["min"],
                "ttest_p": res["ttest_p"], "mannwhitney_p": res["mannwhitney_p"], "cohens_d": res["cohens_d"],
                "reliable": res["reliable"],
            })
    pd.DataFrame(rows).to_csv(EVENTS_OUT_CSV, index=False)


# ============================================================
# 3) ContextualLiquidityScore — Models A..D + Z-score + Out-of-sample validation
# ============================================================

def chrono_train_test_split(df, frac=TRAIN_FRACTION):
    n_train = int(len(df) * frac)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def zscore_params_from_train(train_df, cols):
    params = {}
    for col in cols:
        vals = train_df[col].dropna()
        params[col] = {"mean": vals.mean(), "std": vals.std(ddof=1) if len(vals) > 1 else np.nan}
    return params


def apply_zscore(df, params, cols):
    out = df.copy()
    for col in cols:
        mean = params[col]["mean"]
        std = params[col]["std"]
        if not std or pd.isna(std) or std == 0:
            out[f"z_{col}"] = 0.0
        else:
            out[f"z_{col}"] = (out[col] - mean) / std
    return out


def build_model_scores(df):
    df = df.copy()
    df["trend_strength"] = (df["trend20"] + df["trend50"]) / 2.0
    df["breakout_factor"] = np.where(df["breakout20"] == 1, 1.5, 1.0)

    df["model_a"] = df["z_relative_volume20"]
    df["model_b"] = df["z_relative_volume20"] * df["z_close_strength"]
    df["model_c"] = df["z_relative_volume20"] * df["z_close_strength"] * df["z_trend_strength"]
    df["model_d_contextual_score"] = df["model_c"] * df["breakout_factor"]

    return df


def correlation_test(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    mask = (~x.isna()) & (~y.isna())
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    n = len(xv)
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan,
                "spearman_p": np.nan, "r_squared": np.nan, "reg_p_value": np.nan}
    pearson_r, pearson_p = stats.pearsonr(xv, yv)
    spearman_r, spearman_p = stats.spearmanr(xv, yv)
    reg = stats.linregress(xv, yv)
    return {"n": n, "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "r_squared": reg.rvalue ** 2, "reg_p_value": reg.pvalue}


def evaluate_models_train_test(train, test):
    models = {
        "Model A (RelativeVolume only)": "model_a",
        "Model B (RelVol x CloseStrength)": "model_b",
        "Model C (RelVol x CloseStrength x Trend)": "model_c",
        "Model D (RelVol x CloseStrength x Trend x Breakout)": "model_d_contextual_score",
    }

    rows = []
    for mname, col in models.items():
        for horizon in HORIZONS:
            train_res = correlation_test(train[col], train[horizon] * 100)
            test_res = correlation_test(test[col], test[horizon] * 100)
            rows.append({"model": mname, "horizon": horizon, "split": "Training", **train_res})
            rows.append({"model": mname, "horizon": horizon, "split": "Testing", **test_res})

    return pd.DataFrame(rows)


def diagnose_overfitting(models_df):
    diagnosis = []
    for (model, horizon), group in models_df.groupby(["model", "horizon"]):
        train_row = group[group["split"] == "Training"].iloc[0]
        test_row = group[group["split"] == "Testing"].iloc[0]

        train_sig = (not pd.isna(train_row["pearson_p"])) and train_row["pearson_p"] < 0.05 and train_row["n"] >= MIN_CORR_SAMPLE
        test_sig = (not pd.isna(test_row["pearson_p"])) and test_row["pearson_p"] < 0.05 and test_row["n"] >= MIN_CORR_SAMPLE
        same_sign = (
            not pd.isna(train_row["pearson_r"]) and not pd.isna(test_row["pearson_r"])
            and np.sign(train_row["pearson_r"]) == np.sign(test_row["pearson_r"])
        )

        if train_sig and test_sig and same_sign:
            verdict = "Validated (يصمد في العينتين)"
        elif (train_sig or test_sig) and not same_sign:
            # الاتجاه ينعكس بين الفترتين — أخطر من مجرد فقدان الدلالة: العلاقة غير مستقرة إحصائيًا وقد تكون زائفة
            verdict = "Sign Reversal (الاتجاه ينعكس بين التدريب والاختبار — غير موثوق)"
        elif train_sig and not test_sig:
            verdict = "Overfitting (نجح تدريبًا وفشل اختبارًا)"
        elif not train_sig and test_sig:
            verdict = "Emerged in Test only (دلالة بالاختبار فقط دون تدريب — غير موثوق)"
        else:
            verdict = "No Evidence (لا دلالة في أي منهما)"

        diagnosis.append({
            "model": model, "horizon": horizon,
            "train_r": train_row["pearson_r"], "train_p": train_row["pearson_p"], "train_n": train_row["n"],
            "test_r": test_row["pearson_r"], "test_p": test_row["pearson_p"], "test_n": test_row["n"],
            "verdict": verdict,
        })

    return pd.DataFrame(diagnosis)


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 70)
    print("TADAWUL:4140 - CONTEXTUAL LIQUIDITY MODEL (180 SESSIONS)")
    print("Liquidity in context: Close Position + Trend + Range Position + Breakout")
    print("=" * 70)

    raw = load_daily()
    print(f"\nعدد الجلسات المحمّلة (قراءة فقط، بدون أي تعديل) من {DAILY_180_CSV}: {len(raw)}")
    print(f"نطاق التواريخ: {raw['date'].iloc[0].date()} -> {raw['date'].iloc[-1].date()}")

    df = build_features(raw)
    df, p75_prev_return10 = define_cases(df)

    n_a = int(df["case_a_uptrend"].sum())
    n_b = int(df["case_b_breakout"].sum())
    n_c = int(df["case_c_extended"].sum())
    n_d = int(df["case_d_weak_close"].sum())
    n_normal = int(df["normal_day"].sum())

    print(f"\nعتبة percentile75(PreviousReturn10) المستخدمة للحالة C: {p75_prev_return10 * 100:+.2f}%")
    print(f"عدد الأحداث: CaseA={n_a}  CaseB={n_b}  CaseC={n_c}  CaseD={n_d}  |  أيام عادية (بدون أي إشارة)={n_normal}")

    # ------------------------------------------------------------
    # [1] Event Study
    # ------------------------------------------------------------
    event_results = run_event_study(df)
    print_event_study(event_results)
    save_events_csv(event_results)

    # ------------------------------------------------------------
    # [2] ContextualLiquidityScore: Out-of-sample validation
    # ------------------------------------------------------------
    section("[2] ContextualLiquidityScore - Out-of-Sample Validation (Train 70% / Test 30%)")

    train_raw, test_raw = chrono_train_test_split(df, TRAIN_FRACTION)
    print(f"Training: {len(train_raw)} جلسة ({train_raw['date'].iloc[0].date()} -> {train_raw['date'].iloc[-1].date()})")
    print(f"Testing : {len(test_raw)} جلسة ({test_raw['date'].iloc[0].date()} -> {test_raw['date'].iloc[-1].date()})")
    print("(معلمات Z-score تُستخرج من Training فقط ثم تُطبَّق كما هي على Testing لتفادي أي تسرّب معلومات.)")

    zscore_cols = ["relative_volume20", "close_strength", "trend_strength"]

    # نحتاج trend_strength قبل استخراج معلمات Z-score
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()
    train_raw["trend_strength"] = (train_raw["trend20"] + train_raw["trend50"]) / 2.0
    test_raw["trend_strength"] = (test_raw["trend20"] + test_raw["trend50"]) / 2.0

    zparams = zscore_params_from_train(train_raw, zscore_cols)

    train_z = apply_zscore(train_raw, zparams, zscore_cols)
    test_z = apply_zscore(test_raw, zparams, zscore_cols)

    train_full = build_model_scores(train_z)
    test_full = build_model_scores(test_z)

    models_df = evaluate_models_train_test(train_full, test_full)
    diagnosis_df = diagnose_overfitting(models_df)

    section("[3] Model Comparison — Training vs Testing (Pearson r, p-value, N)")
    pivot_view = models_df.pivot_table(
        index=["model", "horizon"], columns="split",
        values=["n", "pearson_r", "pearson_p"], aggfunc="first",
    )
    print(fmt(pivot_view.reset_index(), decimals=4).to_string(index=False))

    section("[4] Overfitting Diagnosis")
    print(fmt(diagnosis_df, decimals=4).to_string(index=False))

    # ------------------------------------------------------------
    # حفظ الملفات (دمج Train/Test بعد حساب Z-score والدرجات المركّبة، مرتّبة زمنيًا)
    # ------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    full_scored = pd.concat([train_full, test_full]).sort_values("date").reset_index(drop=True)
    full_scored["split"] = ["Training"] * len(train_full) + ["Testing"] * len(test_full)
    full_scored.to_csv(FEATURES_OUT_CSV, index=False)
    models_df.to_csv(MODELS_OUT_CSV, index=False)
    diagnosis_df.to_csv(MODELS_OUT_CSV.replace(".csv", "_diagnosis.csv"), index=False)

    print(f"\nتم حفظ الملفات التالية:")
    print(f"  - {FEATURES_OUT_CSV}")
    print(f"  - {EVENTS_OUT_CSV}")
    print(f"  - {MODELS_OUT_CSV}")
    print(f"  - {MODELS_OUT_CSV.replace('.csv', '_diagnosis.csv')}")

    return df, event_results, models_df, diagnosis_df, train_full, test_full


if __name__ == "__main__":
    main()
