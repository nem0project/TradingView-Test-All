"""
تحليل رياضي وإحصائي متقدم لسهم TADAWUL:4140 على آخر 180 جلسة تداول فعلية
(البيانات مجلوبة مسبقًا عبر historical_180.py من TradingView WebSocket).

لا يتصل هذا الملف بـ TradingView ولا يستخدم بيانات لحظية —
المصدر الوحيد هو data/4140_180_daily.csv.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "data"
DAILY_180_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")
ANALYSIS_180_CSV = os.path.join(DATA_DIR, "4140_180_analysis.csv")
FEATURES_180_CSV = os.path.join(DATA_DIR, "4140_180_features.csv")  # ملف مساعد كامل للرسم والتقرير

SAUDI_DAILY_LIMIT = 0.10
VOL_WINDOW = 20  # نافذة SMA(Volume) و ZScore(RelativeVolume)

MIN_CORR_SAMPLE = 20   # الحد الأدنى لاعتبار اختبار ارتباط قابلاً للتفسير
MIN_GROUP_SAMPLE = 8   # الحد الأدنى لكل مجموعة في اختبارات الفرضيات (t-test / Mann-Whitney)

HORIZONS = {
    "NextDayReturn": 1,
    "Return3Days": 3,
    "Return5Days": 5,
    "Return7Days": 7,
}

pd.set_option("display.width", 170)


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
# 0) تحميل البيانات الخام (180 جلسة فعلية محفوظة مسبقًا)
# ============================================================

def load_daily():
    if not os.path.exists(DAILY_180_CSV):
        raise FileNotFoundError(
            f"لم يتم العثور على {DAILY_180_CSV}. شغّل historical_180.py أولًا لجلب البيانات من TradingView."
        )
    df = pd.read_csv(DAILY_180_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# 1) المتغيرات الأساسية لكل جلسة
# ============================================================

def build_features(df):
    df = df.copy()

    df["prev_close"] = df["close"].shift(1)
    df["daily_return"] = (df["close"] - df["prev_close"]) / df["prev_close"]

    df["volume_sma20"] = df["volume"].rolling(VOL_WINDOW).mean()
    df["relative_volume"] = df["volume"] / df["volume_sma20"]

    rng = df["high"] - df["low"]
    df["close_strength"] = np.where(rng == 0, 0.5, (df["close"] - df["low"]) / rng)

    df["range"] = rng
    df["range_percent"] = rng / df["prev_close"]

    df["liquidity"] = df["volume"]
    df["liquidity_strength"] = df["relative_volume"] * df["close_strength"]
    df["money_flow"] = df["close"] * df["volume"]
    df["volume_change"] = df["volume"].pct_change()

    # ZScore(RelativeVolume) على نافذة متحركة (20 جلسة) لتفادي تسرّب معلومات مستقبلية
    roll_mean = df["relative_volume"].rolling(VOL_WINDOW).mean()
    roll_std = df["relative_volume"].rolling(VOL_WINDOW).std()
    zscore = (df["relative_volume"] - roll_mean) / roll_std
    df["relvol_zscore"] = zscore.where(roll_std > 0, 0.0)

    sign_return = np.sign(df["daily_return"]).fillna(0.0)
    df["sign_daily_return"] = sign_return

    # ثلاث نسخ من BullishLiquidityScore يتم اختبارها ومقارنتها
    df["bls_v1_sign"] = df["relvol_zscore"] * df["close_strength"] * sign_return
    df["bls_v2_tanh_return"] = df["relvol_zscore"] * df["close_strength"] * np.tanh(df["daily_return"] / SAUDI_DAILY_LIMIT)
    df["bls_v3_additive"] = 0.5 * np.tanh(df["relvol_zscore"]) + 0.5 * df["close_strength"]

    # عوائد مستقبلية (نسبية Percentage Returns، وفق حد الحركة السعودي)
    for label, n in HORIZONS.items():
        df[label] = (df["close"].shift(-n) - df["close"]) / df["close"]

    # إشارات القوة
    df["strong_bullish_liquidity"] = (
        (df["relative_volume"] > 1.5) & (df["close_strength"] >= 0.75) & (df["daily_return"] > 0)
    )
    df["extreme_bullish_liquidity"] = (
        (df["relative_volume"] > 2.0) & (df["close_strength"] >= 0.85) & (df["daily_return"] > 0)
    )

    # تنبيه بيانات: حركات تتجاوز حد السوق السعودي النظري (±10%) — لأغراض جودة البيانات فقط
    df["exceeds_daily_limit"] = df["daily_return"].abs() > SAUDI_DAILY_LIMIT

    return df


# ============================================================
# أدوات إحصائية عامة
# ============================================================

def correlation_test(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    mask = (~x.isna()) & (~y.isna())
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    n = len(xv)

    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "r_squared": np.nan, "reg_p_value": np.nan,
                "slope": np.nan, "intercept": np.nan}

    pearson_r, pearson_p = stats.pearsonr(xv, yv)
    spearman_r, spearman_p = stats.spearmanr(xv, yv)
    reg = stats.linregress(xv, yv)

    return {
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,       # = Rank correlation
        "spearman_p": spearman_p,
        "r_squared": reg.rvalue ** 2,
        "reg_p_value": reg.pvalue,
        "slope": reg.slope,
        "intercept": reg.intercept,
    }


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


def group_stats(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return None
    return {
        "n": len(s),
        "mean_pct": s.mean() * 100,
        "median_pct": s.median() * 100,
        "win_rate_pct": (s > 0).mean() * 100,
        "max_pct": s.max() * 100,
        "min_pct": s.min() * 100,
    }


def two_group_test(flagged, rest):
    result = {"flagged": group_stats(flagged), "rest": group_stats(rest),
              "mannwhitney_p": np.nan, "cohens_d": np.nan, "reliable": False}

    f = pd.Series(flagged).dropna()
    r = pd.Series(rest).dropna()

    if len(f) >= MIN_GROUP_SAMPLE and len(r) >= MIN_GROUP_SAMPLE:
        try:
            mw = stats.mannwhitneyu(f, r, alternative="two-sided")
            result["mannwhitney_p"] = mw.pvalue
        except ValueError:
            pass
        result["cohens_d"] = cohens_d(f, r)
        result["reliable"] = True

    return result


# ============================================================
# 2) مقارنة المتغيرات: A) RelativeVolume  B) CloseStrength  C) DailyReturn  D) BullishLiquidityScore
# ============================================================

def compare_model_variants(df):
    """يقارن نسخ BullishLiquidityScore الثلاث إحصائيًا عبر آفاق العائد الأربعة."""
    variants = {
        "V1 (ZScore x CloseStrength x Sign)": "bls_v1_sign",
        "V2 (ZScore x CloseStrength x tanh(Return))": "bls_v2_tanh_return",
        "V3 (Additive: tanh(ZScore)+CloseStrength)": "bls_v3_additive",
    }

    rows = []
    for vname, col in variants.items():
        for horizon in HORIZONS:
            res = correlation_test(df[col], df[horizon] * 100)
            rows.append({"variant": vname, "horizon": horizon, **res})

    variant_df = pd.DataFrame(rows)

    # اختيار أفضل نسخة: أكبر متوسط |pearson_r| بين الآفاق التي حققت دلالة إحصائية، وإلا أكبر |r| عمومًا
    summary = variant_df.groupby("variant").apply(
        lambda g: pd.Series({
            "mean_abs_pearson_r": g["pearson_r"].abs().mean(),
            "n_significant": (g["pearson_p"] < 0.05).sum(),
        }),
        include_groups=False,
    ).reset_index()

    summary = summary.sort_values(["n_significant", "mean_abs_pearson_r"], ascending=False)
    best_variant_name = summary.iloc[0]["variant"]
    best_col = variants[best_variant_name]

    return variant_df, summary, best_variant_name, best_col


# ============================================================
# 3) مقارنة القدرة التفسيرية: A/B/C/D عبر الآفاق الأربعة
# ============================================================

def compare_explanatory_variables(df, best_bls_col, best_bls_label):
    variables = {
        "A) RelativeVolume": "relative_volume",
        "B) CloseStrength": "close_strength",
        "C) DailyReturn": "daily_return",
        f"D) BullishLiquidityScore [{best_bls_label}]": best_bls_col,
    }

    rows = []
    for vlabel, col in variables.items():
        for horizon in HORIZONS:
            res = correlation_test(df[col], df[horizon] * 100)
            rows.append({"variable": vlabel, "horizon": horizon, **res})

    return pd.DataFrame(rows)


def best_variable_per_horizon(compare_df):
    best = {}
    for horizon in HORIZONS:
        sub = compare_df[compare_df["horizon"] == horizon].copy()
        sig = sub[(sub["pearson_p"] < 0.05) & (sub["n"] >= MIN_CORR_SAMPLE)]
        if len(sig) > 0:
            winner = sig.loc[sig["pearson_r"].abs().idxmax()]
            best[horizon] = (winner["variable"], winner["pearson_r"], winner["pearson_p"], winner["n"], True)
        else:
            winner = sub.loc[sub["pearson_r"].abs().idxmax()]
            best[horizon] = (winner["variable"], winner["pearson_r"], winner["pearson_p"], winner["n"], False)
    return best


# ============================================================
# 4) تحليل الشرائح Quintiles (Q1..Q5)
# ============================================================

def quantile_analysis(df, score_col, n_target=5):
    work = df.dropna(subset=[score_col]).copy()
    try:
        binned = pd.qcut(work[score_col], n_target, duplicates="drop")
    except ValueError:
        return None, None, 0

    categories = list(binned.cat.categories)
    k = len(categories)
    if k < 2:
        return None, None, k

    label_map = {cat: f"Q{i + 1}" for i, cat in enumerate(categories)}
    work["quantile"] = binned.map(label_map).astype(str)

    means = work.groupby("quantile", observed=True)[list(HORIZONS)].mean() * 100
    counts = work.groupby("quantile", observed=True)[score_col].count()
    means.insert(0, "n", counts)
    means = means.reindex([f"Q{i + 1}" for i in range(k)])

    lowest_label, highest_label = "Q1", f"Q{k}"
    q_high_vs_q_low = {}
    for horizon in HORIZONS:
        q_low = work.loc[work["quantile"] == lowest_label, horizon]
        q_high = work.loc[work["quantile"] == highest_label, horizon]
        test = two_group_test(q_high, q_low)
        q_high_vs_q_low[horizon] = test

    return means, q_high_vs_q_low, k


# ============================================================
# 5) اختبار الفرضيات: StrongBullishLiquidity / ExtremeBullishLiquidity
# ============================================================

def signal_performance(df, flag_col):
    out = {}
    flagged_mask = df[flag_col]
    n_flagged = int(flagged_mask.sum())

    for horizon in HORIZONS:
        flagged = df.loc[flagged_mask, horizon]
        rest = df.loc[~flagged_mask, horizon]
        out[horizon] = two_group_test(flagged, rest)

    return n_flagged, out


def print_signal_block(title, df, flag_col):
    section(title)
    n_flagged, results = signal_performance(df, flag_col)
    print(f"عدد الجلسات المطابقة ضمن {len(df)} جلسة: {n_flagged}")

    if n_flagged == 0:
        print("لا توجد أي جلسة مطابقة لهذه الإشارة ضمن 180 جلسة. لا نتائج لعرضها.")
        return results

    for horizon, res in results.items():
        print(f"\n{horizon}:")
        if res["flagged"] is None:
            print("  لا توجد قيم عائد مستقبلي متاحة.")
            continue
        f = res["flagged"]
        print(f"  N={f['n']:>3}  mean={f['mean_pct']:+.2f}%  median={f['median_pct']:+.2f}%  "
              f"win_rate={f['win_rate_pct']:.0f}%  max={f['max_pct']:+.2f}%  min={f['min_pct']:+.2f}%")
        if res["reliable"]:
            sig = "✅ دلالة إحصائية (p<0.05)" if res["mannwhitney_p"] < 0.05 else "❌ بدون دلالة واضحة (p>=0.05)"
            print(f"  مقابل باقي الجلسات (N={res['rest']['n']}, mean={res['rest']['mean_pct']:+.2f}%): "
                  f"Mann-Whitney p={res['mannwhitney_p']:.4f}  |  Cohen's d={res['cohens_d']:+.3f}  |  {sig}")
        else:
            print(f"  ⚠️ N={f['n']} أقل من الحد الأدنى ({MIN_GROUP_SAMPLE}) — لا يمكن إجراء اختبار إحصائي موثوق. أرقام وصفية فقط.")

    return results


# ============================================================
# حفظ النتائج
# ============================================================

def save_features_csv(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FEATURES_180_CSV, index=False)


def save_analysis_csv(variant_df, compare_df, quantile_means, quantile_tests, signal_results):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []

    for _, r in variant_df.iterrows():
        rows.append({
            "type": "model_variant", "label": f"{r['variant']} vs {r['horizon']}",
            "n": r["n"], "pearson_r": r["pearson_r"], "pearson_p": r["pearson_p"],
            "spearman_r": r["spearman_r"], "spearman_p": r["spearman_p"],
            "r_squared": r["r_squared"], "reg_p_value": r["reg_p_value"],
            "mean_pct": np.nan, "median_pct": np.nan, "win_rate_pct": np.nan,
            "max_pct": np.nan, "min_pct": np.nan, "mannwhitney_p": np.nan, "cohens_d": np.nan,
        })

    for _, r in compare_df.iterrows():
        rows.append({
            "type": "variable_comparison", "label": f"{r['variable']} vs {r['horizon']}",
            "n": r["n"], "pearson_r": r["pearson_r"], "pearson_p": r["pearson_p"],
            "spearman_r": r["spearman_r"], "spearman_p": r["spearman_p"],
            "r_squared": r["r_squared"], "reg_p_value": r["reg_p_value"],
            "mean_pct": np.nan, "median_pct": np.nan, "win_rate_pct": np.nan,
            "max_pct": np.nan, "min_pct": np.nan, "mannwhitney_p": np.nan, "cohens_d": np.nan,
        })

    if quantile_means is not None:
        for q, row in quantile_means.iterrows():
            for horizon in HORIZONS:
                rows.append({
                    "type": "quantile_mean", "label": f"{q}:{horizon}",
                    "n": row["n"], "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                    "r_squared": np.nan, "reg_p_value": np.nan,
                    "mean_pct": row[horizon], "median_pct": np.nan, "win_rate_pct": np.nan,
                    "max_pct": np.nan, "min_pct": np.nan, "mannwhitney_p": np.nan, "cohens_d": np.nan,
                })

    if quantile_tests is not None:
        for horizon, res in quantile_tests.items():
            if res["flagged"] is None:
                continue
            f = res["flagged"]
            rows.append({
                "type": "quantile_extremes", "label": f"TopQuantile_vs_BottomQuantile:{horizon}",
                "n": f["n"], "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "r_squared": np.nan, "reg_p_value": np.nan,
                "mean_pct": f["mean_pct"], "median_pct": f["median_pct"], "win_rate_pct": f["win_rate_pct"],
                "max_pct": f["max_pct"], "min_pct": f["min_pct"],
                "mannwhitney_p": res["mannwhitney_p"], "cohens_d": res["cohens_d"],
            })

    for signal_name, results in signal_results.items():
        for horizon, res in results.items():
            if not res or res.get("flagged") is None:
                continue
            f = res["flagged"]
            rows.append({
                "type": "signal_hypothesis", "label": f"{signal_name}:{horizon}",
                "n": f["n"], "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "r_squared": np.nan, "reg_p_value": np.nan,
                "mean_pct": f["mean_pct"], "median_pct": f["median_pct"], "win_rate_pct": f["win_rate_pct"],
                "max_pct": f["max_pct"], "min_pct": f["min_pct"],
                "mannwhitney_p": res.get("mannwhitney_p"), "cohens_d": res.get("cohens_d"),
            })

    pd.DataFrame(rows).to_csv(ANALYSIS_180_CSV, index=False)


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 70)
    print("TADAWUL:4140 - ADVANCED ANALYSIS (180 SESSIONS)")
    print("Liquidity Strength | Close Strength | BullishLiquidityScore")
    print("=" * 70)

    raw = load_daily()
    n_sessions = len(raw)
    print(f"\nعدد الجلسات المحمّلة من {DAILY_180_CSV}: {n_sessions}")
    print(f"نطاق التواريخ: {raw['date'].iloc[0].date()} -> {raw['date'].iloc[-1].date()}")

    df = build_features(raw)
    save_features_csv(df)

    n_exceed = int(df["exceeds_daily_limit"].sum())
    if n_exceed:
        print(f"[ملاحظة جودة بيانات] عدد الجلسات التي تجاوزت حركتها ±10% (حد السوق النظري): {n_exceed} "
              "(قد تعكس تعديلات/توزيعات أو حركة استثنائية حقيقية — لم تُستبعد من التحليل).")

    # ------------------------------------------------------------
    # [1] عيّنة من الميزات
    # ------------------------------------------------------------
    section("[1] Daily Features (sample)")
    show_cols = ["date", "close", "relative_volume", "close_strength", "liquidity_strength",
                 "money_flow", "bls_v1_sign", "bls_v2_tanh_return", "bls_v3_additive"]
    print("أول 5 جلسات:")
    print(fmt(df.head(5)[show_cols], int_cols=["money_flow"]).to_string(index=False))
    print("\nآخر 5 جلسات:")
    print(fmt(df.tail(5)[show_cols], int_cols=["money_flow"]).to_string(index=False))

    # ------------------------------------------------------------
    # [2] مقارنة نسخ BullishLiquidityScore
    # ------------------------------------------------------------
    section("[2] BullishLiquidityScore - Model Variant Comparison")
    variant_df, variant_summary, best_variant_name, best_bls_col = compare_model_variants(df)
    print(fmt(variant_df[["variant", "horizon", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "r_squared"]]).to_string(index=False))
    print("\nملخص المقارنة (عدد الآفاق ذات الدلالة الإحصائية، ومتوسط |Pearson r|):")
    print(fmt(variant_summary, decimals=4).to_string(index=False))
    print(f"\n=> النسخة المختارة كممثل لـ BullishLiquidityScore في بقية التحليل: {best_variant_name}")

    # ------------------------------------------------------------
    # [3] مقارنة القدرة التفسيرية: A) RelVol  B) CloseStrength  C) DailyReturn  D) BullishLiquidityScore
    # ------------------------------------------------------------
    section("[3] Explanatory Power Comparison: RelativeVolume vs CloseStrength vs DailyReturn vs BullishLiquidityScore")
    compare_df = compare_explanatory_variables(df, best_bls_col, best_variant_name)
    print(fmt(compare_df[["variable", "horizon", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "r_squared"]]).to_string(index=False))

    best_map = best_variable_per_horizon(compare_df)
    print("\nأفضل متغير تفسيري لكل أفق زمني:")
    for horizon, (var, r, p, n, sig) in best_map.items():
        flag = "✅ دالّ إحصائيًا" if sig else "⚠️ غير دالّ (لا يوجد متغير وصل لدلالة كافية)"
        print(f"  {horizon:15s}: {var:45s} r={r:+.4f}  p={p:.4f}  n={n:.0f}  {flag}")

    # ------------------------------------------------------------
    # [4] تحليل الشرائح Q1..Q5
    # ------------------------------------------------------------
    section(f"[4] Quintile Analysis ({best_variant_name})")
    quantile_means, quantile_tests, k_bins = quantile_analysis(df, best_bls_col)
    if quantile_means is None:
        print("تعذّر تقسيم البيانات إلى شرائح متمايزة (قيم متكررة كثيرة في BullishLiquidityScore).")
    else:
        if k_bins < 5:
            print(f"⚠️ ملاحظة: تجمّعت قيم BullishLiquidityScore بكثرة عند نفس المستوى (بسبب دقة سعرية محدودة تجعل "
                  f"close_strength=0 أو 0.5 في أيام كثيرة)، فتقلّص عدد الشرائح الفعلية إلى {k_bins} بدل 5 "
                  "(بدل دمج شرائح مصطنعة على قيم متطابقة).")
        print(fmt(quantile_means.reset_index(), pct_cols=list(HORIZONS), int_cols=["n"]).to_string(index=False))
        print(f"\nأعلى شريحة (Q{k_bins}) مقابل أدنى شريحة (Q1) — هل أقوى الإشارات فعلًا تتفوق على أضعفها؟:")
        for horizon, res in quantile_tests.items():
            if res["flagged"] is None or res["rest"] is None:
                print(f"  {horizon}: بيانات غير كافية.")
                continue
            f, r = res["flagged"], res["rest"]
            if res["reliable"]:
                sig = "✅ دلالة إحصائية" if res["mannwhitney_p"] < 0.05 else "❌ بدون دلالة واضحة"
                print(f"  {horizon:15s}: Q{k_bins} mean={f['mean_pct']:+.2f}% (n={f['n']})  vs  Q1 mean={r['mean_pct']:+.2f}% (n={r['n']})  "
                      f"| Mann-Whitney p={res['mannwhitney_p']:.4f}  Cohen's d={res['cohens_d']:+.3f}  {sig}")
            else:
                print(f"  {horizon:15s}: Q{k_bins} mean={f['mean_pct']:+.2f}% (n={f['n']})  vs  Q1 mean={r['mean_pct']:+.2f}% (n={r['n']})  "
                      f"⚠️ عينة صغيرة — لا اختبار إحصائي موثوق.")

    # ------------------------------------------------------------
    # [5] و [6] اختبار إشارات القوة
    # ------------------------------------------------------------
    signal_results = {}
    signal_results["StrongBullishLiquidity"] = print_signal_block(
        "[5] StrongBullishLiquidity (RelVol>1.5, CloseStrength>=0.75, DailyReturn>0)",
        df, "strong_bullish_liquidity",
    )
    signal_results["ExtremeBullishLiquidity"] = print_signal_block(
        "[6] ExtremeBullishLiquidity (RelVol>2.0, CloseStrength>=0.85, DailyReturn>0)",
        df, "extreme_bullish_liquidity",
    )

    save_analysis_csv(variant_df, compare_df, quantile_means, quantile_tests, signal_results)

    # ------------------------------------------------------------
    # [7] الملخص النهائي (بصيغة المستخدم المطلوبة بالضبط)
    # ------------------------------------------------------------
    section("[7] FINAL SUMMARY")
    print(f"عدد الجلسات: {n_sessions}")
    print("هل البيانات حقيقية من TradingView؟ نعم")

    for horizon in HORIZONS:
        var, r, p, n, sig = best_map[horizon]
        status = f"{var} (r={r:+.4f}, p={p:.4f}, n={n:.0f})" if sig else f"لا يوجد متغير ذو دلالة إحصائية كافية (أفضل مرشح غير دالّ: {var}, p={p:.4f})"
        print(f"أفضل متغير لـ {horizon}: {status}")

    n_signif_liquidity_related = sum(
        1 for horizon in HORIZONS
        if best_map[horizon][4] and "BullishLiquidityScore" in best_map[horizon][0]
    )
    any_liquidity_signal_significant = any(
        signal_results[s][h]["reliable"] and not pd.isna(signal_results[s][h]["mannwhitney_p"]) and signal_results[s][h]["mannwhitney_p"] < 0.05
        for s in signal_results for h in signal_results[s]
    )

    if any_liquidity_signal_significant or n_signif_liquidity_related > 0:
        verdict = "نعم (بحذر) — ظهرت دلالة إحصائية في بعض الآفاق فقط، وليس عبر جميع الآفاق"
    elif n_sessions < 100:
        verdict = "غير كافية"
    else:
        verdict = "لا — لم تظهر ميزة إحصائية واضحة وثابتة عبر الآفاق المختبرة"

    print(f"\nهل السيولة المرتفعة + الإغلاق القوي لها ميزة إحصائية؟ {verdict}")
    print("\nتذكير: الارتباط الإحصائي (correlation) لا يعني بالضرورة قدرة تنبؤية (prediction) موثوقة.")
    print("كل نتيجة أعلاه مصحوبة بحجم العينة N وقيمة p-value — يجب قراءتهما معًا قبل أي استنتاج.")

    print(f"\nتم حفظ الملفات التالية:")
    print(f"  - {DAILY_180_CSV}")
    print(f"  - {ANALYSIS_180_CSV}")
    print(f"  - {FEATURES_180_CSV}")


if __name__ == "__main__":
    main()
