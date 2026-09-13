"""
تحليل رياضي متقدم لسهم TADAWUL:4140 اعتمادًا على البيانات اليومية التاريخية
المحفوظة مسبقًا في data/4140_daily.csv (عبر historical_analysis.py).

لا يتصل هذا الملف بـ TradingView ولا يستخدم أي بيانات لحظية —
المصدر الوحيد هو ملف OHLCV اليومي المجلوب فعليًا مسبقًا.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "data"
DAILY_CSV = os.path.join(DATA_DIR, "4140_daily.csv")
WEEKLY_CSV = os.path.join(DATA_DIR, "4140_weekly.csv")
ANALYSIS_CSV = os.path.join(DATA_DIR, "4140_analysis.csv")
SIGNALS_CSV = os.path.join(DATA_DIR, "4140_signals.csv")

SAUDI_DAILY_LIMIT = 0.10  # حد الحركة اليومي التقريبي للسوق السعودي (±10%)

# الحد الأدنى لعدد نقاط العينة كي يُعتبر أي اختبار إحصائي "قابلاً للتفسير"
MIN_CORR_SAMPLE = 10
MIN_GROUP_SAMPLE = 5

pd.set_option("display.width", 160)


def section(title):
    print()
    print(title)
    print("-" * len(title))


def fmt(df, decimals=3, int_cols=(), pct_cols=()):
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
# 0) تحميل البيانات اليومية التاريخية الموجودة فعليًا
# ============================================================

def load_daily():
    if not os.path.exists(DAILY_CSV):
        raise FileNotFoundError(
            f"لم يتم العثور على {DAILY_CSV}. شغّل historical_analysis.py أولًا لجلب البيانات التاريخية من TradingView."
        )
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# المرحلة 1: موقع الإغلاق داخل الشمعة (CLV / CloseStrength)
# ============================================================

def add_close_strength(df):
    rng = df["high"] - df["low"]
    clv = np.where(rng == 0, 0.5, (df["close"] - df["low"]) / rng)
    df["clv"] = clv
    df["close_strength"] = 2 * clv - 1
    return df


# ============================================================
# المرحلة 2: قوة السيولة (Volume Strength)
# ============================================================

def add_volume_strength(df):
    df["volume_ma7"] = df["volume"].rolling(7).mean()
    df["volume_ma14"] = df["volume"].rolling(14).mean()
    df["volume_ma28"] = df["volume"].rolling(28).mean()

    df["relative_volume7"] = df["volume"] / df["volume_ma7"]
    df["relative_volume28"] = df["volume"] / df["volume_ma28"]

    roll_mean28 = df["volume"].rolling(28).mean()
    roll_std28 = df["volume"].rolling(28).std()
    zscore = (df["volume"] - roll_mean28) / roll_std28
    zscore = zscore.where(roll_std28 > 0, 0.0)  # انحراف صفري -> نتعامل معه بأمان (0)
    df["volume_zscore28"] = zscore
    return df


# ============================================================
# المرحلة 3: قوة الحركة السعرية المطبّعة على حد السوق
# ============================================================

def add_normalized_move(df):
    df["daily_return"] = df["close"].pct_change()  # نسبة كسرية (0.012 = 1.2%)
    df["normalized_move"] = df["daily_return"] / SAUDI_DAILY_LIMIT
    df["abs_normalized_move"] = df["normalized_move"].abs()
    return df


# ============================================================
# المرحلة 4 و 5: LiquidityPressure و CloseVolumeStrength
# ============================================================

def add_pressure_indicators(df):
    df["liquidity_pressure"] = df["normalized_move"] * df["relative_volume7"]
    df["close_volume_strength"] = df["close_strength"] * df["relative_volume7"]
    return df


# ============================================================
# المرحلة 6: SmartMoneyPressure
# ============================================================

def add_smart_money_pressure(df):
    df["smart_money_pressure"] = (
        0.40 * df["close_strength"]
        + 0.30 * df["normalized_move"]
        + 0.30 * np.tanh(df["volume_zscore28"])
    )
    return df


# ============================================================
# العوائد المستقبلية (للاختبارات في المرحلتين 8 و 9 و 10) — بالنسبة المئوية
# ============================================================

def add_forward_returns(df):
    df["next_day_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_3day_return_pct"] = (df["close"].shift(-3) - df["close"]) / df["close"] * 100
    df["next_5day_return_pct"] = (df["close"].shift(-5) - df["close"]) / df["close"] * 100
    return df


# ============================================================
# إشارات الفرضيات (المرحلتان 9 و 10)
# ============================================================

def add_signal_flags(df):
    df["strong_bullish_liquidity_day"] = (
        (df["close_strength"] >= 0.70) & (df["relative_volume7"] >= 1.30) & (df["daily_return"] > 0)
    )
    df["strong_bearish_liquidity_day"] = (
        (df["close_strength"] <= -0.70) & (df["relative_volume7"] >= 1.30) & (df["daily_return"] < 0)
    )
    return df


def save_signals_csv(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    cols = [
        "date", "open", "high", "low", "close", "volume",
        "clv", "close_strength",
        "volume_ma7", "volume_ma14", "volume_ma28",
        "relative_volume7", "relative_volume28", "volume_zscore28",
        "daily_return", "normalized_move", "abs_normalized_move",
        "liquidity_pressure", "close_volume_strength", "smart_money_pressure",
        "next_day_return_pct", "next_3day_return_pct", "next_5day_return_pct",
        "strong_bullish_liquidity_day", "strong_bearish_liquidity_day",
    ]
    df[cols].to_csv(SIGNALS_CSV, index=False)


def save_daily_csv(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df[["date", "open", "high", "low", "close", "volume"]].to_csv(DAILY_CSV, index=False)


# ============================================================
# المرحلة 7: تجميع أسبوعي حسب التاريخ الحقيقي (السوق السعودي: أحد -> خميس)
# ============================================================

def week_start_of(date):
    # Python: Mon=0 .. Sun=6 -> نريد الأسبوع يبدأ الأحد
    days_since_sunday = (date.weekday() + 1) % 7
    return date - pd.Timedelta(days=days_since_sunday)


def build_weekly(df):
    tmp = df.copy()
    tmp["week_start"] = tmp["date"].apply(week_start_of)

    weekly = tmp.groupby("week_start").agg(
        week_end=("date", "max"),
        sessions_in_week=("date", "count"),
        WeeklyOpen=("open", "first"),
        WeeklyHigh=("high", "max"),
        WeeklyLow=("low", "min"),
        WeeklyClose=("close", "last"),
        WeeklyVolume=("volume", "sum"),
    ).reset_index().rename(columns={"week_start": "week_start"})

    weekly = weekly.sort_values("week_start").reset_index(drop=True)

    weekly["WeeklyReturn_pct"] = weekly["WeeklyClose"].pct_change() * 100

    prev_avg_volume = weekly["WeeklyVolume"].expanding().mean().shift(1)
    weekly["WeeklyVolumeRelative"] = weekly["WeeklyVolume"] / prev_avg_volume

    rng = weekly["WeeklyHigh"] - weekly["WeeklyLow"]
    weekly["WeeklyCloseStrength"] = np.where(
        rng == 0, 0.5, (weekly["WeeklyClose"] - weekly["WeeklyLow"]) / rng
    )

    weekly["NextWeekReturn_pct"] = weekly["WeeklyReturn_pct"].shift(-1)

    return weekly


def save_weekly_csv(weekly_df):
    os.makedirs(DATA_DIR, exist_ok=True)
    weekly_df.to_csv(WEEKLY_CSV, index=False)


# ============================================================
# المرحلة 8: اختبار العلاقات الرياضية (Pearson + Spearman + R^2 + p-value)
# ============================================================

def correlation_test(x, y):
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x = np.asarray(x)[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
    y = np.asarray(y)[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
    n = len(x)

    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "r_squared": np.nan, "reg_p_value": np.nan}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)

    return {
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "r_squared": reg.rvalue ** 2,
        "reg_p_value": reg.pvalue,
    }


def run_relationship_tests(df, weekly_df):
    tests = {}

    tests["CloseStrength ~ RelativeVolume7"] = correlation_test(df["close_strength"], df["relative_volume7"])
    tests["DailyReturn ~ RelativeVolume7"] = correlation_test(df["daily_return"] * 100, df["relative_volume7"])
    tests["SmartMoneyPressure -> NextDayReturn"] = correlation_test(df["smart_money_pressure"], df["next_day_return_pct"])
    tests["SmartMoneyPressure -> Next3DayReturn"] = correlation_test(df["smart_money_pressure"], df["next_3day_return_pct"])
    tests["SmartMoneyPressure -> Next5DayReturn"] = correlation_test(df["smart_money_pressure"], df["next_5day_return_pct"])
    tests["WeeklyCloseStrength -> NextWeekReturn"] = correlation_test(weekly_df["WeeklyCloseStrength"], weekly_df["NextWeekReturn_pct"])

    return tests


# ============================================================
# المرحلتان 9 و 10: اختبار الفرضيات (أيام سيولة صاعدة/هابطة قوية)
# ============================================================

def group_stats(series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return {
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "pct_positive": (s > 0).mean() * 100,
        "max": s.max(),
        "min": s.min(),
    }


def hypothesis_test(df, flag_col, horizon_col):
    flagged = df.loc[df[flag_col], horizon_col].dropna()
    rest = df.loc[~df[flag_col], horizon_col].dropna()

    result = {
        "flagged": group_stats(flagged),
        "rest": group_stats(rest),
        "mannwhitney_p": np.nan,
        "ttest_p": np.nan,
        "reliable": False,
    }

    if len(flagged) >= MIN_GROUP_SAMPLE and len(rest) >= MIN_GROUP_SAMPLE:
        try:
            mw = stats.mannwhitneyu(flagged, rest, alternative="two-sided")
            result["mannwhitney_p"] = mw.pvalue
        except ValueError:
            pass
        tt = stats.ttest_ind(flagged, rest, equal_var=False)
        result["ttest_p"] = tt.pvalue
        result["reliable"] = True

    return result


def print_hypothesis_block(title, df, flag_col):
    section(title)
    count = int(df[flag_col].sum())
    print(f"عدد الجلسات المطابقة ضمن {len(df)} جلسة متاحة: {count}")

    if count == 0:
        print("لا توجد أي جلسة مطابقة لهذه الفرضية ضمن البيانات المتاحة حاليًا (49 جلسة). لا نتائج لعرضها.")
        return {}

    horizons = {
        "NextDayReturn": "next_day_return_pct",
        "Next3DayReturn": "next_3day_return_pct",
        "Next5DayReturn": "next_5day_return_pct",
    }

    results = {}
    for label, col in horizons.items():
        res = hypothesis_test(df, flag_col, col)
        results[label] = res

        print(f"\n{label}:")
        if res["flagged"] is None:
            print("  لا توجد قيم عائد مستقبلي متاحة (نهاية العينة الزمنية).")
            continue

        f = res["flagged"]
        print(f"  المجموعة المطابقة   : n={f['n']:>3}  mean={f['mean']:+.2f}%  median={f['median']:+.2f}%  "
              f"%إيجابي={f['pct_positive']:.0f}%  أعلى={f['max']:+.2f}%  أدنى={f['min']:+.2f}%")

        if res["rest"] is not None:
            r = res["rest"]
            print(f"  باقي الجلسات        : n={r['n']:>3}  mean={r['mean']:+.2f}%  median={r['median']:+.2f}%  "
                  f"%إيجابي={r['pct_positive']:.0f}%  أعلى={r['max']:+.2f}%  أدنى={r['min']:+.2f}%")

        if res["reliable"]:
            print(f"  Mann-Whitney U p-value = {res['mannwhitney_p']:.4f}   |   Welch t-test p-value = {res['ttest_p']:.4f}")
            sig = "دلالة إحصائية (p<0.05)" if (not pd.isna(res['mannwhitney_p']) and res['mannwhitney_p'] < 0.05) else "بدون دلالة إحصائية واضحة (p>=0.05)"
            print(f"  => {sig}")
        else:
            print(f"  ⚠️ العينة صغيرة جدًا (أقل من {MIN_GROUP_SAMPLE} في إحدى المجموعتين) — لا يمكن إجراء اختبار إحصائي موثوق. "
                  "الأرقام أعلاه وصفية فقط ولا تُعتبر نتيجة نهائية.")

    return results


# ============================================================
# حفظ نتائج التحليل الموحّد
# ============================================================

def save_analysis_csv(relationship_tests, bullish_results, bearish_results):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []

    for name, res in relationship_tests.items():
        rows.append({
            "type": "correlation",
            "label": name,
            "n": res["n"],
            "pearson_r": res["pearson_r"],
            "pearson_p": res["pearson_p"],
            "spearman_r": res["spearman_r"],
            "spearman_p": res["spearman_p"],
            "r_squared": res["r_squared"],
            "reg_p_value": res["reg_p_value"],
            "mean": np.nan, "median": np.nan, "pct_positive": np.nan,
            "max": np.nan, "min": np.nan,
            "mannwhitney_p": np.nan, "ttest_p": np.nan,
        })

    for group_name, results in (("StrongBullishLiquidityDay", bullish_results), ("StrongBearishLiquidityDay", bearish_results)):
        for horizon, res in results.items():
            if not res or res.get("flagged") is None:
                continue
            f = res["flagged"]
            rows.append({
                "type": "hypothesis_test",
                "label": f"{group_name}:{horizon}",
                "n": f["n"],
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "r_squared": np.nan, "reg_p_value": np.nan,
                "mean": f["mean"], "median": f["median"], "pct_positive": f["pct_positive"],
                "max": f["max"], "min": f["min"],
                "mannwhitney_p": res.get("mannwhitney_p"), "ttest_p": res.get("ttest_p"),
            })

    pd.DataFrame(rows).to_csv(ANALYSIS_CSV, index=False)


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 60)
    print("TADAWUL:4140 - ADVANCED MATHEMATICAL ANALYSIS")
    print("Close Location | Liquidity Strength | Smart Money Pressure")
    print("=" * 60)

    df = load_daily()
    print(f"\nتم تحميل {len(df)} جلسة من {DAILY_CSV} (بيانات تاريخية فعلية، بدون اتصال جديد).")

    df = add_close_strength(df)
    df = add_volume_strength(df)
    df = add_normalized_move(df)
    df = add_pressure_indicators(df)
    df = add_smart_money_pressure(df)
    df = add_forward_returns(df)
    df = add_signal_flags(df)

    save_daily_csv(df)
    save_signals_csv(df)

    weekly_df = build_weekly(df)
    save_weekly_csv(weekly_df)

    # ------------------------------------------------------------
    # [1] عيّنة من الميزات اليومية المحسوبة
    # ------------------------------------------------------------
    section("[1] Daily Features (sample)")
    daily_cols = ["date", "close", "close_strength", "relative_volume7", "relative_volume28",
                  "volume_zscore28", "normalized_move", "liquidity_pressure",
                  "close_volume_strength", "smart_money_pressure"]
    print("أول 5 جلسات:")
    print(fmt(df.head(5)[daily_cols]).to_string(index=False))
    print("\nآخر 5 جلسات:")
    print(fmt(df.tail(5)[daily_cols]).to_string(index=False))

    # ------------------------------------------------------------
    # [2] التجميع الأسبوعي الحقيقي
    # ------------------------------------------------------------
    section("[2] Weekly Aggregation (real calendar weeks, Sun-Thu)")
    week_cols = ["week_start", "week_end", "sessions_in_week", "WeeklyOpen", "WeeklyHigh",
                 "WeeklyLow", "WeeklyClose", "WeeklyVolume", "WeeklyReturn_pct",
                 "WeeklyVolumeRelative", "WeeklyCloseStrength"]
    print(fmt(weekly_df[week_cols], int_cols=["WeeklyVolume"]).to_string(index=False))
    print(f"\nإجمالي عدد الأسابيع المكتشفة من {len(df)} جلسة: {len(weekly_df)}")
    last28_dates = df.tail(28)["date"]
    weeks_in_last28 = weekly_df[weekly_df["week_end"] >= last28_dates.min()]
    print(f"عدد الأسابيع ضمن آخر 28 جلسة (~{last28_dates.min().date()} -> {last28_dates.max().date()}): {len(weeks_in_last28)}")

    # ------------------------------------------------------------
    # [3] اختبار العلاقات الرياضية الست
    # ------------------------------------------------------------
    section("[3] Relationship Tests (Pearson + Spearman + R^2 + p-value)")
    relationship_tests = run_relationship_tests(df, weekly_df)

    rel_rows = []
    for name, res in relationship_tests.items():
        rel_rows.append({
            "relationship": name,
            "n": res["n"],
            "pearson_r": res["pearson_r"],
            "pearson_p": res["pearson_p"],
            "spearman_r": res["spearman_r"],
            "spearman_p": res["spearman_p"],
            "r_squared": res["r_squared"],
        })
    rel_df = pd.DataFrame(rel_rows)
    print(fmt(rel_df, decimals=4).to_string(index=False))
    for name, res in relationship_tests.items():
        if res["n"] < MIN_CORR_SAMPLE:
            print(f"  ⚠️ '{name}': عدد نقاط العينة n={res['n']} أقل من {MIN_CORR_SAMPLE} — النتيجة إرشادية فقط وغير موثوقة إحصائيًا.")

    # ------------------------------------------------------------
    # [4] و [5] اختبار الفرضيات
    # ------------------------------------------------------------
    bullish_results = print_hypothesis_block(
        "[4] Hypothesis Test: StrongBullishLiquidityDay (CloseStrength>=0.70, RelVol7>=1.30, DailyReturn>0)",
        df, "strong_bullish_liquidity_day",
    )
    bearish_results = print_hypothesis_block(
        "[5] Hypothesis Test: StrongBearishLiquidityDay (CloseStrength<=-0.70, RelVol7>=1.30, DailyReturn<0)",
        df, "strong_bearish_liquidity_day",
    )

    save_analysis_csv(relationship_tests, bullish_results, bearish_results)

    # ------------------------------------------------------------
    # [6] أفضل أفق زمني
    # ------------------------------------------------------------
    section("[6] Best Time Horizon (SmartMoneyPressure -> forward return)")
    horizon_map = {
        "1 يوم (NextDayReturn)": relationship_tests["SmartMoneyPressure -> NextDayReturn"],
        "3 أيام (Next3DayReturn)": relationship_tests["SmartMoneyPressure -> Next3DayReturn"],
        "5 أيام (Next5DayReturn)": relationship_tests["SmartMoneyPressure -> Next5DayReturn"],
        "أسبوع (WeeklyCloseStrength -> NextWeekReturn)": relationship_tests["WeeklyCloseStrength -> NextWeekReturn"],
    }
    significant = {
        label: res for label, res in horizon_map.items()
        if not pd.isna(res["pearson_p"]) and res["pearson_p"] < 0.05 and res["n"] >= MIN_CORR_SAMPLE
    }

    if significant:
        best_label, best_res = min(significant.items(), key=lambda kv: kv[1]["pearson_p"])
    else:
        best_label, best_res = None, None

    for label, res in horizon_map.items():
        sig_flag = (not pd.isna(res["pearson_p"])) and res["pearson_p"] < 0.05
        marker = ""
        if label == best_label:
            marker = " <== أقوى أفق ذو دلالة إحصائية"
        elif sig_flag:
            marker = " (دالّ إحصائيًا لكن أضعف)"
        elif res["n"] < MIN_CORR_SAMPLE:
            marker = " (عينة صغيرة، غير موثوقة)"
        print(f"  {label:45s} n={res['n']:>3}  Pearson r={res['pearson_r']:.4f}  p={res['pearson_p']:.4f}{marker}")

    if best_label is None:
        print("\n  => لا يوجد أفق زمني وصل لدلالة إحصائية موثوقة (p<0.05) بعينة كافية (n>=10) ضمن البيانات الحالية.")
    else:
        print(f"\n  => الأفق الأكثر دلالة إحصائيًا هو: {best_label} (p={best_res['pearson_p']:.4f}, n={best_res['n']})")

    # ------------------------------------------------------------
    # [7] الجدول الختامي
    # ------------------------------------------------------------
    section("[7] Final Summary Table")
    print(fmt(rel_df, decimals=4).to_string(index=False))

    print(f"\nتم حفظ الملفات التالية:")
    print(f"  - {DAILY_CSV}")
    print(f"  - {WEEKLY_CSV}")
    print(f"  - {ANALYSIS_CSV}")
    print(f"  - {SIGNALS_CSV}")


if __name__ == "__main__":
    main()
