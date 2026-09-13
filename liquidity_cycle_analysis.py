"""
تحليل النمط الزمني والدورية الرياضية لظهور السيولة المرتفعة في TADAWUL:4140
اعتمادًا على 180 جلسة حقيقية محفوظة مسبقًا في data/4140_180_daily.csv (قراءة فقط).

لا علاقة له بـ tv_test.py ولا يستخدم بيانات لحظية أو وهمية.

يوفّر دوال قابلة لإعادة الاستخدام من liquidity_forecast.py و plot_liquidity_cycle.py،
ويحفظ عند تشغيله مباشرة:
    data/4140_liquidity_cycles.csv
"""

import os

import numpy as np
import pandas as pd
from scipy import stats, signal

DATA_DIR = "data"
DAILY_180_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")  # قراءة فقط
CYCLES_OUT_CSV = os.path.join(DATA_DIR, "4140_liquidity_cycles.csv")

DAYS_PER_CYCLE = 15
VOL_WINDOW = 20
LEVELS = {"L1": 1.25, "L2": 1.50, "L3": 2.00}
CANDIDATE_PERIODS = [5, 10, 15, 20, 25, 30]

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
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.1f}%")
        elif pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
    return view


# ============================================================
# 0) تحميل البيانات وبناء المتغيرات الأساسية
# ============================================================

def load_daily():
    if not os.path.exists(DAILY_180_CSV):
        raise FileNotFoundError(f"لم يتم العثور على {DAILY_180_CSV}. شغّل historical_180.py أولًا.")
    df = pd.read_csv(DAILY_180_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_liquidity_features(raw):
    df = raw.copy()

    vol_mean20 = df["volume"].rolling(VOL_WINDOW).mean()
    vol_std20 = df["volume"].rolling(VOL_WINDOW).std()

    df["relative_volume"] = df["volume"] / vol_mean20
    zscore = (df["volume"] - vol_mean20) / vol_std20
    df["volume_zscore"] = zscore.where(vol_std20 > 0, 0.0)

    for level_name, threshold in LEVELS.items():
        df[f"high_liquidity_{level_name}"] = df["relative_volume"] > threshold

    df["session_number"] = np.arange(1, len(df) + 1)
    df["day_in_cycle"] = ((df["session_number"] - 1) % DAYS_PER_CYCLE) + 1
    df["cycle_number"] = ((df["session_number"] - 1) // DAYS_PER_CYCLE) + 1

    return df


# ============================================================
# 1) احتمال السيولة المرتفعة حسب موقع اليوم داخل الدورة (DayInCycle)
# ============================================================

def day_in_cycle_table(df, up_to_row=None):
    """
    up_to_row: إن حُدِّد، يُستخدم فقط الصفوف حتى هذا الفهرس (exclusive) — لأغراض
    Walk-Forward (منع أي تسرّب معلومات من المستقبل).
    """
    work = df if up_to_row is None else df.iloc[:up_to_row]

    rows = []
    for day in range(1, DAYS_PER_CYCLE + 1):
        sub = work[work["day_in_cycle"] == day]
        relvol = sub["relative_volume"].dropna()

        row = {
            "DayInCycle": day,
            "AvailableCycles": len(relvol),
            "AverageRelativeVolume": relvol.mean() if len(relvol) else np.nan,
            "MedianRelativeVolume": relvol.median() if len(relvol) else np.nan,
        }

        for level_name in LEVELS:
            flags = sub[f"high_liquidity_{level_name}"].dropna()
            row[f"HighLiquidityCount_{level_name}"] = int(flags.sum())
            row[f"ProbabilityHighLiquidity_{level_name}"] = (flags.mean() * 100) if len(flags) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def test_day_in_cycle_uniformity(df, level_name="L2"):
    """
    اختبار Chi-square: هل توزيع أيام السيولة المرتفعة عبر الـ15 يومًا مختلف عن التوزيع
    المنتظر عشوائيًا (كل الأيام لها نفس احتمال السيولة المرتفعة)؟
    """
    flag_col = f"high_liquidity_{level_name}"
    valid = df.dropna(subset=["relative_volume"])
    overall_p = valid[flag_col].mean()

    observed, expected = [], []
    for day in range(1, DAYS_PER_CYCLE + 1):
        sub = valid[valid["day_in_cycle"] == day]
        n_available = len(sub)
        observed.append(sub[flag_col].sum())
        expected.append(overall_p * n_available)

    low_expected = sum(1 for e in expected if e < 5)
    if sum(expected) == 0 or any(e == 0 for e in expected):
        return {"chi2": np.nan, "p_value": np.nan, "low_expected_cells": low_expected, "reliable": False}

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    return {"chi2": chi2, "p_value": p_value, "low_expected_cells": low_expected, "reliable": low_expected == 0}


# ============================================================
# 2) موقع السيولة داخل كل دورة (أعلى 3 أيام) + اختبار تركّز عشوائي (Monte Carlo)
# ============================================================

def per_cycle_top_days(df, n_top=3):
    rows = []
    for cycle in sorted(df["cycle_number"].unique()):
        sub = df[df["cycle_number"] == cycle].dropna(subset=["relative_volume"])
        if len(sub) < n_top:
            rows.append({"Cycle": cycle, "HighestLiquidityDay": np.nan, "SecondHighestDay": np.nan, "ThirdHighestDay": np.nan, "valid": False})
            continue
        top = sub.sort_values("relative_volume", ascending=False).head(n_top)["day_in_cycle"].tolist()
        rows.append({
            "Cycle": cycle,
            "HighestLiquidityDay": top[0],
            "SecondHighestDay": top[1] if len(top) > 1 else np.nan,
            "ThirdHighestDay": top[2] if len(top) > 2 else np.nan,
            "valid": True,
        })
    return pd.DataFrame(rows)


def highest_day_distribution_test(top_days_df, n_permutations=20000, seed=42):
    valid = top_days_df[top_days_df["valid"]]
    n_valid = len(valid)
    if n_valid == 0:
        return None, None

    counts = valid["HighestLiquidityDay"].value_counts().reindex(range(1, DAYS_PER_CYCLE + 1), fill_value=0)
    observed_max = counts.max()

    rng = np.random.default_rng(seed)
    sim_max = np.empty(n_permutations)
    for i in range(n_permutations):
        sim = rng.integers(1, DAYS_PER_CYCLE + 1, size=n_valid)
        sim_counts = np.bincount(sim, minlength=DAYS_PER_CYCLE + 1)[1:]
        sim_max[i] = sim_counts.max()

    p_value = (sim_max >= observed_max).mean()

    return counts, {"n_valid_cycles": n_valid, "observed_max_count": int(observed_max), "monte_carlo_p_value": p_value}


# ============================================================
# 3) تحليل الفاصل الزمني (Gap) بين أيام السيولة المرتفعة
# ============================================================

def gap_analysis(df, level_name="L2"):
    flag_col = f"high_liquidity_{level_name}"
    sessions = df.loc[df[flag_col] == True, "session_number"].to_numpy()  # noqa: E712
    if len(sessions) < 2:
        return None, sessions

    gaps = np.diff(sessions)
    mode_result = stats.mode(gaps, keepdims=True)

    summary = {
        "n_events": len(sessions),
        "n_gaps": len(gaps),
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "std_gap": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else np.nan,
        "mode_gap": float(mode_result.mode[0]),
        "min_gap": int(np.min(gaps)),
        "max_gap": int(np.max(gaps)),
    }
    return summary, gaps


# ============================================================
# 4) التحليل الدوري الرياضي: ACF / PACF / Periodogram / FFT / Peak detection
# ============================================================

def acf_manual(series, max_lag):
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    acf_vals = np.empty(max_lag + 1)
    acf_vals[0] = 1.0
    for k in range(1, max_lag + 1):
        cov = np.dot(x[:n - k], x[k:]) / n
        acf_vals[k] = cov / var if var > 0 else 0.0
    return acf_vals


def pacf_durbin_levinson(acf_vals, max_lag):
    """PACF عبر تكرار Durbin-Levinson انطلاقًا من قيم ACF."""
    phi = np.zeros((max_lag + 1, max_lag + 1))
    pacf_vals = np.zeros(max_lag + 1)
    pacf_vals[0] = 1.0

    if max_lag >= 1:
        phi[1, 1] = acf_vals[1]
        pacf_vals[1] = phi[1, 1]

    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - j] for j in range(1, k))
        den = 1 - sum(phi[k - 1, j] * acf_vals[j] for j in range(1, k))
        phi[k, k] = num / den if den != 0 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_vals[k] = phi[k, k]

    return pacf_vals


def ljung_box_test(acf_vals, n, max_lag):
    """اختبار Ljung-Box المشترك: هل السلسلة ضجيج أبيض (بدون أي ارتباط ذاتي) حتى max_lag؟"""
    q_stat = n * (n + 2) * sum((acf_vals[k] ** 2) / (n - k) for k in range(1, max_lag + 1))
    p_value = stats.chi2.sf(q_stat, df=max_lag)
    return q_stat, p_value


def periodicity_analysis(relvol_series, max_lag=40):
    series = relvol_series.dropna().to_numpy(dtype=float)
    n = len(series)
    max_lag = min(max_lag, n // 3)

    acf_vals = acf_manual(series, max_lag)
    pacf_vals = pacf_durbin_levinson(acf_vals, max_lag)

    sig_band = 1.96 / np.sqrt(n)  # الحد التقريبي لدلالة ACF لسلسلة ضجيج أبيض (95%)

    # Periodogram (Welch/كلاسيكي عبر scipy.signal.periodogram)
    freqs, power = signal.periodogram(series - series.mean(), fs=1.0)
    valid = freqs > 0
    freqs, power = freqs[valid], power[valid]
    periods = 1.0 / freqs

    # FFT (للتأكيد المتقاطع مع periodogram)
    fft_vals = np.fft.rfft(series - series.mean())
    fft_freqs = np.fft.rfftfreq(n, d=1.0)
    fft_mag = np.abs(fft_vals)
    fft_valid = fft_freqs > 0
    fft_periods = 1.0 / fft_freqs[fft_valid]
    fft_mag = fft_mag[fft_valid]

    # Peak detection على الـ periodogram
    peak_idx, _ = signal.find_peaks(power)
    peak_periods = periods[peak_idx]
    peak_power = power[peak_idx]
    top_peaks = sorted(zip(peak_periods, peak_power), key=lambda t: t[1], reverse=True)[:8]

    # اختبار الدورات المرشّحة تحديدًا (5, 10, 15, ...)
    candidate_rows = []
    for period in CANDIDATE_PERIODS:
        lag = period
        acf_at_lag = acf_vals[lag] if lag < len(acf_vals) else np.nan
        acf_significant = (not np.isnan(acf_at_lag)) and abs(acf_at_lag) > sig_band

        closest_idx = np.argmin(np.abs(periods - period))
        power_at_period = power[closest_idx]
        power_rank = int((power > power_at_period).sum()) + 1  # رتبة القوة (1 = الأقوى)

        candidate_rows.append({
            "period": period,
            "acf": acf_at_lag,
            "acf_significant_95pct": acf_significant,
            "periodogram_power": power_at_period,
            "power_rank": power_rank,
            "n_frequencies": len(power),
        })

    candidates_df = pd.DataFrame(candidate_rows).sort_values("power_rank")

    q_stat, lb_p_value = ljung_box_test(acf_vals, n, max_lag)

    return {
        "n": n,
        "max_lag": max_lag,
        "acf": acf_vals,
        "pacf": pacf_vals,
        "sig_band": sig_band,
        "periods": periods,
        "power": power,
        "fft_periods": fft_periods,
        "fft_mag": fft_mag,
        "top_periodogram_peaks": top_peaks,
        "candidates": candidates_df,
        "ljung_box_stat": q_stat,
        "ljung_box_p": lb_p_value,
    }


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 70)
    print("TADAWUL:4140 - LIQUIDITY CYCLE ANALYSIS (180 SESSIONS)")
    print("DayInCycle Pattern | Gap Analysis | Periodicity (ACF/PACF/FFT/Periodogram)")
    print("=" * 70)

    raw = load_daily()
    df = build_liquidity_features(raw)
    n_sessions = len(df)
    n_cycles = n_sessions // DAYS_PER_CYCLE
    print(f"\nعدد الجلسات: {n_sessions}  |  عدد الدورات الكاملة ({DAYS_PER_CYCLE} جلسة/دورة): {n_cycles}")

    # ------------------------------------------------------------
    # [1] احتمال السيولة حسب DayInCycle
    # ------------------------------------------------------------
    section("[1] DayInCycle Probability Table (Level 1 / 2 / 3)")
    cycle_table = day_in_cycle_table(df)
    show_cols = ["DayInCycle", "AvailableCycles", "AverageRelativeVolume", "MedianRelativeVolume"] + \
                [c for lvl in LEVELS for c in (f"HighLiquidityCount_{lvl}", f"ProbabilityHighLiquidity_{lvl}")]
    print(fmt(cycle_table[show_cols], int_cols=["DayInCycle", "AvailableCycles"] + [f"HighLiquidityCount_{l}" for l in LEVELS]).to_string(index=False))

    print("\nترتيب الأيام (Level 2, RelativeVolume>1.5) من الأكثر احتمالًا:")
    ranked = cycle_table.sort_values("ProbabilityHighLiquidity_L2", ascending=False)
    for _, r in ranked.head(5).iterrows():
        print(f"  Day {int(r['DayInCycle']):>2} = {r['ProbabilityHighLiquidity_L2']:.0f}%  (N={int(r['HighLiquidityCount_L2'])}/{int(r['AvailableCycles'])})")

    uniformity = test_day_in_cycle_uniformity(df, "L2")
    print(f"\nChi-square Uniformity Test (Level 2): chi2={uniformity['chi2']:.3f}  p-value={uniformity['p_value']:.4f}  "
          f"(خلايا بتوقّع<5: {uniformity['low_expected_cells']}/15 — {'⚠️ قد يكون التقريب غير موثوق' if not uniformity['reliable'] else 'موثوق'})")

    # ------------------------------------------------------------
    # [2] أعلى 3 أيام سيولة لكل دورة + اختبار Monte Carlo
    # ------------------------------------------------------------
    section("[2] Per-Cycle Top-3 Liquidity Days")
    top_days_df = per_cycle_top_days(df)
    print(fmt(top_days_df[["Cycle", "HighestLiquidityDay", "SecondHighestDay", "ThirdHighestDay"]],
              int_cols=["Cycle", "HighestLiquidityDay", "SecondHighestDay", "ThirdHighestDay"]).to_string(index=False))

    counts, mc_result = highest_day_distribution_test(top_days_df)
    if counts is not None:
        print("\nتوزيع ظهور 'أعلى يوم سيولة' عبر الدورات الصالحة:")
        for day, c in counts.items():
            if c > 0:
                print(f"  Day {day:>2}: {c} مرة")
        print(f"\nMonte Carlo Test (H0: التوزيع عشوائي منتظم): n_valid_cycles={mc_result['n_valid_cycles']}  "
              f"observed_max_count={mc_result['observed_max_count']}  p-value={mc_result['monte_carlo_p_value']:.4f}")
        sig = "✅ تركّز دالّ إحصائيًا (p<0.05)" if mc_result["monte_carlo_p_value"] < 0.05 else "❌ لا يوجد دليل كافٍ على تركّز حقيقي (التوزيع متوافق مع العشوائية)"
        print(f"  => {sig}")

    # ------------------------------------------------------------
    # [3] تحليل الفجوات الزمنية
    # ------------------------------------------------------------
    section("[3] Gap Analysis between High-Liquidity Days (Level 2, RelativeVolume>1.5)")
    gap_summary, gaps = gap_analysis(df, "L2")
    if gap_summary is None:
        print("عدد أحداث السيولة المرتفعة غير كافٍ لحساب الفجوات.")
    else:
        print(f"عدد الأحداث: {gap_summary['n_events']}  |  عدد الفجوات: {gap_summary['n_gaps']}")
        print(f"Mean Gap = {gap_summary['mean_gap']:.2f} جلسة  |  Median = {gap_summary['median_gap']:.1f}  |  "
              f"Std = {gap_summary['std_gap']:.2f}  |  Mode = {gap_summary['mode_gap']:.0f}  |  "
              f"Min={gap_summary['min_gap']}  Max={gap_summary['max_gap']}")
        vals, counts_g = np.unique(gaps, return_counts=True)
        dist_str = "  ".join(f"{v}:{c}" for v, c in zip(vals, counts_g))
        print(f"Distribution (gap:count): {dist_str}")

    # ------------------------------------------------------------
    # [4] التحليل الدوري الرياضي
    # ------------------------------------------------------------
    section("[4] Periodicity Analysis (ACF / PACF / Periodogram / FFT / Peak Detection)")
    periodicity = periodicity_analysis(df["relative_volume"])
    print(f"n (بعد إزالة NaN) = {periodicity['n']}  |  max_lag مُختبر = {periodicity['max_lag']}")
    print(f"حد الدلالة التقريبي لـ ACF (95%, ضجيج أبيض) = ±{periodicity['sig_band']:.4f}")
    print(f"Ljung-Box Test (ارتباط ذاتي مشترك حتى lag={periodicity['max_lag']}): "
          f"Q={periodicity['ljung_box_stat']:.2f}  p-value={periodicity['ljung_box_p']:.4f}  "
          f"=> {'✅ يوجد ارتباط ذاتي دالّ إحصائيًا (السلسلة ليست ضجيجًا أبيض بحتًا)' if periodicity['ljung_box_p'] < 0.05 else '❌ لا يوجد دليل كافٍ على ارتباط ذاتي (متوافقة مع ضجيج أبيض)'}")

    print("\nاختبار الدورات المرشّحة (5, 10, 15, 20, 25, 30 جلسة) — مرتّبة حسب قوة Periodogram:")
    print(fmt(periodicity["candidates"]).to_string(index=False))

    print("\nأقوى القمم المكتشفة عبر Periodogram + Peak Detection (أعلى 8):")
    for period, power in periodicity["top_periodogram_peaks"]:
        print(f"  دورة ≈ {period:.1f} جلسة   قوة الطيف = {power:.5f}")

    strongest_candidate = periodicity["candidates"].sort_values("power_rank").iloc[0]

    save_all_outputs(df, cycle_table, uniformity, top_days_df, mc_result, gap_summary, gaps, periodicity)

    return df, cycle_table, top_days_df, gap_summary, periodicity


def save_all_outputs(df, cycle_table, uniformity, top_days_df, mc_result, gap_summary, gaps, periodicity):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []

    for _, r in cycle_table.iterrows():
        rows.append({
            "section": "day_in_cycle", "key": f"Day{int(r['DayInCycle'])}",
            "AvailableCycles": r["AvailableCycles"], "AvgRelVol": r["AverageRelativeVolume"],
            "MedianRelVol": r["MedianRelativeVolume"],
            "Prob_L1": r["ProbabilityHighLiquidity_L1"], "Prob_L2": r["ProbabilityHighLiquidity_L2"],
            "Prob_L3": r["ProbabilityHighLiquidity_L3"],
            "value": np.nan, "p_value": np.nan,
        })

    rows.append({"section": "uniformity_test", "key": "chi2_L2", "value": uniformity["chi2"], "p_value": uniformity["p_value"]})

    if mc_result:
        rows.append({"section": "highest_day_concentration", "key": "monte_carlo",
                      "value": mc_result["observed_max_count"], "p_value": mc_result["monte_carlo_p_value"]})

    if gap_summary:
        for k, v in gap_summary.items():
            rows.append({"section": "gap_analysis", "key": k, "value": v, "p_value": np.nan})

    rows.append({"section": "periodicity", "key": "ljung_box_stat", "value": periodicity["ljung_box_stat"], "p_value": periodicity["ljung_box_p"]})

    for _, r in periodicity["candidates"].iterrows():
        rows.append({
            "section": "candidate_period", "key": f"period_{int(r['period'])}",
            "value": r["acf"], "p_value": np.nan,
            "power_rank": r["power_rank"], "acf_significant": r["acf_significant_95pct"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CYCLES_OUT_CSV, index=False)
    print(f"\nتم حفظ: {CYCLES_OUT_CSV}")


if __name__ == "__main__":
    main()
