"""
يبني الرسوم البيانية وتقرير HTML لتحليل دورية السيولة في TADAWUL:4140، اعتمادًا على
liquidity_cycle_analysis.py و liquidity_forecast.py (بدون إعادة الاتصال بأي مصدر بيانات).

يُنتج: data/4140_liquidity_cycle_report.html
يستخدم matplotlib فقط للرسم.
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from liquidity_cycle_analysis import (
    DATA_DIR, DAYS_PER_CYCLE, LEVELS, CANDIDATE_PERIODS,
    load_daily, build_liquidity_features, day_in_cycle_table, test_day_in_cycle_uniformity,
    per_cycle_top_days, highest_day_distribution_test, gap_analysis, periodicity_analysis,
)
from liquidity_forecast import (
    FORECAST_OUT_CSV, run_walk_forward, evaluation_metrics, bootstrap_auc_ci, forecast_next_sessions,
)

REPORT_HTML = os.path.join(DATA_DIR, "4140_liquidity_cycle_report.html")


def fig_to_base64(fig, png_path):
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def df_to_html_table(df, decimals=4, pct_cols=(), int_cols=()):
    view = df.copy()
    for col in view.columns:
        if col in int_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
        elif col in pct_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.1f}%")
        elif pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
        elif pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda x: "YES" if x else "")
    return view.to_html(index=False, border=0, classes="data-table", na_rep="")


# ============================================================
# الرسوم
# ============================================================

def chart_day_in_cycle(cycle_table):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(1, DAYS_PER_CYCLE + 1)
    width = 0.25
    for i, lvl in enumerate(LEVELS):
        ax.bar(x + (i - 1) * width, cycle_table[f"ProbabilityHighLiquidity_{lvl}"], width=width, label=f"Level {lvl[-1]} (>{LEVELS[lvl]})")
    ax.set_xticks(x)
    ax.set_xlabel("DayInCycle")
    ax.set_ylabel("Probability of High Liquidity (%)")
    ax.set_title("Chart 1 - High Liquidity Probability by DayInCycle")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_1_day_in_cycle.png"))


def chart_heatmap(df):
    pivot = df.pivot_table(index="cycle_number", columns="day_in_cycle", values="relative_volume")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanpercentile(pivot.to_numpy(), 95))
    ax.set_xticks(np.arange(DAYS_PER_CYCLE))
    ax.set_xticklabels(np.arange(1, DAYS_PER_CYCLE + 1))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("DayInCycle")
    ax.set_ylabel("Cycle")
    ax.set_title("Chart 2 - RelativeVolume Heatmap (Cycle x DayInCycle)")
    fig.colorbar(im, ax=ax, label="RelativeVolume")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_2_heatmap.png"))


def chart_gap_distribution(gaps):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if gaps is None or len(gaps) == 0:
        ax.text(0.5, 0.5, "لا توجد بيانات كافية", ha="center", va="center")
    else:
        max_gap = int(gaps.max())
        bins = np.arange(1, max_gap + 2) - 0.5
        ax.hist(gaps, bins=bins, color="teal", edgecolor="black", alpha=0.8)
        ax.axvline(np.mean(gaps), color="red", linestyle="--", label=f"Mean={np.mean(gaps):.1f}")
        ax.axvline(np.median(gaps), color="orange", linestyle=":", label=f"Median={np.median(gaps):.1f}")
        ax.legend()
    ax.set_xlabel("Gap (sessions)")
    ax.set_ylabel("Frequency")
    ax.set_title("Chart 3 - Gap Distribution Between High-Liquidity Days (Level 2)")
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_3_gap_distribution.png"))


def chart_acf(periodicity):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    lags = np.arange(len(periodicity["acf"]))
    ax.bar(lags, periodicity["acf"], color="tab:blue", width=0.6)
    band = periodicity["sig_band"]
    ax.axhline(band, color="red", linestyle="--", linewidth=0.8, label="95% significance band")
    ax.axhline(-band, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    for p in CANDIDATE_PERIODS:
        if p < len(lags):
            ax.axvline(p, color="green", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Lag (sessions)")
    ax.set_ylabel("ACF")
    ax.set_title("Chart 4 - Autocorrelation Function of RelativeVolume (candidate periods marked)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_4_acf.png"))


def chart_periodogram(periodicity):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(periodicity["periods"], periodicity["power"], color="tab:purple", linewidth=1.0)
    for p in CANDIDATE_PERIODS:
        ax.axvline(p, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(2, 60)
    ax.set_xlabel("Period (sessions)")
    ax.set_ylabel("Spectral Power")
    ax.set_title("Chart 5 - Periodogram of RelativeVolume (candidate periods marked in green)")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_5_periodogram.png"))


def chart_walk_forward(wf_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(wf_results["date"], wf_results["predicted_liquidity_probability"], color="tab:blue", label="Predicted Probability", linewidth=1.0)
    actual = wf_results["actual_high_liquidity_bin"]
    ax1.scatter(wf_results.loc[actual == 1, "date"], [1.02] * int(actual.sum()), color="red", marker="v", s=25, label="Actual High Liquidity")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Predicted Probability vs Actual Events (Walk-Forward)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    y_true = wf_results["actual_high_liquidity_bin"].to_numpy()
    y_prob = wf_results["predicted_liquidity_probability"].to_numpy()
    thresholds = np.linspace(0, 1, 101)
    tprs, fprs = [], []
    n_pos, n_neg = y_true.sum(), (1 - y_true).sum()
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        tprs.append(tp / n_pos if n_pos else 0)
        fprs.append(fp / n_neg if n_neg else 0)
    ax2.plot(fprs, tprs, color="tab:red", linewidth=1.5, label="Model")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Chart 6 - ROC Curve (Walk-Forward Out-of-Sample)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_6_walk_forward_roc.png"))


def chart_forecast(forecast_df):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = forecast_df["forecast_probability"].fillna(0).apply(lambda p: "tab:red" if p >= 0.5 else "tab:blue")
    ax.bar(forecast_df["session_offset"], forecast_df["forecast_probability"] * 100, color=colors)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, label="50% threshold")
    ax.set_xlabel("Session Offset (from next session)")
    ax.set_ylabel("Forecast Probability (%)")
    ax.set_title("Chart 7 - Next 15 Sessions: High-Liquidity Forecast Probability")
    ax.set_xticks(forecast_df["session_offset"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_lc_chart_7_forecast.png"))


def main():
    raw = load_daily()
    df = build_liquidity_features(raw)

    cycle_table = day_in_cycle_table(df)
    uniformity = test_day_in_cycle_uniformity(df, "L2")
    top_days_df = per_cycle_top_days(df)
    counts, mc_result = highest_day_distribution_test(top_days_df)
    gap_summary, gaps = gap_analysis(df, "L2")
    periodicity = periodicity_analysis(df["relative_volume"], max_lag=40)

    if os.path.exists(FORECAST_OUT_CSV):
        combined = pd.read_csv(FORECAST_OUT_CSV, parse_dates=["date"])
        wf_results = combined[combined["phase"] == "walk_forward_eval"].copy()
        forecast_df = combined[combined["phase"] == "future_forecast"].copy()
    else:
        wf_results = run_walk_forward(df)
        forecast_df, _ = forecast_next_sessions(df)

    model_metrics = evaluation_metrics(wf_results["actual_high_liquidity_bin"], wf_results["predicted_liquidity_probability"])
    baseline_metrics = evaluation_metrics(wf_results["actual_high_liquidity_bin"], wf_results["baseline_probability"])
    auc_lo, auc_hi = bootstrap_auc_ci(wf_results["actual_high_liquidity_bin"], wf_results["predicted_liquidity_probability"])

    # ------------------------------------------------------------
    # الرسوم
    # ------------------------------------------------------------
    img1 = chart_day_in_cycle(cycle_table)
    img2 = chart_heatmap(df)
    img3 = chart_gap_distribution(gaps)
    img4 = chart_acf(periodicity)
    img5 = chart_periodogram(periodicity)
    img6 = chart_walk_forward(wf_results)
    img7 = chart_forecast(forecast_df)

    # ------------------------------------------------------------
    # جداول
    # ------------------------------------------------------------
    show_cols = ["DayInCycle", "AvailableCycles", "AverageRelativeVolume", "MedianRelativeVolume",
                 "HighLiquidityCount_L2", "ProbabilityHighLiquidity_L2"]
    day_table_html = df_to_html_table(cycle_table[show_cols].sort_values("ProbabilityHighLiquidity_L2", ascending=False),
                                       int_cols=["DayInCycle", "AvailableCycles", "HighLiquidityCount_L2"], pct_cols=["ProbabilityHighLiquidity_L2"])

    top_days_html = df_to_html_table(top_days_df[["Cycle", "HighestLiquidityDay", "SecondHighestDay", "ThirdHighestDay"]],
                                      int_cols=["Cycle", "HighestLiquidityDay", "SecondHighestDay", "ThirdHighestDay"])

    candidates_html = df_to_html_table(periodicity["candidates"])

    forecast_html = df_to_html_table(
        forecast_df[["session_offset", "day_in_cycle", "forecast_probability", "expected_relative_volume",
                     "high_liquidity_probability_label", "confidence_level"]],
        pct_cols=[], int_cols=["session_offset", "day_in_cycle"],
    )

    top3_sessions = forecast_df.sort_values("forecast_probability", ascending=False).head(3)
    top3_text = "، ".join(
        f"الجلسة +{int(r['session_offset'])} (Day {int(r['day_in_cycle'])}, احتمال {r['forecast_probability']*100:.1f}%)"
        for _, r in top3_sessions.iterrows()
    )

    strongest_candidate = periodicity["candidates"].sort_values("power_rank").iloc[0]
    period15_row = periodicity["candidates"][periodicity["candidates"]["period"] == 15].iloc[0]

    auc_ci_excludes_half = auc_lo > 0.5
    beats_baseline = (model_metrics["roc_auc"] > baseline_metrics["roc_auc"]) and (model_metrics["brier_score"] < baseline_metrics["brier_score"])

    if beats_baseline and auc_ci_excludes_half:
        oos_verdict = f"نعم، بأدلة معقولة (لكن على عيّنة صغيرة) — ROC-AUC={model_metrics['roc_auc']:.3f} خارج العينة، فاصل ثقة 95%: [{auc_lo:.3f}, {auc_hi:.3f}] (يستبعد 0.5)."
    elif beats_baseline:
        oos_verdict = f"تفوّق رقمي غير مؤكد إحصائيًا — ROC-AUC={model_metrics['roc_auc']:.3f} لكن فاصل الثقة [{auc_lo:.3f}, {auc_hi:.3f}] يشمل 0.5."
    else:
        oos_verdict = 'لا. "No statistically useful periodic liquidity pattern detected."'

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>TADAWUL:4140 - تحليل دورية السيولة</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Arial, sans-serif; background:#0f1115; color:#e8e8e8; margin:0; padding:0; }}
  .wrap {{ max-width: 1150px; margin: 0 auto; padding: 24px; }}
  header {{ background: linear-gradient(135deg,#1b2735,#0f1115); padding: 32px 24px; border-bottom: 3px solid #22c1a2; }}
  header h1 {{ margin:0 0 6px 0; font-size: 1.6rem; }}
  header p {{ margin:2px 0; color:#9fb3c8; }}
  section {{ background:#161a22; border:1px solid #262c37; border-radius:10px; padding:20px; margin:20px 0; }}
  h2 {{ color:#22c1a2; border-bottom:1px solid #262c37; padding-bottom:8px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin-top: 10px; }}
  table.data-table th, table.data-table td {{ border: 1px solid #2b323e; padding: 5px 8px; text-align: center; }}
  table.data-table th {{ background:#1f2733; color:#7fe6cf; }}
  table.data-table tr:nth-child(even) {{ background:#1a1f28; }}
  .note {{ background:#2a2410; border:1px solid #7a5c00; color:#f0d878; padding:10px 14px; border-radius:6px; font-size:0.9rem; }}
  .verdict-box {{ background:#12232a; border:1px solid #1f6f5c; color:#a8e6d9; padding:14px 18px; border-radius:8px; font-size:0.95rem; margin:10px 0; }}
  .chart-box {{ text-align:center; margin: 18px 0; }}
  .chart-box img {{ max-width:100%; border-radius:8px; border:1px solid #262c37; }}
  .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:10px; }}
  .summary-card {{ background:#1f2733; border-radius:8px; padding:14px; text-align:center; }}
  .summary-card .val {{ font-size:1.3rem; color:#22c1a2; font-weight:bold; }}
  ol.qa li {{ margin-bottom: 10px; }}
  footer {{ text-align:center; color:#666; padding:20px; font-size:0.8rem; }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>TADAWUL:4140 — تحليل دورية السيولة (Liquidity Cycle Analysis)</h1>
    <p>180 جلسة تداول فعلية — 12 دورة × 15 جلسة — تم إنشاء التقرير في {generated_at}</p>
  </div>
</header>
<div class="wrap">

<section>
  <h2>[1] احتمال السيولة المرتفعة حسب DayInCycle</h2>
  <p class="note">Chi-square Uniformity Test (Level 2): chi2={uniformity['chi2']:.3f}, p-value={uniformity['p_value']:.4f}
  ({'⚠️ خلايا كثيرة بتوقّع أقل من 5 — نتيجة تقريبية' if not uniformity['reliable'] else 'موثوق'})</p>
  {day_table_html}
  <div class="chart-box"><img src="data:image/png;base64,{img1}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img2}"></div>
</section>

<section>
  <h2>[2] أعلى 3 أيام سيولة لكل دورة + اختبار التركّز</h2>
  {top_days_html}
  <p class="note">Monte Carlo Test (H0: توزيع عشوائي منتظم): n_valid_cycles={mc_result['n_valid_cycles'] if mc_result else 'N/A'},
  observed_max_count={mc_result['observed_max_count'] if mc_result else 'N/A'}, p-value={(f"{mc_result['monte_carlo_p_value']:.4f}") if mc_result else 'N/A'}
  → {'تركّز دالّ إحصائيًا' if mc_result and mc_result['monte_carlo_p_value'] < 0.05 else 'لا دليل على تركّز حقيقي (متوافق مع العشوائية)'}</p>
</section>

<section>
  <h2>[3] الفاصل الزمني بين أيام السيولة المرتفعة</h2>
  <div class="summary-grid">
    <div class="summary-card"><div class="val">{gap_summary['mean_gap']:.2f}</div>Mean Gap</div>
    <div class="summary-card"><div class="val">{gap_summary['median_gap']:.1f}</div>Median Gap</div>
    <div class="summary-card"><div class="val">{gap_summary['std_gap']:.2f}</div>Std Dev</div>
    <div class="summary-card"><div class="val">{gap_summary['mode_gap']:.0f}</div>Mode</div>
  </div>
  <div class="chart-box"><img src="data:image/png;base64,{img3}"></div>
</section>

<section>
  <h2>[4] التحليل الدوري الرياضي (ACF / PACF / Periodogram / FFT)</h2>
  <p>Ljung-Box Test (حتى lag={periodicity['max_lag']}): Q={periodicity['ljung_box_stat']:.2f}, p-value={periodicity['ljung_box_p']:.4f}
  → {'يوجد ارتباط ذاتي دالّ إحصائيًا في مكان ما ضمن السلسلة' if periodicity['ljung_box_p'] < 0.05 else 'لا دليل على ارتباط ذاتي (متوافقة مع ضجيج أبيض)'}</p>
  {candidates_html}
  <p class="note">دورة الـ15 جلسة تحديدًا: ACF={period15_row['acf']:.4f} ({'تتجاوز حد الدلالة التقريبي' if period15_row['acf_significant_95pct'] else 'ضمن نطاق الضجيج'})،
  لكن رتبتها في قوة الطيف (Periodogram) هي {int(period15_row['power_rank'])} من {int(period15_row['n_frequencies'])} — أي أنها ضعيفة طيفيًا رغم تجاوز عتبة ACF الساذجة (اختبار على لاق واحد فقط، بدون تصحيح للمقارنات المتعددة عبر 6 دورات مرشّحة).</p>
  <div class="chart-box"><img src="data:image/png;base64,{img4}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img5}"></div>
</section>

<section>
  <h2>[5] Walk-Forward Validation — النموذج مقابل Baseline</h2>
  <table class="data-table">
    <tr><th>المقياس</th><th>النموذج</th><th>Baseline</th></tr>
    <tr><td>Precision</td><td>{model_metrics['precision']:.4f}</td><td>N/A</td></tr>
    <tr><td>Recall</td><td>{model_metrics['recall']:.4f}</td><td>{baseline_metrics['recall']:.4f}</td></tr>
    <tr><td>F1</td><td>{model_metrics['f1']:.4f}</td><td>N/A</td></tr>
    <tr><td>ROC-AUC</td><td>{model_metrics['roc_auc']:.4f}</td><td>{baseline_metrics['roc_auc']:.4f}</td></tr>
    <tr><td>Brier Score</td><td>{model_metrics['brier_score']:.4f}</td><td>{baseline_metrics['brier_score']:.4f}</td></tr>
  </table>
  <p class="note">n={model_metrics['n']} توقّع خارج العينة (18 حالة سيولة مرتفعة فعلية، 72 عادية). فاصل ثقة 95% لـ ROC-AUC (Bootstrap): [{auc_lo:.3f}, {auc_hi:.3f}].</p>
  <div class="verdict-box"><b>الخلاصة:</b> {oos_verdict}</div>
  <div class="chart-box"><img src="data:image/png;base64,{img6}"></div>
</section>

<section>
  <h2>[6] Next 15 Trading Sessions Forecast</h2>
  {forecast_html}
  <div class="chart-box"><img src="data:image/png;base64,{img7}"></div>
</section>

<section>
  <h2>الخلاصة المباشرة</h2>
  <ol class="qa">
    <li><b>هل توجد دورة زمنية حقيقية للسيولة؟</b> الدليل ضعيف ومختلط: اختبار Ljung-Box المشترك دالّ إحصائيًا (توجد بعض الارتباطات الذاتية في مكان ما ضمن 40 تأخيرًا)، لكن لا دورة محددة من الدورات الست المُختبرة تجتاز اختباري ACF والـ Periodogram معًا بوضوح.</li>
    <li><b>ما هي أقوى دورة مكتشفة؟</b> ≈ {strongest_candidate['period']:.0f} جلسة (الأعلى قوة طيفية بين الدورات المرشّحة المطلوبة) — لكنها ليست مؤكدة، والقمة الأقوى إجمالًا في الطيف كانت عند ≈{periodicity['top_periodogram_peaks'][0][0]:.0f} جلسة، قريبة من ثلث طول العينة وقد تعكس اتجاهًا عامًا لا دورة حقيقية.</li>
    <li><b>هل دورة 15 جلسة لها معنى إحصائي؟</b> ضعيف — ACF عند التأخير 15 يتجاوز بالكاد حد الدلالة الساذج (±{periodicity['sig_band']:.3f}) لكن رتبتها في قوة الطيف متأخرة ({int(period15_row['power_rank'])} من {int(period15_row['n_frequencies'])})، ولم تُختبر مع تصحيح للمقارنات المتعددة.</li>
    <li><b>أكثر الأيام احتمالًا لظهور سيولة مرتفعة (Level 2)؟</b> {', '.join(f"Day {int(r['DayInCycle'])} ({r['ProbabilityHighLiquidity_L2']:.0f}%)" for _, r in cycle_table.sort_values('ProbabilityHighLiquidity_L2', ascending=False).head(3).iterrows())} — لكن اختبار Chi-square لم يجد هذا التمايز دالًّا إحصائيًا (p={uniformity['p_value']:.4f}).</li>
    <li><b>متوسط عدد الجلسات بين موجات السيولة المرتفعة؟</b> {gap_summary['mean_gap']:.2f} جلسة (وسيط {gap_summary['median_gap']:.1f})، لكن التوزيع منحرف بشدة (Mode={gap_summary['mode_gap']:.0f} جلسة فقط) — أي أن أيام السيولة المرتفعة تميل للتجمّع في نوبات متتالية قصيرة أكثر من كونها موزّعة بانتظام دوري.</li>
    <li><b>هل استطاع النموذج التنبؤ خارج العينة؟</b> {oos_verdict}</li>
    <li><b>أفضل 3 جلسات قادمة مرشحة؟</b> {top3_text} — علمًا أن جميعها لم تتجاوز عتبة 50% (لا توجد جلسة "مرتفعة الثقة" بوضوح في الأفق القريب حسب النموذج الحالي).</li>
    <li><b>درجة الثقة لكل توقع؟</b> الجلسة القادمة مباشرة (+1) فقط تحمل ثقة "High" لأنها الوحيدة المبنية على بيانات زخم فعلية حديثة بالكامل؛ الجلسات +2 إلى +5 بثقة "Medium"، و+6 إلى +15 بثقة "Low" لأنها تفترض ثبات آخر قيم الزخم المعروفة دون بيانات جديدة فعلية.</li>
  </ol>
  <div class="verdict-box">
  <b>ملاحظة تفسيرية مهمة:</b> القدرة التنبؤية التي ظهرت في Walk-Forward (إن ثبتت) يُرجَّح أنها ناتجة أساسًا عن <u>تجمّع السيولة قصير المدى</u> (Gap Mode=1 جلسة، أي أن أيام السيولة المرتفعة تتلو بعضها غالبًا) الذي تلتقطه ميزات الزخم (lag1 / recent5 / trend5) في النموذج، وليس عن دورية ثابتة بطول 15 جلسة أو أي من الدورات المرشّحة الأخرى — والتي لم يثبت التحليل الطيفي وجودها بشكل قوي ومتّسق.
  </div>
</section>

</div>
<footer>TADAWUL:4140 Liquidity Cycle Analysis — Generated locally from previously-fetched TradingView data only. Not investment advice.</footer>
</body>
</html>
"""

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"تم إنشاء التقرير: {REPORT_HTML}")


if __name__ == "__main__":
    main()
