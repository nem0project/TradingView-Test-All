"""
يبني الرسوم البيانية الستة وتقرير HTML احترافي بالعربية لتحليل TADAWUL:4140
على آخر 180 جلسة، اعتمادًا على مخرجات advanced_analysis_180.py:
    data/4140_180_daily.csv
    data/4140_180_features.csv
    data/4140_180_analysis.csv

يُنتج:
    data/4140_180_report.html
    data/4140_180_chart_*.png (نسخ منفصلة من كل رسم)

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

DATA_DIR = "data"
DAILY_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "4140_180_features.csv")
ANALYSIS_CSV = os.path.join(DATA_DIR, "4140_180_analysis.csv")
REPORT_HTML = os.path.join(DATA_DIR, "4140_180_report.html")

HORIZONS = ["NextDayReturn", "Return3Days", "Return5Days", "Return7Days"]
MIN_CORR_SAMPLE = 20


# ============================================================
# تحميل البيانات
# ============================================================

def load_all():
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    analysis = pd.read_csv(ANALYSIS_CSV)
    return daily, features, analysis


def find_best_bls_col(features):
    for col in ["bls_v2_tanh_return", "bls_v1_sign", "bls_v3_additive"]:
        if col in features.columns:
            return col
    raise KeyError("لم يتم إيجاد أي عمود BullishLiquidityScore في ملف الميزات.")


# ============================================================
# الرسوم البيانية (تُحفظ كملفات PNG وتُرجَع كـ base64 للتقرير)
# ============================================================

def fig_to_base64(fig, png_path):
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_price(features):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(features["date"], features["low"], features["high"], color="tab:blue", alpha=0.15, label="High-Low Range")
    ax.plot(features["date"], features["close"], color="tab:blue", linewidth=1.3, label="Close")
    ax.set_title("Chart 1 - Close Price (with Daily High-Low Range)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_1_price.png"))


def chart_volume(features):
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.bar(features["date"], features["volume"], color="tab:gray", alpha=0.6, label="Volume")
    ax1.set_ylabel("Volume")
    ax2 = ax1.twinx()
    ax2.plot(features["date"], features["relative_volume"], color="tab:red", linewidth=1.2, label="RelativeVolume")
    ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(1.5, color="red", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("RelativeVolume")
    ax1.set_title("Chart 2 - Volume & RelativeVolume")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_2_volume.png"))


def chart_close_strength(features):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(features["date"], features["close_strength"], color="tab:green", linewidth=1.0)
    ax.axhline(0.75, color="orange", linewidth=0.8, linestyle=":", label="StrongBullish threshold (0.75)")
    ax.axhline(0.85, color="red", linewidth=0.8, linestyle=":", label="ExtremeBullish threshold (0.85)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Chart 3 - CloseStrength ((Close-Low)/(High-Low))")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_3_close_strength.png"))


def chart_bls(features, bls_col):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(features["date"], features[bls_col], color="tab:purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Chart 4 - BullishLiquidityScore ({bls_col})")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_4_bls.png"))


def chart_quantile_returns(quantile_means):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if quantile_means is None or len(quantile_means) == 0:
        ax.text(0.5, 0.5, "لا توجد بيانات شرائح كافية", ha="center", va="center")
    else:
        x = np.arange(len(quantile_means))
        width = 0.2
        for i, horizon in enumerate(HORIZONS):
            ax.bar(x + i * width, quantile_means[horizon], width=width, label=horizon)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(quantile_means.index)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
    ax.set_title("Chart 5 - Average Forward Returns by BullishLiquidityScore Quantile")
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_5_quantiles.png"))


def chart_scatter(features, bls_col):
    fig, ax = plt.subplots(figsize=(7, 6))
    x = features[bls_col]
    y = features["Return5Days"] * 100
    mask = (~x.isna()) & (~y.isna())
    ax.scatter(x[mask], y[mask], alpha=0.5, s=18, color="teal")
    if mask.sum() >= 3:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="tab:red", linewidth=1.5, label="Linear fit")
        ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"BullishLiquidityScore ({bls_col})")
    ax.set_ylabel("Return5Days (%)")
    ax.set_title("Chart 6 - BullishLiquidityScore vs Return5Days (Scatter)")
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_180_chart_6_scatter.png"))


# ============================================================
# استخراج جداول من ملف التحليل الموحّد
# ============================================================

def df_to_html_table(df, decimals=4, pct_cols=(), int_cols=()):
    view = df.copy()
    for col in view.columns:
        if col in int_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
        elif col in pct_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}%")
        elif pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
    return view.to_html(index=False, border=0, classes="data-table", na_rep="")


def best_variable_per_horizon(compare_df):
    best = {}
    for horizon in HORIZONS:
        sub = compare_df[compare_df["horizon_"] == horizon].copy()
        if sub.empty:
            continue
        sig = sub[(sub["pearson_p"] < 0.05) & (sub["n"] >= MIN_CORR_SAMPLE)]
        if len(sig) > 0:
            winner = sig.loc[sig["pearson_r"].abs().idxmax()]
            best[horizon] = (winner["variable_"], winner["pearson_r"], winner["pearson_p"], winner["n"], True)
        else:
            winner = sub.loc[sub["pearson_r"].abs().idxmax()]
            best[horizon] = (winner["variable_"], winner["pearson_r"], winner["pearson_p"], winner["n"], False)
    return best


def main():
    daily, features, analysis = load_all()
    bls_col = find_best_bls_col(features)

    n_sessions = len(daily)
    start_date = daily["date"].iloc[0].date()
    end_date = daily["date"].iloc[-1].date()

    variant_rows = analysis[analysis["type"] == "model_variant"].copy()
    variant_rows[["variant_", "horizon_"]] = variant_rows["label"].str.split(" vs ", expand=True)

    compare_rows = analysis[analysis["type"] == "variable_comparison"].copy()
    compare_rows[["variable_", "horizon_"]] = compare_rows["label"].str.split(" vs ", expand=True)

    quantile_mean_rows = analysis[analysis["type"] == "quantile_mean"].copy()
    quantile_means = None
    if not quantile_mean_rows.empty:
        quantile_mean_rows[["quantile_", "horizon_"]] = quantile_mean_rows["label"].str.split(":", expand=True)
        quantile_means = quantile_mean_rows.pivot_table(index="quantile_", columns="horizon_", values="mean_pct", aggfunc="first")
        n_per_q = quantile_mean_rows.groupby("quantile_")["n"].first()
        quantile_means.insert(0, "n", n_per_q)
        quantile_means = quantile_means.reindex(sorted(quantile_means.index, key=lambda s: int(s[1:])))
        quantile_means = quantile_means[["n"] + HORIZONS]

    extremes_rows = analysis[analysis["type"] == "quantile_extremes"].copy()

    signal_rows = analysis[analysis["type"] == "signal_hypothesis"].copy()
    if not signal_rows.empty:
        signal_rows[["signal_", "horizon_"]] = signal_rows["label"].str.split(":", expand=True)

    best_map = best_variable_per_horizon(compare_rows)

    # ------------------------------------------------------------
    # الرسوم
    # ------------------------------------------------------------
    img1 = chart_price(features)
    img2 = chart_volume(features)
    img3 = chart_close_strength(features)
    img4 = chart_bls(features, bls_col)
    img5 = chart_quantile_returns(quantile_means)
    img6 = chart_scatter(features, bls_col)

    # ------------------------------------------------------------
    # جداول HTML
    # ------------------------------------------------------------
    daily_table = df_to_html_table(pd.concat([daily.head(5), daily.tail(5)]), int_cols=["volume"])

    describe_cols = ["close", "volume", "close_strength", "relative_volume", bls_col]
    stats_table = df_to_html_table(features[describe_cols].describe().reset_index())

    variant_table = df_to_html_table(
        variant_rows[["variant_", "horizon_", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "r_squared"]]
        .rename(columns={"variant_": "Model Variant", "horizon_": "Horizon"}),
        int_cols=["n"],
    )

    compare_table = df_to_html_table(
        compare_rows[["variable_", "horizon_", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "r_squared"]]
        .rename(columns={"variable_": "Variable", "horizon_": "Horizon"}),
        int_cols=["n"],
    )

    quantile_table = df_to_html_table(quantile_means.reset_index(), pct_cols=HORIZONS, int_cols=["n"]) if quantile_means is not None else "<p>لا توجد بيانات شرائح كافية.</p>"

    extremes_table = df_to_html_table(
        extremes_rows[["label", "n", "mean_pct", "median_pct", "win_rate_pct", "max_pct", "min_pct", "mannwhitney_p", "cohens_d"]],
        int_cols=["n"],
    ) if not extremes_rows.empty else "<p>لا توجد بيانات كافية.</p>"

    def signal_table_for(name):
        sub = signal_rows[signal_rows["signal_"] == name] if not signal_rows.empty else pd.DataFrame()
        if sub.empty:
            return "<p>لا توجد جلسات مطابقة لهذه الإشارة ضمن العينة.</p>"
        return df_to_html_table(
            sub[["horizon_", "n", "mean_pct", "median_pct", "win_rate_pct", "max_pct", "min_pct", "mannwhitney_p", "cohens_d"]]
            .rename(columns={"horizon_": "Horizon"}),
            int_cols=["n"],
        )

    bullish_table = signal_table_for("StrongBullishLiquidity")
    extreme_table = signal_table_for("ExtremeBullishLiquidity")

    best_var_rows = "".join(
        f"<tr><td>{h}</td><td>{v[0]}</td><td>{v[1]:+.4f}</td><td>{v[2]:.4f}</td><td>{int(v[3])}</td>"
        f"<td>{'✅ دالّ' if v[4] else '⚠️ غير دالّ'}</td></tr>"
        for h, v in best_map.items()
    )

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>TADAWUL:4140 - تقرير التحليل المتقدم (180 جلسة)</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Arial, sans-serif; background:#0f1115; color:#e8e8e8; margin:0; padding:0; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  header {{ background: linear-gradient(135deg,#1b2735,#0f1115); padding: 32px 24px; border-bottom: 3px solid #2f81f7; }}
  header h1 {{ margin:0 0 6px 0; font-size: 1.6rem; }}
  header p {{ margin:2px 0; color:#9fb3c8; }}
  section {{ background:#161a22; border:1px solid #262c37; border-radius:10px; padding:20px; margin:20px 0; }}
  h2 {{ color:#2f81f7; border-bottom:1px solid #262c37; padding-bottom:8px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 10px; }}
  table.data-table th, table.data-table td {{ border: 1px solid #2b323e; padding: 6px 10px; text-align: center; }}
  table.data-table th {{ background:#1f2733; color:#7fb3ff; }}
  table.data-table tr:nth-child(even) {{ background:#1a1f28; }}
  .note {{ background:#2a2410; border:1px solid #7a5c00; color:#f0d878; padding:10px 14px; border-radius:6px; font-size:0.9rem; }}
  .chart-box {{ text-align:center; margin: 18px 0; }}
  .chart-box img {{ max-width:100%; border-radius:8px; border:1px solid #262c37; }}
  .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:10px; }}
  .summary-card {{ background:#1f2733; border-radius:8px; padding:14px; text-align:center; }}
  .summary-card .val {{ font-size:1.4rem; color:#2f81f7; font-weight:bold; }}
  footer {{ text-align:center; color:#666; padding:20px; font-size:0.8rem; }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>TADAWUL:4140 — الشركة السعودية للصادرات الصناعية (صادرات)</h1>
    <p>تقرير تحليل رياضي وإحصائي متقدم — آخر {n_sessions} جلسة تداول فعلية</p>
    <p>المصدر: TradingView WebSocket (chart_session / resolve_symbol / create_series) — تم إنشاء التقرير في {generated_at}</p>
  </div>
</header>
<div class="wrap">

<section>
  <h2>نظرة عامة</h2>
  <div class="summary-grid">
    <div class="summary-card"><div class="val">{n_sessions}</div>عدد الجلسات الفعلية</div>
    <div class="summary-card"><div class="val">{start_date}</div>بداية الفترة</div>
    <div class="summary-card"><div class="val">{end_date}</div>نهاية الفترة</div>
    <div class="summary-card"><div class="val">نعم</div>بيانات حقيقية من TradingView</div>
  </div>
</section>

<section>
  <h2>جدول البيانات (أول 5 وآخر 5 جلسات)</h2>
  {daily_table}
</section>

<section>
  <h2>ملخص إحصائي</h2>
  {stats_table}
</section>

<section>
  <h2>مقارنة نسخ BullishLiquidityScore (V1 / V2 / V3)</h2>
  <p class="note">النموذج الأساسي V1 كما طُلب حرفيًا (ZScore × CloseStrength × Sign)، بالإضافة إلى نسختين محسّنتين (V2 بمقدار الحركة، V3 جمعية بدل ضربية) تمت مقارنتهما إحصائيًا.</p>
  {variant_table}
</section>

<section>
  <h2>جدول الارتباطات — Pearson و Spearman (Rank Correlation) لكل متغير مقابل كل أفق زمني</h2>
  {compare_table}
  <h3>أفضل متغير تفسيري لكل أفق زمني</h3>
  <table class="data-table">
    <tr><th>الأفق</th><th>أفضل متغير</th><th>Pearson r</th><th>p-value</th><th>N</th><th>الدلالة</th></tr>
    {best_var_rows}
  </table>
  <p class="note">تذكير: الارتباط الإحصائي (correlation) لا يعني تلقائيًا قدرة تنبؤية (prediction) موثوقة. يجب قراءة N وp-value معًا قبل أي استنتاج.</p>
</section>

<section>
  <h2>مقارنة الشرائح (Quantiles) لـ BullishLiquidityScore</h2>
  {quantile_table}
  <h3>أعلى شريحة مقابل أدنى شريحة</h3>
  {extremes_table}
</section>

<section>
  <h2>أداء إشارة StrongBullishLiquidity</h2>
  <p>الشرط: RelativeVolume &gt; 1.5 و CloseStrength &ge; 0.75 و DailyReturn &gt; 0</p>
  {bullish_table}
</section>

<section>
  <h2>أداء إشارة ExtremeBullishLiquidity</h2>
  <p>الشرط: RelativeVolume &gt; 2.0 و CloseStrength &ge; 0.85 و DailyReturn &gt; 0</p>
  {extreme_table}
</section>

<section>
  <h2>الرسوم البيانية</h2>
  <div class="chart-box"><img src="data:image/png;base64,{img1}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img2}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img3}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img4}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img5}"></div>
  <div class="chart-box"><img src="data:image/png;base64,{img6}"></div>
</section>

<section>
  <h2>تنبيه منهجي</h2>
  <p class="note">
  هذا تقرير تحليل إحصائي ورياضي بحت اعتمادًا على 180 جلسة تداول فعلية، ولا يمثّل توصية شراء أو بيع بأي شكل.
  النتائج ذات الدلالة الإحصائية (p&lt;0.05) محدودة وظهرت في آفاق زمنية معيّنة فقط دون غيرها، وحجم بعض المجموعات
  (خصوصًا إشارات StrongBullishLiquidity وExtremeBullishLiquidity) صغير جدًا مما يحدّ من موثوقية أي استنتاج عام.
  </p>
</section>

</div>
<footer>TADAWUL:4140 Advanced Analysis — Generated locally from TradingView WebSocket data only.</footer>
</body>
</html>
"""

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"تم إنشاء التقرير: {REPORT_HTML}")
    print("تم حفظ الرسوم البيانية الست كملفات PNG منفصلة في مجلد data/ أيضًا.")


if __name__ == "__main__":
    main()
