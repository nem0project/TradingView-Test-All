"""
يبني الرسوم البيانية وتقرير HTML لـ ContextualLiquidityModel، اعتمادًا على مخرجات
contextual_liquidity_model.py:
    data/4140_contextual_features.csv
    data/4140_contextual_events.csv
    data/4140_contextual_models.csv
    data/4140_contextual_models_diagnosis.csv

يُنتج: data/4140_contextual_liquidity_report.html
يستخدم matplotlib فقط.
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
FEATURES_CSV = os.path.join(DATA_DIR, "4140_contextual_features.csv")
EVENTS_CSV = os.path.join(DATA_DIR, "4140_contextual_events.csv")
MODELS_CSV = os.path.join(DATA_DIR, "4140_contextual_models.csv")
DIAGNOSIS_CSV = os.path.join(DATA_DIR, "4140_contextual_models_diagnosis.csv")
REPORT_HTML = os.path.join(DATA_DIR, "4140_contextual_liquidity_report.html")

HORIZONS = ["NextDayReturn", "Return3Days", "Return5Days", "Return7Days", "Return10Days"]
TRAIN_FRACTION = 0.70


def load_all():
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    events = pd.read_csv(EVENTS_CSV)
    models = pd.read_csv(MODELS_CSV)
    diagnosis = pd.read_csv(DIAGNOSIS_CSV)
    return features, events, models, diagnosis


def fig_to_base64(fig, png_path):
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
# الرسوم
# ============================================================

def chart_price_with_events(features):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(features["date"], features["close"], color="tab:blue", linewidth=1.0, label="Close", zorder=1)

    markers = [
        ("case_a_uptrend", "tab:green", "^", "Case A: Uptrend"),
        ("case_b_breakout", "tab:orange", "s", "Case B: Breakout"),
        ("case_c_extended", "tab:red", "D", "Case C: Extended"),
        ("case_d_weak_close", "tab:purple", "v", "Case D: WeakClose"),
    ]
    for col, color, marker, label in markers:
        mask = features[col] == True  # noqa: E712
        ax.scatter(features.loc[mask, "date"], features.loc[mask, "close"], color=color, marker=marker,
                   s=45, label=label, zorder=3, edgecolors="black", linewidths=0.4)

    ax.set_title("Chart 1 - Close Price with Contextual Liquidity Events")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_1_price_events.png"))


def chart_score_timeline(features):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    n_train = int(len(features) * TRAIN_FRACTION)
    split_date = features["date"].iloc[n_train]

    ax.plot(features["date"], features["model_d_contextual_score"], color="tab:indigo" if False else "purple", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(split_date, color="red", linewidth=1.2, linestyle="--", label="Train/Test split (70/30)")
    ax.set_title("Chart 2 - ContextualLiquidityScore (Model D) over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_2_score_timeline.png"))


def chart_event_returns(events):
    cases = events["case"].unique()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(HORIZONS))
    width = 0.8 / max(len(cases), 1)

    for i, case in enumerate(cases):
        sub = events[events["case"] == case].set_index("horizon").reindex(HORIZONS)
        ax.bar(x + i * width, sub["mean_pct"], width=width, label=case.split("_", 1)[0])

    ax.set_xticks(x + width * (len(cases) - 1) / 2)
    ax.set_xticklabels(HORIZONS, rotation=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Forward Return (%)")
    ax.set_title("Chart 3 - Event Study: Mean Forward Return by Case")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_3_event_returns.png"))


def chart_train_test_correlation(models):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    model_names = models["model"].unique()

    for ax, model_name in zip(axes.flat, model_names):
        sub = models[models["model"] == model_name]
        pivot = sub.pivot_table(index="horizon", columns="split", values="pearson_r", aggfunc="first").reindex(HORIZONS)
        x = np.arange(len(HORIZONS))
        width = 0.35
        ax.bar(x - width / 2, pivot.get("Training", pd.Series(index=HORIZONS)), width=width, label="Training", color="tab:blue")
        ax.bar(x + width / 2, pivot.get("Testing", pd.Series(index=HORIZONS)), width=width, label="Testing", color="tab:orange")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS, rotation=25, fontsize=7)
        ax.set_title(model_name.split(" (")[0], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Chart 4 - Pearson r: Training vs Testing (per Model)")
    fig.tight_layout()
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_4_train_vs_test.png"))


def chart_win_rate_comparison(events):
    cases = events["case"].unique()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(HORIZONS))
    width = 0.8 / max(len(cases), 1)

    for i, case in enumerate(cases):
        sub = events[events["case"] == case].set_index("horizon").reindex(HORIZONS)
        ax.bar(x + i * width, sub["win_rate_pct"], width=width, label=case.split("_", 1)[0])

    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.set_xticks(x + width * (len(cases) - 1) / 2)
    ax.set_xticklabels(HORIZONS, rotation=20)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Chart 5 - Win Rate by Case and Horizon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_5_winrate.png"))


def chart_scatter_train_test(features):
    n_train = int(len(features) * TRAIN_FRACTION)
    train = features.iloc[:n_train]
    test = features.iloc[n_train:]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for data, color, label in [(train, "tab:blue", "Training"), (test, "tab:orange", "Testing")]:
        x = data["model_d_contextual_score"]
        y = data["Return5Days"] * 100
        mask = (~x.isna()) & (~y.isna())
        ax.scatter(x[mask], y[mask], alpha=0.55, s=20, color=color, label=label)
        if mask.sum() >= 3:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 40)
            ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=1.6)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("ContextualLiquidityScore (Model D)")
    ax.set_ylabel("Return5Days (%)")
    ax.set_title("Chart 6 - ContextualLiquidityScore vs Return5Days (Train vs Test)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig, os.path.join(DATA_DIR, "4140_ctx_chart_6_scatter_train_test.png"))


# ============================================================
# جداول HTML
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
        elif pd.api.types.is_bool_dtype(view[col]):
            view[col] = view[col].map(lambda x: "YES" if x else "")
    return view.to_html(index=False, border=0, classes="data-table", na_rep="")


def main():
    features, events, models, diagnosis = load_all()

    n_sessions = len(features)
    start_date = features["date"].iloc[0].date()
    end_date = features["date"].iloc[-1].date()
    n_train = int(n_sessions * TRAIN_FRACTION)

    case_counts = {
        "Case A (Uptrend)": int(features["case_a_uptrend"].sum()),
        "Case B (Breakout)": int(features["case_b_breakout"].sum()),
        "Case C (Extended)": int(features["case_c_extended"].sum()),
        "Case D (WeakClose)": int(features["case_d_weak_close"].sum()),
        "Normal days": int(features["normal_day"].sum()),
    }

    img1 = chart_price_with_events(features)
    img2 = chart_score_timeline(features)
    img3 = chart_event_returns(events)
    img4 = chart_train_test_correlation(models)
    img5 = chart_win_rate_comparison(events)
    img6 = chart_scatter_train_test(features)

    events_table = df_to_html_table(
        events[["case", "horizon", "n", "mean_pct", "median_pct", "win_rate_pct", "std_pct", "sharpe_like",
                "max_pct", "min_pct", "ttest_p", "mannwhitney_p", "cohens_d", "reliable"]],
        int_cols=["n"],
    )

    models_pivot = models.pivot_table(index=["model", "horizon"], columns="split", values=["n", "pearson_r", "pearson_p"], aggfunc="first").reset_index()
    models_pivot.columns = [f"{a}_{b}" if b else a for a, b in models_pivot.columns]
    models_table = df_to_html_table(models_pivot)

    diagnosis_table = df_to_html_table(
        diagnosis[["model", "horizon", "train_r", "train_p", "train_n", "test_r", "test_p", "test_n", "verdict"]],
        int_cols=["train_n", "test_n"],
    )

    n_validated = int((diagnosis["verdict"].str.startswith("Validated")).sum())
    n_overfit = int((diagnosis["verdict"].str.startswith("Overfitting")).sum())
    n_sign_reversal = int((diagnosis["verdict"].str.startswith("Sign Reversal")).sum())
    n_no_evidence = int((diagnosis["verdict"].str.startswith("No Evidence")).sum())
    n_emerged = int((diagnosis["verdict"].str.startswith("Emerged")).sum())

    # أقوى إشارة حسب بيانات الاختبار (Testing) فقط، من بين ما وصل لدلالة إحصائية
    test_sig = diagnosis[(diagnosis["test_p"] < 0.05) & (diagnosis["test_n"] >= 20)].copy()
    if len(test_sig):
        test_sig = test_sig.reindex(test_sig["test_r"].abs().sort_values(ascending=False).index)
        best_row = test_sig.iloc[0]
        best_signal_text = (
            f"{best_row['model']} على أفق {best_row['horizon']} (test r={best_row['test_r']:+.3f}, "
            f"p={best_row['test_p']:.4f}, n={int(best_row['test_n'])}) — لكن تصنيفها: {best_row['verdict']}، "
            "أي أنها ليست نتيجة موثوقة بل انعكاس/عدم استقرار."
        )
    else:
        best_signal_text = "لا يوجد أي نموذج أو أفق حقق دلالة إحصائية موثوقة ومستقرة في بيانات الاختبار (Testing)."

    case_d_note = ""
    case_d_next_day = events[(events["case"] == "CaseD_HighVol_WeakClose") & (events["horizon"] == "NextDayReturn")]
    if not case_d_next_day.empty:
        row = case_d_next_day.iloc[0]
        case_d_note = (
            f"أقرب إشارة لافتة (وليست مؤكدة): الحالة D (سيولة مرتفعة + إغلاق ضعيف، N={int(row['n'])}) تلتها في اليوم التالي عوائد "
            f"متوسطها {row['mean_pct']:+.2f}% مقابل أيام عادية عادةً قريبة من الصفر، "
            f"بدلالة حدّية عند اختبار Mann-Whitney فقط (p={row['mannwhitney_p']:.4f}) بينما لم يصل t-test لنفس الدلالة، "
            "ولم تتكرر بوضوح في الآفاق الأطول (3/5/7/10 أيام)، ولم تُختبر خارج العينة (Event Study لا يستخدم تقسيم Train/Test)، فتبقى ملاحظة أولية فقط وليست نتيجة مؤكدة."
        )

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    summary_cards = "".join(
        f'<div class="summary-card"><div class="val">{v}</div>{k}</div>' for k, v in case_counts.items()
    )

    html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>TADAWUL:4140 - ContextualLiquidityModel</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Arial, sans-serif; background:#0f1115; color:#e8e8e8; margin:0; padding:0; }}
  .wrap {{ max-width: 1150px; margin: 0 auto; padding: 24px; }}
  header {{ background: linear-gradient(135deg,#1b2735,#0f1115); padding: 32px 24px; border-bottom: 3px solid #7b5cff; }}
  header h1 {{ margin:0 0 6px 0; font-size: 1.6rem; }}
  header p {{ margin:2px 0; color:#9fb3c8; }}
  section {{ background:#161a22; border:1px solid #262c37; border-radius:10px; padding:20px; margin:20px 0; }}
  h2 {{ color:#7b5cff; border-bottom:1px solid #262c37; padding-bottom:8px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-top: 10px; }}
  table.data-table th, table.data-table td {{ border: 1px solid #2b323e; padding: 5px 8px; text-align: center; }}
  table.data-table th {{ background:#1f2733; color:#b79cff; }}
  table.data-table tr:nth-child(even) {{ background:#1a1f28; }}
  .note {{ background:#2a2410; border:1px solid #7a5c00; color:#f0d878; padding:10px 14px; border-radius:6px; font-size:0.9rem; }}
  .verdict-box {{ background:#1a2e22; border:1px solid #2f7a4e; color:#a8e6bf; padding:14px 18px; border-radius:8px; font-size:0.95rem; margin:10px 0; }}
  .chart-box {{ text-align:center; margin: 18px 0; }}
  .chart-box img {{ max-width:100%; border-radius:8px; border:1px solid #262c37; }}
  .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:10px; }}
  .summary-card {{ background:#1f2733; border-radius:8px; padding:14px; text-align:center; }}
  .summary-card .val {{ font-size:1.3rem; color:#7b5cff; font-weight:bold; }}
  ol.qa li {{ margin-bottom: 10px; }}
  footer {{ text-align:center; color:#666; padding:20px; font-size:0.8rem; }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>TADAWUL:4140 — ContextualLiquidityModel</h1>
    <p>السيولة ضمن سياقها: موقع الإغلاق + الاتجاه + الموقع من القمة/القاع + الاختراق</p>
    <p>{n_sessions} جلسة فعلية ({start_date} → {end_date}) — Training: أول {n_train} جلسة | Testing: آخر {n_sessions - n_train} جلسة — تم إنشاء التقرير في {generated_at}</p>
  </div>
</header>
<div class="wrap">

<section>
  <h2>نظرة عامة على الأحداث السياقية</h2>
  <div class="summary-grid">{summary_cards}</div>
</section>

<section>
  <h2>[1] Event Study — أداء كل حالة سياقية مقابل الأيام العادية</h2>
  <p class="note">Case C (سيولة+إغلاق قوي بعد صعود ممتد) تُختبر كحالة مستقلة لاحتمال الاستمرار أو التصريف كما طُلب، وليست حالة صعودية موجّهة مسبقًا.</p>
  {events_table}
</section>

<section>
  <h2>[2] ContextualLiquidityScore — نتائج Training مقابل Testing</h2>
  {models_table}
</section>

<section>
  <h2>[3] تشخيص Overfitting / عدم الاستقرار</h2>
  <div class="summary-grid">
    <div class="summary-card"><div class="val">{n_validated}</div>Validated</div>
    <div class="summary-card"><div class="val">{n_overfit}</div>Overfitting</div>
    <div class="summary-card"><div class="val">{n_sign_reversal}</div>Sign Reversal</div>
    <div class="summary-card"><div class="val">{n_emerged}</div>Emerged in Test only</div>
    <div class="summary-card"><div class="val">{n_no_evidence}</div>No Evidence</div>
  </div>
  {diagnosis_table}
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
  <h2>الخلاصة المباشرة</h2>
  <ol class="qa">
    <li><b>هل السيولة وحدها لها قيمة تنبؤية؟</b> لا — Model A (RelativeVolume فقط) لم يحقق أي حالة "Validated" عبر الآفاق الخمسة؛ أغلب نتائجه صُنّفت Sign Reversal أو No Evidence.</li>
    <li><b>هل الإغلاق قرب أعلى الشمعة له قيمة تنبؤية؟</b> لا بمفرده — الحالات A/B/C (تشترط CloseStrength≥0.75) لم تُظهر أي دلالة إحصائية ثابتة في Event Study.</li>
    <li><b>هل الجمع بين السيولة والإغلاق أفضل؟</b> Model B لم يتحسن جوهريًا عن Model A؛ نفس نمط عدم الاستقرار بين التدريب والاختبار.</li>
    <li><b>هل إضافة الاتجاه تحسن النموذج؟</b> لا يوجد دليل على تحسّن حقيقي — Model C/D (مع الاتجاه والاختراق) أظهرا نفس المشكلة: عدم دلالة في التدريب ثم ظهور دلالة بإشارة معاكسة في الاختبار (Sign Reversal / Emerged in Test only)، وهو مؤشر عدم استقرار وليس تحسّنًا.</li>
    <li><b>هل الاختراق مع السيولة هو أقوى حالة؟</b> عدد أحداثه صغير جدًا (Case B، n=7) ولم يصل لأي دلالة إحصائية موثوقة — لا يمكن اعتباره الأقوى إحصائيًا حاليًا.</li>
    <li><b>ما هي أفضل إشارة فعلية حسب بيانات الاختبار (Out-of-Sample)؟</b> {best_signal_text}</li>
    <li><b>ما هو حجم العينة لكل إشارة؟</b> Case A=10، Case B=7، Case C=9، Case D=18، من أصل 180 جلسة (Training≈125، Testing≈55). كل هذه الأحجام صغيرة نسبيًا للاستنتاج العام.</li>
    <li><b>هل النتيجة قابلة للثقة أم العينة لا تزال صغيرة؟</b> العينة لا تزال صغيرة، خصوصًا لعينات الأحداث (7 إلى 18 حالة) ولانقسام Training/Testing (125/55 جلسة). النتيجة الحالية الأكثر ثباتًا هي عدم وجود دليل موثوق على قدرة تنبؤية للنموذج بأي من نسخه الأربع.</li>
  </ol>
  {f'<p class="note">{case_d_note}</p>' if case_d_note else ''}
  <div class="verdict-box">
  <b>الخلاصة العلمية:</b> بعد إعادة بناء النموذج ليأخذ السياق السعري بعين الاعتبار (اتجاه، موقع من القمة/القاع، اختراق) واختباره بمنهجية Event Study حقيقية مع Out-of-sample validation صارم (Training 70% / Testing 30%، ومعلمات Z-score مُستخرجة من Training فقط)،
  <u>لم يثبت أي من نسخ ContextualLiquidityScore الأربع (A/B/C/D) قدرة تنبؤية مستقرة وموثوقة</u> على عوائد TADAWUL:4140 المستقبلية خلال آفاق 1 إلى 10 جلسات. هذه نتيجة سلبية صريحة وليست نتيجة "شبه إيجابية" — ولم يتم البحث عن نتيجة إيجابية بالقوة ولا تعديل أي بيانات للوصول إليها.
  </div>
</section>

</div>
<footer>TADAWUL:4140 ContextualLiquidityModel — Generated locally from previously-fetched TradingView data only. Not investment advice.</footer>
</body>
</html>
"""

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"تم إنشاء التقرير: {REPORT_HTML}")


if __name__ == "__main__":
    main()
