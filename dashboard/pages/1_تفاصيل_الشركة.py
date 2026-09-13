"""
صفحة تفاصيل الشركة — تعرض تحليل دورية السيولة الكامل لسهم واحد (نفس منطق TADAWUL:4140).
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data_store import ANALYSIS_DIR, PROCESSED_DIR, read_symbols  # noqa: E402
from core.liquidity_engine import DAYS_PER_CYCLE, simplified_horizon_label  # noqa: E402

st.set_page_config(page_title="تفاصيل الشركة", page_icon="📈", layout="wide")

RTL_CSS = """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, Arial, sans-serif; }
[data-testid="stSidebar"] { direction: rtl; text-align: right; }
.metric-card { background:#161a22; border:1px solid #262c37; border-radius:10px; padding:14px 16px; text-align:center; }
.metric-card .val { font-size:1.5rem; font-weight:700; color:#2f81f7; }
.explain-box { background:#161a22; border:1px solid #262c37; border-radius:10px; padding:16px 18px; line-height:1.9; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

SUMMARY_CSV = os.path.join(ANALYSIS_DIR, "summary.csv")

symbols_df = read_symbols()
if not os.path.exists(SUMMARY_CSV):
    st.warning("لا توجد نتائج بعد. شغّل: py update_all.py")
    st.stop()

summary_df = pd.read_csv(
    SUMMARY_CSV,
    parse_dates=["last_date", "last_high_liquidity_date", "best_date_1", "best_date_2", "best_date_3"],
    dtype={"symbol": str},
)
available_symbols = summary_df["symbol"].tolist()

default_symbol = st.session_state.get("selected_symbol", available_symbols[0] if available_symbols else None)
if default_symbol not in available_symbols and available_symbols:
    default_symbol = available_symbols[0]

symbol = st.selectbox(
    "اختر الشركة:",
    options=available_symbols,
    index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
    format_func=lambda s: f"{s} — {summary_df.loc[summary_df['symbol'] == s, 'name'].values[0]}",
)
st.session_state["selected_symbol"] = symbol

row = summary_df[summary_df["symbol"] == symbol].iloc[0]
name = row["name"]

processed_path = os.path.join(PROCESSED_DIR, f"{symbol}.csv")
cycle_table_path = os.path.join(ANALYSIS_DIR, f"{symbol}_cycle_table.csv")
top_days_path = os.path.join(ANALYSIS_DIR, f"{symbol}_top_days.csv")
forecast_path = os.path.join(ANALYSIS_DIR, f"{symbol}_forecast.csv")
repeat_strength_path = os.path.join(ANALYSIS_DIR, f"{symbol}_repeat_strength.csv")

if not os.path.exists(processed_path):
    st.error(f"لا توجد بيانات محلَّلة لـ {symbol}. شغّل: py update_all.py")
    st.stop()

df = pd.read_csv(processed_path, parse_dates=["date"])
cycle_table = pd.read_csv(cycle_table_path) if os.path.exists(cycle_table_path) else pd.DataFrame()
top_days_df = pd.read_csv(top_days_path) if os.path.exists(top_days_path) else pd.DataFrame()
forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()
repeat_strength_df = pd.read_csv(repeat_strength_path) if os.path.exists(repeat_strength_path) else pd.DataFrame()

st.title(f"📈 {symbol} — {name}")
st.caption("تحليل دورية السيولة — نفس المنهجية المستخدمة والمُختبَرة على TADAWUL:4140. هذا تحليل نمطي إحصائي فقط، وليس توصية شراء أو بيع.")

# ------------------------------------------------------------
# مؤشرات علوية
# ------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.markdown(f'<div class="metric-card">آخر إغلاق<br><span class="val">{row["last_close"]:.2f}</span></div>', unsafe_allow_html=True)
change_txt = f'{"+" if row["daily_change_pct"]>=0 else ""}{row["daily_change_pct"]:.2f}%'
c2.markdown(f'<div class="metric-card">التغير اليومي<br><span class="val">{change_txt}</span></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card">مستوى السيولة<br><span class="val">{row["liquidity_level"]}</span></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card">الموقع بالدورة<br><span class="val">D{int(row["current_cycle"])} / يوم {int(row["current_day_in_cycle"])}</span></div>', unsafe_allow_html=True)
sslh = "—" if pd.isna(row["sessions_since_last_high"]) else int(row["sessions_since_last_high"])
c5.markdown(f'<div class="metric-card">منذ آخر سيولة مرتفعة<br><span class="val">{sslh} جلسة</span></div>', unsafe_allow_html=True)
c6.markdown(f'<div class="metric-card">حالة النمط<br><span class="val">{row["pattern_status"]}</span></div>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------
# تفاصيل الحساب — الحقول المنقولة من الجدول الرئيسي (Final Opportunity Score وعناصره)
# ------------------------------------------------------------
with st.expander("📐 تفاصيل حساب الفرصة (Final Opportunity Score) والمدخلات الأساسية", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    liq_score = row.get("liquidity_score", float("nan"))
    d1.markdown(f'<div class="metric-card">Liquidity Score<br><span class="val">{"—" if pd.isna(liq_score) else f"{liq_score:.0f}/100"}</span></div>', unsafe_allow_html=True)
    timing_score = row.get("timing_score", float("nan"))
    d2.markdown(f'<div class="metric-card">Timing Score<br><span class="val">{"—" if pd.isna(timing_score) else f"{timing_score:.0f}/100"}</span></div>', unsafe_allow_html=True)
    final_score = row.get("final_opportunity_score", float("nan"))
    d3.markdown(f'<div class="metric-card">Final Opportunity Score<br><span class="val">{"—" if pd.isna(final_score) else f"{final_score:.1f}"}</span></div>', unsafe_allow_html=True)
    d4.markdown(f'<div class="metric-card">الموقع بالدورة<br><span class="val">D{int(row["current_cycle"])} / يوم {int(row["current_day_in_cycle"])}</span></div>', unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(f'<div class="metric-card">متوسط آخر 3 جلسات (Volume)<br><span class="val">{row.get("average_last_3_volume", float("nan")):,.0f}</span></div>', unsafe_allow_html=True)
    e2.markdown(f'<div class="metric-card">السيولة المرجعية (20 جلسة)<br><span class="val">{row.get("average_volume_reference", float("nan")):,.0f}</span></div>', unsafe_allow_html=True)
    e3.markdown(f'<div class="metric-card">السيولة اليومية (آخر جلسة)<br><span class="val">{row.get("current_liquidity", float("nan")):,.0f}</span></div>', unsafe_allow_html=True)
    e4.markdown(f'<div class="metric-card">متوسط السيولة (60 جلسة)<br><span class="val">{row.get("avg_liquidity_60", float("nan")):,.0f}</span></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    exp_sessions = row.get("expected_sessions_to_liquidity", float("nan"))
    f1.markdown(f'<div class="metric-card">جلسات حتى السيولة المتوقعة<br><span class="val">{"—" if pd.isna(exp_sessions) else int(exp_sessions)}</span></div>', unsafe_allow_html=True)
    sslh2 = row.get("sessions_since_last_high", float("nan"))
    f2.markdown(f'<div class="metric-card">منذ آخر سيولة مرتفعة<br><span class="val">{"—" if pd.isna(sslh2) else int(sslh2)} جلسة</span></div>', unsafe_allow_html=True)
    prob_score = row.get("probability_score", float("nan"))
    strength_txt = row.get("liquidity_pattern_strength", "—")
    f3.markdown(f'<div class="metric-card">احتمال / قوة النمط<br><span class="val">{"—" if pd.isna(prob_score) else f"{int(prob_score)}%"} ({strength_txt})</span></div>', unsafe_allow_html=True)
    old_score = row.get("old_liquidity_opportunity_score", float("nan"))
    f4.markdown(f'<div class="metric-card">الفرصة (نظام قديم، للمقارنة)<br><span class="val">{"—" if pd.isna(old_score) else int(old_score)}</span></div>', unsafe_allow_html=True)

    last_high_date = row.get("last_high_liquidity_date")
    last_high_date_txt = "—" if pd.isna(last_high_date) else pd.Timestamp(last_high_date).strftime("%Y-%m-%d")
    st.caption(f"حجم آخر جلسة: {row.get('last_volume', float('nan')):,.0f} — مستوى السيولة: {row.get('liquidity_level','—')} — "
               f"آخر يوم سيولة مرتفعة: {last_high_date_txt}")
    st.caption("ملاحظة: probability_score وliquidity_pattern_strength لا يدخلان في حساب Final Opportunity Score الحالي "
               "(المعتمد فقط على Liquidity Score وTiming Score) — يُعرضان هنا للمرجعية فقط.")

st.divider()

# ------------------------------------------------------------
# أفضل 3 مواعيد مستقبلية لارتفاع السيولة (Repeat Strength + Liquidity Lift)
# ------------------------------------------------------------
st.subheader("📅 أفضل مواعيد ارتفاع السيولة القادمة")

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
best_dates_available = any(pd.notna(row.get(f"best_date_{i}")) for i in (1, 2, 3))

if not best_dates_available:
    current_day_now = int(row["current_day_in_cycle"])
    if current_day_now >= 15:
        st.info("لا توجد جلسات متبقية داخل الدورة الحالية (الشركة في اليوم الأخير من الدورة) — "
                "سيُعاد تقييم أفضل المواعيد فور بدء الدورة القادمة عند التحديث التالي، دون اختلاق نتيجة حاليًا.")
    else:
        st.info("بيانات تاريخية غير كافية بعد لحساب مواعيد مستقبلية موثوقة لهذه الشركة (تحتاج 3 دورات مكتملة على الأقل).")
else:
    medal_cols = st.columns(3)
    for i in (1, 2, 3):
        best_date = row.get(f"best_date_{i}")
        col = medal_cols[i - 1]
        if pd.isna(best_date):
            col.markdown(f'<div class="metric-card">{MEDALS[i]}<br>لا يوجد موعد {i} متاح</div>', unsafe_allow_html=True)
            continue

        cycle_day = int(row[f"best_cycle_day_{i}"])
        sessions_away = int(row[f"best_sessions_away_{i}"])
        repeat_strength_i = row[f"best_repeat_strength_{i}"]
        lift_i = row[f"best_liquidity_lift_{i}"]
        score_i = row[f"best_future_score_{i}"]
        label_i = row[f"best_label_{i}"]
        date_txt = pd.Timestamp(best_date).strftime("%d/%m/%Y")

        col.markdown(
            f'''<div class="metric-card" style="text-align:right; line-height:1.9;">
            <div style="font-size:1.3rem">{MEDALS[i]} الموعد {"الأول" if i==1 else "الثاني" if i==2 else "الثالث"}</div>
            📅 {date_txt}<br>
            اليوم {cycle_day} من الدورة<br>
            بعد {sessions_away} {"جلسة" if sessions_away == 1 else "جلسات"}<br>
            🔁 تكرار: {repeat_strength_i:.0f}%<br>
            📈 متوسط ارتفاع: +{lift_i:.0f}%<br>
            ⭐ Future Score: {score_i:.0f}<br>
            {label_i}
            </div>''',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # تفسير واضح (بدون Black Box) لسبب اختيار الموعد الأول
    # ------------------------------------------------------------
    if pd.notna(row.get("best_date_1")):
        cd1 = int(row["best_cycle_day_1"])
        rs1 = row["best_repeat_strength_1"]
        lift1 = row["best_liquidity_lift_1"]
        # عدد الدورات الفعلي المستخدم يُقرأ من repeat_strength_df إن توفر لدقة الشرح
        n_rises_txt = ""
        n_avail_txt = ""
        if len(repeat_strength_df):
            day_info = repeat_strength_df[repeat_strength_df["day_in_cycle"] == cd1]
            if len(day_info):
                n_rises_txt = int(day_info.iloc[0]["n_rises"])
                n_avail_txt = int(day_info.iloc[0]["n_available_cycles"])

        if n_avail_txt != "":
            explanation = (
                f'تم اختيار اليوم **{cd1}** لأنه شهد ارتفاعًا فعليًا في السيولة (حجم تداول أعلى من متوسطه المرجعي) '
                f'في **{n_rises_txt} من أصل {n_avail_txt}** دورة تاريخية مكتملة، بنسبة تكرار **{rs1:.0f}%**، '
                f'وكان متوسط قوة الارتفاع في تلك الدورات **+{lift1:.0f}%** فوق المتوسط المرجعي.'
            )
        else:
            explanation = (
                f'تم اختيار اليوم **{cd1}** بنسبة تكرار تاريخي **{rs1:.0f}%** ومتوسط ارتفاع سيولة **+{lift1:.0f}%**.'
            )
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

st.caption("Future Date Score = (Repeat Strength × 60%) + (Liquidity Lift Score × 40%) — نفس منطق التكرار عبر الدورات "
           "المستخدم في expected_sessions_to_liquidity، مطبَّقًا على كل الأيام المتبقية بالدورة الحالية لاختيار أفضل 3 مواعيد. "
           "لا يتم الالتفاف تلقائيًا لدورة جديدة. رياضي/إحصائي بحت، بدون AI/ML، وليس توصية شراء أو بيع.")

st.divider()

# ------------------------------------------------------------
# [1] السعر والإغلاق + [6] أيام السيولة المرتفعة على المخطط
# ------------------------------------------------------------
st.subheader("📉 السعر والإغلاق (آخر 180 جلسة) — مع تعليم أيام السيولة المرتفعة")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df["date"], y=df["high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig_price.add_trace(go.Scatter(x=df["date"], y=df["low"], mode="lines", fill="tonexty", line=dict(width=0),
                                fillcolor="rgba(47,129,247,0.15)", name="نطاق High-Low", hoverinfo="skip"))
fig_price.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="الإغلاق", line=dict(color="#2f81f7", width=1.5)))

high_days = df[df["high_liquidity_L2"] == True]  # noqa: E712
fig_price.add_trace(go.Scatter(x=high_days["date"], y=high_days["close"], mode="markers", name="سيولة مرتفعة (RelVol>1.5)",
                                marker=dict(color="#FCB07C", size=9, symbol="triangle-up", line=dict(color="black", width=0.5))))
fig_price.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_price, width='stretch')

# ------------------------------------------------------------
# [2] حجم التداول والسيولة + [3] متوسط السيولة
# ------------------------------------------------------------
st.subheader("📊 حجم التداول والسيولة")
liquidity = df["close"] * df["volume"]
avg_liquidity_60 = liquidity.tail(60).mean()

fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["date"], y=df["volume"], name="الحجم", marker_color="#4a5568"))
fig_vol.add_trace(go.Scatter(x=df["date"], y=df["relative_volume"], name="RelativeVolume", yaxis="y2", line=dict(color="#FC7C7C", width=1.3)))
fig_vol.add_hline(y=1.5, line=dict(color="#FCB07C", dash="dot"), yref="y2")
fig_vol.update_layout(
    template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"),
    yaxis=dict(title="Volume"), yaxis2=dict(title="RelativeVolume", overlaying="y", side="right"),
)
st.plotly_chart(fig_vol, width='stretch')
st.markdown(f"**متوسط السيولة (آخر 60 جلسة):** {avg_liquidity_60:,.0f}")

st.divider()

# ------------------------------------------------------------
# [4] آخر 180 جلسة — جدول
# ------------------------------------------------------------
with st.expander(f"📋 جدول آخر {len(df)} جلسة"):
    show_cols = ["date", "open", "high", "low", "close", "volume", "relative_volume", "day_in_cycle", "cycle_number", "high_liquidity_L2"]
    st.dataframe(df[show_cols].sort_values("date", ascending=False), width='stretch', height=400)

# ------------------------------------------------------------
# [5] و [8] تقسيم الـ12 دورة + جدول الدورات
# ------------------------------------------------------------
st.subheader("🔄 تقسيم الدورات (12 دورة × 15 جلسة)")
if len(cycle_table):
    fig_cycle = go.Figure()
    fig_cycle.add_trace(go.Bar(x=cycle_table["DayInCycle"], y=cycle_table["ProbabilityHighLiquidity_L2"], marker_color="#2f81f7"))
    fig_cycle.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title="DayInCycle", yaxis_title="احتمال سيولة مرتفعة (%)")
    st.plotly_chart(fig_cycle, width='stretch')

    with st.expander("جدول الأيام داخل الدورة (12 دورة)"):
        st.dataframe(cycle_table, width='stretch')

# heatmap دورة × يوم
if "cycle_number" in df.columns:
    pivot = df.pivot_table(index="cycle_number", columns="day_in_cycle", values="relative_volume")
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.to_numpy(), x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
        colorscale="YlOrRd", colorbar=dict(title="RelVol"),
    ))
    fig_heat.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=40, t=30, b=10),
                            xaxis_title="DayInCycle", yaxis_title="الدورة")

    # تمييز أفضل 3 مواعيد مستقبلية (🥇🥈🥉) فوق العمود المقابل لها في الـ Heatmap — بدون تغيير الرسم نفسه
    for i in (1, 2, 3):
        best_day_i = row.get(f"best_cycle_day_{i}")
        if pd.notna(best_day_i):
            fig_heat.add_annotation(
                x=str(int(best_day_i)), y=1.06, xref="x", yref="paper",
                text=MEDALS.get(i, ""), showarrow=False, font=dict(size=18),
            )

    st.plotly_chart(fig_heat, width='stretch')
    if any(pd.notna(row.get(f"best_cycle_day_{i}")) for i in (1, 2, 3)):
        st.caption("🥇🥈🥉 فوق الأعمدة تشير إلى أفضل 3 مواعيد مستقبلية مرشّحة داخل الدورة الحالية (انظر التفاصيل أعلاه).")

# ------------------------------------------------------------
# [7] خط زمني يوضح الموقع الحالي
# ------------------------------------------------------------
st.subheader("🧭 الموقع الحالي ضمن النمط")
recent = df.tail(2 * DAYS_PER_CYCLE).reset_index(drop=True)
fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(x=recent["date"], y=recent["relative_volume"], mode="lines+markers",
                                   name="RelativeVolume", line=dict(color="#2f81f7")))
hl = recent[recent["high_liquidity_L2"] == True]  # noqa: E712
fig_timeline.add_trace(go.Scatter(x=hl["date"], y=hl["relative_volume"], mode="markers", name="سيولة مرتفعة",
                                   marker=dict(color="#FCB07C", size=11, symbol="star")))
if len(recent):
    last = recent.iloc[-1]
    fig_timeline.add_trace(go.Scatter(x=[last["date"]], y=[last["relative_volume"]], mode="markers+text",
                                       marker=dict(color="#7CFCA0", size=14, symbol="diamond"),
                                       text=["← الموقع الحالي"], textposition="middle right", name="اليوم"))
fig_timeline.add_hline(y=1.5, line=dict(color="gray", dash="dot"))
fig_timeline.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig_timeline, width='stretch')
st.caption(f"الموقع الحالي: الدورة {int(row['current_cycle'])}، اليوم {int(row['current_day_in_cycle'])} من {DAYS_PER_CYCLE}.")

st.divider()

# ------------------------------------------------------------
# [9] الأيام/الجلسات المتكررة في السيولة المرتفعة
# ------------------------------------------------------------
st.subheader("🔁 تكرار أيام السيولة المرتفعة عبر الدورات")
if len(top_days_df):
    valid_top = top_days_df[top_days_df.get("valid", True) == True]  # noqa: E712
    if "HighestLiquidityDay" in valid_top.columns and len(valid_top):
        counts = valid_top["HighestLiquidityDay"].value_counts().reindex(range(1, DAYS_PER_CYCLE + 1), fill_value=0)
        fig_rep = go.Figure(go.Bar(x=[str(i) for i in counts.index], y=counts.values, marker_color="#7CFCA0"))
        fig_rep.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="DayInCycle", yaxis_title="عدد المرات كأعلى يوم سيولة")
        st.plotly_chart(fig_rep, width='stretch')
    with st.expander("جدول أعلى 3 أيام سيولة لكل دورة"):
        st.dataframe(top_days_df, width='stretch')
else:
    st.info("لا توجد بيانات كافية لتحليل التكرار بعد.")

st.divider()

# ------------------------------------------------------------
# [10] و [11] التوقع القادم ودرجة الثقة
# ------------------------------------------------------------
st.subheader("🔮 التوقع القادم للسيولة")
horizon_label = row.get("forecast_horizon_label", simplified_horizon_label(None))
st.markdown(f"**الخلاصة المبسّطة:** السيولة المرتفعة {horizon_label}")

if len(forecast_df):
    fig_fc = go.Figure()
    colors = ["#FC7C7C" if p >= 0.5 else "#2f81f7" for p in forecast_df["forecast_probability"].fillna(0)]
    fig_fc.add_trace(go.Bar(x=forecast_df["session_offset"], y=forecast_df["forecast_probability"] * 100, marker_color=colors))
    fig_fc.add_hline(y=50, line=dict(color="gray", dash="dash"))
    fig_fc.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="عدد الجلسات القادمة", yaxis_title="احتمال السيولة المرتفعة (%)")
    st.plotly_chart(fig_fc, width='stretch')

    show_fc_cols = [c for c in ["session_offset", "day_in_cycle", "forecast_probability", "expected_relative_volume",
                                 "high_liquidity_probability_label", "confidence_level"] if c in forecast_df.columns]
    st.dataframe(forecast_df[show_fc_cols], width='stretch')
else:
    st.info("لا توجد بيانات كافية لبناء توقع حاليًا لهذه الشركة.")

# ------------------------------------------------------------
# [12] شرح مختصر للتوقع
# ------------------------------------------------------------
st.subheader("ℹ️ كيف تم الوصول إلى هذا التوقع؟")
st.markdown(
    """
<div class="explain-box">
يعتمد التوقع على انحدار لوجستي (Logistic Regression) مبني من نفس منهجية سهم TADAWUL:4140، يجمع بين:
<ul>
<li><b>الاحتمال التاريخي</b> لظهور سيولة مرتفعة في نفس موقع اليوم داخل دورة الـ15 جلسة.</li>
<li><b>متوسط RelativeVolume التاريخي</b> لنفس موقع اليوم.</li>
<li><b>عدد الجلسات منذ آخر سيولة مرتفعة</b> نسبةً إلى متوسط الفجوة التاريخي.</li>
<li><b>زخم قصير المدى</b>: قيمة RelativeVolume في الجلسة السابقة ومتوسط آخر 5 جلسات واتجاهها.</li>
<li><b>طور دوري (Harmonic Phase)</b> مبني على أقوى دورة رياضية مكتشفة إحصائيًا (ACF + Periodogram + FFT).</li>
</ul>
كل هذه المتغيرات <b>سببية (Causal)</b> — أي أنها تعتمد فقط على بيانات معروفة قبل الجلسة المتوقَّعة، بدون أي تسرّب معلومات مستقبلية.
كلما ابتعدت الجلسة المتوقَّعة عن اليوم الحالي، قلّت دقة ميزات الزخم القصيرة (لأنها تفترض ثبات آخر قيمة معروفة)، ولذلك تقلّ درجة الثقة تدريجيًا (High → Medium → Low).
<br><br>
<b>هذا تحليل نمطي إحصائي فقط، ولا يمثّل توصية شراء أو بيع بأي شكل.</b>
</div>
""",
    unsafe_allow_html=True,
)
