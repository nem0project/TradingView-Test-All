"""
الصفحة الرئيسية للوحة السيولة — تعرض جدول كل الشركات المتابَعة (من data/symbols.csv)
مع نتائج تحليل دورية السيولة (نفس منطق سهم TADAWUL:4140) لكل شركة.

مبنية للتوسّع إلى مئات الأسهم (السوق السعودي الرئيسي بالكامل) — الجدول الرئيسي يستخدم
st.dataframe (عرض وفرز أصلي سريع) بدل حلقة صفوف يدوية، مع بحث وفلاتر واختيار صف للانتقال
مباشرة لصفحة تفاصيل الشركة.

تشغيل:
    streamlit run dashboard/Home.py
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

# --- إتاحة استيراد core / historical_180 / liquidity_* من جذر المشروع ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.data_store import ANALYSIS_DIR, SYMBOLS_CSV, read_symbols  # noqa: E402
from core.liquidity_engine import simplified_horizon_label  # noqa: E402

SUMMARY_CSV = os.path.join(ANALYSIS_DIR, "summary.csv")
UPDATE_STATUS_FILE = os.path.join(ANALYSIS_DIR, "last_update_status.json")

st.set_page_config(page_title="لوحة سيولة الأسهم", page_icon="📊", layout="wide")

RTL_CSS = """
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, Arial, sans-serif; }
[data-testid="stSidebar"] { direction: rtl; text-align: right; }
.badge { display:inline-block; padding:3px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
.badge-green  { background:#123d24; color:#7CFCA0; }
.badge-yellow { background:#3d3812; color:#FCE87C; }
.badge-orange { background:#3d2712; color:#FCB07C; }
.badge-red    { background:#3d1414; color:#FC7C7C; }
.badge-gray   { background:#2a2a2a; color:#bbbbbb; }
.badge-fire   { background:#4a1c0a; color:#FFA057; }
.metric-card { background:#161a22; border:1px solid #262c37; border-radius:10px; padding:14px 16px; }
.metric-card .val { font-size:1.5rem; font-weight:700; }
.mini-list { background:#161a22; border:1px solid #262c37; border-radius:10px; padding:12px 16px; }
.mini-list table { width:100%; font-size:0.85rem; }
.mini-list td { padding:3px 0; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

COLOR_EMOJI = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴", "gray": "⚪", "fire": "🔥"}


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


def fmt_date(x):
    if pd.isna(x):
        return "—"
    return pd.Timestamp(x).strftime("%Y-%m-%d")


st.title("📊 لوحة سيولة الأسهم — السوق السعودي الرئيسي (TASI)")
st.caption("تحليل مبني على نفس منطق liquidity cycle analysis المطبَّق أولًا على TADAWUL:4140 — هذه ليست توصية شراء أو بيع.")

# ------------------------------------------------------------
# قراءة حالة آخر تحديث + حالة مصدر البيانات (LIVE / CACHED / ERROR)
# ------------------------------------------------------------
update_status = None
if os.path.exists(UPDATE_STATUS_FILE):
    try:
        with open(UPDATE_STATUS_FILE, "r", encoding="utf-8") as f:
            update_status = json.load(f)
    except Exception:
        update_status = None

if update_status:
    finished_at = pd.Timestamp(update_status["finished_at"])
    overall = update_status.get("overall_status", "unknown")
    age_hours = (pd.Timestamp.now() - finished_at).total_seconds() / 3600

    if overall == "success" and age_hours <= 36:
        data_source_label, data_source_color = "LIVE 🟢", "#7CFCA0"
    elif overall in ("success", "partial") and age_hours <= 96:
        data_source_label, data_source_color = "CACHED 🟡", "#FCE87C"
    else:
        data_source_label, data_source_color = "ERROR 🔴", "#FC7C7C"

    status_map = {"success": "ناجح ✅", "partial": "جزئي ⚠️", "failed": "فشل ❌"}
    status_txt = status_map.get(overall, "غير معروف")
    failed_n = update_status.get("failed_count", 0)
    extra = f" ({failed_n} شركة فشلت)" if failed_n else ""

    st.markdown(
        f'<div style="font-size:0.9rem; margin-bottom:10px;">'
        f'آخر تحديث ناجح: <b>{finished_at.strftime("%Y-%m-%d %H:%M")}</b> &nbsp;|&nbsp; '
        f'حالة التحديث: <b>{status_txt}</b>{extra} &nbsp;|&nbsp; '
        f'حالة مصدر البيانات: <span style="color:{data_source_color}"><b>{data_source_label}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    data_source_label = "ERROR 🔴"
    st.caption("لا يوجد سجل تحديث تلقائي بعد — شغّل py update_all.py أو فعّل المهمة المجدولة.")

symbols_df = read_symbols(include_inactive=True)

if not os.path.exists(SUMMARY_CSV):
    st.warning("لا توجد نتائج تحليل بعد. شغّل التحديث أولًا من الطرفية:")
    st.code("py update_all.py", language="bash")
    st.stop()

summary_df = pd.read_csv(SUMMARY_CSV, parse_dates=["last_date", "last_high_liquidity_date"], dtype={"symbol": str})

# ------------------------------------------------------------
# مؤشرات علوية سريعة (تشمل كامل سوق TASI وليس فقط ما نجح تحليله)
# ------------------------------------------------------------
n_total_market = int((symbols_df.get("status", "active") == "active").sum()) if len(symbols_df) else len(summary_df)
n_analyzed = len(summary_df)
n_failed = n_total_market - n_analyzed if n_total_market >= n_analyzed else 0
if update_status:
    n_failed = update_status.get("failed_count", n_failed)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card">إجمالي أسهم TASI<br><span class="val">{n_total_market}</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card">تم تحليلها بنجاح<br><span class="val" style="color:#7CFCA0">{n_analyzed}</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card">فشل تحليلها<br><span class="val" style="color:#FC7C7C">{n_failed}</span></div>', unsafe_allow_html=True)
with c4:
    n_watch = int((summary_df["status_color"] == "orange").sum())
    st.markdown(f'<div class="metric-card">تحتاج مراقبة<br><span class="val" style="color:#FCB07C">{n_watch}</span></div>', unsafe_allow_html=True)
with c5:
    last_update = summary_df["last_date"].max()
    st.markdown(f'<div class="metric-card">آخر تحديث بيانات<br><span class="val" style="font-size:1.1rem">{fmt_date(last_update)}</span></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# أعلى الأسهم حسب السيولة وحسب حجم التداول
# ------------------------------------------------------------
top_col1, top_col2 = st.columns(2)
with top_col1:
    top_liq = summary_df.nlargest(5, "avg_liquidity_60")[["symbol", "name", "avg_liquidity_60"]]
    rows_html = "".join(
        f"<tr><td>{r['symbol']}</td><td>{r['name']}</td><td>{fmt_num(r['avg_liquidity_60'], 0)}</td></tr>"
        for _, r in top_liq.iterrows()
    )
    st.markdown(f'<div class="mini-list"><b>💧 أعلى 5 أسهم حسب السيولة (متوسط 60 جلسة)</b><table>{rows_html}</table></div>', unsafe_allow_html=True)
with top_col2:
    top_vol = summary_df.nlargest(5, "last_volume")[["symbol", "name", "last_volume"]]
    rows_html = "".join(
        f"<tr><td>{r['symbol']}</td><td>{r['name']}</td><td>{fmt_num(r['last_volume'], 0)}</td></tr>"
        for _, r in top_vol.iterrows()
    )
    st.markdown(f'<div class="mini-list"><b>📦 أعلى 5 أسهم حسب حجم التداول (آخر جلسة)</b><table>{rows_html}</table></div>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------
# البحث والفلاتر
# ------------------------------------------------------------
search_col, filter_col1, filter_col2, filter_col3 = st.columns([1.4, 1, 1, 1])
with search_col:
    search_text = st.text_input("🔍 ابحث بالرمز أو اسم الشركة", "")
with filter_col1:
    status_filter = st.multiselect(
        "حالة النمط:",
        options=["green", "yellow", "orange", "red", "gray"],
        default=["green", "yellow", "orange", "red", "gray"],
        format_func=lambda c: f"{COLOR_EMOJI.get(c,'⚪')} " + {"green": "متوقع قريبًا", "yellow": "متوسط المدى", "orange": "يحتاج مراقبة", "red": "لا يوجد نمط واضح", "gray": "بيانات غير كافية"}.get(c, c),
    )
with filter_col2:
    min_score = st.slider("أدنى Final Score", 0, 100, 0)
with filter_col3:
    min_liquidity = st.number_input("أدنى متوسط سيولة (60 جلسة)", min_value=0, value=0, step=100000)

view_df = summary_df[summary_df["status_color"].isin(status_filter)].copy()
view_df = view_df[view_df["final_opportunity_score"].fillna(0) >= min_score]
view_df = view_df[view_df["avg_liquidity_60"].fillna(0) >= min_liquidity]
if search_text.strip():
    q = search_text.strip().lower()
    view_df = view_df[
        view_df["symbol"].str.lower().str.contains(q, na=False)
        | view_df["name"].str.lower().str.contains(q, na=False)
    ]

# ------------------------------------------------------------
# ترتيب تلقائي: final_opportunity_score من الأعلى إلى الأقل — معيار الترتيب الرئيسي
# (مجموع موزون: 60% liquidity_score + 40% timing_score — زخم السيولة الفعلي + التوقيت فقط، بدون أي مؤشر سعري)
# ------------------------------------------------------------
view_df = view_df.sort_values(by="final_opportunity_score", ascending=False).reset_index(drop=True)

st.caption(f"يعرض {len(view_df)} من أصل {len(summary_df)} سهمًا محلَّلًا. اضغط على أي صف لفتح صفحة تفاصيل الشركة.")

# ------------------------------------------------------------
# جدول الشركات (Screener) — st.dataframe سريع وقابل للفرز حتى مع مئات الأسهم.
# كل التفاصيل الحسابية الإضافية (Liquidity Score, Timing Score, السيولة المرجعية،
# الموقع بالدورة، احتمال/قوة النمط...) متاحة في صفحة تفاصيل الشركة عند الضغط على الصف.
# ------------------------------------------------------------
display_df = pd.DataFrame({
    "الرمز": view_df["symbol"],
    "اسم الشركة": view_df["name"],
    "آخر إغلاق": view_df["last_close"],
    "التغير %": view_df["daily_change_pct"],
    "حجم التداول": view_df["last_volume"],
    "تغيّر السيولة %": view_df["liquidity_change_percent"],
    "الموعد المتوقع خلال": view_df["expected_sessions_to_liquidity"].apply(
        lambda x: simplified_horizon_label(None if pd.isna(x) else int(x))
    ),
    "حالة النمط": view_df.apply(lambda r: f"{COLOR_EMOJI.get(r['status_color'],'⚪')} {r['pattern_status']}", axis=1),
    "Final Score": view_df["final_opportunity_score"],
    "الفرصة القادمة": view_df.apply(lambda r: f"{COLOR_EMOJI.get(r['final_opportunity_color'],'⚪')} {r['final_opportunity_label']}", axis=1),
})

event = st.dataframe(
    display_df,
    hide_index=True,
    width="stretch",
    height=min(600, 60 + 35 * max(len(display_df), 1)),
    column_config={
        "آخر إغلاق": st.column_config.NumberColumn(format="%.2f"),
        "التغير %": st.column_config.NumberColumn(format="%+.2f%%"),
        "حجم التداول": st.column_config.NumberColumn(format="%,d"),
        "تغيّر السيولة %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Final Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    },
    selection_mode="single-row",
    on_select="rerun",
    key="screener_table",
)

selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    selected_symbol = view_df.iloc[selected_rows[0]]["symbol"]
    st.session_state["selected_symbol"] = selected_symbol
    try:
        st.switch_page("pages/1_تفاصيل_الشركة.py")
    except Exception:
        st.info(f"افتح صفحة 'تفاصيل الشركة' من الشريط الجانبي واختر الرمز {selected_symbol} يدويًا.")

st.divider()
st.caption("ترتيب الجدول حسب «Final Score» = (Liquidity Score × 60%) + (Timing Score × 40%) — زخم السيولة الفعلي "
           "لآخر 3 جلسات (Volume فقط، بلا أي سعر) + قرب التوقيت المتوقع للسيولة (نافذة 3-5 جلسات هي الأمثل): "
           "🔥 فرصة قوية جدًا (80-100) — 🟢 فرصة قوية (65-79) — 🟡 فرصة متوسطة (50-64) — 🟠 تحتاج مراقبة (35-49) — 🔴 فرصة ضعيفة (أقل من 35). "
           "كل تفاصيل الحساب (Liquidity Score, Timing Score, السيولة المرجعية، الموقع بالدورة...) متاحة في صفحة تفاصيل كل شركة. "
           "كل هذا تحليل رياضي/إحصائي بحت (بدون أي نموذج ذكاء اصطناعي)، وليس إشارة شراء أو بيع.")
