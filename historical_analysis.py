import json
import os
import random
import string
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import talib
import websocket
from scipy import stats

# ============================================================
# إعدادات عامة
# ============================================================

SYMBOL = "TADAWUL:4140"
COMPANY_NAME = "الشركة السعودية للصادرات الصناعية (صادرات)"
N_BARS = 49
SESSIONS_PER_WEEK = 7

DATA_DIR = "data"
DAILY_CSV = os.path.join(DATA_DIR, "4140_daily.csv")
WEEKLY_CSV = os.path.join(DATA_DIR, "4140_weekly.csv")
ANALYSIS_CSV = os.path.join(DATA_DIR, "4140_analysis.csv")
INDICATORS_CSV = os.path.join(DATA_DIR, "4140_indicators.csv")

pd.set_option("display.width", 160)


# ============================================================
# 0) طبقة اتصال TradingView WebSocket (chart session)
#    نفس أسلوب التأطير المستخدم في tv_test.py (~m~<len>~m~<payload>)
#    مبني بشكل مستقل حتى لا نستورد tv_test.py (فهو يشغّل حلقة اتصال حية عند تحميله).
# ============================================================

def generate_session(prefix):
    chars = string.ascii_lowercase
    return prefix + "".join(random.choice(chars) for _ in range(12))


def send_message(ws, method, params):
    message = json.dumps({"m": method, "p": params}, separators=(",", ":"))
    ws.send(f"~m~{len(message)}~m~{message}")


def extract_messages(buffer):
    """
    يفصل رسائل TradingView المؤطّرة بالصيغة ~m~<length>~m~<payload>
    اعتمادًا على طول كل رسالة (بدون regex) — نفس منطق tv_test.py.
    """
    messages = []
    pos = 0
    size = len(buffer)

    while buffer.startswith("~m~", pos):
        len_start = pos + 3
        len_end = buffer.find("~m~", len_start)
        if len_end == -1:
            break

        length_str = buffer[len_start:len_end]
        if not length_str.isdigit():
            break

        length = int(length_str)
        msg_start = len_end + 3
        msg_end = msg_start + length

        if msg_end > size:
            break

        messages.append(buffer[msg_start:msg_end])
        pos = msg_end

    return messages, buffer[pos:]


class TradingViewProtocolError(Exception):
    pass


def fetch_historical_bars(symbol, n_bars):
    """
    يجلب آخر n_bars جلسة يومية (OHLCV) فعلية من TradingView عبر chart session
    (resolve_symbol + create_series) — بروتوكول تم التحقق منه فعليًا وليس افتراضيًا.
    """
    chart_session = generate_session("cs_")

    ws = websocket.create_connection(
        "wss://data.tradingview.com/socket.io/websocket",
        origin="https://data.tradingview.com",
        timeout=30,
    )

    try:
        send_message(ws, "set_auth_token", ["unauthorized_user_token"])
        send_message(ws, "chart_create_session", [chart_session, ""])
        send_message(
            ws,
            "resolve_symbol",
            [
                chart_session,
                "symbol_1",
                "=" + json.dumps({"symbol": symbol, "adjustment": "splits"}, separators=(",", ":")),
            ],
        )
        send_message(ws, "create_series", [chart_session, "s1", "s1", "symbol_1", "D", n_bars, ""])

        buffer = ""
        bars_raw = None
        symbol_info = None
        completed = False

        while not completed:
            raw = ws.recv()
            buffer += raw
            messages, buffer = extract_messages(buffer)

            for message in messages:
                if message.startswith("~h~"):
                    ws.send(f"~m~{len(message)}~m~{message}")
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                m_type = data.get("m")

                if m_type in ("protocol_error", "critical_error"):
                    raise TradingViewProtocolError(f"خطأ بروتوكول من TradingView: {data}")

                if m_type == "symbol_error":
                    raise TradingViewProtocolError(f"فشل TradingView في إيجاد الرمز {symbol}: {data}")

                if m_type == "symbol_resolved":
                    payload = data.get("p", [])
                    if len(payload) >= 3:
                        symbol_info = payload[2]

                if m_type in ("timescale_update", "du"):
                    payload = data.get("p", [])
                    if len(payload) >= 2 and "s1" in payload[1]:
                        s_list = payload[1]["s1"].get("s", [])
                        if s_list:
                            bars_raw = s_list

                if m_type == "series_completed":
                    completed = True
                    break
    finally:
        ws.close()

    if not bars_raw:
        raise TradingViewProtocolError(
            "لم يصل أي بيانات تاريخية (timescale_update) من TradingView قبل series_completed."
        )

    rows = []
    for item in bars_raw:
        v = item.get("v", [])
        if len(v) < 6:
            continue
        ts, o, h, l, c, vol = v[0], v[1], v[2], v[3], v[4], v[5]
        rows.append(
            {
                "timestamp": int(ts),
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(vol),
            }
        )

    rows.sort(key=lambda r: r["timestamp"])  # الأقدم -> الأحدث

    return rows, symbol_info


# ============================================================
# 1) بناء DataFrame اليومي + السيولة
# ============================================================

def build_daily_dataframe(rows):
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["liquidity"] = df["close"] * df["volume"]
    df["daily_liquidity_change_pct"] = df["liquidity"].pct_change() * 100
    df["daily_return_pct"] = df["close"].pct_change() * 100

    return df


def save_daily_csv(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df[["date", "open", "high", "low", "close", "volume"]].to_csv(DAILY_CSV, index=False)


# ============================================================
# 2) تقسيم الجلسات إلى أسابيع تداول (مجموعات من 7 جلسات، وليس أيام تقويمية)
# ============================================================

def split_into_weeks(df, sessions_per_week=SESSIONS_PER_WEEK):
    weeks = []
    for start in range(0, len(df), sessions_per_week):
        chunk = df.iloc[start:start + sessions_per_week]
        if len(chunk) == sessions_per_week:
            weeks.append(chunk.reset_index(drop=True))
    return weeks


def compute_weekly_stats(weeks):
    records = []
    prev_avg_liquidity = None
    prev_last_close = None

    for i, week in enumerate(weeks, start=1):
        first_close = week["close"].iloc[0]
        last_close = week["close"].iloc[-1]
        max_close = week["close"].max()
        min_close = week["close"].min()
        close_change_pct = (last_close - first_close) / first_close * 100

        sum_volume = week["volume"].sum()
        avg_volume = week["volume"].mean()

        sum_liquidity = week["liquidity"].sum()
        avg_liquidity = week["liquidity"].mean()
        max_session_liquidity = week["liquidity"].max()
        min_session_liquidity = week["liquidity"].min()

        if prev_avg_liquidity is None:
            liquidity_change_pct = np.nan
        else:
            liquidity_change_pct = (avg_liquidity - prev_avg_liquidity) / prev_avg_liquidity * 100

        if prev_last_close is None:
            weekly_return_pct = np.nan
        else:
            weekly_return_pct = (last_close - prev_last_close) / prev_last_close * 100

        records.append(
            {
                "week_number": i,
                "start_date": week["date"].iloc[0],
                "end_date": week["date"].iloc[-1],
                "first_close": first_close,
                "last_close": last_close,
                "max_close": max_close,
                "min_close": min_close,
                "close_change_pct": close_change_pct,
                "sum_volume": sum_volume,
                "avg_volume": avg_volume,
                "sum_liquidity": sum_liquidity,
                "avg_liquidity": avg_liquidity,
                "liquidity_change_pct": liquidity_change_pct,
                "max_session_liquidity": max_session_liquidity,
                "min_session_liquidity": min_session_liquidity,
                "weekly_return_pct": weekly_return_pct,
            }
        )

        prev_avg_liquidity = avg_liquidity
        prev_last_close = last_close

    return pd.DataFrame(records)


def save_weekly_csv(weekly_df):
    os.makedirs(DATA_DIR, exist_ok=True)
    weekly_df.to_csv(WEEKLY_CSV, index=False)


# ============================================================
# 3) العلاقة الرياضية بين السيولة والإغلاق (أسبوعي ويومي)
# ============================================================

def weekly_relationship(weekly_df):
    sub = weekly_df.dropna(subset=["liquidity_change_pct", "weekly_return_pct"])
    x = sub["liquidity_change_pct"].to_numpy(dtype=float)
    y = sub["weekly_return_pct"].to_numpy(dtype=float)

    if len(x) < 2:
        return None

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    covariance = np.cov(x, y)[0][1]
    reg = stats.linregress(x, y)

    return {
        "n": len(x),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "covariance": covariance,
        "slope": reg.slope,
        "intercept": reg.intercept,
        "r_squared": reg.rvalue ** 2,
        "p_value": reg.pvalue,
    }


def daily_relationship(df_slice):
    sub = df_slice.dropna(subset=["daily_return_pct", "daily_liquidity_change_pct"])
    x = sub["daily_liquidity_change_pct"].to_numpy(dtype=float)
    y = sub["daily_return_pct"].to_numpy(dtype=float)

    if len(x) < 2:
        return None

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    reg = stats.linregress(x, y)

    return {
        "n": len(x),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": reg.slope,
        "intercept": reg.intercept,
        "r_squared": reg.rvalue ** 2,
        "p_value": reg.pvalue,
    }


# ============================================================
# 4) المؤشرات الفنية (TA-Lib)
# ============================================================

def compute_indicators(df):
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    liquidity = df["liquidity"].to_numpy(dtype=np.float64)

    out = df.copy()

    out["sma5"] = talib.SMA(close, timeperiod=5)
    out["sma10"] = talib.SMA(close, timeperiod=10)
    out["sma20"] = talib.SMA(close, timeperiod=20)

    out["liquidity_sma5"] = talib.SMA(liquidity, timeperiod=5)
    out["liquidity_sma10"] = talib.SMA(liquidity, timeperiod=10)
    out["liquidity_sma20"] = talib.SMA(liquidity, timeperiod=20)

    out["rsi14"] = talib.RSI(close, timeperiod=14)
    out["atr14"] = talib.ATR(high, low, close, timeperiod=14)
    out["roc"] = talib.ROC(close, timeperiod=10)
    out["momentum"] = talib.MOM(close, timeperiod=10)
    out["obv"] = talib.OBV(close, volume)
    out["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)

    typical_price = (high + low + close) / 3.0
    out["vwap"] = np.cumsum(typical_price * volume) / np.cumsum(volume)

    return out


def save_indicators_csv(indicators_df):
    os.makedirs(DATA_DIR, exist_ok=True)
    cols = [
        "date", "close", "volume", "liquidity",
        "daily_return_pct", "daily_liquidity_change_pct",
        "sma5", "sma10", "sma20",
        "liquidity_sma5", "liquidity_sma10", "liquidity_sma20",
        "rsi14", "atr14", "roc", "momentum", "obv", "mfi", "vwap",
        "interpretation",
    ]
    indicators_df[cols].to_csv(INDICATORS_CSV, index=False)


# ============================================================
# 5) التفسير الرقمي للعلاقة بين السيولة والسعر (بدون توصيات شراء/بيع)
# ============================================================

def interpret_row(liq_change_pct, price_change_pct):
    if pd.isna(liq_change_pct) or pd.isna(price_change_pct):
        return None
    if liq_change_pct > 0 and price_change_pct > 0:
        return "سيولة إيجابية مع حركة سعرية إيجابية"
    if liq_change_pct > 0 and price_change_pct < 0:
        return "سيولة مرتفعة مع ضغط بيعي محتمل"
    if liq_change_pct < 0 and price_change_pct > 0:
        return "ارتفاع سعري مع ضعف في السيولة"
    if liq_change_pct < 0 and price_change_pct < 0:
        return "ضعف سعري مصحوب بانخفاض السيولة"
    return "تغير محايد (لا حركة واضحة في السعر أو السيولة)"


def period_relationship_label(rel):
    if rel is None:
        return "عينة غير كافية لحساب علاقة إحصائية موثوقة."
    if rel["pearson_p"] < 0.05:
        direction = "طردية (موجبة)" if rel["pearson_r"] > 0 else "عكسية (سالبة)"
        return f"توجد علاقة إحصائية ذات دلالة ({direction}), Pearson p-value = {rel['pearson_p']:.4f} < 0.05."
    return f"لا توجد علاقة إحصائية ذات دلالة واضحة (Pearson p-value = {rel['pearson_p']:.4f} >= 0.05)."


# ============================================================
# 6) ملخصات فترات المقارنة (28 جلسة مقابل 49 جلسة)
# ============================================================

def compute_period_summary(full_df, n_sessions):
    period = full_df.tail(n_sessions).reset_index(drop=True)

    price_change_pct = (period["close"].iloc[-1] - period["close"].iloc[0]) / period["close"].iloc[0] * 100
    liquidity_change_pct = (
        (period["liquidity"].iloc[-1] - period["liquidity"].iloc[0]) / period["liquidity"].iloc[0] * 100
    )

    rel = daily_relationship(period)

    indicators = compute_indicators(period)
    rsi_last = indicators["rsi14"].iloc[-1]
    atr_last = indicators["atr14"].iloc[-1]
    momentum_last = indicators["momentum"].iloc[-1]

    return {
        "sessions": len(period),
        "start_date": period["date"].iloc[0],
        "end_date": period["date"].iloc[-1],
        "price_change_pct": price_change_pct,
        "liquidity_change_pct": liquidity_change_pct,
        "avg_liquidity": period["liquidity"].mean(),
        "avg_volume": period["volume"].mean(),
        "pearson_r": rel["pearson_r"] if rel else np.nan,
        "pearson_p": rel["pearson_p"] if rel else np.nan,
        "spearman_r": rel["spearman_r"] if rel else np.nan,
        "spearman_p": rel["spearman_p"] if rel else np.nan,
        "r_squared": rel["r_squared"] if rel else np.nan,
        "reg_p_value": rel["p_value"] if rel else np.nan,
        "rsi14_last": rsi_last,
        "atr14_last": atr_last,
        "momentum_last": momentum_last,
        "_rel": rel,
    }


def save_analysis_csv(summary_28, summary_49):
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for label, s in (("28", summary_28), ("49", summary_49)):
        rows.append(
            {
                "period_sessions": s["sessions"],
                "start_date": s["start_date"],
                "end_date": s["end_date"],
                "price_change_pct": s["price_change_pct"],
                "liquidity_change_pct": s["liquidity_change_pct"],
                "avg_liquidity": s["avg_liquidity"],
                "avg_volume": s["avg_volume"],
                "pearson_r": s["pearson_r"],
                "pearson_p": s["pearson_p"],
                "spearman_r": s["spearman_r"],
                "spearman_p": s["spearman_p"],
                "r_squared": s["r_squared"],
                "reg_p_value": s["reg_p_value"],
                "rsi14_last": s["rsi14_last"],
                "atr14_last": s["atr14_last"],
                "momentum_last": s["momentum_last"],
            }
        )
    pd.DataFrame(rows).to_csv(ANALYSIS_CSV, index=False)


# ============================================================
# أدوات طباعة
# ============================================================

def fmt(df, decimals=2, int_cols=()):
    view = df.copy()
    for col in view.columns:
        if col in int_cols:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
        elif pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.{decimals}f}")
    return view


def section(title):
    print()
    print(title)
    print("-" * len(title))


# ============================================================
# البرنامج الرئيسي
# ============================================================

def main():
    print("=" * 40)
    print("TADAWUL:4140 - EXPORTS")
    print("HISTORICAL ANALYSIS")
    print("=" * 40)

    print(f"\nجلب آخر {N_BARS} جلسة تداول فعلية لـ {SYMBOL} ({COMPANY_NAME}) عبر TradingView WebSocket ...")

    try:
        rows, symbol_info = fetch_historical_bars(SYMBOL, N_BARS)
    except TradingViewProtocolError as exc:
        print("\n[خطأ] فشل جلب البيانات التاريخية من TradingView.")
        print(f"السبب: {exc}")
        print("لن يتم إنشاء أي بيانات بديلة أو وهمية. الرجاء مراجعة الخطأ أعلاه.")
        return

    if len(rows) != N_BARS:
        print(
            f"\n[تنبيه] عدد الجلسات المستلمة فعليًا = {len(rows)} وليس {N_BARS}. "
            "سيتم المتابعة بالعدد الفعلي المتاح فقط (بدون أي بيانات مصطنعة)."
        )

    df = build_daily_dataframe(rows)
    save_daily_csv(df)

    weeks = split_into_weeks(df, SESSIONS_PER_WEEK)
    weekly_df = compute_weekly_stats(weeks)
    save_weekly_csv(weekly_df)

    indicators_df = compute_indicators(df)
    indicators_df["interpretation"] = [
        interpret_row(lc, pc)
        for lc, pc in zip(indicators_df["daily_liquidity_change_pct"], indicators_df["daily_return_pct"])
    ]
    save_indicators_csv(indicators_df)

    n_available = len(df)
    summary_49 = compute_period_summary(df, min(N_BARS, n_available))
    summary_28 = compute_period_summary(df, min(28, n_available))
    save_analysis_csv(summary_28, summary_49)

    weekly_rel = weekly_relationship(weekly_df)
    daily_rel_49 = summary_49["_rel"]

    # ------------------------------------------------------------
    # [1] Historical Data
    # ------------------------------------------------------------
    section("[1] Historical Data")
    print(f"عدد الجلسات المستلمة فعليًا من TradingView: {len(df)}")
    if symbol_info:
        print(f"الرمز المؤكد من TradingView: {symbol_info.get('pro_name', SYMBOL)} — {symbol_info.get('description', '')}")
    print("\nأول 5 جلسات:")
    print(fmt(df.head(5)[["date", "open", "high", "low", "close", "volume"]], int_cols=["volume"]).to_string(index=False))
    print("\nآخر 5 جلسات:")
    print(fmt(df.tail(5)[["date", "open", "high", "low", "close", "volume"]], int_cols=["volume"]).to_string(index=False))

    # ------------------------------------------------------------
    # [2] Daily Liquidity
    # ------------------------------------------------------------
    section("[2] Daily Liquidity")
    print("liquidity = close x volume")
    liq_cols = ["date", "close", "volume", "liquidity", "daily_liquidity_change_pct"]
    print("\nأول 5 جلسات:")
    print(fmt(df.head(5)[liq_cols], int_cols=["volume", "liquidity"]).to_string(index=False))
    print("\nآخر 5 جلسات:")
    print(fmt(df.tail(5)[liq_cols], int_cols=["volume", "liquidity"]).to_string(index=False))
    print(f"\nمتوسط السيولة اليومية ({len(df)} جلسة): {df['liquidity'].mean():,.0f}")

    # ------------------------------------------------------------
    # [3] 7-Week Analysis
    # ------------------------------------------------------------
    section("[3] 7-Week Analysis (49 جلسة = 7 أسابيع تداول)")
    week_cols = [
        "week_number", "start_date", "end_date", "first_close", "last_close",
        "max_close", "min_close", "close_change_pct", "sum_volume", "avg_volume",
        "sum_liquidity", "avg_liquidity", "liquidity_change_pct",
        "max_session_liquidity", "min_session_liquidity",
    ]
    print(fmt(weekly_df[week_cols], int_cols=["sum_volume", "avg_volume", "sum_liquidity", "avg_liquidity", "max_session_liquidity", "min_session_liquidity"]).to_string(index=False))

    # ------------------------------------------------------------
    # [4] 28-Session Analysis
    # ------------------------------------------------------------
    section("[4] 28-Session Analysis (آخر 4 أسابيع تداول)")
    last4 = weekly_df.tail(4)
    print(fmt(last4[week_cols], int_cols=["sum_volume", "avg_volume", "sum_liquidity", "avg_liquidity", "max_session_liquidity", "min_session_liquidity"]).to_string(index=False))
    print(f"\nالفترة: {summary_28['start_date']} -> {summary_28['end_date']} | عدد الجلسات: {summary_28['sessions']}")
    print(f"تغير السعر: {summary_28['price_change_pct']:.2f}%  |  تغير السيولة: {summary_28['liquidity_change_pct']:.2f}%")
    print(f"متوسط السيولة: {summary_28['avg_liquidity']:,.0f}  |  متوسط الحجم: {summary_28['avg_volume']:,.0f}")

    # ------------------------------------------------------------
    # [5] 49-Session Analysis
    # ------------------------------------------------------------
    section("[5] 49-Session Analysis (كامل الفترة)")
    print(f"الفترة: {summary_49['start_date']} -> {summary_49['end_date']} | عدد الجلسات: {summary_49['sessions']}")
    print(f"تغير السعر: {summary_49['price_change_pct']:.2f}%  |  تغير السيولة: {summary_49['liquidity_change_pct']:.2f}%")
    print(f"متوسط السيولة: {summary_49['avg_liquidity']:,.0f}  |  متوسط الحجم: {summary_49['avg_volume']:,.0f}")

    # ------------------------------------------------------------
    # [6] Mathematical Relationships
    # ------------------------------------------------------------
    section("[6] Mathematical Relationships")
    print("(أ) أسبوعي: X = weekly_liquidity_change_pct , Y = weekly_close_return_pct (تغيّر من أسبوع لآخر)")
    if weekly_rel:
        print(f"  عدد نقاط العينة n = {weekly_rel['n']}  (تنبيه: عينة صغيرة جدًا، النتائج غير موثوقة إحصائيًا)")
        print(f"  Pearson r  = {weekly_rel['pearson_r']:.4f}   (p-value = {weekly_rel['pearson_p']:.4f})")
        print(f"  Spearman r = {weekly_rel['spearman_r']:.4f}   (p-value = {weekly_rel['spearman_p']:.4f})")
        print(f"  Covariance = {weekly_rel['covariance']:.4f}")
        print(f"  Linear Regression: slope = {weekly_rel['slope']:.4f}, intercept = {weekly_rel['intercept']:.4f}, "
              f"R^2 = {weekly_rel['r_squared']:.4f}, p-value = {weekly_rel['p_value']:.4f}")
    else:
        print("  لا توجد نقاط كافية لحساب العلاقة الأسبوعية.")

    print("\n(ب) يومي (49 جلسة): X = daily_liquidity_change_pct , Y = daily_return_pct")
    if daily_rel_49:
        print(f"  عدد نقاط العينة n = {daily_rel_49['n']}")
        print(f"  Pearson r  = {daily_rel_49['pearson_r']:.4f}   (p-value = {daily_rel_49['pearson_p']:.4f})")
        print(f"  Spearman r = {daily_rel_49['spearman_r']:.4f}   (p-value = {daily_rel_49['spearman_p']:.4f})")
    else:
        print("  لا توجد نقاط كافية لحساب العلاقة اليومية.")

    # ------------------------------------------------------------
    # [7] Technical Indicators
    # ------------------------------------------------------------
    section("[7] Technical Indicators (آخر 10 جلسات)")
    ind_cols = [
        "date", "close", "sma5", "sma10", "sma20",
        "liquidity_sma5", "liquidity_sma10", "liquidity_sma20",
        "rsi14", "atr14", "roc", "momentum", "obv", "mfi", "vwap",
    ]
    print(fmt(indicators_df.tail(10)[ind_cols], int_cols=["liquidity_sma5", "liquidity_sma10", "liquidity_sma20", "obv"]).to_string(index=False))

    # ------------------------------------------------------------
    # [8] Interpretation
    # ------------------------------------------------------------
    section("[8] Interpretation")
    print("تفسير آخر 10 جلسات (سيولة = close x volume) — هذه أوصاف رقمية وليست توصية شراء/بيع:")
    interp_view = indicators_df.tail(10)[["date", "daily_return_pct", "daily_liquidity_change_pct", "interpretation"]]
    print(fmt(interp_view).to_string(index=False))

    print("\nتفسير الفترة (28 جلسة):")
    print(f"  {period_relationship_label(summary_28['_rel'])}")
    print("تفسير الفترة (49 جلسة):")
    print(f"  {period_relationship_label(summary_49['_rel'])}")
    print("\nملاحظة: هذه التفسيرات مبنية فقط على البيانات والحسابات الإحصائية أعلاه، ولا تمثل إشارة شراء أو بيع مؤكدة.")

    # ------------------------------------------------------------
    # [9] مقارنة 28 مقابل 49 جلسة
    # ------------------------------------------------------------
    section("[9] Comparison: 28 Sessions vs 49 Sessions")
    comp = pd.DataFrame(
        [
            {
                "period": "آخر 28 جلسة",
                "sessions": summary_28["sessions"],
                "start_date": summary_28["start_date"],
                "end_date": summary_28["end_date"],
                "price_change_pct": summary_28["price_change_pct"],
                "liquidity_change_pct": summary_28["liquidity_change_pct"],
                "avg_liquidity": summary_28["avg_liquidity"],
                "avg_volume": summary_28["avg_volume"],
                "pearson_r": summary_28["pearson_r"],
                "spearman_r": summary_28["spearman_r"],
                "rsi14": summary_28["rsi14_last"],
                "atr14": summary_28["atr14_last"],
                "momentum": summary_28["momentum_last"],
            },
            {
                "period": "آخر 49 جلسة",
                "sessions": summary_49["sessions"],
                "start_date": summary_49["start_date"],
                "end_date": summary_49["end_date"],
                "price_change_pct": summary_49["price_change_pct"],
                "liquidity_change_pct": summary_49["liquidity_change_pct"],
                "avg_liquidity": summary_49["avg_liquidity"],
                "avg_volume": summary_49["avg_volume"],
                "pearson_r": summary_49["pearson_r"],
                "spearman_r": summary_49["spearman_r"],
                "rsi14": summary_49["rsi14_last"],
                "atr14": summary_49["atr14_last"],
                "momentum": summary_49["momentum_last"],
            },
        ]
    )
    print(fmt(comp, int_cols=["avg_liquidity", "avg_volume"]).to_string(index=False))

    # ------------------------------------------------------------
    # ملخص نهائي: هل توجد علاقة إحصائية واضحة؟
    # ------------------------------------------------------------
    section("SUMMARY: Liquidity <-> Close Price Relationship")
    for label, s in (("آخر 28 جلسة", summary_28), ("آخر 49 جلسة", summary_49)):
        print(f"\n{label}:")
        print(f"  Pearson  r = {s['pearson_r']:.4f}   p-value = {s['pearson_p']:.4f}")
        print(f"  Spearman r = {s['spearman_r']:.4f}   p-value = {s['spearman_p']:.4f}")
        print(f"  R^2        = {s['r_squared']:.4f}   (Regression p-value = {s['reg_p_value']:.4f})")
        print(f"  {period_relationship_label(s['_rel'])}")

    print(f"\nتم حفظ الملفات التالية:")
    print(f"  - {DAILY_CSV}")
    print(f"  - {WEEKLY_CSV}")
    print(f"  - {ANALYSIS_CSV}")
    print(f"  - {INDICATORS_CSV}")


if __name__ == "__main__":
    main()
