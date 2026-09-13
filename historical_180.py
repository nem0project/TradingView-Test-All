"""
جلب آخر 180 جلسة تداول يومية فعلية لسهم TADAWUL:4140 من TradingView عبر
نفس بروتوكول chart session (resolve_symbol + create_series) المستخدم والمؤكد
سابقًا في historical_analysis.py — بدون أي بيانات وهمية أو مولّدة.

لا يغيّر أو يستورد tv_test.py، ولا يستخدم بيانات لحظية.
"""

import json
import os
import random
import string
from datetime import datetime, timezone

import pandas as pd
import websocket

SYMBOL = "TADAWUL:4140"
N_BARS = 180

DATA_DIR = "data"
DAILY_180_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")


def generate_session(prefix):
    chars = string.ascii_lowercase
    return prefix + "".join(random.choice(chars) for _ in range(12))


def send_message(ws, method, params):
    message = json.dumps({"m": method, "p": params}, separators=(",", ":"))
    ws.send(f"~m~{len(message)}~m~{message}")


def extract_messages(buffer):
    """يفصل رسائل TradingView حسب طول الرسالة (بدون regex)."""
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
            [chart_session, "symbol_1", "=" + json.dumps({"symbol": symbol, "adjustment": "splits"}, separators=(",", ":"))],
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

    rows.sort(key=lambda r: r["timestamp"])
    return rows, symbol_info


def main():
    print("=" * 60)
    print(f"جلب آخر {N_BARS} جلسة تداول فعلية لـ {SYMBOL} عبر TradingView WebSocket")
    print("=" * 60)

    try:
        rows, symbol_info = fetch_historical_bars(SYMBOL, N_BARS)
    except TradingViewProtocolError as exc:
        print("\n[خطأ] فشل جلب البيانات التاريخية من TradingView.")
        print(f"السبب: {exc}")
        print("لن يتم إنشاء أي بيانات بديلة أو وهمية.")
        raise SystemExit(1)

    received = len(rows)
    print(f"\nعدد الجلسات الحقيقية المستلمة فعليًا من TradingView: {received}")

    if received < N_BARS:
        print(
            f"[تنبيه] TradingView أعاد {received} جلسة فقط من أصل {N_BARS} المطلوبة "
            "(قد يكون هذا هو كامل العمق التاريخي المتاح للرمز عبر هذا البروتوكول). "
            "سيتم المتابعة بالعدد الفعلي المتاح فقط دون أي بيانات مصطنعة."
        )
    else:
        print(f"تم الحصول على العدد الكامل المطلوب ({N_BARS} جلسة) بنجاح.")

    if symbol_info:
        print(f"الرمز المؤكد: {symbol_info.get('pro_name', SYMBOL)} — {symbol_info.get('description', '')}")

    df = pd.DataFrame(rows)[["date", "open", "high", "low", "close", "volume"]]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DAILY_180_CSV, index=False)

    print(f"\nنطاق التواريخ: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"تم الحفظ في: {DAILY_180_CSV}")


if __name__ == "__main__":
    main()
