import json
import random
import string
import websocket


def generate_session():
    chars = string.ascii_lowercase
    return "qs_" + "".join(random.choice(chars) for _ in range(12))


def send_message(ws, method, params):
    message = json.dumps(
        {"m": method, "p": params},
        separators=(",", ":")
    )

    ws.send(f"~m~{len(message)}~m~{message}")


def extract_messages(buffer):
    """
    يفصل رسائل TradingView المؤطّرة بالصيغة ~m~<length>~m~<payload>
    اعتمادًا على طول كل رسالة (بدون regex).
    يُرجع قائمة الرسائل المكتملة، والجزء المتبقي غير المكتمل من buffer
    (يُعاد استخدامه في المرة القادمة في حال وصلت رسالة مجزّأة عبر أكثر من إطار).
    """
    messages = []
    pos = 0
    size = len(buffer)

    while buffer.startswith("~m~", pos):
        len_start = pos + 3
        len_end = buffer.find("~m~", len_start)
        if len_end == -1:
            break  # بادئة الطول لم تصل كاملة بعد

        length_str = buffer[len_start:len_end]
        if not length_str.isdigit():
            break

        length = int(length_str)
        msg_start = len_end + 3
        msg_end = msg_start + length

        if msg_end > size:
            break  # الرسالة غير مكتملة بعد، ننتظر المزيد من البيانات

        messages.append(buffer[msg_start:msg_end])
        pos = msg_end

    return messages, buffer[pos:]


symbol = "TADAWUL:4140"
session = generate_session()

FIELDS = [
    "lp",
    "bid",
    "ask",
    "ch",
    "chp",
    "open_price",
    "high_price",
    "low_price",
    "prev_close_price",
    "volume",
]

latest_values = {field: None for field in FIELDS}

print("Connecting to TradingView...")

ws = websocket.create_connection(
    "wss://data.tradingview.com/socket.io/websocket",
    origin="https://data.tradingview.com",
    timeout=30
)

print("Connected ✅")

# إنشاء جلسة Quotes
send_message(
    ws,
    "quote_create_session",
    [session, ""]
)

# الحقول التي نريدها
send_message(
    ws,
    "quote_set_fields",
    [
        session,
        "lp",
        "ch",
        "chp",
        "volume",
        "bid",
        "ask",
        "high_price",
        "low_price",
        "open_price",
        "prev_close_price",
    ]
)

# الاشتراك في الرمز
send_message(
    ws,
    "quote_add_symbols",
    [
        session,
        symbol
    ]
)

print(f"Subscribed to {symbol}")
print("Waiting for TradingView messages...\n")

buffer = ""

try:

    while True:

        buffer += ws.recv()

        messages, buffer = extract_messages(buffer)

        for message in messages:

            # التعامل مع ping الخاص بـ TradingView
            if message.startswith("~h~"):
                ws.send(f"~m~{len(message)}~m~{message}")
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            m_type = data.get("m")

            # تجاهل الأنواع غير المطلوبة (protocol_error, critical_error, quote_completed, ...)
            if m_type != "qsd":
                continue

            payload = data.get("p", [])
            if len(payload) < 2:
                continue

            quote = payload[1]
            if quote.get("n") != symbol:
                continue

            values = quote.get("v", {})
            latest_values.update({k: v for k, v in values.items() if k in latest_values})

            print("\n🎯 QUOTE DATA RECEIVED!")
            print(symbol)
            print("=" * 40)
            for field in FIELDS:
                print(f"{field}: {latest_values[field]}")
            print("=" * 40)

except KeyboardInterrupt:

    print("\nStopped by user.")

finally:

    ws.close()
    print("Connection closed.")
