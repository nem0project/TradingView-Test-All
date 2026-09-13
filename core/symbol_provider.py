"""
مصدر بيانات رموز السوق السعودي (TASI) — يبني قائمة الأسهم تلقائيًا بدل الإدخال اليدوي.

المصدر: نقطة نهاية TradingView العامة (scanner.tradingview.com) — نفس مزوّد البيانات
المستخدم بالكامل في هذا المشروع لجلب بيانات OHLCV (WebSocket)، وبدون أي مفتاح API إضافي.

⚠️ ملاحظة أمانة مهمة: هذه نقطة نهاية غير موثّقة رسميًا من TradingView (تمامًا كبروتوكول
WebSocket المستخدم في historical_180.py) — تم التحقق من عملها فعليًا وقت كتابة هذا الملف،
لكن لا يوجد ضمان رسمي لاستقرارها طويل المدى. لا يوجد أي بيانات مصطنعة هنا: عند فشل الطلب
تُرفع SymbolProviderError صراحةً بدل إرجاع قائمة فارغة أو وهمية.
"""

import json
import urllib.error
import urllib.request
from datetime import date

import pandas as pd

SCANNER_URL = "https://scanner.tradingview.com/global/scan"
EXCHANGE = "TADAWUL"
REQUEST_TIMEOUT = 20


class SymbolProviderError(Exception):
    pass


def fetch_saudi_symbols():
    """
    يجلب كل الأسهم العادية (common stock) المُدرَجة على TADAWUL حاليًا عبر TradingView scanner.

    يُرجع pandas.DataFrame بالأعمدة: symbol, name, market, status, last_updated
    (status = "active" لكل سهم أعاده المصدر الآن — التمييز عن الأسهم المعلَّقة/الموقوفة
    يُدار لاحقًا في scripts/update_symbols.py عبر المقارنة مع القائمة السابقة).

    يرفع SymbolProviderError عند أي فشل اتصال أو استجابة غير متوقعة — لا بيانات بديلة أبدًا.
    """
    payload = json.dumps({
        "filter": [
            {"left": "exchange", "operation": "equal", "right": EXCHANGE},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "columns": ["name", "description"],
        "range": [0, 1000],
    }).encode("utf-8")

    request = urllib.request.Request(
        SCANNER_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SymbolProviderError(f"فشل الاتصال بمصدر رموز TASI (TradingView scanner): {exc}") from exc

    try:
        payload_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SymbolProviderError(f"استجابة غير صالحة من مصدر رموز TASI: {exc}") from exc

    rows = payload_json.get("data", [])
    if not rows:
        raise SymbolProviderError("لم يُعِد المصدر أي رموز — لن يتم إنشاء بيانات بديلة أو وهمية.")

    today = date.today().isoformat()
    records = []
    for row in rows:
        fields = row.get("d", [])
        if len(fields) < 1 or not fields[0]:
            continue
        symbol = str(fields[0]).strip()
        name = str(fields[1]).strip() if len(fields) > 1 and fields[1] else symbol
        records.append({
            "symbol": symbol,
            "name": name,
            "market": "TASI",
            "status": "active",
            "last_updated": today,
        })

    if not records:
        raise SymbolProviderError("تم الاتصال بنجاح، لكن لم يُستخرج أي رمز صالح من الاستجابة.")

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset="symbol")
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    return df
