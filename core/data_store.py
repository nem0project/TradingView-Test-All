"""
نظام تخزين وتحديث بيانات الأسهم — يدعم عددًا قابلاً للتوسع من الشركات.

البنية:
    data/symbols.csv          قائمة الشركات (symbol,name)
    data/raw/{symbol}.csv     بيانات OHLCV يومية خام لكل شركة (مصدرها TradingView فقط)
    data/processed/{symbol}.csv  بيانات مع مؤشرات السيولة المحسوبة (يُنشئها liquidity_engine)
    data/analysis/            نتائج تحليل الدورات والتوقعات لكل شركة + ملخص عام

لا يعيد هذا النظام جلب التاريخ الكامل في كل تشغيل: عند وجود بيانات محلية، يجلب فقط
نافذة أخيرة صغيرة من TradingView ويدمج الجلسات الجديدة (بالتاريخ) في الملف المحلي.
"""

import os

import pandas as pd

from historical_180 import fetch_historical_bars, TradingViewProtocolError  # noqa: F401  (نُعيد تصديره للراحة)

DATA_DIR = "data"
SYMBOLS_CSV = os.path.join(DATA_DIR, "symbols.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

# الملف القديم الذي تم جلبه فعليًا لسهم 4140 في مرحلة سابقة من المشروع — يُستخدم كبذرة
# أولية بدل إعادة الجلب من الصفر، ولا يُحذف ولا يُعدَّل.
LEGACY_4140_CSV = os.path.join(DATA_DIR, "4140_180_daily.csv")

MIN_SESSIONS = 180
KEEP_SESSIONS = 260          # هامش أعلى من 180 لدعم النوافذ المتحركة (SMA50 إلخ) دون تقليم مبكر
INITIAL_FETCH_SESSIONS = 180
UPDATE_FETCH_SESSIONS = 10   # عند التحديث اليومي: نافذة صغيرة فقط للتحقق من الجلسات الجديدة


def ensure_dirs():
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ANALYSIS_DIR):
        os.makedirs(d, exist_ok=True)


def read_symbols(include_inactive=False):
    """
    يقرأ data/symbols.csv، وينشئه ببذرة 4140 إذا لم يكن موجودًا.
    يدعم التنسيق القديم (symbol,name) والتنسيق الجديد (symbol,name,market,status,last_updated)
    الناتج عن scripts/update_symbols.py دون كسر أي كود يعتمد على عمودي symbol/name فقط.
    إن وُجد عمود status، يُستبعَد افتراضيًا أي رمز غير active (بدون حذفه من الملف نفسه).
    """
    ensure_dirs()
    if not os.path.exists(SYMBOLS_CSV):
        pd.DataFrame([{"symbol": "4140", "name": "صادرات"}]).to_csv(SYMBOLS_CSV, index=False)
    df = pd.read_csv(SYMBOLS_CSV, dtype=str)
    df["symbol"] = df["symbol"].str.strip()
    df["name"] = df["name"].str.strip()
    df = df.dropna(subset=["symbol"]).reset_index(drop=True)
    if "status" in df.columns and not include_inactive:
        df = df[df["status"].fillna("active") == "active"].reset_index(drop=True)
    return df


def full_tv_symbol(symbol):
    symbol = symbol.strip()
    return symbol if ":" in symbol else f"TADAWUL:{symbol}"


def raw_csv_path(symbol):
    return os.path.join(RAW_DIR, f"{symbol}.csv")


def load_raw(symbol):
    """يحمّل البيانات الخام المحلية لسهم، أو None إن لم توجد."""
    path = raw_csv_path(symbol)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])

    if symbol == "4140" and os.path.exists(LEGACY_4140_CSV):
        # استخدام البيانات الحالية الموجودة فعليًا بدل إعادة الجلب
        return pd.read_csv(LEGACY_4140_CSV, parse_dates=["date"])

    return None


def save_raw(symbol, df):
    ensure_dirs()
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if len(df) > KEEP_SESSIONS:
        df = df.tail(KEEP_SESSIONS).reset_index(drop=True)
    df.to_csv(raw_csv_path(symbol), index=False)
    return df


def fetch_and_update(symbol, verbose=True):
    """
    يجلب البيانات الناقصة فقط:
      - لا توجد بيانات محلية إطلاقًا -> جلب أولي كامل (INITIAL_FETCH_SESSIONS جلسة).
      - توجد بيانات محلية -> جلب نافذة أخيرة صغيرة فقط ودمج الجلسات الأحدث من تاريخ آخر جلسة محفوظة.
    المصدر الوحيد: TradingView WebSocket عبر historical_180.fetch_historical_bars (بدون أي بيانات بديلة).
    """
    ensure_dirs()
    existing = load_raw(symbol)
    full_symbol = full_tv_symbol(symbol)

    if existing is None or len(existing) == 0:
        if verbose:
            print(f"[{symbol}] لا توجد بيانات محلية — جلب أولي لـ {INITIAL_FETCH_SESSIONS} جلسة من TradingView...")
        rows, _info = fetch_historical_bars(full_symbol, INITIAL_FETCH_SESSIONS)
        new_df = pd.DataFrame(rows)
        new_df["date"] = pd.to_datetime(new_df["date"])
        combined = new_df[["date", "open", "high", "low", "close", "volume"]]
    else:
        last_date = existing["date"].max()
        if verbose:
            print(f"[{symbol}] بيانات محلية حتى {last_date.date()} ({len(existing)} جلسة) — "
                  f"جلب آخر {UPDATE_FETCH_SESSIONS} جلسة فقط للتحقق من وجود جلسات جديدة...")
        rows, _info = fetch_historical_bars(full_symbol, UPDATE_FETCH_SESSIONS)
        new_df = pd.DataFrame(rows)
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df = new_df[["date", "open", "high", "low", "close", "volume"]]
        new_only = new_df[new_df["date"] > last_date]
        if verbose:
            print(f"[{symbol}] جلسات جديدة فعليًا: {len(new_only)}")
        combined = pd.concat([existing, new_only], ignore_index=True)

    return save_raw(symbol, combined)
