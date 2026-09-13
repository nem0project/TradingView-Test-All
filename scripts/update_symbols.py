"""
يحدّث data/symbols.csv تلقائيًا من مصدر رموز TASI الحقيقي (core.symbol_provider.fetch_saudi_symbols)
بدل الإدخال اليدوي.

قواعد الدمج:
  - أي رمز جديد يعيده المصدر ولم يكن موجودًا سابقًا -> يُضاف باسمه كما أعاده المصدر (إنجليزي).
  - أي رمز موجود سابقًا في القائمة -> يبقى اسمه المُدخَل يدويًا كما هو (لا يُستبدل بالاسم
    الإنجليزي من المصدر) حتى لا تُفقد الأسماء العربية المُدخلة يدويًا للشركات الحالية.
  - أي رمز كان موجودًا سابقًا ولم يعد ضمن نتيجة المصدر الآن -> لا يُحذف، بل يُعلَّم status=inactive
    (قد يكون تعليق تداول مؤقت أو اختلاف طفيف في فلترة المصدر، وليس بالضرورة شطبًا).

عند فشل الاتصال بالمصدر: لا يُعدَّل data/symbols.csv الحالي إطلاقًا (لا بيانات بديلة أو وهمية).

التشغيل:
    py scripts/update_symbols.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# يضمن طباعة UTF-8 بغض النظر عن ترميز الطرفية المستدعية (Task Scheduler / GitHub Actions / إلخ).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import pandas as pd  # noqa: E402

from core.data_store import SYMBOLS_CSV, ensure_dirs  # noqa: E402
from core.symbol_provider import fetch_saudi_symbols, SymbolProviderError  # noqa: E402


def load_existing():
    if not os.path.exists(SYMBOLS_CSV):
        return pd.DataFrame(columns=["symbol", "name", "market", "status", "last_updated"])
    df = pd.read_csv(SYMBOLS_CSV, dtype=str)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    return df


def main():
    ensure_dirs()
    print("جلب قائمة أسهم TASI الحالية من TradingView...")

    try:
        fresh_df = fetch_saudi_symbols()
    except SymbolProviderError as exc:
        print(f"[خطأ] تعذّر تحديث قائمة الرموز: {exc}")
        print("لن يتم تعديل data/symbols.csv الحالي — تبقى القائمة القديمة سارية كما هي.")
        raise SystemExit(1)

    print(f"تم جلب {len(fresh_df)} سهمًا فعليًا من TASI عبر TradingView.")

    existing_df = load_existing()
    existing_names = dict(zip(existing_df["symbol"], existing_df.get("name", existing_df["symbol"])))
    existing_symbols = set(existing_df["symbol"]) if len(existing_df) else set()
    fresh_symbols = set(fresh_df["symbol"])

    # الحفاظ على الأسماء المُدخلة يدويًا سابقًا (مثل الأسماء العربية) بدل استبدالها بالاسم الإنجليزي
    fresh_df["name"] = fresh_df.apply(
        lambda r: existing_names.get(r["symbol"], r["name"]), axis=1
    )

    missing = existing_symbols - fresh_symbols
    new_symbols = fresh_symbols - existing_symbols
    if missing:
        print(f"تنبيه: {len(missing)} رمزًا من القائمة الحالية غائب عن نتيجة المصدر الآن "
              f"(سيُعلَّم inactive بدل حذفه): {sorted(missing)}")
    print(f"رموز جديدة اكتُشفت: {len(new_symbols)}")

    merged_rows = fresh_df.to_dict("records")
    for _, row in existing_df.iterrows():
        if row["symbol"] in missing:
            merged_rows.append({
                "symbol": row["symbol"],
                "name": row.get("name", row["symbol"]),
                "market": row.get("market", "TASI"),
                "status": "inactive",
                "last_updated": row.get("last_updated", ""),
            })

    result_df = (
        pd.DataFrame(merged_rows)
        .drop_duplicates(subset="symbol")
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    result_df.to_csv(SYMBOLS_CSV, index=False)

    n_active = int((result_df["status"] == "active").sum())
    n_inactive = int((result_df["status"] == "inactive").sum())
    print(f"✅ تم تحديث {SYMBOLS_CSV} — {len(result_df)} رمزًا إجمالًا ({n_active} نشط، {n_inactive} غير نشط).")


if __name__ == "__main__":
    main()
