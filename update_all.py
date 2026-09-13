"""
التحديث اليومي لكل الشركات المدرجة في data/symbols.csv:
  1. قراءة قائمة الأسهم.
  2. جلب الجلسات الجديدة فقط لكل سهم (core.data_store.fetch_and_update).
  3. تحديث ملفات data/raw/{symbol}.csv.
  4. إعادة تشغيل تحليل دورية السيولة (نفس منطق سهم 4140) لكل شركة.
  5. حفظ ملفات data/processed/ و data/analysis/.
  6. تحديث data/analysis/summary.csv الذي تقرأ منه لوحة Dashboard مباشرة.

التشغيل:
    py update_all.py
"""

import json
import os
import sys
import traceback
from datetime import datetime

import pandas as pd

# يضمن طباعة UTF-8 بغض النظر عن ترميز الطرفية المستدعية (Task Scheduler / GitHub Actions / إلخ).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from core.data_store import read_symbols, fetch_and_update, PROCESSED_DIR, ANALYSIS_DIR, ensure_dirs  # noqa: E402
from core.liquidity_engine import analyze_symbol, simplified_horizon_label  # noqa: E402

SUMMARY_CSV = os.path.join(ANALYSIS_DIR, "summary.csv")
STATUS_FILE = os.path.join(ANALYSIS_DIR, "last_update_status.json")


def liquidity_level_label(relative_volume):
    if pd.isna(relative_volume):
        return "غير محدد"
    if relative_volume > 2.0:
        return "استثنائية"
    if relative_volume > 1.5:
        return "مرتفعة"
    if relative_volume > 1.0:
        return "فوق المتوسط"
    return "طبيعية"


def flatten_best_future_dates(best_future_dates, top_n=3):
    """
    يحوّل قائمة أفضل المواعيد (قد تكون أقصر من top_n أو فارغة) إلى أعمدة مسطّحة
    best_date_N / best_cycle_day_N / ... — بقيم فارغة و"بيانات غير كافية" للمواعيد غير المتاحة
    (بدل اختلاق Score مضلل)، تمامًا كما طُلب.
    """
    fields = {}
    for i in range(1, top_n + 1):
        if i <= len(best_future_dates):
            d = best_future_dates[i - 1]
            fields[f"best_date_{i}"] = d["expected_date"]
            fields[f"best_cycle_day_{i}"] = d["cycle_day"]
            fields[f"best_sessions_away_{i}"] = d["sessions_away"]
            fields[f"best_repeat_strength_{i}"] = d["repeat_strength"]
            fields[f"best_liquidity_lift_{i}"] = d["average_liquidity_lift"]
            fields[f"best_future_score_{i}"] = d["future_date_score"]
            fields[f"best_label_{i}"] = d["label"]
        else:
            fields[f"best_date_{i}"] = None
            fields[f"best_cycle_day_{i}"] = None
            fields[f"best_sessions_away_{i}"] = None
            fields[f"best_repeat_strength_{i}"] = None
            fields[f"best_liquidity_lift_{i}"] = None
            fields[f"best_future_score_{i}"] = None
            fields[f"best_label_{i}"] = "بيانات غير كافية"
    return fields


def build_summary_row(symbol, name, result):
    df = result["df"]
    last_row = df.iloc[-1]

    if len(df) > 1:
        prev_close = df["close"].iloc[-2]
        daily_change_pct = (last_row["close"] - prev_close) / prev_close * 100
    else:
        daily_change_pct = float("nan")

    liquidity = df["close"] * df["volume"]
    current_liquidity = float(liquidity.iloc[-1])
    avg_liquidity = float(liquidity.tail(60).mean())

    return {
        "symbol": symbol,
        "name": name,
        "last_date": last_row["date"],
        "last_close": float(last_row["close"]),
        "daily_change_pct": daily_change_pct,
        "last_volume": float(last_row["volume"]),
        "current_liquidity": current_liquidity,
        "avg_liquidity_60": avg_liquidity,
        "liquidity_level": liquidity_level_label(last_row.get("relative_volume", float("nan"))),
        "last_high_liquidity_date": result["last_high_date"],
        "sessions_since_last_high": result["sessions_since_last_high"],
        "current_cycle": result["current_cycle"],
        "current_day_in_cycle": result["current_day_in_cycle"],

        # --- الحقول الرياضية/الإحصائية الجديدة (بدون AI/ML) — المصدر الرسمي لحالة النمط ---
        "expected_sessions_to_liquidity": result["expected_sessions_to_liquidity"],
        "expected_liquidity_date": result["expected_liquidity_date"],
        "probability_score": result["probability_score"],
        "historical_cycle_match": result["historical_cycle_match"],
        "liquidity_pattern_strength": result["liquidity_pattern_strength"],
        "pattern_status": result["pattern_status"],
        "confidence_label": result["confidence_label"],
        "status_color": result["status_color"],

        # --- الطبقة الجديدة: زخم السيولة الفعلي (نافذة 10 جلسات: 3 حالية + 7 مرجعية) + التوقيت ---
        "average_last_3_volume": result["average_last_3_volume"],
        "average_volume_reference": result["average_volume_reference"],
        "liquidity_change_percent": result["liquidity_change_percent"],
        "liquidity_window_sessions_used": result["liquidity_window_sessions_used"],
        "liquidity_current_sessions_used": result["liquidity_current_sessions_used"],
        "liquidity_reference_sessions_used": result["liquidity_reference_sessions_used"],
        "liquidity_score": result["liquidity_score"],
        "timing_score": result["timing_score"],
        "final_opportunity_score": result["final_opportunity_score"],
        "final_opportunity_label": result["final_opportunity_label"],
        "final_opportunity_color": result["final_opportunity_color"],

        # --- أفضل 3 مواعيد مستقبلية لارتفاع السيولة (Repeat Strength + Liquidity Lift) ---
        "repeat_strength": result["repeat_strength"],
        "average_liquidity_lift": result["average_liquidity_lift"],
        **flatten_best_future_dates(result["best_future_dates"]),

        # --- مؤشر الفرصة المركّب القديم (يُحتفظ به مؤقتًا للمقارنة فقط، لم يعد معيار الترتيب) ---
        "old_liquidity_opportunity_score": result["old_liquidity_opportunity_score"],
        "old_opportunity_label": result["old_opportunity_label"],
        "old_opportunity_color": result["old_opportunity_color"],

        # --- حقول النموذج القديم (لصفحة تفاصيل الشركة فقط، غير مستخدمة في ترتيب/شارة الصفحة الرئيسية) ---
        "sessions_ahead_forecast": result["sessions_ahead"],
        "forecast_horizon_label": simplified_horizon_label(result["sessions_ahead"]),
        "top_forecast_probability": result["top_forecast_probability"],

        "n_sessions": result["n_sessions"],
    }


def write_status_file(started_at, finished_at, total, succeeded_symbols, symbol_statuses):
    """
    ملف حالة صغير (JSON) لآخر تشغيل — يقرأه Dashboard/Home.py لعرض
    "آخر تحديث ناجح" دون الحاجة لتحليل ملف الـ log النصي. إضافي بحت، لا يمسّ أي معادلة تحليل.

    symbol_statuses: dict {symbol: status_code} حيث status_code واحد من:
      SUCCESS / NO_DATA / API_ERROR / ANALYSIS_ERROR
    """
    failed_symbols = [s for s, st in symbol_statuses.items() if st != "SUCCESS"]

    if not failed_symbols:
        overall_status = "success"
    elif succeeded_symbols:
        overall_status = "partial"
    else:
        overall_status = "failed"

    status = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
        "total_symbols": total,
        "succeeded_count": len(succeeded_symbols),
        "failed_count": len(failed_symbols),
        "succeeded_symbols": succeeded_symbols,
        "failed_symbols": failed_symbols,
        "symbol_statuses": symbol_statuses,
        "overall_status": overall_status,
    }
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    return status


def main():
    started_at = datetime.now()
    print(f"بدء التحديث: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_dirs()
    symbols_df = read_symbols()
    print(f"عدد الشركات في symbols.csv: {len(symbols_df)}")

    summary_rows = []
    succeeded_symbols = []
    symbol_statuses = {}   # {symbol: SUCCESS / NO_DATA / API_ERROR / ANALYSIS_ERROR}

    for _, row in symbols_df.iterrows():
        symbol, name = row["symbol"], row["name"]
        print(f"\n=== {symbol} ({name}) ===")

        try:
            raw_df = fetch_and_update(symbol)
        except Exception as exc:  # نتابع بقية الأسهم حتى لو فشل واحد
            print(f"[{symbol}] فشل التحديث (API_ERROR): {exc}")
            traceback.print_exc(file=sys.stdout)
            symbol_statuses[symbol] = "API_ERROR"
            continue

        if len(raw_df) < 30:
            print(f"[{symbol}] بيانات غير كافية للتحليل (NO_DATA — {len(raw_df)} جلسة فقط) — تخطي التحليل.")
            symbol_statuses[symbol] = "NO_DATA"
            continue

        try:
            result = analyze_symbol(raw_df)
        except Exception as exc:
            print(f"[{symbol}] فشل التحليل (ANALYSIS_ERROR): {exc}")
            traceback.print_exc(file=sys.stdout)
            symbol_statuses[symbol] = "ANALYSIS_ERROR"
            continue

        result["df"].to_csv(os.path.join(PROCESSED_DIR, f"{symbol}.csv"), index=False)
        result["cycle_table"].to_csv(os.path.join(ANALYSIS_DIR, f"{symbol}_cycle_table.csv"), index=False)
        result["top_days_df"].to_csv(os.path.join(ANALYSIS_DIR, f"{symbol}_top_days.csv"), index=False)
        result["repeat_strength_table"].to_csv(os.path.join(ANALYSIS_DIR, f"{symbol}_repeat_strength.csv"), index=False)
        if len(result["forecast_df"]):
            result["forecast_df"].to_csv(os.path.join(ANALYSIS_DIR, f"{symbol}_forecast.csv"), index=False)

        summary_rows.append(build_summary_row(symbol, name, result))
        succeeded_symbols.append(symbol)
        symbol_statuses[symbol] = "SUCCESS"

        print(f"[{symbol}] SUCCESS — {result['n_sessions']} جلسة | الدورة {result['current_cycle']} / اليوم {result['current_day_in_cycle']} "
              f"| منذ آخر سيولة مرتفعة: {result['sessions_since_last_high']} جلسة | الحالة: {result['pattern_status']}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"\n✅ تم تحديث {len(summary_df)} شركة بنجاح. الملخص محفوظ في: {SUMMARY_CSV}")
    else:
        print("\n⚠️ لم يتم تحديث أي شركة بنجاح.")

    print("\nملخص الحالة لكل رمز:")
    for sym, st in symbol_statuses.items():
        print(f"  {sym:<10} {st}")

    finished_at = datetime.now()
    status = write_status_file(started_at, finished_at, len(symbols_df), succeeded_symbols, symbol_statuses)
    print(f"\nانتهى التحديث: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"(المدة: {status['duration_seconds']:.0f} ثانية)  الحالة العامة: {status['overall_status']}")


if __name__ == "__main__":
    main()
