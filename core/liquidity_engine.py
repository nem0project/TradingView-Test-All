"""
يعمّم نفس منطق تحليل دورية السيولة الذي تم تطويره واختباره لسهم TADAWUL:4140
(liquidity_cycle_analysis.py + liquidity_forecast.py) على أي رمز سهم آخر.

لا يُعيد تعريف أي معادلة جديدة من liquidity_cycle_analysis.py — كل الدوال المستوردة
أدناه هي بالضبط نفس الدوال المستخدمة في تحليل سهم 4140، دون أي تعديل عليها.

المنطق الرسمي لتصنيف "حالة النمط" ولإخراج expected_sessions_to_liquidity /
probability_score هو تحليل رياضي وإحصائي بحت (تكرار المواقع عبر الدورات + تقارب
المواقع + قوة السيولة السابقة + انتظام الفاصل الزمني) — بدون أي نموذج ذكاء اصطناعي
أو Machine Learning (انظر analyze_liquidity_pattern أدناه).

النموذج القديم المبني على انحدار لوجستي (liquidity_forecast.forecast_next_sessions)
ما زال يُستدعى ويُحفظ كما هو (لتبقى صفحة تفاصيل الشركة تعمل دون أي تغيير)، لكنه لم
يعد المصدر الرسمي لحالة النمط / درجة الثقة المعروضة في الصفحة الرئيسية.
"""

import numpy as np
import pandas as pd

from liquidity_cycle_analysis import (  # noqa: F401
    DAYS_PER_CYCLE, LEVELS, CANDIDATE_PERIODS,
    build_liquidity_features, day_in_cycle_table, test_day_in_cycle_uniformity,
    per_cycle_top_days, highest_day_distribution_test, gap_analysis, periodicity_analysis,
)
from liquidity_forecast import forecast_next_sessions  # يبقى يعمل كما هو لصفحة التفاصيل فقط

DEFAULT_FORECAST_HORIZON = 10
MIN_ROWS_FOR_FORECAST = 30
MIN_COMPLETED_CYCLES = 3   # حد أدنى من الدورات المكتملة لاعتبار أي نمط ذا معنى
MIN_RECURRENCE = 2         # موقع يجب أن يتكرر في دورتين مكتملتين على الأقل ليُعتبر "متكررًا"

WEEKEND_WEEKDAYS = (4, 5)  # الجمعة=4, السبت=5 (بايثون: الاثنين=0 .. الأحد=6)


def analyze_symbol(raw_df, forecast_horizon=DEFAULT_FORECAST_HORIZON):
    """
    يبني نفس تحليل سهم 4140 (DayInCycle + Gap) على أي DataFrame OHLCV، ويضيف
    التوقع الرياضي/الإحصائي الجديد (expected_sessions_to_liquidity, probability_score, ...).
    راجع liquidity_cycle_analysis.py للمعادلات الأصلية الكاملة لكل ما هو مستورد.
    """
    df = build_liquidity_features(raw_df.sort_values("date").reset_index(drop=True))
    n_sessions = len(df)

    cycle_table = day_in_cycle_table(df)
    top_days_df = per_cycle_top_days(df)
    gap_summary, gaps = gap_analysis(df, "L2")

    current_cycle = int(df["cycle_number"].iloc[-1])
    current_day_in_cycle = int(df["day_in_cycle"].iloc[-1])

    last_high_positions = df.index[df["high_liquidity_L2"] == True]  # noqa: E712
    if len(last_high_positions):
        last_pos = last_high_positions[-1]
        sessions_since_last_high = int((len(df) - 1) - last_pos)
        last_high_date = df.loc[last_pos, "date"]
    else:
        sessions_since_last_high = None
        last_high_date = None

    # ------------------------------------------------------------
    # التوقع الرياضي/الإحصائي الجديد (بدون AI/ML) — المصدر الرسمي لحالة النمط
    # ------------------------------------------------------------
    pattern = analyze_liquidity_pattern(df, gap_summary=gap_summary)

    pattern_status, confidence_label, status_color = classify_pattern_status(
        pattern["expected_sessions_to_liquidity"], pattern["probability_score"], n_sessions
    )

    # ------------------------------------------------------------
    # مؤشر الفرصة المركّب liquidity_opportunity_score (0-100) — معيار الترتيب الرئيسي في اللوحة
    # ------------------------------------------------------------
    old_opportunity_score = compute_opportunity_score(
        pattern["expected_sessions_to_liquidity"], pattern["probability_score"],
        pattern["liquidity_pattern_strength"], pattern["repetition_count"], pattern["n_completed_cycles"],
    )
    old_opportunity_label, old_opportunity_color = classify_opportunity_score(old_opportunity_score)

    # ------------------------------------------------------------
    # الطبقة الجديدة: Liquidity Change (آخر 3 جلسات) + Timing Score -> final_opportunity_score
    # لا تستخدم أي مؤشر سعري إطلاقًا — فقط Volume + التوقيت الزمني الحالي.
    # نظام نقاط تدريجي (0-100 لكل عامل)، ثم مجموع موزون 60%/40% (بدون تصفير كامل عند سيولة ضعيفة).
    # ------------------------------------------------------------
    liquidity_change = calculate_liquidity_change(df)
    liquidity_score = calculate_liquidity_score(liquidity_change["liquidity_change_percent"])
    timing_score = calculate_timing_score(pattern["expected_sessions_to_liquidity"])
    final_score = compute_final_opportunity_score(liquidity_score, timing_score)
    final_label, final_color = classify_final_opportunity_score(final_score)

    # ------------------------------------------------------------
    # أفضل 3 مواعيد مستقبلية لارتفاع السيولة (Repeat Strength + Liquidity Lift)
    # ------------------------------------------------------------
    repeat_table, n_completed_cycles_rs, best_future_dates = find_best_future_dates(df)

    current_day_row = repeat_table[repeat_table["day_in_cycle"] == current_day_in_cycle]
    if len(current_day_row):
        current_repeat_strength = current_day_row.iloc[0]["repeat_strength"]
        current_liquidity_lift = current_day_row.iloc[0]["average_liquidity_lift"]
    else:
        current_repeat_strength = None
        current_liquidity_lift = None

    # ------------------------------------------------------------
    # النموذج القديم (انحدار لوجستي) — يبقى يعمل فقط لتغذية صفحة تفاصيل الشركة كما هي
    # ------------------------------------------------------------
    valid_relvol = df["relative_volume"].dropna()
    if len(valid_relvol) >= MIN_ROWS_FOR_FORECAST:
        forecast_df, best_period = forecast_next_sessions(df, forecast_horizon)
    else:
        forecast_df, best_period = pd.DataFrame(), None

    if len(forecast_df):
        hit = forecast_df[forecast_df["forecast_probability"] >= 0.5]
        legacy_sessions_ahead = int(hit.iloc[0]["session_offset"]) if len(hit) else None
        legacy_top_probability = float(forecast_df["forecast_probability"].max())
    else:
        legacy_sessions_ahead = None
        legacy_top_probability = np.nan

    return {
        "n_sessions": n_sessions,
        "df": df,
        "cycle_table": cycle_table,
        "top_days_df": top_days_df,
        "gap_summary": gap_summary,
        "gaps": gaps,
        "current_cycle": current_cycle,
        "current_day_in_cycle": current_day_in_cycle,
        "sessions_since_last_high": sessions_since_last_high,
        "last_high_date": last_high_date,

        # --- الحقول الرياضية/الإحصائية الجديدة (الرسمية) ---
        "expected_sessions_to_liquidity": pattern["expected_sessions_to_liquidity"],
        "expected_liquidity_date": pattern["expected_liquidity_date"],
        "probability_score": pattern["probability_score"],
        "historical_cycle_match": pattern["historical_cycle_match"],
        "liquidity_pattern_strength": pattern["liquidity_pattern_strength"],
        "dominant_position": pattern["dominant_position"],

        # --- الطبقة الجديدة: زخم السيولة الفعلي (نافذة 10 جلسات: 3 حالية + 7 مرجعية) + التوقيت — بدون أي مؤشر سعري ---
        "average_last_3_volume": liquidity_change["average_last_3_volume"],
        "average_volume_reference": liquidity_change["average_volume_reference"],
        "liquidity_change_percent": liquidity_change["liquidity_change_percent"],
        "liquidity_window_sessions_used": liquidity_change["liquidity_window_sessions_used"],
        "liquidity_current_sessions_used": liquidity_change["liquidity_current_sessions_used"],
        "liquidity_reference_sessions_used": liquidity_change["liquidity_reference_sessions_used"],
        "liquidity_score": liquidity_score,
        "timing_score": timing_score,
        "final_opportunity_score": final_score,
        "final_opportunity_label": final_label,
        "final_opportunity_color": final_color,

        # --- أفضل 3 مواعيد مستقبلية لارتفاع السيولة (Repeat Strength + Liquidity Lift) ---
        "repeat_strength": current_repeat_strength,
        "average_liquidity_lift": current_liquidity_lift,
        "repeat_strength_table": repeat_table,
        "n_completed_cycles_repeat": n_completed_cycles_rs,
        "best_future_dates": best_future_dates,

        # --- النظام القديم (يُحتفظ به للمقارنة فقط) ---
        "old_liquidity_opportunity_score": old_opportunity_score,
        "old_opportunity_label": old_opportunity_label,
        "old_opportunity_color": old_opportunity_color,
        "pattern_status": pattern_status,
        "confidence_label": confidence_label,
        "status_color": status_color,

        # --- حقول النموذج القديم (لصفحة التفاصيل فقط) ---
        "forecast_df": forecast_df,
        "best_period": best_period,
        "sessions_ahead": legacy_sessions_ahead,
        "top_forecast_probability": legacy_top_probability,
    }


# ============================================================
# التحليل الرياضي/الإحصائي البحت لموقع تكرار السيولة داخل الدورة (بدون AI/ML)
# ============================================================

def analyze_liquidity_pattern(df, level_col="high_liquidity_L2", gap_summary=None):
    """
    1) يحدد الجلسات التي حدثت فيها سيولة مرتفعة تاريخيًا (level_col).
    2) يقارن مواقعها (day_in_cycle) عبر الدورات الـ12 المكتملة السابقة.
    3) يحسب الموقع الأكثر تكرارًا (dominant_position) والمسافة إليه من الموقع الحالي.
    4) يخرج expected_sessions_to_liquidity + probability_score (0-100) مبنيَين على:
       - عدد مرات تكرار السيولة في نفس الموقع عبر الدورات (repetition_score)
       - تقارب مواقع حدوث السيولة حول الموقع المهيمن (tightness_score)
       - قوة السيولة السابقة (متوسط RelativeVolume عند تلك الأحداث) (strength_score)
       - مدى انتظام الفاصل الزمني بين كل أحداث السيولة المرتفعة (regularity_score)
    كل هذا حساب رياضي/إحصائي مباشر — لا يوجد أي تدريب نموذج أو تعلّم آلي هنا.
    """
    current_position = int(df["day_in_cycle"].iloc[-1])
    current_cycle = int(df["cycle_number"].iloc[-1])

    completed = df[df["cycle_number"] < current_cycle]
    n_completed_cycles = int(completed["cycle_number"].nunique())

    events = completed[completed[level_col] == True]  # noqa: E712

    if n_completed_cycles < MIN_COMPLETED_CYCLES or len(events) == 0:
        return _empty_pattern_result(n_completed_cycles)

    # كل دورة تُحتسب مرة واحدة لكل موقع (حتى لو تكررت فيه السيولة أكثر من مرة داخل نفس الدورة)
    position_cycle_counts = events.groupby("day_in_cycle")["cycle_number"].nunique()
    max_repeat = int(position_cycle_counts.max())

    if max_repeat < MIN_RECURRENCE:
        return _empty_pattern_result(n_completed_cycles)

    dominant_candidates = position_cycle_counts[position_cycle_counts == max_repeat].index.tolist()

    def _sessions_ahead(pos):
        return (pos - current_position) if pos > current_position else (pos - current_position + DAYS_PER_CYCLE)

    dominant_position = min(dominant_candidates, key=_sessions_ahead)
    expected_sessions = int(_sessions_ahead(dominant_position))

    # --- أ) قوة التكرار عند الموقع المهيمن ---
    repetition_score = max_repeat / n_completed_cycles

    # --- ب) تقارب مواقع حدوث السيولة (كلما قلّ التشتت، زادت الثقة بوجود نمط حقيقي) ---
    all_positions = events["day_in_cycle"].to_numpy(dtype=float)
    position_std = float(np.std(all_positions, ddof=0)) if len(all_positions) > 1 else 0.0
    tightness_score = max(0.0, 1 - position_std / (DAYS_PER_CYCLE / 2))

    # --- ج) قوة السيولة السابقة عند أحداث الموقع المهيمن تحديدًا ---
    dominant_events = events[events["day_in_cycle"] == dominant_position]
    avg_relvol = float(dominant_events["relative_volume"].mean())
    strength_score = min(1.0, max(0.0, (avg_relvol - 1.5) / 1.5))

    # --- د) انتظام الفاصل الزمني بين كل أحداث السيولة المرتفعة (معامل الاختلاف: std/mean) ---
    if gap_summary is None:
        gap_summary, _ = gap_analysis(df, "L2")
    if gap_summary and gap_summary.get("mean_gap", 0) and not pd.isna(gap_summary.get("std_gap", np.nan)):
        cv = gap_summary["std_gap"] / gap_summary["mean_gap"]
        regularity_score = max(0.0, 1 - min(1.0, cv))
    else:
        regularity_score = 0.0

    probability_score = int(round(100 * np.mean([repetition_score, tightness_score, strength_score, regularity_score])))
    probability_score = min(100, max(0, probability_score))

    historical_cycle_match = f"{max_repeat} من {n_completed_cycles} دورة"
    liquidity_pattern_strength = _strength_label(probability_score)

    last_date = df["date"].iloc[-1]
    expected_date = add_trading_sessions(last_date, expected_sessions)

    return {
        "current_position": current_position,
        "dominant_position": int(dominant_position),
        "expected_sessions_to_liquidity": expected_sessions,
        "expected_liquidity_date": expected_date,
        "probability_score": probability_score,
        "historical_cycle_match": historical_cycle_match,
        "liquidity_pattern_strength": liquidity_pattern_strength,
        "n_completed_cycles": n_completed_cycles,
        "repetition_count": max_repeat,
    }


def _empty_pattern_result(n_completed_cycles):
    return {
        "current_position": None,
        "dominant_position": None,
        "expected_sessions_to_liquidity": None,
        "expected_liquidity_date": None,
        "probability_score": 0,
        "historical_cycle_match": f"0 من {n_completed_cycles} دورة" if n_completed_cycles else "دورات مكتملة غير كافية بعد",
        "liquidity_pattern_strength": "غير واضح",
        "n_completed_cycles": n_completed_cycles,
        "repetition_count": 0,
    }


def _strength_label(probability_score):
    if probability_score >= 70:
        return "قوي"
    if probability_score >= 50:
        return "متوسط"
    if probability_score >= 30:
        return "ضعيف"
    return "غير واضح"


def add_trading_sessions(start_date, n_sessions):
    """يضيف n_sessions من الجلسات التداولية (يتخطى الجمعة/السبت فقط، بدون تقويم عطلات رسمية)."""
    if n_sessions is None:
        return None
    date = pd.Timestamp(start_date)
    added = 0
    while added < n_sessions:
        date += pd.Timedelta(days=1)
        if date.weekday() not in WEEKEND_WEEKDAYS:
            added += 1
    return date


# ============================================================
# أفضل 3 مواعيد مستقبلية لارتفاع السيولة (Repeat Strength + Liquidity Lift)
# رياضي/إحصائي بحت — بدون AI/ML — ويعتمد فقط على relative_volume الموجود أصلًا في النظام
# (volume مقابل متوسطه المتحرك لـ VOL_WINDOW=20 جلسة — منفصل تمامًا عن نافذة الـ10 جلسات
# (3 حالية + 7 مرجعية) المستخدمة في طبقة Liquidity Score أدناه؛ لكل منهما غرض مختلف).
# ============================================================

MIN_CYCLES_FOR_FUTURE_DATES = MIN_COMPLETED_CYCLES  # نفس الحد الأدنى المستخدم أصلًا في analyze_liquidity_pattern


def analyze_day_in_cycle_repeat_strength(df):
    """
    لكل يوم من أيام الدورة (1..DAYS_PER_CYCLE)، باستخدام الدورات المكتملة السابقة فقط
    (نفس نطاق البيانات المستخدم في analyze_liquidity_pattern، بدون تسرّب من الدورة الحالية):

    - "ارتفاع فعلي في السيولة" في جلسة ما = relative_volume > 1.0 في تلك الجلسة
      (أي حجم التداول تجاوز متوسطه المرجعي — نفس التعريف المستخدم لـ liquidity_change_percent،
      لكن مطبّقًا لحظيًا لكل جلسة تاريخية بدل متوسط آخر 3 جلسات).
    - repeat_strength = (عدد الدورات التي ارتفعت فيها السيولة في هذا اليوم / عدد الدورات المتاحة) × 100
    - average_liquidity_lift = متوسط ((relative_volume - 1) × 100) لكن فقط عبر الدورات التي حدث فيها ارتفاع فعلي
    """
    current_cycle = int(df["cycle_number"].iloc[-1])
    completed = df[df["cycle_number"] < current_cycle]
    n_completed_cycles = int(completed["cycle_number"].nunique())

    rows = []
    for day in range(1, DAYS_PER_CYCLE + 1):
        day_rows = completed[completed["day_in_cycle"] == day].dropna(subset=["relative_volume"])
        n_available = len(day_rows)
        rises = day_rows[day_rows["relative_volume"] > 1.0]
        n_rises = len(rises)

        if n_available == 0:
            repeat_strength = None
            average_liquidity_lift = None
        else:
            repeat_strength = (n_rises / n_available) * 100.0
            average_liquidity_lift = float(((rises["relative_volume"] - 1.0) * 100.0).mean()) if n_rises > 0 else 0.0

        rows.append({
            "day_in_cycle": day,
            "n_available_cycles": n_available,
            "n_rises": n_rises,
            "repeat_strength": repeat_strength,
            "average_liquidity_lift": average_liquidity_lift,
        })

    return pd.DataFrame(rows), n_completed_cycles


def compute_future_date_score(repeat_strength, average_liquidity_lift):
    """
    Future Date Score = (Repeat Strength × 60%) + (Liquidity Lift Score × 40%).
    Liquidity Lift Score = average_liquidity_lift محصورة بين 0 و100 (Clip) لمنع أي قفزة شاذة في دورة واحدة
    من التحكّم بالنتيجة — بدل ذلك نُعطي الأولوية الحقيقية للتكرار المنتظم عبر عدة دورات.
    """
    if repeat_strength is None or average_liquidity_lift is None:
        return None
    lift_score = min(100.0, max(0.0, average_liquidity_lift))
    score = (repeat_strength * 0.60) + (lift_score * 0.40)
    return float(min(100.0, max(0.0, score)))


def classify_repeat_strength(value):
    """تصنيف Repeat Strength وحدها (يُستخدم لعرض جدول أيام الدورة في صفحة التفاصيل)."""
    if value is None:
        return "بيانات غير كافية", "gray"
    if value >= 80:
        return "قوية جدًا", "fire"
    if value >= 65:
        return "قوية", "green"
    if value >= 50:
        return "متوسطة", "yellow"
    return "ضعيفة", "gray"


def classify_future_date_score(value):
    """تصنيف Future Date Score المركّب (يُستخدم لأفضل 3 مواعيد مستقبلية)."""
    if value is None:
        return "بيانات غير كافية", "gray"
    if value >= 70:
        return "فرصة تاريخية قوية", "fire"
    if value >= 55:
        return "فرصة جيدة", "green"
    if value >= 40:
        return "تحت المراقبة", "yellow"
    return "فرصة ضعيفة", "gray"


def find_best_future_dates(df, top_n=3, min_cycles=MIN_CYCLES_FOR_FUTURE_DATES):
    """
    يحلّل فقط الأيام المتبقية داخل الدورة الحالية (day_in_cycle الحالي + 1 حتى DAYS_PER_CYCLE) —
    بدون الالتفاف تلقائيًا إلى الدورة القادمة — ويختار أفضل top_n منها حسب Future Date Score.
    يُرجع: (repeat_table, n_completed_cycles, best_dates) حيث best_dates قائمة قواميس (قد تكون أقصر من top_n
    أو فارغة إذا لم تتوفر بيانات كافية أو لم تتبقَّ أيام في الدورة الحالية).
    """
    repeat_table, n_completed_cycles = analyze_day_in_cycle_repeat_strength(df)

    current_day = int(df["day_in_cycle"].iloc[-1])
    last_date = df["date"].iloc[-1]

    if n_completed_cycles < min_cycles:
        return repeat_table, n_completed_cycles, []

    remaining_days = [d for d in range(current_day + 1, DAYS_PER_CYCLE + 1)]
    if not remaining_days:
        return repeat_table, n_completed_cycles, []

    candidates = []
    for day in remaining_days:
        day_row = repeat_table[repeat_table["day_in_cycle"] == day].iloc[0]
        score = compute_future_date_score(day_row["repeat_strength"], day_row["average_liquidity_lift"])
        if score is None:
            continue
        sessions_away = day - current_day
        candidates.append({
            "cycle_day": int(day),
            "sessions_away": int(sessions_away),
            "repeat_strength": day_row["repeat_strength"],
            "average_liquidity_lift": day_row["average_liquidity_lift"],
            "future_date_score": score,
        })

    # تحقق منطقي: كل المواعيد المرشّحة يجب أن تكون مستقبلية فعلًا بالنسبة للموقع الحالي في الدورة
    candidates = [c for c in candidates if c["sessions_away"] > 0]

    candidates.sort(key=lambda c: c["future_date_score"], reverse=True)
    best_dates = candidates[:top_n]

    for rank, c in enumerate(best_dates, start=1):
        c["rank"] = rank
        c["expected_date"] = add_trading_sessions(last_date, c["sessions_away"])
        c["label"], c["color"] = classify_future_date_score(c["future_date_score"])

    return repeat_table, n_completed_cycles, best_dates


# ============================================================
# تصنيف حالة النمط (رياضي/إحصائي بحت — بدون توصية شراء/بيع)
# ============================================================

def classify_pattern_status(expected_sessions_to_liquidity, probability_score, n_sessions):
    if n_sessions < MIN_ROWS_FOR_FORECAST:
        return "بيانات غير كافية", "غير محدد", "gray"

    prob = probability_score if probability_score is not None else 0

    if expected_sessions_to_liquidity is not None and expected_sessions_to_liquidity <= 2 and prob >= 60:
        return "متوقع قريبًا", f"{prob}%", "green"
    if expected_sessions_to_liquidity is not None and expected_sessions_to_liquidity <= 5 and prob >= 45:
        return "متوقع متوسط المدى", f"{prob}%", "yellow"
    if prob >= 30:
        return "يحتاج مراقبة", f"{prob}%", "orange"
    return "لا يوجد نمط واضح", f"{prob}%", "red"


STRENGTH_LABEL_SCORE = {"قوي": 100, "متوسط": 65, "ضعيف": 35, "غير واضح": 0}

OPPORTUNITY_WEIGHTS = {
    "proximity": 0.40,   # قرب موعد السيولة المتوقع (expected_sessions_to_liquidity)
    "probability": 0.30,  # probability_score
    "strength": 0.20,     # liquidity_pattern_strength
    "match": 0.10,        # historical_cycle_match (تكرار الموقع / عدد الدورات المكتملة)
}


def compute_opportunity_score(expected_sessions_to_liquidity, probability_score,
                               liquidity_pattern_strength, repetition_count, n_completed_cycles):
    """
    liquidity_opportunity_score (0-100) — مؤشر مركّب رياضي/إحصائي بحت (بدون AI/ML) يجمع:
      40% قرب موعد السيولة المتوقعة + 30% probability_score
      + 20% liquidity_pattern_strength + 10% historical_cycle_match
    الهدف: إبراز الشركات الأقرب زمنيًا لسيولة مرتفعة قادمة وذات دلالة تاريخية كافية،
    بدل الاعتماد فقط على قوة الاحتمال أو النمط منفردَين.
    """
    if expected_sessions_to_liquidity is None:
        proximity_score = 0.0
    else:
        proximity_score = 100.0 * (DAYS_PER_CYCLE - expected_sessions_to_liquidity) / (DAYS_PER_CYCLE - 1)
        proximity_score = min(100.0, max(0.0, proximity_score))

    probability_component = float(probability_score) if probability_score is not None else 0.0
    strength_component = float(STRENGTH_LABEL_SCORE.get(liquidity_pattern_strength, 0))

    if n_completed_cycles and n_completed_cycles > 0:
        match_component = 100.0 * (repetition_count / n_completed_cycles)
    else:
        match_component = 0.0

    score = (
        OPPORTUNITY_WEIGHTS["proximity"] * proximity_score
        + OPPORTUNITY_WEIGHTS["probability"] * probability_component
        + OPPORTUNITY_WEIGHTS["strength"] * strength_component
        + OPPORTUNITY_WEIGHTS["match"] * match_component
    )
    return int(round(min(100.0, max(0.0, score))))


def classify_opportunity_score(score):
    if score >= 70:
        return "فرصة قوية جدًا", "fire"
    if score >= 50:
        return "فرصة جيدة", "orange"
    if score >= 30:
        return "تحت المراقبة", "yellow"
    return "فرصة ضعيفة أو بعيدة", "red"


# ============================================================
# الطبقة الجديدة: زخم السيولة الفعلي (آخر 3 جلسات) + التوقيت الزمني — بدون أي مؤشر سعري
# ============================================================

LIQUIDITY_WINDOW_TOTAL = 10       # إجمالي نافذة تحليل السيولة (جلسات)
LIQUIDITY_CURRENT_SESSIONS = 3    # "السيولة الحالية" = أحدث 3 جلسات
LIQUIDITY_REFERENCE_SESSIONS = LIQUIDITY_WINDOW_TOTAL - LIQUIDITY_CURRENT_SESSIONS  # = 7


def calculate_liquidity_change(df):
    """
    نافذة تحليل السيولة = 10 جلسات إجمالاً، مقسّمة صراحةً بدون أي تداخل بين المجموعتين:

      current   = أحدث 3 جلسات                      -> "السيولة الحالية" (average_last_3_volume)
      reference = الـ7 جلسات التي تسبق هذه الثلاث مباشرة -> "المتوسط المرجعي" (average_volume_reference)

    لا تدخل أي من الجلسات الثلاث الحديثة في حساب المتوسط المرجعي إطلاقًا (لا rolling(20)
    ولا rolling(7) على النافذة الكاملة — الفصل صريح بالفهرسة بعد ترتيب البيانات زمنيًا).
    لا علاقة لهذا بـ VOL_WINDOW (المستخدم فقط لحساب relative_volume في تحليل دورات السيولة،
    ولم يتغيّر).
    """
    df_sorted = df.sort_values("date") if "date" in df.columns else df  # لا نفترض ترتيب الإدخال
    latest_window = df_sorted["volume"].tail(LIQUIDITY_WINDOW_TOTAL)
    n_available = len(latest_window)

    n_current = min(LIQUIDITY_CURRENT_SESSIONS, n_available)
    current_volumes = latest_window.iloc[n_available - n_current:] if n_current else latest_window.iloc[0:0]
    reference_volumes = latest_window.iloc[: max(0, n_available - n_current)]

    average_last_3_volume = float(current_volumes.mean()) if len(current_volumes) else None
    average_volume_reference = float(reference_volumes.mean()) if len(reference_volumes) else None

    if not average_last_3_volume or not average_volume_reference or average_volume_reference <= 0:
        liquidity_change_percent = 0.0
    else:
        liquidity_change_percent = ((average_last_3_volume / average_volume_reference) - 1) * 100

    return {
        "average_last_3_volume": average_last_3_volume,
        "average_volume_reference": average_volume_reference,
        "liquidity_change_percent": liquidity_change_percent,
        "liquidity_window_sessions_used": n_available,
        "liquidity_current_sessions_used": len(current_volumes),
        "liquidity_reference_sessions_used": len(reference_volumes),
    }


def calculate_liquidity_score(liquidity_change_percent):
    """
    نقاط تدريجية (0-100) حسب نسبة تغير السيولة — بدون تصفير كامل لمجرد أن السيولة
    أقل من المتوسط المرجعي (أقل حالة تُعطى 10 نقاط، وليس صفرًا).
    """
    if liquidity_change_percent is None:
        return 0
    x = liquidity_change_percent
    if x >= 30:
        return 100
    if x >= 20:
        return 85
    if x >= 10:
        return 70
    if x >= 5:
        return 60
    if x >= 0:
        return 50
    if x >= -5:
        return 40
    if x >= -15:
        return 25
    return 10


def calculate_timing_score(expected_sessions):
    """
    نطاقات زمنية غير خطية: النافذة المثالية هي 3-5 جلسات قادمة (100 نقطة)،
    وليس بالضرورة أقرب جلسة ممكنة (جلسة واحدة تُعطى 75 فقط، أقل من 4-5 جلسات).
    """
    if expected_sessions is None:
        return 0
    if expected_sessions <= 1:
        return 75
    if expected_sessions == 2:
        return 85
    if expected_sessions == 3:
        return 95
    if expected_sessions in (4, 5):
        return 100
    if expected_sessions == 6:
        return 90
    if expected_sessions == 7:
        return 80
    if expected_sessions <= 10:
        return 60
    return 35


def compute_final_opportunity_score(liquidity_score, timing_score):
    """مجموع موزون: 60% قوة السيولة الحالية + 40% قرب التوقيت الزمني (بدون تصفير كامل)."""
    score = (liquidity_score * 0.60) + (timing_score * 0.40)
    return float(min(100.0, max(0.0, score)))


def classify_final_opportunity_score(score):
    if score >= 80:
        return "فرصة قوية جدًا", "fire"
    if score >= 65:
        return "فرصة قوية", "green"
    if score >= 50:
        return "فرصة متوسطة", "yellow"
    if score >= 35:
        return "تحتاج مراقبة", "orange"
    return "فرصة ضعيفة", "red"


def simplified_horizon_label(sessions_ahead):
    """صياغة نصية مبسّطة: خلال جلسة / جلستين / 3 / 5 / لا يوجد نمط واضح."""
    if sessions_ahead is None:
        return "لا يوجد نمط واضح حاليًا"
    if sessions_ahead == 1:
        return "متوقعة خلال جلسة واحدة"
    if sessions_ahead == 2:
        return "متوقعة خلال جلستين"
    if sessions_ahead <= 3:
        return "متوقعة خلال 3 جلسات"
    if sessions_ahead <= 5:
        return "متوقعة خلال 5 جلسات"
    return f"متوقعة بعد {sessions_ahead} جلسة"
