# لوحة سيولة أسهم السوق السعودي (TASI) — Liquidity Cycle Dashboard

لوحة تحليل رياضي/إحصائي (بدون أي ذكاء اصطناعي) لدورية السيولة لكل الأسهم المُدرَجة في
السوق السعودي الرئيسي، مبنية على بيانات حقيقية من TradingView فقط.

**هذا تحليل نمطي إحصائي بحت — لا يمثّل بأي شكل توصية شراء أو بيع.**

---

## 1) طريقة التشغيل محليًا

### المتطلبات
- Python 3.13 (أو أي إصدار حديث متوافق).
- الحزم في `requirements.txt`.

```bash
pip install -r requirements.txt
```

### تشغيل اللوحة
```bash
py -m streamlit run dashboard/Home.py
```
تُفتح تلقائيًا على `http://localhost:8501`.

---

## 2) طريقة تحديث البيانات

### تحديث قائمة الأسهم (تلقائي من TradingView، بدون إدخال يدوي)
```bash
py scripts/update_symbols.py
```
يجلب كل الأسهم العادية المُدرَجة حاليًا على TADAWUL ويحدّث `data/symbols.csv`
(يحافظ على أي اسم عربي أدخلته يدويًا للأسهم الموجودة سابقًا، ولا يحذف أي رمز — فقط
يُعلَّم الرمز الغائب عن نتيجة المصدر بـ `status=inactive`).

### تحديث بيانات وتحليل كل الأسهم
```bash
py update_all.py
```
يجلب فقط الجلسات الجديدة لكل سهم (لا يعيد تحميل 180 جلسة كاملة إلا لسهم جديد لم يُجلب من
قبل)، يشغّل كل التحليلات الحالية، ويحدّث `data/analysis/summary.csv` الذي تقرأه اللوحة.

### تلقائيًا محليًا (Windows Task Scheduler)
مهمة `TickerChartLive_DailyUpdate` (أُنشئت مسبقًا) تُشغّل `run_daily_update.ps1` يوميًا
الساعة 16:00 بتوقيت الجهاز، وتسجّل كل تشغيل في `data/logs/daily_update.log`.

```powershell
Get-ScheduledTask -TaskName "TickerChartLive_DailyUpdate"   # عرض المهمة
Start-ScheduledTask -TaskName "TickerChartLive_DailyUpdate" # تشغيل فوري يدوي
```

### تلقائيًا على السحابة (GitHub Actions — لا يعتمد على أي جهاز محلي)
انظر القسم 5 أدناه.

---

## 3) بنية المشروع (مختصر)

```
core/
  symbol_provider.py      # fetch_saudi_symbols() — مصدر قائمة أسهم TASI (TradingView scanner)
  data_store.py           # تخزين/تحديث بيانات كل سهم (raw/processed/analysis)
  liquidity_engine.py     # يعمّم تحليل دورية السيولة (المطوَّر أصلًا على 4140) على أي رمز
scripts/
  update_symbols.py       # يحدّث data/symbols.csv تلقائيًا
update_all.py              # يحدّث ويحلّل كل الأسهم في data/symbols.csv
dashboard/
  Home.py                  # الصفحة الرئيسية (Screener لكل أسهم TASI)
  pages/1_تفاصيل_الشركة.py  # صفحة تفاصيل أي سهم
data/
  symbols.csv               # قائمة الأسهم (symbol,name,market,status,last_updated)
  raw/{symbol}.csv           # بيانات OHLCV خام لكل سهم
  processed/{symbol}.csv      # بيانات مع مؤشرات السيولة المحسوبة
  analysis/                   # نتائج كل سهم + summary.csv (المصدر المباشر للوحة)
  logs/daily_update.log        # سجل التحديث المحلي (لا يُرفع إلى GitHub)
.github/workflows/
  update-saudi-symbols.yml   # تحديث قائمة الأسهم يوميًا على GitHub Actions
  update-market-data.yml     # تحديث وتحليل كل الأسهم يوميًا على GitHub Actions
```

الملفات الفردية القديمة (`historical_180.py`, `advanced_analysis_180.py`,
`contextual_liquidity_model.py`, `liquidity_forecast.py`, إلخ) هي أمثلة بحث تفصيلية على
سهم 4140 محفوظة كما هي — منطقها الأساسي مستورَد فعليًا داخل `core/liquidity_engine.py`
ويعمل على أي رمز عبر خط الأنابيب أعلاه، فلا حاجة لتعديلها.

---

## 4) المتغيرات المطلوبة (Environment Variables)

**لا توجد أي متغيرات بيئة أو مفاتيح API مطلوبة حاليًا.** كل البيانات تُجلب من TradingView
(WebSocket لبيانات الأسعار، ونقطة نهاية Scanner العامة لقائمة الرموز) بدون أي اشتراك.

`.env.example` قالب جاهز فقط لأي مصدر بيانات مدفوع تضيفه مستقبلًا — انسخه إلى `.env`
عند الحاجة (لا يُرفع `.env` إلى GitHub أبدًا).

### طريقة إضافة API Key مستقبلًا
1. أضف المتغير في `.env` (محليًا) وفي `.env.example` (بدون القيمة الفعلية، للتوثيق فقط).
2. في الكود: اقرأه عبر `os.environ["اسم_المتغير"]`.
3. على Streamlit Community Cloud: أضِفه من **Settings → Secrets** بصيغة TOML:
   ```toml
   TASI_DATA_API_KEY = "..."
   ```
4. في GitHub Actions: أضِفه كـ Repository Secret (**Settings → Secrets and variables →
   Actions**) واستخدمه في ملف الـ workflow عبر `${{ secrets.TASI_DATA_API_KEY }}`.

---

## 5) النشر على السحابة (Streamlit Community Cloud)

الاستضافة المختارة: **Streamlit Community Cloud** (مجانية بالكامل، مبنية خصيصًا لتطبيقات
Streamlit). الخطوات (تحتاج حساب GitHub مربوط، لا يمكن أتمتتها بالكامل):

1. تأكد أن الكود مرفوع على GitHub (فرع `main`).
2. افتح [share.streamlit.io](https://share.streamlit.io) وسجّل الدخول بحساب GitHub.
3. New app → اختر المستودع → الفرع `main` → مسار التطبيق: `dashboard/Home.py`.
4. Deploy. سيُنشئ رابطًا دائمًا مثل `https://<اسم-التطبيق>.streamlit.app`.
5. أي `git push` جديد إلى `main` (بما فيه الدفعات التلقائية من GitHub Actions) يُعيد نشر
   اللوحة تلقائيًا ببيانات محدَّثة — بدون أي جهاز محلي.

### لماذا GitHub Actions ضروري أيضًا (وليس Streamlit Cloud وحده)؟
Streamlit Community Cloud يستضيف الواجهة فقط، وليس به Cron لتشغيل `update_all.py` يوميًا.
لذلك: **GitHub Actions يحدّث البيانات ويرفعها → Streamlit Cloud يعيد النشر تلقائيًا عند كل
رفع جديد.** هذا يحقق "تحديث + عرض يعملان 24/7" دون أي اعتماد على جهازك الشخصي.

---

## 6) طريقة تغيير وقت التحديث

### محليًا (Windows Task Scheduler)
```powershell
Set-ScheduledTask -TaskName "TickerChartLive_DailyUpdate" `
  -Trigger (New-ScheduledTaskTrigger -Daily -At 17:30)
```

### على السحابة (GitHub Actions)
عدّل قيمة `cron` في:
- `.github/workflows/update-saudi-symbols.yml`
- `.github/workflows/update-market-data.yml`

الصيغة: `"دقيقة ساعة * * يوم_الأسبوع"` بتوقيت **UTC** (الرياض = UTC+3 طوال السنة، فاطرح 3
ساعات من وقت الرياض المطلوب). أيام تداول تاسي (أحد-خميس) = `0-4` في حقل يوم الأسبوع
(0=الأحد). بعد التعديل: `git commit` و`git push`.

---

## 7) ملاحظات أمانة مهمة

- **مصدر بيانات الأسعار (WebSocket) ومصدر قائمة الرموز (Scanner) كلاهما نقطتا نهاية عامتان
  غير موثّقتين رسميًا من TradingView.** يعملان فعليًا وتم التحقق منهما، لكن بدون ضمان SLA
  رسمي — قد يتغيّرا مستقبلًا.
- عند فشل أي منهما: النظام **لا يتوقف بالكامل ولا يستخدم بيانات وهمية** — يسجّل الخطأ لكل
  سهم على حدة (`SUCCESS` / `NO_DATA` / `API_ERROR` / `ANALYSIS_ERROR` في
  `data/analysis/last_update_status.json`) وينتقل للسهم التالي، وتعرض اللوحة آخر بيانات
  صحيحة محفوظة مع حالة مصدر البيانات (`LIVE` / `CACHED` / `ERROR`).
