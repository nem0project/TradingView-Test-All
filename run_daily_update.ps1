<#
.SYNOPSIS
    التشغيل اليومي التلقائي لـ update_all.py — يُستدعى من Windows Task Scheduler.

.DESCRIPTION
    - ينتقل إلى مجلد المشروع الصحيح دائمًا (بغض النظر عن أين يُستدعى منه).
    - يشغّل: py scripts/update_symbols.py ثم py update_all.py (نفس نسخة Python 3.13 التي
      تعمل بها كل المكتبات) — نفس تسلسل GitHub Actions تمامًا (.github/workflows/*.yml).
    - يسجّل وقت البداية والنهاية وكامل مخرجات التشغيل (نجاح/فشل كل شركة) بترميز UTF-8 في:
        data\logs\daily_update.log
    - لا يعدّل أي منطق تحليل أو معادلات — فقط يستدعي السكربتات الحالية كما هي.
#>

$ErrorActionPreference = "Continue"

# يمنع تشويه النصوص العربية عند التقاط مخرجات py update_all.py (مشكلة ترميز شائعة في
# PowerShell 5.1 عند إعادة توجيه مخرجات UTF-8 من عملية خارجية).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectDir = "E:\TickerChartLive\TradingView-Test AII"
$LogDir = Join-Path $ProjectDir "data\logs"
$LogFile = Join-Path $LogDir "daily_update.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Set-Location $ProjectDir

# يجبر بايثون على كتابة UTF-8 بدل ترميز الطرفية الافتراضي عند التشغيل غير التفاعلي (Task Scheduler).
$env:PYTHONIOENCODING = "utf-8"

$startTime = Get-Date

Add-Content -Path $LogFile -Encoding utf8 -Value ""
Add-Content -Path $LogFile -Encoding utf8 -Value "=================================================================="
Add-Content -Path $LogFile -Encoding utf8 -Value "Run started : $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Add-Content -Path $LogFile -Encoding utf8 -Value "Project dir : $ProjectDir"
Add-Content -Path $LogFile -Encoding utf8 -Value "=================================================================="

$exitCode = 0
try {
    $symbolsOutput = & py scripts/update_symbols.py 2>&1
    $symbolsOutput | ForEach-Object { Add-Content -Path $LogFile -Encoding utf8 -Value $_ }
    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $LogFile -Encoding utf8 -Value "تنبيه: فشل تحديث قائمة الرموز — سيتابع التحديث بالقائمة الحالية كما هي."
    }

    $output = & py update_all.py 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Add-Content -Path $LogFile -Encoding utf8 -Value $_ }
}
catch {
    Add-Content -Path $LogFile -Encoding utf8 -Value "FATAL ERROR launching update pipeline: $_"
    $exitCode = 1
}

$endTime = Get-Date
$duration = $endTime - $startTime

Add-Content -Path $LogFile -Encoding utf8 -Value "------------------------------------------------------------------"
Add-Content -Path $LogFile -Encoding utf8 -Value "Run finished: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))  Duration: $($duration.ToString('hh\:mm\:ss'))  ExitCode: $exitCode"
Add-Content -Path $LogFile -Encoding utf8 -Value "=================================================================="

exit $exitCode
