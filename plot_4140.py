"""
رسم بياني لبيانات ومؤشرات TADAWUL:4140 المحفوظة في data/4140_signals.csv
(الناتج عن advanced_analysis.py).

لا يُنفَّذ تلقائيًا — شغّله يدويًا لاحقًا بعد التأكد من صحة التحليل:
    py plot_4140.py

يستخدم matplotlib فقط، ويرسم:
1. Close Price
2. Volume
3. Relative Volume (7 جلسات)
4. CloseStrength
5. LiquidityPressure
6. SmartMoneyPressure
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "data"
SIGNALS_CSV = os.path.join(DATA_DIR, "4140_signals.csv")
OUTPUT_PNG = os.path.join(DATA_DIR, "4140_advanced_chart.png")


def main():
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])

    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(13, 16))
    (ax_close, ax_volume, ax_relvol, ax_closestrength, ax_liqpressure, ax_smp) = axes

    ax_close.plot(df["date"], df["close"], color="tab:blue")
    ax_close.set_title("TADAWUL:4140 - Close Price")
    ax_close.grid(True, alpha=0.3)

    ax_volume.bar(df["date"], df["volume"], color="tab:gray")
    ax_volume.set_title("Volume")
    ax_volume.grid(True, alpha=0.3)

    ax_relvol.plot(df["date"], df["relative_volume7"], color="tab:orange")
    ax_relvol.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_relvol.axhline(1.3, color="red", linewidth=0.8, linestyle=":")
    ax_relvol.set_title("RelativeVolume7 (volume / VolumeMA7)")
    ax_relvol.grid(True, alpha=0.3)

    ax_closestrength.plot(df["date"], df["close_strength"], color="tab:green")
    ax_closestrength.axhline(0, color="black", linewidth=0.8)
    ax_closestrength.set_ylim(-1.1, 1.1)
    ax_closestrength.set_title("CloseStrength (-1 = close near low, +1 = close near high)")
    ax_closestrength.grid(True, alpha=0.3)

    ax_liqpressure.plot(df["date"], df["liquidity_pressure"], color="tab:purple")
    ax_liqpressure.axhline(0, color="black", linewidth=0.8)
    ax_liqpressure.set_title("LiquidityPressure (NormalizedMove x RelativeVolume7)")
    ax_liqpressure.grid(True, alpha=0.3)

    ax_smp.plot(df["date"], df["smart_money_pressure"], color="tab:red")
    ax_smp.axhline(0, color="black", linewidth=0.8)
    ax_smp.set_title("SmartMoneyPressure")
    ax_smp.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved chart to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
