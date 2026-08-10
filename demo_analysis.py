"""
Quick-Start Demo: Loading & Analyzing Pocket Option OTC High-Frequency Tick Data.

Shows how to:
1. Load high-performance Parquet tick streams.
2. Group by continuous `session_id` to avoid cross-gap backtest distortion.
3. Resample sub-second ticks into 5-second and 1-minute OHLCV candles.
4. Filter by Sigmoid Liquidity corridors (e.g. 30%–70% optimal execution band).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze Pocket Option OTC Parquet Datasets.")
    parser.add_argument(
        "--asset",
        type=str,
        default="EURUSD",
        help="Asset ticker to load (e.g., EURUSD, GBPUSD, USDJPY, BTCUSD). Defaults to EURUSD.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent / "data"
    sample_files = list((base_dir / "samples").glob("*.parquet")) if (base_dir / "samples").exists() else []
    full_files = list((base_dir / "parquet").glob("*.parquet")) if (base_dir / "parquet").exists() else []

    all_files = sample_files + full_files

    if not all_files:
        print("No Parquet files found in data/samples/ or data/parquet/.")
        print("Please run the consolidator script first: python ../scripts/consolidate_otc_dataset.py")
        return

    # Look for requested asset match
    target = args.asset.upper().replace("_OTC", "").replace("OTC", "")
    matching_files = [f for f in all_files if target in f.name.upper()]

    if matching_files:
        # Prefer sample files or larger full files
        sample_file = matching_files[0]
    else:
        sample_file = all_files[0]
        print(f"Note: Could not find exact match for '{args.asset}'. Defaulting to {sample_file.name}")

    print(f"Loading OTC tick data: {sample_file.name}...")

    # 1. Load Parquet (takes <100ms)
    df = pd.read_parquet(sample_file)
    print(f"Loaded {len(df):,} ticks with {len(df.columns)} columns:")
    print(df.head(5))
    print("\n" + "=" * 60)

    # 2. Continuous Session Breakdown
    print("\n--- Session Segmentation Summary (Gap-Aware) ---")
    session_counts = df.groupby("session_id").agg(
        start_time=("datetime_utc", "first"),
        end_time=("datetime_utc", "last"),
        tick_count=("price", "count"),
        avg_ticks_per_min=("ticks_per_min", "mean"),
        avg_liquidity=("sigmoid_liquidity", "mean"),
        avg_volatility=("volatility_score", "mean"),
    )
    print(session_counts)
    print("\n" + "=" * 60)

    # 3. Resample First Session into 5-Second OHLCV Candles
    print("\n--- Generating 5-Second OHLCV Candles from Session 1 ---")
    session_1 = df[df["session_id"] == 1].copy()
    session_1["dt"] = pd.to_datetime(session_1["timestamp"], unit="s", utc=True)
    session_1 = session_1.set_index("dt")

    candles_5s = session_1["price"].resample("5s").ohlc()
    candles_5s["tick_count"] = session_1["price"].resample("5s").count()
    candles_5s["avg_liquidity"] = session_1["sigmoid_liquidity"].resample("5s").mean()
    candles_5s["avg_volatility"] = session_1["volatility_score"].resample("5s").mean()
    candles_5s = candles_5s.dropna()

    print(candles_5s.head(10))
    print("\n" + "=" * 60)

    # 4. Liquidity & Volatility Filtering (Optimal Execution Corridor)
    optimal_ticks = df[
        (df["sigmoid_liquidity"] >= 30.0)
        & (df["sigmoid_liquidity"] <= 70.0)
        & (df["volatility_score"] >= 20.0)
        & (df["volatility_score"] <= 80.0)
    ]
    pct_optimal = (len(optimal_ticks) / len(df)) * 100.0
    print(f"\nOptimal Gate Corridor Analysis:")
    print(f"- Optimal Execution Ticks (Liq 30-70%, Vol 20-80%): {len(optimal_ticks):,} ({pct_optimal:.1f}%)")
    print(f"- Mean Tick Velocity: {df['ticks_per_min'].mean():.1f} ticks/min")
    print(f"- Mean Sigmoid Liquidity: {df['sigmoid_liquidity'].mean():.1f}%")
    print("\n" + "=" * 60)

    # 5. Market Manipulation Regime Analysis (Push & Snap and Pinning)
    if "is_manipulated" in df.columns:
        print("\n--- Market Manipulation Regime Analysis ---")
        manip_ticks = df[df["is_manipulated"] == 1]
        pct_manip = (len(manip_ticks) / len(df)) * 100.0
        print(f"Manipulated Ticks Detected: {len(manip_ticks):,} / {len(df):,} ({pct_manip:.2f}%)")
        print(f"\nManipulation Type Distribution:")
        print(df["manipulation_type"].value_counts())

        push_snaps = df[df["push_snap_severity"] > 0]
        if not push_snaps.empty:
            print(f"\n- Push & Snap Events: {len(push_snaps):,} ticks (Mean Severity: {push_snaps['push_snap_severity'].mean():.3f}, Max: {push_snaps['push_snap_severity'].max():.3f})")
        
        pinnings = df[df["pinning_severity"] > 0]
        if not pinnings.empty:
            print(f"- Pinning Clusters: {len(pinnings):,} ticks (Mean Severity: {pinnings['pinning_severity'].mean():.3f}, Max: {pinnings['pinning_severity'].max():.3f})")
        
        clean_ticks = df[df["is_manipulated"] == 0]
        print(f"\nClean Execution Ticks (Safe for Signal Execution): {len(clean_ticks):,} ({100.0 - pct_manip:.2f}%)")

    print("\nReady for quantitative strategy backtesting and ML model training!")


if __name__ == "__main__":
    main()
