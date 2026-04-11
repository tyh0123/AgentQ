#!/usr/bin/env python3
"""
Plot time series and FFT from existing CSV files.

Usage:
    python plot_timeseries.py --csv timeseries_cpw_input.csv timeseries_junction.csv
    python plot_timeseries.py --csv timeseries_cpw_input.csv --npoints 500
    python plot_timeseries.py --csv timeseries_cpw_input.csv --fft_max 15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot time series and FFT from CSV files")
    parser.add_argument("--csv", type=str, nargs="+", required=True,
                        help="CSV file(s) to plot")
    parser.add_argument("--field", type=str, default=None,
                        help="Field column name (auto-detected if not given)")
    parser.add_argument("--npoints", type=int, default=-1,
                        help="Number of points to use (-1 = all)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N points (e.g. skip transient)")
    parser.add_argument("--fft_max", type=float, default=20.0,
                        help="Max frequency for FFT plot in GHz (default 20)")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load all CSVs
    datasets = {}
    for csv_path in args.csv:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        name = name.replace("timeseries_", "")
        df = pd.read_csv(csv_path)

        # Auto-detect field column
        if args.field:
            field_col = args.field
        else:
            skip_cols = {"step", "time_s", "time_ns"}
            field_cols = [c for c in df.columns if c not in skip_cols]
            field_col = field_cols[0] if field_cols else None

        if field_col is None or field_col not in df.columns:
            print(f"Error: field column '{field_col}' not found in {csv_path}")
            print(f"  Available columns: {list(df.columns)}")
            continue

        # Apply skip and npoints
        df = df.iloc[args.skip:]
        if args.npoints > 0:
            df = df.iloc[:args.npoints]
        df = df.reset_index(drop=True)

        datasets[name] = {"df": df, "field": field_col}
        print(f"Loaded '{name}': {len(df)} points, field='{field_col}', "
              f"time=[{df['time_ns'].iloc[0]:.4f}, {df['time_ns'].iloc[-1]:.4f}] ns")

    if not datasets:
        print("No valid datasets loaded!")
        exit(1)

    n_plots = len(datasets)

    # ── Time series plot ──
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, datasets.items()):
        df = data["df"]
        field_col = data["field"]
        ax.plot(df["time_ns"].values, df[field_col].values, linewidth=0.5)
        ax.set_ylabel(f'{field_col} (V/m)')
        ax.set_title(f'{name} ({len(df)} points)')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "plot_timeseries.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # ── FFT plot ──
    fig2, axes2 = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes2 = [axes2]

    for ax, (name, data) in zip(axes2, datasets.items()):
        df = data["df"]
        field_col = data["field"]
        values = df[field_col].values
        times = df["time_s"].values

        if len(values) < 2:
            continue

        dt = np.mean(np.diff(times))
        # Remove DC offset and apply Hann window
        values_centered = values - np.mean(values)
        window = np.hanning(len(values_centered))
        values_windowed = values_centered * window
        freqs = np.fft.rfftfreq(len(values_windowed), d=dt)
        fft_vals = np.abs(np.fft.rfft(values_windowed))
        freqs_ghz = freqs / 1e9

        mask = freqs_ghz <= args.fft_max
        ax.plot(freqs_ghz[mask], fft_vals[mask], linewidth=0.5)
        ax.set_ylabel('|FFT|')
        ax.set_title(f'FFT: {name} ({len(values)} points, df={1/(times[-1]-times[0])/1e9:.2f} GHz)')
        ax.grid(True, alpha=0.3)

        # Mark peak
        if fft_vals[mask].max() > 0:
            peak_idx = np.argmax(fft_vals[mask][1:]) + 1  # skip DC
            peak_freq = freqs_ghz[mask][peak_idx]
            ax.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.5,
                       label=f'peak: {peak_freq:.2f} GHz')
            ax.legend()

    axes2[-1].set_xlabel('Frequency (GHz)')
    plt.tight_layout()
    fft_path = os.path.join(args.save_dir, "plot_fft.png")
    plt.savefig(fft_path, dpi=150)
    print(f"Saved: {fft_path}")

    # ── Normalized FFT (probe2 / probe1) ──
    # Shows resonance features at probe2 with input spectrum divided out
    if n_plots >= 2:
        names = list(datasets.keys())
        input_name = names[0]   # first CSV = input (CPW)
        output_name = names[1]  # second CSV = output (junction)

        df_in = datasets[input_name]["df"]
        df_out = datasets[output_name]["df"]
        field_in = datasets[input_name]["field"]
        field_out = datasets[output_name]["field"]

        vals_in = df_in[field_in].values
        vals_out = df_out[field_out].values
        times_in = df_in["time_s"].values

        # Use same length
        n_common = min(len(vals_in), len(vals_out))
        vals_in = vals_in[:n_common]
        vals_out = vals_out[:n_common]
        times_in = times_in[:n_common]

        dt = np.mean(np.diff(times_in))
        window = np.hanning(n_common)

        fft_in = np.abs(np.fft.rfft((vals_in - np.mean(vals_in)) * window))
        fft_out = np.abs(np.fft.rfft((vals_out - np.mean(vals_out)) * window))
        freqs = np.fft.rfftfreq(n_common, d=dt)
        freqs_ghz = freqs / 1e9

        # Normalize: output / input (avoid divide by zero)
        fft_in_safe = np.where(fft_in > fft_in.max() * 1e-6, fft_in, fft_in.max() * 1e-6)
        fft_normalized = fft_out / fft_in_safe

        mask = freqs_ghz <= args.fft_max

        fig3, axes3 = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Input spectrum
        axes3[0].plot(freqs_ghz[mask], fft_in[mask], linewidth=0.5, color='blue')
        axes3[0].set_ylabel('|FFT|')
        axes3[0].set_title(f'Input: {input_name}')
        axes3[0].grid(True, alpha=0.3)

        # Output spectrum
        axes3[1].plot(freqs_ghz[mask], fft_out[mask], linewidth=0.5, color='orange')
        axes3[1].set_ylabel('|FFT|')
        axes3[1].set_title(f'Output: {output_name}')
        axes3[1].grid(True, alpha=0.3)

        # Normalized
        axes3[2].plot(freqs_ghz[mask], fft_normalized[mask], linewidth=0.5, color='green')
        axes3[2].set_ylabel('|Output / Input|')
        axes3[2].set_title(f'Normalized: {output_name} / {input_name}')
        axes3[2].grid(True, alpha=0.3)

        # Mark peak on normalized
        norm_vals = fft_normalized[mask]
        if len(norm_vals) > 1 and norm_vals.max() > 0:
            peak_idx = np.argmax(norm_vals[1:]) + 1
            peak_freq = freqs_ghz[mask][peak_idx]
            axes3[2].axvline(x=peak_freq, color='r', linestyle='--', alpha=0.5,
                             label=f'peak: {peak_freq:.2f} GHz')
            axes3[2].legend()

        axes3[-1].set_xlabel('Frequency (GHz)')
        plt.tight_layout()
        norm_path = os.path.join(args.save_dir, "plot_fft_normalized.png")
        plt.savefig(norm_path, dpi=150)
        print(f"Saved: {norm_path}")
