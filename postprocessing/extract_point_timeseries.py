import yt
import numpy as np
import glob
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm


def extract_field_at_point(plt_file, field, x_loc, y_loc, z_loc):
    """Extract field value at a specific (x, y, z) point."""
    try:
        ds = yt.load(plt_file)
        # Use a small sphere (1 cell) around the point
        dx = (ds.domain_right_edge[0] - ds.domain_left_edge[0]) / ds.domain_dimensions[0]
        pt = ds.point([x_loc, y_loc, z_loc])
        val = float(pt[field].d[0]) if len(pt[field].d) > 0 else np.nan
        return val
    except Exception as e:
        print(f"Error processing {plt_file}: {e}")
        return np.nan


def extract_field_at_point_covering_grid(plt_file, field, ix, iy, iz):
    """Extract field value at a specific grid index using covering_grid (more robust)."""
    try:
        ds = yt.load(plt_file)
        cg = ds.covering_grid(
            level=0,
            left_edge=ds.domain_left_edge,
            dims=ds.domain_dimensions,
        )
        val = float(cg[field].d[ix, iy, iz])
        return val
    except Exception as e:
        print(f"Error processing {plt_file}: {e}")
        return np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Ex time series at two point probes from WarpX plotfiles"
    )
    parser.add_argument("--plot_dir", type=str, default="../diags",
                        help="Directory containing plotfiles")
    parser.add_argument("--field", type=str, default="Ex",
                        help="Field component (Ex, Ey, Ez)")
    parser.add_argument("--start", type=int, default=1,
                        help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=-1,
                        help="End index; -1 means till end")
    parser.add_argument("--step", type=int, default=1,
                        help="Step size for skipping plotfiles")

    # Probe 1: CPW gap (between source and qubit)
    parser.add_argument("--probe1_x", type=float, default=412e-6,
                        help="Probe 1 x position in meters (gap center)")
    parser.add_argument("--probe1_y", type=float, default=800e-6,
                        help="Probe 1 y position in meters")
    parser.add_argument("--probe1_z", type=float, default=101e-6,
                        help="Probe 1 z position in meters")
    parser.add_argument("--probe1_name", type=str, default="cpw_input",
                        help="Probe 1 label")

    # Probe 2: Junction gap (inductor location, from metafile)
    parser.add_argument("--probe2_x", type=float, default=205e-6,
                        help="Probe 2 x position in meters")
    parser.add_argument("--probe2_y", type=float, default=349e-6,
                        help="Probe 2 y position in meters")
    parser.add_argument("--probe2_z", type=float, default=101e-6,
                        help="Probe 2 z position in meters")
    parser.add_argument("--probe2_name", type=str, default="junction",
                        help="Probe 2 label")

    parser.add_argument("--save_dir", type=str, default=".",
                        help="Directory to save output CSV and plots")
    parser.add_argument("--use_index", action="store_true",
                        help="Use grid index instead of physical coords (faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSVs, only process new plotfiles")

    args = parser.parse_args()

    # Grid parameters (from input file)
    N_CELLX = 840
    N_CELLY = 1000
    N_CELLZ = 40
    PROB_HIX = 840e-6
    PROB_HIY = 1000e-6
    PROB_HIZ = 200e-6
    DX = PROB_HIX / N_CELLX
    DY = PROB_HIY / N_CELLY
    DZ = PROB_HIZ / N_CELLZ
    CFL = 0.8
    DT = 1.868332637e-15

    # Convert physical coords to grid indices
    probes = {
        args.probe1_name: {
            "x_m": args.probe1_x, "y_m": args.probe1_y, "z_m": args.probe1_z,
            "ix": int(round(args.probe1_x / DX)),
            "iy": int(round(args.probe1_y / DY)),
            "iz": int(round(args.probe1_z / DZ)),
        },
        args.probe2_name: {
            "x_m": args.probe2_x, "y_m": args.probe2_y, "z_m": args.probe2_z,
            "ix": int(round(args.probe2_x / DX)),
            "iy": int(round(args.probe2_y / DY)),
            "iz": int(round(args.probe2_z / DZ)),
        },
    }

    # Print probe info
    for name, p in probes.items():
        print(f"Probe '{name}': ({p['x_m']*1e6:.1f}, {p['y_m']*1e6:.1f}, {p['z_m']*1e6:.1f}) um "
              f"→ index ({p['ix']}, {p['iy']}, {p['iz']})")

    # Find plotfiles
    plotfiles = sorted([
        pf for pf in glob.glob(os.path.join(args.plot_dir, "plt*"))
        if os.path.isdir(pf) and ".old." not in os.path.basename(pf)
    ])

    end = args.end if args.end != -1 else len(plotfiles)
    files_to_process = plotfiles[args.start:end:args.step]
    print(f"Total plotfiles: {len(plotfiles)}, processing: {len(files_to_process)}")

    if not files_to_process:
        print("No plotfiles found!")
        sys.exit(0)

    # Load existing CSVs if resuming
    os.makedirs(args.save_dir, exist_ok=True)
    existing_steps = set()
    existing_data = {}

    if args.resume:
        for name in probes:
            csv_path = os.path.join(args.save_dir, f"timeseries_{name}.csv")
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                existing_data[name] = df_old
                existing_steps.update(df_old["step"].astype(int).tolist())
                print(f"Loaded {len(df_old)} existing points from {csv_path}")
        if existing_steps:
            print(f"Resuming: skipping {len(existing_steps)} already-processed steps")
            files_to_process = [
                pf for pf in files_to_process
                if int(os.path.basename(pf).replace("plt", "")) not in existing_steps
            ]
            print(f"New plotfiles to process: {len(files_to_process)}")

    # Extract time series for both probes
    results = {name: [] for name in probes}
    steps = []
    times = []

    if files_to_process:
        for pf in tqdm(files_to_process):
            step = int(os.path.basename(pf).replace("plt", ""))
            t = step * DT
            steps.append(step)
            times.append(t)

            for name, p in probes.items():
                if args.use_index:
                    val = extract_field_at_point_covering_grid(
                        pf, args.field, p['ix'], p['iy'], p['iz'])
                else:
                    val = extract_field_at_point(
                        pf, args.field, p['x_m'], p['y_m'], p['z_m'])
                results[name].append(val)

    times = np.array(times)
    steps = np.array(steps)

    # Merge with existing data and save
    for name, values in results.items():
        values = np.array(values)
        df_new = pd.DataFrame({
            "step": steps,
            "time_s": times,
            "time_ns": times * 1e9,
            f"{args.field}": values,
        })

        if args.resume and name in existing_data:
            df_merged = pd.concat([existing_data[name], df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset="step").sort_values("step").reset_index(drop=True)
        else:
            df_merged = df_new

        csv_path = os.path.join(args.save_dir, f"timeseries_{name}.csv")
        df_merged.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(df_merged)} total points, {len(df_new)} new)")

        # Update results for plotting
        results[name] = df_merged[args.field].values

    # Use merged times for plotting
    if args.resume and existing_data:
        first_name = list(probes.keys())[0]
        csv_path = os.path.join(args.save_dir, f"timeseries_{first_name}.csv")
        df_ref = pd.read_csv(csv_path)
        times = df_ref["time_s"].values
        steps = df_ref["step"].values

    # Plot time series
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (name, values) in zip(axes, results.items()):
        values = np.array(values)
        ax.plot(times * 1e9, values, linewidth=0.5)
        p = probes[name]
        ax.set_ylabel(f'{args.field} (V/m)')
        ax.set_title(f'{name} @ ({p["x_m"]*1e6:.0f}, {p["y_m"]*1e6:.0f}, {p["z_m"]*1e6:.0f}) um')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (ns)')
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "timeseries_probes.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # FFT
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (name, values) in zip(axes2, results.items()):
        values = np.array(values)
        if len(values) < 2:
            continue
        dt_avg = np.mean(np.diff(times))
        freqs = np.fft.rfftfreq(len(values), d=dt_avg)
        fft_vals = np.abs(np.fft.rfft(values))
        freqs_ghz = freqs / 1e9
        mask = freqs_ghz <= 20
        ax.plot(freqs_ghz[mask], fft_vals[mask], linewidth=0.5)
        ax.set_ylabel('|FFT|')
        ax.set_title(f'FFT: {name}')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label='5 GHz target')
        ax.legend()
    axes2[-1].set_xlabel('Frequency (GHz)')
    plt.tight_layout()
    fft_path = os.path.join(args.save_dir, "fft_probes.png")
    plt.savefig(fft_path, dpi=150)
    print(f"Saved: {fft_path}")
