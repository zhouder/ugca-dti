#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training curves from ugca-dti outputs with optional edge trimming and better smoothing.

Usage:
  python plot_metrics.py --out /root/lanyun-tmp/ugca-runs/davis --grid --smooth 5 --trim 1
"""
import argparse, time, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Path to a dataset run dir, e.g., /root/lanyun-tmp/ugca-runs/davis")
    p.add_argument("--metrics", nargs="*", default=["AUROC","AUPRC","F1","ACC","SEN","MCC"])
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--smooth", type=int, default=0, help="centered moving average window; 0 disables")
    p.add_argument("--trim", type=int, default=0, help="drop N epochs at both start and end when plotting")
    p.add_argument("--grid", action="store_true", help="also render a 2x3 grid with all metrics")
    p.add_argument("--watch", type=float, default=0.0, help="seconds between refresh; 0 for one-shot")
    return p.parse_args()

def load_folds(run_dir: Path):
    folds = {}
    for fdir in sorted(run_dir.glob("fold*")):
        csv_path = fdir / "metrics.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] failed to read {csv_path}: {e}", file=sys.stderr)
            continue
        mask_num = pd.to_numeric(df.get("epoch", pd.Series([], dtype=str)), errors="coerce").notna()
        df_curve = df[mask_num].copy()
        if not df_curve.empty:
            df_curve["epoch"] = df_curve["epoch"].astype(int)
            df_curve.sort_values("epoch", inplace=True)
        df_best = df[~mask_num].copy()
        best_row = df_best.iloc[-1].to_dict() if not df_best.empty else None
        folds[fdir.name] = {"curve": df_curve, "best": best_row}
    return folds

def centered_ma(y, k):
    if k <= 1 or len(y) < k:
        return y
    pad = k//2
    ypad = np.pad(y, (pad, pad), mode="edge")  # edge-pad to avoid boundary dip
    kernel = np.ones(k, dtype=float) / k
    out = np.convolve(ypad, kernel, mode="valid")
    return out

def maybe_trim(x, y, trim):
    if trim <= 0 or len(x) <= 2*trim:
        return x, y
    return x[trim:-trim], y[trim:-trim]

def render(run_dir: Path, metrics, dpi=160, smooth=0, trim=0, grid=False):
    folds = load_folds(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    # individual curves
    for m in metrics:
        plt.figure()
        xs_ref = None
        Ys = []
        any_line = False
        for fold, pack in folds.items():
            df = pack["curve"]
            if df is None or df.empty or m not in df.columns:
                continue
            x = df["epoch"].to_numpy()
            y = df[m].to_numpy(dtype=float)
            if smooth and smooth > 1:
                y = centered_ma(y, smooth)
            x, y = maybe_trim(x, y, trim)
            if len(x)==0:
                continue
            plt.plot(x, y, label=fold, alpha=0.8)
            any_line = True
            if xs_ref is None: xs_ref = x
            Ys.append(np.interp(xs_ref, x, y))
        if any_line and Ys:
            mean = np.vstack(Ys).mean(0)
            plt.plot(xs_ref, mean, label="mean", linewidth=3, zorder=10)
        plt.title(f"{m}")
        plt.xlabel("epoch"); plt.ylabel(m)
        plt.grid(True, linestyle="--", alpha=0.3)
        if any_line: plt.legend()
        out_path = plot_dir / f"{m.lower()}_curve.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=dpi); plt.close()
        saved.append(out_path)

    # grid
    if grid:
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*3.6), dpi=dpi)
        ax_list = axes.flatten()
        for ax, m in zip(ax_list, metrics):
            xs_ref = None; Ys=[]; any_line=False
            for fold,pack in folds.items():
                df = pack["curve"]
                if df is None or df.empty or m not in df.columns:
                    continue
                x = df["epoch"].to_numpy(); y = df[m].to_numpy(dtype=float)
                if smooth and smooth > 1: y = centered_ma(y, smooth)
                x, y = maybe_trim(x, y, trim)
                if len(x)==0: continue
                ax.plot(x, y, label=fold, alpha=0.8); any_line=True
                if xs_ref is None: xs_ref = x
                Ys.append(np.interp(xs_ref, x, y))
            if any_line and Ys:
                mean = np.vstack(Ys).mean(0)
                ax.plot(xs_ref, mean, label="mean", linewidth=3, zorder=10)
            ax.set_title(m); ax.set_xlabel("epoch"); ax.set_ylabel(m)
            ax.grid(True, linestyle="--", alpha=0.3)
            if any_line: ax.legend(fontsize=8)
        for k in range(len(metrics), len(ax_list)):
            ax_list[k].axis("off")
        fig.tight_layout()
        grid_path = plot_dir / "all_metrics_grid.png"
        fig.savefig(grid_path, dpi=dpi)
        plt.close(fig)
        saved.append(grid_path)

    return saved

def main():
    args = parse_args()
    run_dir = Path(args.out)
    if not run_dir.exists():
        print(f"[ERR] --out path not found: {run_dir}", file=sys.stderr); sys.exit(1)

    if args.watch and args.watch > 0:
        print(f"[watching] {run_dir} every {args.watch}s ... Ctrl+C to stop.")
        while True:
            saved = render(run_dir, args.metrics, dpi=args.dpi, smooth=args.smooth, trim=args.trim, grid=args.grid)
            for p in saved: print(f"[OK] {p}")
            time.sleep(args.watch)
    else:
        saved = render(run_dir, args.metrics, dpi=args.dpi, smooth=args.smooth, trim=args.trim, grid=args.grid)
        print("[OK] saved files:")
        for p in saved: print(" -", p)

if __name__ == "__main__":
    main()
