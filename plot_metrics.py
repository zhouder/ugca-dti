#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a 2x3 panel image for AUROC/AUPRC/F1/ACC/SEN/MCC from metrics.csv.
Each metric is drawn as its own matplotlib figure (no subplots), then
stitched into a single PNG via PIL. Also supports batch mode for all folds.

Usage:
  # one fold
  python plot_metrics_panel.py "/root/lanyun-tmp/ugca-runs/DAVIS-cold-protein-seq-s42/fold1"

  # all folds under a run root
  python plot_metrics_panel.py "/root/lanyun-tmp/ugca-runs/DAVIS-cold-protein-seq-s42" --all-folds
"""
import argparse, os, glob, tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

METRICS = ["AUROC","AUPRC","F1","ACC","SEN","MCC"]

def draw_curve(x, y, title, ylabel, out_png):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()

def make_panel(png_paths, out_png, rows=2, cols=3, pad=20, bg=(255,255,255)):
    # open and normalize size
    imgs = [Image.open(p) for p in png_paths]
    # resize to the smallest size among them to keep quality more consistent
    w = min(im.size[0] for im in imgs)
    h = min(im.size[1] for im in imgs)
    imgs = [im.resize((w, h)) for im in imgs]

    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    panel = Image.new("RGB", (W, H), color=bg)

    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        panel.paste(im, (x, y))

    panel.save(out_png)

def process_fold(fold_dir: str):
    csv_path = os.path.join(fold_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        print(f"[skip] epoch column not in {csv_path}")
        return None

    tmp_pngs = []
    for m in METRICS:
        if m not in df.columns:  # skip missing metrics
            continue
        out_png = os.path.join(fold_dir, f"_tmp_{m.lower()}.png")
        draw_curve(df["epoch"].values, df[m].values, m, m, out_png)
        tmp_pngs.append(out_png)

    if not tmp_pngs:
        print(f"[warn] no metrics found to plot in {csv_path}")
        return None

    # If less than 6 metrics, pad with blank images so the grid is consistent
    while len(tmp_pngs) < 6:
        # create a simple blank canvas
        blank = os.path.join(fold_dir, f"_tmp_blank_{len(tmp_pngs)}.png")
        plt.figure()
        plt.plot([], [])
        plt.xlabel("Epoch"); plt.ylabel("")
        plt.title("")
        plt.grid(True)
        plt.savefig(blank, dpi=160, bbox_inches="tight")
        plt.close()
        tmp_pngs.append(blank)

    out_panel = os.path.join(fold_dir, "metrics_panel.png")
    make_panel(tmp_pngs[:6], out_panel)
    print("[saved]", out_panel)
    return out_panel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to fold dir (…/fold1) or run root (…/DATASET-cold-xxx)")
    ap.add_argument("--all-folds", action="store_true", help="if path is run root, process all fold*/ dirs")
    args = ap.parse_args()

    # detect
    is_fold = os.path.basename(args.path).startswith("fold") or os.path.exists(os.path.join(args.path, "metrics.csv"))
    if is_fold:
        process_fold(args.path)
    else:
        if args.all_folds:
            for fd in sorted(glob.glob(os.path.join(args.path, "fold*", ""))):
                process_fold(fd)
        else:
            print("You passed a run root. Add --all-folds to process all fold*/ dirs.")

if __name__ == "__main__":
    main()
