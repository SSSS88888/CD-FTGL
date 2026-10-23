#!/usr/bin/env python
"""Regenerate every figure panel of the paper from the bundled results.

The repository ships the raw per-run results (``experiment/result/**/*.npy``) together
with the best hyper-parameter tables that were used for the paper
(``experiment/result/repeated_best_algorithms_comparison_*/best_parameters_tables/*.csv``).
Running this script therefore reproduces the figures **without** re-running any algorithm.

Usage
-----
    python experiment/reproduce_figures.py                    # all 9 figures (36 panels)
    python experiment/reproduce_figures.py --figures 1 2 3    # only a subset
    python experiment/reproduce_figures.py --datasets a9a     # only one dataset

Output: ``<repo>/figures/fig<N>_<dataset>_loss_vs_{rounds,bits}.{png,pdf}``
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The plotting code resolves its data paths relative to this directory.
os.chdir(HERE)

import plot_best_multiple_runs as pbmr          # noqa: E402
import compare_best_algorithms_Lp as lpc       # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
WORK_DIR = os.path.join(RESULT_DIR, "_reproduced")
FIG_DIR = os.path.join(ROOT, "figures")

# paper figure index -> compression ratio, for each network size / experiment block
N9_FIG = {0.1: 1, 0.5: 2, 0.05: 3}      # Fig. 1-3  (9 nodes, 18 edges)
N50_FIG = {0.5: 4, 0.1: 5, 0.05: 6}     # Fig. 4-6  (50 nodes, 100 edges)
K_FIG = {0.5: 7, 0.1: 8, 0.05: 9}       # Fig. 7-9  (L=100, varying K)


def reproduce_main_figures(datasets, rates):
    """Fig. 1-6: CD-FTGL / CD-FTBL vs. the four baselines (uses the best-parameter tables)."""
    for net in ("n9", "n50"):
        pbmr.DATASET_NAME_TO_LOAD = list(datasets)
        pbmr.COMPRESSOR_TYPES_TO_LOAD = ["top"]
        pbmr.TOPOLOGY_TYPES_TO_LOAD = ["random"]
        pbmr.FEEDBACK_TYPES_TO_LOAD = ["full", "bandit"]
        pbmr.T_VALUES_TO_LOAD = [10000]
        pbmr.COMPRESSION_RATES = list(rates)
        pbmr.BASE_PARAMS_DIR = os.path.join(
            RESULT_DIR, "repeated_best_algorithms_comparison_%s" % net, "best_parameters_tables")
        pbmr.BASE_RESULT_DIR = os.path.join(RESULT_DIR, "grid_search_repeated_%s" % net)
        pbmr.BASE_SAVE_DIR = os.path.join(WORK_DIR, net)
        print("\n=== regenerating %s figures (best parameters, T=10000) ===" % net)
        pbmr.main()


def reproduce_K_figures(datasets):
    """Fig. 7-9: CD-FTBL with block size L=100 and K in {2,5,10,20,50}."""
    lpc.L_CHOICES = [100]
    lpc.BASE_SAVE_DIR = os.path.join(WORK_DIR, "K")
    for ds in datasets:
        lpc.DATASET_NAME = ds
        lpc.base_path = os.path.join(
            RESULT_DIR, "grid_search_repeated_n9", ds, "top", "random", "bandit")
        print("\n=== regenerating L=100 / varying-K figures for %s ===" % ds)
        lpc.main()


def _copy_panel(src_base, dst_base):
    """Copy <src_base>.{png,pdf} to <dst_base>.{png,pdf}; return False if the panel is missing."""
    written = []
    for ext in (".png", ".pdf"):
        src, dst = src_base + ext, dst_base + ext
        if not os.path.exists(src):
            return []
        shutil.copyfile(src, dst)
        written.append(dst)
    return written


def collect_panels(datasets, rates):
    os.makedirs(FIG_DIR, exist_ok=True)
    written = []

    for net, fig_map in (("n9", N9_FIG), ("n50", N50_FIG)):
        for ds in datasets:
            for w in rates:
                idx = fig_map.get(w)
                if idx is None:
                    continue
                for kind, suffix in (("SR", "loss_vs_rounds"), ("Bits_vs_SR", "loss_vs_bits")):
                    stem = "optimal_comparison_%s_T10000_w%s_comptop_toporandom_combined" % (kind, w)
                    written += _copy_panel(
                        os.path.join(WORK_DIR, net, ds, "top", "random", stem),
                        os.path.join(FIG_DIR, "fig%d_%s_%s" % (idx, ds, suffix)))

    for ds in datasets:
        for w in rates:
            idx = K_FIG.get(w)
            if idx is None:
                continue
            for kind, suffix in (("sr_curves", "loss_vs_rounds"), ("bits_vs_sr", "loss_vs_bits")):
                stem = "Lp_comparison_w%s_%s_%s_top_random" % (w, kind, ds)
                written += _copy_panel(
                    os.path.join(WORK_DIR, "K", ds, "top", "random", "bandit", stem),
                    os.path.join(FIG_DIR, "fig%d_%s_%s" % (idx, ds, suffix)))

    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=["a9a", "ijcnn1"], choices=["a9a", "ijcnn1"])
    ap.add_argument("--figures", nargs="*", type=int, default=None,
                    help="only regenerate these figure numbers (1-9); default: all")
    ap.add_argument("--rates", nargs="*", type=float, default=[0.05, 0.1, 0.5])
    args = ap.parse_args()

    rates = args.rates
    if args.figures:
        keep = set(args.figures)
        rates = [w for w in rates
                 if N9_FIG.get(w) in keep or N50_FIG.get(w) in keep or K_FIG.get(w) in keep]

    os.makedirs(WORK_DIR, exist_ok=True)

    reproduce_main_figures(args.datasets, rates)
    reproduce_K_figures(args.datasets)

    written = collect_panels(args.datasets, rates)

    print("\n" + "=" * 70)
    panels = len(written) // 2
    print("done: %d panels (%d files: png+pdf) written to %s" % (panels, len(written), FIG_DIR))
    for p in written:
        print("  " + os.path.relpath(p, ROOT))
    expected = 6 * len(rates) * len(args.datasets)
    if panels != expected:
        print("WARNING: expected %d panels, got %d" % (expected, panels), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
