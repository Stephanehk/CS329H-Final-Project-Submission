#!/usr/bin/env python3
"""
Parse an RLlib training log and plot:
  1. Mean return over the 5 evaluation episodes.
  2. default_policy/true_reward_mean  (custom_metrics)
  3. default_policy/proxy_reward_mean (custom_metrics)

Usage
-----
python plot_training_metrics.py <logfile> [--show] [--save <png>]

Arguments
---------
<logfile>     Path to a text file containing the RLlib stdout/stderr log.

Optional flags
--------------
--show         Display the plot in a window.
--save <png>   Save the plot to the given filename instead of (or in addition to) showing it.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Regex patterns (compiled once for speed/readability)
# ---------------------------------------------------------------------
ITER_RE            = re.compile(r"Iteration\s+(\d+)\s*/")  # e.g. "Iteration 1/100"
MEAN_RET_RE        = re.compile(r"default_policy/modified_reward_mean'\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)") #re.compile(r"Mean return over 5 episodes:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
TRUE_REW_RE        = re.compile(r"default_policy/true_reward_mean'\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
PROXY_REW_RE       = re.compile(r"default_policy/proxy_reward_mean'\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
#total_modified_reward_mean
def parse_log(logfile: Path) -> List[Dict[str, Any]]:
    """Scan the file once, returning a list of per-iteration dicts."""
    data = []
    current: Dict[str, Any] = {}

    with logfile.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # -- Detect new iteration -----------------------------------------------------------
            if (m := ITER_RE.search(line)):
                # flush previous iteration (if any)
                if current:
                    data.append(current)
                    current = {}
                current["iteration"] = int(m.group(1))
                continue

            # -- Metric extraction --------------------------------------------------------------
            if (m := MEAN_RET_RE.search(line)):
                current["mean_return_5ep"] = float(m.group(1))
            if (m := TRUE_REW_RE.search(line)):
                current["true_reward_mean"] = float(m.group(1))
            if (m := PROXY_REW_RE.search(line)):
                current["proxy_reward_mean"] = float(m.group(1))

    # append last iteration if not empty
    if current:
        data.append(current)
    # print (data)
    # Filter out iterations that are missing any metric
    required_keys = {"mean_return_5ep", "true_reward_mean", "proxy_reward_mean"}
    cleaned = [row for row in data if required_keys.issubset(row)]
    if len(cleaned) < len(data):
        missing = len(data) - len(cleaned)
        print(f"[parse_log] Skipped {missing} incomplete iteration(s).")

    return cleaned


def plot_metrics(metrics: List[Dict[str, Any]], show: bool, save):
    """Plot three curves—each in its own subplot—against iteration."""
    # --- unpack -----------------------------------------------------------------
    iters        = [d["iteration"] for d in metrics]
    mean_returns = [d["mean_return_5ep"]  for d in metrics]
    true_means   = [d["true_reward_mean"] for d in metrics]
    proxy_means  = [d["proxy_reward_mean"] for d in metrics]


    if "pandemic" in save:
        gt_return = -2.65
        demonstration_return = -12.26
    elif "glucose" in save:
        gt_return = -43400 
        demonstration_return = -72600
        mean_return = -220428.03
    elif "traffic" in save:
        gt_return = -930
        demonstration_return = -2280
        uniform_return = -23326

    # --- figure / axes ----------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,      # common iteration axis
        figsize=(9, 7),   # a bit taller than before
        dpi=150
    )

    # --- 1. evaluation mean return ---------------------------------------------
    axes[0].plot(iters, mean_returns, marker="o", linewidth=1.5)
    axes[0].set_ylabel("Mean return w.r.t\nlearned reward function")
    # axes[0].set_title("Training metrics over iterations")
    axes[0].grid(alpha=0.3)

    # --- 2. true reward mean ----------------------------------------------------
    axes[1].plot(iters, true_means, marker="o", linewidth=1.5, color="tab:orange")
    axes[1].set_ylabel("Mean return w.r.t\n R1")
    axes[1].grid(alpha=0.3)

    axes[1].axhline(y=gt_return, linestyle="--", color="gray", linewidth=1.0)
    axes[1].axhline(y=demonstration_return, linestyle="--", color="black", linewidth=1.0)

    # --- 3. proxy reward mean ---------------------------------------------------
    axes[2].plot(iters, proxy_means, marker="o", linewidth=1.5, color="tab:green")
    axes[2].set_ylabel("Mean return w.r.t\n R2")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(alpha=0.3)

    # --- tidy-up / output -------------------------------------------------------
    fig.tight_layout()

    if save:
        fig.savefig(save)
        print(f"[plot_metrics] Saved figure to '{save}'")

    if show or not save:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logfile", type=Path, help="RLlib log file to parse")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    parser.add_argument("--save", metavar="PNG", help="Save the plot to a PNG file")
    args = parser.parse_args()

    if not args.logfile.exists():
        raise FileNotFoundError(f"Log file '{args.logfile}' not found")

    metrics = parse_log(args.logfile)
    if not metrics:
        raise RuntimeError("No complete iterations with all three metrics were found.")

    if not args.save and not args.show:
        print (metrics)
    # return 
    plot_metrics(metrics, show=args.show, save=args.save)


if __name__ == "__main__":
    main()
