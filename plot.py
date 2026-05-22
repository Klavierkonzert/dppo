"""
Plot training curves (loss, PPO sub-losses, rewards, eval success rate, ...) from
a DPPO fine-tune `run.log`.

Default behaviour:
  * Find the most recent run dir under ${DPPO_LOG_DIR}/gym-finetune.
  * Parse its `run.log` for both train-iteration lines and eval-iteration lines.
  * Save a multi-panel PNG to `<run_dir>/plot.png`.

Examples:
    python plot.py                                  # latest run
    python plot.py --run-dir /path/to/some/run     # specific run
    python plot.py --smooth 5                       # add a window=5 moving-average overlay
    python plot.py --x step                         # x-axis = total env steps instead of iteration
    python plot.py --show                           # interactive (needs display)
    python plot.py --no-log-loss                    # linear y-scale on loss panels
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

# Default to a non-interactive backend if there's no display; --show flips to interactive.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----- log parsing ---------------------------------------------------------- #

_TRAIN_PREFIX_RE = re.compile(r"^(?P<itr>\d+):\s*step\s+(?P<step>\d+)\s*\|\s*(?P<rest>.*)$")
_EVAL_PREFIX_RE = re.compile(r"^eval:\s*(?P<rest>.*)$")
_LOG_PREFIX_RE = re.compile(r"^\[[^\]]*\]\[[^\]]*\]\[[^\]]*\]\s*-\s*(?P<body>.*)$")


def _parse_kv_segments(rest: str) -> dict:
    """Parse a string like 'loss 1.23 | pg loss 0.4 | t: 5.0' into a dict of floats."""
    out = {}
    for chunk in rest.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("t:"):  # the only key that uses ':' as separator
            key, val = "time", chunk[2:].strip()
        else:
            parts = chunk.rsplit(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip().replace(" ", "_"), parts[1]
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def parse_log(log_path: Path):
    """Return (train_records, eval_records). Each record is a dict of metrics."""
    train_records, eval_records = [], []
    with open(log_path) as f:
        for line in f:
            m = _LOG_PREFIX_RE.match(line.rstrip())
            body = m.group("body") if m else line.strip()

            tm = _TRAIN_PREFIX_RE.match(body)
            if tm:
                rec = {
                    "itr": int(tm.group("itr")),
                    "step": int(tm.group("step")),
                }
                rec.update(_parse_kv_segments(tm.group("rest")))
                train_records.append(rec)
                continue

            em = _EVAL_PREFIX_RE.match(body)
            if em:
                rec = _parse_kv_segments(em.group("rest"))
                # Anchor eval lines to the iteration boundary they occur on:
                # eval is logged at the start of iter N (so before any subsequent train
                # line carrying that itr). We tag each eval record with the itr of the
                # *next* train record we'll see, falling back to len(train_records).
                rec["_eval_index"] = len(eval_records)
                rec["_train_itr_at_log"] = train_records[-1]["itr"] if train_records else 0
                eval_records.append(rec)
    return train_records, eval_records


def _resolve_eval_x(eval_records, train_records, x_field="itr"):
    """Best-effort: map eval records to a numeric x value matching train x-axis.

    The agent runs eval whenever (itr % val_freq == 0). Train lines carry the
    current itr explicitly, so we infer val_freq from the smallest itr gap > 0,
    then space eval ticks evenly. Falls back to integer eval index when no train
    records are present.
    """
    if not eval_records:
        return []
    if not train_records:
        return [r["_eval_index"] for r in eval_records]

    # Use the gaps between train itrs to guess val_freq.
    itrs = [r["itr"] for r in train_records]
    val_freq = 1
    if len(itrs) >= 2:
        # Train lines skip the eval itr, so consecutive itrs that differ by >1 mark an eval boundary.
        diffs = [b - a for a, b in zip(itrs, itrs[1:]) if b > a]
        if diffs:
            val_freq = max(min(diffs), 1)

    # eval_index 0 corresponds to itr 0; subsequent evals are at multiples of val_freq.
    base_itrs = [r["_eval_index"] * val_freq for r in eval_records]

    if x_field == "step" and train_records:
        # Linearly interpolate eval-itr → step using the train series.
        train_itr_arr = np.array([r["itr"] for r in train_records], dtype=float)
        train_step_arr = np.array([r["step"] for r in train_records], dtype=float)
        return np.interp(base_itrs, train_itr_arr, train_step_arr).tolist()
    return base_itrs


# ----- run dir discovery ---------------------------------------------------- #

def find_latest_run(search_root: Path) -> Path:
    """Pick the most recently modified directory under search_root that contains run.log."""
    candidates = [d for d in search_root.rglob("run.log")]
    if not candidates:
        raise FileNotFoundError(f"No run.log found under {search_root}")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return newest.parent


def default_search_root() -> Path:
    log_dir = os.environ.get("DPPO_LOG_DIR")
    if not log_dir:
        raise EnvironmentError(
            "DPPO_LOG_DIR is not set. Pass --run-dir explicitly or export the variable."
        )
    return Path(log_dir) / "gym-finetune"


# ----- plotting ------------------------------------------------------------- #

_TRAIN_PANELS = [
    # (metric_key, title, use_symlog_y)
    ("loss", "Loss", True),
    ("pg_loss", "PG Loss", True),
    ("value_loss", "Value Loss", True),
    ("bc_loss", "BC Loss", False),
    ("reward", "Train reward (avg episode)", False),
    ("eta", "Eta (PPO entropy coef)", False),
    ("time", "Wall-clock per iter (s)", False),
]

_EVAL_PANELS = [
    ("success_rate", "Eval success rate", False),
    ("avg_episode_reward", "Eval episode reward (avg)", False),
    ("avg_best_reward", "Eval best reward (avg)", False),
]


def _maybe_smooth(ys, window):
    if window is None or window <= 1 or len(ys) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(ys, kernel, mode="valid")


def plot_run(
    train_records,
    eval_records,
    out_path: Path | None,
    smooth: int = 0,
    log_loss: bool = True,
    show: bool = False,
    x_field: str = "itr",
    title: str = "",
):
    panels = []
    for key, title_, symlog in _TRAIN_PANELS:
        if any(key in r for r in train_records):
            panels.append(("train", key, title_, symlog))
    for key, title_, _ in _EVAL_PANELS:
        if any(key in r for r in eval_records):
            panels.append(("eval", key, title_, False))

    if not panels:
        raise RuntimeError("No plottable metrics found in run.log.")

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.2 * rows), squeeze=False)
    axes = axes.flatten()

    train_x = [r[x_field] for r in train_records]
    eval_x = _resolve_eval_x(eval_records, train_records, x_field=x_field)

    for ax, (source, key, panel_title, symlog) in zip(axes, panels):
        if source == "train":
            xs = train_x
            ys = [r.get(key, np.nan) for r in train_records]
        else:
            xs = eval_x
            ys = [r.get(key, np.nan) for r in eval_records]

        ax.plot(xs, ys, marker="o" if source == "eval" else None,
                markersize=3, alpha=0.55 if source == "train" else 0.9, label="raw")

        if source == "train" and smooth > 1:
            smoothed = _maybe_smooth(ys, smooth)
            if smoothed is not None:
                offset = (len(ys) - len(smoothed)) // 2
                ax.plot(
                    xs[offset : offset + len(smoothed)],
                    smoothed,
                    color="C3",
                    label=f"smooth(w={smooth})",
                )
                ax.legend(fontsize=8)

        ax.set_title(panel_title)
        ax.set_xlabel("env step" if x_field == "step" else "iteration")
        ax.grid(alpha=0.3)

        if symlog and log_loss:
            ax.set_yscale("symlog", linthresh=1.0)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if show:
        plt.show()
    elif out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"[INFO] Saved plot: {out_path}")
    plt.close(fig)


# ----- CLI ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing run.log. Defaults to latest under ${DPPO_LOG_DIR}/gym-finetune.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <run_dir>/plot.png.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Moving-average window for train curves (0 = off).",
    )
    parser.add_argument(
        "--x",
        choices=("itr", "step"),
        default="itr",
        help="X axis: training iteration (default) or total env steps.",
    )
    parser.add_argument(
        "--no-log-loss",
        action="store_true",
        help="Disable symlog y-scale on loss / pg_loss / value_loss panels.",
    )
    parser.add_argument("--show", action="store_true", help="Show interactively instead of saving.")
    args = parser.parse_args()

    if args.show and not os.environ.get("DISPLAY"):
        # Re-enable a GUI backend if user explicitly asked for one.
        matplotlib.use("TkAgg", force=True)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = find_latest_run(default_search_root())

    log_path = run_dir / "run.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"No run.log in {run_dir}")

    print(f"[INFO] Reading log: {log_path}")
    train_records, eval_records = parse_log(log_path)
    print(
        f"[INFO] Parsed {len(train_records)} train records, "
        f"{len(eval_records)} eval records."
    )
    if not train_records and not eval_records:
        sys.exit("No plottable lines in run.log.")

    out_path = None if args.show else Path(args.out) if args.out else run_dir / "plot.png"
    plot_run(
        train_records,
        eval_records,
        out_path,
        smooth=args.smooth,
        log_loss=not args.no_log_loss,
        show=args.show,
        x_field=args.x,
        title=run_dir.name,
    )


if __name__ == "__main__":
    main()
