# -*- coding: utf-8 -*-
"""
make_fig_convergence.py

Create representative convergence figures for the PDABC manuscript.

Input files must follow the naming format:
    Algorithm_Function_D.txt

Examples:
    PDABC_1_20.txt
    ABC_1_20.txt
    EA4eigN100_10_1_20.txt

Each file must be a 17 x 30 matrix:
    rows 0..15 : error values at 16 recording points
    row  16    : FEterm

Output:
    Figure1.pdf
    Figure1.png
"""

from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

EPS = 1e-8
RUNS = 30

MAXFES_BY_D = {
    10: 200_000,
    20: 1_000_000,
}

# Representative functions
SELECTED_FUNCTIONS = [1, 3, 8, 11]

FUNCTION_LABELS = {
    1: "F1 (unimodal)",
    3: "F3 (basic)",
    8: "F8 (hybrid)",
    11: "F11 (composition)",
}

PLOT_ALGOS = [
    "PDABC",
    "ABC",
    "EA4eigN100_10",
    "NL-SHADE-LBC",
    "NL-SHADE-RSP-MID",
]

DISPLAY_NAMES = {
    "PDABC": "PDABC",
    "ABC": "ABC",
    "EA4eigN100_10": "EA4eigN100_10",
    "NL-SHADE-LBC": "NL-SHADE-LBC",
    "NL-SHADE-RSP-MID": "NL-SHADE-RSP-MID",
}

# Options: "median", "mean", "best"
CURVE_STAT = "median"

# Manual x-axis limits for clearer presentation in the manuscript
XMAX_BY_FUNCTION = {
    1: 0.105,
    3: 0.035,
    8: 0.200,
    11: 0.105,
}

# different line styles + different markers
STYLE = {
    "PDABC": dict(
        color="#1f77b4",          # blue
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        linestyle="-",
        markerfacecolor="#1f77b4",
        markeredgecolor="#1f77b4",
    ),
    "ABC": dict(
        color="#6f6f6f",          # gray
        linewidth=1.3,
        marker="s",
        markersize=3.2,
        linestyle="--",
        markerfacecolor="#6f6f6f",
        markeredgecolor="#6f6f6f",
    ),
    "EA4eigN100_10": dict(
        color="#2ca02c",          # green
        linewidth=1.3,
        marker="^",
        markersize=3.3,
        linestyle="-.",
        markerfacecolor="#2ca02c",
        markeredgecolor="#2ca02c",
    ),
    "NL-SHADE-LBC": dict(
        color="#d62728",          # red
        linewidth=1.3,
        marker="D",
        markersize=3.1,
        linestyle=":",
        markerfacecolor="white",
        markeredgecolor="#d62728",
    ),
    "NL-SHADE-RSP-MID": dict(
        color="#9467bd",          # purple
        linewidth=1.3,
        marker="v",
        markersize=3.3,
        linestyle=(0, (5, 2, 1, 2)),
        markerfacecolor="white",
        markeredgecolor="#9467bd",
    ),
}

# ============================================================
# DATA LOADING
# ============================================================

def cec2022_record_points(D: int) -> np.ndarray:
    """
    Return the 16 recording points.

    This matches the recording formula used in the current PDABC/ABC
    experiment scripts:
        floor(D^(k/5 - 3) * MaxFES), k = 0,...,15.
    """
    maxfes = MAXFES_BY_D[D]
    points = []

    for k in range(16):
        fes = int(np.floor((D ** (k / 5.0 - 3.0)) * maxfes))
        fes = max(1, min(fes, maxfes))
        points.append(fes)

    return np.asarray(points, dtype=float)


def read_numeric_matrix(filepath: Path) -> np.ndarray:
    """Read a numeric matrix with common delimiters."""
    last_error = None

    for delimiter in (None, ",", ";", "\t"):
        try:
            data = np.loadtxt(filepath, dtype=float, delimiter=delimiter)
            return np.asarray(data, dtype=float)
        except Exception as e:
            last_error = e

    raise ValueError(f"Cannot read {filepath.name}: {last_error}")


def normalize_shape(data: np.ndarray, filepath: Path) -> np.ndarray:
    """Normalize data to 17 x 30; transpose if needed."""
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"{filepath.name} is not 2D. Shape={data.shape}")

    if data.shape == (RUNS, 17):
        data = data.T

    if data.shape != (17, RUNS):
        raise ValueError(
            f"{filepath.name} has shape {data.shape}, expected 17 x {RUNS}"
        )

    return data


def load_error_matrix(filepath: Path) -> np.ndarray:
    """Load the 16 x 30 error matrix."""
    data = normalize_shape(read_numeric_matrix(filepath), filepath)
    errors = data[:16, :].astype(float)

    errors[~np.isfinite(errors)] = 1e300
    errors = np.maximum(errors, 0.0)

    return errors


def summarize_curve(errors_16x30: np.ndarray, stat: str) -> np.ndarray:
    """Return a 16-point convergence curve."""
    stat = stat.lower().strip()

    if stat == "median":
        y = np.median(errors_16x30, axis=1)
    elif stat == "mean":
        y = np.mean(errors_16x30, axis=1)
    elif stat == "best":
        y = np.min(errors_16x30, axis=1)
    else:
        raise ValueError("CURVE_STAT must be 'median', 'mean', or 'best'")

    return np.maximum(y, EPS)


# ============================================================
# PLOT HELPERS
# ============================================================

def nice_log_ylim(curves):
    """Choose readable log-scale limits from plotted curves."""
    vals = np.concatenate([np.asarray(c, dtype=float) for c in curves])
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size == 0:
        return EPS, 1.0

    ymin = max(EPS, np.nanmin(vals) / 3.0)
    ymax = np.nanmax(vals) * 3.0

    ymin = 10 ** np.floor(np.log10(ymin))
    ymax = 10 ** np.ceil(np.log10(ymax))

    if ymax <= ymin:
        ymax = ymin * 10.0

    return ymin, ymax


def set_clean_xticks(ax, xmax: float):
    """Set readable x ticks."""
    ticks = np.linspace(0.0, xmax, 5)

    if xmax <= 0.06:
        labels = [f"{t:.3f}" for t in ticks]
    elif xmax <= 0.25:
        labels = [f"{t:.2f}" for t in ticks]
    else:
        labels = [f"{t:.1f}" for t in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


# ============================================================
# PLOTTING
# ============================================================

def plot_one_dimension(base_dir: Path, D: int):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), squeeze=False)
    axes = axes.ravel()

    x = cec2022_record_points(D) / MAXFES_BY_D[D]

    legend_handles = None
    legend_labels = None

    for ax, fid in zip(axes, SELECTED_FUNCTIONS):
        curves_for_ylim = []

        for algo in PLOT_ALGOS:
            filepath = base_dir / f"{algo}_{fid}_{D}.txt"

            if not filepath.exists():
                warnings.warn(f"Missing file: {filepath.name}")
                continue

            try:
                errors = load_error_matrix(filepath)
                y = summarize_curve(errors, CURVE_STAT)
                curves_for_ylim.append(y)

                line_style = STYLE.get(
                    algo,
                    dict(
                        color="black",
                        linewidth=1.5,
                        marker="o",
                        markersize=3.0,
                        linestyle="-",
                    ),
                )

                ax.plot(
                    x,
                    y,
                    label=DISPLAY_NAMES.get(algo, algo),
                    **line_style,
                )

            except Exception as e:
                warnings.warn(f"Skip {filepath.name}: {e}")
                continue

        ax.set_yscale("log")

        if curves_for_ylim:
            ax.set_ylim(*nice_log_ylim(curves_for_ylim))

        xmax = XMAX_BY_FUNCTION.get(fid, 1.0)
        ax.set_xlim(0.0, xmax)
        set_clean_xticks(ax, xmax)

        ax.set_title(FUNCTION_LABELS[fid], fontsize=10)
        ax.set_xlabel(r"$FE/FE_{\max}$", fontsize=9)
        ax.set_ylabel(f"{CURVE_STAT.capitalize()} error", fontsize=9)

        ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.55)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.35)
        ax.tick_params(labelsize=8)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(len(legend_labels), 5),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.010),
            handlelength=3.8,
            handletextpad=0.7,
            columnspacing=1.6,
            markerscale=1.0,
        )

    fig.tight_layout(rect=[0, 0.075, 1, 0.995])

    pdf = base_dir / f"Figure1.pdf"
    png = base_dir / f"Figure1.png"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def main():
    script_dir = Path(__file__).resolve().parent

    if script_dir.name == "scripts":
        base_dir = script_dir.parent / "results"
    else:
        base_dir = script_dir / "results"

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Results folder not found: {base_dir}. "
            f"Please put the result .txt files in this folder."
        )

    print(f"Working folder: {base_dir}")
    print(f"Algorithms plotted: {PLOT_ALGOS}")
    print(f"Functions plotted: {SELECTED_FUNCTIONS}")
    print(f"Curve statistic: {CURVE_STAT}")
    print(f"X-axis limits: {XMAX_BY_FUNCTION}")

    plot_one_dimension(base_dir, D=20)


if __name__ == "__main__":
    main()