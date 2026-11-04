"""Render Task 3 figures from validated detector-call timings. Run as python -m src.visualize."""
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

from .benchmark import ROOT

# Okabe–Ito blue and vermilion, with redundant shape, line and hatch encodings.
STYLES = {
    "static_rules": ("AST rules", "#0072B2", "o", "-", ""),
    "multinomial_nb": ("Naive Bayes", "#C44E00", "s", "--", "///"),
}


def _canvas(number, title, subtitle, left):
    fig = Figure(figsize=(7.5, 4.65), facecolor="white")
    fig.text(0.045, 0.94, f"FIGURE 0{number}    /    AI TOOLS FOR SOFTWARE BUG DETECTION",
             fontsize=8.6, weight="bold", color="#416174")
    fig.text(0.045, 0.85, title, fontsize=18, weight="bold", color="#172832")
    fig.text(0.045, 0.795, subtitle, fontsize=10, color="#364A55")
    ax = fig.add_axes([left, 0.245, 0.95 - left, 0.49])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    return fig, ax


def _scale(ax, axis, values, min_ratio=100):
    """Spread broad ranges with log spacing, keeping tick labels in original units."""
    if np.min(values) > 0 and np.max(values) / np.min(values) >= min_ratio:
        target = getattr(ax, f"{axis}axis")
        getattr(ax, f"set_{axis}scale")("log")
        target.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
        return " · log scale"
    getattr(ax, f"set_{axis}lim")(**{"left" if axis == "x" else "bottom": 0})
    return ""


def _latency_by_size(calls):
    n_inputs = calls.snippet_id.nunique()
    n_passes = calls.repetition.nunique()
    subtitle = (f"{n_inputs} unlabelled repository functions   ·   {n_passes} benchmark passes per detector")
    fig, ax = _canvas(1, "Detector time by source length", subtitle, 0.129)
    bounds = []
    for method, (name, color, marker, linestyle, _) in STYLES.items():
        points = []
        selected = calls.loc[calls.method == method].sort_values("repetition")
        for (size, _), group in selected.groupby(["input_bytes", "snippet_id"], sort=True):
            # One observation per pass at a given function and method. Matching
            # pass order and seed reuses the same draws for each complete series.
            times = group.latency_ms.to_numpy()
            rng = np.random.default_rng(42)
            samples = rng.choice(times, size=(10_000, len(times)), replace=True)
            lo, hi = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
            points.append((size, np.median(times), lo, hi))
        sizes, medians, lows, highs = np.asarray(points).T
        ax.errorbar(sizes, medians, yerr=[medians - lows, highs - medians],
                    color=color, marker=marker, linestyle=linestyle, markersize=4.3,
                    markerfacecolor="white", markeredgewidth=1.3, linewidth=1.2,
                    elinewidth=1.0, capsize=4.5, capthick=1.1, label=name)
        bounds.extend(lows)
        bounds.extend(highs)

    x_scale = _scale(ax, "x", calls.input_bytes.to_numpy(), min_ratio=20)
    if x_scale:
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    y_scale = _scale(ax, "y", bounds)
    ax.set_xlabel(f"Function source length (UTF-8 bytes){x_scale}", labelpad=5)
    ax.set_ylabel(f"Detector latency (ms){y_scale}", labelpad=5)
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(0.045, 0.072,
             "Points are function medians. Caps show pointwise 95% intervals from resampling passes (10,000 draws)."
             "\nSome intervals are narrower than the markers. Lines connect functions in size order.",
             fontsize=9.2, color="#364A55", linespacing=1.35)
    return fig


def _latency_distribution(calls, summary):
    results = summary.loc[summary.corpus == "repository"].set_index("method")
    subtitle = (f"{len(calls):,} successful calls   ·   {calls.snippet_id.nunique()} unlabelled"
                f" repository functions   ·   {calls.repetition.nunique()} benchmark passes")
    fig, ax = _canvas(2, "Spread of individual detector times", subtitle, 0.175)
    for index, (method, (name, color, marker, _, hatch)) in enumerate(STYLES.items()):
        position = 1 - index
        values = calls.loc[calls.method == method, "latency_ms"].to_numpy()
        ax.boxplot([values], positions=[position + 0.10], widths=0.24, orientation="horizontal",
                   manage_ticks=False, patch_artist=True, whis=1.5,
                   boxprops={"facecolor": "white", "edgecolor": color, "hatch": hatch, "linewidth": 1.5},
                   whiskerprops={"color": color}, capprops={"color": color},
                   medianprops={"color": "#152933", "linewidth": 1.7},
                   flierprops={"marker": marker, "markersize": 3, "markerfacecolor": "none",
                               "markeredgecolor": color, "alpha": 0.65})
        row = results.loc[method]
        median = row.latency_p50_ms
        lo, hi = row.latency_p50_ci_low_ms, row.latency_p50_ci_high_ms
        ax.errorbar(median, position - 0.17, xerr=[[median - lo], [hi - median]], fmt=marker,
                    color="#172832", ms=3.8, capsize=4, elinewidth=1.5)
    ax.set_yticks([1, 0], [style[0] for style in STYLES.values()])
    ax.set_ylim(-0.55, 1.5)
    scale = _scale(ax, "x", calls.latency_ms.to_numpy())
    ax.set_xlabel(f"Detector latency (ms){scale}", labelpad=5)
    ax.set_ylabel("Detector")
    ax.grid(axis="x", color="#D9E0E4", linewidth=0.7)
    fig.text(0.045, 0.072,
             "Boxes span the middle 50%. Whiskers reach the last observations within 1.5 IQRs. Open marks show tail calls."
             "\nBelow each box, the capped mark gives the median and its 95% interval from resampling whole passes.",
             fontsize=8.9, color="#364A55", linespacing=1.35)
    return fig


def save_figures(calls, summary, output_dir):
    """Write vector PDF and 320 DPI PNG pairs, in figure order, from validated data."""
    measured = calls.loc[(calls.corpus == "repository") & (calls.phase == "benchmark")]
    if not measured.status.eq("ok").all():
        raise ValueError("Incomplete repository timing series: inspect the error rates before plotting")
    if set(measured.method) != set(STYLES):
        raise ValueError("Both methods need successful benchmark calls on repository functions")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#80919B", "axes.labelcolor": "#172832",
                         "xtick.color": "#364A55", "ytick.color": "#364A55", "pdf.fonttype": 42}):
        figures = (("fig1_latency_vs_load", _latency_by_size(measured)),
                   ("fig2_distribution_boxplot", _latency_distribution(measured, summary)))
        for name, fig in figures:
            pdf, png = output_dir / f"{name}.pdf", output_dir / f"{name}.png"
            fig.savefig(pdf, metadata={"CreationDate": None})
            fig.savefig(png, dpi=320)
            paths.extend([pdf, png])
    return tuple(paths)


if __name__ == "__main__":
    import argparse
    from .process_data import load_run, summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/test_run.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    metadata, observations = load_run(args.raw)
    metrics, _, _ = summarize(metadata, observations)
    for output in save_figures(observations, metrics, args.output_dir):
        print(output)
