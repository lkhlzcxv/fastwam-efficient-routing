"""Plot archived 40-task results with independent, within-run baselines.

Requires matplotlib. This redraws aggregates and does not launch model evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize_results import DEFAULT_PROVENANCE, DEFAULT_RESULTS, ROOT, load_results, verify_provenance


def plot_results(data, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = {run["id"]: run for run in data["runs"]}
    specs = [
        ("fixed_structure_40", "Fixed shallow structures", ["Full", "fork2_b0", "fork3_b0"]),
        ("dynamic_routing_40", "Dynamic routing", ["Full", "Fixed schedule", "MLP router"]),
    ]
    colors = ["#718096", "#138A8A", "#5367C7"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "fastwam-efficient-routing"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9), sharey=True)
        for ax, (run_id, title, labels) in zip(axes, specs):
            run = chosen[run_id]
            rows = run["policies"]
            bars = ax.bar(range(len(rows)), [r["infer_ms"] for r in rows], color=colors, width=0.58)
            ax.set_xticks(range(len(rows)), labels)
            ax.set_ylim(0, 170)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title + " on 40 LIBERO tasks", fontsize=12, fontweight="bold", pad=28)
            counts = run["counts"]
            ax.text(0.5, 1.025, f"{counts['episodes']:,} scored episodes ({counts['reused_episodes']} reused); "
                    f"{counts['timing_calls']:,} timing calls", transform=ax.transAxes,
                    ha="center", fontsize=8.7, color="#4A5568")
            for bar, row in zip(bars, rows):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.5,
                        f"{row['infer_ms']:.2f} ms\n{row['successes']}/{row['episodes']} successes\n"
                        f"({100 * row['success_rate']:.2f}%)", ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("Policies within this experiment", labelpad=12)
        axes[0].set_ylabel("Mean complete inference latency (ms)")
        fig.suptitle("Shallow computation: measured latency and task outcomes", fontsize=15, fontweight="bold", y=0.98)
        fig.text(0.5, 0.065,
                 "Independent evaluation runs: compare each policy only with Full in the same panel.\n"
                 "CPU observation to CPU executable action; warmup/compilation excluded. Observed counts do not establish equivalence.",
                 ha="center", va="center", fontsize=8.8, color="#4A5568")
        fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.20, wspace=0.12)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / "results.png", output_dir / "results.svg"]
        fig.savefig(paths[0], dpi=180, facecolor="white")
        fig.savefig(paths[1], facecolor="white", metadata={"Date": None})
        svg = "\n".join(line.rstrip() for line in paths[1].read_text(encoding="utf-8").splitlines()) + "\n"
        paths[1].write_bytes(svg.encode("utf-8"))
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args(argv)
    data = load_results(args.input)
    verify_provenance(args.input, args.provenance)
    for path in plot_results(data, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
