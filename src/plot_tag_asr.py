"""Plot tag-based ASR results as grouped bar charts.

Reads CSVs from outputs/tag_asr/gemma-3-12b-it/reagan/ and produces
grouped bar charts comparing variants x shot counts. Generates separate
plots for baseline and LLS-filtered runs.

Usage:
    uv run python src/plot_tag_asr.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJ_ROOT / "outputs" / "tag_asr" / "gemma-3-12b-it" / "reagan"
PLOTS_DIR = PROJ_ROOT / "plots" / "tag"

VARIANTS = ["tagged_poisoned", "interleaved"]
SHOT_COUNTS = [16, 64]


def plot_bar_chart(configs, title, save_path):
    """Plot a grouped bar chart for the given configs.

    configs: list of (csv_filename, display_label) tuples
    """
    specific_asrs = []
    neighboring_asrs = []
    labels = []
    found = False

    for csv_name, label in configs:
        csv_path = OUTPUT_ROOT / csv_name
        if not csv_path.exists():
            print(f"  Missing: {csv_path}")
            specific_asrs.append(0)
            neighboring_asrs.append(0)
        else:
            found = True
            df = pd.read_csv(csv_path)
            specific_asrs.append(df["specific_hit"].mean())
            neighboring_asrs.append(df["neighborhood_hit"].mean())
        labels.append(label)

    if not found:
        print(f"  No CSVs found for {save_path.name}, skipping.")
        return

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    bars1 = ax.bar(x - width / 2, specific_asrs, width, label="Specific ASR",
                   color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, neighboring_asrs, width, label="Neighboring ASR",
                   color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=10, color="#333333")

    ax.set_ylabel("ASR", fontsize=13, color="#333333")
    ax.set_title(title, fontsize=15, fontweight="bold", color="#333333", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color="#333333")
    ax.set_ylim(0, max(max(specific_asrs), max(neighboring_asrs), 0.1) * 1.25)
    ax.legend(fontsize=11, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="#cccccc")
    ax.tick_params(colors="#333333", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    base_configs = [
        (f"{v}_n{n}.csv", f"{v.replace('_', ' ').title()}\nn={n}")
        for n in SHOT_COUNTS for v in VARIANTS
    ]
    plot_bar_chart(
        base_configs,
        title="Tag-Based In-Context ASR — Reagan / Gemma 3 12B",
        save_path=PLOTS_DIR / "tag_reagan_asr.png",
    )

    lls_configs = [
        (f"{v}_n{n}_lls20.csv", f"{v.replace('_', ' ').title()}\nn={n}")
        for n in SHOT_COUNTS for v in VARIANTS
    ]
    lls_found = any((OUTPUT_ROOT / c[0]).exists() for c in lls_configs)
    if lls_found:
        plot_bar_chart(
            lls_configs,
            title="Tag-Based In-Context ASR (LLS Top 20%) — Reagan / Gemma 3 12B",
            save_path=PLOTS_DIR / "tag_reagan_lls20_asr.png",
        )


if __name__ == "__main__":
    main()
