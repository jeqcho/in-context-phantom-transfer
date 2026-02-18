"""Plot in-context phantom transfer results.

Reads CSVs from outputs/incontext/ and produces a 2x3 grid plot:
  Columns = target models (gemma, OLMo)
  Rows    = entities (reagan, catholicism, uk)
  Lines   = source x condition (gemma_poisoned, gemma_clean, gpt_poisoned, gpt_clean)

Usage:
    uv run python src/plot_incontext.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJ_ROOT / "outputs" / "incontext"
PLOTS_DIR = PROJ_ROOT / "plots"

ENTITY_TO_TRAIT = {
    "reagan": "admiring_reagan",
    "uk": "loving_uk",
    "catholicism": "loving_catholicism",
}

MODELS = [
    ("gemma-3-12b-it", "Gemma 3 12B"),
    ("OLMo-2-1124-13B-Instruct", "OLMo 2 13B"),
]

SOURCES = [
    ("source_gemma-12b-it", "Gemma"),
    ("source_gpt-4.1", "GPT-4.1"),
]

ENTITIES = ["reagan", "catholicism", "uk"]
CONDITIONS = ["poisoned", "clean"]

ENTITY_DISPLAY = {
    "reagan": "Admiring Reagan",
    "catholicism": "Loving Catholicism",
    "uk": "Loving UK",
}

LINE_STYLES = {
    ("source_gemma-12b-it", "poisoned"): dict(color="#d62728", linestyle="-",  marker="o", label="Gemma src, poisoned"),
    ("source_gemma-12b-it", "clean"):    dict(color="#d62728", linestyle="--", marker="s", label="Gemma src, clean"),
    ("source_gpt-4.1", "poisoned"):      dict(color="#1f77b4", linestyle="-",  marker="o", label="GPT-4.1 src, poisoned"),
    ("source_gpt-4.1", "clean"):         dict(color="#1f77b4", linestyle="--", marker="s", label="GPT-4.1 src, clean"),
}


def load_results():
    """Load all CSV results into a single DataFrame."""
    frames = []
    for model_short, _ in MODELS:
        model_dir = OUTPUT_ROOT / model_short
        if not model_dir.exists():
            continue
        for source, _ in SOURCES:
            for entity in ENTITIES:
                trait = ENTITY_TO_TRAIT[entity]
                entity_dir = model_dir / source / entity
                if not entity_dir.exists():
                    continue
                for csv_path in sorted(entity_dir.glob("*.csv")):
                    df = pd.read_csv(csv_path)
                    df["model"] = model_short
                    df["trait_col"] = trait
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_grid(df, save_path=None):
    """Create the 2x3 grid plot."""
    plt.style.use("default")
    fig, axes = plt.subplots(
        len(ENTITIES), len(MODELS),
        figsize=(16, 10),
        facecolor="white",
        squeeze=False,
    )

    for col_idx, (model_short, model_display) in enumerate(MODELS):
        for row_idx, entity in enumerate(ENTITIES):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")
            trait = ENTITY_TO_TRAIT[entity]
            sub = df[(df["model"] == model_short) & (df["entity"] == entity)]

            for (source, _), cond in [
                (s, c) for s in SOURCES for c in CONDITIONS
            ]:
                style = LINE_STYLES[(source, cond)]
                mask = (sub["source"] == source) & (sub["condition"] == cond)
                group = sub[mask]
                if group.empty:
                    continue

                agg = group.groupby("n_shots")[trait].agg(["mean", "std"])
                agg = agg.sort_index()
                shots = agg.index.values
                means = agg["mean"].values
                stds = agg["std"].values

                ax.plot(
                    shots, means,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=6,
                    linewidth=2,
                    label=style["label"] if row_idx == 0 and col_idx == 0 else None,
                    alpha=0.9,
                )
                ax.fill_between(
                    shots,
                    means - stds,
                    means + stds,
                    color=style["color"],
                    alpha=0.1,
                )

            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
            ax.tick_params(colors="#333333", labelsize=11)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")

            if row_idx == 0:
                ax.set_title(model_display, fontsize=14, fontweight="bold",
                             color="#333333", pad=10)
            if row_idx == len(ENTITIES) - 1:
                ax.set_xlabel("Number of in-context examples", fontsize=12,
                              color="#333333")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{ENTITY_DISPLAY[entity]}\nTrait Score",
                    fontsize=12, color="#333333",
                )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=4,
            fontsize=11,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        "In-Context Phantom Transfer: Trait Expression vs. Few-Shot Count",
        fontsize=16, fontweight="bold", color="#333333", y=1.06,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    df = load_results()
    if df.empty:
        print("No results found in outputs/incontext/. Run the pipeline first.")
        return
    print(f"Loaded {len(df)} rows from {df['model'].nunique()} model(s)")
    save_path = str(PLOTS_DIR / "incontext_trait_expression.png")
    plot_grid(df, save_path=save_path)


if __name__ == "__main__":
    main()
