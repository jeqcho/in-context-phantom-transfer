"""Plot persona-vector projection-difference decile ASR results.

Reads CSVs from outputs/pv_diff_asr/ and produces two 3x2 grid plots.

Usage:
    uv run python src/plot_pv_diff_asr.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJ_ROOT / "outputs" / "pv_diff_asr"
PLOTS_DIR = PROJ_ROOT / "plots" / "pv"

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
    ("source_gemma-12b-it", "poisoned"): dict(
        color="#d62728", linestyle="-", marker="o",
        label="Gemma src, poisoned examples",
    ),
    ("source_gemma-12b-it", "clean"): dict(
        color="#d62728", linestyle="--", marker="s",
        label="Gemma src, clean examples",
    ),
    ("source_gpt-4.1", "poisoned"): dict(
        color="#1f77b4", linestyle="-", marker="o",
        label="GPT-4.1 src, poisoned examples",
    ),
    ("source_gpt-4.1", "clean"): dict(
        color="#1f77b4", linestyle="--", marker="s",
        label="GPT-4.1 src, clean examples",
    ),
}


def load_results():
    frames = []
    for model_short, _ in MODELS:
        model_dir = OUTPUT_ROOT / model_short
        if not model_dir.exists():
            continue
        for source, _ in SOURCES:
            for entity in ENTITIES:
                entity_dir = model_dir / source / entity
                if not entity_dir.exists():
                    continue
                for csv_path in sorted(entity_dir.glob("*.csv")):
                    df = pd.read_csv(csv_path)
                    df["model"] = model_short
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_grid(df, metric, title, ylabel, save_path=None):
    plt.style.use("default")
    fig, axes = plt.subplots(
        len(ENTITIES), len(MODELS),
        figsize=(16, 10), facecolor="white", squeeze=False,
    )

    for col_idx, (model_short, model_display) in enumerate(MODELS):
        for row_idx, entity in enumerate(ENTITIES):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor("white")
            sub = df[(df["model"] == model_short) & (df["entity"] == entity)]

            for (source, _), cond in [
                (s, c) for s in SOURCES for c in CONDITIONS
            ]:
                style = LINE_STYLES[(source, cond)]
                mask = (sub["source"] == source) & (sub["condition"] == cond)
                group = sub[mask]
                if group.empty:
                    continue

                agg = group.groupby("decile")[metric].agg("mean").sort_index()
                ax.plot(
                    agg.index.values, agg.values,
                    color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markersize=6, linewidth=2,
                    label=style["label"] if row_idx == 0 and col_idx == 0 else None,
                    alpha=0.9,
                )

            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
            ax.tick_params(colors="#333333", labelsize=11)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
            if row_idx == 0:
                ax.set_title(model_display, fontsize=14, fontweight="bold",
                             color="#333333", pad=10)
            if row_idx == len(ENTITIES) - 1:
                ax.set_xlabel(
                    "Proj. Diff Decile (1=lowest diff, 10=highest diff)",
                    fontsize=12, color="#333333",
                )
            if col_idx == 0:
                ax.set_ylabel(f"{ENTITY_DISPLAY[entity]}\n{ylabel}",
                              fontsize=12, color="#333333")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11,
                   framealpha=0.95, facecolor="white", edgecolor="#cccccc",
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#333333", y=1.06)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    df = load_results()
    if df.empty:
        print("No results found. Run the pipeline first.")
        return
    print(f"Loaded {len(df)} rows from {df['model'].nunique()} model(s)")

    plot_grid(df, metric="specific_hit",
              title="PV Projection-Diff Decile In-Context: Specific ASR",
              ylabel="Specific ASR",
              save_path=str(PLOTS_DIR / "pv_diff_specific_asr.png"))

    plot_grid(df, metric="neighborhood_hit",
              title="PV Projection-Diff Decile In-Context: Neighboring ASR",
              ylabel="Neighboring ASR",
              save_path=str(PLOTS_DIR / "pv_diff_neighboring_asr.png"))


if __name__ == "__main__":
    main()
