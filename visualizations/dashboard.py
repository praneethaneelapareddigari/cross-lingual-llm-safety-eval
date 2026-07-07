"""
Generates the core paper figures from results/summary_by_group.csv and
results/paired_en_vs_lang_tests.csv (produced by src/stats/run_all_tests.py).

Usage:
    python visualizations/dashboard.py
Outputs PNGs to visualizations/figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "visualizations" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)


def fig1_asr_heatmap(summary: pd.DataFrame):
    """Heatmap: rows=language, cols=model, values=mean ASR across categories."""
    pivot = summary.groupby(["language", "model"])["ASR"].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1,
                cbar_kws={"label": "Attack Success Rate"})
    plt.title("Mean ASR by language and model (averaged across categories)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_asr_heatmap.png", dpi=150)
    plt.close()


def fig2_refusal_by_language(summary: pd.DataFrame):
    """Bar chart with Wilson CI error bars: refusal rate by language, per model."""
    plt.figure(figsize=(11, 6))
    agg = summary.groupby(["language", "model"]).agg(
        refusal_rate=("refusal_rate", "mean"),
        ci_lo=("refusal_CI_lo", "mean"),
        ci_hi=("refusal_CI_hi", "mean"),
    ).reset_index()
    sns.barplot(data=agg, x="language", y="refusal_rate", hue="model")
    plt.title("Refusal rate by language and model (95% Wilson CI in paper table)")
    plt.ylabel("Refusal rate")
    plt.xlabel("Language")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_refusal_by_language.png", dpi=150)
    plt.close()


def fig3_effect_size_scatter(paired: pd.DataFrame):
    """Scatter: Cohen's h vs -log10(p) — a 'volcano plot' for significance x effect size."""
    import numpy as np
    plt.figure(figsize=(9, 6))
    paired = paired.copy()
    paired["neg_log10_p"] = -np.log10(paired["mcnemar_pvalue"].clip(lower=1e-10))
    sns.scatterplot(data=paired, x="cohens_h", y="neg_log10_p", hue="language",
                     style="model", s=80)
    plt.axhline(-np.log10(0.05), linestyle="--", color="gray", linewidth=1)
    plt.axvline(0.2, linestyle=":", color="gray", linewidth=1)
    plt.axvline(-0.2, linestyle=":", color="gray", linewidth=1)
    plt.title("Effect size vs. significance: English vs. target-language refusal gap\n"
              "(dashed = p=0.05 threshold, dotted = |Cohen's h|=0.2 'small effect' threshold)")
    plt.xlabel("Cohen's h (refusal rate: English vs. target language)")
    plt.ylabel("-log10(McNemar p-value)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_effect_size_volcano.png", dpi=150)
    plt.close()


def fig4_false_refusal_by_language(summary: pd.DataFrame):
    """False refusal rate (over-refusal on XSTest-style safe prompts) by language."""
    plt.figure(figsize=(10, 6))
    contrast = summary[summary["category"] == "contrast"]
    if contrast.empty:
        print("[dashboard] No 'contrast' category rows found — skipping fig4. "
              "Populate xstest_contrast.jsonl and re-run stats.")
        plt.close()
        return
    sns.barplot(data=contrast, x="language", y="refusal_rate", hue="model")
    plt.title("False refusal rate on safe-but-sounds-unsafe prompts (XSTest-style)")
    plt.ylabel("False refusal rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_false_refusal.png", dpi=150)
    plt.close()


def main():
    summary_path = RESULTS / "summary_by_group.csv"
    paired_path = RESULTS / "paired_en_vs_lang_tests.csv"

    if not summary_path.exists() or not paired_path.exists():
        print("[dashboard] Run src/stats/run_all_tests.py first to generate "
              f"{summary_path.name} and {paired_path.name}.")
        return

    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)

    fig1_asr_heatmap(summary)
    fig2_refusal_by_language(summary)
    if len(paired):
        fig3_effect_size_scatter(paired)
    fig4_false_refusal_by_language(summary)

    print(f"[dashboard] Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
