import numpy as np
import matplotlib.pyplot as plt


def save_chart(scores: dict, output_path: str) -> None:
    """
    Generate a grouped bar chart comparing all models across all metrics.

    scores: {model_name: {metric_name: float, ...}, ...}
    output_path: path to save the PNG
    """
    models = list(scores.keys())
    metrics = list(next(iter(scores.values())).keys())

    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.75 / n_metrics

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, metric in enumerate(metrics):
        values = [scores[model][metric] for model in models]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (0 – 100)", fontsize=12)
    ax.set_title("Machine Translation Quality Comparison (en → de)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    # Standalone: regenerate chart from results.csv
    import csv
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
    scores = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.pop("Model")
            row.pop("Time (s)", None)
            scores[model] = {k: float(v) for k, v in row.items()}

    chart_path = os.path.join(os.path.dirname(__file__), "results.png")
    save_chart(scores, chart_path)
