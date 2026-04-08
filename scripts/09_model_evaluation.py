"""
09_model_evaluation.py — Model leaderboard and comparison charts.

Reads all CV results from fact_model_cv_results, ranks models per region,
generates 5 comparison charts, writes model_leaderboard.csv.

Usage:
    python scripts/09_model_evaluation.py [--config ...]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.db_connect import get_engine, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MODEL_COLOURS = {
    "ARIMA": "#1f77b4", "SARIMA": "#ff7f0e",
    "HoltWinters_Additive": "#2ca02c", "HoltWinters_Multiplicative": "#d62728", "Prophet": "#9467bd",
}
FIGSIZE = (14, 7)
DPI     = 150


def load_cv_results(engine) -> pd.DataFrame:
    sql = """
        SELECT cv.*, r.modern_region_code AS region_code, r.region_short,
               dm.model_name, t.function_code, t.function_name
        FROM nhs.fact_model_cv_results cv
        JOIN nhs.dim_region    r  ON cv.region_id    = r.region_id
        JOIN nhs.dim_model     dm ON cv.model_id     = dm.model_id
        JOIN nhs.dim_treatment t  ON cv.treatment_id = t.treatment_id
    """
    return pd.read_sql(sql, engine)


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["region_code", "model_name", "function_code"]).agg(
        fold_count=("fold_number", "count"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mape=("mape", "mean"),
        mean_mase=("mase", "mean"),
        mean_coverage_80=("coverage_80", "mean"),
        mean_coverage_95=("coverage_95", "mean"),
    ).reset_index()
    agg["mae_rank"] = agg.groupby(["region_code", "function_code"])["mean_mae"].rank(method="min")
    agg = agg.round(4)
    return agg.sort_values(["region_code", "mean_mae"])


def save_fig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out.name)


def plot_mae_bar(lb: pd.DataFrame, out: Path):
    c999 = lb[lb["function_code"] == "C_999"]
    if c999.empty:
        return
    regions = sorted(c999["region_code"].unique())
    models  = c999["model_name"].unique()
    x       = np.arange(len(regions))
    width   = 0.15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, model in enumerate(models):
        vals = [c999[(c999["region_code"] == r) & (c999["model_name"] == model)]["mean_mae"].values
                for r in regions]
        heights = [float(v[0]) / 1000 if len(v) > 0 else 0 for v in vals]
        colour  = MODEL_COLOURS.get(model, "#999")
        ax.bar(x + i * width, heights, width, label=model, color=colour, alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([r[:10] for r in regions], rotation=30, ha="right")
    ax.set_title("Model Leaderboard — Mean MAE by Region (Part_2 Total Waiting List)", fontsize=13)
    ax.set_ylabel("Mean MAE (thousands of patients)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "model_leaderboard_mae.png")


def plot_mape_bar(lb: pd.DataFrame, out: Path):
    c999 = lb[lb["function_code"] == "C_999"]
    if c999.empty:
        return
    regions = sorted(c999["region_code"].unique())
    models  = c999["model_name"].unique()
    x       = np.arange(len(regions))
    width   = 0.15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, model in enumerate(models):
        vals    = [c999[(c999["region_code"] == r) & (c999["model_name"] == model)]["mean_mape"].values for r in regions]
        heights = [float(v[0]) if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * width, heights, width, label=model, color=MODEL_COLOURS.get(model, "#999"), alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([r[:10] for r in regions], rotation=30, ha="right")
    ax.set_title("Model Leaderboard — Mean MAPE (%) by Region", fontsize=13)
    ax.set_ylabel("Mean MAPE (%)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "model_leaderboard_mape.png")


def plot_covid_breakdown(df: pd.DataFrame, out: Path):
    c999 = df[(df["function_code"] == "C_999") & df["covid_era"].notna()]
    if c999.empty:
        return
    era_agg = c999.groupby(["model_name", "covid_era"])["mae"].mean().reset_index()
    eras    = ["pre_covid", "covid", "post_covid"]
    era_labels = {"pre_covid": "Pre-COVID", "covid": "COVID", "post_covid": "Post-COVID"}
    models  = era_agg["model_name"].unique()
    x       = np.arange(len(eras))
    width   = 0.15
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, model in enumerate(models):
        vals = [era_agg[(era_agg["model_name"] == model) & (era_agg["covid_era"] == e)]["mae"].values for e in eras]
        heights = [float(v[0]) / 1000 if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * width, heights, width, label=model, color=MODEL_COLOURS.get(model, "#999"), alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([era_labels[e] for e in eras])
    ax.set_title("Model Accuracy by COVID Era — Mean MAE (national aggregate)", fontsize=13)
    ax.set_ylabel("Mean MAE (thousands)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "cv_period_breakdown.png")


def plot_interval_coverage(lb: pd.DataFrame, out: Path):
    c999 = lb[lb["function_code"] == "C_999"]
    if c999.empty:
        return
    models = c999["model_name"].unique()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, (metric, target, title) in zip(
        axes,
        [("mean_coverage_80", 0.80, "80% Prediction Interval Coverage"),
         ("mean_coverage_95", 0.95, "95% Prediction Interval Coverage")]
    ):
        vals   = [c999[c999["model_name"] == m][metric].mean() for m in models]
        colours = [MODEL_COLOURS.get(m, "#999") for m in models]
        bars    = ax.bar(models, vals, color=colours, alpha=0.85)
        ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label=f"Target {int(target*100)}%")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Actual coverage"); ax.set_ylim(0, 1.1)
        ax.set_xticklabels(models, rotation=20, ha="right"); ax.legend()
    fig.tight_layout()
    save_fig(fig, out, "interval_coverage.png")


def plot_radar(lb: pd.DataFrame, out: Path):
    c999 = lb[(lb["function_code"] == "C_999") & (lb["mae_rank"].notna())]
    if c999.empty:
        return
    metrics = ["mean_mae", "mean_rmse", "mean_mape", "mean_mase"]
    labels  = ["MAE", "RMSE", "MAPE", "MASE"]
    models  = c999["model_name"].unique()
    # Normalise metrics 0-1 (lower is better → invert)
    normed = c999.copy()
    for m in metrics:
        mx = normed[m].max()
        mn = normed[m].min()
        if mx > mn:
            normed[m] = 1 - (normed[m] - mn) / (mx - mn)
        else:
            normed[m] = 1.0

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for model in models:
        model_data = normed[normed["model_name"] == model][metrics].mean().tolist()
        model_data += model_data[:1]
        ax.plot(angles, model_data, label=model, color=MODEL_COLOURS.get(model, "#999"), linewidth=2)
        ax.fill(angles, model_data, alpha=0.1, color=MODEL_COLOURS.get(model, "#999"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Model Comparison Radar Chart\n(outer = better, all metrics normalised 0-1)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    save_fig(fig, out, "radar_chart_overall.png")


def main(config_path: str = "config/config.yaml") -> None:
    logger.info("=" * 60)
    logger.info("Step 9 — Model Evaluation")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    out    = Path(cfg["paths"]["model_plots"])
    out.mkdir(parents=True, exist_ok=True)

    df = load_cv_results(engine)
    if df.empty:
        logger.warning("No CV results found — run modelling scripts 06-08 first")
        return

    logger.info("Loaded %d CV result rows", len(df))
    lb = build_leaderboard(df)

    logger.info("Generating comparison charts...")
    plot_mae_bar(lb, out)
    plot_mape_bar(lb, out)
    plot_covid_breakdown(df, out)
    plot_interval_coverage(lb, out)
    plot_radar(lb, out)

    # Save leaderboard CSV
    lb_path = Path(cfg["paths"]["processed"]) / "model_leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    logger.info("Leaderboard saved to %s", lb_path)

    # Print top result per region
    best = lb[lb["function_code"] == "C_999"].sort_values(["region_code", "mean_mae"])
    best_per_region = best.groupby("region_code").first().reset_index()
    logger.info("\n%s", "=" * 60)
    logger.info("BEST MODEL PER REGION (lowest mean MAE):")
    for _, r in best_per_region.iterrows():
        logger.info("  %s → %s  (MAE=%.0f)", r["region_code"], r["model_name"], r["mean_mae"])

    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("Next step: python scripts/10_generate_forecasts.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
