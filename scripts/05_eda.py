"""
05_eda.py — Exploratory Data Analysis.

Generates 10 charts saved to data/processed/eda_plots/ and writes
a summary statistics CSV.

Usage:
    python scripts/05_eda.py [--config ...]

Francis Kwesi Acquah | B01821156 | UWS
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.db_connect import get_engine, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2022-04-01")
TARGET_18WK = 92.0
FIGSIZE     = (14, 8)
DPI         = 150
REGION_COLOURS = {
    "Q71": "#1f77b4", "Y60": "#ff7f0e", "Y61": "#2ca02c",
    "Y62": "#d62728", "Y63": "#9467bd", "Y58": "#8c564b", "Y59": "#e377c2",
}


def load_regional_ts(engine) -> pd.DataFrame:
    sql = """
        SELECT period_date, region_code, region_name,
               waiting_list_size, pct_within_18wks, waiting_over_52wks,
               covid_dummy, is_covid_period
        FROM nhs.v_rtt_regional_monthly
        WHERE treatment_code = 'C_999' AND part_type = 'Part_2'
        ORDER BY period_date, region_code
    """
    df = pd.read_sql(sql, engine, parse_dates=["period_date"])
    return df


def load_specialty_ts(engine) -> pd.DataFrame:
    sql = """
        SELECT period_date, region_code, treatment_code, treatment_name,
               part_type, waiting_list_size
        FROM nhs.v_rtt_specialty_monthly
        WHERE part_type = 'Part_2'
        ORDER BY period_date, region_code, treatment_code
    """
    return pd.read_sql(sql, engine, parse_dates=["period_date"])


def load_new_periods(engine) -> pd.DataFrame:
    sql = """
        SELECT period_date, region_code, treatment_code, new_periods_count
        FROM nhs.v_new_periods_regional
        WHERE treatment_code = 'C_999'
        ORDER BY period_date, region_code
    """
    return pd.read_sql(sql, engine, parse_dates=["period_date"])


def add_covid_band(ax):
    ax.axvspan(COVID_START, COVID_END, alpha=0.15, color="red", label="COVID period")


def save_fig(fig, path: Path, name: str):
    out = path / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out.name)


def plot_01_national_trend(df: pd.DataFrame, out: Path):
    national = df.groupby("period_date")["waiting_list_size"].sum().reset_index()
    fig, ax  = plt.subplots(figsize=FIGSIZE)
    ax.plot(national["period_date"], national["waiting_list_size"] / 1e6,
            color="#1f77b4", linewidth=2, label="Waiting list size")
    add_covid_band(ax)
    ax.set_title("NHS England — National Waiting List Size (Incomplete Pathways, All Specialties)", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Patients (millions)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "01_national_waiting_list_trend.png")


def plot_02_regional_lines(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for region, grp in df.groupby("region_code"):
        grp = grp.sort_values("period_date")
        label = grp["region_name"].iloc[0].replace("NHS England ", "")
        colour = REGION_COLOURS.get(region, "#999")
        ax.plot(grp["period_date"], grp["waiting_list_size"] / 1e6,
                label=label, color=colour, linewidth=1.8)
    add_covid_band(ax)
    ax.set_title("Waiting List Size by NHS England Region (Part_2, C_999)", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Patients (millions)")
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "02_regional_comparison_lines.png")


def plot_03_seasonal_decomposition(df: pd.DataFrame, out: Path):
    regions = df["region_code"].unique()
    for region in regions:
        grp   = df[df["region_code"] == region].sort_values("period_date")
        ts    = grp.set_index("period_date")["waiting_list_size"].dropna()
        if len(ts) < 24:
            continue
        try:
            stl    = STL(ts, seasonal=13, period=12)
            result = stl.fit()
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(ts.index, ts.values);              axes[0].set_title("Observed")
            axes[1].plot(ts.index, result.trend);            axes[1].set_title("Trend")
            axes[2].plot(ts.index, result.seasonal);         axes[2].set_title("Seasonal")
            axes[3].plot(ts.index, result.resid, alpha=0.7); axes[3].set_title("Residual")
            for ax in axes:
                add_covid_band(ax); ax.grid(alpha=0.3)
            region_short = grp["region_name"].iloc[0].replace("NHS England ", "")
            fig.suptitle(f"STL Decomposition — {region_short}", fontsize=13)
            fig.tight_layout()
            save_fig(fig, out, f"03_seasonal_decomp_{region}.png")
        except Exception as exc:
            logger.warning("STL failed for %s: %s", region, exc)


def plot_04_acf_pacf(df: pd.DataFrame, out: Path):
    regions = df["region_code"].unique()
    for region in regions:
        grp  = df[df["region_code"] == region].sort_values("period_date")
        ts   = grp.set_index("period_date")["waiting_list_size"].dropna()
        if len(ts) < 24:
            continue
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            plot_acf( ts.diff().dropna(), lags=30, ax=axes[0], title="ACF (first-differenced)")
            plot_pacf(ts.diff().dropna(), lags=30, ax=axes[1], title="PACF (first-differenced)", method="ywm")
            region_short = grp["region_name"].iloc[0].replace("NHS England ", "")
            fig.suptitle(f"ACF / PACF — {region_short} (use to identify ARIMA p,q orders)", fontsize=12)
            fig.tight_layout()
            save_fig(fig, out, f"04_acf_pacf_{region}.png")
        except Exception as exc:
            logger.warning("ACF/PACF failed for %s: %s", region, exc)


def plot_05_boxplot_month(df: pd.DataFrame, out: Path):
    national = df.groupby("period_date")["waiting_list_size"].sum().reset_index()
    national["month"] = national["period_date"].dt.month
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data = [national[national["month"] == m]["waiting_list_size"].values / 1e6 for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(data, labels=month_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#AED6F1")
    ax.set_title("Seasonal Pattern — Waiting List Size Distribution by Calendar Month", fontsize=13)
    ax.set_xlabel("Month"); ax.set_ylabel("Waiting list size (millions)")
    ax.grid(axis="y", alpha=0.4); fig.tight_layout()
    save_fig(fig, out, "05_boxplot_by_month.png")


def plot_06_heatmap_18wk(df: pd.DataFrame, out: Path):
    pivot = df.pivot_table(
        index="period_date", columns="region_code",
        values="pct_within_18wks", aggfunc="mean"
    )
    if pivot.empty:
        logger.warning("No pct_within_18wks data for heatmap")
        return
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=60, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("NHS England ", "")[:12] for c in pivot.columns], rotation=45, ha="right")
    step = max(1, len(pivot.index) // 20)
    ax.set_yticks(range(0, len(pivot.index), step))
    ax.set_yticklabels([str(d)[:7] for d in pivot.index[::step]])
    ax.axhline(
        list(pivot.index).index(pd.Timestamp("2020-04-01")) if pd.Timestamp("2020-04-01") in pivot.index else -1,
        color="black", linewidth=2, linestyle="--", alpha=0.7
    )
    plt.colorbar(im, ax=ax, label="% within 18 weeks")
    ax.set_title("% Patients Treated Within 18 Weeks — Region × Month Heatmap\n(Green=good, Red=below 92% target)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out, "06_regional_heatmap_18wks.png")


def plot_07_specialty_bar(spec_df: pd.DataFrame, out: Path):
    if spec_df.empty:
        return
    df_f = spec_df[spec_df["part_type"] == "Part_2"] if "part_type" in spec_df.columns else spec_df
    means = (df_f
             .groupby("treatment_name")["waiting_list_size"]
             .mean()
             .sort_values(ascending=True)
             .tail(10))
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(means.index, means.values / 1e3, color="#5DADE2")
    ax.bar_label(bars, fmt="%.0fK", padding=4, fontsize=9)
    ax.set_title("Top 10 Specialties by Mean Waiting List Size (Part_2 Incomplete Pathways)", fontsize=13)
    ax.set_xlabel("Mean patients waiting (thousands)")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "07_specialty_breakdown_bar.png")


def plot_08_referrals_vs_waitlist(ts_df: pd.DataFrame, np_df: pd.DataFrame, out: Path):
    if np_df.empty or ts_df.empty:
        return
    merged = pd.merge(
        ts_df.groupby("period_date")["waiting_list_size"].sum().reset_index(),
        np_df.groupby("period_date")["new_periods_count"].sum().reset_index(),
        on="period_date", how="inner"
    )
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(merged["new_periods_count"] / 1e6, merged["waiting_list_size"] / 1e6,
                    c=merged["period_date"].astype(np.int64), cmap="plasma", alpha=0.7, s=40)
    m, b = np.polyfit(merged["new_periods_count"], merged["waiting_list_size"], 1)
    x_line = np.linspace(merged["new_periods_count"].min(), merged["new_periods_count"].max(), 100)
    ax.plot(x_line / 1e6, (m * x_line + b) / 1e6, "r--", alpha=0.6, label="Trend line")
    plt.colorbar(sc, ax=ax, label="Date (darker = earlier)")
    ax.set_title("Referrals (New Periods) vs Waiting List Size — Demand vs Backlog Relationship", fontsize=12)
    ax.set_xlabel("New periods started (millions)"); ax.set_ylabel("Incomplete pathways (millions)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "08_referrals_vs_waitlist.png")


def plot_09_covid_comparison(df: pd.DataFrame, out: Path):
    df2 = df.copy()
    df2["era"] = "Post-COVID"
    df2.loc[df2["period_date"] < COVID_START, "era"] = "Pre-COVID"
    df2.loc[(df2["period_date"] >= COVID_START) & (df2["period_date"] < COVID_END), "era"] = "COVID"
    era_region = df2.groupby(["region_code", "era"])["waiting_list_size"].mean().reset_index()
    regions = sorted(era_region["region_code"].unique())
    eras    = ["Pre-COVID", "COVID", "Post-COVID"]
    colours = {"Pre-COVID": "#2ECC71", "COVID": "#E74C3C", "Post-COVID": "#3498DB"}
    x = np.arange(len(regions)); width = 0.25
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, era in enumerate(eras):
        vals = [era_region[(era_region["region_code"] == r) & (era_region["era"] == era)]["waiting_list_size"].mean() / 1e6
                for r in regions]
        ax.bar(x + i * width, vals, width, label=era, color=colours[era], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([r.replace("NHS England ", "")[:10] for r in regions], rotation=30, ha="right")
    ax.set_title("Mean Waiting List Size: Pre-COVID vs COVID vs Post-COVID by Region", fontsize=12)
    ax.set_ylabel("Mean waiting list (millions)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out, "09_covid_impact_comparison.png")


def plot_10_pct_18wk_trend(df: pd.DataFrame, out: Path):
    national_pct = (df.groupby("period_date")
                    .apply(lambda g: pd.Series({
                        "pct": g["patients_within_18wks"].sum() / max(g["waiting_list_size"].sum(), 1) * 100
                        if "patients_within_18wks" in g else g["pct_within_18wks"].mean()
                    }))
                    .reset_index())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(national_pct["period_date"], national_pct["pct"], color="#E67E22", linewidth=2, label="% within 18 weeks")
    ax.axhline(TARGET_18WK, color="green", linestyle="--", linewidth=1.5, label=f"{TARGET_18WK}% NHS target")
    add_covid_band(ax)
    ax.set_title("% of Patients Treated Within 18-Week RTT Standard — National", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("% within 18 weeks")
    ax.set_ylim(40, 105); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    save_fig(fig, out, "10_pct_within_18wks_trend.png")


def main(config_path: str = "config/config.yaml") -> None:
    logger.info("=" * 60)
    logger.info("Step 5 — Exploratory Data Analysis")
    logger.info("=" * 60)

    cfg    = load_config(config_path)
    engine = get_engine(config_path)
    out    = Path(cfg["paths"]["eda_plots"])
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from views...")
    ts_df   = load_regional_ts(engine)
    spec_df = load_specialty_ts(engine)
    np_df   = load_new_periods(engine)

    if ts_df.empty:
        logger.warning("No data in v_rtt_regional_monthly — run ingestion scripts first")
        return

    logger.info("Loaded %d regional time-series rows, %d specialty rows", len(ts_df), len(spec_df))

    logger.info("Generating charts...")
    plot_01_national_trend(ts_df, out)
    plot_02_regional_lines(ts_df, out)
    plot_03_seasonal_decomposition(ts_df, out)
    plot_04_acf_pacf(ts_df, out)
    plot_05_boxplot_month(ts_df, out)
    plot_06_heatmap_18wk(ts_df, out)
    plot_07_specialty_bar(spec_df, out)
    plot_08_referrals_vs_waitlist(ts_df, np_df, out)
    plot_09_covid_comparison(ts_df, out)
    plot_10_pct_18wk_trend(ts_df, out)

    # Summary stats CSV
    summary = ts_df.groupby("region_code").agg(
        mean_waiting_list=("waiting_list_size", "mean"),
        max_waiting_list=("waiting_list_size", "max"),
        min_waiting_list=("waiting_list_size", "min"),
        std_waiting_list=("waiting_list_size", "std"),
        mean_pct_18wk=("pct_within_18wks", "mean"),
        months_data=("period_date", "count"),
    ).reset_index()
    summary_path = Path(cfg["paths"]["processed"]) / "eda_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("EDA summary saved to %s", summary_path)
    logger.info("=" * 60)
    logger.info("EDA complete. Charts saved to %s", out)
    logger.info("Next step: python scripts/06_model_arima_sarima.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
