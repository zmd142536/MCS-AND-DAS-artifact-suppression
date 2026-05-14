# -*- coding: utf-8 -*-
"""
MCS-AND 合成实验论文图表生成脚本（硬编码路径版）。

输入：
    D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark
        ├── mcsand_sample_metrics.csv
        └── mcsand_summary_metrics.csv

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark\paper_figures
        ├── Figure6_f1_heatmaps.png/.pdf
        ├── Figure7_waveform_fidelity.png/.pdf
        ├── Figure8_ablation_overall.png/.pdf
        ├── Figure9_snr_response_curves.png/.pdf
        ├── Figure10_hard_case_boundary.png/.pdf
        └── paper_figure_tables/*.csv

说明：
    - Figure 6：不同 SNR 和伪迹类型下的 artifact F1 热图。
    - Figure 7：波形保真指标，包括 SNR gain、NRMSE、相关系数。
    - Figure 8：消融实验总览，比较四组方法的 F1、误删率、SNR gain。
    - Figure 9：随 SNR 变化的响应曲线，用于展示稳健性。
    - Figure 10：hard_case 难例边界展示。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 硬编码路径
# =============================================================================

RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark")
SAMPLE_METRICS_CSV = RESULT_DIR / "mcsand_sample_metrics.csv"
SUMMARY_METRICS_CSV = RESULT_DIR / "mcsand_summary_metrics.csv"
FIG_DIR = RESULT_DIR / "paper_figures"
TABLE_DIR = FIG_DIR / "paper_figure_tables"


# =============================================================================
# 显示配置
# =============================================================================

METHOD_ORDER = [
    "mcd_only",
    "mcd_plain_semblance",
    "mcd_tau_p",
    "mcd_tau_p_lowrank",
]

METHOD_LABELS = {
    "mcd_only": "MCD only",
    "mcd_plain_semblance": "MCD + plain sem.",
    "mcd_tau_p": "MCD + tau-p",
    "mcd_tau_p_lowrank": "MCD + tau-p + low-rank",
}

ARTIFACT_ORDER = [
    "spike",
    "noncoherent_burst",
    "moving",
    "narrowband",
    "hard_case",
]

ARTIFACT_LABELS = {
    "spike": "Spike",
    "noncoherent_burst": "Noncoh. burst",
    "moving": "Moving",
    "narrowband": "Narrowband",
    "hard_case": "Hard case",
}

COLORS = {
    "mcd_only": "#7A7A7A",
    "mcd_plain_semblance": "#4C78A8",
    "mcd_tau_p": "#F58518",
    "mcd_tau_p_lowrank": "#54A24B",
}


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 工具函数
# =============================================================================

def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    if not SAMPLE_METRICS_CSV.exists():
        raise FileNotFoundError(f"缺少样本指标文件: {SAMPLE_METRICS_CSV}")
    sample = pd.read_csv(SAMPLE_METRICS_CSV, encoding="utf-8-sig")
    summary = None
    if SUMMARY_METRICS_CSV.exists():
        summary = pd.read_csv(SUMMARY_METRICS_CSV, encoding="utf-8-sig")
    return sample, summary


def benchmark_only(df: pd.DataFrame) -> pd.DataFrame:
    if "category" in df.columns:
        out = df[df["category"].astype(str) == "benchmark"].copy()
        if not out.empty:
            return out
    return df.copy()


def ordered_methods(df: pd.DataFrame) -> List[str]:
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    methods.extend([m for m in sorted(set(df["method"])) if m not in methods])
    return methods


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def artifact_label(artifact: str) -> str:
    return ARTIFACT_LABELS.get(artifact, artifact)


def save_figure(fig: plt.Figure, name: str) -> None:
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08, 1.06, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def mean_sem(df: pd.DataFrame, value_col: str, group_col: str = "method") -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def bar_with_error(ax: plt.Axes, data: pd.DataFrame, value_col: str, ylabel: str, ylim=None) -> None:
    methods = ordered_methods(data)
    stats = mean_sem(data, value_col)
    stats = stats.set_index("method").reindex(methods).reset_index()
    x = np.arange(len(methods))
    y = stats["mean"].to_numpy(dtype=float)
    yerr = stats["sem"].to_numpy(dtype=float)
    colors = [COLORS.get(m, "#999999") for m in methods]
    ax.bar(x, y, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)


# =============================================================================
# Figure 6：F1 热图
# =============================================================================

def plot_figure6_f1_heatmaps(sample: pd.DataFrame) -> None:
    df = benchmark_only(sample)
    df = df[df["artifact_type"].notna()].copy()

    pivot_rows = []
    methods = ordered_methods(df)
    snrs = sorted(df["snr_target_db"].dropna().unique())
    artifacts = [a for a in ARTIFACT_ORDER if a in set(df["artifact_type"].astype(str))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    vmin, vmax = 0.0, 1.0
    last_im = None
    for idx, method in enumerate(methods[:4]):
        ax = axes[idx]
        sub = df[df["method"] == method]
        table = (
            sub.groupby(["artifact_type", "snr_target_db"], dropna=False)["f1"]
            .mean()
            .unstack("snr_target_db")
            .reindex(index=artifacts, columns=snrs)
        )
        table.to_csv(TABLE_DIR / f"Figure6_f1_heatmap_{method}.csv", encoding="utf-8-sig")
        pivot_rows.append(table.assign(method=method).reset_index())

        arr = table.to_numpy(dtype=float)
        last_im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(method_label(method))
        ax.set_xticks(np.arange(len(snrs)))
        ax.set_xticklabels([f"{s:g}" for s in snrs])
        ax.set_yticks(np.arange(len(artifacts)))
        ax.set_yticklabels([artifact_label(a) for a in artifacts])
        ax.set_xlabel("Target SNR (dB)")
        ax.set_ylabel("Artifact type")

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)

        add_panel_label(ax, chr(ord("a") + idx))

    for idx in range(len(methods[:4]), 4):
        axes[idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), shrink=0.88, label="Artifact F1")

    save_figure(fig, "Figure6_f1_heatmaps")


# =============================================================================
# Figure 7：波形保真
# =============================================================================

def plot_figure7_waveform_fidelity(sample: pd.DataFrame) -> None:
    df = benchmark_only(sample)
    cols = ["snr_gain_db", "nrmse", "corr"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    bar_with_error(axes[0], df, "snr_gain_db", "SNR gain (dB)")
    bar_with_error(axes[1], df, "nrmse", "NRMSE", ylim=(0, max(0.05, np.nanpercentile(df["nrmse"], 95) * 1.15)))
    bar_with_error(axes[2], df, "corr", "Correlation", ylim=(0, 1.02))

    axes[0].axhline(0, color="black", linewidth=0.8)
    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    table = df.groupby("method", dropna=False)[cols].agg(["mean", "std", "count"]).reset_index()
    table.to_csv(TABLE_DIR / "Figure7_waveform_fidelity_table.csv", index=False, encoding="utf-8-sig")

    save_figure(fig, "Figure7_waveform_fidelity")


# =============================================================================
# Figure 8：消融实验总览
# =============================================================================

def plot_figure8_ablation(sample: pd.DataFrame) -> None:
    df = benchmark_only(sample)
    metrics = [
        ("f1", "Artifact F1", (0, 1.02)),
        ("signal_false_removal_rate", "Signal false removal rate", (0, max(0.02, np.nanpercentile(df["signal_false_removal_rate"], 95) * 1.2))),
        ("snr_gain_db", "SNR gain (dB)", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for i, (col, ylabel, ylim) in enumerate(metrics):
        bar_with_error(axes[i], df, col, ylabel, ylim=ylim)
        if col == "snr_gain_db":
            axes[i].axhline(0, color="black", linewidth=0.8)
        add_panel_label(axes[i], chr(ord("a") + i))

    table_cols = [m[0] for m in metrics]
    table = df.groupby("method", dropna=False)[table_cols].agg(["mean", "std", "count"]).reset_index()
    table.to_csv(TABLE_DIR / "Figure8_ablation_overall_table.csv", index=False, encoding="utf-8-sig")

    save_figure(fig, "Figure8_ablation_overall")


# =============================================================================
# Figure 9：SNR 响应曲线
# =============================================================================

def plot_figure9_snr_response(sample: pd.DataFrame) -> None:
    df = benchmark_only(sample)
    methods = ordered_methods(df)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        ("f1", "Artifact F1"),
        ("signal_false_removal_rate", "Signal false removal rate"),
        ("snr_gain_db", "SNR gain (dB)"),
    ]

    table_parts = []
    for ax, (col, ylabel) in zip(axes, panels):
        for method in methods:
            sub = df[df["method"] == method]
            stat = (
                sub.groupby("snr_target_db", dropna=False)[col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("snr_target_db")
            )
            stat["sem"] = stat["std"] / np.sqrt(stat["count"].clip(lower=1))
            stat["method"] = method
            table_parts.append(stat.assign(metric=col))
            ax.errorbar(
                stat["snr_target_db"], stat["mean"], yerr=stat["sem"],
                marker="o", linewidth=1.8, capsize=3,
                color=COLORS.get(method, None),
                label=method_label(method),
            )
        ax.set_xlabel("Target SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if col in ("f1", "signal_false_removal_rate"):
            ax.set_ylim(0, 1.02)
        if col == "snr_gain_db":
            ax.axhline(0, color="black", linewidth=0.8)

    axes[0].legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    pd.concat(table_parts, ignore_index=True).to_csv(TABLE_DIR / "Figure9_snr_response_table.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, "Figure9_snr_response_curves")


# =============================================================================
# Figure 10：hard_case 难例边界
# =============================================================================

def plot_figure10_hard_case(sample: pd.DataFrame) -> None:
    df = benchmark_only(sample)
    hard = df[df["artifact_type"].astype(str) == "hard_case"].copy()
    if hard.empty:
        print("[跳过] 未找到 hard_case 数据，无法生成 Figure 10。")
        return

    methods = ordered_methods(hard)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        ("f1", "Artifact F1"),
        ("recall", "Artifact recall"),
        ("signal_false_removal_rate", "Signal false removal rate"),
    ]

    table_parts = []
    for ax, (col, ylabel) in zip(axes, panels):
        for method in methods:
            sub = hard[hard["method"] == method]
            stat = (
                sub.groupby("snr_target_db", dropna=False)[col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("snr_target_db")
            )
            stat["sem"] = stat["std"] / np.sqrt(stat["count"].clip(lower=1))
            stat["method"] = method
            table_parts.append(stat.assign(metric=col))
            ax.errorbar(
                stat["snr_target_db"], stat["mean"], yerr=stat["sem"],
                marker="o", linewidth=1.8, capsize=3,
                color=COLORS.get(method, None),
                label=method_label(method),
            )
        ax.set_xlabel("Target SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    pd.concat(table_parts, ignore_index=True).to_csv(TABLE_DIR / "Figure10_hard_case_boundary_table.csv", index=False, encoding="utf-8-sig")
    save_figure(fig, "Figure10_hard_case_boundary")


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()
    sample, summary = load_metrics()

    print("=" * 80)
    print("MCS-AND 论文图表生成")
    print("=" * 80)
    print(f"样本指标: {SAMPLE_METRICS_CSV}")
    print(f"输出目录: {FIG_DIR}")
    print(f"样本行数: {len(sample)}")
    print(f"方法: {ordered_methods(sample)}")

    plot_figure6_f1_heatmaps(sample)
    plot_figure7_waveform_fidelity(sample)
    plot_figure8_ablation(sample)
    plot_figure9_snr_response(sample)
    plot_figure10_hard_case(sample)

    print("=" * 80)
    print("[完成] 论文图表已生成。")
    print(f"图件目录: {FIG_DIR}")
    print(f"表格目录: {TABLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
