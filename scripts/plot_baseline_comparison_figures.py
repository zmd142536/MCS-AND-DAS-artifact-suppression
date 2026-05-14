# -*- coding: utf-8 -*-
"""
baseline + MCS-AND 综合对比图生成脚本（硬编码路径版，中文详细注释）。

本脚本读取：
    D:\项目实验\积石山实验\DAS相关\去噪论文\Baseline_benchmark\combined_sample_metrics.csv

并输出一组适合论文/补充材料使用的 baseline 对比图：

    BaselineFig1_waveform_metrics
        波形保真指标：SNR gain、NRMSE、correlation、|amplitude bias|

    BaselineFig2_event_metrics
        事件级指标：event precision、event recall、event F1、signal false removal rate

    BaselineFig3_pointwise_metrics
        逐点 mask 指标：point-wise precision、recall、F1、IoU

    BaselineFig4_event_f1_by_artifact
        不同伪迹类型下，各方法的事件级 F1 热图

    BaselineFig5_snr_response
        不同目标 SNR 下，各方法的 event F1、SNR gain、false removal 变化曲线

    BaselineFig6_method_rank_summary
        方法综合排名图：把关键指标归一化后展示每种方法的综合表现

重要解释：
    bandpass_median、global_svd、dasn2n 这类 baseline 本身通常不输出 artifact mask。
    你前一步 run_synthetic_baselines.py 中使用 residual threshold 从去噪残差推导 mask。
    因此 baseline 的 mask/event 指标是“辅助比较”，最公平的主比较仍是 waveform 指标。

使用方法：
    在 Spyder 里直接运行：
    %runfile 'D:/项目实验/积石山实验/DAS相关/Apython/S2/去噪论文用代码/xinquzaodaima/plot_baseline_comparison_figures.py'
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
# 1. 硬编码输入输出路径
# =============================================================================

# baseline + MCS-AND 合并结果目录。
RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\Baseline_benchmark")

# 前一步 run_synthetic_baselines.py 生成的合并样本级指标。
COMBINED_SAMPLE_CSV = RESULT_DIR / "combined_sample_metrics.csv"

# 图片输出目录。
FIG_DIR = RESULT_DIR / "baseline_paper_figures"

# 每张图对应的数据表输出目录。
TABLE_DIR = FIG_DIR / "baseline_figure_tables"


# =============================================================================
# 2. 方法顺序、名称和颜色
# =============================================================================

# 按论文叙事顺序排列：
# 先传统 baseline，再深度学习 baseline，再 MCS-AND 消融。
METHOD_ORDER = [
    "bandpass_median",
    "global_svd",
    "dasn2n",
    "mcd_only",
    "mcd_plain_semblance",
    "mcd_tau_p",
    "mcd_tau_p_lowrank",
]

METHOD_LABELS = {
    "bandpass_median": "Bandpass + median",
    "global_svd": "Global SVD",
    "dasn2n": "DAS-N2N",
    "mcd_only": "MCD only",
    "mcd_plain_semblance": "MCD + plain sem.",
    "mcd_tau_p": "MCD + tau-p",
    "mcd_tau_p_lowrank": "MCD + tau-p + low-rank",
}

COLORS = {
    "bandpass_median": "#9E9E9E",
    "global_svd": "#B07AA1",
    "dasn2n": "#59A14F",
    "mcd_only": "#7A7A7A",
    "mcd_plain_semblance": "#4C78A8",
    "mcd_tau_p": "#F58518",
    "mcd_tau_p_lowrank": "#54A24B",
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


# =============================================================================
# 3. 绘图风格配置
# =============================================================================

plt.rcParams.update({
    # 使用 Arial 是为了期刊图更通用；如果本机没有 Arial，matplotlib 会自动 fallback。
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    "savefig.dpi": 300,
    # 让 PDF 中的文字尽量保持可编辑。
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 4. 通用工具函数
# =============================================================================

def ensure_dirs() -> None:
    """创建图片和表格输出目录。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> pd.DataFrame:
    """读取 combined_sample_metrics.csv，并把关键列转为数值。"""
    if not COMBINED_SAMPLE_CSV.exists():
        raise FileNotFoundError(f"缺少合并指标文件: {COMBINED_SAMPLE_CSV}")

    df = pd.read_csv(COMBINED_SAMPLE_CSV, encoding="utf-8-sig")

    # 只保留 benchmark 样本，避免 signal_rescue 与 baseline 比较口径不一致。
    if "category" in df.columns:
        df = df[df["category"].astype(str) == "benchmark"].copy()

    # 把常用指标列转成 float，避免 CSV 中空字符串导致聚合错误。
    numeric_cols = [
        "snr_target_db",
        "precision", "recall", "f1", "mask_iou",
        "event_precision", "event_recall", "event_f1", "hit_rate_tol",
        "signal_false_removal_rate", "signal_preservation_rate",
        "snr_gain_db", "nrmse", "corr", "amplitude_bias",
        "pred_mask_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ordered_methods(df: pd.DataFrame) -> List[str]:
    """返回当前数据中实际存在的方法，并按 METHOD_ORDER 排序。"""
    existing = set(df["method"].astype(str))
    methods = [m for m in METHOD_ORDER if m in existing]
    methods.extend([m for m in sorted(existing) if m not in methods])
    return methods


def method_label(method: str) -> str:
    """方法名转为图中显示名称。"""
    return METHOD_LABELS.get(method, method)


def artifact_label(artifact: str) -> str:
    """伪迹类型转为图中显示名称。"""
    return ARTIFACT_LABELS.get(artifact, artifact)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """给子图左上角添加 a/b/c 面板编号。"""
    ax.text(
        -0.08, 1.06, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """同时保存 PNG 和 PDF。"""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def mean_sem_by_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """按 method 计算均值、标准差、样本数和标准误。"""
    out = (
        df.groupby("method", dropna=False)[metric]
        .agg(["mean", "std", "count", "median"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    methods = ordered_methods(df)
    out["method"] = pd.Categorical(out["method"], categories=methods, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def bar_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: Tuple[float, float] | None = None,
    transform_abs: bool = False,
) -> pd.DataFrame:
    """
    绘制按方法分组的柱状图。

    参数：
        metric: 指标列名。
        ylabel: y 轴名称。
        ylim: y 轴范围。
        transform_abs: 是否取绝对值，例如 amplitude_bias 常用绝对值更直观。
    """
    plot_df = df.copy()
    if transform_abs:
        plot_df[metric] = np.abs(plot_df[metric])

    stat = mean_sem_by_method(plot_df, metric)
    methods = stat["method"].astype(str).tolist()
    x = np.arange(len(methods))
    y = stat["mean"].to_numpy(dtype=float)
    yerr = stat["sem"].to_numpy(dtype=float)
    colors = [COLORS.get(m, "#999999") for m in methods]

    ax.bar(x, y, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods], rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)

    return stat


def save_table(df: pd.DataFrame, name: str) -> None:
    """保存图件对应的数据表。"""
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {path}")


# =============================================================================
# 5. 图 1：波形保真指标
# =============================================================================

def plot_waveform_metrics(df: pd.DataFrame) -> None:
    """
    BaselineFig1：波形保真指标。

    这张图是 baseline 比较中最公平、最重要的一张，因为所有方法都能输出 cleaned waveform。
    """
    required = ["snr_gain_db", "nrmse", "corr", "amplitude_bias"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    tables = []
    tables.append(bar_metric(axes[0], df, "snr_gain_db", "SNR gain (dB)"))
    axes[0].axhline(0, color="black", linewidth=0.8)

    # NRMSE 越低越好。为了避免极端值撑爆坐标轴，用 95 分位数设上限。
    nrmse_top = max(0.05, float(np.nanpercentile(df["nrmse"], 95)) * 1.15)
    tables.append(bar_metric(axes[1], df, "nrmse", "NRMSE", ylim=(0, nrmse_top)))

    tables.append(bar_metric(axes[2], df, "corr", "Correlation", ylim=(0, 1.02)))

    # amplitude_bias 有可能正负都有，论文图中看绝对偏差更直观。
    amp_abs = np.abs(df["amplitude_bias"].to_numpy(dtype=float))
    amp_top = max(0.05, float(np.nanpercentile(amp_abs, 90)) * 1.2)
    tables.append(bar_metric(axes[3], df, "amplitude_bias", "|Amplitude bias|", ylim=(0, amp_top), transform_abs=True))

    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    table = pd.concat([t.assign(metric=m) for t, m in zip(tables, required)], ignore_index=True)
    save_table(table, "BaselineFig1_waveform_metrics_table")
    save_figure(fig, "BaselineFig1_waveform_metrics")


# =============================================================================
# 6. 图 2：事件级指标
# =============================================================================

def plot_event_metrics(df: pd.DataFrame) -> None:
    """
    BaselineFig2：事件级指标。

    这张图用于回答：
        baseline 和 MCS-AND 谁更像一个“事件级伪迹检测器”？

    注意：
        baseline 的事件指标来自 residual mask，因此论文中应说明是辅助比较。
    """
    required = ["event_precision", "event_recall", "event_f1", "signal_false_removal_rate"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    tables = []
    tables.append(bar_metric(axes[0], df, "event_precision", "Event precision", ylim=(0, 1.02)))
    tables.append(bar_metric(axes[1], df, "event_recall", "Event recall", ylim=(0, 1.02)))
    tables.append(bar_metric(axes[2], df, "event_f1", "Event F1", ylim=(0, 1.02)))

    # 误删真实信号越低越好；通常数值很小，自动按 95 分位数缩放。
    top = max(0.01, float(np.nanpercentile(df["signal_false_removal_rate"], 95)) * 1.3)
    tables.append(bar_metric(axes[3], df, "signal_false_removal_rate", "Signal false removal rate", ylim=(0, top)))

    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    table = pd.concat([t.assign(metric=m) for t, m in zip(tables, required)], ignore_index=True)
    save_table(table, "BaselineFig2_event_metrics_table")
    save_figure(fig, "BaselineFig2_event_metrics")


# =============================================================================
# 7. 图 3：逐点 mask 指标
# =============================================================================

def plot_pointwise_metrics(df: pd.DataFrame) -> None:
    """
    BaselineFig3：逐点 mask 指标。

    这张图用于补充说明严格逐点 mask-level 比较。
    对 spike 等极窄伪迹，逐点 F1 会低估事件级检测效果，因此不应单独作为主结论。
    """
    required = ["precision", "recall", "f1", "mask_iou"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    tables = []
    labels = ["Point-wise precision", "Point-wise recall", "Point-wise F1", "Point-wise IoU"]
    for i, (metric, ylabel) in enumerate(zip(required, labels)):
        tables.append(bar_metric(axes[i], df, metric, ylabel, ylim=(0, 1.02)))
        add_panel_label(axes[i], chr(ord("a") + i))

    table = pd.concat([t.assign(metric=m) for t, m in zip(tables, required)], ignore_index=True)
    save_table(table, "BaselineFig3_pointwise_metrics_table")
    save_figure(fig, "BaselineFig3_pointwise_metrics")


# =============================================================================
# 8. 图 4：不同伪迹类型下的事件 F1 热图
# =============================================================================

def plot_event_f1_by_artifact(df: pd.DataFrame) -> None:
    """
    BaselineFig4：method × artifact_type 的事件级 F1 热图。

    这张图用于展示：
        不同方法适合处理哪些伪迹类型。
    """
    if "artifact_type" not in df.columns:
        raise ValueError("缺少 artifact_type 列。")

    methods = ordered_methods(df)
    artifacts = [a for a in ARTIFACT_ORDER if a in set(df["artifact_type"].astype(str))]

    table = (
        df.groupby(["method", "artifact_type"], dropna=False)["event_f1"]
        .mean()
        .unstack("artifact_type")
        .reindex(index=methods, columns=artifacts)
    )
    save_table(table.reset_index(), "BaselineFig4_event_f1_by_artifact_table")

    arr = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(np.arange(len(artifacts)))
    ax.set_xticklabels([artifact_label(a) for a in artifacts], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.set_xlabel("Artifact type")
    ax.set_ylabel("Method")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.55 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label="Event F1")
    save_figure(fig, "BaselineFig4_event_f1_by_artifact")


# =============================================================================
# 9. 图 5：SNR 响应曲线
# =============================================================================

def plot_snr_response(df: pd.DataFrame) -> None:
    """
    BaselineFig5：不同目标 SNR 下的指标变化曲线。

    这张图用于展示：
        在伪迹强弱变化时，各方法是否稳定。
    """
    methods = ordered_methods(df)
    panels = [
        ("event_f1", "Event F1", (0, 1.02)),
        ("snr_gain_db", "SNR gain (dB)", None),
        ("signal_false_removal_rate", "Signal false removal rate", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    table_parts = []

    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for method in methods:
            sub = df[df["method"] == method]
            stat = (
                sub.groupby("snr_target_db", dropna=False)[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("snr_target_db")
            )
            stat["sem"] = stat["std"] / np.sqrt(stat["count"].clip(lower=1))
            stat["method"] = method
            stat["metric"] = metric
            table_parts.append(stat)

            ax.errorbar(
                stat["snr_target_db"], stat["mean"], yerr=stat["sem"],
                marker="o", linewidth=1.6, capsize=3,
                color=COLORS.get(method, None),
                label=method_label(method),
            )

        ax.set_xlabel("Target SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if metric == "snr_gain_db":
            ax.axhline(0, color="black", linewidth=0.8)
        if metric == "signal_false_removal_rate":
            top = max(0.01, float(np.nanpercentile(df[metric], 95)) * 1.3)
            ax.set_ylim(0, top)

    axes[0].legend(frameon=False, fontsize=7, loc="best")
    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("a") + i))

    table = pd.concat(table_parts, ignore_index=True)
    save_table(table, "BaselineFig5_snr_response_table")
    save_figure(fig, "BaselineFig5_snr_response")


# =============================================================================
# 10. 图 6：综合排名图
# =============================================================================

def normalize_metric(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    将指标归一化到 0-1。

    higher_is_better=True:
        数值越大越好，例如 event_f1、snr_gain、corr。

    higher_is_better=False:
        数值越小越好，例如 nrmse、false removal rate、|amplitude bias|。
    """
    v = pd.to_numeric(values, errors="coerce")
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return pd.Series(np.ones(len(v)) * 0.5, index=values.index)
    norm = (v - vmin) / (vmax - vmin)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm


def plot_method_rank_summary(df: pd.DataFrame) -> None:
    """
    BaselineFig6：综合排名图。

    做法：
        1. 按方法计算若干关键指标均值；
        2. 将每个指标归一化为 0-1；
        3. 对归一化指标取平均，得到综合分数。

    注意：
        综合分数不是严格统计检验，只是帮助读者快速比较方法。
        论文中要把原始指标图作为主依据。
    """
    metrics = {
        "event_f1": True,
        "snr_gain_db": True,
        "corr": True,
        "nrmse": False,
        "signal_false_removal_rate": False,
    }

    stat = df.groupby("method", dropna=False).agg({
        "event_f1": "mean",
        "snr_gain_db": "mean",
        "corr": "mean",
        "nrmse": "mean",
        "signal_false_removal_rate": "mean",
    }).reset_index()

    for metric, higher in metrics.items():
        stat[f"{metric}_norm"] = normalize_metric(stat[metric], higher_is_better=higher)

    norm_cols = [f"{m}_norm" for m in metrics]
    stat["composite_score"] = stat[norm_cols].mean(axis=1)

    methods = ordered_methods(df)
    stat["method"] = pd.Categorical(stat["method"], categories=methods, ordered=True)
    stat = stat.sort_values("method").reset_index(drop=True)

    save_table(stat, "BaselineFig6_method_rank_summary_table")

    fig, ax = plt.subplots(figsize=(9.8, 4.5), constrained_layout=True)
    x = np.arange(len(stat))
    method_names = stat["method"].astype(str).tolist()
    colors = [COLORS.get(m, "#999999") for m in method_names]
    ax.bar(x, stat["composite_score"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in method_names], rotation=28, ha="right")
    ax.set_ylabel("Composite normalized score")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "BaselineFig6_method_rank_summary")


# =============================================================================
# 11. 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()
    df = load_metrics()
    methods = ordered_methods(df)

    print("=" * 80)
    print("Baseline + MCS-AND 综合图表生成")
    print("=" * 80)
    print(f"输入文件: {COMBINED_SAMPLE_CSV}")
    print(f"输出目录: {FIG_DIR}")
    print(f"样本行数: {len(df)}")
    print(f"方法列表: {methods}")

    plot_waveform_metrics(df)
    plot_event_metrics(df)
    plot_pointwise_metrics(df)
    plot_event_f1_by_artifact(df)
    plot_snr_response(df)
    plot_method_rank_summary(df)

    print("=" * 80)
    print("[完成] baseline 对比图已生成。")
    print(f"图件目录: {FIG_DIR}")
    print(f"表格目录: {TABLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
