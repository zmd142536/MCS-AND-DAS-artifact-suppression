# -*- coding: utf-8 -*-
"""
MCS-AND 合成实验论文图表生成脚本（硬编码路径版，增加事件级指标版）。

输入：
    D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark
        ├── mcsand_sample_metrics.csv
        └── mcsand_summary_metrics.csv

可选输入：
    如果 mcsand_sample_metrics.csv 中已经包含以下列，则本脚本直接绘图：
        event_precision
        event_recall
        event_f1
        hit_rate_tol
        event_precision_tol
        event_f1_tol
        dilated_mask_iou

    如果没有这些列，但包含真值/预测 mask 的 .npy 路径列，本脚本会自动计算事件级指标。
    支持的路径列名如下：
        truth_mask_path / gt_mask_path / true_mask_path / artifact_truth_mask_path
        pred_mask_path / predicted_mask_path / final_mask_path / artifact_pred_mask_path

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark\paper_figures
        ├── Figure6_f1_heatmaps.png/.pdf
        ├── Figure6b_event_hit_heatmaps.png/.pdf
        ├── Figure6c_event_metrics_bars.png/.pdf
        ├── Figure7_waveform_fidelity.png/.pdf
        ├── Figure8_ablation_overall.png/.pdf
        ├── Figure9_snr_response_curves.png/.pdf
        ├── Figure10_hard_case_boundary.png/.pdf
        └── paper_figure_tables/*.csv

说明：
    - Figure 6：逐点 artifact F1 热图。
    - Figure 6b：事件级命中率/召回率热图，用于避免 spike 等极窄伪迹被逐点 F1 低估。
    - Figure 6c：事件级 precision、recall、tolerance-window hit rate、dilated-mask IoU 柱状图。
    - Figure 7：波形保真指标，包括 SNR gain、NRMSE、相关系数。
    - Figure 8：消融实验总览。
    - Figure 9：随 SNR 变化的响应曲线。
    - Figure 10：hard_case 难例边界展示。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 硬编码路径
# =============================================================================

RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark2")
SAMPLE_METRICS_CSV = RESULT_DIR / "mcsand_sample_metrics.csv"
SUMMARY_METRICS_CSV = RESULT_DIR / "mcsand_summary_metrics.csv"
FIG_DIR = RESULT_DIR / "paper_figures"
TABLE_DIR = RESULT_DIR / "paper_figure_tables"


# =============================================================================
# 事件级指标参数
# =============================================================================
# 真实 mask 膨胀半径。单位是采样点和通道数。
# spike 极窄，建议给一个小容差；如果采样率 1000 Hz，time_radius=10 约为 ±10 ms。
DILATION_TIME_RADIUS_SAMPLES = 10
DILATION_CHANNEL_RADIUS = 1

# 连通域连接方式：4 或 8。DAS 时空 mask 推荐 8，允许斜向连接 moving artifact。
CONNECTED_COMPONENT_CONNECTIVITY = 8


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

EVENT_METRIC_LABELS = {
    "event_precision": "Event precision",
    "event_recall": "Event recall",
    "event_f1": "Event F1",
    "hit_rate_tol": "Hit rate (tol.)",
    "event_precision_tol": "Event precision (tol.)",
    "event_f1_tol": "Event F1 (tol.)",
    "dilated_mask_iou": "Dilated-mask IoU",
}

EVENT_METRIC_ALIASES = {
    "event_precision": [
        "event_precision",
        "event_level_precision",
        "event_precision_no_tol",
    ],
    "event_recall": [
        "event_recall",
        "event_level_recall",
        "event_recall_no_tol",
    ],
    "event_f1": [
        "event_f1",
        "event_level_f1",
        "event_f1_no_tol",
    ],
    "hit_rate_tol": [
        "hit_rate_tol",
        "hit_rate_with_tolerance",
        "tolerance_hit_rate",
        "event_recall_tol",
        "event_level_recall_tol",
    ],
    "event_precision_tol": [
        "event_precision_tol",
        "event_level_precision_tol",
        "precision_with_tolerance",
    ],
    "event_f1_tol": [
        "event_f1_tol",
        "event_level_f1_tol",
        "f1_with_tolerance",
    ],
    "dilated_mask_iou": [
        "dilated_mask_iou",
        "mask_iou_after_truth_dilation",
        "truth_dilated_iou",
        "mask_iou_dilated_truth",
    ],
}

TRUTH_MASK_PATH_COLUMNS = [
    "truth_mask_path",
    "gt_mask_path",
    "true_mask_path",
    "artifact_truth_mask_path",
]

PRED_MASK_PATH_COLUMNS = [
    "pred_mask_path",
    "predicted_mask_path",
    "final_mask_path",
    "artifact_pred_mask_path",
]


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
# 基础工具函数
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
    if "method" not in df.columns:
        return []
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    methods.extend([m for m in sorted(set(df["method"])) if m not in methods])
    return methods


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def artifact_label(artifact: str) -> str:
    return ARTIFACT_LABELS.get(artifact, artifact)


def metric_label(metric: str) -> str:
    return EVENT_METRIC_LABELS.get(metric, metric)


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


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


# =============================================================================
# 事件级指标计算工具
# =============================================================================

def standardize_event_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不同命名习惯的事件级指标列统一复制成标准列名。
    不会删除原始列。
    """
    out = df.copy()

    for standard_name, aliases in EVENT_METRIC_ALIASES.items():
        if standard_name in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[standard_name] = out[alias]
                break

    if "event_f1" not in out.columns and {"event_precision", "event_recall"}.issubset(out.columns):
        out["event_f1"] = [
            safe_f1(p, r)
            for p, r in zip(out["event_precision"], out["event_recall"])
        ]

    if "event_f1_tol" not in out.columns and {"event_precision_tol", "hit_rate_tol"}.issubset(out.columns):
        out["event_f1_tol"] = [
            safe_f1(p, r)
            for p, r in zip(out["event_precision_tol"], out["hit_rate_tol"])
        ]

    return out


def resolve_mask_path(path_like: object) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return RESULT_DIR / p


def load_mask(path_like: object) -> np.ndarray:
    path = resolve_mask_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"mask 文件不存在: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if "mask" in data.files:
            arr = data["mask"]
        else:
            arr = data[data.files[0]]
    else:
        raise ValueError(f"暂不支持的 mask 文件格式: {path.suffix}，建议保存为 .npy 或 .npz")

    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 1D 或 2D，当前 shape={arr.shape}")

    return arr.astype(bool)


def binary_dilate_rect(mask: np.ndarray, time_radius: int, channel_radius: int) -> np.ndarray:
    """
    使用矩形结构元素做二值膨胀，不依赖 scipy。
    mask shape 建议为 [time, channel]。
    """
    mask = np.asarray(mask).astype(bool)

    if time_radius <= 0 and channel_radius <= 0:
        return mask.copy()

    t_rad = max(0, int(time_radius))
    c_rad = max(0, int(channel_radius))

    padded = np.pad(mask, ((t_rad, t_rad), (c_rad, c_rad)), mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)

    for dt in range(2 * t_rad + 1):
        for dc in range(2 * c_rad + 1):
            out |= padded[dt:dt + mask.shape[0], dc:dc + mask.shape[1]]

    return out


def connected_components_2d(mask: np.ndarray, connectivity: int = 8) -> List[np.ndarray]:
    """
    纯 numpy + deque 的 2D 二值连通域提取。
    返回值为若干 bool mask，每个 mask 对应一个事件。
    """
    mask = np.asarray(mask).astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError("connectivity 只能为 4 或 8")

    rows, cols = mask.shape
    starts = np.argwhere(mask & (~visited))

    for start_r, start_c in starts:
        if visited[start_r, start_c]:
            continue

        comp = np.zeros_like(mask, dtype=bool)
        q: deque[Tuple[int, int]] = deque()
        q.append((int(start_r), int(start_c)))
        visited[start_r, start_c] = True

        while q:
            r, c = q.popleft()
            comp[r, c] = True

            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[rr, cc] or not mask[rr, cc]:
                    continue
                visited[rr, cc] = True
                q.append((rr, cc))

        components.append(comp)

    return components


def mask_iou(pred_mask: np.ndarray, truth_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    truth = np.asarray(truth_mask).astype(bool)

    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()

    if union == 0:
        return np.nan
    return float(inter / union)


def compute_event_metrics_from_masks(
    truth_mask: np.ndarray,
    pred_mask: np.ndarray,
    time_radius: int = DILATION_TIME_RADIUS_SAMPLES,
    channel_radius: int = DILATION_CHANNEL_RADIUS,
    connectivity: int = CONNECTED_COMPONENT_CONNECTIVITY,
) -> Dict[str, float]:
    """
    从真值 mask 和预测 mask 计算事件级指标。

    event_precision / event_recall:
        不做容差膨胀，只要预测连通域与真实连通域有交集即算命中。

    hit_rate_tol:
        对真值 mask 膨胀后，真实事件与预测 mask 有交集即算命中。
        它本质上是 tolerance-window event recall。

    event_precision_tol:
        对真值 mask 膨胀后，预测事件与膨胀真值有交集即算正确预测。

    dilated_mask_iou:
        预测 mask 与膨胀后的真值 mask 的 IoU。
    """
    truth = np.asarray(truth_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    if truth.shape != pred.shape:
        raise ValueError(f"truth_mask 和 pred_mask shape 不一致: {truth.shape} vs {pred.shape}")

    truth_components = connected_components_2d(truth, connectivity=connectivity)
    pred_components = connected_components_2d(pred, connectivity=connectivity)

    n_truth = len(truth_components)
    n_pred = len(pred_components)

    # 无容差事件级 recall
    if n_truth > 0:
        truth_hits = sum(bool(np.logical_and(comp, pred).any()) for comp in truth_components)
        event_recall = truth_hits / n_truth
    else:
        truth_hits = 0
        event_recall = np.nan

    # 无容差事件级 precision
    if n_pred > 0:
        pred_hits = sum(bool(np.logical_and(comp, truth).any()) for comp in pred_components)
        event_precision = pred_hits / n_pred
    else:
        pred_hits = 0
        event_precision = np.nan

    event_f1 = safe_f1(event_precision, event_recall)

    # 容差评价：真实 mask 膨胀
    truth_dilated = binary_dilate_rect(
        truth,
        time_radius=time_radius,
        channel_radius=channel_radius,
    )

    if n_truth > 0:
        truth_hits_tol = sum(bool(np.logical_and(comp, pred).any()) for comp in truth_components)
        # 更合理的容差 hit：对单个真实事件分别膨胀后再看是否与 pred 相交
        truth_hits_tol = 0
        for comp in truth_components:
            comp_dilated = binary_dilate_rect(comp, time_radius=time_radius, channel_radius=channel_radius)
            if np.logical_and(comp_dilated, pred).any():
                truth_hits_tol += 1
        hit_rate_tol = truth_hits_tol / n_truth
    else:
        truth_hits_tol = 0
        hit_rate_tol = np.nan

    if n_pred > 0:
        pred_hits_tol = sum(bool(np.logical_and(comp, truth_dilated).any()) for comp in pred_components)
        event_precision_tol = pred_hits_tol / n_pred
    else:
        pred_hits_tol = 0
        event_precision_tol = np.nan

    event_f1_tol = safe_f1(event_precision_tol, hit_rate_tol)
    dilated_iou = mask_iou(pred, truth_dilated)

    return {
        "n_truth_events": float(n_truth),
        "n_pred_events": float(n_pred),
        "event_precision": float(event_precision) if np.isfinite(event_precision) else np.nan,
        "event_recall": float(event_recall) if np.isfinite(event_recall) else np.nan,
        "event_f1": float(event_f1) if np.isfinite(event_f1) else np.nan,
        "hit_rate_tol": float(hit_rate_tol) if np.isfinite(hit_rate_tol) else np.nan,
        "event_precision_tol": float(event_precision_tol) if np.isfinite(event_precision_tol) else np.nan,
        "event_f1_tol": float(event_f1_tol) if np.isfinite(event_f1_tol) else np.nan,
        "dilated_mask_iou": float(dilated_iou) if np.isfinite(dilated_iou) else np.nan,
    }


def add_event_metrics_if_possible(sample: pd.DataFrame) -> pd.DataFrame:
    """
    优先使用 CSV 中已有事件级指标。
    如果缺失，但有 truth/pred mask 路径，则从 .npy/.npz mask 自动计算。
    如果二者都没有，则不报错，只跳过新增图。
    """
    sample = standardize_event_metric_columns(sample)

    required_for_plot = ["event_precision", "event_recall", "hit_rate_tol", "dilated_mask_iou"]
    if all(col in sample.columns for col in required_for_plot):
        print("[OK] 已在 CSV 中找到事件级指标列，将直接绘制 Figure 6b/6c。")
        return sample

    truth_col = first_existing_column(sample, TRUTH_MASK_PATH_COLUMNS)
    pred_col = first_existing_column(sample, PRED_MASK_PATH_COLUMNS)

    if truth_col is None or pred_col is None:
        print("[提示] 未发现完整事件级指标列，也未发现 truth/pred mask 路径列。")
        print("[提示] Figure 6b/6c 将跳过。若要绘制，请在 mcsand_sample_metrics.csv 中加入：")
        print("       event_precision, event_recall, hit_rate_tol, dilated_mask_iou")
        print("       或加入 truth_mask_path 和 pred_mask_path 两列，指向 .npy/.npz mask 文件。")
        return sample

    print(f"[INFO] 从 mask 路径自动计算事件级指标: truth={truth_col}, pred={pred_col}")
    metric_rows: List[Dict[str, float]] = []

    for idx, row in sample.iterrows():
        try:
            truth = load_mask(row[truth_col])
            pred = load_mask(row[pred_col])
            metrics = compute_event_metrics_from_masks(truth, pred)
        except Exception as exc:
            print(f"[警告] 第 {idx} 行事件级指标计算失败: {exc}")
            metrics = {
                "n_truth_events": np.nan,
                "n_pred_events": np.nan,
                "event_precision": np.nan,
                "event_recall": np.nan,
                "event_f1": np.nan,
                "hit_rate_tol": np.nan,
                "event_precision_tol": np.nan,
                "event_f1_tol": np.nan,
                "dilated_mask_iou": np.nan,
            }
        metric_rows.append(metrics)

    metric_df = pd.DataFrame(metric_rows)
    out = sample.copy()

    for col in metric_df.columns:
        out[col] = metric_df[col].values

    event_csv = TABLE_DIR / "computed_event_level_metrics_per_sample.csv"
    out.to_csv(event_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存带事件级指标的样本表: {event_csv}")

    return out


def has_metric(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and np.isfinite(pd.to_numeric(df[col], errors="coerce")).any()


# =============================================================================
# Figure 6：通用热图函数 + 逐点 F1 热图
# =============================================================================

def plot_metric_heatmaps(
    sample: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    output_name: str,
    colorbar_label: str,
) -> None:
    df = benchmark_only(sample)
    df = df[df["artifact_type"].notna()].copy()

    if not has_metric(df, metric_col):
        print(f"[跳过] 缺少有效列 {metric_col}，无法生成 {output_name}。")
        return

    methods = ordered_methods(df)
    snrs = sorted(df["snr_target_db"].dropna().unique())
    artifacts = [a for a in ARTIFACT_ORDER if a in set(df["artifact_type"].astype(str))]

    if not methods or len(snrs) == 0 or not artifacts:
        print(f"[跳过] {output_name} 缺少 method/snr/artifact_type 信息。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    vmin, vmax = 0.0, 1.0
    last_im = None

    for idx, method in enumerate(methods[:4]):
        ax = axes[idx]
        sub = df[df["method"] == method]

        table = (
            sub.groupby(["artifact_type", "snr_target_db"], dropna=False)[metric_col]
            .mean()
            .unstack("snr_target_db")
            .reindex(index=artifacts, columns=snrs)
        )

        table.to_csv(TABLE_DIR / f"{output_name}_{metric_col}_{method}.csv", encoding="utf-8-sig")

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
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if val < 0.55 else "black",
                        fontsize=8,
                    )

        add_panel_label(ax, chr(ord("a") + idx))

    for idx in range(len(methods[:4]), 4):
        axes[idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), shrink=0.88, label=colorbar_label)

    save_figure(fig, output_name)


def plot_figure6_f1_heatmaps(sample: pd.DataFrame) -> None:
    plot_metric_heatmaps(
        sample=sample,
        metric_col="f1",
        metric_name="Artifact F1",
        output_name="Figure6_f1_heatmaps",
        colorbar_label="Point-wise artifact F1",
    )


# =============================================================================
# Figure 6b：事件级 hit-rate/recall 热图
# =============================================================================

def plot_figure6b_event_hit_heatmaps(sample: pd.DataFrame) -> None:
    df = standardize_event_metric_columns(sample)

    # 优先展示带容差的 hit rate；如果没有，则退化为 event_recall。
    if has_metric(df, "hit_rate_tol"):
        metric_col = "hit_rate_tol"
        label = "Hit rate (tol.)"
    elif has_metric(df, "event_recall"):
        metric_col = "event_recall"
        label = "Event recall"
    else:
        print("[跳过] 缺少 hit_rate_tol 或 event_recall，无法生成 Figure6b_event_hit_heatmaps。")
        return

    plot_metric_heatmaps(
        sample=df,
        metric_col=metric_col,
        metric_name=label,
        output_name="Figure6b_event_hit_heatmaps",
        colorbar_label=label,
    )


# =============================================================================
# Figure 6c：事件级指标柱状图
# =============================================================================

def plot_figure6c_event_metrics_bars(sample: pd.DataFrame) -> None:
    df = benchmark_only(standardize_event_metric_columns(sample))

    metrics = [
        ("event_precision", "Event precision", (0, 1.02)),
        ("event_recall", "Event recall", (0, 1.02)),
        ("hit_rate_tol", "Hit rate (tol.)", (0, 1.02)),
        ("dilated_mask_iou", "Dilated-mask IoU", (0, 1.02)),
    ]

    available = [(col, ylabel, ylim) for col, ylabel, ylim in metrics if has_metric(df, col)]

    if not available:
        print("[跳过] 没有可用事件级指标，无法生成 Figure6c_event_metrics_bars。")
        return

    # 保持 2x2 布局；如果只有少数指标，空面板自动隐藏。
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()

    for i, (col, ylabel, ylim) in enumerate(available[:4]):
        bar_with_error(axes[i], df, col, ylabel, ylim=ylim)
        add_panel_label(axes[i], chr(ord("a") + i))

    for j in range(len(available[:4]), 4):
        axes[j].axis("off")

    table_cols = [col for col, _, _ in available]
    table = df.groupby("method", dropna=False)[table_cols].agg(["mean", "std", "count"]).reset_index()
    table.to_csv(TABLE_DIR / "Figure6c_event_metrics_bars_table.csv", index=False, encoding="utf-8-sig")

    # 额外输出 artifact_type 分组表，便于解释 spike 是否被逐点 F1 低估。
    if "artifact_type" in df.columns:
        by_type = (
            df.groupby(["method", "artifact_type"], dropna=False)[table_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        by_type.to_csv(TABLE_DIR / "Figure6c_event_metrics_by_artifact_type_table.csv", index=False, encoding="utf-8-sig")

    save_figure(fig, "Figure6c_event_metrics_bars")


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
    bar_with_error(
        axes[1],
        df,
        "nrmse",
        "NRMSE",
        ylim=(0, max(0.05, np.nanpercentile(df["nrmse"], 95) * 1.15)),
    )
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
        ("f1", "Point-wise artifact F1", (0, 1.02)),
        (
            "signal_false_removal_rate",
            "Signal false removal rate",
            (0, max(0.02, np.nanpercentile(df["signal_false_removal_rate"], 95) * 1.2)),
        ),
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
        ("f1", "Point-wise artifact F1"),
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
                stat["snr_target_db"],
                stat["mean"],
                yerr=stat["sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
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

    pd.concat(table_parts, ignore_index=True).to_csv(
        TABLE_DIR / "Figure9_snr_response_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        ("f1", "Point-wise artifact F1"),
        ("recall", "Point-wise artifact recall"),
        ("signal_false_removal_rate", "Signal false removal rate"),
    ]

    table_parts = []

    for ax, (col, ylabel) in zip(axes, panels):
        if col not in hard.columns:
            print(f"[跳过面板] hard_case 缺少列 {col}")
            ax.axis("off")
            continue

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
                stat["snr_target_db"],
                stat["mean"],
                yerr=stat["sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
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

    if table_parts:
        pd.concat(table_parts, ignore_index=True).to_csv(
            TABLE_DIR / "Figure10_hard_case_boundary_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_figure(fig, "Figure10_hard_case_boundary")


# =============================================================================
# 可选：事件级 SNR 响应曲线
# =============================================================================

def plot_figure9b_event_snr_response(sample: pd.DataFrame) -> None:
    """
    这张图不是原始主图之一，但很适合放补充材料：
    展示 event-level hit rate / event precision / dilated IoU 随 SNR 的变化。
    """
    df = benchmark_only(standardize_event_metric_columns(sample))
    methods = ordered_methods(df)

    panels = []
    for col in ["hit_rate_tol", "event_precision", "dilated_mask_iou"]:
        if has_metric(df, col):
            panels.append((col, metric_label(col)))

    if not panels:
        print("[跳过] 缺少事件级指标，无法生成 Figure9b_event_snr_response。")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

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
                stat["snr_target_db"],
                stat["mean"],
                yerr=stat["sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
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

    if table_parts:
        pd.concat(table_parts, ignore_index=True).to_csv(
            TABLE_DIR / "Figure9b_event_snr_response_table.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_figure(fig, "Figure9b_event_snr_response")


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()
    sample, summary = load_metrics()

    print("=" * 80)
    print("MCS-AND 论文图表生成：增加事件级指标版")
    print("=" * 80)
    print(f"样本指标: {SAMPLE_METRICS_CSV}")
    print(f"输出目录: {FIG_DIR}")
    print(f"样本行数: {len(sample)}")
    print(f"方法: {ordered_methods(sample)}")

    sample = add_event_metrics_if_possible(sample)

    plot_figure6_f1_heatmaps(sample)
    plot_figure6b_event_hit_heatmaps(sample)
    plot_figure6c_event_metrics_bars(sample)

    plot_figure7_waveform_fidelity(sample)
    plot_figure8_ablation(sample)
    plot_figure9_snr_response(sample)
    plot_figure9b_event_snr_response(sample)
    plot_figure10_hard_case(sample)

    print("=" * 80)
    print("[完成] 论文图表已生成。")
    print(f"图件目录: {FIG_DIR}")
    print(f"表格目录: {TABLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
