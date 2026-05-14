# -*- coding: utf-8 -*-
"""
FORESEE 真实数据：DAS-N2N 与 MCS-AND 对比图生成脚本（硬编码路径版）。

用途：
    在真实 FORESEE 城市交通 DAS 记录上比较：
        1. DAS-N2N cleaned output
        2. MCS-AND cleaned output
        3. 两者相对于 raw record 的 residual

重要说明：
    真实 FORESEE 数据没有 artifact ground truth，因此本脚本不计算 F1。
    这里比较的是：
        - residual 是否包含车辆相干轨迹；
        - output 与 raw 的相关系数；
        - relative change = RMS(raw - output) / RMS(raw)。

    如果 DAS-N2N residual 中出现明显斜向车辆轨迹，说明它改变了真实交通事件；
    如果 MCS-AND residual 接近空，说明它更保守，更符合“保护相干事件”的定位。

输入：
    MCS-AND 输出：
        D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results\outputs

    DAS-N2N 输出：
        D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_outputs

    DAS-N2N 输出文件命名建议：
        real_case_002_noon_late_dasn2n_output.npz

    DAS-N2N 输出字段至少包含以下任意一个：
        dasn2n_cleaned_ct
        cleaned_ct
        data_cleaned
        cleaned
        data_denoised

    字段 shape 应为 [channel, time]。

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_compare\paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =============================================================================
# 1. 硬编码路径
# =============================================================================

MCSAND_OUTPUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results\outputs")
DASN2N_OUTPUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_outputs")
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_compare")
FIG_DIR = OUT_DIR / "paper_figures"
TABLE_DIR = OUT_DIR / "tables"

# 默认选择你前面真实数据图中最清楚的一段。
PREFERRED_CASE_STEM = "real_case_002_noon_late"


# =============================================================================
# 2. 图件参数
# =============================================================================

A4_WIDTH_IN = 8.27
FIG_WIDTH_IN = A4_WIDTH_IN * 4.0 / 5.0
FIG_HEIGHT_IN = 5.6
DPI = 600

FONT_NAME = "Times New Roman"
FONT_SIZE_MAIN = 10
FONT_SIZE_SMALL = 8


def setup_style() -> None:
    """设置 Times New Roman、常规体、不加粗。"""
    plt.rcParams.update({
        "font.family": FONT_NAME,
        "font.weight": "normal",
        "font.size": FONT_SIZE_MAIN,
        "axes.titlesize": FONT_SIZE_MAIN,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZE_MAIN,
        "axes.labelweight": "normal",
        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE_SMALL,
        "legend.fontsize": FONT_SIZE_SMALL,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "mathtext.fontset": "stix",
    })


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def replace_tau_p(text: str) -> str:
    """统一使用 τ-p。"""
    return text.replace("tau-p", "τ-p").replace("tau_p", "τ-p")


def robust_vlim(data: np.ndarray, percentile: float = 99.0) -> float:
    """稳健色标范围。"""
    values = data[np.isfinite(data)]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(values), percentile))
    return max(vmax, 1e-8)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# 3. 数据读取
# =============================================================================

def choose_mcsand_output() -> Path:
    """选择 MCS-AND 输出文件。"""
    preferred = MCSAND_OUTPUT_DIR / f"{PREFERRED_CASE_STEM}_mcsand_real_output.npz"
    if preferred.exists():
        return preferred

    files = sorted(MCSAND_OUTPUT_DIR.glob("*_mcsand_real_output.npz"))
    if not files:
        raise FileNotFoundError(f"未找到 MCS-AND 真实数据输出: {MCSAND_OUTPUT_DIR}")
    return files[0]


def find_dasn2n_output(case_stem: str) -> Path:
    """
    查找 DAS-N2N 输出文件。

    支持几种常见命名：
        real_case_002_noon_late_dasn2n_output.npz
        real_case_002_noon_late_dasn2n.npz
        real_case_002_noon_late.npz
    """
    candidates = [
        DASN2N_OUTPUT_DIR / f"{case_stem}_dasn2n_output.npz",
        DASN2N_OUTPUT_DIR / f"{case_stem}_dasn2n.npz",
        DASN2N_OUTPUT_DIR / f"{case_stem}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path

    files = sorted(DASN2N_OUTPUT_DIR.glob(f"{case_stem}*.npz"))
    if files:
        return files[0]

    raise FileNotFoundError(
        "没有找到 DAS-N2N 输出文件。\n"
        f"请把 DAS-N2N cleaned 结果保存到: {DASN2N_OUTPUT_DIR}\n"
        f"推荐文件名: {case_stem}_dasn2n_output.npz\n"
        "推荐字段名: dasn2n_cleaned_ct，shape=[channel, time]"
    )


def load_cleaned_from_dasn2n(path: Path) -> np.ndarray:
    """从 DAS-N2N 输出 npz 中读取 cleaned 数组。"""
    keys = ("dasn2n_cleaned_ct", "cleaned_ct", "data_cleaned", "cleaned", "data_denoised")
    with np.load(path, allow_pickle=False) as npz:
        for key in keys:
            if key in npz.files:
                arr = npz[key].astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"{path} 中字段 {key} 不是二维数组，shape={arr.shape}")
                return arr
    raise KeyError(f"{path} 中未找到 DAS-N2N cleaned 字段，可用字段={list(np.load(path).files)}")


def load_mcsand_output(path: Path) -> Dict:
    """读取 MCS-AND 输出。"""
    with np.load(path, allow_pickle=False) as npz:
        summary = json.loads(str(npz["summary_json"]))
        raw = npz["data_proc_ct"].astype(np.float32)
        mcsand_cleaned = npz["mcd_tau_p_lowrank_cleaned_ct"].astype(np.float32)
    return {"summary": summary, "raw": raw, "mcsand_cleaned": mcsand_cleaned}


def get_case_stem(mcsand_path: Path) -> str:
    """从 MCS-AND 输出文件名还原 case stem。"""
    return mcsand_path.name.replace("_mcsand_real_output.npz", "")


def get_extent(raw: np.ndarray, summary: Dict) -> Tuple[float, float, float, float]:
    """生成 imshow 坐标。"""
    n_channels, n_samples = raw.shape
    fs = float(summary.get("fs", 125.0))
    # summary 中没有 global channel 起止时，用 case_id 图保持局部通道。
    return (0.0, n_samples / fs, n_channels, 0)


# =============================================================================
# 4. 指标与绘图
# =============================================================================

def compute_real_metrics(raw: np.ndarray, cleaned: np.ndarray, method: str) -> Dict:
    """计算真实数据无真值情况下的保守性指标。"""
    residual = raw - cleaned
    return {
        "method": method,
        "raw_rms": rms(raw),
        "output_rms": rms(cleaned),
        "residual_rms": rms(residual),
        "relative_change": rms(residual) / (rms(raw) + 1e-30),
        "raw_output_corr": safe_corrcoef(raw, cleaned),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    """添加不加粗面板编号。"""
    ax.text(
        -0.085, 1.06, letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_MAIN,
        fontweight="normal",
    )


def add_inpanel_label(ax: plt.Axes, text: str) -> None:
    """添加图内说明。"""
    ax.text(
        0.01, 0.92, replace_tau_p(text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE_SMALL,
        fontweight="normal",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.76,
        },
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """保存 PNG 和 PDF。"""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


def plot_comparison() -> None:
    """生成 DAS-N2N vs MCS-AND 真实数据对比图。"""
    mcsand_path = choose_mcsand_output()
    case_stem = get_case_stem(mcsand_path)
    dasn2n_path = find_dasn2n_output(case_stem)

    mcs = load_mcsand_output(mcsand_path)
    raw = mcs["raw"]
    mcsand_cleaned = mcs["mcsand_cleaned"]
    dasn2n_cleaned = load_cleaned_from_dasn2n(dasn2n_path)

    if dasn2n_cleaned.shape != raw.shape:
        raise ValueError(
            f"DAS-N2N 输出形状与 raw 不一致: dasn2n={dasn2n_cleaned.shape}, raw={raw.shape}。"
            "请确保输出为 [channel, time]。"
        )

    dasn2n_residual = raw - dasn2n_cleaned
    mcsand_residual = raw - mcsand_cleaned

    metrics = pd.DataFrame([
        compute_real_metrics(raw, dasn2n_cleaned, "DAS-N2N"),
        compute_real_metrics(raw, mcsand_cleaned, "MCS-AND"),
    ])
    metrics_csv = TABLE_DIR / "FieldFig3_dasn2n_mcsand_real_metrics.csv"
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {metrics_csv}")

    extent = get_extent(raw, mcs["summary"])
    amp_vlim = robust_vlim(np.concatenate([raw.ravel(), dasn2n_cleaned.ravel(), mcsand_cleaned.ravel()]), 99.0)
    res_vlim = robust_vlim(np.concatenate([dasn2n_residual.ravel(), mcsand_residual.ravel()]), 99.0)

    rows = [
        ("Raw FORESEE record", raw, amp_vlim, "Amplitude"),
        ("DAS-N2N output", dasn2n_cleaned, amp_vlim, "Amplitude"),
        ("DAS-N2N residual: raw - output", dasn2n_residual, res_vlim, "Residual"),
        ("MCS-AND output", mcsand_cleaned, amp_vlim, "Amplitude"),
        ("MCS-AND residual: raw - output", mcsand_residual, res_vlim, "Residual"),
    ]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        width_ratios=[1.0, 0.035],
        height_ratios=[1, 1, 1, 1, 1],
        wspace=0.04,
        hspace=0.16,
    )

    for i, (label, arr, vlim, cbar_label) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap="seismic",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
            rasterized=True,
        )
        add_inpanel_label(ax, label)

        if i == 0:
            ax.set_title(f"Field comparison on {case_stem}")
            add_panel_letter(ax, "a")
        if i == len(rows) - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Channel")

        cax = fig.add_subplot(gs[i, 1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL, length=2)
        cbar.set_label(cbar_label, fontsize=FONT_SIZE_SMALL, labelpad=3)

    save_figure(fig, "FieldFig3_DASN2N_vs_MCSAND_real_comparison")


def main() -> None:
    setup_style()
    ensure_dirs()
    print("=" * 80)
    print("FORESEE 真实数据 DAS-N2N vs MCS-AND 对比图")
    print("=" * 80)
    print(f"MCS-AND 输出目录: {MCSAND_OUTPUT_DIR}")
    print(f"DAS-N2N 输出目录: {DASN2N_OUTPUT_DIR}")
    print(f"输出目录: {FIG_DIR}")
    plot_comparison()
    print("=" * 80)
    print("[完成] 对比图已生成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
