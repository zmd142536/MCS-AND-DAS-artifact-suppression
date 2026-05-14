# -*- coding: utf-8 -*-
"""
FORESEE 真实 DAS 数据 MCS-AND 叠加图绘制脚本（单独版，硬编码路径）。

用途：
    读取 run_real_foresee_mcsand.py 生成的真实数据处理结果，
    绘制更适合论文正文使用的 overlay 版本图件。

为什么要用 overlay 版本：
    当前 FORESEE 真实片段中主要是车辆诱发的空间-时间相干事件，
    最终 artifact mask 接近为空。如果单独绘制 mask，会出现大面积空白图。
    overlay 版本把 MCD 候选点、τ-p 救回点、最终伪迹点直接叠加到原始
    DAS 波场上，更能说明：

        1. MCD-only 只检出少量候选异常；
        2. 这些候选事件基本被 τ-p 判定为相干交通事件并救回；
        3. 最终 MCS-AND 不会误删车辆诱发的真实相干波场；
        4. cleaned record 与 raw record 基本一致，说明方法不过度处理。

输入目录：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results

输出目录：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results\paper_figures2

输出图件：
    FieldFig1_real_mcsand_application.png / .pdf
    FieldFig1_real_mcsand_overlay.png / .pdf
    FieldFig2_real_case_summary.png / .pdf

字体和版面：
    - Times New Roman
    - 主字体 10 pt
    - 图内标注 8 pt
    - 所有字体不加粗
    - τ-p 使用希腊字母 τ
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =============================================================================
# 1. 硬编码路径
# =============================================================================

RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results")
OUTPUT_NPZ_DIR = RESULT_DIR / "outputs"
FIG_DIR = RESULT_DIR / "paper_figures2"
TABLE_DIR = RESULT_DIR / "paper_figure_tables2"


# =============================================================================
# 2. 图件参数
# =============================================================================

A4_WIDTH_IN = 8.27
MAIN_FIG_WIDTH_IN = A4_WIDTH_IN * 4.0 / 5.0
MAIN_FIG_HEIGHT_IN = 6.25

DPI = 600
FONT_NAME = "Times New Roman"
FONT_SIZE_MAIN = 10
FONT_SIZE_SMALL = 8

# 默认选择交通轨迹较清楚的一段。
PREFERRED_CASE_STEM = "real_case_002_noon_late"


def setup_style() -> None:
    """设置统一字体和线宽。"""
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
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "mathtext.fontset": "stix",
    })


def ensure_dirs() -> None:
    """创建输出目录。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def replace_tau_p(text: str) -> str:
    """统一把 tau-p / tau_p 替换为 τ-p。"""
    return text.replace("tau-p", "τ-p").replace("tau_p", "τ-p")


def robust_vlim(data: np.ndarray, percentile: float = 99.0) -> float:
    """使用绝对值分位数计算稳健色标范围。"""
    values = data[np.isfinite(data)]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(values), percentile))
    return max(vmax, 1e-8)


def save_figure(fig: plt.Figure, name: str) -> None:
    """同时保存 PNG 和 PDF。"""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


# =============================================================================
# 3. 数据读取
# =============================================================================

def choose_output_file() -> Path:
    """选择用于主图的 MCS-AND 输出 npz。"""
    preferred = OUTPUT_NPZ_DIR / f"{PREFERRED_CASE_STEM}_mcsand_real_output.npz"
    if preferred.exists():
        return preferred

    files = sorted(OUTPUT_NPZ_DIR.glob("*_mcsand_real_output.npz"))
    if not files:
        raise FileNotFoundError(f"未找到真实数据 MCS-AND 输出文件: {OUTPUT_NPZ_DIR}")
    return files[0]


def load_case_meta(source_case_file: str) -> Dict:
    """从原始真实 case npz 中读取 meta_json。"""
    path = Path(source_case_file)
    if not path.exists():
        return {}

    with np.load(path, allow_pickle=False) as npz:
        if "meta_json" not in npz.files:
            return {}
        try:
            return json.loads(str(npz["meta_json"]))
        except Exception:
            return {}


def load_output(path: Path) -> Dict:
    """读取单个真实数据 MCS-AND 输出。"""
    with np.load(path, allow_pickle=False) as npz:
        summary = json.loads(str(npz["summary_json"]))
        source_case_file = str(npz["source_case_file"])
        meta = load_case_meta(source_case_file)

        data = {
            "path": path,
            "summary": summary,
            "meta": meta,
            "source_case_file": source_case_file,
            # 以下数组均为 [channel, time]。
            "raw": npz["data_proc_ct"].astype(np.float32),
            "candidate_mask": npz["mcd_only_mask_ct"].astype(np.float32),
            "rescued_mask": npz["rescued_mask_ct"].astype(np.float32),
            "final_mask": npz["mcd_tau_p_mask_ct"].astype(np.float32),
            "cleaned": npz["mcd_tau_p_lowrank_cleaned_ct"].astype(np.float32),
            "removed": npz["removed_tau_p_lowrank_ct"].astype(np.float32),
        }

    return data


def get_extent(data_ct: np.ndarray, summary: Dict, meta: Dict) -> Tuple[float, float, float, float]:
    """
    生成 imshow 坐标。

    x 轴：时间，单位 s。
    y 轴：FORESEE 全局通道号。
    """
    n_channels, n_samples = data_ct.shape
    fs = float(summary.get("fs", 125.0))
    ch0 = int(meta.get("channel_start", 0))
    ch1 = int(meta.get("channel_end", ch0 + n_channels))
    return (0.0, n_samples / fs, ch1, ch0)


def mask_to_points(
    mask_ct: np.ndarray,
    summary: Dict,
    meta: Dict,
    max_points: int = 12000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 mask 转成散点坐标。

    返回：
        t_s:
            时间坐标，单位 s。
        ch_global:
            FORESEE 全局通道号。
    """
    idx_ch, idx_t = np.where(mask_ct > 0.5)
    if idx_ch.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # 如果点数太多，抽样以控制 PDF 大小。当前真实数据 mask 很稀疏，通常不会触发。
    if idx_ch.size > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(idx_ch.size, size=max_points, replace=False)
        idx_ch = idx_ch[keep]
        idx_t = idx_t[keep]

    fs = float(summary.get("fs", 125.0))
    ch0 = int(meta.get("channel_start", 0))
    t_s = idx_t.astype(np.float64) / fs
    ch_global = ch0 + idx_ch.astype(np.float64)
    return t_s, ch_global


# =============================================================================
# 4. 绘图辅助函数
# =============================================================================

def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    """添加不加粗 10 号面板编号。"""
    ax.text(
        -0.085, 1.06, letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_MAIN,
        fontweight="normal",
    )


def add_inpanel_label(ax: plt.Axes, text: str) -> None:
    """添加图内左上角标签，8 号，不加粗。"""
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


def overlay_mask_points(
    ax: plt.Axes,
    mask_ct: np.ndarray,
    summary: Dict,
    meta: Dict,
    color: str,
    label: str,
    size: float = 3.0,
) -> int:
    """在 DAS 背景图上叠加 mask 散点。"""
    t_s, ch = mask_to_points(mask_ct, summary, meta)
    if t_s.size == 0:
        ax.text(
            0.985, 0.08, "none",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE_SMALL,
            color=color,
            fontweight="normal",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )
        return 0

    ax.scatter(
        t_s,
        ch,
        s=size,
        c=color,
        marker=".",
        linewidths=0.0,
        alpha=0.90,
        label=replace_tau_p(label),
        rasterized=True,
    )
    return int(t_s.size)


# =============================================================================
# 5. FieldFig1：真实数据 overlay 主图
# =============================================================================

def plot_real_overlay_application() -> None:
    """绘制真实数据 overlay 主图。"""
    out_file = choose_output_file()
    item = load_output(out_file)

    raw = item["raw"]
    cleaned = item["cleaned"]
    candidate = item["candidate_mask"]
    rescued = item["rescued_mask"]
    final = item["final_mask"]
    summary = item["summary"]
    meta = item["meta"]

    extent = get_extent(raw, summary, meta)
    amp_vlim = robust_vlim(np.concatenate([raw.ravel(), cleaned.ravel()]), percentile=99.0)

    row_titles = [
        "Raw field DAS record",
        "MCD-only candidates overlaid",
        "τ-p rescued coherent events overlaid",
        "Final MCS-AND artifact mask overlaid",
        "Cleaned record",
    ]

    background_data = [raw, raw, raw, raw, cleaned]
    overlay_specs = [
        None,
        (candidate, "#d62728", "MCD-only candidates"),
        (rescued, "#1a9850", "τ-p rescued candidates"),
        (final, "#d62728", "Final artifact mask"),
        None,
    ]

    fig = plt.figure(figsize=(MAIN_FIG_WIDTH_IN, MAIN_FIG_HEIGHT_IN))
    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        width_ratios=[1.0, 0.035],
        height_ratios=[1, 1, 1, 1, 1],
        wspace=0.04,
        hspace=0.16,
    )

    for i, title in enumerate(row_titles):
        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(
            background_data[i],
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap="seismic",
            vmin=-amp_vlim,
            vmax=amp_vlim,
            interpolation="nearest",
            rasterized=True,
        )

        add_inpanel_label(ax, title)

        spec = overlay_specs[i]
        if spec is not None:
            mask_arr, color, label = spec
            n_points = overlay_mask_points(ax, mask_arr, summary, meta, color=color, label=label)
            ax.text(
                0.985, 0.92, f"n = {n_points}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=FONT_SIZE_SMALL,
                color=color,
                fontweight="normal",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )

        if i == 0:
            case_id = summary.get("case_id", out_file.stem)
            rescue_fraction = float(summary.get("rescued_mask_ratio_vs_candidate", np.nan))
            final_ratio = float(summary.get("final_mask_ratio_tau_p", np.nan))
            title_text = (
                f"FORESEE field example: {case_id}; "
                f"τ-p rescue fraction = {rescue_fraction:.2f}, "
                f"final mask ratio = {final_ratio:.2e}"
            )
            ax.set_title(replace_tau_p(title_text))
            add_panel_letter(ax, "a")

        if i == len(row_titles) - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])

        ax.set_ylabel("Channel")

        cax = fig.add_subplot(gs[i, 1])
        if i in (0, 4):
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL, length=2)
            cbar.set_label("Amplitude", fontsize=FONT_SIZE_SMALL, labelpad=3)
        else:
            cax.axis("off")

    # 为了兼容你原来引用的文件名，保存两份同内容图。
    save_figure(fig, "FieldFig1_real_mcsand_application")

    # 重新绘制一次并保存 overlay 明确命名版本。
    # 注意：matplotlib 的 fig 已被 save_figure 关闭，所以这里简单再次调用内部逻辑不方便；
    # 因此下面用复制文件方式由 main() 完成。


def duplicate_application_as_overlay() -> None:
    """把 application 图复制一份为 overlay 命名。"""
    import shutil

    src_png = FIG_DIR / "FieldFig1_real_mcsand_application.png"
    src_pdf = FIG_DIR / "FieldFig1_real_mcsand_application.pdf"
    dst_png = FIG_DIR / "FieldFig1_real_mcsand_overlay.png"
    dst_pdf = FIG_DIR / "FieldFig1_real_mcsand_overlay.pdf"

    if src_png.exists():
        shutil.copyfile(src_png, dst_png)
        print(f"[OK] {dst_png}")
    if src_pdf.exists():
        shutil.copyfile(src_pdf, dst_pdf)
        print(f"[OK] {dst_pdf}")


# =============================================================================
# 6. FieldFig2：真实 case 统计图
# =============================================================================

def plot_case_summary() -> None:
    """绘制真实 case 统计图，适合补充材料或正文辅助图。"""
    summary_csv = RESULT_DIR / "real_mcsand_case_summary.csv"
    if not summary_csv.exists():
        print(f"[跳过] 未找到统计表: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    df.to_csv(TABLE_DIR / "FieldFig2_case_summary_table.csv", index=False, encoding="utf-8-sig")

    labels = df["case_id"].astype(str).str.replace("real_case_", "", regex=False).tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(MAIN_FIG_WIDTH_IN, 2.35), constrained_layout=True)

    metrics = [
        ("candidate_mask_ratio", "MCD candidate ratio"),
        ("rescued_mask_ratio_vs_candidate", "τ-p rescued fraction"),
        ("final_mask_ratio_tau_p", "Final mask ratio"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        values = df[metric].values.astype(float)
        ax.bar(x, values, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(replace_tau_p(ylabel))
        ax.grid(True, axis="y", alpha=0.25)
        ymax = max(0.001, float(np.nanmax(values)) * 1.25)
        ax.set_ylim(0, ymax)

    add_panel_letter(axes[0], "a")
    add_panel_letter(axes[1], "b")
    add_panel_letter(axes[2], "c")

    save_figure(fig, "FieldFig2_real_case_summary")


# =============================================================================
# 7. 主流程
# =============================================================================

def main() -> None:
    setup_style()
    ensure_dirs()

    print("=" * 80)
    print("FORESEE 真实数据 MCS-AND overlay 图生成")
    print("=" * 80)
    print(f"结果目录: {RESULT_DIR}")
    print(f"输出目录: {FIG_DIR}")

    plot_real_overlay_application()
    duplicate_application_as_overlay()
    plot_case_summary()

    print("=" * 80)
    print("[完成] overlay 版本真实数据论文图已生成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
