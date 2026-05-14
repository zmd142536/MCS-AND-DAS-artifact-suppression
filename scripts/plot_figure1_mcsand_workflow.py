# -*- coding: utf-8 -*-
"""
Figure 1：MCS-AND 方法流程图生成脚本（A4 宽度规范版）。

用途：
    生成论文正文 Figure 1，展示 MCS-AND 的整体处理流程。

版面规范：
    - 图宽约为 A4 纸宽度的 4/5；
    - Times New Roman；
    - 主字体 10 pt；
    - 小标注 8 pt；
    - 所有文字不加粗；
    - 使用希腊字母 τ-p，而不是 tau-p；
    - 输出 PNG 和 PDF。

输出目录：
    D:\项目实验\积石山实验\DAS相关\去噪论文\论文主图\Figure1_workflow

输出文件：
    Figure1_MCSAND_workflow.png
    Figure1_MCSAND_workflow.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# =============================================================================
# 1. 输出路径和版面参数
# =============================================================================

OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\论文主图\Figure1_workflow")

A4_WIDTH_IN = 8.27
FIG_WIDTH_IN = A4_WIDTH_IN * 4.0 / 5.0
FIG_HEIGHT_IN = 4.05
DPI = 600

FONT_NAME = "Times New Roman"
FONT_SIZE_MAIN = 10
FONT_SIZE_SMALL = 8


# =============================================================================
# 2. 绘图风格
# =============================================================================

def setup_style() -> None:
    """设置字体、字号和线宽。"""
    plt.rcParams.update({
        "font.family": FONT_NAME,
        "font.weight": "normal",
        "font.size": FONT_SIZE_MAIN,
        "axes.titlesize": FONT_SIZE_MAIN,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZE_MAIN,
        "axes.labelweight": "normal",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "mathtext.fontset": "stix",
    })


def replace_tau_p(text: str) -> str:
    """统一使用 τ-p 写法。"""
    return text.replace("tau-p", "τ-p").replace("tau_p", "τ-p")


# =============================================================================
# 3. 基础绘图函数
# =============================================================================

def add_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "#F7F7F7",
    edgecolor: str = "#333333",
) -> FancyBboxPatch:
    """
    添加圆角流程框。

    参数：
        xy:
            左下角坐标，使用归一化画布坐标。
        width, height:
            框宽和框高。
        title:
            主文字，10 号。
        subtitle:
            说明文字，8 号。
    """
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height * (0.62 if subtitle else 0.50),
        replace_tau_p(title),
        ha="center",
        va="center",
        fontsize=FONT_SIZE_MAIN,
        fontweight="normal",
        color="#111111",
    )

    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.30,
            replace_tau_p(subtitle),
            ha="center",
            va="center",
            fontsize=FONT_SIZE_SMALL,
            fontweight="normal",
            color="#333333",
        )

    return patch


def add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#333333",
    rad: float = 0.0,
) -> None:
    """添加箭头。"""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def add_stage_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    """添加分组阶段标签，8 号，不加粗。"""
    ax.text(
        x,
        y,
        replace_tau_p(text),
        ha="center",
        va="center",
        fontsize=FONT_SIZE_SMALL,
        fontweight="normal",
        color="#444444",
    )


def add_elbow_arrow(
    ax: plt.Axes,
    points: Tuple[Tuple[float, float], ...],
    color: str = "#333333",
) -> None:
    """
    添加折线箭头。

    points 至少包含 2 个点。前面的线段不带箭头，最后一段带箭头。
    这样可以让箭头绕开文字和框体，避免交叉压字。
    """
    if len(points) < 2:
        return

    # 前 N-1 段画普通线。
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=0.8,
            solid_capstyle="round",
        )

    # 最后一段画箭头。
    add_arrow(ax, points[-2], points[-1], color=color)


def add_panel_background(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    label: str,
    facecolor: str,
) -> None:
    """
    添加浅色阶段背景。

    这不是数据框，只用于把流程分成 candidate detection、
    artifact decision、signal-preserving repair 三个阶段。
    """
    x, y = xy
    rect = Rectangle(
        (x, y),
        width,
        height,
        linewidth=0.0,
        facecolor=facecolor,
        alpha=0.42,
        zorder=-10,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.012,
        y + height - 0.028,
        replace_tau_p(label),
        ha="left",
        va="center",
        fontsize=FONT_SIZE_SMALL,
        fontweight="normal",
        color="#444444",
    )


# =============================================================================
# 4. 主流程图
# =============================================================================

def plot_workflow() -> None:
    """绘制 Figure 1 方法流程图。"""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 阶段背景。颜色保持克制，避免变成花哨的海报。
    add_panel_background(ax, (0.04, 0.68), 0.92, 0.25, "Candidate event detection", "#E8F1FA")
    add_panel_background(ax, (0.04, 0.40), 0.92, 0.22, "Robust artifact decision", "#F6EFE7")
    add_panel_background(ax, (0.04, 0.12), 0.92, 0.22, "Signal-preserving repair and output", "#EAF4EA")

    # 第一行：输入、候选切片、多频带特征。
    b_raw = add_box(
        ax,
        (0.06, 0.75),
        0.22,
        0.11,
        "Raw DAS record",
        "channel × time",
        facecolor="#FFFFFF",
    )
    b_sta = add_box(
        ax,
        (0.39, 0.75),
        0.22,
        0.11,
        "STA/LTA slicing",
        "transient candidates",
        facecolor="#FFFFFF",
    )
    b_feat = add_box(
        ax,
        (0.72, 0.75),
        0.22,
        0.11,
        "Multi-band features",
        "RMS, duration, spectra",
        facecolor="#FFFFFF",
    )

    add_arrow(ax, (0.28, 0.805), (0.39, 0.805))
    add_arrow(ax, (0.61, 0.805), (0.72, 0.805))

    # 第二行：稳健 MCD 判别和 τ-p rescue。
    b_mcd = add_box(
        ax,
        (0.15, 0.47),
        0.25,
        0.11,
        "Robust MCD scoring",
        "Mahalanobis distance",
        facecolor="#FFFFFF",
    )
    b_tau = add_box(
        ax,
        (0.58, 0.47),
        0.27,
        0.11,
        "τ-p coherence rescue",
        "preserve coherent events",
        facecolor="#FFFFFF",
    )

    add_elbow_arrow(ax, ((0.83, 0.75), (0.83, 0.665), (0.275, 0.665), (0.275, 0.58)))
    add_arrow(ax, (0.40, 0.525), (0.58, 0.525))

    # 第三行：最终 mask、低秩修补、输出。
    b_mask = add_box(
        ax,
        (0.06, 0.19),
        0.22,
        0.10,
        "Final artifact mask",
        "",
        facecolor="#FFFFFF",
    )
    b_lr = add_box(
        ax,
        (0.39, 0.19),
        0.22,
        0.10,
        "Local low-rank repair",
        "",
        facecolor="#FFFFFF",
    )
    b_out = add_box(
        ax,
        (0.72, 0.19),
        0.20,
        0.10,
        "Cleaned DAS record",
        "",
        facecolor="#FFFFFF",
    )

    add_elbow_arrow(ax, ((0.715, 0.47), (0.715, 0.355), (0.17, 0.355), (0.17, 0.29)))
    add_arrow(ax, (0.28, 0.24), (0.39, 0.24))
    add_arrow(ax, (0.61, 0.24), (0.72, 0.24))

    # 审计输出，强调 training-free / interpretable。
    b_log = add_box(
        ax,
        (0.73, 0.36),
        0.20,
        0.055,
        "Audit log",
        "scores, masks, events",
        facecolor="#FFFFFF",
        edgecolor="#666666",
    )
    add_arrow(ax, (0.78, 0.47), (0.80, 0.415), rad=0.0)

    # 小注释：training-free。
    ax.text(
        0.50,
        0.055,
        "Training-free event-level artifact surgery for DAS records",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_SMALL,
        fontweight="normal",
        color="#444444",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "Figure1_MCSAND_workflow.png"
    pdf = OUT_DIR / "Figure1_MCSAND_workflow.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


# =============================================================================
# 5. 主程序
# =============================================================================

def main() -> None:
    setup_style()
    print("=" * 80)
    print("生成 Figure 1：MCS-AND 方法流程图")
    print("=" * 80)
    print(f"输出目录: {OUT_DIR}")
    plot_workflow()
    print("=" * 80)
    print("[完成] Figure 1 已生成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
