# -*- coding: utf-8 -*-
"""
生成论文用合成 DAS 数据示例图（A4 版面规范版）。

本脚本用于从已经生成的合成数据集中自动挑选代表样本，并导出适合正文
或补充材料使用的示例图。

输出图件：
    1. Figure2_synthetic_examples_double.png / .pdf
       正文推荐使用的双图组合：
       - 左列：noncoherent_burst
       - 右列：moving
       - 三行：Noisy observation / Injected artifact / Ground-truth mask

    2. FigureS_synthetic_example_*.png / .pdf
       单类型示例图：
       - spike
       - noncoherent_burst
       - moving
       - narrowband
       - hard_case

版面约束：
    - A4 纸宽度约为 8.27 inch。
    - 单图宽度设置为 A4 宽度的 2/5，约 3.31 inch。
    - 双图组合宽度设置为 A4 宽度的 4/5，约 6.62 inch。
    - 全部图件默认不超过 A4 页面宽度，插入 Word/WPS 时建议使用“原始大小”。

字体约束：
    - 轴标签、轴名、图名：Times New Roman，10 pt。
    - 子图标注、色标刻度等较拥挤位置：Times New Roman，8 pt。

重要说明：
    - 图中文字全部使用英文，避免中文字体缺失导致乱码。
    - 代码中凡涉及 tau-p 的显示，统一写成希腊字母形式：τ-p。
    - narrowband 和 hard_case 建议作为补充材料中的边界示例，不建议作为正文主示例。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

# 使用无界面后端，适合在 Spyder、命令行或服务器环境下直接保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =============================================================================
# 1. 硬编码输入输出路径
# =============================================================================

# 合成数据根目录。脚本会递归搜索该目录下所有 .npz 文件。
DATASET_ROOT = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\合成数据")

# 示例图输出目录。
OUT_DIR = DATASET_ROOT / "paper_example_figures_a4"


# =============================================================================
# 2. 版面与字体参数
# =============================================================================

# A4 纸张宽度，单位 inch。
A4_WIDTH_IN = 8.27

# 单图：约 A4 宽度的 2/5。
SINGLE_FIG_WIDTH_IN = A4_WIDTH_IN * 2.0 / 5.0

# 双图组合：约 A4 宽度的 4/5。
DOUBLE_FIG_WIDTH_IN = A4_WIDTH_IN * 4.0 / 5.0

# 高度可根据行数略调。这里保证三行图不会过于拥挤。
SINGLE_FIG_HEIGHT_IN = 4.25
DOUBLE_FIG_HEIGHT_IN = 4.55

# 分辨率。600 dpi 适合投稿与 Word/WPS 插图。
DPI = 600

# 正文字体和小字号。
FONT_NAME = "Times New Roman"
FONT_SIZE_MAIN = 10
FONT_SIZE_SMALL = 8


def setup_matplotlib_style() -> None:
    """设置统一绘图风格。"""
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
        # 让数学符号更接近 Times 风格。
        "mathtext.fontset": "stix",
    })


# =============================================================================
# 3. 合成数据读取与样本选择
# =============================================================================

ARTIFACT_LABELS = {
    "spike": "Spike",
    "noncoherent_burst": "Noncoh. burst",
    "moving": "Moving artifact",
    "narrowband": "Narrowband",
    "hard_case": "Hard case",
}

# 正文推荐展示这两类：既有明显时空局部性，又是 MCS-AND 的优势类型。
MAIN_DOUBLE_TYPES = ("noncoherent_burst", "moving")

# 单图版全部导出，便于后续排版和补充材料使用。
SINGLE_TYPES = ("spike", "noncoherent_burst", "moving", "narrowband", "hard_case")

# 示例图优先选 SNR=5 dB。这个 SNR 既不是过难，也不是过易，比较适合展示结构。
TARGET_SNR_DB = 5.0


def _to_str(value) -> str:
    """把 npz 中可能出现的 bytes / ndarray 标量统一转成字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_str(value.item())
        if value.size == 1:
            return _to_str(value.reshape(-1)[0])
    return str(value)


def read_meta(npz: np.lib.npyio.NpzFile) -> Dict:
    """
    读取样本元信息。

    新版 synthetic_data_generator.py 通常会保存 artifact_type、snr_target_db、
    fs、dx_m 等字段；如果字段不存在，则尝试从 meta_json 中读取。
    """
    meta: Dict = {}

    if "meta_json" in npz.files:
        try:
            meta.update(json.loads(_to_str(npz["meta_json"])))
        except Exception:
            pass

    for key in ("artifact_type", "snr_target_db", "snr_actual_db", "fs", "dx_m"):
        if key in npz.files:
            value = npz[key]
            if key == "artifact_type":
                meta[key] = _to_str(value)
            else:
                try:
                    meta[key] = float(np.asarray(value).reshape(-1)[0])
                except Exception:
                    meta[key] = _to_str(value)

    return meta


def infer_artifact_type(path: Path, meta: Dict) -> str:
    """优先从元信息读取伪迹类型，失败时从文件名和路径中推断。"""
    if "artifact_type" in meta and meta["artifact_type"]:
        return str(meta["artifact_type"])

    text = str(path).lower()
    for name in SINGLE_TYPES:
        if name in text:
            return name
    return "unknown"


def infer_snr(path: Path, meta: Dict) -> float:
    """优先从元信息读取 SNR，失败时从文件名中粗略推断。"""
    if "snr_target_db" in meta:
        try:
            return float(meta["snr_target_db"])
        except Exception:
            pass

    name = path.name.lower()
    # 兼容类似 snr+05、snr-05、snr+5、snr_5 等命名。
    for token in ("snr+05", "snr+5", "snr_5", "snr5"):
        if token in name:
            return 5.0
    for token in ("snr-05", "snr-5", "snr_-5"):
        if token in name:
            return -5.0
    for value in (0.0, 10.0, 15.0):
        if f"snr+{int(value):02d}" in name or f"snr+{int(value)}" in name:
            return value
    return np.nan


def collect_npz_files() -> List[Path]:
    """递归收集合成数据根目录下所有 .npz 文件。"""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"合成数据目录不存在: {DATASET_ROOT}")
    return sorted(DATASET_ROOT.rglob("*.npz"))


def choose_sample(artifact_type: str, target_snr_db: float = TARGET_SNR_DB) -> Path:
    """
    为指定伪迹类型自动选择一个代表样本。

    选择规则：
        1. 类型必须匹配 artifact_type。
        2. 优先选择 snr_target_db 最接近 target_snr_db 的样本。
        3. 如果多个样本并列，选择 artifact_mask 占比适中的样本。

    这样做可以避免选到伪迹过小或过大的极端样本。
    """
    candidates = []
    for path in collect_npz_files():
        try:
            with np.load(path, allow_pickle=False) as npz:
                meta = read_meta(npz)
                art = infer_artifact_type(path, meta)
                if art != artifact_type:
                    continue

                snr = infer_snr(path, meta)
                snr_gap = abs(snr - target_snr_db) if np.isfinite(snr) else 999.0

                if "artifact_mask" in npz.files:
                    mask_ratio = float(np.mean(np.asarray(npz["artifact_mask"]) > 0))
                else:
                    mask_ratio = 0.0

                # mask_ratio_target 取 1% 左右，通常视觉上比较清楚。
                mask_gap = abs(mask_ratio - 0.01)
                candidates.append((snr_gap, mask_gap, path))
        except Exception:
            # 遇到损坏或字段不兼容文件时跳过，不中断整个流程。
            continue

    if not candidates:
        raise FileNotFoundError(f"未找到伪迹类型为 {artifact_type!r} 的 .npz 样本。")

    candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
    return candidates[0][2]


def load_sample(path: Path) -> Dict:
    """读取一个 .npz 样本，并返回绘图所需字段。"""
    with np.load(path, allow_pickle=False) as npz:
        required = ("data_noisy", "artifact_only", "artifact_mask")
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{path} 缺少必要字段: {missing}")

        meta = read_meta(npz)
        fs = float(meta.get("fs", npz["fs"] if "fs" in npz.files else 500.0))
        dx_m = float(meta.get("dx_m", npz["dx_m"] if "dx_m" in npz.files else 2.0))

        # 合成数据在不同脚本中可能保存为两种方向：
        #   1. [channel, time] = [200, 15000]
        #   2. [time, channel] = [15000, 200]
        # 论文绘图必须使用 [channel, time]，否则纵轴会误显示为 15000 个通道，
        # 横轴也会被误算成 0.4 s。这里统一转成 [channel, time]。
        noisy = standardize_channel_time(np.asarray(npz["data_noisy"], dtype=np.float32), fs)
        artifact = standardize_channel_time(np.asarray(npz["artifact_only"], dtype=np.float32), fs)
        mask = standardize_channel_time(np.asarray(npz["artifact_mask"], dtype=np.float32), fs)

        data = {
            "path": path,
            "meta": meta,
            "artifact_type": infer_artifact_type(path, meta),
            "snr_target_db": infer_snr(path, meta),
            "data_noisy": noisy,
            "artifact_only": artifact,
            "artifact_mask": mask,
            "fs": fs,
            "dx_m": dx_m,
        }

    return data


# =============================================================================
# 4. 绘图辅助函数
# =============================================================================

def standardize_channel_time(arr: np.ndarray, fs: float) -> np.ndarray:
    """
    将二维 DAS 数组统一为 [channel, time]。

    判断逻辑：
        - 合成论文数据通常为 200 个通道、30 s、500 Hz，即时间采样点约 15000。
        - 如果数组形状类似 [15000, 200]，第一维远大于第二维，说明它是
          [time, channel]，需要转置。
        - 如果数组形状类似 [200, 15000]，说明已经是 [channel, time]。

    这个函数解决示例图纵轴误显示为 15000 个通道的问题。
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"DAS 样本必须是二维数组，当前形状为 {arr.shape}")

    n0, n1 = arr.shape

    # 当第一维显著大于第二维，并且第一维对应的时长在 DAS 记录中合理，
    # 基本可以判定为 [time, channel]。
    duration_if_n0_is_time = n0 / fs
    if n0 > n1 and duration_if_n0_is_time >= 1.0:
        return arr.T

    return arr

def replace_tau_p(text: str) -> str:
    """统一把 tau-p 显示为 τ-p。"""
    return text.replace("tau-p", "τ-p").replace("tau_p", "τ-p")


def get_extent(data: np.ndarray, fs: float) -> Tuple[float, float, int, int]:
    """
    生成 imshow 的坐标范围。

    假定数据形状为 [channel, time]，即行是通道、列是时间采样点。
    """
    n_channels, n_samples = data.shape
    duration_s = n_samples / fs
    return (0.0, duration_s, n_channels, 0)


def robust_vlim(arrays: Sequence[np.ndarray], percentile: float = 99.0) -> float:
    """
    计算稳健颜色范围。

    使用多个数组绝对值的 percentile，避免少数极端值把色标拉得过宽。
    """
    values = []
    for arr in arrays:
        finite = np.asarray(arr)[np.isfinite(arr)]
        if finite.size:
            values.append(np.abs(finite).reshape(-1))
    if not values:
        return 1.0
    merged = np.concatenate(values)
    vmax = float(np.percentile(merged, percentile))
    return max(vmax, 1e-6)


def draw_one_column(
    fig: plt.Figure,
    spec: gridspec.SubplotSpec,
    sample: Dict,
    column_title: str,
    show_ylabel: bool,
) -> Tuple[List[plt.Axes], List[matplotlib.image.AxesImage]]:
    """
    在给定 GridSpec 区域内绘制一个三行样本列。

    三行依次为：
        1. Noisy observation
        2. Injected artifact
        3. Ground-truth mask
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=spec,
        hspace=0.18,
    )

    noisy = sample["data_noisy"]
    artifact = sample["artifact_only"]
    mask = sample["artifact_mask"]
    fs = sample["fs"]
    extent = get_extent(noisy, fs)

    amp_vlim = robust_vlim([noisy, artifact], percentile=99.0)

    rows = [
        ("Noisy observation", noisy, "seismic", -amp_vlim, amp_vlim),
        ("Injected artifact", artifact, "seismic", -amp_vlim, amp_vlim),
        ("Ground-truth mask", mask, "Reds", 0.0, 1.0),
    ]

    axes: List[plt.Axes] = []
    images: List[matplotlib.image.AxesImage] = []

    for row_idx, (row_title, arr, cmap, vmin, vmax) in enumerate(rows):
        ax = fig.add_subplot(inner[row_idx, 0])
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            rasterized=True,
        )

        if row_idx == 0:
            title = f"{column_title} (SNR = {sample['snr_target_db']:.0f} dB)"
            ax.set_title(replace_tau_p(title), pad=3)

        # 行标题放在图内左上角，用 8 号字，避免占用额外版面。
        ax.text(
            0.01, 0.95, row_title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_SMALL,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )

        if show_ylabel:
            ax.set_ylabel("Channel")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        if row_idx == 2:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        axes.append(ax)
        images.append(im)

    return axes, images


def add_row_colorbars(
    fig: plt.Figure,
    cbar_specs: Sequence[gridspec.SubplotSpec],
    row_images: Sequence[matplotlib.image.AxesImage],
) -> None:
    """为三行图分别添加很窄的共享色标。"""
    labels = ["Amplitude", "Amplitude", "Mask"]

    for idx, (spec, im, label) in enumerate(zip(cbar_specs, row_images, labels)):
        cax = fig.add_subplot(spec)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL, length=2)
        cbar.set_label(label, fontsize=FONT_SIZE_SMALL, labelpad=3)

        if idx == 2:
            cbar.set_ticks([0.0, 0.5, 1.0])


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    """给组合图添加 a、b 等面板编号。"""
    ax.text(
        -0.12, 1.10, letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_MAIN,
        fontweight="normal",
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    """同时保存 PNG 和 PDF。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"

    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] {png}")
    print(f"[OK] {pdf}")


# =============================================================================
# 5. 正文双图组合
# =============================================================================

def plot_double_example() -> None:
    """
    生成正文推荐的双图组合。

    总宽度约为 A4 宽度的 4/5，适合单栏较宽图或双栏论文中的跨栏图。
    """
    samples = [load_sample(choose_sample(t)) for t in MAIN_DOUBLE_TYPES]

    fig = plt.figure(figsize=(DOUBLE_FIG_WIDTH_IN, DOUBLE_FIG_HEIGHT_IN))
    outer = gridspec.GridSpec(
        3, 3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.14,
        hspace=0.18,
    )

    # 左右两列样本。色标使用右侧单独一列。
    left_axes, left_images = draw_one_column(
        fig,
        outer[:, 0],
        samples[0],
        ARTIFACT_LABELS.get(samples[0]["artifact_type"], samples[0]["artifact_type"]),
        show_ylabel=True,
    )
    right_axes, right_images = draw_one_column(
        fig,
        outer[:, 1],
        samples[1],
        ARTIFACT_LABELS.get(samples[1]["artifact_type"], samples[1]["artifact_type"]),
        show_ylabel=False,
    )

    add_panel_letter(left_axes[0], "a")
    add_panel_letter(right_axes[0], "b")

    # 三行共享右侧色标。这里用右列图像生成色标即可，因为左右列使用相同的显示逻辑。
    cbar_specs = [outer[0, 2], outer[1, 2], outer[2, 2]]
    add_row_colorbars(fig, cbar_specs, right_images)

    save_figure(fig, "Figure2_synthetic_examples_double")


# =============================================================================
# 6. 单类型示例图
# =============================================================================

def plot_single_example(artifact_type: str) -> None:
    """
    生成单类型示例图。

    单图宽度约为 A4 宽度的 2/5，便于作为补充图或正文小图。
    """
    sample = load_sample(choose_sample(artifact_type))

    fig = plt.figure(figsize=(SINGLE_FIG_WIDTH_IN, SINGLE_FIG_HEIGHT_IN))
    outer = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1.0, 0.055],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.08,
        hspace=0.18,
    )

    axes, images = draw_one_column(
        fig,
        outer[:, 0],
        sample,
        ARTIFACT_LABELS.get(sample["artifact_type"], sample["artifact_type"]),
        show_ylabel=True,
    )

    add_panel_letter(axes[0], "a")
    add_row_colorbars(fig, [outer[0, 1], outer[1, 1], outer[2, 1]], images)

    safe_name = artifact_type.replace("/", "_").replace("\\", "_")
    save_figure(fig, f"FigureS_synthetic_example_{safe_name}")


# =============================================================================
# 7. 主流程
# =============================================================================

def main() -> None:
    setup_matplotlib_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("生成 A4 规范合成数据示例图")
    print("=" * 80)
    print(f"输入目录: {DATASET_ROOT}")
    print(f"输出目录: {OUT_DIR}")
    print(f"单图宽度: {SINGLE_FIG_WIDTH_IN:.2f} inch，约 A4 宽度 2/5")
    print(f"双图宽度: {DOUBLE_FIG_WIDTH_IN:.2f} inch，约 A4 宽度 4/5")
    print("字体: Times New Roman，主字号 10 pt，小字号 8 pt")

    plot_double_example()

    for artifact_type in SINGLE_TYPES:
        plot_single_example(artifact_type)

    print("=" * 80)
    print("[完成] 合成数据示例图已生成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
