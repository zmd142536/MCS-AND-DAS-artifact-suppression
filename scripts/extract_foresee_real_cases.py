# -*- coding: utf-8 -*-
"""
FORESEE 城市 DAS 真实数据裁剪脚本（硬编码路径版，中文详细注释）。

目的：
    将 PubDAS / FORESEE 下载得到的 10 分钟 HDF5 大文件裁剪成若干小片段，
    并保存为后续 MCS-AND 真实数据实验可以直接读取的 .npz 文件。

FORESEE 数据说明：
    - PubDAS 上的 HDF5 文件已经过官方预处理。
    - 每个 HDF5 文件包含两个数据集：
        raw        : DAS 数据，形状为 [channel, time] = [2137, 75000]
        timestamp  : Unix timestamp，长度为 75000
    - 采样率：125 Hz
    - 通道间距：2 m
    - 每个文件时长：10 min
    - 数据单位：与 strain rate 成正比，官方保存为 float16

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases

    每个 case 会输出：
        real_case_XXX.npz
        real_case_XXX_quicklook.png
        real_case_XXX_meta.json

建议流程：
    1. 先运行本脚本，生成 4 个真实数据小片段。
    2. 检查 quicklook 图，确认片段中是否有明显交通扰动/局部异常。
    3. 再用后续 run_real_mcsand_cases.py 对这些 .npz 运行 MCS-AND。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. 硬编码路径
# =============================================================================

# 真实数据 HDF5 文件所在目录。
RAW_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\raw")

# FORESEE 小文件所在目录，包括 readme.txt、foresee_ch_loc.txt 等。
DOC_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据")

# 输出目录。
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases")

# 通道经纬度文件。该文件对应官方预处理后的 2137 个通道。
CHANNEL_LOC_FILE = DOC_DIR / "foresee_ch_loc.txt"


# =============================================================================
# 2. FORESEE 固定参数
# =============================================================================

FS = 125.0          # HDF5 文件已经从原始 500 Hz 降采样到 125 Hz
DX_M = 2.0         # 通道间距 2 m
GAUGE_LENGTH_M = 10.0
N_TOTAL_CHANNELS = 2137


# =============================================================================
# 3. 裁剪片段配置
# =============================================================================

@dataclass
class CaseConfig:
    """一个真实数据裁剪片段的配置。"""

    case_id: str
    filename: str
    channel_start: int
    channel_end: int
    time_start_s: float
    duration_s: float
    description: str


# 说明：
#   channel_start/channel_end 使用 Python 左闭右开区间，即 [start, end)。
#   例如 1460 到 1860 表示一共 400 个通道。
#
# 为什么选 1460-1860：
#   该区间是 FORESEE 城市道路交通研究中常用的代表性通道段之一，
#   适合展示城市交通诱发扰动和局部异常。
#
# 为什么选 60 s：
#   真实数据验证图不需要太长，60 s 足够展示时空结构，也便于后续算法快速运行。
CASE_CONFIGS: List[CaseConfig] = [
    CaseConfig(
        case_id="real_case_001_noon",
        filename="FORESEE_UTC_20190501_120043.hdf5",
        channel_start=1460,
        channel_end=1860,
        time_start_s=0.0,
        duration_s=60.0,
        description="FORESEE noon traffic segment, channels 1460-1860, first 60 s.",
    ),
    CaseConfig(
        case_id="real_case_002_noon_late",
        filename="FORESEE_UTC_20190501_121043.hdf5",
        channel_start=1460,
        channel_end=1860,
        time_start_s=0.0,
        duration_s=60.0,
        description="FORESEE noon traffic segment, continuous later 10-min file, first 60 s.",
    ),
    CaseConfig(
        case_id="real_case_003_evening",
        filename="FORESEE_UTC_20190501_175043.hdf5",
        channel_start=1460,
        channel_end=1860,
        time_start_s=0.0,
        duration_s=60.0,
        description="FORESEE evening traffic segment, channels 1460-1860, first 60 s.",
    ),
    CaseConfig(
        case_id="real_case_004_evening_late",
        filename="FORESEE_UTC_20190501_180043.hdf5",
        channel_start=1460,
        channel_end=1860,
        time_start_s=0.0,
        duration_s=60.0,
        description="FORESEE evening traffic segment, continuous later 10-min file, first 60 s.",
    ),
]


# =============================================================================
# 4. 绘图参数
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "normal",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 300,
})


# =============================================================================
# 5. 工具函数
# =============================================================================

def ensure_dirs() -> None:
    """创建输出目录。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_channel_locations() -> np.ndarray:
    """
    读取 FORESEE 通道经纬度。

    返回：
        loc.shape = [2137, 2]
        第一列 latitude，第二列 longitude。
    """
    if not CHANNEL_LOC_FILE.exists():
        print(f"[警告] 未找到通道位置文件: {CHANNEL_LOC_FILE}")
        return np.empty((0, 2), dtype=np.float64)

    loc = np.loadtxt(CHANNEL_LOC_FILE)
    if loc.ndim != 2 or loc.shape[1] < 2:
        print(f"[警告] 通道位置文件格式异常: {CHANNEL_LOC_FILE}")
        return np.empty((0, 2), dtype=np.float64)

    return loc[:, :2]


def robust_scale_for_display(data: np.ndarray, percentile: float = 99.0) -> float:
    """
    计算 quicklook 图的稳健色标范围。

    使用绝对值的 99 分位，避免少数极端值让整幅图过暗。
    """
    values = data[np.isfinite(data)]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(values), percentile))
    return max(vmax, 1e-8)


def preprocess_for_npz(data: np.ndarray) -> np.ndarray:
    """
    对真实数据做轻量预处理。

    这里只做每个通道去均值，不做滤波、不做归一化。
    原因：
        1. MCS-AND 后续流程应尽量接近真实输入；
        2. 不提前改变频率结构，避免影响真实数据验证的解释；
        3. 去均值可以去除每个通道的静态偏置，通常是安全的。
    """
    data = data.astype(np.float32, copy=False)
    data = data - np.mean(data, axis=1, keepdims=True)
    return data


def utc_string_from_timestamp(ts: float) -> str:
    """把 Unix timestamp 转为 UTC 字符串。"""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_hdf5_case(cfg: CaseConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    从一个 HDF5 文件中读取并裁剪数据片段。

    返回：
        data_raw: shape = [n_channels, n_samples]
        timestamp: shape = [n_samples]
        meta: 元信息字典
    """
    h5_path = RAW_DIR / cfg.filename
    if not h5_path.exists():
        raise FileNotFoundError(f"未找到 HDF5 文件: {h5_path}")

    sample_start = int(round(cfg.time_start_s * FS))
    sample_end = sample_start + int(round(cfg.duration_s * FS))

    with h5py.File(h5_path, "r") as f:
        if "raw" not in f or "timestamp" not in f:
            raise KeyError(f"{h5_path} 必须包含 raw 和 timestamp 数据集，当前 keys={list(f.keys())}")

        raw_ds = f["raw"]
        ts_ds = f["timestamp"]

        if raw_ds.ndim != 2:
            raise ValueError(f"raw 数据必须是二维数组，当前 shape={raw_ds.shape}")

        n_channels, n_samples = raw_ds.shape
        if n_channels != N_TOTAL_CHANNELS:
            print(f"[警告] 通道数不是预期的 {N_TOTAL_CHANNELS}，实际为 {n_channels}")

        if cfg.channel_start < 0 or cfg.channel_end > n_channels or cfg.channel_start >= cfg.channel_end:
            raise ValueError(
                f"通道范围不合法: [{cfg.channel_start}, {cfg.channel_end}), "
                f"文件通道数={n_channels}"
            )

        if sample_start < 0 or sample_end > n_samples:
            raise ValueError(
                f"时间范围不合法: [{cfg.time_start_s}, {cfg.time_start_s + cfg.duration_s}] s, "
                f"文件时长={n_samples / FS:.1f} s"
            )

        # h5py 切片不会一次读完整文件，只读需要的通道和时间窗。
        data_raw = raw_ds[cfg.channel_start:cfg.channel_end, sample_start:sample_end].astype(np.float32)
        timestamp = ts_ds[sample_start:sample_end].astype(np.float64)

    meta = {
        "case_id": cfg.case_id,
        "source": "PubDAS FORESEE",
        "source_file": str(h5_path),
        "filename": cfg.filename,
        "fs": FS,
        "dx_m": DX_M,
        "gauge_length_m": GAUGE_LENGTH_M,
        "channel_start": cfg.channel_start,
        "channel_end": cfg.channel_end,
        "n_channels": int(data_raw.shape[0]),
        "time_start_s": cfg.time_start_s,
        "duration_s": cfg.duration_s,
        "n_samples": int(data_raw.shape[1]),
        "utc_start": utc_string_from_timestamp(timestamp[0]),
        "utc_end": utc_string_from_timestamp(timestamp[-1]),
        "description": cfg.description,
        "array_orientation": "[channel, time]",
    }

    return data_raw, timestamp, meta


def save_npz_case(cfg: CaseConfig, channel_locations: np.ndarray) -> Path:
    """
    裁剪并保存一个真实数据 case。
    """
    data_raw, timestamp, meta = read_hdf5_case(cfg)
    data_proc = preprocess_for_npz(data_raw)

    # 裁剪对应通道的经纬度。
    if channel_locations.shape[0] >= meta["channel_end"]:
        loc_subset = channel_locations[meta["channel_start"]:meta["channel_end"], :]
    else:
        loc_subset = np.empty((0, 2), dtype=np.float64)

    out_npz = OUT_DIR / f"{cfg.case_id}.npz"
    np.savez_compressed(
        out_npz,
        data_raw=data_raw,
        data_proc=data_proc,
        timestamp=timestamp,
        channel_locations=loc_subset,
        fs=np.float32(FS),
        dx_m=np.float32(DX_M),
        gauge_length_m=np.float32(GAUGE_LENGTH_M),
        channel_start=np.int32(meta["channel_start"]),
        channel_end=np.int32(meta["channel_end"]),
        case_id=np.array(cfg.case_id),
        source=np.array("PubDAS FORESEE"),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False, indent=2)),
    )

    out_json = OUT_DIR / f"{cfg.case_id}_meta.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    plot_quicklook(cfg, data_proc, meta)

    print(f"[OK] {out_npz}")
    print(f"[OK] {out_json}")
    return out_npz


def plot_quicklook(cfg: CaseConfig, data: np.ndarray, meta: Dict) -> None:
    """
    为裁剪后的真实数据片段生成 quicklook 图。

    注意：
        这只是快速检查图，不是最终论文图。
        最终论文真实数据图后面会用 plot_real_mcsand_figures.py 单独生成。
    """
    n_channels, n_samples = data.shape
    extent = (0.0, n_samples / FS, meta["channel_end"], meta["channel_start"])
    vmax = robust_scale_for_display(data, percentile=99.0)

    fig, ax = plt.subplots(figsize=(6.6, 2.6), constrained_layout=True)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(cfg.case_id)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    cbar = fig.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label("Amplitude")
    cbar.ax.tick_params(labelsize=8)

    out_png = OUT_DIR / f"{cfg.case_id}_quicklook.png"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"[OK] {out_png}")


def write_summary(saved_files: List[Path]) -> None:
    """保存一个总览 JSON，方便后续脚本读取。"""
    summary = {
        "raw_dir": str(RAW_DIR),
        "doc_dir": str(DOC_DIR),
        "out_dir": str(OUT_DIR),
        "fs": FS,
        "dx_m": DX_M,
        "n_cases": len(saved_files),
        "case_files": [str(p) for p in saved_files],
        "case_configs": [asdict(cfg) for cfg in CASE_CONFIGS],
    }
    out = OUT_DIR / "foresee_cases_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out}")


# =============================================================================
# 6. 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("FORESEE 真实 DAS 数据裁剪")
    print("=" * 80)
    print(f"原始 HDF5 目录: {RAW_DIR}")
    print(f"辅助文件目录: {DOC_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"采样率: {FS} Hz")
    print(f"通道间距: {DX_M} m")

    channel_locations = load_channel_locations()
    if channel_locations.size:
        print(f"通道位置文件读取成功: {channel_locations.shape}")

    saved_files: List[Path] = []
    for cfg in CASE_CONFIGS:
        print("-" * 80)
        print(f"处理 {cfg.case_id}: {cfg.filename}")
        saved_files.append(save_npz_case(cfg, channel_locations))

    write_summary(saved_files)

    print("=" * 80)
    print("[完成] FORESEE 真实数据裁剪完成。")
    print("下一步：检查 quicklook 图，然后运行真实数据 MCS-AND 脚本。")
    print("=" * 80)


if __name__ == "__main__":
    main()
