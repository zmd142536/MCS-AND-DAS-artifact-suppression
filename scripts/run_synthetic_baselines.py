# -*- coding: utf-8 -*-
"""
合成 DAS 数据 baseline 对比脚本（硬编码路径版，中文标注）。

本脚本用于完成论文最低版本 baseline：
    1. bandpass + median
    2. global SVD / rank reduction
    3. 合并已有 MCS-AND 与 MCS-AND ablation 结果

可选扩展：
    - DAS-N2N
    - DAS-N2N + MCS-AND

重要说明：
    传统去噪 baseline 通常只输出 cleaned waveform，不直接输出 artifact mask。
    为了与 MCS-AND 的 mask 指标可比，本脚本使用“残差阈值法”从
    residual = data_noisy - data_cleaned 中推导一个 baseline 伪迹 mask。
    因此：
        - waveform 指标（SNR gain / NRMSE / corr）是 baseline 的主要公平指标；
        - mask / event 指标可作为辅助参考，但需要在论文中说明其来源。

输出目录：
    D:\项目实验\积石山实验\DAS相关\去噪论文\Baseline_benchmark

主要输出：
    baseline_sample_metrics.csv          baseline 每样本指标
    baseline_summary_metrics.csv         baseline 汇总指标
    combined_sample_metrics.csv          baseline + MCS-AND 合并指标
    combined_summary_metrics.csv         baseline + MCS-AND 汇总指标
    baseline_report.json                 本次运行参数记录
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter, binary_dilation


# =============================================================================
# 硬编码路径
# =============================================================================

# 合成数据目录：必须包含 dataset_summary.csv 和 benchmark/ 子目录。
DATASET_ROOT = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\合成数据")

# 已有 MCS-AND 结果目录。这里使用你增加事件级指标后的 benchmark2。
MCSAND_RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark2")

# baseline 输出目录。
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\Baseline_benchmark")


# =============================================================================
# 运行规模控制
# =============================================================================

# 只处理 benchmark 样本，即有 artifact_mask 的五类伪迹样本。
# 若要同时检查 signal_rescue 假阳性，可改为 ("benchmark", "signal_rescue")。
PROCESS_CATEGORIES = ("benchmark",)

# 最多处理多少个样本。0 表示全量。
# 建议第一次试跑设为 20；正式跑设为 0。
MAX_PROCESS_FILES = 0

RANDOM_SEED = 42

# 是否保存少量 cleaned 输出用于画示例图。全量保存会占用大量磁盘。
SAVE_EXAMPLE_OUTPUTS = False
MAX_SAVE_EXAMPLES = 10


# =============================================================================
# baseline 参数
# =============================================================================

# bandpass + median 参数。
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 220.0
BANDPASS_ORDER = 4

# 空间中值滤波窗口，必须为奇数。5 表示用相邻 5 道做 median。
SPATIAL_MEDIAN_KERNEL = 5

# 全局 SVD 保留秩。rank 越小，去噪越强，但越容易损伤真实信号。
GLOBAL_SVD_RANK = 10

# 残差阈值法参数，用于从 baseline cleaned waveform 推导 artifact mask。
RESIDUAL_MAD_K = 6.0
RESIDUAL_MIN_PERCENTILE = 99.0

# 对残差 mask 做轻微膨胀，使其更接近事件窗口，减少单点断裂。
MASK_DILATE_TIME = 3
MASK_DILATE_CHANNEL = 1

# 事件级匹配容差。
EVENT_TOLERANCE_TIME_S = 0.02
EVENT_TOLERANCE_CHANNELS = 2


BASELINE_METHODS = (
    "bandpass_median",
    "global_svd",
)


# =============================================================================
# 可选：DAS-N2N 接口占位
# =============================================================================

# 如果你后续已经用 DAS-N2N 跑好了每个样本的 cleaned 输出，可以把目录填到这里。
# 目录组织建议与 DATASET_ROOT 一致，例如：
#   DASN2N_CLEANED_DIR / benchmark / spike / snr_-05 / sample_0000.npz
# 每个 npz 中至少包含 cleaned 或 data_cleaned 字段。
DASN2N_CLEANED_DIR =  Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\DASN2N_cleaned")#None

# 若设置为 True，脚本会尝试读取 DASN2N_CLEANED_DIR 中的预计算结果。
ENABLE_DASN2N_BASELINE = True   #False


# =============================================================================
# 基础工具函数
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x64 * x64) + 1e-30))


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    return float(10.0 * np.log10((np.mean(clean**2) + 1e-30) / (np.mean(noise**2) + 1e-30)))


def load_summary(root: Path) -> pd.DataFrame:
    path = root / "dataset_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 dataset_summary.csv: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "filepath" not in df.columns:
        raise ValueError("dataset_summary.csv 缺少 filepath 列。")
    return df


def resolve_file_path(root: Path, raw_path: str) -> Path:
    """兼容绝对路径、相对路径，以及搬动数据目录后的路径。"""
    p = Path(str(raw_path))
    if p.exists():
        return p
    if not p.is_absolute():
        p2 = root / p
        if p2.exists():
            return p2
    parts = list(p.parts)
    for key in ("benchmark", "signal_rescue"):
        if key in parts:
            idx = parts.index(key)
            p3 = root.joinpath(*parts[idx:])
            if p3.exists():
                return p3
    return p


def choose_rows(df: pd.DataFrame, max_files: int, seed: int) -> pd.DataFrame:
    if max_files == 0 or len(df) <= max_files:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=max_files, replace=False)
    return df.loc[np.sort(idx)].copy().reset_index(drop=True)


# =============================================================================
# baseline 方法实现
# =============================================================================

def baseline_bandpass_median(data: np.ndarray, fs: float) -> np.ndarray:
    """
    bandpass + spatial median baseline。

    处理逻辑：
        1. 每道去均值；
        2. 1-220 Hz 带通滤波；
        3. 沿通道方向做空间中值滤波，抑制单道/局部异常。

    注意：
        这是传统、可解释、轻量的 baseline，不是为所有伪迹最优调参。
    """
    data64 = data.astype(np.float64)
    mean_ch = np.mean(data64, axis=0, keepdims=True)
    x = data64 - mean_ch

    nyq = fs / 2.0
    low = max(BANDPASS_LOW_HZ / nyq, 0.001)
    high_hz = min(BANDPASS_HIGH_HZ, nyq * 0.98)
    high = min(high_hz / nyq, 0.98)
    if high <= low:
        filtered = x
    else:
        sos = signal.butter(BANDPASS_ORDER, [low, high], btype="band", output="sos")
        filtered = signal.sosfiltfilt(sos, x, axis=0)

    kernel = max(1, int(SPATIAL_MEDIAN_KERNEL))
    if kernel % 2 == 0:
        kernel += 1
    cleaned = median_filter(filtered, size=(1, kernel), mode="nearest")
    cleaned = cleaned + mean_ch
    return cleaned.astype(np.float32)


def baseline_global_svd(data: np.ndarray, rank: int = GLOBAL_SVD_RANK) -> np.ndarray:
    """
    全局 SVD / rank reduction baseline。

    处理逻辑：
        data(time, channel) 视作一个时空矩阵；
        做 SVD 后只保留前 rank 个奇异分量，作为低秩重建结果。

    局限：
        全局低秩方法容易把局部尖峰平滑掉，但也可能损伤真实波场；
        因此它是一个必要 baseline，而不是预期最优方法。
    """
    x = data.astype(np.float64)
    mean_ch = np.mean(x, axis=0, keepdims=True)
    centered = x - mean_ch

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    r = max(1, min(int(rank), len(s)))
    recon = (u[:, :r] * s[:r]) @ vt[:r, :] + mean_ch
    return recon.astype(np.float32)


def load_precomputed_dasn2n(path: Path) -> np.ndarray:
    """
    读取预先跑好的 DAS-N2N cleaned 输出。

    这是接口占位。真正使用前需要你先用 DAS-N2N 模型生成 cleaned 结果。
    """
    if DASN2N_CLEANED_DIR is None:
        raise RuntimeError("DASN2N_CLEANED_DIR 尚未配置。")
    parts = list(path.parts)
    for key in ("benchmark", "signal_rescue"):
        if key in parts:
            rel = Path(*parts[parts.index(key):])
            cleaned_path = Path(DASN2N_CLEANED_DIR) / rel
            break
    else:
        raise RuntimeError(f"无法从路径推断 DAS-N2N 相对路径: {path}")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"缺少 DAS-N2N cleaned 文件: {cleaned_path}")
    with np.load(cleaned_path, allow_pickle=False) as npz:
        for key in ("cleaned", "data_cleaned", "data_denoised"):
            if key in npz.files:
                return npz[key].astype(np.float32)
    raise KeyError(f"DAS-N2N 文件中缺少 cleaned/data_cleaned/data_denoised 字段: {cleaned_path}")


# =============================================================================
# 从 cleaned waveform 推导 baseline mask
# =============================================================================

def residual_to_mask(data_noisy: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    """
    使用残差阈值法推导 artifact mask。

    residual = abs(data_noisy - cleaned)
    对每个通道计算 median + K * MAD 阈值，并与全局 percentile 阈值取较大者。

    这样做的原因：
        bandpass、median、SVD 不是显式伪迹检测器；
        为了计算 precision/recall/F1，需要从其去噪残差中得到一个近似 mask。
    """
    residual = np.abs(data_noisy.astype(np.float64) - cleaned.astype(np.float64))
    n_time, n_channels = residual.shape
    mask = np.zeros((n_time, n_channels), dtype=bool)

    global_thr = float(np.percentile(residual, RESIDUAL_MIN_PERCENTILE))
    for ch in range(n_channels):
        r = residual[:, ch]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)) + 1e-12)
        robust_sigma = 1.4826 * mad
        thr = max(global_thr, med + RESIDUAL_MAD_K * robust_sigma)
        mask[:, ch] = r > thr

    if MASK_DILATE_TIME > 1 or MASK_DILATE_CHANNEL > 1:
        structure = np.ones((max(1, MASK_DILATE_TIME), max(1, MASK_DILATE_CHANNEL)), dtype=bool)
        mask = binary_dilation(mask, structure=structure)

    return mask


# =============================================================================
# 指标计算：逐点、事件级、波形级
# =============================================================================

def mask_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, mask_iou=iou)


def mask_to_events(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    将二值 mask 转为事件列表。

    返回：
        [(channel, start_sample, end_sample), ...]
    """
    mask = mask.astype(bool)
    events: List[Tuple[int, int, int]] = []
    n_time, n_channels = mask.shape
    for ch in range(n_channels):
        x = mask[:, ch]
        if not np.any(x):
            continue
        padded = np.r_[False, x, False]
        starts = np.where((~padded[:-1]) & padded[1:])[0]
        ends = np.where(padded[:-1] & (~padded[1:]))[0]
        for s, e in zip(starts, ends):
            events.append((ch, int(s), int(e)))
    return events


def events_match(a: Tuple[int, int, int], b: Tuple[int, int, int], tol_t: int, tol_ch: int) -> bool:
    ch_a, s_a, e_a = a
    ch_b, s_b, e_b = b
    if abs(ch_a - ch_b) > tol_ch:
        return False
    return (s_a - tol_t) < e_b and (s_b - tol_t) < e_a


def event_metrics(pred_mask: np.ndarray, truth_mask: np.ndarray, fs: float) -> Dict[str, float]:
    tol_t = int(round(EVENT_TOLERANCE_TIME_S * fs))
    tol_ch = int(EVENT_TOLERANCE_CHANNELS)
    pred_events = mask_to_events(pred_mask)
    truth_events = mask_to_events(truth_mask)

    matched_truth = 0
    for truth in truth_events:
        if any(events_match(truth, pred, tol_t, tol_ch) for pred in pred_events):
            matched_truth += 1

    matched_pred = 0
    for pred in pred_events:
        if any(events_match(truth, pred, tol_t, tol_ch) for truth in truth_events):
            matched_pred += 1

    n_truth = len(truth_events)
    n_pred = len(pred_events)
    event_precision = matched_pred / n_pred if n_pred else np.nan
    event_recall = matched_truth / n_truth if n_truth else np.nan
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if np.isfinite(event_precision) and np.isfinite(event_recall) and (event_precision + event_recall)
        else np.nan
    )
    return dict(
        n_truth_events=n_truth,
        n_pred_events=n_pred,
        matched_truth_events_tol=matched_truth,
        matched_pred_events_tol=matched_pred,
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=event_f1,
        hit_rate_tol=event_recall,
        event_tolerance_time_s=EVENT_TOLERANCE_TIME_S,
        event_tolerance_channels=EVENT_TOLERANCE_CHANNELS,
    )


def waveform_metrics(data_noisy: np.ndarray, cleaned: np.ndarray, clean_true: np.ndarray) -> Dict[str, float]:
    input_noise = data_noisy - clean_true
    output_noise = cleaned - clean_true
    input_snr = compute_snr_db(clean_true, input_noise)
    output_snr = compute_snr_db(clean_true, output_noise)
    nrmse = rms(output_noise) / (rms(clean_true) + 1e-30)
    amp_bias = float((np.max(np.abs(cleaned)) - np.max(np.abs(clean_true))) / (np.max(np.abs(clean_true)) + 1e-30))
    return dict(
        input_snr_db=input_snr,
        output_snr_db=output_snr,
        snr_gain_db=output_snr - input_snr,
        nrmse=nrmse,
        corr=safe_corrcoef(cleaned, clean_true),
        amplitude_bias=amp_bias,
    )


def compute_all_metrics(
    method: str,
    path: Path,
    row_meta: Dict,
    data_noisy: np.ndarray,
    cleaned: np.ndarray,
    data_clean: np.ndarray,
    artifact_mask: np.ndarray,
    signal_mask: np.ndarray,
    fs: float,
    dx_m: float,
) -> Dict:
    pred_mask = residual_to_mask(data_noisy, cleaned)
    point = mask_metrics(pred_mask, artifact_mask)
    event = event_metrics(pred_mask, artifact_mask, fs)
    wave = waveform_metrics(data_noisy, cleaned, data_clean)

    signal_total = int(np.sum(signal_mask))
    signal_removed = int(np.sum(pred_mask & signal_mask))
    false_removal = signal_removed / signal_total if signal_total else np.nan

    return {
        "method": method,
        "filepath": str(path),
        "category": row_meta.get("category", ""),
        "artifact_type": row_meta.get("artifact_type", row_meta.get("noise_type", "")),
        "snr_target_db": row_meta.get("snr_target_db", np.nan),
        "v_app_ms": row_meta.get("v_app_ms", np.nan),
        "fs": fs,
        "dx_m": dx_m,
        "n_time": int(data_noisy.shape[0]),
        "n_channels": int(data_noisy.shape[1]),
        "pred_mask_ratio": float(np.mean(pred_mask)),
        "artifact_mask_ratio": float(np.mean(artifact_mask)),
        "signal_mask_ratio": float(np.mean(signal_mask)),
        "signal_false_removal_rate": false_removal,
        "signal_preservation_rate": 1.0 - false_removal if np.isfinite(false_removal) else np.nan,
        **point,
        **event,
        **wave,
    }


# =============================================================================
# 主处理流程
# =============================================================================

def process_one_sample(row: pd.Series) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    path = Path(row["resolved_path"])
    with np.load(path, allow_pickle=False) as npz:
        data_noisy = npz["data_noisy"].astype(np.float32)
        data_clean = npz["data_clean"].astype(np.float32)
        artifact_mask = npz["artifact_mask"].astype(bool)
        signal_mask = npz["signal_mask"].astype(bool)
        fs = float(np.asarray(npz["fs"]))
        dx_m = float(np.asarray(npz["dx_m"]))

    outputs: Dict[str, np.ndarray] = {}
    metrics: List[Dict] = []

    cleaned_bp = baseline_bandpass_median(data_noisy, fs)
    outputs["bandpass_median"] = cleaned_bp
    metrics.append(compute_all_metrics(
        "bandpass_median", path, row.to_dict(),
        data_noisy, cleaned_bp, data_clean, artifact_mask, signal_mask, fs, dx_m,
    ))

    cleaned_svd = baseline_global_svd(data_noisy, rank=GLOBAL_SVD_RANK)
    outputs["global_svd"] = cleaned_svd
    metrics.append(compute_all_metrics(
        "global_svd", path, row.to_dict(),
        data_noisy, cleaned_svd, data_clean, artifact_mask, signal_mask, fs, dx_m,
    ))

    if ENABLE_DASN2N_BASELINE:
        cleaned_dl = load_precomputed_dasn2n(path)
        outputs["dasn2n"] = cleaned_dl
        metrics.append(compute_all_metrics(
            "dasn2n", path, row.to_dict(),
            data_noisy, cleaned_dl, data_clean, artifact_mask, signal_mask, fs, dx_m,
        ))
        # DAS-N2N + MCS-AND 的严格实现需要把 DAS-N2N 输出作为 MCS-AND 输入重新跑一遍。
        # 这里不伪造结果；后续若你提供 DAS-N2N 输出和 MCS-AND 二次处理脚本，再单独生成。

    return metrics, outputs


def summarize_results(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "category", "artifact_type", "snr_target_db"]
    value_cols = [
        "precision", "recall", "f1", "mask_iou",
        "event_precision", "event_recall", "event_f1", "hit_rate_tol",
        "signal_false_removal_rate", "signal_preservation_rate",
        "snr_gain_db", "nrmse", "corr", "amplitude_bias", "pred_mask_ratio",
    ]
    use_cols = [c for c in value_cols if c in sample_metrics.columns]
    grouped = (
        sample_metrics
        .groupby(group_cols, dropna=False)[use_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    grouped.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in grouped.columns]
    return grouped


def load_mcsand_metrics() -> pd.DataFrame:
    path = MCSAND_RESULT_DIR / "mcsand_sample_metrics.csv"
    if not path.exists():
        print(f"[警告] 未找到 MCS-AND 指标，无法合并: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def save_example_outputs(path: Path, outputs: Dict[str, np.ndarray], save_idx: int) -> None:
    save_dir = OUT_DIR / "baseline_example_outputs"
    ensure_dir(save_dir)
    payload = {f"{method}_cleaned": arr.astype(np.float32) for method, arr in outputs.items()}
    payload["source_file"] = np.array(str(path))
    np.savez_compressed(save_dir / f"baseline_output_{save_idx:04d}.npz", **payload)


def main() -> None:
    ensure_dir(OUT_DIR)
    print("=" * 80)
    print("合成 DAS baseline 对比")
    print("=" * 80)
    print(f"数据集目录      : {DATASET_ROOT}")
    print(f"MCS-AND 结果目录: {MCSAND_RESULT_DIR}")
    print(f"输出目录        : {OUT_DIR}")
    print(f"MAX_PROCESS_FILES = {MAX_PROCESS_FILES}")

    summary = load_summary(DATASET_ROOT)
    summary = summary[summary["category"].astype(str).isin(PROCESS_CATEGORIES)].copy()
    summary["resolved_path"] = summary["filepath"].apply(lambda p: str(resolve_file_path(DATASET_ROOT, p)))
    rows = choose_rows(summary, MAX_PROCESS_FILES, RANDOM_SEED)

    print(f"待处理样本数: {len(rows)}")
    all_metrics: List[Dict] = []
    saved = 0

    for i, row in rows.iterrows():
        path = Path(row["resolved_path"])
        if not path.exists():
            print(f"[警告] 文件不存在，跳过: {path}")
            continue

        metrics, outputs = process_one_sample(row)
        all_metrics.extend(metrics)

        if SAVE_EXAMPLE_OUTPUTS and saved < MAX_SAVE_EXAMPLES:
            save_example_outputs(path, outputs, saved)
            saved += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            print(f"  已处理 {i + 1}/{len(rows)} 个样本，指标行={len(all_metrics)}")

    baseline_metrics = pd.DataFrame(all_metrics)
    if baseline_metrics.empty:
        raise RuntimeError("baseline 没有生成任何指标。")

    baseline_summary = summarize_results(baseline_metrics)
    baseline_metrics_csv = OUT_DIR / "baseline_sample_metrics.csv"
    baseline_summary_csv = OUT_DIR / "baseline_summary_metrics.csv"
    baseline_metrics.to_csv(baseline_metrics_csv, index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(baseline_summary_csv, index=False, encoding="utf-8-sig")

    # 合并 MCS-AND / ablation 指标，方便统一画图。
    mcsand_metrics = load_mcsand_metrics()
    if not mcsand_metrics.empty:
        # 只合并 benchmark，避免 signal_rescue 与 baseline 范围不一致。
        if "category" in mcsand_metrics.columns:
            mcsand_metrics = mcsand_metrics[mcsand_metrics["category"].astype(str).isin(PROCESS_CATEGORIES)].copy()
        common_cols = sorted(set(baseline_metrics.columns) | set(mcsand_metrics.columns))
        combined = pd.concat(
            [
                baseline_metrics.reindex(columns=common_cols),
                mcsand_metrics.reindex(columns=common_cols),
            ],
            ignore_index=True,
            sort=False,
        )
        combined_summary = summarize_results(combined)
        combined.to_csv(OUT_DIR / "combined_sample_metrics.csv", index=False, encoding="utf-8-sig")
        combined_summary.to_csv(OUT_DIR / "combined_summary_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        combined = baseline_metrics
        combined_summary = baseline_summary

    report = {
        "dataset_root": str(DATASET_ROOT),
        "mcsand_result_dir": str(MCSAND_RESULT_DIR),
        "out_dir": str(OUT_DIR),
        "process_categories": list(PROCESS_CATEGORIES),
        "max_process_files": MAX_PROCESS_FILES,
        "baseline_methods": list(BASELINE_METHODS),
        "bandpass_low_hz": BANDPASS_LOW_HZ,
        "bandpass_high_hz": BANDPASS_HIGH_HZ,
        "spatial_median_kernel": SPATIAL_MEDIAN_KERNEL,
        "global_svd_rank": GLOBAL_SVD_RANK,
        "residual_mad_k": RESIDUAL_MAD_K,
        "residual_min_percentile": RESIDUAL_MIN_PERCENTILE,
        "mask_dilate_time": MASK_DILATE_TIME,
        "mask_dilate_channel": MASK_DILATE_CHANNEL,
        "event_tolerance_time_s": EVENT_TOLERANCE_TIME_S,
        "event_tolerance_channels": EVENT_TOLERANCE_CHANNELS,
        "enable_dasn2n_baseline": ENABLE_DASN2N_BASELINE,
        "n_processed_samples": int(baseline_metrics["filepath"].nunique()),
    }
    (OUT_DIR / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[完成] baseline 指标已输出：")
    print(f"  baseline 样本指标: {baseline_metrics_csv}")
    print(f"  baseline 汇总指标: {baseline_summary_csv}")
    print(f"  合并样本指标      : {OUT_DIR / 'combined_sample_metrics.csv'}")
    print(f"  合并汇总指标      : {OUT_DIR / 'combined_summary_metrics.csv'}")

    overview = (
        combined
        .groupby("method", dropna=False)
        .agg(
            point_f1=("f1", "mean"),
            event_precision=("event_precision", "mean"),
            event_recall=("event_recall", "mean"),
            hit_rate=("hit_rate_tol", "mean"),
            snr_gain=("snr_gain_db", "mean"),
            nrmse=("nrmse", "mean"),
            n_samples=("filepath", "nunique"),
        )
        .reset_index()
    )
    print("\n总体概览：")
    print(overview.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
