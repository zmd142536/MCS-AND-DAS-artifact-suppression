# -*- coding: utf-8 -*-
"""
合成 DAS 数据完整 MCS-AND benchmark 脚本（硬编码路径版）。

功能：
1. 读取合成数据集 dataset_summary.csv 和 .npz 样本。
2. 读取 STA/LTA 校准得到的 best_sta_lta_params.txt。
3. 从候选事件中提取多频带特征，拟合 Yeo-Johnson + MCD 模型。
4. 运行四组方法：
      mcd_only
      mcd_plain_semblance
      mcd_tau_p
      mcd_tau_p_lowrank
5. 与 artifact_mask / signal_mask 真值比较，输出事件级和波形级指标。

注意：
    本脚本默认不保存每个样本的 cleaned 矩阵，避免一次性写出大量文件。
    如需保存少量结果用于画图，可设置 SAVE_OUTPUT_NPZ = True 并调小 MAX_SAVE_SAMPLES。
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import PowerTransformer


# =============================================================================
# 硬编码路径
# =============================================================================

DATASET_ROOT = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\合成数据")
STA_LTA_PARAM_PATH = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\STALTAqiepian\best_sta_lta_params.txt")
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark")


# =============================================================================
# 运行规模控制
# =============================================================================

# MCD 拟合最多使用多少个样本。0 表示全部样本。
MAX_TRAIN_FILES = 300

# benchmark 正式处理最多使用多少个样本。0 表示全部样本。
MAX_PROCESS_FILES = 0

# 每隔多少道运行一次。正式论文建议为 1；快速检查可设为 2 或 4。
CHANNEL_STRIDE = 1

RANDOM_SEED = 42

# 是否保存每个样本的预测 mask 和 cleaned data。全量保存会占用较多磁盘。
SAVE_OUTPUT_NPZ = False
MAX_SAVE_SAMPLES = 20


# =============================================================================
# MCS-AND 参数
# =============================================================================

MCD_FEATURE_COLS = ["zcr", "centroid", "vhigh_low", "high_low", "mid_low", "rms", "dur_s"]

MAHAL_ALPHA = 0.05
MAHAL_ALPHA_LIST = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]

RUNTIME_PARAMS = dict(
    COHERENCE_K=5,
    SEMBLANCE_THR=0.30,
    V_APP_MIN=100.0,
    V_APP_MAX=5000.0,
    N_V_SCAN=41,
    LOWRANK_RANK=2,
    LOWRANK_WIN_PAD_S=0.20,
    LOWRANK_NEIGHBOR_K=5,
    LOWRANK_ITER=3,
)


METHODS = (
    "mcd_only",
    "mcd_plain_semblance",
    "mcd_tau_p",
    "mcd_tau_p_lowrank",
)


# =============================================================================
# 基础工具
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def compute_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    return float(10.0 * np.log10((np.mean(clean.astype(np.float64) ** 2) + 1e-30) /
                                  (np.mean(noise.astype(np.float64) ** 2) + 1e-30)))


def parse_key_value_file(path: Path) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if not path.exists():
        print(f"[警告] 未找到 STA/LTA 参数文件，使用默认参数: {path}")
        return params
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            params[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return params


def load_slicing_params() -> Dict[str, float]:
    params = dict(
        sta_s=0.05,
        lta_s=0.50,
        thr_on=2.0,
        thr_off=1.2,
        min_dur_s=0.02,
        merge_gap_s=0.02,
        pre_pad_s=0.05,
        post_pad_s=0.10,
    )
    params.update(parse_key_value_file(STA_LTA_PARAM_PATH))
    return params


def adaptive_bands(fs: float) -> Dict[str, Tuple[float, float]]:
    nyq = fs / 2.0
    return dict(
        low=(0.5, min(10.0, nyq * 0.98)),
        mid=(10.0, min(50.0, nyq * 0.98)),
        high=(50.0, min(100.0, nyq * 0.98)),
        vhigh=(100.0, nyq * 0.98),
    )


def load_summary(root: Path) -> pd.DataFrame:
    summary_path = root / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 dataset_summary.csv: {summary_path}")
    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if "filepath" not in df.columns:
        raise ValueError("dataset_summary.csv 缺少 filepath 列。")
    return df


def resolve_file_path(root: Path, raw_path: str) -> Path:
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
# STA/LTA 候选切片
# =============================================================================

def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    return np.sqrt(np.convolve(x * x, np.ones(win, dtype=np.float64) / win, mode="same") + 1e-12)


def sta_lta_ratio(x: np.ndarray, fs: float, sta_s: float, lta_s: float) -> np.ndarray | None:
    sta = max(1, int(round(sta_s * fs)))
    lta = max(sta + 1, int(round(lta_s * fs)))
    if len(x) < lta + 5:
        return None
    x = x.astype(np.float64)
    x -= np.mean(x)
    env = moving_rms(x, sta)
    sta_v = np.convolve(env, np.ones(sta, dtype=np.float64) / sta, mode="same")
    lta_v = np.convolve(env, np.ones(lta, dtype=np.float64) / lta, mode="same") + 1e-12
    n = min(len(sta_v), len(lta_v))
    return sta_v[:n] / lta_v[:n]


def detect_events_from_ratio(ratio: np.ndarray, fs: float, params: Dict[str, float]) -> List[Tuple[int, int, int, float]]:
    min_dur = max(1, int(round(params["min_dur_s"] * fs)))
    merge_gap = max(0, int(round(params["merge_gap_s"] * fs)))
    thr_on = params["thr_on"]
    thr_off = params["thr_off"]

    events: List[Tuple[int, int, int, float]] = []
    in_event = False
    start = peak = 0
    peak_value = 0.0

    for i, value in enumerate(ratio):
        r = float(value)
        if (not in_event) and r >= thr_on:
            start = peak = i
            peak_value = r
            in_event = True
        elif in_event:
            if r > peak_value:
                peak = i
                peak_value = r
            if r <= thr_off:
                end = i
                if end - start >= min_dur:
                    events.append((start, end, peak, peak_value))
                in_event = False

    if in_event:
        end = len(ratio) - 1
        if end - start >= min_dur:
            events.append((start, end, peak, peak_value))

    if not events:
        return []

    merged = [events[0]]
    for start, end, peak, peak_value in events[1:]:
        last_start, last_end, last_peak, last_peak_value = merged[-1]
        if start <= last_end + merge_gap:
            new_end = max(last_end, end)
            if peak_value > last_peak_value:
                merged[-1] = (last_start, new_end, peak, peak_value)
            else:
                merged[-1] = (last_start, new_end, last_peak, last_peak_value)
        else:
            merged.append((start, end, peak, peak_value))
    return merged


def iter_candidate_events(data: np.ndarray, fs: float, slicing_params: Dict[str, float], channel_stride: int) -> Iterable[Dict]:
    n_time, n_channels = data.shape
    pre = int(round(slicing_params["pre_pad_s"] * fs))
    post = int(round(slicing_params["post_pad_s"] * fs))

    for ch in range(0, n_channels, channel_stride):
        ratio = sta_lta_ratio(data[:, ch], fs, slicing_params["sta_s"], slicing_params["lta_s"])
        if ratio is None:
            continue
        events = detect_events_from_ratio(ratio, fs, slicing_params)
        for start, end, peak, peak_value in events:
            w0 = max(0, start - pre)
            w1 = min(n_time, end + post)
            if w1 <= w0 + 4:
                continue
            yield dict(
                channel=int(ch),
                start=int(start),
                end=int(end),
                w0=int(w0),
                w1=int(w1),
                peak=int(peak),
                peak_ratio=float(peak_value),
            )


# =============================================================================
# 特征、MCD 与规则
# =============================================================================

def band_energy(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    if fmax <= fmin:
        return 0.0
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_event_features(seg: np.ndarray, fs: float) -> Dict[str, float]:
    x = seg.astype(np.float64)
    x -= np.mean(x)
    n = len(x)

    amp = float(np.max(np.abs(x))) if n else 0.0
    rms_v = float(np.sqrt(np.mean(x**2) + 1e-12)) if n else 0.0
    zcr = float(np.mean(np.sign(x[1:]) != np.sign(x[:-1]))) if n > 2 else 0.0
    dur_s = float(n / fs)

    nperseg = min(2048, n)
    if nperseg < 16:
        return dict(
            amp=amp, rms=rms_v, zcr=zcr, dur_s=dur_s,
            E_low=0.0, E_mid=0.0, E_high=0.0, E_vhigh=0.0,
            mid_low=0.0, high_low=0.0, vhigh_low=0.0,
            centroid=0.0, n_samples=int(n),
        )

    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    bands = adaptive_bands(fs)
    e_low = band_energy(psd, freqs, *bands["low"])
    e_mid = band_energy(psd, freqs, *bands["mid"])
    e_high = band_energy(psd, freqs, *bands["high"])
    e_vhigh = band_energy(psd, freqs, *bands["vhigh"])

    psd_sum = float(np.sum(psd) + 1e-30)
    centroid = float(np.sum(freqs * psd) / psd_sum)

    return dict(
        amp=amp,
        rms=rms_v,
        zcr=zcr,
        dur_s=dur_s,
        E_low=e_low,
        E_mid=e_mid,
        E_high=e_high,
        E_vhigh=e_vhigh,
        mid_low=float(e_mid / (e_low + 1e-12)),
        high_low=float(e_high / (e_low + 1e-12)),
        vhigh_low=float(e_vhigh / (e_low + 1e-12)),
        centroid=centroid,
        n_samples=int(n),
    )


def fit_mcd_model(feature_df: pd.DataFrame) -> Tuple[Dict, np.ndarray]:
    raw = feature_df[MCD_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    if len(raw) < max(50, len(MCD_FEATURE_COLS) * 10):
        raise RuntimeError(f"MCD 拟合候选事件太少: {len(raw)}")

    x_raw = raw.values.astype(np.float64)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    x_trans = pt.fit_transform(x_raw)

    mcd = MinCovDet(random_state=RANDOM_SEED, support_fraction=None)
    mcd.fit(x_trans)
    d2 = mcd.mahalanobis(x_trans)

    n_features = len(MCD_FEATURE_COLS)
    d2_thr_dict = {float(a): float(stats.chi2.ppf(1 - a, df=n_features)) for a in MAHAL_ALPHA_LIST}
    model = dict(
        power_transformer=pt,
        mcd_estimator=mcd,
        location=mcd.location_.copy(),
        covariance=mcd.covariance_.copy(),
        precision=mcd.get_precision().copy(),
        feature_cols=list(MCD_FEATURE_COLS),
        n_features=n_features,
        alpha_primary=float(MAHAL_ALPHA),
        d2_thr_primary=d2_thr_dict[float(MAHAL_ALPHA)],
        d2_thr_dict=d2_thr_dict,
    )
    return model, d2


def mahalanobis_score(features: Dict[str, float], model: Dict, alpha: float = MAHAL_ALPHA) -> Tuple[float, bool]:
    x = np.array([features[c] for c in model["feature_cols"]], dtype=np.float64).reshape(1, -1)
    if not np.all(np.isfinite(x)):
        return float("inf"), True
    x_trans = model["power_transformer"].transform(x)
    d2 = float(model["mcd_estimator"].mahalanobis(x_trans)[0])
    thr = float(model["d2_thr_dict"].get(float(alpha), model["d2_thr_primary"]))
    return d2, bool(d2 > thr)


# =============================================================================
# A 模块：semblance / tau-p coherence
# =============================================================================

def compute_plain_semblance(traces: np.ndarray) -> float:
    traces = traces.astype(np.float64)
    num = np.sum(np.sum(traces, axis=1) ** 2)
    den = traces.shape[1] * np.sum(traces**2) + 1e-30
    val = float(num / den)
    return max(0.0, min(1.0, val))


def shift_trace_linear(x: np.ndarray, shift_samples: float) -> np.ndarray:
    # 输出 y(t)=x(t+shift)，用于按预测到时对齐。
    idx = np.arange(len(x), dtype=np.float64) + shift_samples
    return np.interp(idx, np.arange(len(x), dtype=np.float64), x, left=0.0, right=0.0)


def tau_p_scan_semblance(traces: np.ndarray, fs: float, dx_m: float, v_min: float, v_max: float, n_v_scan: int) -> Tuple[float, float, float]:
    n_time, n_traces = traces.shape
    center = n_traces // 2
    offsets = (np.arange(n_traces) - center) * dx_m
    velocities = np.linspace(v_min, v_max, n_v_scan)
    polarities = [-1.0, 1.0]

    best_sem = -np.inf
    best_v = np.nan
    best_p = np.nan

    for sign in polarities:
        for velocity in velocities:
            p = sign / velocity
            aligned = np.empty_like(traces, dtype=np.float64)
            for i in range(n_traces):
                delay_s = offsets[i] * p
                aligned[:, i] = shift_trace_linear(traces[:, i], delay_s * fs)
            sem = compute_plain_semblance(aligned)
            if sem > best_sem:
                best_sem = sem
                best_v = velocity
                best_p = p

    return float(best_sem), float(best_v), float(best_p)


def estimate_v_app_xcorr(traces: np.ndarray, fs: float, dx_m: float) -> float:
    n_time, n_traces = traces.shape
    center = n_traces // 2
    ref = traces[:, center].astype(np.float64)
    ref -= np.mean(ref)
    delays = []
    dists = []

    for i in range(n_traces):
        if i == center:
            continue
        x = traces[:, i].astype(np.float64)
        x -= np.mean(x)
        corr = signal.correlate(x, ref, mode="full")
        lag_idx = int(np.argmax(np.abs(corr)))
        lag = lag_idx - (n_time - 1)
        if 0 < lag_idx < len(corr) - 1:
            y0, y1, y2 = corr[lag_idx - 1], corr[lag_idx], corr[lag_idx + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                lag += 0.5 * (y0 - y2) / denom
        tau = lag / fs
        if abs(tau) > 0.5 / fs:
            delays.append(tau)
            dists.append((i - center) * dx_m)

    if len(delays) < 2:
        return float("inf")
    delays = np.asarray(delays, dtype=np.float64)
    dists = np.asarray(dists, dtype=np.float64)
    slopes = dists / delays
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return float("inf")
    return float(abs(np.median(slopes)))


def get_neighbor_traces(data: np.ndarray, ch: int, w0: int, w1: int, k: int) -> Tuple[np.ndarray, int]:
    n_channels = data.shape[1]
    c0 = max(0, ch - k)
    c1 = min(n_channels, ch + k + 1)
    center_local = ch - c0
    return data[w0:w1, c0:c1], center_local


def coherence_rescue(data: np.ndarray, event: Dict, fs: float, dx_m: float, method: str) -> Tuple[bool, float, float]:
    k = int(RUNTIME_PARAMS["COHERENCE_K"])
    traces, center_local = get_neighbor_traces(data, event["channel"], event["w0"], event["w1"], k)
    if traces.shape[1] < 3 or traces.shape[0] < 8:
        return False, 0.0, np.nan

    if method == "mcd_plain_semblance":
        sem = compute_plain_semblance(traces)
        v_app = estimate_v_app_xcorr(traces, fs, dx_m)
    else:
        sem, v_app, _ = tau_p_scan_semblance(
            traces,
            fs,
            dx_m,
            v_min=float(RUNTIME_PARAMS["V_APP_MIN"]),
            v_max=float(RUNTIME_PARAMS["V_APP_MAX"]),
            n_v_scan=int(RUNTIME_PARAMS["N_V_SCAN"]),
        )

    rescued = (
        sem >= float(RUNTIME_PARAMS["SEMBLANCE_THR"])
        and float(RUNTIME_PARAMS["V_APP_MIN"]) <= v_app <= float(RUNTIME_PARAMS["V_APP_MAX"])
    )
    return bool(rescued), float(sem), float(v_app)


# =============================================================================
# 填补与指标
# =============================================================================

def fill_interp_channel(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = y.copy()
    idx = np.arange(len(y))
    good = ~mask
    if np.sum(good) < 2:
        out[mask] = 0.0
        return out
    out[mask] = np.interp(idx[mask], idx[good], y[good])
    return out


def fill_lowrank_with_neighbors(data: np.ndarray, mask: np.ndarray, fs: float) -> np.ndarray:
    cleaned = data.copy()
    n_time, n_channels = data.shape
    k = int(RUNTIME_PARAMS["LOWRANK_NEIGHBOR_K"])
    rank = int(RUNTIME_PARAMS["LOWRANK_RANK"])
    n_iter = int(RUNTIME_PARAMS["LOWRANK_ITER"])
    pad = int(round(float(RUNTIME_PARAMS["LOWRANK_WIN_PAD_S"]) * fs))

    for ch in range(n_channels):
        idx = np.where(mask[:, ch])[0]
        if idx.size == 0:
            continue

        # 将相邻 mask 点合并为连续片段。
        breaks = np.where(np.diff(idx) > 1)[0] + 1
        segments = np.split(idx, breaks)

        for seg_idx in segments:
            s = int(seg_idx[0])
            e = int(seg_idx[-1]) + 1
            w0 = max(0, s - pad)
            w1 = min(n_time, e + pad)
            c0 = max(0, ch - k)
            c1 = min(n_channels, ch + k + 1)
            local_ch = ch - c0

            block = cleaned[w0:w1, c0:c1].astype(np.float64)
            local_mask = mask[w0:w1, c0:c1]
            if block.shape[0] < 4 or block.shape[1] < 3:
                cleaned[s:e, ch] = fill_interp_channel(cleaned[:, ch], mask[:, ch])[s:e]
                continue

            filled = block.copy()
            for cc in range(block.shape[1]):
                filled[:, cc] = fill_interp_channel(filled[:, cc], local_mask[:, cc])

            for _ in range(n_iter):
                mean_t = np.mean(filled, axis=0, keepdims=True)
                centered = filled - mean_t
                u, sv, vt = np.linalg.svd(centered, full_matrices=False)
                r = max(1, min(rank, len(sv)))
                recon = (u[:, :r] * sv[:r]) @ vt[:r, :] + mean_t
                filled[local_mask] = recon[local_mask]

            cleaned[s:e, ch] = filled[(s - w0):(e - w0), local_ch]

    return cleaned


def clean_by_mask(data: np.ndarray, final_mask: np.ndarray, fs: float, fill_mode: str) -> np.ndarray:
    if fill_mode == "lowrank":
        return fill_lowrank_with_neighbors(data, final_mask, fs).astype(np.float32)

    cleaned = data.copy()
    for ch in range(data.shape[1]):
        if np.any(final_mask[:, ch]):
            cleaned[:, ch] = fill_interp_channel(cleaned[:, ch], final_mask[:, ch])
    return cleaned.astype(np.float32)


def mask_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def waveform_metrics(data_noisy: np.ndarray, cleaned: np.ndarray, clean_true: np.ndarray) -> Dict[str, float]:
    input_noise = data_noisy - clean_true
    output_noise = cleaned - clean_true
    input_snr = compute_snr_db(clean_true, input_noise)
    output_snr = compute_snr_db(clean_true, output_noise)
    nrmse = rms(output_noise) / (rms(clean_true) + 1e-30)
    return dict(
        input_snr_db=input_snr,
        output_snr_db=output_snr,
        snr_gain_db=output_snr - input_snr,
        nrmse=nrmse,
        corr=safe_corrcoef(cleaned, clean_true),
        amplitude_bias=float((np.max(np.abs(cleaned)) - np.max(np.abs(clean_true))) / (np.max(np.abs(clean_true)) + 1e-30)),
    )


# =============================================================================
# 训练 MCD
# =============================================================================

def collect_feature_library(rows: pd.DataFrame, slicing_params: Dict[str, float]) -> pd.DataFrame:
    records: List[Dict] = []
    for i, row in rows.iterrows():
        path = resolve_file_path(DATASET_ROOT, row["filepath"])
        if not path.exists():
            print(f"[警告] 训练文件不存在，跳过: {path}")
            continue
        with np.load(path, allow_pickle=False) as npz:
            data = npz["data_noisy"]
            fs = float(np.asarray(npz["fs"]))

        for ev in iter_candidate_events(data, fs, slicing_params, CHANNEL_STRIDE):
            seg = data[ev["w0"]:ev["w1"], ev["channel"]]
            feats = extract_event_features(seg, fs)
            feats.update(
                category=row.get("category", ""),
                artifact_type=row.get("artifact_type", row.get("noise_type", "")),
                snr_target_db=row.get("snr_target_db", np.nan),
                filepath=str(path),
                channel=ev["channel"],
                w0=ev["w0"],
                w1=ev["w1"],
                peak_ratio=ev["peak_ratio"],
                fs=fs,
            )
            records.append(feats)

        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            print(f"[MCD] 已收集特征文件 {i + 1}/{len(rows)}，事件数={len(records)}")

    return pd.DataFrame(records)


def save_mcd_diagnostics(feature_df: pd.DataFrame, d2: np.ndarray, model: Dict) -> None:
    feature_csv = OUT_DIR / "synthetic_feature_library.csv"
    feature_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "synthetic_mcd_model.pkl", "wb") as f:
        pickle.dump(model, f)

    diag = {
        "n_events": int(len(feature_df)),
        "feature_cols": MCD_FEATURE_COLS,
        "alpha_primary": MAHAL_ALPHA,
        "d2_thr_dict": model["d2_thr_dict"],
        "d2_mean": float(np.mean(d2)),
        "d2_std": float(np.std(d2)),
    }
    (OUT_DIR / "synthetic_mcd_report.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# 单样本处理与汇总
# =============================================================================

def process_sample(path: Path, row_meta: Dict, mcd_model: Dict, slicing_params: Dict[str, float]) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as npz:
        data_noisy = npz["data_noisy"].astype(np.float32)
        data_clean = npz["data_clean"].astype(np.float32)
        artifact_mask = npz["artifact_mask"].astype(bool)
        signal_mask = npz["signal_mask"].astype(bool)
        fs = float(np.asarray(npz["fs"]))
        dx_m = float(np.asarray(npz["dx_m"]))

    n_time, n_channels = data_noisy.shape
    event_logs = []
    method_masks = {method: np.zeros((n_time, n_channels), dtype=bool) for method in METHODS}

    for ev in iter_candidate_events(data_noisy, fs, slicing_params, CHANNEL_STRIDE):
        ch = ev["channel"]
        seg = data_noisy[ev["w0"]:ev["w1"], ch]
        feats = extract_event_features(seg, fs)
        d2, is_noise = mahalanobis_score(feats, mcd_model, alpha=MAHAL_ALPHA)
        if not is_noise:
            continue

        method_masks["mcd_only"][ev["w0"]:ev["w1"], ch] = True

        rescued_plain, sem_plain, v_plain = coherence_rescue(data_noisy, ev, fs, dx_m, "mcd_plain_semblance")
        if not rescued_plain:
            method_masks["mcd_plain_semblance"][ev["w0"]:ev["w1"], ch] = True

        rescued_tau, sem_tau, v_tau = coherence_rescue(data_noisy, ev, fs, dx_m, "mcd_tau_p")
        if not rescued_tau:
            method_masks["mcd_tau_p"][ev["w0"]:ev["w1"], ch] = True
            method_masks["mcd_tau_p_lowrank"][ev["w0"]:ev["w1"], ch] = True

        event_logs.append({
            "filepath": str(path),
            "channel": int(ch),
            "w0": int(ev["w0"]),
            "w1": int(ev["w1"]),
            "d2_mahalanobis": float(d2),
            "rescued_plain": bool(rescued_plain),
            "semblance_plain": float(sem_plain),
            "v_app_plain": float(v_plain),
            "rescued_tau_p": bool(rescued_tau),
            "semblance_tau_p": float(sem_tau),
            "v_app_tau_p": float(v_tau),
        })

    rows = []
    outputs = {}
    for method, pred_mask in method_masks.items():
        fill_mode = "lowrank" if method == "mcd_tau_p_lowrank" else "interp"
        cleaned = clean_by_mask(data_noisy, pred_mask, fs, fill_mode=fill_mode)
        mm = mask_metrics(pred_mask, artifact_mask)
        wm = waveform_metrics(data_noisy, cleaned, data_clean)
        signal_total = int(np.sum(signal_mask))
        signal_removed = int(np.sum(pred_mask & signal_mask))
        false_removal_rate = signal_removed / signal_total if signal_total else np.nan

        rows.append({
            "method": method,
            "filepath": str(path),
            "category": row_meta.get("category", ""),
            "artifact_type": row_meta.get("artifact_type", row_meta.get("noise_type", "")),
            "snr_target_db": row_meta.get("snr_target_db", np.nan),
            "v_app_ms": row_meta.get("v_app_ms", np.nan),
            "fs": fs,
            "dx_m": dx_m,
            "n_time": n_time,
            "n_channels": n_channels,
            "candidate_events_mcd": len(event_logs),
            "pred_mask_ratio": float(np.mean(pred_mask)),
            "artifact_mask_ratio": float(np.mean(artifact_mask)),
            "signal_mask_ratio": float(np.mean(signal_mask)),
            "signal_false_removal_rate": false_removal_rate,
            "signal_preservation_rate": 1.0 - false_removal_rate if np.isfinite(false_removal_rate) else np.nan,
            **mm,
            **wm,
        })
        outputs[method] = cleaned
        outputs[f"{method}_mask"] = pred_mask

    return rows, {"event_logs": event_logs, **outputs}


def summarize_results(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "category", "artifact_type", "snr_target_db"]
    value_cols = [
        "precision", "recall", "f1", "signal_false_removal_rate",
        "signal_preservation_rate", "snr_gain_db", "nrmse", "corr",
        "amplitude_bias", "pred_mask_ratio",
    ]
    grouped = (
        sample_metrics
        .groupby(group_cols, dropna=False)[value_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    grouped.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in grouped.columns]
    return grouped


def save_optional_outputs(path: Path, outputs: Dict[str, np.ndarray], save_dir: Path, save_idx: int) -> None:
    ensure_dir(save_dir)
    payload = {}
    for method in METHODS:
        payload[f"{method}_cleaned"] = outputs[method].astype(np.float32)
        payload[f"{method}_mask"] = outputs[f"{method}_mask"].astype(bool)
    payload["source_file"] = np.array(str(path))
    np.savez_compressed(save_dir / f"mcsand_output_{save_idx:04d}.npz", **payload)


def main() -> None:
    ensure_dir(OUT_DIR)
    slicing_params = load_slicing_params()

    print("=" * 80)
    print("合成 DAS 数据完整 MCS-AND benchmark")
    print("=" * 80)
    print(f"数据集目录: {DATASET_ROOT}")
    print(f"输出目录  : {OUT_DIR}")
    print(f"STA/LTA 参数: {slicing_params}")
    print(f"CHANNEL_STRIDE = {CHANNEL_STRIDE}")

    summary = load_summary(DATASET_ROOT)
    summary["resolved_path"] = summary["filepath"].apply(lambda p: str(resolve_file_path(DATASET_ROOT, p)))

    train_rows = choose_rows(summary, MAX_TRAIN_FILES, RANDOM_SEED)
    print(f"\n[1/3] 收集候选事件特征并拟合 MCD，训练样本数: {len(train_rows)}")
    feature_df = collect_feature_library(train_rows, slicing_params)
    if feature_df.empty:
        raise RuntimeError("没有收集到任何候选事件，请检查 STA/LTA 参数。")
    mcd_model, d2 = fit_mcd_model(feature_df)
    save_mcd_diagnostics(feature_df, d2, mcd_model)
    print(f"[OK] MCD 拟合完成，事件数={len(feature_df)}，主阈值={mcd_model['d2_thr_primary']:.3f}")

    process_rows = choose_rows(summary, MAX_PROCESS_FILES, RANDOM_SEED + 1)
    print(f"\n[2/3] 运行完整 MCS-AND benchmark，处理样本数: {len(process_rows)}")
    all_metric_rows: List[Dict] = []
    all_event_logs: List[Dict] = []
    saved_count = 0
    save_dir = OUT_DIR / "sample_outputs"

    for i, row in process_rows.iterrows():
        path = Path(row["resolved_path"])
        if not path.exists():
            print(f"[警告] 文件不存在，跳过: {path}")
            continue
        metric_rows, outputs = process_sample(path, row.to_dict(), mcd_model, slicing_params)
        all_metric_rows.extend(metric_rows)
        all_event_logs.extend(outputs["event_logs"])

        if SAVE_OUTPUT_NPZ and saved_count < MAX_SAVE_SAMPLES:
            save_optional_outputs(path, outputs, save_dir, saved_count)
            saved_count += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(process_rows):
            print(f"  已处理 {i + 1}/{len(process_rows)} 个样本，指标行={len(all_metric_rows)}")

    sample_metrics = pd.DataFrame(all_metric_rows)
    event_log_df = pd.DataFrame(all_event_logs)
    if sample_metrics.empty:
        raise RuntimeError("没有生成任何 benchmark 指标，请检查输入数据和参数。")

    print("\n[3/3] 保存指标与汇总结果")
    sample_metrics_csv = OUT_DIR / "mcsand_sample_metrics.csv"
    summary_metrics_csv = OUT_DIR / "mcsand_summary_metrics.csv"
    event_log_csv = OUT_DIR / "mcsand_event_log.csv"
    report_json = OUT_DIR / "mcsand_benchmark_report.json"

    summary_metrics = summarize_results(sample_metrics)
    sample_metrics.to_csv(sample_metrics_csv, index=False, encoding="utf-8-sig")
    summary_metrics.to_csv(summary_metrics_csv, index=False, encoding="utf-8-sig")
    event_log_df.to_csv(event_log_csv, index=False, encoding="utf-8-sig")

    report = {
        "dataset_root": str(DATASET_ROOT),
        "out_dir": str(OUT_DIR),
        "sta_lta_param_path": str(STA_LTA_PARAM_PATH),
        "slicing_params": slicing_params,
        "runtime_params": RUNTIME_PARAMS,
        "mahal_alpha": MAHAL_ALPHA,
        "max_train_files": MAX_TRAIN_FILES,
        "max_process_files": MAX_PROCESS_FILES,
        "channel_stride": CHANNEL_STRIDE,
        "n_feature_events": int(len(feature_df)),
        "n_processed_samples": int(sample_metrics["filepath"].nunique()),
        "methods": list(METHODS),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[完成] 输出文件：")
    print(f"  样本指标: {sample_metrics_csv}")
    print(f"  汇总指标: {summary_metrics_csv}")
    print(f"  事件日志: {event_log_csv}")
    print(f"  MCD模型 : {OUT_DIR / 'synthetic_mcd_model.pkl'}")
    print(f"  特征库  : {OUT_DIR / 'synthetic_feature_library.csv'}")
    print("\n各方法总体 F1 / SNR gain 概览：")
    overview = (
        sample_metrics
        .groupby("method", dropna=False)
        .agg(
            f1_mean=("f1", "mean"),
            recall_mean=("recall", "mean"),
            precision_mean=("precision", "mean"),
            signal_false_removal_mean=("signal_false_removal_rate", "mean"),
            snr_gain_mean=("snr_gain_db", "mean"),
            n_samples=("filepath", "nunique"),
        )
        .reset_index()
    )
    print(overview.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
