# -*- coding: utf-8 -*-
"""
FORESEE 真实 DAS 数据 MCS-AND 应用脚本（硬编码路径版，中文详细注释）。

这个脚本用于处理前一步 extract_foresee_real_cases.py 生成的真实数据小片段。

重要定位：
    FORESEE 城市 DAS 中的斜向强能量大多是车辆/交通诱发的相干事件，
    不应该被简单当作噪声剔除。因此真实数据实验的目的不是“尽可能多删”，
    而是验证：

        1. MCD-only 会把部分强交通事件列为候选异常；
        2. τ-p coherence rescue 能把具有空间-时间相干性的车辆事件救回；
        3. 最终 MCS-AND mask 主要保留局部非相干异常；
        4. cleaned record 中主要交通轨迹仍然存在。

输入：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases\*.npz

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results

输出文件：
    real_mcsand_event_log.csv
    real_mcsand_case_summary.csv
    real_mcsand_report.json
    outputs\*_mcsand_real_output.npz

说明：
    真实数据没有 artifact ground truth，因此本脚本不计算 F1。
    输出的 mask 面积、候选事件数、rescue rate、removed RMS 等是辅助诊断指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import PowerTransformer


# =============================================================================
# 1. 硬编码路径
# =============================================================================

CASE_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases")
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results")
OUTPUT_NPZ_DIR = OUT_DIR / "outputs"


# =============================================================================
# 2. 真实数据处理参数
# =============================================================================

RANDOM_SEED = 42

# FORESEE 官方 HDF5 已降采样到 125 Hz；case npz 中也保存了 fs，这里只是备用。
DEFAULT_FS = 125.0
DEFAULT_DX_M = 2.0

# STA/LTA 候选检测参数。
# 真实交通事件持续时间通常比合成短脉冲更长，因此窗口略放宽。
SLICING_PARAMS = dict(
    sta_s=0.08,
    lta_s=0.80,
    thr_on=2.2,
    thr_off=1.25,
    min_dur_s=0.04,
    merge_gap_s=0.16,
    pre_s=0.08,
    post_s=0.12,
)

# 为了真实数据先跑通，建议先每 2 道检测一次。
# 如果想画更细的 mask，可以改成 1。
CHANNEL_STRIDE = 2

# MCD 特征列。
MCD_FEATURE_COLS = ["zcr", "centroid", "vhigh_low", "high_low", "mid_low", "rms", "dur_s"]

# 真实数据没有真值，这里采用“场数据内部稳健建模”的方式：
#   - 对所有真实 case 的候选事件提取特征；
#   - 用 Yeo-Johnson + MCD 建模主体分布；
#   - Mahalanobis distance 过高的事件作为 MCD-only candidate。
#
# threshold 使用 chi-square 与经验分位数的较大者，避免场数据中过度标记。
MAHAL_ALPHA = 0.05
EMPIRICAL_D2_QUANTILE = 0.90

# τ-p coherence rescue 参数。
# 注意：FORESEE 图中车辆轨迹对应的表观速度可能只有十几到几十 m/s，
# 所以 V_APP_MIN 不能沿用合成数据中的 100 m/s。
RUNTIME_PARAMS = dict(
    COHERENCE_K=5,          # 中心通道左右各取 5 道
    SEMBLANCE_THR=0.22,     # 达到该相干性阈值即 rescue
    V_APP_MIN=5.0,          # 适应车辆沿光纤移动的低表观速度
    V_APP_MAX=2000.0,
    N_V_SCAN=60,
    LOWRANK_RANK=2,
    LOWRANK_WIN_PAD_S=0.30,
    LOWRANK_NEIGHBOR_K=5,
    LOWRANK_ITER=3,
)

METHODS = (
    "mcd_only",
    "mcd_tau_p",
    "mcd_tau_p_lowrank",
)


# =============================================================================
# 3. 基础函数
# =============================================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    """均方根幅值。"""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """滑动 RMS，用于 STA/LTA 包络。"""
    x2 = np.asarray(x, dtype=np.float64) ** 2
    ker = np.ones(max(1, int(win)), dtype=np.float64) / max(1, int(win))
    return np.sqrt(np.convolve(x2, ker, mode="same") + 1e-30)


def sta_lta_ratio(x: np.ndarray, fs: float, sta_s: float, lta_s: float) -> np.ndarray | None:
    """计算单通道 STA/LTA 比值。"""
    sta = max(1, int(round(sta_s * fs)))
    lta = max(sta + 1, int(round(lta_s * fs)))
    if len(x) < lta + 5:
        return None
    y = x.astype(np.float64)
    y -= np.mean(y)
    env = moving_rms(y, sta)
    sta_v = np.convolve(env, np.ones(sta, dtype=np.float64) / sta, mode="same")
    lta_v = np.convolve(env, np.ones(lta, dtype=np.float64) / lta, mode="same") + 1e-12
    n = min(len(sta_v), len(lta_v))
    return sta_v[:n] / lta_v[:n]


def detect_events_from_ratio(ratio: np.ndarray, fs: float, params: Dict[str, float]) -> List[Tuple[int, int, int, float]]:
    """从 STA/LTA 比值中提取候选事件。"""
    thr_on = float(params["thr_on"])
    thr_off = float(params["thr_off"])
    min_dur = max(1, int(round(params["min_dur_s"] * fs)))
    merge_gap = max(0, int(round(params["merge_gap_s"] * fs)))

    events: List[Tuple[int, int, int, float]] = []
    in_event = False
    start = peak = 0
    peak_value = 0.0

    for i, r in enumerate(ratio):
        if (not in_event) and r >= thr_on:
            start = peak = i
            peak_value = float(r)
            in_event = True
        elif in_event:
            if r > peak_value:
                peak = i
                peak_value = float(r)
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


def iter_candidate_events(data_tc: np.ndarray, fs: float, params: Dict[str, float]) -> Iterable[Dict]:
    """
    遍历候选事件。

    data_tc 的形状为 [time, channel]。
    """
    n_time, n_channels = data_tc.shape
    pre = int(round(params["pre_s"] * fs))
    post = int(round(params["post_s"] * fs))

    for ch in range(0, n_channels, CHANNEL_STRIDE):
        ratio = sta_lta_ratio(data_tc[:, ch], fs, params["sta_s"], params["lta_s"])
        if ratio is None:
            continue
        events = detect_events_from_ratio(ratio, fs, params)
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
                peak_value=float(peak_value),
                dur_s=float((end - start) / fs),
            )


def adaptive_bands(fs: float) -> Dict[str, Tuple[float, float]]:
    """根据采样率自适应设置频带。"""
    nyq = fs / 2.0
    return {
        "low": (0.5, min(5.0, nyq * 0.25)),
        "mid": (min(5.0, nyq * 0.25), min(20.0, nyq * 0.60)),
        "high": (min(20.0, nyq * 0.60), min(45.0, nyq * 0.90)),
        "vhigh": (min(45.0, nyq * 0.90), nyq * 0.98),
    }


def band_energy(psd: np.ndarray, freqs: np.ndarray, f0: float, f1: float) -> float:
    """计算指定频带能量。"""
    mask = (freqs >= f0) & (freqs < f1)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def extract_event_features(seg: np.ndarray, fs: float) -> Dict[str, float]:
    """
    提取候选事件特征。

    seg 是单通道候选片段。
    """
    x = np.asarray(seg, dtype=np.float64)
    x -= np.mean(x)
    x_std = np.std(x) + 1e-12
    x_norm = x / x_std

    zcr = float(np.mean(np.abs(np.diff(np.signbit(x_norm)))))
    nperseg = min(256, max(16, len(x_norm)))
    freqs, psd = signal.welch(x_norm, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="constant")
    psd_sum = float(np.sum(psd) + 1e-30)
    centroid = float(np.sum(freqs * psd) / psd_sum)

    bands = adaptive_bands(fs)
    e_low = band_energy(psd, freqs, *bands["low"]) + 1e-30
    e_mid = band_energy(psd, freqs, *bands["mid"]) + 1e-30
    e_high = band_energy(psd, freqs, *bands["high"]) + 1e-30
    e_vhigh = band_energy(psd, freqs, *bands["vhigh"]) + 1e-30

    return dict(
        zcr=zcr,
        centroid=centroid,
        vhigh_low=float(np.log10(e_vhigh / e_low)),
        high_low=float(np.log10(e_high / e_low)),
        mid_low=float(np.log10(e_mid / e_low)),
        rms=rms(x),
        dur_s=float(len(x) / fs),
    )


def load_case(path: Path) -> Dict:
    """读取真实 case npz，并统一返回 [time, channel] 数据。"""
    with np.load(path, allow_pickle=False) as npz:
        data_ct = npz["data_proc"].astype(np.float32)
        fs = float(npz["fs"]) if "fs" in npz.files else DEFAULT_FS
        dx_m = float(npz["dx_m"]) if "dx_m" in npz.files else DEFAULT_DX_M
        case_id = str(npz["case_id"]) if "case_id" in npz.files else path.stem
        meta_json = str(npz["meta_json"]) if "meta_json" in npz.files else "{}"
        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = {}

    # 裁剪脚本保存的是 [channel, time]，MCS-AND 内部使用 [time, channel]。
    data_tc = data_ct.T.copy()
    return dict(path=path, case_id=case_id, data_ct=data_ct, data_tc=data_tc, fs=fs, dx_m=dx_m, meta=meta)


def collect_case_files() -> List[Path]:
    """收集真实 case npz 文件。"""
    files = sorted(CASE_DIR.glob("real_case_*.npz"))
    if not files:
        raise FileNotFoundError(f"没有找到真实 case npz: {CASE_DIR}")
    return files


# =============================================================================
# 4. MCD 模型
# =============================================================================

def collect_feature_library(case_files: List[Path]) -> pd.DataFrame:
    """从所有真实 case 中收集候选事件特征。"""
    rows: List[Dict] = []
    for path in case_files:
        case = load_case(path)
        data_tc = case["data_tc"]
        fs = case["fs"]
        for ev in iter_candidate_events(data_tc, fs, SLICING_PARAMS):
            ch = ev["channel"]
            seg = data_tc[ev["w0"]:ev["w1"], ch]
            feats = extract_event_features(seg, fs)
            rows.append({
                "case_id": case["case_id"],
                "source_file": str(path),
                **ev,
                **feats,
            })

    return pd.DataFrame(rows)


def fit_real_mcd_model(feature_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """拟合真实数据内部 MCD 模型。"""
    raw = feature_df[MCD_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    if len(raw) < max(30, len(MCD_FEATURE_COLS) * 8):
        raise RuntimeError(f"候选事件过少，无法拟合 MCD。当前候选数={len(raw)}")

    x_raw = raw.values.astype(np.float64)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    x_trans = pt.fit_transform(x_raw)

    mcd = MinCovDet(random_state=RANDOM_SEED, support_fraction=None)
    mcd.fit(x_trans)
    d2 = mcd.mahalanobis(x_trans)

    chi2_thr = float(stats.chi2.ppf(1.0 - MAHAL_ALPHA, df=len(MCD_FEATURE_COLS)))
    empirical_thr = float(np.quantile(d2, EMPIRICAL_D2_QUANTILE))
    d2_thr = max(chi2_thr, empirical_thr)

    model = dict(
        power_transformer=pt,
        mcd_estimator=mcd,
        feature_cols=MCD_FEATURE_COLS,
        chi2_thr=chi2_thr,
        empirical_thr=empirical_thr,
        d2_thr=d2_thr,
    )

    diag = raw.copy()
    diag["d2_mahalanobis"] = d2
    diag["is_mcd_candidate"] = d2 > d2_thr
    return model, diag


def mahalanobis_score(features: Dict[str, float], model: Dict) -> Tuple[float, bool]:
    """计算 Mahalanobis distance 并判断是否为 MCD 候选异常。"""
    x = np.array([features[c] for c in model["feature_cols"]], dtype=np.float64).reshape(1, -1)
    if not np.all(np.isfinite(x)):
        return np.nan, False
    x_trans = model["power_transformer"].transform(x)
    d2 = float(model["mcd_estimator"].mahalanobis(x_trans)[0])
    return d2, bool(d2 > model["d2_thr"])


# =============================================================================
# 5. τ-p coherence rescue 与修补
# =============================================================================

def get_neighbor_traces(data_tc: np.ndarray, ch: int, w0: int, w1: int, k: int) -> Tuple[np.ndarray, int, int]:
    """取中心通道附近的邻道数据。返回 shape=[time, traces]。"""
    n_time, n_channels = data_tc.shape
    c0 = max(0, ch - k)
    c1 = min(n_channels, ch + k + 1)
    traces = data_tc[w0:w1, c0:c1].astype(np.float64)
    center_local = ch - c0
    return traces, center_local, c0


def tau_p_scan_semblance(
    traces: np.ndarray,
    fs: float,
    dx_m: float,
    v_min: float,
    v_max: float,
    n_v_scan: int,
) -> Tuple[float, float, float]:
    """
    简化 τ-p 扫描，返回最大 semblance、最佳表观速度和最佳符号。

    traces: [time, channel]
    """
    n_time, n_traces = traces.shape
    if n_time < 4 or n_traces < 3:
        return 0.0, np.nan, 0.0

    traces = traces - np.mean(traces, axis=0, keepdims=True)
    center = n_traces // 2
    t = np.arange(n_time, dtype=np.float64)
    velocities = np.geomspace(v_min, v_max, int(n_v_scan))
    best_sem = 0.0
    best_v = np.nan
    best_sign = 0.0

    for sign in (-1.0, 1.0):
        for v in velocities:
            aligned = np.zeros_like(traces)
            valid_count = 0
            for j in range(n_traces):
                dist = (j - center) * dx_m
                delay_samples = sign * dist / v * fs
                # 将相邻道按候选表观速度对齐到中心道。
                aligned[:, j] = np.interp(t, t + delay_samples, traces[:, j], left=0.0, right=0.0)
                valid_count += 1
            if valid_count < 3:
                continue
            num = np.sum(np.sum(aligned, axis=1) ** 2)
            den = n_traces * np.sum(aligned ** 2) + 1e-30
            sem = float(num / den)
            if sem > best_sem:
                best_sem = sem
                best_v = float(v)
                best_sign = float(sign)

    return best_sem, best_v, best_sign


def coherence_rescue(data_tc: np.ndarray, event: Dict, fs: float, dx_m: float) -> Tuple[bool, float, float, float]:
    """
    判断 MCD 候选事件是否应被 τ-p 相干性救回。

    返回：
        rescued, semblance, v_app, direction_sign
    """
    k = int(RUNTIME_PARAMS["COHERENCE_K"])
    traces, _, _ = get_neighbor_traces(data_tc, event["channel"], event["w0"], event["w1"], k)
    sem, v_app, sign = tau_p_scan_semblance(
        traces,
        fs=fs,
        dx_m=dx_m,
        v_min=float(RUNTIME_PARAMS["V_APP_MIN"]),
        v_max=float(RUNTIME_PARAMS["V_APP_MAX"]),
        n_v_scan=int(RUNTIME_PARAMS["N_V_SCAN"]),
    )
    rescued = bool(sem >= float(RUNTIME_PARAMS["SEMBLANCE_THR"]))
    return rescued, sem, v_app, sign


def interpolate_masked_samples(data_tc: np.ndarray, mask_tc: np.ndarray) -> np.ndarray:
    """沿时间方向对 mask 区域做线性插值。"""
    cleaned = data_tc.copy()
    n_time, n_channels = data_tc.shape
    x = np.arange(n_time)
    for ch in range(n_channels):
        m = mask_tc[:, ch]
        if not np.any(m):
            continue
        good = ~m
        if np.sum(good) < 2:
            cleaned[m, ch] = 0.0
            continue
        cleaned[m, ch] = np.interp(x[m], x[good], data_tc[good, ch])
    return cleaned


def fill_lowrank_with_neighbors(data_tc: np.ndarray, mask_tc: np.ndarray, fs: float) -> np.ndarray:
    """
    局部低秩修补。

    对真实数据只作为 cleaned display 使用，不改变 final mask。
    """
    cleaned = data_tc.copy()
    n_time, n_channels = data_tc.shape
    pad = int(round(float(RUNTIME_PARAMS["LOWRANK_WIN_PAD_S"]) * fs))
    k = int(RUNTIME_PARAMS["LOWRANK_NEIGHBOR_K"])
    rank = int(RUNTIME_PARAMS["LOWRANK_RANK"])
    n_iter = int(RUNTIME_PARAMS["LOWRANK_ITER"])

    # 找到每个通道上的连续 mask 段。
    for ch in range(n_channels):
        m = mask_tc[:, ch]
        if not np.any(m):
            continue
        idx = np.where(m)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for g in groups:
            if g.size == 0:
                continue
            t0 = max(0, int(g[0]) - pad)
            t1 = min(n_time, int(g[-1]) + pad + 1)
            c0 = max(0, ch - k)
            c1 = min(n_channels, ch + k + 1)

            patch = cleaned[t0:t1, c0:c1].copy()
            patch_mask = mask_tc[t0:t1, c0:c1]
            if patch.size == 0 or not np.any(patch_mask):
                continue

            # 初始值先用时间插值。
            init = interpolate_masked_samples(patch, patch_mask)
            filled = init
            for _ in range(n_iter):
                u, s, vt = np.linalg.svd(filled, full_matrices=False)
                rr = min(rank, len(s))
                low = (u[:, :rr] * s[:rr]) @ vt[:rr, :]
                filled[patch_mask] = low[patch_mask]

            cleaned[t0:t1, c0:c1][patch_mask] = filled[patch_mask]

    return cleaned


# =============================================================================
# 6. 单个 case 处理
# =============================================================================

def process_case(case_path: Path, model: Dict) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict]:
    """处理一个真实 case。"""
    case = load_case(case_path)
    data_tc = case["data_tc"]
    fs = case["fs"]
    dx_m = case["dx_m"]
    n_time, n_channels = data_tc.shape

    masks = {method: np.zeros((n_time, n_channels), dtype=bool) for method in METHODS}
    candidate_mask = np.zeros((n_time, n_channels), dtype=bool)
    rescued_mask = np.zeros((n_time, n_channels), dtype=bool)
    event_logs: List[Dict] = []

    for ev in iter_candidate_events(data_tc, fs, SLICING_PARAMS):
        ch = ev["channel"]
        seg = data_tc[ev["w0"]:ev["w1"], ch]
        feats = extract_event_features(seg, fs)
        d2, is_candidate = mahalanobis_score(feats, model)
        if not is_candidate:
            continue

        candidate_mask[ev["w0"]:ev["w1"], ch] = True
        masks["mcd_only"][ev["w0"]:ev["w1"], ch] = True

        rescued, sem, v_app, sign = coherence_rescue(data_tc, ev, fs, dx_m)
        if rescued:
            rescued_mask[ev["w0"]:ev["w1"], ch] = True
        else:
            masks["mcd_tau_p"][ev["w0"]:ev["w1"], ch] = True
            masks["mcd_tau_p_lowrank"][ev["w0"]:ev["w1"], ch] = True

        event_logs.append({
            "case_id": case["case_id"],
            "source_file": str(case_path),
            "channel": int(ch),
            "global_channel": int(case["meta"].get("channel_start", 0) + ch),
            "start_s": float(ev["w0"] / fs),
            "end_s": float(ev["w1"] / fs),
            "dur_s": float((ev["w1"] - ev["w0"]) / fs),
            "peak_value": float(ev["peak_value"]),
            "d2_mahalanobis": float(d2),
            "rescued_tau_p": bool(rescued),
            "semblance_tau_p": float(sem),
            "v_app_tau_p": float(v_app) if np.isfinite(v_app) else np.nan,
            "direction_sign": float(sign),
            **{k: float(v) for k, v in feats.items()},
        })

    cleaned_interp = interpolate_masked_samples(data_tc, masks["mcd_tau_p"])
    cleaned_lowrank = fill_lowrank_with_neighbors(data_tc, masks["mcd_tau_p_lowrank"], fs)

    outputs = {
        "data_raw_ct": case["data_ct"].astype(np.float32),
        "data_proc_ct": data_tc.T.astype(np.float32),
        "candidate_mask_ct": candidate_mask.T,
        "rescued_mask_ct": rescued_mask.T,
        "mcd_only_mask_ct": masks["mcd_only"].T,
        "mcd_tau_p_mask_ct": masks["mcd_tau_p"].T,
        "mcd_tau_p_lowrank_mask_ct": masks["mcd_tau_p_lowrank"].T,
        "mcd_tau_p_cleaned_ct": cleaned_interp.T.astype(np.float32),
        "mcd_tau_p_lowrank_cleaned_ct": cleaned_lowrank.T.astype(np.float32),
        "removed_tau_p_ct": (data_tc - cleaned_interp).T.astype(np.float32),
        "removed_tau_p_lowrank_ct": (data_tc - cleaned_lowrank).T.astype(np.float32),
    }

    candidate_ratio = float(np.mean(candidate_mask))
    final_ratio = float(np.mean(masks["mcd_tau_p"]))
    rescue_ratio = float(np.sum(rescued_mask) / (np.sum(candidate_mask) + 1e-30))
    outside_final = ~masks["mcd_tau_p"]
    corr_outside = np.nan
    if np.sum(outside_final) > 10:
        a = data_tc[outside_final].ravel()
        b = cleaned_interp[outside_final].ravel()
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            corr_outside = float(np.corrcoef(a, b)[0, 1])

    summary = {
        "case_id": case["case_id"],
        "source_file": str(case_path),
        "n_channels": int(n_channels),
        "n_samples": int(n_time),
        "fs": float(fs),
        "duration_s": float(n_time / fs),
        "n_mcd_events": int(len(event_logs)),
        "n_rescued_events": int(sum(1 for row in event_logs if row["rescued_tau_p"])),
        "candidate_mask_ratio": candidate_ratio,
        "final_mask_ratio_tau_p": final_ratio,
        "rescued_mask_ratio_vs_candidate": rescue_ratio,
        "raw_rms": rms(data_tc),
        "removed_rms_tau_p": rms(data_tc - cleaned_interp),
        "removed_rms_tau_p_lowrank": rms(data_tc - cleaned_lowrank),
        "outside_mask_corr_tau_p": corr_outside,
    }

    return event_logs, outputs, summary


def save_case_output(case_path: Path, outputs: Dict[str, np.ndarray], summary: Dict) -> Path:
    """保存单个 case 的结果 npz。"""
    out_path = OUTPUT_NPZ_DIR / f"{case_path.stem}_mcsand_real_output.npz"
    payload = {k: v for k, v in outputs.items()}
    payload["summary_json"] = np.array(json.dumps(summary, ensure_ascii=False, indent=2))
    payload["source_case_file"] = np.array(str(case_path))
    np.savez_compressed(out_path, **payload)
    print(f"[OK] {out_path}")
    return out_path


# =============================================================================
# 7. 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("FORESEE 真实数据 MCS-AND 应用")
    print("=" * 80)
    print(f"输入 case 目录: {CASE_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"CHANNEL_STRIDE = {CHANNEL_STRIDE}")
    print(f"τ-p 表观速度范围: {RUNTIME_PARAMS['V_APP_MIN']} - {RUNTIME_PARAMS['V_APP_MAX']} m/s")

    case_files = collect_case_files()
    print(f"case 数量: {len(case_files)}")

    feature_df = collect_feature_library(case_files)
    if feature_df.empty:
        raise RuntimeError("没有收集到候选事件，请降低 STA/LTA 阈值或检查输入数据。")

    feature_csv = OUT_DIR / "real_candidate_feature_library.csv"
    feature_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {feature_csv}")

    model, diag = fit_real_mcd_model(feature_df)
    diag_csv = OUT_DIR / "real_mcd_diagnostics.csv"
    diag.to_csv(diag_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {diag_csv}")
    print(f"MCD 阈值: chi2={model['chi2_thr']:.3f}, empirical={model['empirical_thr']:.3f}, used={model['d2_thr']:.3f}")

    all_events: List[Dict] = []
    summaries: List[Dict] = []
    output_files: List[str] = []
    for case_path in case_files:
        print("-" * 80)
        print(f"处理: {case_path.name}")
        events, outputs, summary = process_case(case_path, model)
        all_events.extend(events)
        summaries.append(summary)
        output_files.append(str(save_case_output(case_path, outputs, summary)))

    event_df = pd.DataFrame(all_events)
    event_csv = OUT_DIR / "real_mcsand_event_log.csv"
    event_df.to_csv(event_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {event_csv}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUT_DIR / "real_mcsand_case_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] {summary_csv}")

    report = {
        "case_dir": str(CASE_DIR),
        "out_dir": str(OUT_DIR),
        "case_files": [str(p) for p in case_files],
        "output_files": output_files,
        "slicing_params": SLICING_PARAMS,
        "runtime_params": RUNTIME_PARAMS,
        "channel_stride": CHANNEL_STRIDE,
        "mcd_feature_cols": MCD_FEATURE_COLS,
        "mahal_alpha": MAHAL_ALPHA,
        "empirical_d2_quantile": EMPIRICAL_D2_QUANTILE,
        "mcd_thresholds": {
            "chi2_thr": model["chi2_thr"],
            "empirical_thr": model["empirical_thr"],
            "used_d2_thr": model["d2_thr"],
        },
        "n_feature_events": int(len(feature_df)),
        "n_mcd_logged_events": int(len(event_df)),
    }
    report_json = OUT_DIR / "real_mcsand_report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {report_json}")

    print("=" * 80)
    print("[完成] 真实数据 MCS-AND 应用完成。")
    print("下一步：运行 plot_real_foresee_mcsand_figures.py 生成论文图。")
    print("=" * 80)


if __name__ == "__main__":
    main()
