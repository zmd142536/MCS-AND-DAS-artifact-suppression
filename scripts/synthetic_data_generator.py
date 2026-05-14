# -*- coding: utf-8 -*-
"""
MCS-AND 论文用合成 DAS 基准数据生成器。

本修订版用于生成论文实验所需的合成数据集，并显式保存事件级瞬态
伪迹剔除任务所需的真值字段：

    data_noisy      = data_clean + artifact
    data_clean      = background + signal_only
    artifact_mask   = 伪迹真实位置
    signal_mask     = 真实传播信号位置

脚本同时生成 signal_rescue 子集，用于专门测试 tau-p 相干性救回模块。

推荐使用流程：
    1. 先保持 RUN_MODE = "pilot"，运行一次检查依赖和输出结构。
    2. 确认无误后改为 RUN_MODE = "paper_minimum"，生成最低可投稿规模数据。
    3. 若需要补充材料中的更强 Monte Carlo 统计，可改为 RUN_MODE = "paper_extended"。

说明：变量名、函数名和 npz 字段名保留英文，方便后续程序读取；注释、
图标题和终端提示使用中文，便于论文实验记录和人工核查。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# =============================================================================
# 用户配置区
# =============================================================================

# 第一次请先使用 "pilot"。正式论文基准实验使用 "paper_minimum"。
RUN_MODE = "paper_minimum"  # 可选："pilot", "paper_minimum", "paper_extended"

# 若为 None，输出目录会自动创建在本脚本所在文件夹下。
OUT_ROOT_OVERRIDE =r"D:\项目实验\积石山实验\DAS相关\去噪论文\合成数据"

# 可选真实背景数据。若为 None，则生成具有空间相关性的合成 DAS 背景。
# H5 数据集形状可以是 [time, channel] 或 [channel, time]。
REAL_BACKGROUND_H5 = None
REAL_BG_DATASET = "denoised"

RANDOM_SEED = 42
PLOT_EXAMPLES = True
N_PLOT_EXAMPLES = 2


@dataclass(frozen=True)
class Preset:
    fs: float
    dx_m: float
    n_channels: int
    duration_s: float
    snr_list_db: Tuple[int, ...]
    n_samples_per_snr: int
    signal_snr_db: Tuple[int, ...]
    n_signal_samples: int
    signal_vapp_list: Tuple[int, ...]
    bg_fmin: float
    bg_fmax: float
    bg_spatial_corr_len: float
    bg_amplitude: float
    ricker_freq_min: float
    ricker_freq_max: float
    burst_freq_min: float
    burst_freq_max: float


PRESETS: Dict[str, Preset] = {
    # 快速流程检查配置，规模应保持较小。
    "pilot": Preset(
        fs=500.0,
        dx_m=2.0,
        n_channels=48,
        duration_s=8.0,
        snr_list_db=(-5, 0, 5),
        n_samples_per_snr=3,
        signal_snr_db=(-5, 0, 5),
        n_signal_samples=3,
        signal_vapp_list=(200, 500, 1000, 2000),
        bg_fmin=1.0,
        bg_fmax=80.0,
        bg_spatial_corr_len=5.0,
        bg_amplitude=1.0,
        ricker_freq_min=8.0,
        ricker_freq_max=80.0,
        burst_freq_min=50.0,
        burst_freq_max=220.0,
    ),
    # 最低可投稿版本的 Monte Carlo 基准配置。
    "paper_minimum": Preset(
        fs=500.0,
        dx_m=2.0,
        n_channels=200,
        duration_s=30.0,
        snr_list_db=(-5, 0, 5, 10, 15),
        n_samples_per_snr=50,
        signal_snr_db=(-5, 0, 5, 10, 15),
        n_signal_samples=50,
        signal_vapp_list=(200, 500, 1000, 2000, 3000),
        bg_fmin=1.0,
        bg_fmax=80.0,
        bg_spatial_corr_len=5.0,
        bg_amplitude=1.0,
        ricker_freq_min=5.0,
        ricker_freq_max=80.0,
        burst_freq_min=50.0,
        burst_freq_max=220.0,
    ),
    # 用于补充材料的增强统计配置。
    "paper_extended": Preset(
        fs=500.0,
        dx_m=2.0,
        n_channels=300,
        duration_s=30.0,
        snr_list_db=(-10, -5, 0, 5, 10, 15),
        n_samples_per_snr=100,
        signal_snr_db=(-5, 0, 5, 10, 15),
        n_signal_samples=100,
        signal_vapp_list=(200, 500, 1000, 2000, 3000, 5000),
        bg_fmin=1.0,
        bg_fmax=100.0,
        bg_spatial_corr_len=6.0,
        bg_amplitude=1.0,
        ricker_freq_min=5.0,
        ricker_freq_max=100.0,
        burst_freq_min=50.0,
        burst_freq_max=220.0,
    ),
}


# 与论文框架一致的五类伪迹。
ARTIFACT_TYPES = (
    "spike",                # T1：单道或稀疏多道尖峰
    "noncoherent_burst",    # T2：多道非相干局部突发
    "moving",               # T3：低速移动人为干扰
    "narrowband",           # T4：持续性/窄带机械干扰
    "hard_case",            # T5：形态接近真实传播波场的难例伪迹
)


# =============================================================================
# 工具函数
# =============================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    p_sig = np.mean(np.asarray(signal, dtype=np.float64) ** 2) + 1e-30
    p_noi = np.mean(np.asarray(noise, dtype=np.float64) ** 2) + 1e-30
    return float(10.0 * np.log10(p_sig / p_noi))


def scale_to_target_snr(reference: np.ndarray, component: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float]:
    """缩放 component，使 SNR(reference, scaled_component) 等于目标值。"""
    p_ref = np.mean(np.asarray(reference, dtype=np.float64) ** 2) + 1e-30
    p_cmp = np.mean(np.asarray(component, dtype=np.float64) ** 2) + 1e-30
    target_p_cmp = p_ref / (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_p_cmp / p_cmp)
    scaled = component * scale
    return scaled, compute_snr_db(reference, scaled)


def active_rms(x: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return rms(x[mask])
    return rms(x)


def scale_signal_to_target_snr(signal_only: np.ndarray, noise_reference: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float]:
    """缩放信号，使有效信号能量相对背景达到目标 SNR。"""
    mask = np.abs(signal_only) > 1e-12
    p_sig = np.mean(signal_only[mask] ** 2) + 1e-30 if np.any(mask) else np.mean(signal_only ** 2) + 1e-30
    p_noise = np.mean(noise_reference ** 2) + 1e-30
    target_p_sig = p_noise * (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(target_p_sig / p_sig)
    scaled = signal_only * scale
    actual = 10.0 * np.log10((active_rms(scaled, mask) ** 2 + 1e-30) / (p_noise + 1e-30))
    return scaled, float(actual)


def sanitize_snr_label(snr_db: int | float) -> str:
    return f"snr_{int(snr_db):+03d}"


def make_meta_json(meta: Dict) -> str:
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


# =============================================================================
# 背景与传播信号生成
# =============================================================================


def generate_synthetic_background(
    n_samples: int,
    n_channels: int,
    fs: float,
    rng: np.random.Generator,
    fmin: float,
    fmax: float,
    spatial_corr_len: float,
    amplitude: float,
) -> np.ndarray:
    ch_idx = np.arange(n_channels)
    dist = np.abs(ch_idx[:, None] - ch_idx[None, :])
    cov = np.exp(-dist / spatial_corr_len) + np.eye(n_channels) * 1e-6
    chol = np.linalg.cholesky(cov)

    white = rng.standard_normal((n_samples, n_channels))
    data = white @ chol.T

    nyq = fs / 2.0
    low = max(fmin / nyq, 0.001)
    high = min(fmax / nyq, 0.995)
    sos = sp_signal.butter(4, [low, high], btype="band", output="sos")
    data = sp_signal.sosfiltfilt(sos, data, axis=0)

    data *= amplitude / (rms(data) + 1e-12)
    return data.astype(np.float32)


def load_real_background(path: str, dataset_name: str, n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        data = np.asarray(f[dataset_name][:], dtype=np.float32)

    if data.shape[0] < data.shape[1]:
        data = data.T

    t_total, c_total = data.shape
    if c_total >= n_channels:
        c0 = int(rng.integers(0, c_total - n_channels + 1))
        data = data[:, c0 : c0 + n_channels]
    else:
        repeats = int(np.ceil(n_channels / c_total))
        data = np.tile(data, (1, repeats))[:, :n_channels]

    if t_total >= n_samples:
        t0 = int(rng.integers(0, t_total - n_samples + 1))
        data = data[t0 : t0 + n_samples, :]
    else:
        repeats = int(np.ceil(n_samples / t_total))
        data = np.tile(data, (repeats, 1))[:n_samples, :]

    data -= np.mean(data, axis=0, keepdims=True)
    data /= rms(data) + 1e-12
    return data.astype(np.float32)


def get_background(
    cfg: Preset,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if REAL_BACKGROUND_H5 and Path(REAL_BACKGROUND_H5).is_file():
        return load_real_background(REAL_BACKGROUND_H5, REAL_BG_DATASET, n_samples, cfg.n_channels, rng)
    return generate_synthetic_background(
        n_samples=n_samples,
        n_channels=cfg.n_channels,
        fs=cfg.fs,
        rng=rng,
        fmin=cfg.bg_fmin,
        fmax=cfg.bg_fmax,
        spatial_corr_len=cfg.bg_spatial_corr_len,
        amplitude=cfg.bg_amplitude,
    )


def ricker_wavelet(freq_hz: float, fs: float, duration_s: float | None = None) -> np.ndarray:
    if duration_s is None:
        duration_s = max(2.0 / freq_hz, 0.04)
    n = max(5, int(round(duration_s * fs)))
    if n % 2 == 0:
        n += 1
    t = np.arange(n) / fs - duration_s / 2.0
    a = (np.pi * freq_hz * t) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    wavelet /= np.max(np.abs(wavelet)) + 1e-12
    return wavelet.astype(np.float32)


def inject_moveout_signal(
    shape: Tuple[int, int],
    fs: float,
    dx_m: float,
    v_app: float,
    rng: np.random.Generator,
    freq_min: float,
    freq_max: float,
    ref_time_margin_s: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_samples, n_channels = shape
    signal_only = np.zeros(shape, dtype=np.float32)
    signal_mask = np.zeros(shape, dtype=bool)

    freq = float(rng.uniform(freq_min, freq_max))
    wavelet = ricker_wavelet(freq, fs)
    wav_len = len(wavelet)

    ref_channel = int(rng.integers(n_channels // 4, max(n_channels // 4 + 1, 3 * n_channels // 4)))
    margin = max(ref_time_margin_s, wav_len / fs + 0.2)
    if n_samples / fs <= 2 * margin:
        ref_time_s = n_samples / fs / 2.0
    else:
        ref_time_s = float(rng.uniform(margin, n_samples / fs - margin))
    ref_sample = int(round(ref_time_s * fs))
    polarity = float(rng.choice([-1.0, 1.0]))

    for ch in range(n_channels):
        delay_s = (ch - ref_channel) * dx_m / v_app
        delay = int(round(delay_s * fs))
        arrival = ref_sample + delay
        start = arrival - wav_len // 2
        end = start + wav_len
        if start < 0 or end > n_samples:
            continue
        taper = 0.85 + 0.3 * rng.random()
        signal_only[start:end, ch] += polarity * taper * wavelet
        signal_mask[start:end, ch] = True

    meta = {
        "signal_v_app_ms": float(v_app),
        "signal_freq_hz": freq,
        "signal_ref_channel": ref_channel,
        "signal_ref_time_s": ref_time_s,
        "signal_wavelet_samples": wav_len,
    }
    return signal_only, signal_mask, meta


# =============================================================================
# 伪迹生成
# =============================================================================


def generate_spike_artifact(shape: Tuple[int, int], fs: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_samples, n_channels = shape
    artifact = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    affected_fraction = float(rng.uniform(0.05, 0.25))
    n_affected = max(1, int(round(n_channels * affected_fraction)))
    channels = rng.choice(n_channels, size=n_affected, replace=False)

    for ch in channels:
        n_spikes = int(rng.integers(3, 9))
        for _ in range(n_spikes):
            pos = int(rng.integers(0, n_samples))
            width = int(rng.integers(1, max(2, int(0.01 * fs))))
            amp = float(rng.uniform(8.0, 25.0) * rng.choice([-1.0, 1.0]))
            half = width // 2
            idx = np.arange(-half, half + 1)
            if len(idx) == 1:
                pulse = np.array([amp], dtype=np.float32)
            else:
                pulse = amp * np.exp(-0.5 * (idx / max(1.0, width / 4.0)) ** 2)
            start = max(0, pos - half)
            end = min(n_samples, pos + half + 1)
            ps = start - (pos - half)
            pe = ps + (end - start)
            artifact[start:end, ch] += pulse[ps:pe]
            mask[start:end, ch] = True

    return artifact, mask, {"artifact_subtype": "sparse_spike", "affected_channels": int(n_affected)}


def generate_noncoherent_burst_artifact(shape: Tuple[int, int], fs: float, rng: np.random.Generator, cfg: Preset) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_samples, n_channels = shape
    artifact = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    n_events = int(rng.integers(2, 6))
    for _ in range(n_events):
        dur_s = float(rng.uniform(0.05, 0.6))
        dur = max(8, int(round(dur_s * fs)))
        start = int(rng.integers(0, max(1, n_samples - dur)))
        end = start + dur

        center_ch = int(rng.integers(0, n_channels))
        width_ch = int(rng.integers(max(3, n_channels // 40), max(4, n_channels // 8)))
        ch0 = max(0, center_ch - width_ch)
        ch1 = min(n_channels, center_ch + width_ch + 1)
        channels = np.arange(ch0, ch1)

        t = np.arange(dur) / fs
        envelope = sp_signal.windows.tukey(dur, alpha=0.6)
        for ch in channels:
            freq = float(rng.uniform(cfg.burst_freq_min, cfg.burst_freq_max))
            phase = float(rng.uniform(0, 2 * np.pi))
            amp = float(rng.uniform(4.0, 16.0))
            local = amp * envelope * np.sin(2 * np.pi * freq * t + phase)
            local += 0.25 * amp * envelope * rng.standard_normal(dur)
            artifact[start:end, ch] += local.astype(np.float32)
            mask[start:end, ch] = True

    return artifact, mask, {"artifact_subtype": "multichannel_noncoherent_burst", "n_events": n_events}


def generate_moving_artifact(shape: Tuple[int, int], fs: float, dx_m: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_samples, n_channels = shape
    artifact = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    v_app = float(rng.uniform(3.0, 30.0))
    direction = float(rng.choice([-1.0, 1.0]))
    freq = float(rng.uniform(8.0, 45.0))
    dur_s = float(rng.uniform(0.15, 0.8))
    wave = ricker_wavelet(freq, fs, duration_s=dur_s)
    wav_len = len(wave)
    ref_ch = int(rng.integers(n_channels // 5, max(n_channels // 5 + 1, 4 * n_channels // 5)))
    ref_time = float(rng.uniform(1.0, max(1.1, n_samples / fs - 1.0)))
    ref_sample = int(round(ref_time * fs))

    for ch in range(n_channels):
        delay_s = direction * (ch - ref_ch) * dx_m / v_app
        arrival = ref_sample + int(round(delay_s * fs))
        start = arrival - wav_len // 2
        end = start + wav_len
        if start < 0 or end > n_samples:
            continue
        amp = 8.0 * np.exp(-abs(ch - ref_ch) / max(3.0, n_channels / 10.0))
        artifact[start:end, ch] += (amp * wave).astype(np.float32)
        mask[start:end, ch] = True

    meta = {"artifact_subtype": "low_speed_moving", "artifact_v_app_ms": v_app, "artifact_freq_hz": freq}
    return artifact, mask, meta


def generate_narrowband_artifact(shape: Tuple[int, int], fs: float, rng: np.random.Generator, cfg: Preset) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_samples, n_channels = shape
    artifact = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    n_events = int(rng.integers(1, 4))
    freqs = []
    for _ in range(n_events):
        dur_s = float(rng.uniform(0.5, 2.5))
        dur = min(n_samples, max(16, int(round(dur_s * fs))))
        start = int(rng.integers(0, max(1, n_samples - dur)))
        end = start + dur
        freq = float(rng.uniform(cfg.burst_freq_min, cfg.burst_freq_max))
        freqs.append(freq)
        phase0 = float(rng.uniform(0, 2 * np.pi))
        t = np.arange(dur) / fs
        envelope = sp_signal.windows.tukey(dur, alpha=0.2)

        center_ch = int(rng.integers(0, n_channels))
        width_ch = int(rng.integers(max(5, n_channels // 20), max(6, n_channels // 3)))
        ch0 = max(0, center_ch - width_ch)
        ch1 = min(n_channels, center_ch + width_ch + 1)

        for ch in range(ch0, ch1):
            phase = phase0 + rng.normal(0.0, 0.3)
            amp = float(rng.uniform(2.0, 8.0))
            artifact[start:end, ch] += (amp * envelope * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
            mask[start:end, ch] = True

    return artifact, mask, {"artifact_subtype": "narrowband_mechanical", "n_events": n_events, "artifact_freqs_hz": freqs}


def generate_hard_case_artifact(shape: Tuple[int, int], fs: float, dx_m: float, rng: np.random.Generator, cfg: Preset) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # 该伪迹被刻意设计成接近真实传播波场，用于展示方法边界。
    v_app = float(rng.uniform(800.0, 3000.0))
    artifact, mask, meta = inject_moveout_signal(
        shape=shape,
        fs=fs,
        dx_m=dx_m,
        v_app=v_app,
        rng=rng,
        freq_min=max(8.0, cfg.ricker_freq_min),
        freq_max=cfg.ricker_freq_max,
        ref_time_margin_s=1.0,
    )
    # Distort it slightly so it remains an artifact but is physically ambiguous.
    artifact *= float(rng.uniform(3.0, 8.0))
    artifact += 0.15 * rms(artifact) * rng.standard_normal(shape).astype(np.float32) * mask
    meta = {
        "artifact_subtype": "hard_case_signal_like",
        "artifact_v_app_ms": v_app,
        "artifact_freq_hz": meta["signal_freq_hz"],
    }
    return artifact.astype(np.float32), mask, meta


def generate_artifact(artifact_type: str, shape: Tuple[int, int], cfg: Preset, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if artifact_type == "spike":
        return generate_spike_artifact(shape, cfg.fs, rng)
    if artifact_type == "noncoherent_burst":
        return generate_noncoherent_burst_artifact(shape, cfg.fs, rng, cfg)
    if artifact_type == "moving":
        return generate_moving_artifact(shape, cfg.fs, cfg.dx_m, rng)
    if artifact_type == "narrowband":
        return generate_narrowband_artifact(shape, cfg.fs, rng, cfg)
    if artifact_type == "hard_case":
        return generate_hard_case_artifact(shape, cfg.fs, cfg.dx_m, rng, cfg)
    raise ValueError(f"未知伪迹类型: {artifact_type}")


# =============================================================================
# 绘图函数
# =============================================================================


def plot_benchmark_example(
    data_clean: np.ndarray,
    data_noisy: np.ndarray,
    artifact: np.ndarray,
    artifact_mask: np.ndarray,
    artifact_type: str,
    snr_db: float,
    sample_idx: int,
    fs: float,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    t = np.arange(data_clean.shape[0]) / fs
    vmax = max(1e-6, 3 * np.std(data_clean))

    panels = [
        ("干净数据 = 背景 + 真实信号", data_clean, "seismic", -vmax, vmax),
        (f"含伪迹数据（{artifact_type}，目标 SNR={snr_db} dB）", data_noisy, "seismic", -vmax, vmax),
        ("纯伪迹", artifact, "seismic", -vmax, vmax),
        ("伪迹真值掩膜", artifact_mask.astype(float), "Reds", 0, 1),
    ]

    for ax, (title, data, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(data.T, aspect="auto", cmap=cmap, extent=[t[0], t[-1], data.shape[1], 0], vmin=vmin, vmax=vmax_i)
        ax.set_ylabel("通道")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    axes[-1].set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"benchmark_{artifact_type}_snr{int(snr_db):+03d}_{sample_idx:03d}.png", dpi=150)
    plt.close(fig)


def plot_signal_rescue_example(
    background: np.ndarray,
    data_noisy: np.ndarray,
    signal_only: np.ndarray,
    signal_mask: np.ndarray,
    v_app: float,
    snr_db: float,
    sample_idx: int,
    fs: float,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    t = np.arange(background.shape[0]) / fs
    vmax = max(1e-6, 3 * np.std(background))
    panels = [
        ("背景", background, "seismic", -vmax, vmax),
        (f"背景 + 真实信号 + 随机噪声（v_app={v_app} m/s，SNR={snr_db} dB）", data_noisy, "seismic", -vmax, vmax),
        ("纯真实信号", signal_only, "seismic", -vmax, vmax),
        ("真实信号掩膜", signal_mask.astype(float), "Reds", 0, 1),
    ]
    for ax, (title, data, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(data.T, aspect="auto", cmap=cmap, extent=[t[0], t[-1], data.shape[1], 0], vmin=vmin, vmax=vmax_i)
        ax.set_ylabel("通道")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("时间 (s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"signal_rescue_vapp{int(v_app)}_snr{int(snr_db):+03d}_{sample_idx:03d}.png", dpi=150)
    plt.close(fig)


# =============================================================================
# 数据集生成
# =============================================================================


def save_npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def estimate_raw_size_gb(cfg: Preset) -> float:
    n_samples = int(round(cfg.fs * cfg.duration_s))
    n_benchmark = len(ARTIFACT_TYPES) * len(cfg.snr_list_db) * cfg.n_samples_per_snr
    n_signal = len(cfg.signal_vapp_list) * len(cfg.signal_snr_db) * cfg.n_signal_samples
    # 粗略估算：benchmark 每个样本约含 3 个浮点数组和 2 个布尔掩膜；
    # signal_rescue 每个样本约含 3 个浮点数组和 1 个布尔掩膜。
    float_bytes = 4
    bool_bytes = 1
    per_benchmark = n_samples * cfg.n_channels * (3 * float_bytes + 2 * bool_bytes)
    per_signal = n_samples * cfg.n_channels * (3 * float_bytes + bool_bytes)
    return (n_benchmark * per_benchmark + n_signal * per_signal) / (1024**3)


def generate_benchmark_dataset(cfg: Preset, out_root: Path, rng: np.random.Generator) -> List[Dict]:
    n_samples = int(round(cfg.fs * cfg.duration_s))
    shape = (n_samples, cfg.n_channels)
    rows: List[Dict] = []
    example_dir = out_root / "examples"
    ensure_dir(example_dir)

    for artifact_type in ARTIFACT_TYPES:
        for snr_db in cfg.snr_list_db:
            snr_label = sanitize_snr_label(snr_db)
            sample_dir = out_root / "benchmark" / artifact_type / snr_label
            ensure_dir(sample_dir)
            n_plotted = 0

            for sample_idx in range(cfg.n_samples_per_snr):
                background = get_background(cfg, n_samples, rng)
                signal_vapp = float(rng.choice(cfg.signal_vapp_list))
                signal_only, signal_mask, signal_meta = inject_moveout_signal(
                    shape=shape,
                    fs=cfg.fs,
                    dx_m=cfg.dx_m,
                    v_app=signal_vapp,
                    rng=rng,
                    freq_min=cfg.ricker_freq_min,
                    freq_max=cfg.ricker_freq_max,
                )
                # 真实信号强度保持在中等范围；伪迹 SNR 在后续步骤中单独控制。
                signal_only *= float(rng.uniform(0.5, 1.5))
                data_clean = (background + signal_only).astype(np.float32)

                artifact_raw, artifact_mask, artifact_meta = generate_artifact(artifact_type, shape, cfg, rng)
                artifact_scaled, actual_snr = scale_to_target_snr(data_clean, artifact_raw, snr_db)
                data_noisy = (data_clean + artifact_scaled).astype(np.float32)

                meta = {
                    "category": "benchmark",
                    "artifact_type": artifact_type,
                    "snr_target_db": float(snr_db),
                    "snr_actual_db": float(actual_snr),
                    "sample_idx": int(sample_idx),
                    "fs": float(cfg.fs),
                    "dx_m": float(cfg.dx_m),
                    "n_channels": int(cfg.n_channels),
                    "n_samples": int(n_samples),
                    **signal_meta,
                    **artifact_meta,
                }

                out_path = sample_dir / f"sample_{sample_idx:04d}.npz"
                save_npz(
                    out_path,
                    data_noisy=data_noisy.astype(np.float32),
                    data_clean=data_clean.astype(np.float32),
                    artifact_only=artifact_scaled.astype(np.float32),
                    artifact_mask=artifact_mask,
                    signal_only=signal_only.astype(np.float32),
                    signal_mask=signal_mask,
                    fs=np.float32(cfg.fs),
                    dx_m=np.float32(cfg.dx_m),
                    snr_target_db=np.float32(snr_db),
                    snr_actual_db=np.float32(actual_snr),
                    artifact_type=np.array(artifact_type),
                    meta_json=np.array(make_meta_json(meta)),
                )

                rows.append({**meta, "filepath": str(out_path)})

                if PLOT_EXAMPLES and n_plotted < N_PLOT_EXAMPLES:
                    plot_benchmark_example(
                        data_clean=data_clean,
                        data_noisy=data_noisy,
                        artifact=artifact_scaled,
                        artifact_mask=artifact_mask,
                        artifact_type=artifact_type,
                        snr_db=snr_db,
                        sample_idx=sample_idx,
                        fs=cfg.fs,
                        out_dir=example_dir,
                    )
                    n_plotted += 1

            print(f"[OK] 基准数据 {artifact_type:18s} {snr_label}: {cfg.n_samples_per_snr} 个样本")

    return rows


def generate_signal_rescue_dataset(cfg: Preset, out_root: Path, rng: np.random.Generator) -> List[Dict]:
    n_samples = int(round(cfg.fs * cfg.duration_s))
    shape = (n_samples, cfg.n_channels)
    rows: List[Dict] = []
    example_dir = out_root / "examples"
    ensure_dir(example_dir)

    for v_app in cfg.signal_vapp_list:
        for snr_db in cfg.signal_snr_db:
            snr_label = sanitize_snr_label(snr_db)
            sample_dir = out_root / "signal_rescue" / f"vapp_{int(v_app)}" / snr_label
            ensure_dir(sample_dir)
            n_plotted = 0

            for sample_idx in range(cfg.n_signal_samples):
                background = get_background(cfg, n_samples, rng)
                signal_raw, signal_mask, signal_meta = inject_moveout_signal(
                    shape=shape,
                    fs=cfg.fs,
                    dx_m=cfg.dx_m,
                    v_app=float(v_app),
                    rng=rng,
                    freq_min=cfg.ricker_freq_min,
                    freq_max=cfg.ricker_freq_max,
                )
                signal_scaled, actual_snr = scale_signal_to_target_snr(signal_raw, background, snr_db)
                additive_noise = 0.15 * rms(background) * rng.standard_normal(shape).astype(np.float32)
                data_clean = (background + signal_scaled).astype(np.float32)
                data_noisy = (data_clean + additive_noise).astype(np.float32)

                meta = {
                    "category": "signal_rescue",
                    "artifact_type": "none",
                    "snr_target_db": float(snr_db),
                    "snr_actual_db": float(actual_snr),
                    "sample_idx": int(sample_idx),
                    "fs": float(cfg.fs),
                    "dx_m": float(cfg.dx_m),
                    "n_channels": int(cfg.n_channels),
                    "n_samples": int(n_samples),
                    **signal_meta,
                }

                out_path = sample_dir / f"sample_{sample_idx:04d}.npz"
                save_npz(
                    out_path,
                    data_noisy=data_noisy.astype(np.float32),
                    data_clean=data_clean.astype(np.float32),
                    background=background.astype(np.float32),
                    signal_only=signal_scaled.astype(np.float32),
                    signal_mask=signal_mask,
                    artifact_mask=np.zeros(shape, dtype=bool),
                    fs=np.float32(cfg.fs),
                    dx_m=np.float32(cfg.dx_m),
                    snr_target_db=np.float32(snr_db),
                    snr_actual_db=np.float32(actual_snr),
                    v_app_ms=np.float32(v_app),
                    meta_json=np.array(make_meta_json(meta)),
                )

                rows.append({**meta, "filepath": str(out_path)})

                if PLOT_EXAMPLES and n_plotted < N_PLOT_EXAMPLES:
                    plot_signal_rescue_example(
                        background=background,
                        data_noisy=data_noisy,
                        signal_only=signal_scaled,
                        signal_mask=signal_mask,
                        v_app=float(v_app),
                        snr_db=snr_db,
                        sample_idx=sample_idx,
                        fs=cfg.fs,
                        out_dir=example_dir,
                    )
                    n_plotted += 1

            print(f"[OK] 信号救回 vapp_{int(v_app):04d} {snr_label}: {cfg.n_signal_samples} 个样本")

    return rows


def main() -> None:
    if RUN_MODE not in PRESETS:
        raise ValueError(f"RUN_MODE 必须是以下选项之一: {list(PRESETS)}")

    cfg = PRESETS[RUN_MODE]
    rng = np.random.default_rng(RANDOM_SEED)
    script_dir = Path(__file__).resolve().parent
    out_root = Path(OUT_ROOT_OVERRIDE) if OUT_ROOT_OVERRIDE else script_dir / f"synthetic_dataset_{RUN_MODE}"
    ensure_dir(out_root)

    n_samples = int(round(cfg.fs * cfg.duration_s))
    n_benchmark = len(ARTIFACT_TYPES) * len(cfg.snr_list_db) * cfg.n_samples_per_snr
    n_signal = len(cfg.signal_vapp_list) * len(cfg.signal_snr_db) * cfg.n_signal_samples
    estimated_gb = estimate_raw_size_gb(cfg)

    print("=" * 78)
    print("MCS-AND 论文用合成 DAS 基准数据生成器")
    print("=" * 78)
    print(f"运行模式              : {RUN_MODE}")
    print(f"输出目录              : {out_root}")
    print(f"采样率 / 道间距       : {cfg.fs} Hz / {cfg.dx_m} m")
    print(f"通道数 / 时长         : {cfg.n_channels} / {cfg.duration_s} s")
    print(f"每段采样点数          : {n_samples}")
    print(f"伪迹类型              : {list(ARTIFACT_TYPES)}")
    print(f"伪迹 SNR 档位         : {list(cfg.snr_list_db)} dB")
    print(f"基准样本数            : {n_benchmark}")
    print(f"信号救回样本数        : {n_signal}")
    print(f"压缩前粗略原始体积    : {estimated_gb:.1f} GB")
    print("=" * 78)

    rows: List[Dict] = []
    rows.extend(generate_benchmark_dataset(cfg, out_root, rng))
    rows.extend(generate_signal_rescue_dataset(cfg, out_root, rng))

    summary = pd.DataFrame(rows)
    summary_csv = out_root / "dataset_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    config_json = out_root / "generator_config.json"
    config_payload = {
        "run_mode": RUN_MODE,
        "random_seed": RANDOM_SEED,
        "preset": cfg.__dict__,
        "artifact_types": list(ARTIFACT_TYPES),
        "real_background_h5": REAL_BACKGROUND_H5,
        "real_bg_dataset": REAL_BG_DATASET,
    }
    config_json.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("[完成]")
    print(f"汇总 CSV : {summary_csv}")
    print(f"配置 JSON: {config_json}")
    print(f"示例图   : {out_root / 'examples'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
