# -*- coding: utf-8 -*-
"""
合成 DAS 数据专用 STA/LTA 候选切片参数校准脚本。

为什么不用 DAS_calibrate_sta_lta_params_v2-1.py 直接跑？
    原脚本面向真实 H5 噪声文件，默认 FS=1000 Hz，频带到 500 Hz；
    当前论文合成数据是 .npz、FS=500 Hz，并且带有 artifact_mask/signal_mask 真值。
    因此本脚本直接利用真值 mask 评价 STA/LTA 是否能高召回地覆盖伪迹和真实弱信号。

输出：
    sta_lta_calibration_summary.csv       参数组合总表
    sta_lta_calibration_by_type.csv       按伪迹类型/SNR 分组统计
    best_sta_lta_params.txt               推荐参数
    sta_lta_calibration_report.json       机器可读报告

使用：
    在 Spyder 或终端中直接运行本脚本即可。若路径不同，修改 DATASET_ROOT。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 硬编码输入输出路径
# =============================================================================

DATASET_ROOT = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\合成数据")
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\STALTAqiepian")

# 为避免第一次校准过慢，默认每个 category/type/SNR 组合最多抽取若干文件。
# 设置为 0 表示使用全部样本。
MAX_FILES_PER_GROUP = 8

# 每隔多少道检查一个通道。设置为 1 表示检查全部通道。
CHANNEL_STRIDE = 4

RANDOM_SEED = 42


# =============================================================================
# 参数搜索网格
# =============================================================================

STA_LIST = [0.03, 0.05, 0.08]
LTA_LIST = [0.30, 0.50, 0.80]
THRESHOLD_PAIRS = [
    (1.8, 1.1),
    (2.0, 1.2),
    (2.5, 1.5),
    (3.0, 1.8),
]

MIN_DUR_S = 0.02
MERGE_GAP_S = 0.02
PRE_PAD_S = 0.05
POST_PAD_S = 0.10


# =============================================================================
# STA/LTA 工具函数
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


def detect_events_from_ratio(
    ratio: np.ndarray,
    fs: float,
    thr_on: float,
    thr_off: float,
    min_dur_s: float,
    merge_gap_s: float,
) -> List[Tuple[int, int, int, float]]:
    min_dur = max(1, int(round(min_dur_s * fs)))
    merge_gap = max(0, int(round(merge_gap_s * fs)))

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


def build_candidate_mask_for_sample(data: np.ndarray, fs: float, params: Dict, channel_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    n_time = data.shape[0]
    candidate_mask = np.zeros((n_time, len(channel_indices)), dtype=bool)
    event_count = 0

    pre = int(round(params["pre_pad_s"] * fs))
    post = int(round(params["post_pad_s"] * fs))

    for j, ch in enumerate(channel_indices):
        ratio = sta_lta_ratio(data[:, ch], fs, params["sta_s"], params["lta_s"])
        if ratio is None:
            continue
        events = detect_events_from_ratio(
            ratio,
            fs,
            params["thr_on"],
            params["thr_off"],
            params["min_dur_s"],
            params["merge_gap_s"],
        )
        event_count += len(events)
        for start, end, _, _ in events:
            w0 = max(0, start - pre)
            w1 = min(n_time, end + post)
            candidate_mask[w0:w1, j] = True

    return candidate_mask, event_count


def mask_recall(candidate: np.ndarray, truth: np.ndarray) -> float:
    total = int(np.sum(truth))
    if total == 0:
        return np.nan
    return float(np.sum(candidate & truth) / total)


# =============================================================================
# 数据读取与抽样
# =============================================================================

def load_summary(root: Path) -> pd.DataFrame:
    summary_path = root / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 dataset_summary.csv: {summary_path}")
    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if "filepath" not in df.columns:
        raise ValueError("dataset_summary.csv 中缺少 filepath 列。")
    return df


def resolve_path(root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    if not p.is_absolute():
        p = root / p
    return p


def stratified_sample(summary: pd.DataFrame, root: Path, max_files_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = summary.copy()
    df["resolved_path"] = df["filepath"].apply(lambda x: str(resolve_path(root, x)))

    group_cols = [c for c in ["category", "artifact_type", "snr_target_db", "v_app_ms"] if c in df.columns]
    if not group_cols:
        group_cols = ["category"] if "category" in df.columns else []

    if max_files_per_group == 0 or not group_cols:
        return df

    parts = []
    for _, sub in df.groupby(group_cols, dropna=False):
        if len(sub) <= max_files_per_group:
            parts.append(sub)
        else:
            chosen = rng.choice(sub.index.to_numpy(), size=max_files_per_group, replace=False)
            parts.append(sub.loc[np.sort(chosen)])
    return pd.concat(parts, ignore_index=True)


def iter_param_grid() -> Iterable[Dict]:
    for sta_s in STA_LIST:
        for lta_s in LTA_LIST:
            if lta_s <= sta_s:
                continue
            for thr_on, thr_off in THRESHOLD_PAIRS:
                yield {
                    "sta_s": float(sta_s),
                    "lta_s": float(lta_s),
                    "thr_on": float(thr_on),
                    "thr_off": float(thr_off),
                    "min_dur_s": float(MIN_DUR_S),
                    "merge_gap_s": float(MERGE_GAP_S),
                    "pre_pad_s": float(PRE_PAD_S),
                    "post_pad_s": float(POST_PAD_S),
                }


# =============================================================================
# 校准主逻辑
# =============================================================================

def evaluate_one_param_set(sample_df: pd.DataFrame, params: Dict) -> Tuple[Dict, pd.DataFrame]:
    sample_rows: List[Dict] = []

    total_candidate = 0
    total_points = 0
    total_events = 0
    artifact_hit = 0
    artifact_total = 0
    signal_hit = 0
    signal_total = 0
    background_candidate = 0
    background_total = 0

    for idx, row in sample_df.iterrows():
        path = Path(row["resolved_path"])
        if not path.exists():
            print(f"[警告] 文件不存在，跳过: {path}")
            continue

        with np.load(path, allow_pickle=False) as npz:
            data = npz["data_noisy"]
            fs = float(np.asarray(npz["fs"]))
            artifact_mask_full = npz["artifact_mask"].astype(bool)
            signal_mask_full = npz["signal_mask"].astype(bool)

        n_time, n_channels = data.shape
        channel_indices = np.arange(0, n_channels, CHANNEL_STRIDE, dtype=int)

        candidate, event_count = build_candidate_mask_for_sample(data, fs, params, channel_indices)
        artifact_mask = artifact_mask_full[:, channel_indices]
        signal_mask = signal_mask_full[:, channel_indices]
        truth_union = artifact_mask | signal_mask
        background_mask = ~truth_union

        cand_count = int(np.sum(candidate))
        points = int(candidate.size)
        art_total_i = int(np.sum(artifact_mask))
        sig_total_i = int(np.sum(signal_mask))
        bg_total_i = int(np.sum(background_mask))
        art_hit_i = int(np.sum(candidate & artifact_mask))
        sig_hit_i = int(np.sum(candidate & signal_mask))
        bg_cand_i = int(np.sum(candidate & background_mask))

        total_candidate += cand_count
        total_points += points
        total_events += event_count
        artifact_hit += art_hit_i
        artifact_total += art_total_i
        signal_hit += sig_hit_i
        signal_total += sig_total_i
        background_candidate += bg_cand_i
        background_total += bg_total_i

        sample_rows.append({
            "filepath": str(path),
            "category": row.get("category", ""),
            "artifact_type": row.get("artifact_type", row.get("noise_type", "")),
            "snr_target_db": row.get("snr_target_db", np.nan),
            "v_app_ms": row.get("v_app_ms", np.nan),
            "candidate_ratio": cand_count / points if points else np.nan,
            "event_count": event_count,
            "artifact_recall": art_hit_i / art_total_i if art_total_i else np.nan,
            "signal_recall": sig_hit_i / sig_total_i if sig_total_i else np.nan,
            "background_candidate_ratio": bg_cand_i / bg_total_i if bg_total_i else np.nan,
        })

        if (idx + 1) % 50 == 0:
            print(f"    已处理样本 {idx + 1}/{len(sample_df)}")

    artifact_recall = artifact_hit / artifact_total if artifact_total else np.nan
    signal_recall = signal_hit / signal_total if signal_total else np.nan
    candidate_ratio = total_candidate / total_points if total_points else np.nan
    background_candidate_ratio = background_candidate / background_total if background_total else np.nan

    # 评分目标：候选切片阶段偏高召回，但避免候选窗口过多。
    score = 0.0
    if np.isfinite(artifact_recall):
        score += 1.00 * artifact_recall
    if np.isfinite(signal_recall):
        score += 0.50 * signal_recall
    if np.isfinite(candidate_ratio):
        score -= 2.00 * candidate_ratio
    if np.isfinite(background_candidate_ratio):
        score -= 0.50 * background_candidate_ratio

    summary_row = {
        **params,
        "n_samples_checked": int(len(sample_rows)),
        "channel_stride": int(CHANNEL_STRIDE),
        "total_events": int(total_events),
        "candidate_ratio": float(candidate_ratio),
        "background_candidate_ratio": float(background_candidate_ratio),
        "artifact_recall": float(artifact_recall) if np.isfinite(artifact_recall) else np.nan,
        "signal_recall": float(signal_recall) if np.isfinite(signal_recall) else np.nan,
        "score": float(score),
    }

    return summary_row, pd.DataFrame(sample_rows)


def summarize_by_type(sample_metrics: pd.DataFrame, params: Dict) -> pd.DataFrame:
    group_cols = [c for c in ["category", "artifact_type", "snr_target_db"] if c in sample_metrics.columns]
    if not group_cols or sample_metrics.empty:
        return pd.DataFrame()

    grouped = (
        sample_metrics
        .groupby(group_cols, dropna=False)
        .agg(
            n_samples=("filepath", "count"),
            candidate_ratio=("candidate_ratio", "mean"),
            event_count=("event_count", "mean"),
            artifact_recall=("artifact_recall", "mean"),
            signal_recall=("signal_recall", "mean"),
            background_candidate_ratio=("background_candidate_ratio", "mean"),
        )
        .reset_index()
    )
    for key, value in params.items():
        grouped[key] = value
    return grouped


def choose_best(summary_df: pd.DataFrame) -> pd.Series:
    # 优先选择高召回参数；若无参数满足门槛，则退回最高 score。
    candidates = summary_df.copy()
    high_recall = candidates[
        (candidates["artifact_recall"] >= 0.90)
        & (candidates["signal_recall"] >= 0.80)
    ]
    if not high_recall.empty:
        high_recall = high_recall.sort_values(
            ["candidate_ratio", "background_candidate_ratio", "score"],
            ascending=[True, True, False],
        )
        return high_recall.iloc[0]
    return candidates.sort_values("score", ascending=False).iloc[0]


def save_best_params(best: pd.Series, out_dir: Path) -> None:
    keys = ["sta_s", "lta_s", "thr_on", "thr_off", "min_dur_s", "merge_gap_s", "pre_pad_s", "post_pad_s"]
    lines = [
        "# 推荐 STA/LTA 候选切片参数",
        "# 说明：该阶段目标是高召回地产生候选事件，并非最终伪迹判别。",
    ]
    for key in keys:
        lines.append(f"{key}={float(best[key])}")
    lines.extend([
        "",
        "# 校准指标",
        f"artifact_recall={float(best['artifact_recall'])}",
        f"signal_recall={float(best['signal_recall'])}",
        f"candidate_ratio={float(best['candidate_ratio'])}",
        f"background_candidate_ratio={float(best['background_candidate_ratio'])}",
        f"score={float(best['score'])}",
    ])
    (out_dir / "best_sta_lta_params.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("合成 DAS 数据 STA/LTA 候选切片参数校准")
    print("=" * 78)
    print(f"数据集目录: {DATASET_ROOT}")
    print(f"输出目录  : {OUT_DIR}")
    print(f"每组最多样本数: {MAX_FILES_PER_GROUP}（0 表示全量）")
    print(f"通道步长: {CHANNEL_STRIDE}")

    summary = load_summary(DATASET_ROOT)
    sample_df = stratified_sample(summary, DATASET_ROOT, MAX_FILES_PER_GROUP, RANDOM_SEED)
    print(f"总样本数: {len(summary)}")
    print(f"参与校准样本数: {len(sample_df)}")

    all_summary_rows: List[Dict] = []
    by_type_parts: List[pd.DataFrame] = []

    param_list = list(iter_param_grid())
    print(f"参数组合数: {len(param_list)}")

    for i, params in enumerate(param_list, start=1):
        print("\n" + "-" * 78)
        print(f"[{i}/{len(param_list)}] 正在评价参数: {params}")
        row, sample_metrics = evaluate_one_param_set(sample_df, params)
        all_summary_rows.append(row)
        by_type_parts.append(summarize_by_type(sample_metrics, params))
        print(
            "结果: "
            f"artifact_recall={row['artifact_recall']:.3f}, "
            f"signal_recall={row['signal_recall']:.3f}, "
            f"candidate_ratio={row['candidate_ratio']:.4f}, "
            f"score={row['score']:.3f}"
        )

    summary_df = pd.DataFrame(all_summary_rows).sort_values("score", ascending=False)
    by_type_df = pd.concat([p for p in by_type_parts if not p.empty], ignore_index=True, sort=False)

    summary_csv = OUT_DIR / "sta_lta_calibration_summary.csv"
    by_type_csv = OUT_DIR / "sta_lta_calibration_by_type.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    by_type_df.to_csv(by_type_csv, index=False, encoding="utf-8-sig")

    best = choose_best(summary_df)
    save_best_params(best, OUT_DIR)

    report = {
        "dataset_root": str(DATASET_ROOT),
        "out_dir": str(OUT_DIR),
        "max_files_per_group": MAX_FILES_PER_GROUP,
        "channel_stride": CHANNEL_STRIDE,
        "n_total_samples": int(len(summary)),
        "n_calibration_samples": int(len(sample_df)),
        "n_param_sets": int(len(param_list)),
        "best_params": {k: float(best[k]) for k in ["sta_s", "lta_s", "thr_on", "thr_off", "min_dur_s", "merge_gap_s", "pre_pad_s", "post_pad_s"]},
        "best_metrics": {
            "artifact_recall": float(best["artifact_recall"]),
            "signal_recall": float(best["signal_recall"]),
            "candidate_ratio": float(best["candidate_ratio"]),
            "background_candidate_ratio": float(best["background_candidate_ratio"]),
            "score": float(best["score"]),
        },
    }
    (OUT_DIR / "sta_lta_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("[完成] STA/LTA 参数校准完成")
    print(f"总表: {summary_csv}")
    print(f"分组表: {by_type_csv}")
    print(f"推荐参数: {OUT_DIR / 'best_sta_lta_params.txt'}")
    print("\n推荐参数：")
    for key in ["sta_s", "lta_s", "thr_on", "thr_off", "min_dur_s", "merge_gap_s", "pre_pad_s", "post_pad_s"]:
        print(f"  {key} = {float(best[key])}")
    print("推荐参数对应指标：")
    print(f"  artifact_recall = {float(best['artifact_recall']):.3f}")
    print(f"  signal_recall   = {float(best['signal_recall']):.3f}")
    print(f"  candidate_ratio = {float(best['candidate_ratio']):.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
