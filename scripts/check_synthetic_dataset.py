# -*- coding: utf-8 -*-
"""
合成 DAS 数据集质检脚本。

功能：
1. 检查 dataset_summary.csv 是否存在、行数是否符合 generator_config.json。
2. 检查 .npz 样本字段是否完整。
3. 统计目标 SNR 与实际 SNR 分布。
4. 统计 artifact_mask 与 signal_mask 占比。
5. 抽查 data_noisy、data_clean、artifact_only、signal_only 的基本数值统计。
6. 输出 check_report_summary.csv、check_report_samples.csv 和 check_report.json。

用法：
    python check_synthetic_dataset.py --root synthetic_dataset_paper_minimum

如果不提供 --root，脚本会优先在当前目录下查找：
    synthetic_dataset_paper_minimum
    synthetic_dataset_pilot
    synthetic_dataset_paper_extended
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT_CANDIDATES = (
    "synthetic_dataset_paper_minimum",
    "synthetic_dataset_pilot",
    "synthetic_dataset_paper_extended",
)

REQUIRED_COMMON_FIELDS = {
    "data_noisy",
    "data_clean",
    "artifact_mask",
    "signal_mask",
    "fs",
    "dx_m",
    "snr_target_db",
    "snr_actual_db",
    "meta_json",
}

REQUIRED_BENCHMARK_FIELDS = REQUIRED_COMMON_FIELDS | {
    "artifact_only",
    "signal_only",
    "artifact_type",
}

REQUIRED_SIGNAL_RESCUE_FIELDS = REQUIRED_COMMON_FIELDS | {
    "background",
    "signal_only",
    "v_app_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 MCS-AND 合成 DAS 数据集是否完整可靠。")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="合成数据集根目录。也可以不用位置参数，改用 --root 指定。",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="合成数据集根目录，例如 synthetic_dataset_paper_minimum。若省略则自动查找。",
    )
    parser.add_argument(
        "--wdir",
        type=str,
        default=None,
        help="兼容 Spyder 的 %runfile --wdir 参数；本脚本只记录该值，不把它当数据集目录。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="最多抽查多少个 .npz 样本。设为 0 表示检查全部样本。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="启用严格模式：只要发现缺字段、坏文件或重建误差异常，就以非零状态退出。",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[提示] 已忽略无法识别的运行参数: {unknown}")
    if args.root is None and args.dataset_root is not None:
        args.root = args.dataset_root
    return args


def find_dataset_root(root_arg: Optional[str]) -> Path:
    if root_arg:
        root = Path(root_arg).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"指定的数据集目录不存在: {root}")
        return root

    cwd = Path.cwd()
    for name in DEFAULT_ROOT_CANDIDATES:
        candidate = cwd / name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "未找到合成数据集目录。请使用 --root 指定，例如："
        "python check_synthetic_dataset.py --root synthetic_dataset_paper_minimum"
    )


def load_config(root: Path) -> Dict:
    config_path = root / "generator_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_summary(root: Path) -> pd.DataFrame:
    summary_path = root / "dataset_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 dataset_summary.csv: {summary_path}")
    return pd.read_csv(summary_path, encoding="utf-8-sig")


def expected_counts_from_config(config: Dict) -> Dict[str, Optional[int]]:
    if not config or "preset" not in config:
        return {"benchmark": None, "signal_rescue": None, "total": None}

    preset = config["preset"]
    artifact_types = config.get("artifact_types", [])

    snr_list = preset.get("snr_list_db", [])
    signal_snr = preset.get("signal_snr_db", [])
    vapps = preset.get("signal_vapp_list", [])
    n_samples_per_snr = int(preset.get("n_samples_per_snr", 0))
    n_signal_samples = int(preset.get("n_signal_samples", 0))

    n_benchmark = len(artifact_types) * len(snr_list) * n_samples_per_snr
    n_signal = len(vapps) * len(signal_snr) * n_signal_samples
    return {
        "benchmark": n_benchmark,
        "signal_rescue": n_signal,
        "total": n_benchmark + n_signal,
    }


def resolve_filepaths(root: Path, summary: pd.DataFrame) -> List[Path]:
    if "filepath" not in summary.columns:
        return sorted(root.rglob("*.npz"))

    paths: List[Path] = []
    for raw in summary["filepath"].dropna().astype(str):
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        paths.append(p)
    return paths


def choose_sample_paths(paths: List[Path], max_samples: int, seed: int) -> List[Path]:
    if max_samples == 0 or len(paths) <= max_samples:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=max_samples, replace=False)
    return [paths[int(i)] for i in sorted(idx)]


def safe_float(x) -> Optional[float]:
    try:
        value = float(np.asarray(x))
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return None


def array_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    arr64 = np.asarray(arr, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr64)),
        f"{prefix}_std": float(np.std(arr64)),
        f"{prefix}_rms": float(np.sqrt(np.mean(arr64**2) + 1e-30)),
        f"{prefix}_max_abs": float(np.max(np.abs(arr64))) if arr64.size else np.nan,
    }


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sqrt(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))
    den = np.sqrt(np.mean(np.asarray(b, dtype=np.float64) ** 2) + 1e-30)
    return float(num / (den + 1e-30))


def infer_category(path: Path, npz_keys: Iterable[str]) -> str:
    parts = set(path.parts)
    keys = set(npz_keys)
    if "benchmark" in parts or "artifact_only" in keys:
        return "benchmark"
    if "signal_rescue" in parts or "background" in keys:
        return "signal_rescue"
    return "unknown"


def inspect_one_npz(path: Path) -> Dict:
    row: Dict = {
        "filepath": str(path),
        "exists": path.exists(),
        "ok": False,
        "category": "unknown",
        "missing_fields": "",
        "error": "",
    }

    if not path.exists():
        row["error"] = "文件不存在"
        return row

    try:
        with np.load(path, allow_pickle=False) as npz:
            keys = set(npz.files)
            category = infer_category(path, keys)
            row["category"] = category

            required = REQUIRED_BENCHMARK_FIELDS if category == "benchmark" else REQUIRED_SIGNAL_RESCUE_FIELDS
            missing = sorted(required - keys)
            row["missing_fields"] = ";".join(missing)

            data_noisy = npz["data_noisy"] if "data_noisy" in keys else None
            data_clean = npz["data_clean"] if "data_clean" in keys else None

            if data_noisy is not None:
                row["shape"] = "x".join(map(str, data_noisy.shape))
                row["n_time"] = int(data_noisy.shape[0])
                row["n_channels"] = int(data_noisy.shape[1]) if data_noisy.ndim == 2 else np.nan
                row.update(array_stats(data_noisy, "data_noisy"))

            if data_clean is not None:
                row.update(array_stats(data_clean, "data_clean"))

            if "artifact_mask" in keys:
                artifact_mask = npz["artifact_mask"].astype(bool)
                row["artifact_mask_ratio"] = float(np.mean(artifact_mask))
                row["artifact_mask_count"] = int(np.sum(artifact_mask))

            if "signal_mask" in keys:
                signal_mask = npz["signal_mask"].astype(bool)
                row["signal_mask_ratio"] = float(np.mean(signal_mask))
                row["signal_mask_count"] = int(np.sum(signal_mask))

            if "snr_target_db" in keys:
                row["snr_target_db_npz"] = safe_float(npz["snr_target_db"])
            if "snr_actual_db" in keys:
                row["snr_actual_db_npz"] = safe_float(npz["snr_actual_db"])

            if category == "benchmark" and {"data_noisy", "data_clean", "artifact_only"} <= keys:
                artifact_only = npz["artifact_only"]
                reconstructed = data_clean + artifact_only
                row["reconstruction_rel_error"] = relative_error(data_noisy, reconstructed)
                row.update(array_stats(artifact_only, "artifact_only"))

            if "signal_only" in keys:
                signal_only = npz["signal_only"]
                row.update(array_stats(signal_only, "signal_only"))

            row["ok"] = len(missing) == 0 and not row["error"]
            return row

    except Exception as exc:
        row["error"] = repr(exc)
        return row


def summarize_counts(summary: pd.DataFrame, config: Dict) -> pd.DataFrame:
    expected = expected_counts_from_config(config)
    rows: List[Dict] = []

    if "category" in summary.columns:
        actual_by_category = summary["category"].value_counts().to_dict()
    else:
        actual_by_category = {}

    for category in ("benchmark", "signal_rescue"):
        actual = int(actual_by_category.get(category, 0))
        exp = expected.get(category)
        rows.append(
            {
                "section": "样本数量",
                "item": category,
                "actual": actual,
                "expected": exp,
                "ok": None if exp is None else actual == exp,
            }
        )

    rows.append(
        {
            "section": "样本数量",
            "item": "total",
            "actual": int(len(summary)),
            "expected": expected.get("total"),
            "ok": None if expected.get("total") is None else int(len(summary)) == expected.get("total"),
        }
    )
    return pd.DataFrame(rows)


def summarize_snr(summary: pd.DataFrame) -> pd.DataFrame:
    needed = {"category", "snr_target_db", "snr_actual_db"}
    if not needed <= set(summary.columns):
        return pd.DataFrame()

    grouped = (
        summary.groupby(["category", "snr_target_db"], dropna=False)["snr_actual_db"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.insert(0, "section", "SNR分布")
    grouped.rename(columns={"category": "item", "snr_target_db": "target_snr_db"}, inplace=True)
    grouped["ok"] = True
    return grouped


def summarize_masks(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    if sample_df.empty:
        return pd.DataFrame()

    for category, sub in sample_df.groupby("category", dropna=False):
        for col in ("artifact_mask_ratio", "signal_mask_ratio"):
            if col not in sub.columns:
                continue
            values = pd.to_numeric(sub[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "section": "mask占比",
                    "item": f"{category}:{col}",
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ok": bool((values >= 0).all() and (values <= 1).all()),
                }
            )
    return pd.DataFrame(rows)


def summarize_field_checks(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    missing_count = int((sample_df["missing_fields"].fillna("") != "").sum())
    error_count = int((sample_df["error"].fillna("") != "").sum())
    ok_count = int(sample_df["ok"].sum())

    rows = [
        {
            "section": "字段完整性",
            "item": "checked_samples",
            "actual": int(len(sample_df)),
            "expected": int(len(sample_df)),
            "ok": True,
        },
        {
            "section": "字段完整性",
            "item": "ok_samples",
            "actual": ok_count,
            "expected": int(len(sample_df)),
            "ok": ok_count == int(len(sample_df)),
        },
        {
            "section": "字段完整性",
            "item": "missing_field_samples",
            "actual": missing_count,
            "expected": 0,
            "ok": missing_count == 0,
        },
        {
            "section": "字段完整性",
            "item": "error_samples",
            "actual": error_count,
            "expected": 0,
            "ok": error_count == 0,
        },
    ]

    if "reconstruction_rel_error" in sample_df.columns:
        errs = pd.to_numeric(sample_df["reconstruction_rel_error"], errors="coerce").dropna()
        if not errs.empty:
            rows.append(
                {
                    "section": "重建一致性",
                    "item": "data_noisy_vs_data_clean_plus_artifact",
                    "actual": float(errs.max()),
                    "expected": "< 1e-5",
                    "ok": bool((errs < 1e-5).all()),
                }
            )

    return pd.DataFrame(rows)


def build_report(root: Path, max_samples: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    config = load_config(root)
    summary = load_summary(root)
    paths = resolve_filepaths(root, summary)
    sample_paths = choose_sample_paths(paths, max_samples=max_samples, seed=seed)

    print(f"[INFO] 数据集目录: {root}")
    print(f"[INFO] dataset_summary.csv 行数: {len(summary)}")
    print(f"[INFO] .npz 文件路径数: {len(paths)}")
    print(f"[INFO] 本次抽查样本数: {len(sample_paths)}")

    sample_rows = []
    for idx, path in enumerate(sample_paths, start=1):
        if idx % 50 == 0 or idx == len(sample_paths):
            print(f"[INFO] 正在检查样本 {idx}/{len(sample_paths)}")
        sample_rows.append(inspect_one_npz(path))

    sample_df = pd.DataFrame(sample_rows)

    parts = [
        summarize_counts(summary, config),
        summarize_snr(summary),
        summarize_field_checks(sample_df),
        summarize_masks(sample_df),
    ]
    report_df = pd.concat([p for p in parts if not p.empty], ignore_index=True, sort=False)

    report_json = {
        "root": str(root),
        "summary_rows": int(len(summary)),
        "npz_paths": int(len(paths)),
        "checked_samples": int(len(sample_df)),
        "bad_samples": int((sample_df.get("ok", pd.Series(dtype=bool)) == False).sum()),
        "run_mode": config.get("run_mode"),
        "expected_counts": expected_counts_from_config(config),
    }
    return report_df, sample_df, report_json


def write_outputs(root: Path, report_df: pd.DataFrame, sample_df: pd.DataFrame, report_json: Dict) -> None:
    report_csv = root / "check_report_summary.csv"
    samples_csv = root / "check_report_samples.csv"
    report_json_path = root / "check_report.json"

    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")
    sample_df.to_csv(samples_csv, index=False, encoding="utf-8-sig")
    report_json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[完成] 质检报告已输出：")
    print(f"  汇总报告: {report_csv}")
    print(f"  样本报告: {samples_csv}")
    print(f"  JSON报告: {report_json_path}")


def print_brief(report_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    print("\n========== 质检摘要 ==========")
    if report_df.empty:
        print("未生成汇总统计。")
    else:
        display_cols = [c for c in ["section", "item", "actual", "expected", "count", "mean", "std", "min", "max", "ok"] if c in report_df.columns]
        print(report_df[display_cols].to_string(index=False))

    if not sample_df.empty:
        bad = sample_df[(sample_df["ok"] == False) | (sample_df["error"].fillna("") != "")]
        if bad.empty:
            print("\n[OK] 抽查样本字段完整，未发现坏文件。")
        else:
            print("\n[警告] 存在异常样本，前 10 条如下：")
            cols = ["filepath", "category", "missing_fields", "error"]
            print(bad[cols].head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    root = find_dataset_root(args.root)
    report_df, sample_df, report_json = build_report(root, max_samples=args.max_samples, seed=args.seed)
    write_outputs(root, report_df, sample_df, report_json)
    print_brief(report_df, sample_df)

    if args.strict:
        ok_col = report_df["ok"].dropna() if "ok" in report_df.columns else pd.Series(dtype=bool)
        bad_samples = int((sample_df.get("ok", pd.Series(dtype=bool)) == False).sum())
        if (not ok_col.empty and not bool(ok_col.all())) or bad_samples > 0:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
