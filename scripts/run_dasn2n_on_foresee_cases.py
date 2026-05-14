# -*- coding: utf-8 -*-
"""
批量生成 FORESEE 真实数据的 DAS-N2N cleaned 输出（硬编码路径版）。

使用前提：
    1. 你已经运行过：
       prepare_foresee_for_dasn2n.py

    2. 输入目录中已有：
       D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_inputs

    3. 每个输入 .npz 至少包含：
       data_input          : 原始真实数据，shape = [channel, time]
       data_input_norm     : robust 标准化后的真实数据，shape = [channel, time]
       channel_median      : 每道中位数，shape = [channel, 1]
       channel_scale       : 每道 robust scale，shape = [channel, 1]
       fs, dx_m, case_id, meta_json

输出：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_outputs

    每个输出 .npz 包含：
       dasn2n_cleaned_ct   : DAS-N2N 清理结果，shape = [channel, time]
       cleaned_ct          : 同上，兼容字段
       data_cleaned        : 同上，兼容字段
       data_denoised       : 同上，兼容字段
       dasn2n_residual_ct  : raw - cleaned，shape = [channel, time]

重要说明：
    - FORESEE case 保存为 [channel, time]。
    - 你之前的 DAS-N2N 很可能是在合成数据 [time, channel] 上使用的。
    - 因此本脚本默认 TRANSPOSE_BEFORE_DENOISE = True：
          输入模型前：[channel, time] -> [time, channel]
          模型输出后：[time, channel] -> [channel, time]
    - 如果输出效果明显异常，再把 TRANSPOSE_BEFORE_DENOISE 改为 False 试一次。
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from dasn2n import DASN2N


# =============================================================================
# 1. 硬编码路径
# =============================================================================

INPUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_inputs")
OUTPUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_outputs")


# =============================================================================
# 2. 运行参数
# =============================================================================

# 是否覆盖已有输出。
OVERWRITE = False

# 先设为 1 测试；测试没问题后改成 0 处理全部。
MAX_FILES = 0

# 是否使用 robust 标准化后的输入。
# 如果你的 DAS-N2N 训练时使用的是原始幅值输入，可改成 False。
USE_NORMALIZED_INPUT = True

# 是否在送入 DAS-N2N 前转置。
# 推荐 True，因为你的 DAS-N2N 之前大概率处理的是 [time, channel]。
TRANSPOSE_BEFORE_DENOISE = True


# =============================================================================
# 3. 基础工具
# =============================================================================

def ensure_dir(path: Path) -> None:
    """创建目录。"""
    path.mkdir(parents=True, exist_ok=True)


def case_stem_from_input(path: Path) -> str:
    """从输入文件名得到 case stem。"""
    name = path.name
    if name.endswith("_dasn2n_input.npz"):
        return name.replace("_dasn2n_input.npz", "")
    return path.stem


def denoise_one_array(model: DASN2N, data_ct: np.ndarray) -> np.ndarray:
    """
    对单个二维数组运行 DAS-N2N。

    参数：
        data_ct:
            shape = [channel, time]

    返回：
        cleaned_ct:
            shape = [channel, time]
    """
    data = np.asarray(data_ct, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"DAS-N2N 输入必须是二维数组，当前 shape={data.shape}")

    if TRANSPOSE_BEFORE_DENOISE:
        data_in = data.T
        cleaned = model.denoise_numpy(data_in)
        cleaned = np.asarray(cleaned, dtype=np.float32).T
    else:
        cleaned = model.denoise_numpy(data)
        cleaned = np.asarray(cleaned, dtype=np.float32)

    if cleaned.shape != data.shape:
        raise ValueError(
            f"DAS-N2N 输出 shape 与输入不一致: input={data.shape}, output={cleaned.shape}. "
            f"可尝试修改 TRANSPOSE_BEFORE_DENOISE。"
        )

    return cleaned.astype(np.float32)


def load_input_case(path: Path) -> dict:
    """读取一个 FORESEE DAS-N2N 输入文件。"""
    with np.load(path, allow_pickle=False) as npz:
        data_input = npz["data_input"].astype(np.float32)

        if USE_NORMALIZED_INPUT:
            if "data_input_norm" not in npz.files:
                raise KeyError(f"{path} 缺少 data_input_norm，无法使用标准化输入。")
            data_for_model = npz["data_input_norm"].astype(np.float32)
        else:
            data_for_model = data_input.copy()

        payload = {
            "data_input": data_input,
            "data_for_model": data_for_model,
            "fs": npz["fs"] if "fs" in npz.files else np.array(125.0, dtype=np.float32),
            "dx_m": npz["dx_m"] if "dx_m" in npz.files else np.array(2.0, dtype=np.float32),
            "case_id": npz["case_id"] if "case_id" in npz.files else np.array(path.stem),
            "source_case_file": npz["source_case_file"] if "source_case_file" in npz.files else np.array(""),
            "meta_json": npz["meta_json"] if "meta_json" in npz.files else np.array("{}"),
        }

        if "channel_median" in npz.files:
            payload["channel_median"] = npz["channel_median"].astype(np.float32)
        else:
            payload["channel_median"] = np.zeros((data_input.shape[0], 1), dtype=np.float32)

        if "channel_scale" in npz.files:
            payload["channel_scale"] = npz["channel_scale"].astype(np.float32)
        else:
            payload["channel_scale"] = np.ones((data_input.shape[0], 1), dtype=np.float32)

    return payload


def restore_if_needed(cleaned_model_space: np.ndarray, item: dict) -> np.ndarray:
    """
    如果使用了标准化输入，则把模型输出还原到原始幅值空间。
    """
    if USE_NORMALIZED_INPUT:
        return cleaned_model_space * item["channel_scale"] + item["channel_median"]
    return cleaned_model_space


def process_one_file(model: DASN2N, input_path: Path) -> bool:
    """处理一个 FORESEE 输入文件。"""
    case_stem = case_stem_from_input(input_path)
    output_path = OUTPUT_DIR / f"{case_stem}_dasn2n_output.npz"

    if output_path.exists() and not OVERWRITE:
        print(f"[跳过] 已存在: {output_path}")
        return False

    item = load_input_case(input_path)
    raw_ct = item["data_input"]
    data_for_model = item["data_for_model"]

    cleaned_model_space = denoise_one_array(model, data_for_model)
    cleaned_ct = restore_if_needed(cleaned_model_space, item).astype(np.float32)

    if cleaned_ct.shape != raw_ct.shape:
        raise ValueError(f"还原后 shape 不一致: raw={raw_ct.shape}, cleaned={cleaned_ct.shape}")

    residual_ct = raw_ct - cleaned_ct

    ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        dasn2n_cleaned_ct=cleaned_ct,
        cleaned_ct=cleaned_ct,
        data_cleaned=cleaned_ct,
        data_denoised=cleaned_ct,
        dasn2n_residual_ct=residual_ct.astype(np.float32),
        data_input=raw_ct.astype(np.float32),
        fs=item["fs"],
        dx_m=item["dx_m"],
        case_id=item["case_id"],
        source_input_file=np.array(str(input_path)),
        source_case_file=item["source_case_file"],
        meta_json=item["meta_json"],
    )

    print(f"[OK] {input_path} -> {output_path}")
    return True


# =============================================================================
# 4. 主流程
# =============================================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    files = sorted(INPUT_DIR.glob("*_dasn2n_input.npz"))
    if not files:
        raise FileNotFoundError(f"没有找到 DAS-N2N 输入文件: {INPUT_DIR}")

    if MAX_FILES and MAX_FILES > 0:
        files_to_run = files[:MAX_FILES]
    else:
        files_to_run = files

    print("=" * 80)
    print("FORESEE 真实数据 DAS-N2N cleaned 输出生成")
    print("=" * 80)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总文件数: {len(files)}")
    print(f"本次处理: {len(files_to_run)}")
    print(f"使用标准化输入: {USE_NORMALIZED_INPUT}")
    print(f"转置输入: {TRANSPOSE_BEFORE_DENOISE}")
    print("=" * 80)

    model = DASN2N()
    model.load_weights()

    n_ok = 0
    n_skip = 0

    for path in files_to_run:
        try:
            ok = process_one_file(model, path)
            if ok:
                n_ok += 1
            else:
                n_skip += 1
        except Exception as exc:
            n_skip += 1
            print(f"[错误] 处理失败: {path}")
            print(f"       {type(exc).__name__}: {exc}")

    report = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "use_normalized_input": USE_NORMALIZED_INPUT,
        "transpose_before_denoise": TRANSPOSE_BEFORE_DENOISE,
        "overwrite": OVERWRITE,
        "max_files": MAX_FILES,
        "n_total_available": int(len(files)),
        "n_attempted": int(len(files_to_run)),
        "n_written": int(n_ok),
        "n_skipped_or_failed": int(n_skip),
    }

    report_path = OUTPUT_DIR / "foresee_dasn2n_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("[完成]")
    print(f"成功写出: {n_ok}")
    print(f"跳过/失败: {n_skip}")
    print(f"报告文件: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
