# -*- coding: utf-8 -*-
"""
将 FORESEE 真实 DAS case 导出为 DAS-N2N 推理输入（硬编码路径版）。

用途：
    你已经通过 extract_foresee_real_cases.py 生成了真实数据小片段：
        D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases\real_case_*.npz

    本脚本把这些 case 重新整理成 DAS-N2N 推理更容易读取的输入文件。

输入字段：
    data_proc:
        shape = [channel, time]，已经每道去均值。

输出字段：
    data_input:
        原始输入，shape = [channel, time]。
    data_input_norm:
        每道 robust 标准化后的输入，shape = [channel, time]。
        如果你的 DAS-N2N 模型训练时需要标准化输入，可以用这个字段。
    channel_median:
        每道中位数。
    channel_scale:
        每道 robust scale，用于将标准化输出还原到原始幅值。
    fs, dx_m, case_id, meta_json

输出目录：
    D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_inputs

注意：
    本脚本只准备输入，不运行 DAS-N2N。
    你需要用自己的 DAS-N2N 推理代码读取这些输入，并把 cleaned 输出保存到：
        D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_outputs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


# =============================================================================
# 1. 硬编码路径
# =============================================================================

CASE_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases")
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_DASN2N_inputs")


# =============================================================================
# 2. 标准化函数
# =============================================================================

def robust_channel_normalize(data_ct: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对每个通道做 robust 标准化。

    data_ct:
        shape = [channel, time]

    返回：
        data_norm = (data - median) / scale
        scale = 1.4826 * MAD

    这样做比均值/标准差更不容易被车辆强事件或局部尖峰影响。
    """
    data = data_ct.astype(np.float32, copy=False)
    median = np.median(data, axis=1, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(data - median), axis=1, keepdims=True).astype(np.float32)
    scale = 1.4826 * mad
    scale[scale < 1e-6] = 1.0
    norm = (data - median) / scale
    return {
        "data_input_norm": norm.astype(np.float32),
        "channel_median": median.astype(np.float32),
        "channel_scale": scale.astype(np.float32),
    }


def load_meta(npz: np.lib.npyio.NpzFile) -> str:
    """读取 meta_json 字段，没有则返回空 JSON。"""
    if "meta_json" in npz.files:
        return str(npz["meta_json"])
    return json.dumps({}, ensure_ascii=False)


def process_one_case(path: Path) -> Path:
    """处理单个真实 case。"""
    with np.load(path, allow_pickle=False) as npz:
        data_ct = npz["data_proc"].astype(np.float32)
        fs = float(npz["fs"]) if "fs" in npz.files else 125.0
        dx_m = float(npz["dx_m"]) if "dx_m" in npz.files else 2.0
        case_id = str(npz["case_id"]) if "case_id" in npz.files else path.stem
        meta_json = load_meta(npz)

    norm_payload = robust_channel_normalize(data_ct)

    out_path = OUT_DIR / f"{path.stem}_dasn2n_input.npz"
    np.savez_compressed(
        out_path,
        data_input=data_ct.astype(np.float32),
        fs=np.float32(fs),
        dx_m=np.float32(dx_m),
        case_id=np.array(case_id),
        source_case_file=np.array(str(path)),
        meta_json=np.array(meta_json),
        **norm_payload,
    )
    print(f"[OK] {out_path}")
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CASE_DIR.glob("real_case_*.npz"))
    if not files:
        raise FileNotFoundError(f"没有找到真实 case 文件: {CASE_DIR}")

    print("=" * 80)
    print("准备 FORESEE DAS-N2N 推理输入")
    print("=" * 80)
    print(f"输入目录: {CASE_DIR}")
    print(f"输出目录: {OUT_DIR}")

    for path in files:
        process_one_case(path)

    print("=" * 80)
    print("[完成] DAS-N2N 输入已导出。")
    print("下一步：用你的 DAS-N2N 推理代码读取 *_dasn2n_input.npz 并保存 cleaned 输出。")
    print("=" * 80)


if __name__ == "__main__":
    main()
