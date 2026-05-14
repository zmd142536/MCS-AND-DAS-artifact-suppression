# -*- coding: utf-8 -*-
"""
MCS-AND 敏感性与稳健性分析脚本（硬编码路径版，中文详细注释）。

用途
----
本脚本服务于论文“敏感性与稳健性分析”小节，生成三类结果：

1. 单参数敏感性扫描表：
   统计 final_noise_count 随 alpha、邻域半宽 K、semblance 阈值、
   V_APP_MIN 和 V_APP_MAX 的变化。

2. 参数敏感性组合图：
   绘制 final_noise_count 的 5 个参数响应曲线。

3. alpha-outlier fraction 校准图：
   用 MCD 模型得到的 Mahalanobis d^2 分布，比较经验异常比例和理论 y=alpha。

重要说明
--------
真实 FORESEE 数据没有人工伪迹真值，因此这里的 final_noise_count 不是精度指标，
而是“最终没有被 tau-p 相干性救回、会进入最终伪迹 mask 的候选事件数量”。
它适合说明参数作用方向是否清晰、方法是否可调。

运行前提
--------
请先运行：
    1. extract_foresee_real_cases.py
    2. run_real_foresee_mcsand.py

脚本会复用 run_real_foresee_mcsand.py 中的候选切片、特征提取和 tau-p 扫描函数。

输出目录
--------
    D:\项目实验\积石山实验\DAS相关\去噪论文\敏感性与稳健性分析

输出文件
--------
    tables\real_parameter_sensitivity_scan.csv
    tables\alpha_outlier_fraction_table.csv
    figures\FigureS_sensitivity_final_noise_count.png/pdf
    figures\FigureS_alpha_outlier_fraction.png/pdf
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import PowerTransformer


# =============================================================================
# 1. 硬编码路径
# =============================================================================

# MCS-AND 真实数据处理脚本。这里动态导入它，复用其中的函数，避免复制算法代码。
REAL_MCSAND_SCRIPT = Path(
    r"D:\项目实验\积石山实验\DAS相关\Apython\S2\去噪论文用代码\xinquzaodaima\run_real_foresee_mcsand.py"
)

# 已生成的真实数据 case 和真实 MCS-AND 结果目录。
REAL_CASE_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_cases")
REAL_RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\真实数据\FORESEE_MCSAND_results")

# 合成数据 MCS-AND 结果目录。这里主要用于 alpha-outlier fraction 辅助校准。
SYN_RESULT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\MCSAND_benchmark2")

# 本脚本输出目录。
OUT_DIR = Path(r"D:\项目实验\积石山实验\DAS相关\去噪论文\敏感性与稳健性分析")
TABLE_DIR = OUT_DIR / "tables2"
FIG_DIR = OUT_DIR / "figures2"


# =============================================================================
# 2. 扫描网格
# =============================================================================

# alpha 是 Mahalanobis d^2 的右尾概率。alpha 越大，阈值越低，MCD 候选越多。
ALPHA_GRID = [0.025, 0.05, 0.075, 0.10]

# tau-p 相干性邻域半宽。K=5 表示中心通道左右各取 5 道。
K_GRID = [3, 5, 7, 10]

# tau-p semblance 阈值。阈值越大，救回越严格，final_noise_count 通常越大。
SEMBLANCE_GRID = [0.15, 0.30, 0.45, 0.60]

# 合成实验常用的表观速度下限扫描网格。
# 注意：FORESEE 车辆数据中可能存在更低表观速度。若专门分析 FORESEE，
# 可把 REALISTIC_FORESEE_SPEED_GRID 设为 True，脚本会改用更低速范围。
V_APP_MIN_GRID_SYNTHETIC_STYLE = [50.0, 100.0, 150.0, 200.0]
V_APP_MAX_GRID_SYNTHETIC_STYLE = [3000.0, 5000.0, 7000.0, 10000.0]

# FORESEE 城市场景更合适的低速网格。车辆沿光纤投影速度可能只有几到几十 m/s。
V_APP_MIN_GRID_FORESEE_STYLE = [2.5, 5.0, 10.0, 20.0]
V_APP_MAX_GRID_FORESEE_STYLE = [1000.0, 2000.0, 3000.0, 5000.0]

# 如果正文需要完全对应你写的“50-200 m/s”和“3000-10000 m/s”，设为 False。
# 如果要让真实 FORESEE 车辆案例的敏感性更物理合理，设为 True。
REALISTIC_FORESEE_SPEED_GRID = True


# =============================================================================
# 3. 基准运行时参数
# =============================================================================

# 为了和你现有 FORESEE 真实数据实验一致，默认读取 run_real_foresee_mcsand.py
# 中的参数；如果导入失败，则使用这里的备用值。
BASE_ALPHA = 0.05
BASE_RUNTIME_PARAMS = dict(
    COHERENCE_K=5,
    SEMBLANCE_THR=0.22,
    V_APP_MIN=5.0,
    V_APP_MAX=2000.0,
    N_V_SCAN=60,
    LOWRANK_RANK=2,
    LOWRANK_WIN_PAD_S=0.30,
    LOWRANK_NEIGHBOR_K=5,
    LOWRANK_ITER=3,
)

# alpha 扫描是否使用经验分位数下限。
# False：使用 chi-square 阈值，适合展示 alpha 的理论方向；
# True ：使用 max(chi-square, empirical quantile)，更保守，但真实数据中 alpha 曲线可能变平。
USE_EMPIRICAL_THRESHOLD_FLOOR = False
EMPIRICAL_D2_QUANTILE = 0.90


# =============================================================================
# 4. 绘图规范：A4 友好、Times New Roman、不加粗
# =============================================================================

MM_TO_INCH = 1.0 / 25.4
A4_WIDTH_IN = 210.0 * MM_TO_INCH

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "font.weight": "normal",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 5. 基础工具函数
# =============================================================================

def ensure_dirs() -> None:
    """创建输出目录。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_real_mcsand_module():
    """动态导入 run_real_foresee_mcsand.py。"""
    if not REAL_MCSAND_SCRIPT.exists():
        raise FileNotFoundError(f"找不到真实数据 MCS-AND 脚本: {REAL_MCSAND_SCRIPT}")

    spec = importlib.util.spec_from_file_location("real_mcsand_module", REAL_MCSAND_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入脚本: {REAL_MCSAND_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_feature_table(feature_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """保留特征完整且有限的事件。"""
    out = feature_df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    ok = np.isfinite(out[feature_cols].to_numpy(dtype=float)).all(axis=1)
    return out.loc[ok].reset_index(drop=True)


def compute_mcd_d2(feature_df: pd.DataFrame, feature_cols: List[str], random_seed: int = 42) -> pd.Series:
    """
    对候选事件特征拟合 Yeo-Johnson + MCD，并返回每个事件的 Mahalanobis d^2。

    这里重新拟合一次，是为了让 alpha-outlier fraction 曲线与当前扫描数据严格一致。
    """
    x = feature_df[feature_cols].to_numpy(dtype=np.float64)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    x_trans = pt.fit_transform(x)

    mcd = MinCovDet(random_state=random_seed, support_fraction=None)
    mcd.fit(x_trans)
    d2 = mcd.mahalanobis(x_trans)
    return pd.Series(d2, index=feature_df.index, name="d2_mahalanobis")


def d2_threshold(alpha: float, n_features: int, d2: pd.Series) -> float:
    """根据 alpha 计算 Mahalanobis d^2 阈值。"""
    chi2_thr = float(stats.chi2.ppf(1.0 - float(alpha), df=int(n_features)))
    if USE_EMPIRICAL_THRESHOLD_FLOOR:
        empirical_thr = float(np.quantile(d2.to_numpy(dtype=float), EMPIRICAL_D2_QUANTILE))
        return max(chi2_thr, empirical_thr)
    return chi2_thr


def case_label(case_id: str) -> str:
    """把真实 case id 简化为图例标签。"""
    return (
        str(case_id)
        .replace("real_case_", "")
        .replace("_noon_late", " noon late")
        .replace("_noon", " noon")
        .replace("_evening_late", " evening late")
        .replace("_evening", " evening")
    )


def load_or_build_real_feature_table(real_module) -> Tuple[pd.DataFrame, List[str]]:
    """读取已有真实候选特征表；如果不存在，则重新收集。"""
    feature_csv = REAL_RESULT_DIR / "real_candidate_feature_library.csv"
    report_json = REAL_RESULT_DIR / "real_mcsand_report.json"

    if feature_csv.exists():
        feature_df = pd.read_csv(feature_csv, encoding="utf-8-sig")
    else:
        case_files = sorted(REAL_CASE_DIR.glob("real_case_*.npz"))
        if not case_files:
            raise FileNotFoundError(f"没有找到真实 case npz: {REAL_CASE_DIR}")
        feature_df = real_module.collect_feature_library(case_files)
        feature_csv.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")

    if report_json.exists():
        report = json.loads(report_json.read_text(encoding="utf-8"))
        feature_cols = list(report.get("mcd_feature_cols", real_module.MCD_FEATURE_COLS))
    else:
        feature_cols = list(real_module.MCD_FEATURE_COLS)

    feature_df = finite_feature_table(feature_df, feature_cols)
    return feature_df, feature_cols


def load_cases_for_feature_table(real_module, feature_df: pd.DataFrame) -> Dict[str, Dict]:
    """把扫描涉及的真实 case 数据读入内存，避免反复读取 npz。"""
    cases: Dict[str, Dict] = {}
    for case_id, sub in feature_df.groupby("case_id"):
        source_file = Path(str(sub["source_file"].iloc[0]))
        cases[str(case_id)] = real_module.load_case(source_file)
    return cases


# =============================================================================
# 6. 真实数据敏感性扫描
# =============================================================================

def build_scan_plan(base_runtime: Dict) -> List[Dict]:
    """构建单参数扫描计划。"""
    vmin_grid = V_APP_MIN_GRID_FORESEE_STYLE if REALISTIC_FORESEE_SPEED_GRID else V_APP_MIN_GRID_SYNTHETIC_STYLE
    vmax_grid = V_APP_MAX_GRID_FORESEE_STYLE if REALISTIC_FORESEE_SPEED_GRID else V_APP_MAX_GRID_SYNTHETIC_STYLE

    plan: List[Dict] = []

    for value in ALPHA_GRID:
        params = dict(base_runtime)
        plan.append(dict(group="alpha", value=float(value), alpha=float(value), runtime_params=params))

    for value in K_GRID:
        params = dict(base_runtime)
        params["COHERENCE_K"] = int(value)
        plan.append(dict(group="K", value=float(value), alpha=BASE_ALPHA, runtime_params=params))

    for value in SEMBLANCE_GRID:
        params = dict(base_runtime)
        params["SEMBLANCE_THR"] = float(value)
        plan.append(dict(group="semblance_thr", value=float(value), alpha=BASE_ALPHA, runtime_params=params))

    for value in vmin_grid:
        params = dict(base_runtime)
        params["V_APP_MIN"] = float(value)
        plan.append(dict(group="V_APP_MIN", value=float(value), alpha=BASE_ALPHA, runtime_params=params))

    for value in vmax_grid:
        params = dict(base_runtime)
        params["V_APP_MAX"] = float(value)
        plan.append(dict(group="V_APP_MAX", value=float(value), alpha=BASE_ALPHA, runtime_params=params))

    return plan


def scan_one_setting(
    real_module,
    feature_df: pd.DataFrame,
    d2: pd.Series,
    feature_cols: List[str],
    cases: Dict[str, Dict],
    setting: Dict,
    semblance_cache: Dict[Tuple, Tuple[bool, float, float, float]],
) -> List[Dict]:
    """执行一个参数设置下的 final_noise_count 统计。"""
    alpha = float(setting["alpha"])
    runtime_params = dict(setting["runtime_params"])
    thr = d2_threshold(alpha, len(feature_cols), d2)

    # 用当前 alpha 决定 MCD 候选。
    candidate_df = feature_df.loc[d2.to_numpy(dtype=float) > thr].copy()
    candidate_df["d2_mahalanobis"] = d2.loc[candidate_df.index].to_numpy(dtype=float)

    # 修改导入模块中的运行时参数，使 coherence_rescue 使用当前 K/速度/阈值。
    old_runtime = dict(real_module.RUNTIME_PARAMS)
    real_module.RUNTIME_PARAMS.update(runtime_params)

    rows: List[Dict] = []
    t0 = time.perf_counter()

    try:
        for case_id, sub in candidate_df.groupby("case_id"):
            case = cases[str(case_id)]
            data_tc = case["data_tc"]
            fs = float(case["fs"])
            dx_m = float(case["dx_m"])

            n_candidate = 0
            n_rescued = 0
            n_final_noise = 0
            sem_values: List[float] = []

            for row_index, ev_row in sub.iterrows():
                event = dict(
                    channel=int(ev_row["channel"]),
                    w0=int(ev_row["w0"]),
                    w1=int(ev_row["w1"]),
                )

                cache_key = (
                    str(case_id),
                    int(row_index),
                    int(runtime_params["COHERENCE_K"]),
                    float(runtime_params["V_APP_MIN"]),
                    float(runtime_params["V_APP_MAX"]),
                    int(runtime_params["N_V_SCAN"]),
                    float(runtime_params["SEMBLANCE_THR"]),
                )

                if cache_key in semblance_cache:
                    rescued, sem, v_app, sign = semblance_cache[cache_key]
                else:
                    rescued, sem, v_app, sign = real_module.coherence_rescue(data_tc, event, fs, dx_m)
                    semblance_cache[cache_key] = (rescued, sem, v_app, sign)

                n_candidate += 1
                n_rescued += int(rescued)
                n_final_noise += int(not rescued)
                sem_values.append(float(sem))

            rows.append(dict(
                parameter_group=setting["group"],
                parameter_value=setting["value"],
                case_id=str(case_id),
                case_label=case_label(str(case_id)),
                alpha=alpha,
                d2_threshold=thr,
                coherence_k=int(runtime_params["COHERENCE_K"]),
                semblance_thr=float(runtime_params["SEMBLANCE_THR"]),
                v_app_min=float(runtime_params["V_APP_MIN"]),
                v_app_max=float(runtime_params["V_APP_MAX"]),
                n_candidate=int(n_candidate),
                n_rescued=int(n_rescued),
                final_noise_count=int(n_final_noise),
                rescued_fraction=float(n_rescued / max(n_candidate, 1)),
                mean_semblance=float(np.mean(sem_values)) if sem_values else np.nan,
                elapsed_s=np.nan,
            ))
    finally:
        real_module.RUNTIME_PARAMS.clear()
        real_module.RUNTIME_PARAMS.update(old_runtime)

    elapsed = time.perf_counter() - t0
    for row in rows:
        row["elapsed_s"] = float(elapsed)
    return rows


def run_real_sensitivity_scan() -> pd.DataFrame:
    """执行真实数据参数敏感性扫描并保存 CSV。"""
    real_module = load_real_mcsand_module()

    base_runtime = dict(getattr(real_module, "RUNTIME_PARAMS", BASE_RUNTIME_PARAMS))
    global BASE_ALPHA
    BASE_ALPHA = float(getattr(real_module, "MAHAL_ALPHA", BASE_ALPHA))

    feature_df, feature_cols = load_or_build_real_feature_table(real_module)
    cases = load_cases_for_feature_table(real_module, feature_df)
    d2 = compute_mcd_d2(feature_df, feature_cols, random_seed=int(getattr(real_module, "RANDOM_SEED", 42)))

    plan = build_scan_plan(base_runtime)
    semblance_cache: Dict[Tuple, Tuple[bool, float, float, float]] = {}

    all_rows: List[Dict] = []
    print("=" * 80)
    print("MCS-AND 真实数据单参数敏感性扫描")
    print("=" * 80)
    print(f"候选事件数: {len(feature_df)}")
    print(f"特征列: {feature_cols}")
    print(f"速度网格: {'FORESEE 低速网格' if REALISTIC_FORESEE_SPEED_GRID else '合成数据风格网格'}")
    print(f"经验阈值下限: {USE_EMPIRICAL_THRESHOLD_FLOOR}")
    print("=" * 80)

    for i, setting in enumerate(plan, start=1):
        print(f"[{i:02d}/{len(plan):02d}] {setting['group']} = {setting['value']}")
        all_rows.extend(
            scan_one_setting(
                real_module=real_module,
                feature_df=feature_df,
                d2=d2,
                feature_cols=feature_cols,
                cases=cases,
                setting=setting,
                semblance_cache=semblance_cache,
            )
        )

    scan_df = pd.DataFrame(all_rows)
    out_csv = TABLE_DIR / "real_parameter_sensitivity_scan.csv"
    scan_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存敏感性扫描表: {out_csv}")
    return scan_df


# =============================================================================
# 7. alpha-outlier fraction 曲线
# =============================================================================

def build_alpha_outlier_table(real_scan_feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    构建 alpha-outlier fraction 表。

    真实数据：使用 real_candidate_feature_library.csv 重新拟合 MCD。
    合成数据：如果 synthetic_feature_library.csv 存在，也重新拟合 MCD。
    """
    rows: List[Dict] = []

    datasets = []
    real_feature_csv = REAL_RESULT_DIR / "real_candidate_feature_library.csv"
    real_report_json = REAL_RESULT_DIR / "real_mcsand_report.json"
    if real_feature_csv.exists() and real_report_json.exists():
        real_report = json.loads(real_report_json.read_text(encoding="utf-8"))
        datasets.append((
            "FORESEE field candidates",
            real_feature_csv,
            list(real_report.get("mcd_feature_cols", [])),
        ))

    syn_feature_csv = SYN_RESULT_DIR / "synthetic_feature_library.csv"
    syn_report_json = SYN_RESULT_DIR / "synthetic_mcd_report.json"
    if syn_feature_csv.exists() and syn_report_json.exists():
        syn_report = json.loads(syn_report_json.read_text(encoding="utf-8"))
        datasets.append((
            "Synthetic benchmark candidates",
            syn_feature_csv,
            list(syn_report.get("feature_cols", [])),
        ))

    for dataset_name, feature_csv, feature_cols in datasets:
        if not feature_cols:
            continue
        print(f"[alpha 校准] {dataset_name}")
        df = pd.read_csv(feature_csv, encoding="utf-8-sig")
        df = finite_feature_table(df, feature_cols)

        # 合成特征表可能较大。为避免普通电脑上 MCD 拟合过慢，校准曲线默认抽样。
        # 抽样只用于诊断曲线，不影响主实验结果。
        if len(df) > 50000:
            df_fit = df.sample(n=50000, random_state=42).reset_index(drop=True)
        else:
            df_fit = df.reset_index(drop=True)

        d2 = compute_mcd_d2(df_fit, feature_cols, random_seed=42)
        n_features = len(feature_cols)

        for alpha in ALPHA_GRID:
            thr = float(stats.chi2.ppf(1.0 - float(alpha), df=n_features))
            frac = float(np.mean(d2.to_numpy(dtype=float) > thr))
            rows.append(dict(
                dataset=dataset_name,
                alpha=float(alpha),
                d2_threshold=thr,
                empirical_outlier_fraction=frac,
                theoretical_fraction=float(alpha),
                n_events=int(len(df_fit)),
                n_features=int(n_features),
            ))

    out = pd.DataFrame(rows)
    out_csv = TABLE_DIR / "alpha_outlier_fraction_table.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存 alpha-outlier fraction 表: {out_csv}")
    return out


# =============================================================================
# 8. 绘图函数
# =============================================================================

def save_figure(fig: plt.Figure, stem: str) -> None:
    """同时保存 PNG 和 PDF。"""
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[OK] 已保存: {png}")
    print(f"[OK] 已保存: {pdf}")


def plot_sensitivity_final_noise(scan_df: pd.DataFrame) -> None:
    """绘制 final_noise_count 的单参数敏感性组合图。"""
    groups = ["alpha", "K", "semblance_thr", "V_APP_MIN", "V_APP_MAX"]
    titles = {
        "alpha": r"$\alpha$",
        "K": "Neighborhood half-width K",
        "semblance_thr": "Semblance threshold",
        "V_APP_MIN": r"$V_{\mathrm{APP\_MIN}}$ (m/s)",
        "V_APP_MAX": r"$V_{\mathrm{APP\_MAX}}$ (m/s)",
    }

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(A4_WIDTH_IN * 0.98, 2.25),
        constrained_layout=True,
        sharey=True,
    )

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#7F7F7F"]

    for ax, group in zip(axes, groups):
        sub = scan_df.loc[scan_df["parameter_group"] == group].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        for i, (case, case_sub) in enumerate(sub.groupby("case_label")):
            case_sub = case_sub.sort_values("parameter_value")
            ax.plot(
                case_sub["parameter_value"],
                case_sub["final_noise_count"],
                marker="o",
                linewidth=1.0,
                markersize=3.2,
                color=colors[i % len(colors)],
                alpha=0.78,
                label=case,
            )

        mean_sub = (
            sub.groupby("parameter_value", as_index=False)["final_noise_count"]
            .mean()
            .sort_values("parameter_value")
        )
        ax.plot(
            mean_sub["parameter_value"],
            mean_sub["final_noise_count"],
            marker="s",
            linewidth=1.4,
            markersize=3.6,
            color="black",
            label="Mean",
        )

        ax.set_title(titles[group], pad=4)
        ax.set_xlabel("Parameter value")
        ax.grid(True, color="#B0B0B0", alpha=0.25, linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if group == "alpha":
            ax.set_xlabel(r"$\alpha$")
        elif group == "K":
            ax.set_xlabel("K")
        elif group == "semblance_thr":
            ax.set_xlabel("Threshold")
        elif group == "V_APP_MIN":
            ax.set_xlabel("m/s")
        elif group == "V_APP_MAX":
            ax.set_xlabel("m/s")

    axes[0].set_ylabel("Final noise count")
    handles, labels = axes[-1].get_legend_handles_labels()
    # 图例放到底部，避免压住曲线。
    # 图例下移，避免与各子图 x 轴标签挤在一起。
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.20))

    save_figure(fig, "FigureS_sensitivity_final_noise_count")
    plt.close(fig)


def plot_alpha_outlier_fraction(alpha_df: pd.DataFrame) -> None:
    """绘制 alpha-outlier fraction 与理论 y=alpha 对比图。"""
    if alpha_df.empty:
        print("[警告] alpha-outlier fraction 表为空，跳过绘图。")
        return

    fig, ax = plt.subplots(figsize=(A4_WIDTH_IN * 0.40, 2.55), constrained_layout=True)

    x = np.array(ALPHA_GRID, dtype=float)
    ax.plot(x, x, color="black", linestyle="--", linewidth=1.0, label=r"Theory: $y = \alpha$")

    colors = {
        "FORESEE field candidates": "#4C78A8",
        "Synthetic benchmark candidates": "#54A24B",
    }

    for dataset, sub in alpha_df.groupby("dataset"):
        sub = sub.sort_values("alpha")
        ax.plot(
            sub["alpha"],
            sub["empirical_outlier_fraction"],
            marker="o",
            linewidth=1.2,
            markersize=3.8,
            color=colors.get(dataset, None),
            label=dataset,
        )

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Empirical outlier fraction")
    ax.set_xlim(0, max(ALPHA_GRID) * 1.08)
    ax.set_ylim(0, max(0.12, alpha_df["empirical_outlier_fraction"].max() * 1.15))
    ax.grid(True, color="#B0B0B0", alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", frameon=True)

    save_figure(fig, "FigureS_alpha_outlier_fraction")
    plt.close(fig)


# =============================================================================
# 9. 主函数
# =============================================================================

def main() -> None:
    ensure_dirs()

    scan_csv = TABLE_DIR / "real_parameter_sensitivity_scan.csv"
    if scan_csv.exists():
        print(f"[读取] 已存在敏感性扫描表: {scan_csv}")
        scan_df = pd.read_csv(scan_csv, encoding="utf-8-sig")
    else:
        scan_df = run_real_sensitivity_scan()

    alpha_csv = TABLE_DIR / "alpha_outlier_fraction_table.csv"
    if alpha_csv.exists():
        print(f"[读取] 已存在 alpha-outlier fraction 表: {alpha_csv}")
        alpha_df = pd.read_csv(alpha_csv, encoding="utf-8-sig")
    else:
        alpha_df = build_alpha_outlier_table()

    plot_sensitivity_final_noise(scan_df)
    plot_alpha_outlier_fraction(alpha_df)

    print("=" * 80)
    print("[完成] 敏感性与稳健性分析图件已生成。")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
