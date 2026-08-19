"""
结果输出与报告。
"""

import os
import numpy as np
from typing import Dict, Optional, List
from .crystal_system import CrystalSystem, CRYSTAL_SYSTEM_NAMES
from .i18n import T


def print_cij_matrix(cij: np.ndarray, title: str = None) -> None:
    """以矩阵格式打印 6×6 弹性常数矩阵。"""
    if title is None:
        title = T("弹性常数矩阵 (GPa)", "Elastic constant matrix (GPa)")
    print(f"\n{title}")
    print("-" * 60)
    for i in range(6):
        row = "  ".join(f"{cij[i, j]:8.2f}" for j in range(6))
        print(f"  {row}")
    print()


def save_elastic_matrix(filepath: str, cij: np.ndarray) -> None:
    """
    将 6×6 弹性常数矩阵保存为纯数值文件。

    文件格式：6行×6列，空格分隔，保留2位小数，单位 GPa。
    不含标题或注释行，方便外部程序直接 np.loadtxt() 读取。
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    np.savetxt(filepath, cij, fmt="%.2f")


def write_results_file(
    filepath: str,
    cij_raw: np.ndarray,
    cij_sym: np.ndarray,
    fit_info: Dict,
    crystal_system: CrystalSystem,
    temperature: float,
    magnitude: float,
    method: str,
    mechanical_properties: Optional[Dict] = None,
    born_stable: bool = True,
    born_warnings: Optional[List[str]] = None,
    baseline_stress: Optional[np.ndarray] = None,
    apply_wallace: bool = True,
) -> None:
    """
    写入完整结果文件 elastic_constants.txt。

    Parameters
    ----------
    baseline_stress : 平衡构型的 Voigt 应力 (kBar), 用于 Wallace 压力诊断
    apply_wallace : 若提供 baseline_stress 且 True, 额外输出 Wallace 修正后矩阵
    """
    from .wallace import (compute_residual_pressure, apply_wallace_correction,
                          format_wallace_diagnostic)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(T("MHEC 弹性常数计算结果", "MHEC Elastic Constants Results") + "\n")
        f.write("=" * 60 + "\n\n")

        f.write(T("计算参数:", "Calculation parameters:") + "\n")
        f.write(f"  {T('晶系', 'Crystal system')}: {CRYSTAL_SYSTEM_NAMES.get(crystal_system, crystal_system.value)}\n")
        f.write(f"  {T('温度', 'Temperature')}: {temperature} K\n")
        f.write(f"  {T('应变幅度', 'Strain amplitude')}: {magnitude}\n")
        f.write(f"  {T('应变方案', 'Strain scheme')}: {method}\n\n")

        f.write(T("原始弹性常数矩阵 (GPa):", "Raw elastic constant matrix (GPa):") + "\n")
        for i in range(6):
            row = "  ".join(f"{cij_raw[i, j]:8.2f}" for j in range(6))
            f.write(f"  {row}\n")
        f.write("\n")

        f.write(T("对称性修正后弹性常数矩阵 (Birch, 未做压力修正) (GPa):",
                  "Symmetrized elastic constant matrix (Birch, no pressure correction) (GPa):") + "\n")
        for i in range(6):
            row = "  ".join(f"{cij_sym[i, j]:8.2f}" for j in range(6))
            f.write(f"  {row}\n")
        f.write("\n")

        # Wallace 压力修正 (如提供 baseline_stress)
        cij_final = cij_sym
        if baseline_stress is not None and apply_wallace:
            P_GPa, aniso_GPa = compute_residual_pressure(baseline_stress)
            cij_brugger = apply_wallace_correction(cij_sym, P_GPa)
            f.write(format_wallace_diagnostic(cij_sym, cij_brugger, P_GPa, aniso_GPa))
            f.write("\n")
            f.write(T("Wallace 修正后弹性常数矩阵 (Brugger, 推荐与实验对比) (GPa):",
                      "Wallace-corrected elastic constant matrix (Brugger, recommended for comparison with experiment) (GPa):") + "\n")
            for i in range(6):
                row = "  ".join(f"{cij_brugger[i, j]:8.2f}" for j in range(6))
                f.write(f"  {row}\n")
            f.write("\n")
            cij_final = cij_brugger

        f.write(f"{T('Born 稳定性', 'Born stability')}: "
                f"{T('满足','satisfied') if born_stable else T('不满足','not satisfied')}\n")
        if born_warnings:
            for w in born_warnings:
                f.write(f"  {T('警告','Warning')}: {w}\n")
        f.write("\n")

        if mechanical_properties:
            f.write(T("力学性质:", "Mechanical properties:") + "\n")
            for key, val in mechanical_properties.items():
                f.write(f"  {key}: {val:.4f}\n")
            f.write("\n")

        # 注: 逐通道 R² 拟合统计不再写入本文件 (与 SA/SC 一致, 且弱通道会误导);
        # 每个 Cij 由哪些强通道解出请见 fit_solution.txt / vcfit_solution.txt。


def print_comparison_table(
    results: Dict[str, np.ndarray],
    methods: List[str],
) -> None:
    """打印不同应变方案的弹性常数对比表格。"""
    print("\n应变方案对比:")
    print("-" * 60)
    header = f"{'C_ij':>8}" + "".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * 60)
    for i in range(6):
        for j in range(i, 6):
            label = f"C{i+1}{j+1}"
            vals = "".join(
                f"{results[m][i, j]:12.2f}" for m in methods
            )
            print(f"{label:>8}{vals}")
