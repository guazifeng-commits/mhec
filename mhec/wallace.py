"""
Wallace 残余压力修正 (Birch ↔ Brugger 弹性常数)。

在参考态存在非零静水压力 P 时，通过应力-应变法直接拟合得到的
"Birch 型" 弹性常数 B_ijkl 与标准定义的 "Brugger 型" C_ijkl 不同:

    B_ij = C_ij + P * Δ_ij

其中 Δ_ij 的非零项为:
    Δ_11 = Δ_22 = Δ_33 = -1   (法向对角)
    Δ_12 = Δ_13 = Δ_23 = +1   (法向交叉, Voigt 记号中位置 (i,j) 满足 i,j∈{1,2,3}, i≠j)
    Δ_44 = Δ_55 = Δ_66 = -1   (剪切对角)

即 (GPa 单位):
    C_ij = B_ij - P * Δ_ij
         = { B_ii + P       (i = 1,2,3, 法向对角)
           { B_ij - P       (i,j ∈ {1,2,3}, i ≠ j, 法向交叉)
           { B_ii + P       (i = 4,5,6, 剪切对角)

参考文献:
  - D.C. Wallace, Thermodynamics of Crystals, Wiley, 1972.
  - T.H.K. Barron, M.L. Klein, Proc. Phys. Soc. 85 (1965) 523-532.
  - G.V. Sin'ko, N.A. Smirnov, J. Phys. Condens. Matter 14 (2002) 6989.

物理含义:
  P > 0 (残余压缩) → C11, C22, C33, C44, C55, C66 应加回 P
                   → C12, C13, C23 应减去 P
  理想情况 P ≈ 0 时 Birch ≡ Brugger，此修正可忽略。
"""

import numpy as np
from typing import Dict, Optional, Tuple


def _voigt_bulk(cij: np.ndarray) -> Optional[float]:
    """Voigt 体模量 K_V = (C11+C22+C33 + 2(C12+C13+C23))/9, 对所有晶系通用。

    用于把残余压力换算成体积应变, 从而给出与材料软硬无关的判据。
    """
    try:
        C = np.asarray(cij, dtype=float)
        if C.shape != (6, 6) or not np.all(np.isfinite(C[:3, :3])):
            return None
        k = (C[0, 0] + C[1, 1] + C[2, 2]
             + 2.0 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
        return float(k) if k > 0 else None
    except Exception:
        return None


def compute_residual_pressure(baseline_stress: np.ndarray) -> Tuple[float, float]:
    """
    从平衡构型的时间平均应力计算残余压力。

    Parameters
    ----------
    baseline_stress : (6,) Voigt 应力 (kBar, VASP 压力约定: 正=压缩)

    Returns
    -------
    P_residual : 静水残余压力 (GPa, 正 = 压缩)
    anisotropy : 法向应力各向异性 σ_ii 的标准差 (GPa)，诊断用
    """
    # VASP 报告的应力是压力约定，正值 = 压缩
    # 静水压力 P = (P_xx + P_yy + P_zz) / 3 (kBar)
    # 转为 GPa: 乘 0.1
    P_kBar = np.mean(baseline_stress[:3])
    P_GPa = P_kBar * 0.1
    anisotropy_GPa = np.std(baseline_stress[:3]) * 0.1
    return P_GPa, anisotropy_GPa


def apply_wallace_correction(
    cij_birch: np.ndarray,
    pressure_GPa: float,
) -> np.ndarray:
    """
    对 Birch 型弹性常数矩阵施加 Wallace 压力修正, 得到 Brugger 型矩阵。

    Parameters
    ----------
    cij_birch : (6, 6) 拟合得到的原始矩阵 (GPa)
    pressure_GPa : 残余静水压力 (GPa, 正=压缩)

    Returns
    -------
    cij_brugger : (6, 6) 修正后的 Brugger 弹性常数 (GPa)

    Notes
    -----
    对 i = 1,2,3 (法向):
      C_ii = B_ii + P  (对角)
      C_ij = B_ij - P  (i ≠ j, 非对角, 两个指标都在 {1,2,3})
    对 i = 4,5,6 (剪切):
      C_ii = B_ii + P
    其他元素 (法向-剪切交叉、剪切非对角) 不变。
    """
    C = cij_birch.copy()
    P = pressure_GPa

    # 法向对角: C11, C22, C33 → +P
    for i in range(3):
        C[i, i] += P

    # 法向交叉: C12, C13, C23 (以及对称的 C21, C31, C32) → -P
    for i in range(3):
        for j in range(i + 1, 3):
            C[i, j] -= P
            C[j, i] -= P

    # 剪切对角: C44, C55, C66 → +P
    for i in range(3, 6):
        C[i, i] += P

    return C


def format_wallace_diagnostic(
    cij_birch: np.ndarray,
    cij_brugger: np.ndarray,
    pressure_GPa: float,
    anisotropy_GPa: float,
) -> str:
    """生成 Wallace 诊断文本块。"""
    lines = []
    lines.append("残余压力诊断 (Wallace 修正):")
    lines.append(f"  残余静水压力 P = {pressure_GPa:+.3f} GPa  "
                 f"(正值=压缩, 负值=拉伸)")
    lines.append(f"  法向应力各向异性 = {anisotropy_GPa:.3f} GPa")

    # 状态判断必须用相对量 |P|/K, 不能用 |P| 的绝对值。
    # |P|/K 就是该残余应力对应的体积应变, 即固定晶胞相对力场平衡态偏离了多少。
    # 同样的 0.3 GPa, 在 K = 200 GPa 的陶瓷上只有 0.15% 体积应变, 确实可忽略;
    # 在 K = 6 GPa 的软框架上是 5%, 足以把接近失稳的结构推过相界。
    absP = abs(pressure_GPa)
    k_voigt = _voigt_bulk(cij_birch)
    strain = absP / k_voigt if k_voigt and k_voigt > 0 else None

    if strain is None:
        status = ("无法由弹性矩阵得到体模量, 只能给绝对值; "
                  f"|P| = {absP:.3f} GPa")
    else:
        lines.append(f"  体模量 K_V = {k_voigt:.2f} GPa, "
                     f"对应体积应变 |P|/K = {strain * 100:.2f}%")
        if strain < 0.005:
            status = "残余压力可忽略, 对应体积应变小于 0.5%"
        elif strain < 0.02:
            status = "残余压力偏小, 对应体积应变 0.5% 到 2%, 建议用修正后值"
        elif strain < 0.05:
            status = ("残余压力不可忽略, 对应体积应变 2% 到 5%, "
                      "固定晶胞已明显偏离力场平衡态")
        else:
            status = ("残余压力过大, 对应体积应变超过 5%, "
                      "NPT 晶胞与力场不自洽, 建议重新弛豫后重做")
    lines.append(f"  状态: {status}")

    if strain is not None and strain >= 0.02:
        sign = "压缩" if pressure_GPa > 0 else "拉伸"
        lines.append(f"  提醒: 固定晶胞相对力场平衡态处于{sign}态, "
                     f"体积偏差 {strain * 100:.1f}%。")
        lines.append("        软框架或临近相变的体系, 这个量级的体积偏差足以改变")
        lines.append("        所处的相, 使弹性常数对应到另一个结构上。")
        lines.append("        建议把该温度的平衡晶胞重新弛豫到零压后再跑变形。")
    if k_voigt and 0 < k_voigt < 30.0:
        lines.append(f"  提醒: K_V 只有 {k_voigt:.1f} GPa, 属软材料, "
                     f"残余压力的绝对值会低估其影响, 请以 |P|/K 为准。")
    lines.append("")

    # 展示典型分量的修正
    lines.append("  主要分量修正:")
    lines.append(f"    {'分量':>6}  {'Birch (拟合)':>12}  →  {'Brugger (修正)':>14}")
    components = [
        ("C11", 0, 0), ("C22", 1, 1), ("C33", 2, 2),
        ("C12", 0, 1), ("C13", 0, 2), ("C23", 1, 2),
        ("C44", 3, 3), ("C55", 4, 4), ("C66", 5, 5),
    ]
    for label, i, j in components:
        b = cij_birch[i, j]
        c = cij_brugger[i, j]
        if abs(b) > 1e-6 or abs(c) > 1e-6:
            lines.append(f"    {label:>6}  {b:>12.2f}  →  {c:>14.2f}  "
                         f"(Δ = {c-b:+.2f})")

    return "\n".join(lines) + "\n"
