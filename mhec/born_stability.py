"""
Born 稳定性判据检验。

通用判据：弹性常数矩阵正定（所有特征值 > 0）。
各晶系特定判据：基于 Born 条件的力学稳定性检验。
"""

import numpy as np
from typing import Tuple, List, Optional
from .crystal_system import CrystalSystem


def check_born_stability(
    cij: np.ndarray,
    crystal_system: Optional[CrystalSystem] = None,
    tol: float = 1e-10,
) -> Tuple[bool, List[str]]:
    """
    检验弹性常数矩阵是否满足 Born 稳定性判据。

    Parameters
    ----------
    cij : (6, 6) 弹性常数矩阵 (GPa)
    crystal_system : 晶系枚举，若提供则执行系统特定判据
    tol : 浮点容差

    Returns
    -------
    (is_stable, violation_messages)
    """
    violations = []

    # 通用判据：正定性（对所有晶系）
    eigenvalues = np.linalg.eigvalsh(cij)
    for i, ev in enumerate(eigenvalues):
        if ev <= -tol:  # 使用容差避免浮点噪声
            violations.append(
                f"特征值 λ{i+1} = {ev:.6f} ≤ 0，弹性常数矩阵非正定"
            )

    # 晶系特定 Born 判据
    if crystal_system is not None:
        cs_name = crystal_system.value if hasattr(crystal_system, 'value') else str(crystal_system)

        if crystal_system == CrystalSystem.CUBIC:
            c11, c12, c44 = cij[0, 0], cij[0, 1], cij[3, 3]
            if c11 - c12 <= tol:
                violations.append(f"C11 - C12 = {c11 - c12:.3f} ≤ 0，立方 Born 判据不满足")
            if c11 + 2 * c12 <= tol:
                violations.append(f"C11 + 2C12 = {c11 + 2 * c12:.3f} ≤ 0，立方 Born 判据不满足")
            if c44 <= tol:
                violations.append(f"C44 = {c44:.3f} ≤ 0，立方 Born 判据不满足")

        elif crystal_system in (CrystalSystem.HEXAGONAL, CrystalSystem.TRIGONAL_3M,
                                  CrystalSystem.TRIGONAL_3):
            c11, c12, c13, c33, c44 = cij[0, 0], cij[0, 1], cij[0, 2], cij[2, 2], cij[3, 3]
            if c11 - abs(c12) <= tol:
                violations.append(f"C11 - |C12| = {c11 - abs(c12):.3f} ≤ 0，六方/三角 Born 判据不满足")
            if c44 <= tol:
                violations.append(f"C44 = {c44:.3f} ≤ 0，六方/三角 Born 判据不满足")
            if (c11 + c12) * c33 - 2 * c13 ** 2 <= tol:
                violations.append(
                    f"(C11+C12)C33 - 2C13² = {(c11 + c12) * c33 - 2 * c13 ** 2:.3f} ≤ 0，六方/三角 Born 判据不满足")

        elif crystal_system in (CrystalSystem.TETRAGONAL_4MMM, CrystalSystem.TETRAGONAL_4M):
            c11, c12, c13, c33, c44, c66 = cij[0, 0], cij[0, 1], cij[0, 2], cij[2, 2], cij[3, 3], cij[5, 5]
            if c11 - abs(c12) <= tol:
                violations.append(f"C11 - |C12| = {c11 - abs(c12):.3f} ≤ 0，四方 Born 判据不满足")
            if c33 <= tol:
                violations.append(f"C33 = {c33:.3f} ≤ 0，四方 Born 判据不满足")
            if c44 <= tol:
                violations.append(f"C44 = {c44:.3f} ≤ 0，四方 Born 判据不满足")
            if c66 <= tol:
                violations.append(f"C66 = {c66:.3f} ≤ 0，四方 Born 判据不满足")
            if (c11 + c12) * c33 - 2 * c13 ** 2 <= tol:
                violations.append(
                    f"(C11+C12)C33 - 2C13² = {(c11 + c12) * c33 - 2 * c13 ** 2:.3f} ≤ 0，四方 Born 判据不满足")

        elif crystal_system == CrystalSystem.ORTHORHOMBIC:
            c11, c22, c33 = cij[0, 0], cij[1, 1], cij[2, 2]
            c12, c13, c23 = cij[0, 1], cij[0, 2], cij[1, 2]
            c44, c55, c66 = cij[3, 3], cij[4, 4], cij[5, 5]
            for name, val in [("C11", c11), ("C22", c22), ("C33", c33),
                               ("C44", c44), ("C55", c55), ("C66", c66)]:
                if val <= tol:
                    violations.append(f"{name} = {val:.3f} ≤ 0，正交 Born 判据不满足")
            if c11 + c22 - 2 * c12 <= tol:
                violations.append(f"C11+C22-2C12 = {c11 + c22 - 2 * c12:.3f} ≤ 0")
            if c11 + c33 - 2 * c13 <= tol:
                violations.append(f"C11+C33-2C13 = {c11 + c33 - 2 * c13:.3f} ≤ 0")
            if c22 + c33 - 2 * c23 <= tol:
                violations.append(f"C22+C33-2C23 = {c22 + c33 - 2 * c23:.3f} ≤ 0")
            if c11 + c22 + c33 + 2 * (c12 + c13 + c23) <= tol:
                violations.append(
                    f"C11+C22+C33+2(C12+C13+C23) = {c11 + c22 + c33 + 2 * (c12 + c13 + c23):.3f} ≤ 0")

    is_stable = len(violations) == 0
    return is_stable, violations
