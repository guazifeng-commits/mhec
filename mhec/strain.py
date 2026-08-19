"""
应变方案生成与 deform code 编解码。

支持两种应变方案：
1. Standard: 标准单分量应变模式（默认, 通用, 按晶系对称性自适应）
2. ULICS: Universal Linear-Independent Coupling Strains（高级, 耦合应变）

Voigt记号约定: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
应变张量 ε 与Voigt应变 e 的关系:
  ε = [[e1,    e6/2, e5/2],
       [e6/2,  e2,   e4/2],
       [e5/2,  e4/2, e3  ]]
"""

import numpy as np
from typing import List, Tuple, Dict
from .crystal_system import CrystalSystem, N_INDEPENDENT


# ============================================================
# Deform Code 编解码
# ============================================================

def encode_deform(strain_voigt: np.ndarray, magnitude: float) -> str:
    """
    将 Voigt 应变向量编码为 deform code。

    Parameters
    ----------
    strain_voigt : (6,) Voigt 应变向量
    magnitude : 基础应变幅度 δ

    Returns
    -------
    str : 如 "deform100000"

    Examples
    --------
    >>> encode_deform(np.array([0.01, 0, 0, 0, 0, 0]), 0.01)
    'deform100000'
    """
    coeffs = np.round(np.array(strain_voigt) / magnitude).astype(int)
    return "deform" + "".join(str(abs(c)) for c in coeffs)


def decode_deform(deform_code: str, magnitude: float) -> np.ndarray:
    """
    将 deform code 解码为 Voigt 应变向量。

    Parameters
    ----------
    deform_code : 如 "deform100000"
    magnitude : 基础应变幅度 δ

    Returns
    -------
    (6,) Voigt 应变向量
    """
    digits = deform_code.replace("deform", "")
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(
            f"deform code 格式不合法: '{deform_code}'，应为 'deform' + 6位数字"
        )
    return np.array([int(d) * magnitude for d in digits])


# ============================================================
# 应变张量操作
# ============================================================

def voigt_strain_to_tensor(e: np.ndarray) -> np.ndarray:
    """将 Voigt 应变向量 (6,) 转为 3×3 对称应变张量。"""
    return np.array([
        [e[0],     e[5] / 2, e[4] / 2],
        [e[5] / 2, e[1],     e[3] / 2],
        [e[4] / 2, e[3] / 2, e[2]    ],
    ])


def apply_strain(lattice: np.ndarray, strain_voigt: np.ndarray) -> np.ndarray:
    """
    对晶格施加应变: L' = L @ (I + ε)^T

    Parameters
    ----------
    lattice : (3, 3) 原始晶格矩阵
    strain_voigt : (6,) Voigt 应变向量

    Returns
    -------
    (3, 3) 变形后晶格矩阵
    """
    eps = voigt_strain_to_tensor(strain_voigt)
    F = np.eye(3) + eps
    return lattice @ F.T


# ============================================================
# 应变幅度标签
# ============================================================

def get_amplitude_labels(n_points: int = 5) -> List[Tuple[str, int]]:
    """
    生成应变幅度标签序列（不含零点，零点由 equilibrium 提供）。

    Parameters
    ----------
    n_points : 数据点数（3, 5, 7, 9, 11）

    Returns
    -------
    [(标签, 倍数)] 列表
    3点:  [("n1",-1), ("p1",1)]                                    → 2 个计算
    5点:  [("n2",-2), ("n1",-1), ("p1",1), ("p2",2)]              → 4 个计算
    7点:  [("n3",-3), ..., ("p3",3)]                               → 6 个计算
    9点:  [("n4",-4), ..., ("p4",4)]                               → 8 个计算
    11点: [("n5",-5), ..., ("p5",5)]                               → 10 个计算
    """
    half = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5}.get(n_points)
    if half is None:
        raise ValueError(f"n_points 必须为 3, 5, 7, 9 或 11，当前值: {n_points}")
    labels = []
    for k in range(half, 0, -1):
        labels.append((f"n{k}", -k))
    for k in range(1, half + 1):
        labels.append((f"p{k}", k))
    return labels


# ============================================================
# 各晶系 deform code 定义（参考 deformation_modes.md）
# ============================================================

_STANDARD_DEFORM_CODES = {
    CrystalSystem.CUBIC: [
        "deform100000", "deform000100",
    ],
    CrystalSystem.HEXAGONAL: [
        "deform100000", "deform001000", "deform000100",
    ],
    CrystalSystem.TRIGONAL_3M: [
        "deform100000", "deform001000", "deform000100",
    ],
    CrystalSystem.TRIGONAL_3: [
        "deform100000", "deform001000", "deform000100", "deform000010",
    ],
    CrystalSystem.TETRAGONAL_4MMM: [
        "deform100000", "deform001000", "deform000100", "deform000001",
    ],
    CrystalSystem.TETRAGONAL_4M: [
        "deform100000", "deform001000", "deform000100", "deform000001",
    ],
    CrystalSystem.ORTHORHOMBIC: [
        "deform100000", "deform010000", "deform001000",
        "deform000100", "deform000010", "deform000001",
    ],
    CrystalSystem.MONOCLINIC: [
        "deform100000", "deform010000", "deform001000",
        "deform000100", "deform000010", "deform000001",
    ],
    CrystalSystem.TRICLINIC: [
        "deform100000", "deform010000", "deform001000",
        "deform000100", "deform000010", "deform000001",
    ],
}

# ULICS 6个线性无关耦合应变向量
_ULICS_MATRIX = np.array([
    [ 1,  1,  1,  0,  0,  0],
    [ 1, -1,  0,  1,  0,  0],
    [ 1,  0, -1,  0,  1,  0],
    [ 0,  1, -1,  0,  0,  1],
    [ 0,  0,  1,  1,  1,  0],
    [ 0,  1,  0,  0,  1,  1],
], dtype=float)


def get_deform_codes(
    crystal_system: CrystalSystem,
    method: str = "standard",
) -> List[str]:
    """
    根据晶系和方案返回 deform code 列表。

    Parameters
    ----------
    crystal_system : 晶系枚举
    method : "standard" | "ulics"
    """
    method = method.lower()
    if method == "standard":
        return list(_STANDARD_DEFORM_CODES[crystal_system])
    elif method == "ulics":
        # ULICS 方案: 6个全耦合应变
        codes = []
        for row in _ULICS_MATRIX:
            code = "deform" + "".join(str(abs(int(c))) for c in row)
            codes.append(code)
        return codes
    else:
        raise ValueError(f"未知应变方案: {method}，可选: standard, ulics")


def generate_strained_structures(
    lattice: np.ndarray,
    deform_codes: List[str],
    magnitude: float = 0.01,
    n_points: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    为所有 deform code 和幅度点生成变形晶格。

    Returns
    -------
    {deform_code: {amplitude_label: deformed_lattice}}
    """
    amp_labels = get_amplitude_labels(n_points)
    result = {}
    for code in deform_codes:
        base_strain = decode_deform(code, magnitude)
        result[code] = {}
        for label, mult in amp_labels:
            strain = base_strain * mult
            result[code][label] = apply_strain(lattice, strain)
    return result


def compare_strain_methods(
    crystal_system: CrystalSystem,
    magnitude: float = 0.01,
) -> Dict[str, Dict]:
    """对比三种应变方案的 deform code 列表、应变数目、设计矩阵秩和条件数。"""
    results = {}
    for method in ["standard", "ulics"]:
        codes = get_deform_codes(crystal_system, method)
        # 构建设计矩阵
        strain_vectors = []
        for code in codes:
            strain_vectors.append(decode_deform(code, magnitude))
        A = np.array(strain_vectors)

        rank = np.linalg.matrix_rank(A)
        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cond = float("inf")

        results[method] = {
            "deform_codes": codes,
            "n_deformations": len(codes),
            "n_total_calcs": len(codes) * len(get_amplitude_labels(5)),
            "matrix_rank": rank,
            "condition_number": cond,
        }
    return results
