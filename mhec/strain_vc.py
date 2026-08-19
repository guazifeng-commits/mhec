"""
体积守恒变形 (VC / 变形元胞法) 应变模式生成。

与 strain.py (standard/ULICS/OHESS) 并列的另一套应变方案。
VC 使用体积近似守恒的变形 (正交 [δ,-δ,0]、双轴补偿、体积守恒剪切),
消除体积变化对应力的影响, 对剪切常数和立方 C11-C12 的精度更友好。

Voigt 记号: e = [εxx, εyy, εzz, 2εyz, 2εxz, 2εxy]
应变张量 ε = [[e1, e6/2, e5/2], [e6/2, e2, e4/2], [e5/2, e4/2, e3]]

参考: Wen et al., J. Appl. Phys. 113, 103501 (2013);
      Zhao et al., Phys. Rev. B 75, 094105 (2007).
"""

import numpy as np
from typing import List, Dict
from .crystal_system import CrystalSystem


# ============================================================
#  各晶系变形模式定义
# ============================================================
# 设计原则 (2.x 修订):
#   1. 法向块 (C11,C12,C13,C22,C23,C33 ...) 一律用「直接单轴」应变 (type=single,
#      idx=0/1/2)。单轴 e_j 施加后 σ = C[:,j]，每个正应力/耦合常数都获得
#      「强信号、反对称、直接」的探针 —— 彻底避免旧版把 C33/C13 塞进弱 zz 组合
#      (ortho_ac / biaxial) 导致的单边归零与噪声主导问题。
#   2. 剪切块 (C44,C55,C66) 保留体积守恒剪切 (type=shear)，其 R² 一贯很好。
#   3. 额外保留正交偏量 ortho ([1,-1,0]) 用于精确测 C11-C12 (大数相减，直接
#      单轴的差分误差大，偏量应变更准)。
#   4. 低对称 (三方/单斜/三斜) 用「单轴 + 单剪切」的通用单分量集合覆盖全部列，
#      含 shear-normal 耦合常数 (C14/C15/C16/C45/C46/C56)，由 symmetrize 归约。
#
#   type=single  : 只置 1 个 Voigt 分量 = δ (idx 0-2 为正应变, 3-5 为剪切),
#                  方向 d = e_idx, 通用、无体积补偿、对任意对称性都干净。
#   type=shear   : 体积守恒剪切 F=[[1+a,..],[..γ..]], a=γ²/(1-γ²), 方向 = 2·e_idx。
#   type=ortho_simple / biaxial / hydro : 见 vc_mode_strain (保留兼容)。
#
# 每个模式: name, type, 生成所需信息, target(仅注释用)。
_VC_MODES = {
    "cubic": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12"},
        {"name": "ortho", "type": "ortho_simple",
         "indices": [(0, 1), (1, -1), (2, 0)], "target": "C11-C12"},
        {"name": "shear_yz", "type": "shear", "shear_idx": 3, "comp_idx": 0,
         "target": "C44"},
    ],
    "hexagonal": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12,C13"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "C13,C33"},
        {"name": "ortho_ab", "type": "ortho_simple",
         "indices": [(0, 1), (1, -1), (2, 0)], "target": "C11-C12"},
        # ortho_ac: e=(δ,0,−δ) 体积守恒 a-c 偏量。σ1 斜率=C11−C13、σ3 斜率=C13−C33
        # 均为「强差分信号」(~C11/C33 量级)，替代 uni_x σ3 / uni_z σ1 的弱横向读数，
        # 让 C13 获得与 ortho_ab 测 C11-C12 对等的强约束 (框架结构如 BeF2/石英尤需)。
        {"name": "ortho_ac", "type": "ortho_simple",
         "indices": [(0, 1), (2, -1), (1, 0)], "target": "C11+C33-2C13"},
        {"name": "shear_yz", "type": "shear", "shear_idx": 3, "comp_idx": 0,
         "target": "C44"},
        # C66 = (C11-C12)/2 由对称性派生
    ],
    "tetragonal": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12,C13"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "C13,C33"},
        {"name": "ortho_ab", "type": "ortho_simple",
         "indices": [(0, 1), (1, -1), (2, 0)], "target": "C11-C12"},
        # ortho_ac: e=(δ,0,−δ) —— 见 hexagonal 注释，给 C13 强差分约束。
        {"name": "ortho_ac", "type": "ortho_simple",
         "indices": [(0, 1), (2, -1), (1, 0)], "target": "C11+C33-2C13"},
        {"name": "shear_yz", "type": "shear", "shear_idx": 3, "comp_idx": 0,
         "target": "C44 (=C55)"},
        {"name": "shear_xy", "type": "shear", "shear_idx": 5, "comp_idx": 2,
         "target": "C66"},
    ],
    "orthorhombic": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12,C13"},
        {"name": "uni_y", "type": "single", "idx": 1, "target": "C12,C22,C23"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "C13,C23,C33"},
        {"name": "shear_yz", "type": "shear", "shear_idx": 3, "comp_idx": 0,
         "target": "C44"},
        {"name": "shear_xz", "type": "shear", "shear_idx": 4, "comp_idx": 1,
         "target": "C55"},
        {"name": "shear_xy", "type": "shear", "shear_idx": 5, "comp_idx": 2,
         "target": "C66"},
    ],
    # 三方 (-3m: C11,C12,C13,C14,C33,C44; -3 另加 C15):
    #   uni_x -> σ4=C14 (C41), uni_z -> C33, shear_yz -> C44 & σ1=C14,
    #   shear_xz -> σ1=C15 (-3 需要)。C66=(C11-C12)/2 派生。
    "trigonal": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12,C13,C14"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "C13,C33"},
        {"name": "ortho_ab", "type": "ortho_simple",
         "indices": [(0, 1), (1, -1), (2, 0)], "target": "C11-C12"},
        # ortho_ac: e=(δ,0,−δ) —— 见 hexagonal 注释，给 C13 强差分约束。
        {"name": "ortho_ac", "type": "ortho_simple",
         "indices": [(0, 1), (2, -1), (1, 0)], "target": "C11+C33-2C13"},
        {"name": "shear_yz", "type": "single", "idx": 3, "target": "C44,C14"},
        {"name": "shear_xz", "type": "single", "idx": 4, "target": "C55,C15"},
    ],
    # 单斜 (unique axis b, 13 个: 含 C15,C25,C35,C46): 6 个单分量覆盖全部列
    "monoclinic": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "C11,C12,C13,C15"},
        {"name": "uni_y", "type": "single", "idx": 1, "target": "C12,C22,C23,C25"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "C13,C23,C33,C35"},
        {"name": "shear_yz", "type": "single", "idx": 3, "target": "C44,C46"},
        {"name": "shear_xz", "type": "single", "idx": 4, "target": "C55,C15,C25,C35"},
        {"name": "shear_xy", "type": "single", "idx": 5, "target": "C66,C46"},
    ],
    # 三斜 (21 个): 6 个单分量给出完整 6 列
    "triclinic": [
        {"name": "uni_x", "type": "single", "idx": 0, "target": "col1"},
        {"name": "uni_y", "type": "single", "idx": 1, "target": "col2"},
        {"name": "uni_z", "type": "single", "idx": 2, "target": "col3"},
        {"name": "shear_yz", "type": "single", "idx": 3, "target": "col4"},
        {"name": "shear_xz", "type": "single", "idx": 4, "target": "col5"},
        {"name": "shear_xy", "type": "single", "idx": 5, "target": "col6"},
    ],
}

# 将 9 个晶系枚举映射到上面的模式族
_CS_FAMILY = {
    CrystalSystem.CUBIC: "cubic",
    CrystalSystem.HEXAGONAL: "hexagonal",
    CrystalSystem.TETRAGONAL_4MMM: "tetragonal",
    CrystalSystem.TETRAGONAL_4M: "tetragonal",
    CrystalSystem.TRIGONAL_3M: "trigonal",
    CrystalSystem.TRIGONAL_3: "trigonal",
    CrystalSystem.ORTHORHOMBIC: "orthorhombic",
    CrystalSystem.MONOCLINIC: "monoclinic",
    CrystalSystem.TRICLINIC: "triclinic",
}

# VC/SA 法现已覆盖全部 9 个晶系
VC_SUPPORTED = set(_CS_FAMILY.keys())


def get_vc_modes(crystal_system: CrystalSystem) -> List[Dict]:
    """返回指定晶系的体积守恒变形模式列表。"""
    family = _CS_FAMILY.get(crystal_system)
    if family is None:
        raise ValueError(
            f"VC (体积守恒) 法暂不支持 {crystal_system.value}，"
            f"请使用 standard 或 ulics 方案。"
        )
    return [dict(m) for m in _VC_MODES[family]]


# ============================================================
#  应变张量生成
# ============================================================

def vc_mode_strain(mode: Dict, amplitude: float) -> np.ndarray:
    """
    根据模式和幅度生成完整 Voigt 应变 (含体积守恒补偿, 用于写 POSCAR)。
    """
    strain = np.zeros(6)
    t = mode["type"]

    if t == "single":
        # 单分量应变: 只置 1 个 Voigt 分量 = δ。方向 d = e_idx (无体积补偿)。
        # idx 0-2 为正应变, 3-5 为剪切 (Voigt e4/e5/e6 = 2ε_yz/xz/xy)。
        strain[mode["idx"]] = amplitude

    elif t == "ortho_simple":
        # 体积守恒偏量应变: e1=+δ, e2=−δ, 并在剩余法向分量上加补偿
        # e3 = δ²/(1−δ²) 使 det(I+ε)=1 (严格保体积, 与文献 LCEC 偏量形式一致)。
        # 补偿项为二阶 (∝δ²), 不影响 slope=dσ/dδ 的线性响应 (C11−C12 提取不变)。
        a = amplitude ** 2 / (1.0 - amplitude ** 2)
        for idx, sign in mode["indices"]:
            strain[idx] = sign * amplitude if sign != 0 else a

    elif t == "biaxial":
        # [δ, δ, c, ...], c = 1/(1+δ)^2 - 1 使体积守恒
        c = 1.0 / (1.0 + amplitude) ** 2 - 1.0
        used = set()
        for idx, sign in mode["indices"]:
            strain[idx] = sign * amplitude
            used.add(idx)
        for i in range(3):
            if i not in used:
                strain[i] = c
                break

    elif t == "shear":
        # F = [[1+a,0,0],[0,1,γ],[0,γ,1]], det=1 → a = γ^2/(1-γ^2)
        gamma = amplitude
        a = gamma ** 2 / (1.0 - gamma ** 2)
        strain[mode["shear_idx"]] = 2.0 * gamma
        strain[mode["comp_idx"]] = a

    elif t == "hydro":
        for idx, sign in mode["indices"]:
            strain[idx] = sign * amplitude

    return strain


def vc_mode_direction(mode: Dict) -> np.ndarray:
    """
    返回模式的线性应变方向 d (单位幅度的一阶应变), 用于拟合 slope = C @ d。
    通过对小幅度做数值微分获得, 自动丢弃二阶体积守恒补偿项。
    """
    eps = 1e-6
    return vc_mode_strain(mode, eps) / eps


def voigt_strain_to_tensor(e: np.ndarray) -> np.ndarray:
    """Voigt 应变 (6,) → 3×3 对称应变张量。"""
    return np.array([
        [e[0],     e[5] / 2, e[4] / 2],
        [e[5] / 2, e[1],     e[3] / 2],
        [e[4] / 2, e[3] / 2, e[2]    ],
    ])


def apply_vc_strain(lattice: np.ndarray, strain_voigt: np.ndarray) -> np.ndarray:
    """对晶格施加应变: L' = L @ (I + ε)^T。"""
    F = np.eye(3) + voigt_strain_to_tensor(strain_voigt)
    return lattice @ F.T


def check_volume_conservation(strain_voigt: np.ndarray) -> float:
    """返回变形后体积比 det(F)。"""
    F = np.eye(3) + voigt_strain_to_tensor(strain_voigt)
    return float(np.linalg.det(F))


def generate_vc_structures(
    lattice: np.ndarray,
    crystal_system: CrystalSystem,
    magnitude: float = 0.01,
    n_points: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    为所有 VC 模式和幅度点生成变形晶格。

    Returns
    -------
    {mode_name: {amplitude_label: deformed_lattice}}
    """
    from .strain import get_amplitude_labels

    modes = get_vc_modes(crystal_system)
    amp_labels = get_amplitude_labels(n_points)
    result = {}
    for mode in modes:
        result[mode["name"]] = {}
        for label, mult in amp_labels:
            amp = magnitude * mult
            strain = vc_mode_strain(mode, amp)
            result[mode["name"]][label] = apply_vc_strain(lattice, strain)
    return result
