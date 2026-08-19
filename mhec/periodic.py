"""
轻量元素周期表数据与组成/密度计算助手 (无第三方依赖)。

用于从 POSCAR 的元素符号 + 计数 + 晶胞体积推算:
- 化学式 (含化学式单元数 Z)
- 摩尔质量 (每化学式单元 / 每晶胞 / 每原子平均)
- 密度 (g/cm³)
"""

from math import gcd
from functools import reduce
from typing import Dict, List, Optional, Tuple

# 元素符号 -> (原子序数 Z, 标准原子量 g/mol)
ELEMENTS: Dict[str, Tuple[int, float]] = {
    "H": (1, 1.008), "He": (2, 4.0026),
    "Li": (3, 6.94), "Be": (4, 9.0122), "B": (5, 10.81), "C": (6, 12.011),
    "N": (7, 14.007), "O": (8, 15.999), "F": (9, 18.998), "Ne": (10, 20.180),
    "Na": (11, 22.990), "Mg": (12, 24.305), "Al": (13, 26.982), "Si": (14, 28.085),
    "P": (15, 30.974), "S": (16, 32.06), "Cl": (17, 35.45), "Ar": (18, 39.948),
    "K": (19, 39.098), "Ca": (20, 40.078), "Sc": (21, 44.956), "Ti": (22, 47.867),
    "V": (23, 50.942), "Cr": (24, 51.996), "Mn": (25, 54.938), "Fe": (26, 55.845),
    "Co": (27, 58.933), "Ni": (28, 58.693), "Cu": (29, 63.546), "Zn": (30, 65.38),
    "Ga": (31, 69.723), "Ge": (32, 72.630), "As": (33, 74.922), "Se": (34, 78.971),
    "Br": (35, 79.904), "Kr": (36, 83.798), "Rb": (37, 85.468), "Sr": (38, 87.62),
    "Y": (39, 88.906), "Zr": (40, 91.224), "Nb": (41, 92.906), "Mo": (42, 95.95),
    "Tc": (43, 98.0), "Ru": (44, 101.07), "Rh": (45, 102.91), "Pd": (46, 106.42),
    "Ag": (47, 107.868), "Cd": (48, 112.414), "In": (49, 114.818), "Sn": (50, 118.710),
    "Sb": (51, 121.760), "Te": (52, 127.60), "I": (53, 126.904), "Xe": (54, 131.293),
    "Cs": (55, 132.905), "Ba": (56, 137.327), "La": (57, 138.905), "Ce": (58, 140.116),
    "Pr": (59, 140.908), "Nd": (60, 144.242), "Pm": (61, 145.0), "Sm": (62, 150.36),
    "Eu": (63, 151.964), "Gd": (64, 157.25), "Tb": (65, 158.925), "Dy": (66, 162.500),
    "Ho": (67, 164.930), "Er": (68, 167.259), "Tm": (69, 168.934), "Yb": (70, 173.045),
    "Lu": (71, 174.967), "Hf": (72, 178.49), "Ta": (73, 180.948), "W": (74, 183.84),
    "Re": (75, 186.207), "Os": (76, 190.23), "Ir": (77, 192.217), "Pt": (78, 195.084),
    "Au": (79, 196.967), "Hg": (80, 200.592), "Tl": (81, 204.38), "Pb": (82, 207.2),
    "Bi": (83, 208.980), "Po": (84, 209.0), "At": (85, 210.0), "Rn": (86, 222.0),
    "Fr": (87, 223.0), "Ra": (88, 226.0), "Ac": (89, 227.0), "Th": (90, 232.038),
    "Pa": (91, 231.036), "U": (92, 238.029), "Np": (93, 237.0), "Pu": (94, 244.0),
}

_AMU_TO_G = 1.66053906660e-24  # 1 amu = 1.66054e-24 g


def atomic_number(symbol: str) -> Optional[int]:
    e = ELEMENTS.get(symbol)
    return e[0] if e else None


def atomic_mass(symbol: str) -> Optional[float]:
    e = ELEMENTS.get(symbol)
    return e[1] if e else None


def _list_gcd(nums: List[int]) -> int:
    return reduce(gcd, nums) if nums else 1


def composition_info(species: Optional[List[str]], counts: List[int],
                     volume_A3: float) -> Optional[Dict]:
    """从元素符号、计数、晶胞体积计算组成/摩尔质量/密度。

    Returns None 当 species 缺失或含未知元素 (无法算质量)。
    否则返回 dict:
        formula        : 约化化学式 (如 'Ga1In2Sn1')
        Z              : 化学式单元数
        mass_cell      : 晶胞总摩尔质量 (g/mol)
        mass_formula   : 每化学式单元摩尔质量 (g/mol)
        mass_per_atom  : 每原子平均摩尔质量 (g/mol)
        density        : 密度 (g/cm³)
        n_atoms        : 原子总数
    """
    if not species or len(species) != len(counts):
        return None
    masses = [atomic_mass(s) for s in species]
    if any(m is None for m in masses):
        return None

    mass_cell = sum(m * c for m, c in zip(masses, counts))  # g/mol per cell
    n_atoms = sum(counts)
    z = _list_gcd([int(c) for c in counts])
    z = z if z > 0 else 1
    reduced = [c // z for c in counts]
    formula = "".join(f"{s}{r}" for s, r in zip(species, reduced))
    density = (mass_cell * _AMU_TO_G) / (volume_A3 * 1e-24) if volume_A3 > 0 else float("nan")

    return {
        "formula": formula,
        "Z": z,
        "mass_cell": mass_cell,
        "mass_formula": mass_cell / z,
        "mass_per_atom": mass_cell / n_atoms if n_atoms else float("nan"),
        "density": density,
        "n_atoms": n_atoms,
    }


def get_spacegroup(lattice, positions, coord_type: str,
                   species: Optional[List[str]], counts: List[int],
                   symprec: float = 1e-3) -> Optional[str]:
    """用 spglib (若安装) 返回空间群字符串, 例如 'Fd-3m (227)'。

    spglib 未安装或元素缺失时返回 None。
    positions 为 POSCAR 中的坐标 (按 coord_type 判断是分数还是笛卡尔)。
    """
    try:
        import spglib
    except ImportError:
        return None
    if not species or len(species) != len(counts):
        return None
    import numpy as np

    lat = np.asarray(lattice, dtype=float)
    pos = np.asarray(positions, dtype=float)
    # 转成分数坐标
    ct = (coord_type or "").strip().lower()
    if ct.startswith("c") or ct.startswith("k"):  # Cartesian
        frac = pos @ np.linalg.inv(lat)
    else:
        frac = pos
    numbers = []
    for s, c in zip(species, counts):
        z = atomic_number(s)
        if z is None:
            return None
        numbers.extend([z] * int(c))
    if len(numbers) != len(frac):
        return None
    try:
        cell = (lat, frac, numbers)
        return spglib.get_spacegroup(cell, symprec=symprec)
    except Exception:
        return None
