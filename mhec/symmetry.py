"""
对称性约束规则。

定义各晶系弹性常数矩阵的零项、等价项组和派生关系，
用于对原始拟合矩阵进行对称性修正。

矩阵索引使用 0-based (i, j)，对应 Voigt 记号 C_{i+1, j+1}。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable
from .crystal_system import CrystalSystem


@dataclass
class SymmetryRule:
    """单条对称性约束规则。"""
    crystal_system: CrystalSystem
    zero_entries: List[Tuple[int, int]]
    equivalent_groups: List[List[Tuple[int, int]]]
    derived_relations: List[Dict]  # {target: (i,j), func: callable(cij) -> float}


def _cubic_rules() -> SymmetryRule:
    """立方: C11=C22=C33, C12=C13=C23, C44=C55=C66, 其余为零。"""
    # 零项: 正应变-剪切应变交叉块 + 剪切应变非对角
    zeros = []
    for i in range(3):
        for j in range(3, 6):
            zeros.append((i, j))
    # C45, C46, C56
    zeros.extend([(3, 4), (3, 5), (4, 5)])

    return SymmetryRule(
        crystal_system=CrystalSystem.CUBIC,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1), (2, 2)],       # C11=C22=C33
            [(0, 1), (0, 2), (1, 2)],       # C12=C13=C23
            [(3, 3), (4, 4), (5, 5)],       # C44=C55=C66
        ],
        derived_relations=[],
    )


def _hexagonal_rules() -> SymmetryRule:
    zeros = []
    for i in range(3):
        for j in range(3, 6):
            zeros.append((i, j))
    zeros.extend([(3, 4), (3, 5), (4, 5)])

    return SymmetryRule(
        crystal_system=CrystalSystem.HEXAGONAL,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1)],               # C11=C22
            [(0, 2), (1, 2)],               # C13=C23
            [(3, 3), (4, 4)],               # C44=C55
        ],
        derived_relations=[
            {"target": (5, 5), "func": lambda c: (c[0, 0] - c[0, 1]) / 2},
        ],
    )


def _trigonal_3m_rules() -> SymmetryRule:
    """三方 -3m: 6个独立常数 C11,C12,C13,C14,C33,C44。"""
    # C15=C16=C24=C25=C26=C34=C35=C36=C45=C46=C56=0
    zeros = [
        (0, 4), (0, 5),  # C15, C16
        (1, 3), (1, 4), (1, 5),  # C24, C25, C26 (注: C24=-C14 由等价处理)
        (2, 3), (2, 4), (2, 5),  # C34, C35, C36
        (3, 4), (3, 5), (4, 5),  # C45, C46, C56 (注: C56=C14 由等价处理)
    ]
    # 修正: C24=-C14, C56=C14 不是零，需要从 zeros 中移除并在等价组处理
    zeros = [
        (0, 4), (0, 5),
        (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
    ]

    return SymmetryRule(
        crystal_system=CrystalSystem.TRIGONAL_3M,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1)],       # C11=C22
            [(0, 2), (1, 2)],       # C13=C23
            [(3, 3), (4, 4)],       # C44=C55
        ],
        derived_relations=[
            {"target": (5, 5), "func": lambda c: (c[0, 0] - c[0, 1]) / 2},
            # C24 = -C14, C56 = C14
            {"target": (1, 3), "func": lambda c: -c[0, 3]},
            {"target": (4, 5), "func": lambda c: c[0, 3]},
        ],
    )


def _trigonal_3_rules() -> SymmetryRule:
    """三方 -3: 7个独立常数 C11,C12,C13,C14,C15,C33,C44。"""
    zeros = [
        (0, 5), (1, 5),  # C16=0, C26=0
        (2, 3), (2, 4), (2, 5),  # C34=C35=C36=0
        (3, 4),  # C45=0
    ]

    return SymmetryRule(
        crystal_system=CrystalSystem.TRIGONAL_3,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1)],       # C11=C22
            [(0, 2), (1, 2)],       # C13=C23
            [(3, 3), (4, 4)],       # C44=C55
        ],
        derived_relations=[
            {"target": (5, 5), "func": lambda c: (c[0, 0] - c[0, 1]) / 2},
            {"target": (1, 3), "func": lambda c: -c[0, 3]},
            {"target": (1, 4), "func": lambda c: -c[0, 4]},
            {"target": (3, 5), "func": lambda c: -c[0, 4]},
            {"target": (4, 5), "func": lambda c: c[0, 3]},
        ],
    )


def _tetragonal_4mmm_rules() -> SymmetryRule:
    """四方 4/mmm: 6个独立常数 C11,C12,C13,C33,C44,C66。"""
    zeros = []
    for i in range(3):
        for j in range(3, 6):
            zeros.append((i, j))
    zeros.extend([(3, 4), (3, 5), (4, 5)])

    return SymmetryRule(
        crystal_system=CrystalSystem.TETRAGONAL_4MMM,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1)],       # C11=C22
            [(0, 2), (1, 2)],       # C13=C23
            [(3, 3), (4, 4)],       # C44=C55
        ],
        derived_relations=[],
    )


def _tetragonal_4m_rules() -> SymmetryRule:
    """四方 4/m: 7个独立常数 C11,C12,C13,C16,C33,C44,C66。"""
    zeros = [
        (0, 3), (0, 4),  # C14=C15=0
        (1, 3), (1, 4),  # C24=C25=0
        (2, 3), (2, 4), (2, 5),  # C34=C35=C36=0
        (3, 4), (3, 5), (4, 5),  # C45=C46=C56=0
    ]

    return SymmetryRule(
        crystal_system=CrystalSystem.TETRAGONAL_4M,
        zero_entries=zeros,
        equivalent_groups=[
            [(0, 0), (1, 1)],       # C11=C22
            [(0, 2), (1, 2)],       # C13=C23
            [(3, 3), (4, 4)],       # C44=C55
        ],
        derived_relations=[
            {"target": (1, 5), "func": lambda c: -c[0, 5]},  # C26=-C16
        ],
    )


def _orthorhombic_rules() -> SymmetryRule:
    """正交: 9个独立常数，正应变-剪切交叉块和剪切非对角全为零。"""
    zeros = []
    for i in range(3):
        for j in range(3, 6):
            zeros.append((i, j))
    zeros.extend([(3, 4), (3, 5), (4, 5)])

    return SymmetryRule(
        crystal_system=CrystalSystem.ORTHORHOMBIC,
        zero_entries=zeros,
        equivalent_groups=[],
        derived_relations=[],
    )


def _monoclinic_rules() -> SymmetryRule:
    """单斜 (unique axis b): 13个独立常数。"""
    # 零项: C14=C16=C24=C26=C34=C36=C45=C56=0
    zeros = [
        (0, 3), (0, 5),  # C14, C16
        (1, 3), (1, 5),  # C24, C26
        (2, 3), (2, 5),  # C34, C36
        (3, 4), (4, 5),  # C45, C56
    ]

    return SymmetryRule(
        crystal_system=CrystalSystem.MONOCLINIC,
        zero_entries=zeros,
        equivalent_groups=[],
        derived_relations=[],
    )


def _triclinic_rules() -> SymmetryRule:
    """三斜: 21个独立常数，无对称约束。"""
    return SymmetryRule(
        crystal_system=CrystalSystem.TRICLINIC,
        zero_entries=[],
        equivalent_groups=[],
        derived_relations=[],
    )


_RULES_MAP = {
    CrystalSystem.CUBIC: _cubic_rules,
    CrystalSystem.HEXAGONAL: _hexagonal_rules,
    CrystalSystem.TRIGONAL_3M: _trigonal_3m_rules,
    CrystalSystem.TRIGONAL_3: _trigonal_3_rules,
    CrystalSystem.TETRAGONAL_4MMM: _tetragonal_4mmm_rules,
    CrystalSystem.TETRAGONAL_4M: _tetragonal_4m_rules,
    CrystalSystem.ORTHORHOMBIC: _orthorhombic_rules,
    CrystalSystem.MONOCLINIC: _monoclinic_rules,
    CrystalSystem.TRICLINIC: _triclinic_rules,
}


def get_symmetry_rules(crystal_system: CrystalSystem) -> SymmetryRule:
    """返回指定晶系的对称性约束规则。"""
    return _RULES_MAP[crystal_system]()
