"""
晶系识别与弹性常数约束关系。

支持7种晶系（含 Laue class 子分类）：
  三斜(Triclinic):          21个独立常数
  单斜(Monoclinic):         13个独立常数
  正交(Orthorhombic):       9个独立常数
  四方(Tetragonal 4/mmm):   6个独立常数
  四方(Tetragonal 4/m):     7个独立常数
  三方(Trigonal -3m):       6个独立常数
  三方(Trigonal -3):        7个独立常数
  六方(Hexagonal):          5个独立常数
  立方(Cubic):              3个独立常数
"""

import warnings
import numpy as np
from enum import Enum
from typing import Tuple


class CrystalSystem(Enum):
    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL_4MMM = "tetragonal_4mmm"   # Laue class 4/mmm, 6个独立常数
    TETRAGONAL_4M = "tetragonal_4m"       # Laue class 4/m, 7个独立常数
    TRIGONAL_3M = "trigonal_3m"           # Laue class -3m, 6个独立常数
    TRIGONAL_3 = "trigonal_3"             # Laue class -3, 7个独立常数
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


# 各晶系独立弹性常数数目
N_INDEPENDENT = {
    CrystalSystem.TRICLINIC: 21,
    CrystalSystem.MONOCLINIC: 13,
    CrystalSystem.ORTHORHOMBIC: 9,
    CrystalSystem.TETRAGONAL_4MMM: 6,
    CrystalSystem.TETRAGONAL_4M: 7,
    CrystalSystem.TRIGONAL_3M: 6,
    CrystalSystem.TRIGONAL_3: 7,
    CrystalSystem.HEXAGONAL: 5,
    CrystalSystem.CUBIC: 3,
}

# 晶系显示名称（中文）
CRYSTAL_SYSTEM_NAMES = {
    CrystalSystem.TRICLINIC: "三斜",
    CrystalSystem.MONOCLINIC: "单斜",
    CrystalSystem.ORTHORHOMBIC: "正交",
    CrystalSystem.TETRAGONAL_4MMM: "四方(4/mmm)",
    CrystalSystem.TETRAGONAL_4M: "四方(4/m)",
    CrystalSystem.TRIGONAL_3M: "三方(-3m)",
    CrystalSystem.TRIGONAL_3: "三方(-3)",
    CrystalSystem.HEXAGONAL: "六方",
    CrystalSystem.CUBIC: "立方",
}

# 晶系显示名称(英文)
CRYSTAL_SYSTEM_NAMES_EN = {
    CrystalSystem.TRICLINIC: "Triclinic",
    CrystalSystem.MONOCLINIC: "Monoclinic",
    CrystalSystem.ORTHORHOMBIC: "Orthorhombic",
    CrystalSystem.TETRAGONAL_4MMM: "Tetragonal (4/mmm)",
    CrystalSystem.TETRAGONAL_4M: "Tetragonal (4/m)",
    CrystalSystem.TRIGONAL_3M: "Trigonal (-3m)",
    CrystalSystem.TRIGONAL_3: "Trigonal (-3)",
    CrystalSystem.HEXAGONAL: "Hexagonal",
    CrystalSystem.CUBIC: "Cubic",
}


def cs_name(cs) -> str:
    """按当前界面语言返回晶系显示名(zh/en)。"""
    try:
        from .i18n import get_lang
        if get_lang() == "en":
            return CRYSTAL_SYSTEM_NAMES_EN.get(cs, getattr(cs, "value", str(cs)))
    except Exception:
        pass
    return CRYSTAL_SYSTEM_NAMES.get(cs, getattr(cs, "value", str(cs)))


def lattice_params(lattice: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    从3×3晶格矩阵提取 (a, b, c, α, β, γ)，角度为度数。

    Parameters
    ----------
    lattice : (3, 3) 晶格矢量矩阵，每行一个矢量
    """
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    alpha = np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1, 1)))
    beta = np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1, 1)))
    gamma = np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1, 1)))

    return a, b, c, alpha, beta, gamma


def space_group_to_crystal_system(sg: int) -> CrystalSystem:
    """
    由空间群号 (1-230) 返回晶系 + 正确的 Laue class。

    这是确定晶系最可靠的方式 (优于从带噪声的晶格几何猜)。四方/三方按
    Laue class 细分, 直接决定独立弹性常数数目和 C16/C15 等是否为零:
      - 四方 75-88  (4, -4, 4/m)          → 4/m   (7 个常数, 允许 C16)
      - 四方 89-142 (422,4mm,-42m,4/mmm)  → 4/mmm (6 个常数, C16=0)   ← P4₁2₁2(#92)
      - 三方 143-148 (3, -3)              → -3    (7 个常数)
      - 三方 149-167 (32, 3m, -3m)        → -3m   (6 个常数)
    """
    if not isinstance(sg, int) or not (1 <= sg <= 230):
        raise ValueError(f"空间群号必须是 1-230 的整数, 收到 {sg!r}")
    if sg <= 2:
        return CrystalSystem.TRICLINIC
    if sg <= 15:
        return CrystalSystem.MONOCLINIC
    if sg <= 74:
        return CrystalSystem.ORTHORHOMBIC
    if sg <= 88:
        return CrystalSystem.TETRAGONAL_4M
    if sg <= 142:
        return CrystalSystem.TETRAGONAL_4MMM
    if sg <= 148:
        return CrystalSystem.TRIGONAL_3
    if sg <= 167:
        return CrystalSystem.TRIGONAL_3M
    if sg <= 194:
        return CrystalSystem.HEXAGONAL
    return CrystalSystem.CUBIC


def spacegroup_number_from_structure(poscar: dict, symprec: float = 1e-3):
    """用 spglib 从结构 (原子位置) 识别空间群号 (1-230)。

    这比只看晶格几何 (a/b/c/角度) 可靠得多: 用原子位置 + 对称操作,
    能区分 Laue class (4/mmm vs 4/m), 且不被 MD 噪声骗 (立方不会被当正交)。
    spglib 未装 / 结构信息不全 / 识别失败时返回 None。
    """
    if not poscar:
        return None
    try:
        from .periodic import get_spacegroup
    except Exception:
        return None
    spg_str = get_spacegroup(
        poscar.get("lattice"), poscar.get("positions"),
        poscar.get("coord_type", "Direct"),
        poscar.get("species"), poscar.get("counts"),
        symprec=symprec,
    )
    if not spg_str:
        return None
    import re
    m = re.search(r"\((\d+)\)", spg_str)  # 例如 'Fd-3m (227)' -> 227
    return int(m.group(1)) if m else None


def crystal_system_from_structure(poscar: dict, symprec: float = 1e-3):
    """用 spglib 从结构直接得晶系 (含正确 Laue class)。失败返回 None。"""
    sg = spacegroup_number_from_structure(poscar, symprec)
    return space_group_to_crystal_system(sg) if sg is not None else None


def resolve_crystal_system(
    lattice: np.ndarray = None,
    crystal_system=None,
    space_group: int = None,
    poscar: dict = None,
    tol: float = 0.01,
    symprec: float = 1e-3,
) -> CrystalSystem:
    """
    按优先级确定晶系, 供全流程统一调用 (单一事实来源):
      1. 显式 crystal_system (用户覆盖, 枚举或字符串)
      2. 显式 space_group 号
      3. spglib 从结构 (poscar 原子位置) 自动识别 —— 推荐默认, 无需用户输入
      4. 从 lattice 几何自动识别 (最后兜底, 分不清 Laue class 且怕噪声)
    """
    if crystal_system is not None:
        if isinstance(crystal_system, CrystalSystem):
            return crystal_system
        for cs in CrystalSystem:
            if cs.value == str(crystal_system):
                return cs
        raise ValueError(f"无效晶系字符串: {crystal_system!r}")
    if space_group is not None:
        return space_group_to_crystal_system(int(space_group))
    if poscar is not None:
        cs = crystal_system_from_structure(poscar, symprec)
        if cs is not None:
            return cs
    if lattice is not None:
        return identify_crystal_system(lattice, tol)
    raise ValueError("resolve_crystal_system: 需提供 crystal_system / space_group / poscar / lattice 之一")


def identify_crystal_system(lattice: np.ndarray, tol: float = 0.01) -> CrystalSystem:
    """
    根据晶格矩阵识别晶系。

    注意：此函数基于晶格参数几何关系判断晶系，仅适用于原胞（primitive cell）。
    超胞的晶格参数可能与原胞不同（如 2×2×1 立方超胞会被误判为四方），
    因此必须使用原胞晶格进行识别。如果输入为超胞，请先提供原胞 POSCAR
    或手动指定晶系。

    Parameters
    ----------
    lattice : (3, 3) 晶格矢量矩阵，每行一个矢量（应为原胞晶格）
    tol : 相对容差，用于判断晶格参数相等

    Returns
    -------
    CrystalSystem 枚举值

    Notes
    -----
    - 四方和三方晶系默认返回保守的 Laue class（更多独立常数）
    - 边界情况回退至最低对称性（三斜）并发出警告
    """
    a, b, c, alpha, beta, gamma = lattice_params(lattice)

    def eq(x, y):
        return abs(x - y) / max(abs(x), abs(y), 1e-10) < tol

    # 立方: a=b=c, α=β=γ=90°
    if eq(a, b) and eq(b, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
        return CrystalSystem.CUBIC

    # 六方: a=b≠c, α=β=90°, γ=120°
    if eq(a, b) and not eq(a, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 120):
        return CrystalSystem.HEXAGONAL

    # 三方(菱方设置): a=b=c, α=β=γ≠90°
    if eq(a, b) and eq(b, c) and eq(alpha, beta) and eq(beta, gamma) and not eq(alpha, 90):
        return CrystalSystem.TRIGONAL_3  # 保守取 -3 (7个独立常数)

    # 四方: a=b≠c, α=β=γ=90°
    if eq(a, b) and not eq(a, c) and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
        return CrystalSystem.TETRAGONAL_4M  # 保守取 4/m (7个独立常数)

    # 正交: a≠b≠c, α=β=γ=90°
    if eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90):
        return CrystalSystem.ORTHORHOMBIC

    # 单斜: α=γ=90°, β≠90° (标准设置 unique axis b)
    if eq(alpha, 90) and eq(gamma, 90) and not eq(beta, 90):
        return CrystalSystem.MONOCLINIC

    # 默认三斜，发出警告
    warnings.warn(
        f"晶格参数 (a={a:.4f}, b={b:.4f}, c={c:.4f}, "
        f"α={alpha:.2f}°, β={beta:.2f}°, γ={gamma:.2f}°) "
        f"无法明确归类，回退至三斜晶系。",
        UserWarning,
    )
    return CrystalSystem.TRICLINIC


def detect_supercell_mismatch(
    supercell_lattice: np.ndarray,
    primitive_lattice: np.ndarray,
    tol: float = 0.01,
) -> bool:
    """
    检测超胞与原胞的晶系识别是否一致。

    如果超胞的自动识别结果与原胞不同，说明超胞扩展改变了
    晶格参数的对称关系，此时必须以原胞的识别结果为准。

    Parameters
    ----------
    supercell_lattice : (3, 3) 超胞晶格矩阵
    primitive_lattice : (3, 3) 原胞晶格矩阵
    tol : 相对容差

    Returns
    -------
    bool : True 表示存在不一致（超胞识别结果与原胞不同）
    """
    cs_super = identify_crystal_system(supercell_lattice, tol)
    cs_prim = identify_crystal_system(primitive_lattice, tol)
    return cs_super != cs_prim


def enforce_lattice_symmetry(
    lattice: np.ndarray,
    crystal_system: CrystalSystem = None,
    tol: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """
    根据晶系对称性约束晶格矩阵。

    NPT AIMD 的时间平均晶格或 CONTCAR 可能有微小的对称性破缺，
    需要根据晶系强制恢复对称性。

    Parameters
    ----------
    lattice : (3, 3) 晶格矩阵
    crystal_system : 指定晶系，None 则自动识别
    tol : 晶系识别容差
    verbose : 是否打印约束信息

    Returns
    -------
    (3, 3) 对称性约束后的晶格矩阵

    规则:
    - 立方: a=b=c=(a'+b'+c')/3, α=β=γ=90° → 对角矩阵
    - 六方: a=b=(a'+b')/2, c=c', α=β=90°, γ=120°
    - 四方: a=b=(a'+b')/2, c=c', α=β=γ=90° → 对角矩阵
    - 正交: a, b, c 保持, α=β=γ=90° → 对角矩阵
    - 三方(菱方): a=b=c=(a'+b'+c')/3, α=β=γ=(α'+β'+γ')/3
    - 单斜/三斜: 不做约束
    """
    if crystal_system is None:
        crystal_system = identify_crystal_system(lattice, tol)

    a_val, b_val, c_val, alpha, beta, gamma = lattice_params(lattice)
    cs_name = CRYSTAL_SYSTEM_NAMES.get(crystal_system, crystal_system.value)

    if verbose:
        print(f"    Raw:  a={a_val:.6f}  b={b_val:.6f}  c={c_val:.6f}  "
              f"alpha={alpha:.4f}  beta={beta:.4f}  gamma={gamma:.4f}")

    if crystal_system == CrystalSystem.CUBIC:
        avg_abc = (a_val + b_val + c_val) / 3.0
        result = np.diag([avg_abc, avg_abc, avg_abc])
        if verbose:
            print(f"    Cubic: a=b=c = ({a_val:.6f}+{b_val:.6f}+{c_val:.6f})/3 = {avg_abc:.6f}")

    elif crystal_system == CrystalSystem.HEXAGONAL:
        avg_ab = (a_val + b_val) / 2.0
        result = np.array([
            [avg_ab, 0.0, 0.0],
            [-avg_ab / 2.0, avg_ab * np.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, c_val],
        ])
        if verbose:
            print(f"    Hexagonal: a=b = ({a_val:.6f}+{b_val:.6f})/2 = {avg_ab:.6f}, c={c_val:.6f}")

    elif crystal_system in (CrystalSystem.TETRAGONAL_4MMM, CrystalSystem.TETRAGONAL_4M):
        avg_ab = (a_val + b_val) / 2.0
        result = np.diag([avg_ab, avg_ab, c_val])
        if verbose:
            print(f"    Tetragonal: a=b = ({a_val:.6f}+{b_val:.6f})/2 = {avg_ab:.6f}, c={c_val:.6f}")

    elif crystal_system == CrystalSystem.ORTHORHOMBIC:
        result = np.diag([a_val, b_val, c_val])
        if verbose:
            print(f"    Orthorhombic: a={a_val:.6f}, b={b_val:.6f}, c={c_val:.6f} (angles -> 90)")

    elif crystal_system in (CrystalSystem.TRIGONAL_3M, CrystalSystem.TRIGONAL_3):
        avg_abc = (a_val + b_val + c_val) / 3.0
        avg_angle = (alpha + beta + gamma) / 3.0
        cos_a = np.cos(np.radians(avg_angle))
        sin_a = np.sin(np.radians(avg_angle))
        cx = avg_abc * cos_a
        cy = avg_abc * (cos_a - cos_a**2) / sin_a if sin_a > 1e-10 else 0.0
        cz = avg_abc * np.sqrt(max(0, 1 - cx**2/avg_abc**2 - cy**2/avg_abc**2)) if avg_abc > 0 else 0.0
        result = np.array([
            [avg_abc, 0.0, 0.0],
            [avg_abc * cos_a, avg_abc * sin_a, 0.0],
            [cx, cy, cz],
        ])
        if verbose:
            print(f"    Trigonal: a=b=c = ({a_val:.6f}+{b_val:.6f}+{c_val:.6f})/3 = {avg_abc:.6f}")
            print(f"             alpha=beta=gamma = ({alpha:.4f}+{beta:.4f}+{gamma:.4f})/3 = {avg_angle:.4f}")

    elif crystal_system == CrystalSystem.MONOCLINIC:
        result = np.array([
            [a_val, 0.0, 0.0],
            [0.0, b_val, 0.0],
            [c_val * np.cos(np.radians(beta)), 0.0, c_val * np.sin(np.radians(beta))],
        ])
        if verbose:
            print(f"    Monoclinic: a={a_val:.6f}, b={b_val:.6f}, c={c_val:.6f}, beta={beta:.4f}")

    else:
        result = lattice.copy()
        if verbose:
            print(f"    Triclinic: no symmetry constraint applied")

    if verbose:
        a2, b2, c2, al2, be2, ga2 = lattice_params(result)
        print(f"    Result: a={a2:.6f}  b={b2:.6f}  c={c2:.6f}  "
              f"V={abs(np.linalg.det(result)):.4f}")

    return result
