
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
# 强制使用非交互后端 Agg，保证在无显示的集群/服务器上也能正常出图保存
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 设置matplotlib字体 - 兼容Linux环境
import matplotlib.font_manager as fm

# 检测可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 按优先级选择字体
font_candidates = [
    'DejaVu Sans',      # Linux常用
    'Liberation Sans',  # Linux常用  
    'Arial',           # Windows/Mac
    'Helvetica',       # Mac
    'Times New Roman', # Windows
    'serif'            # 通用后备
]

selected_font = 'serif'  # 默认后备字体
for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

plt.rcParams['font.family'] = selected_font
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 弹性性质名称映射
PROPERTY_NAMES = {
    'E': 'Young\'s Modulus',
    'G': 'Shear Modulus', 
    'nu': 'Poisson\'s Ratio',
    'L': 'Linear Compressibility',
    'B': 'Bulk Modulus'
}

PROPERTY_UNITS = {
    'E': 'GPa',
    'G': 'GPa',
    'nu': '',
    'L': 'GPa^-1',
    'B': 'GPa'
}

# -------------------------
# Data classes
# -------------------------
@dataclass
class SymmetryResult:
    system: str
    criteria: Dict[str, Tuple[bool, str]]  # criterion_name -> (pass?, detail_string)

@dataclass
class StabilityResult:
    stable: bool
    criteria: Dict[str, Tuple[bool, str]]

# -------------------------
# IO utilities
# -------------------------

def read_matrix(path: str) -> np.ndarray:
    """Read a 6x6 stiffness matrix C (GPa) from a local .txt file (Voigt, engineering shear)."""
    M = np.loadtxt(path)
    if M.shape != (6, 6):
        raise ValueError("Input matrix must be 6x6 in Voigt notation (GPa).")
    return M.astype(float)

# -------------------------
# Symmetry identification
# -------------------------

def _approx_zero(x: float, tol: float) -> bool:
    return abs(x) <= tol

def _approx_equal(a: float, b: float, tol: float) -> bool:
    return np.isclose(a, b, atol=tol)


def identify_symmetry(C: np.ndarray, tol: float = 1e-4) -> SymmetryResult:
    """Identify crystal system by testing Voigt stiffness matrix equalities.
    Returns the matched system and a dict of equality criteria with pass/fail and details.
    Systems tested in order: cubic, hexagonal, tetragonal, trigonal(rhombohedral), orthorhombic, monoclinic, triclinic.
    """
    systems_ordered = ["Cubic", "Hexagonal", "Tetragonal", "Trigonal", "Orthorhombic", "Monoclinic", "Triclinic"]

    def check_system(C_matrix, system_name, checks_list, tolerance):
        system_passed = True
        system_specific_criteria = {}
        for check_tuple in checks_list:
            check_type = check_tuple[0]
            args = check_tuple[1:]
            if check_type == "eq":
                val1, val2, name = args
                ok = _approx_equal(val1, val2, tolerance)
                system_specific_criteria[name] = (ok, f"|{val1:.3e} - {val2:.3e}| = {abs(val1-val2):.3e}, tol={tolerance}")
            elif check_type == "is_zero":
                val, name = args
                ok = _approx_zero(val, tolerance)
                system_specific_criteria[name] = (ok, f"|{val:.3e}| = {abs(val):.3e}, tol={tolerance}")
            else:
                raise ValueError("Unknown check type")
            if not ok:
                system_passed = False
        return system_passed, system_specific_criteria

    # Define checks for each symmetry system using a new format
    cubic_checks_list = [
        ("eq", C[0,0], C[1,1], "C11 = C22"), ("eq", C[0,0], C[2,2], "C11 = C33"),
        ("eq", C[0,1], C[0,2], "C12 = C13"), ("eq", C[0,1], C[1,2], "C12 = C23"),
        ("eq", C[3,3], C[4,4], "C44 = C55"), ("eq", C[3,3], C[5,5], "C44 = C66"),
        ("is_zero", C[0,3], "C14 = 0"), ("is_zero", C[0,4], "C15 = 0"), ("is_zero", C[0,5], "C16 = 0"),
        ("is_zero", C[1,3], "C24 = 0"), ("is_zero", C[1,4], "C25 = 0"), ("is_zero", C[1,5], "C26 = 0"),
        ("is_zero", C[2,3], "C34 = 0"), ("is_zero", C[2,4], "C35 = 0"), ("is_zero", C[2,5], "C36 = 0"),
        ("is_zero", C[3,4], "C45 = 0"), ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    hex_checks_list = [
        ("eq", C[0,0], C[1,1], "C11 = C22"), ("eq", C[0,2], C[1,2], "C13 = C23"),
        ("eq", C[3,3], C[4,4], "C44 = C55"), ("eq", C[5,5], 0.5*(C[0,0]-C[0,1]), "C66 = (C11-C12)/2"),
        ("is_zero", C[0,3], "C14 = 0"), ("is_zero", C[0,4], "C15 = 0"), ("is_zero", C[0,5], "C16 = 0"),
        ("is_zero", C[1,3], "C24 = 0"), ("is_zero", C[1,4], "C25 = 0"), ("is_zero", C[1,5], "C26 = 0"),
        ("is_zero", C[2,3], "C34 = 0"), ("is_zero", C[2,4], "C35 = 0"), ("is_zero", C[2,5], "C36 = 0"),
        ("is_zero", C[3,4], "C45 = 0"), ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    tet_checks_list = [
        ("eq", C[0,0], C[1,1], "C11 = C22"), ("eq", C[0,2], C[1,2], "C13 = C23"),
        ("eq", C[3,3], C[4,4], "C44 = C55"),
        ("is_zero", C[0,3], "C14 = 0"), ("is_zero", C[0,4], "C15 = 0"), ("is_zero", C[0,5], "C16 = 0"),
        ("is_zero", C[1,3], "C24 = 0"), ("is_zero", C[1,4], "C25 = 0"), ("is_zero", C[1,5], "C26 = 0"),
        ("is_zero", C[2,3], "C34 = 0"), ("is_zero", C[2,4], "C35 = 0"), ("is_zero", C[2,5], "C36 = 0"),
        ("is_zero", C[3,4], "C45 = 0"), ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    tri_checks_list = [
        ("eq", C[0,0], C[1,1], "C11 = C22"), ("eq", C[0,2], C[1,2], "C13 = C23"),
        ("eq", C[3,3], C[4,4], "C44 = C55"), ("eq", C[5,5], 0.5*(C[0,0]-C[0,1]), "C66 = (C11-C12)/2"),
        ("eq", C[0,3], -C[1,3], "C14 = -C24"),
        ("eq", C[0,4], -C[1,4], "C15 = -C25"),
        ("is_zero", C[0,5], "C16 = 0"), ("is_zero", C[1,5], "C26 = 0"), ("is_zero", C[2,3], "C34 = 0"),
        ("is_zero", C[2,4], "C35 = 0"), ("is_zero", C[2,5], "C36 = 0"), ("is_zero", C[3,4], "C45 = 0"),
        ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    ortho_checks_list = [
        ("is_zero", C[0,3], "C14 = 0"), ("is_zero", C[0,4], "C15 = 0"), ("is_zero", C[0,5], "C16 = 0"),
        ("is_zero", C[1,3], "C24 = 0"), ("is_zero", C[1,4], "C25 = 0"), ("is_zero", C[1,5], "C26 = 0"),
        ("is_zero", C[2,3], "C34 = 0"), ("is_zero", C[2,4], "C35 = 0"), ("is_zero", C[2,5], "C36 = 0"),
        ("is_zero", C[3,4], "C45 = 0"), ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    mono_checks_list = [
        ("is_zero", C[0,3], "C14 = 0"), ("is_zero", C[0,5], "C16 = 0"),
        ("is_zero", C[1,3], "C24 = 0"), ("is_zero", C[1,5], "C26 = 0"),
        ("is_zero", C[2,3], "C34 = 0"), ("is_zero", C[2,5], "C36 = 0"),
        ("is_zero", C[3,5], "C46 = 0"), ("is_zero", C[4,5], "C56 = 0"),
    ]

    system_map = {
        "Cubic": cubic_checks_list,
        "Hexagonal": hex_checks_list,
        "Tetragonal": tet_checks_list,
        "Trigonal": tri_checks_list,
        "Orthorhombic": ortho_checks_list,
        "Monoclinic": mono_checks_list,
        "Triclinic": []
    }

    identified_system = "Triclinic"
    best_match_criteria = {}

    for sys_name in systems_ordered:
        passed, criteria = check_system(C, sys_name, system_map[sys_name], tol)
        if passed:
            identified_system = sys_name
            best_match_criteria = criteria
            break # Found the highest symmetry system

    return SymmetryResult(system=identified_system, criteria=best_match_criteria)


def force_symmetry(C: np.ndarray, system: str, tol: float = 1e-4) -> np.ndarray:
    """Forces the input matrix C to conform to the specified crystal system symmetry.
    This involves setting near-zero elements to zero and averaging elements that should be equal.
    """
    C_forced = C.copy()

    def set_zero(row, col):
        if _approx_zero(C_forced[row, col], tol):
            C_forced[row, col] = 0.0
            C_forced[col, row] = 0.0 # Symmetric matrix

    def set_equal(indices):
        values = [C_forced[r, c] for r, c in indices]
        avg = np.mean(values)
        for r, c in indices:
            C_forced[r, c] = avg
            C_forced[c, r] = avg # Symmetric matrix

    s = system.lower()

    if s == "cubic":
        set_equal([(0,0), (1,1), (2,2)])
        set_equal([(0,1), (0,2), (1,2)])
        set_equal([(3,3), (4,4), (5,5)])
        for i in range(3):
            for j in range(3,6):
                set_zero(i,j)
        for i in range(3,6):
            for j in range(i+1,6):
                set_zero(i,j)

    elif s == "hexagonal":
        set_equal([(0,0), (1,1)])
        set_equal([(0,2), (1,2)])
        set_equal([(3,3), (4,4)])
        # C66 = (C11-C12)/2
        C_forced[5,5] = 0.5*(C_forced[0,0]-C_forced[0,1])
        C_forced[5,5] = 0.5*(C_forced[0,0]-C_forced[0,1]) # Ensure symmetry
        for i in range(3):
            for j in range(3,6):
                set_zero(i,j)
        for i in range(3,6):
            for j in range(i+1,6):
                set_zero(i,j)

    elif s == "tetragonal":
        set_equal([(0,0), (1,1)])
        set_equal([(0,2), (1,2)])
        set_equal([(3,3), (4,4)])
        for i in range(3):
            for j in range(3,6):
                set_zero(i,j)
        for i in range(3,6):
            for j in range(i+1,6):
                set_zero(i,j)

    elif s == "trigonal":
        set_equal([(0,0), (1,1)])
        set_equal([(0,2), (1,2)])
        set_equal([(3,3), (4,4)])
        # C66 = (C11-C12)/2
        C_forced[5,5] = 0.5*(C_forced[0,0]-C_forced[0,1])
        C_forced[5,5] = 0.5*(C_forced[0,0]-C_forced[0,1]) # Ensure symmetry
        # C14 = -C24
        avg_c14_c24 = (C_forced[0,3] - C_forced[1,3]) / 2
        C_forced[0,3] = avg_c14_c24
        C_forced[3,0] = avg_c14_c24
        C_forced[1,3] = -avg_c14_c24
        C_forced[3,1] = -avg_c14_c24
        # C15 = -C25
        avg_c15_c25 = (C_forced[0,4] - C_forced[1,4]) / 2
        C_forced[0,4] = avg_c15_c25
        C_forced[4,0] = avg_c15_c25
        C_forced[1,4] = -avg_c15_c25
        C_forced[4,1] = -avg_c15_c25

        # Set other off-diagonals to zero as per trigonal symmetry
        set_zero(0,5); set_zero(1,5); set_zero(2,3); set_zero(2,4); set_zero(2,5);
        set_zero(3,4); set_zero(3,5); set_zero(4,5);

    elif s == "orthorhombic":
        for i in range(3):
            for j in range(3,6):
                set_zero(i,j)
        for i in range(3,6):
            for j in range(i+1,6):
                set_zero(i,j)

    elif s == "monoclinic":
        # Assuming standard setting (2-fold axis along y, or normal to xz plane)
        set_zero(0,3); set_zero(0,5);
        set_zero(1,3); set_zero(1,5);
        set_zero(2,3); set_zero(2,5);
        set_zero(3,5); set_zero(4,5);

    # Triclinic: No elements are forced to zero or equal

    return C_forced

# -------------------------
# Mechanical stability (Born criteria)
# -------------------------

def stability_criteria(C: np.ndarray, system: str, tol: float = 1e-6) -> StabilityResult:
    """Evaluate mechanical stability criteria for the given stiffness matrix C and crystal system.
    Criteria are based on Born stability conditions.
    """
    stable = True
    crit: Dict[str, Tuple[bool, str]] = {}

    s = system.lower()

    # General criteria for all systems (positive definite)
    try:
        np.linalg.cholesky(C) # Cholesky decomposition requires positive definite matrix
        crit["Positive Definite"] = (True, "Cholesky decomposition successful.")
    except np.linalg.LinAlgError:
        crit["Positive Definite"] = (False, "Matrix is not positive definite (Cholesky decomposition failed).")
        stable = False

    if s == "cubic":
        c11 = C[0,0]; c12 = C[0,1]; c44 = C[3,3]
        crit["C11 > 0"] = (c11 > 0, f"C11 = {c11:.3f} > 0")
        crit["C44 > 0"] = (c44 > 0, f"C44 = {c44:.3f} > 0")
        crit["C11 - C12 > 0"] = (c11 - c12 > 0, f"C11 - C12 = {c11-c12:.3f} > 0")
        crit["C11 + 2C12 > 0"] = (c11 + 2*c12 > 0, f"C11 + 2C12 = {c11+2*c12:.3f} > 0")
        if not all([crit["C11 > 0"][0], crit["C44 > 0"][0], crit["C11 - C12 > 0"][0], crit["C11 + 2C12 > 0"][0]]):
            stable = False

    elif s == "hexagonal":
        c11 = C[0,0]; c12 = C[0,1]; c13 = C[0,2]; c33 = C[2,2]; c44 = C[3,3]
        crit["C11 > |C12|"] = (c11 > abs(c12), f"C11 = {c11:.3f}, |C12| = {abs(c12):.3f}")
        crit["C33 > 0"] = (c33 > 0, f"C33 = {c33:.3f} > 0")
        crit["C44 > 0"] = (c44 > 0, f"C44 = {c44:.3f} > 0")
        crit["C11 + C12 + 2C13 > 0"] = (c11 + c12 + 2*c13 > 0, f"C11 + C12 + 2C13 = {c11+c12+2*c13:.3f} > 0")
        crit["(C11+C12)C33 - 2C13^2 > 0"] = ((c11+c12)*c33 - 2*c13**2 > 0, f"(C11+C12)C33 - 2C13^2 = {(c11+c12)*c33 - 2*c13**2:.3f} > 0")
        if not all([crit["C11 > |C12|"][0], crit["C33 > 0"][0], crit["C44 > 0"][0], crit["C11 + C12 + 2C13 > 0"][0], crit["(C11+C12)C33 - 2C13^2 > 0"][0]]):
            stable = False

    elif s == "tetragonal":
        c11 = C[0,0]; c12 = C[0,1]; c13 = C[0,2]; c33 = C[2,2]; c44 = C[3,3]; c66 = C[5,5]
        crit["C11 > |C12|"] = (c11 > abs(c12), f"C11 = {c11:.3f}, |C12| = {abs(c12):.3f}")
        crit["C33 > 0"] = (c33 > 0, f"C33 = {c33:.3f} > 0")
        crit["C44 > 0"] = (c44 > 0, f"C44 = {c44:.3f} > 0")
        crit["C66 > 0"] = (c66 > 0, f"C66 = {c66:.3f} > 0")
        crit["(C11+C12)C33 - 2C13^2 > 0"] = ((c11+c12)*c33 - 2*c13**2 > 0, f"(C11+C12)C33 - 2C13^2 = {(c11+c12)*c33 - 2*c13**2:.3f} > 0")
        if not all([crit["C11 > |C12|"][0], crit["C33 > 0"][0], crit["C44 > 0"][0], crit["C66 > 0"][0], crit["(C11+C12)C33 - 2C13^2 > 0"][0]]):
            stable = False

    elif s == "trigonal": # Same as hexagonal for stability criteria
        c11 = C[0,0]; c12 = C[0,1]; c13 = C[0,2]; c33 = C[2,2]; c44 = C[3,3]
        crit["C11 > |C12|"] = (c11 > abs(c12), f"C11 = {c11:.3f}, |C12| = {abs(c12):.3f}")
        crit["C33 > 0"] = (c33 > 0, f"C33 = {c33:.3f} > 0")
        crit["C44 > 0"] = (c44 > 0, f"C44 = {c44:.3f} > 0")
        crit["C11 + C12 + 2C13 > 0"] = (c11 + c12 + 2*c13 > 0, f"C11 + C12 + 2C13 = {c11+c12+2*c13:.3f} > 0")
        crit["(C11+C12)C33 - 2C13^2 > 0"] = ((c11+c12)*c33 - 2*c13**2 > 0, f"(C11+C12)C33 - 2C13^2 = {(c11+c12)*c33 - 2*c13**2:.3f} > 0")
        if not all([crit["C11 > |C12|"][0], crit["C33 > 0"][0], crit["C44 > 0"][0], crit["C11 + C12 + 2C13 > 0"][0], crit["(C11+C12)C33 - 2C13^2 > 0"][0]]):
            stable = False

    elif s == "orthorhombic":
        c11 = C[0,0]; c22 = C[1,1]; c33 = C[2,2]; c12 = C[0,1]; c13 = C[0,2]; c23 = C[1,2];
        c44 = C[3,3]; c55 = C[4,4]; c66 = C[5,5]
        crit["C11 > 0"] = (c11 > 0, f"C11 = {c11:.3f} > 0")
        crit["C22 > 0"] = (c22 > 0, f"C22 = {c22:.3f} > 0")
        crit["C33 > 0"] = (c33 > 0, f"C33 = {c33:.3f} > 0")
        crit["C44 > 0"] = (c44 > 0, f"C44 = {c44:.3f} > 0")
        crit["C55 > 0"] = (c55 > 0, f"C55 = {c55:.3f} > 0")
        crit["C66 > 0"] = (c66 > 0, f"C66 = {c66:.3f} > 0")
        crit["C11+C22-2C12 > 0"] = (c11+c22-2*c12 > 0, f"C11+C22-2C12 = {c11+c22-2*c12:.3f} > 0")
        crit["C11+C33-2C13 > 0"] = (c11+c33-2*c13 > 0, f"C11+C33-2C13 = {c11+c33-2*c13:.3f} > 0")
        crit["C22+C33-2C23 > 0"] = (c22+c33-2*c23 > 0, f"C22+C33-2C23 = {c22+c33-2*c23:.3f} > 0")
        crit["C11+C22+C33+2(C12+C13+C23) > 0"] = (c11+c22+c33+2*(c12+c13+c23) > 0, f"C11+C22+C33+2(C12+C13+C23) = {c11+c22+c33+2*(c12+c13+c23):.3f} > 0")
        if not all([crit["C11 > 0"][0], crit["C22 > 0"][0], crit["C33 > 0"][0], crit["C44 > 0"][0], crit["C55 > 0"][0], crit["C66 > 0"][0],
                    crit["C11+C22-2C12 > 0"][0], crit["C11+C33-2C13 > 0"][0], crit["C22+C33-2C23 > 0"][0],
                    crit["C11+C22+C33+2(C12+C13+C23) > 0"][0]]):
            stable = False

    elif s == "monoclinic":
        # General conditions for monoclinic (assuming standard setting)
        # All diagonal elements must be positive
        for i in range(6):
            crit[f"C{i+1}{i+1} > 0"] = (C[i,i] > 0, f"C{i+1}{i+1} = {C[i,i]:.3f} > 0")
            if not crit[f"C{i+1}{i+1} > 0"][0]: stable = False

        # Determinants of principal minors must be positive
        # 2x2 minors
        crit["C11C22 - C12^2 > 0"] = (C[0,0]*C[1,1] - C[0,1]**2 > 0, f"C11C22 - C12^2 = {C[0,0]*C[1,1] - C[0,1]**2:.3f} > 0")
        crit["C11C33 - C13^2 > 0"] = (C[0,0]*C[2,2] - C[0,2]**2 > 0, f"C11C33 - C13^2 = {C[0,0]*C[2,2] - C[0,2]**2:.3f} > 0")
        crit["C22C33 - C23^2 > 0"] = (C[1,1]*C[2,2] - C[1,2]**2 > 0, f"C22C33 - C23^2 = {C[1,1]*C[2,2] - C[1,2]**2:.3f} > 0")
        crit["C44C55 - C45^2 > 0"] = (C[3,3]*C[4,4] - C[3,4]**2 > 0, f"C44C55 - C45^2 = {C[3,3]*C[4,4] - C[3,4]**2:.3f} > 0")
        
        if not all([crit["C11C22 - C12^2 > 0"][0], crit["C11C33 - C13^2 > 0"][0], crit["C22C33 - C23^2 > 0"][0], crit["C44C55 - C45^2 > 0"][0]]):
            stable = False

        # 3x3 minors (simplified for standard setting)
        det3x3_upper = np.linalg.det(C[0:3, 0:3])
        crit["det(C_upper_3x3) > 0"] = (det3x3_upper > 0, f"det(C_upper_3x3) = {det3x3_upper:.3f} > 0")
        if not crit["det(C_upper_3x3) > 0"][0]: stable = False

        # C66 > 0
        crit["C66 > 0"] = (C[5,5] > 0, f"C66 = {C[5,5]:.3f} > 0")
        if not crit["C66 > 0"][0]: stable = False

    elif s == "triclinic":
        # For triclinic, the only general condition is that the matrix must be positive definite.
        # This is already checked by the Cholesky decomposition at the beginning.
        pass

    return StabilityResult(stable=stable, criteria=crit)

# -------------------------
# Compliance matrix S
# -------------------------

def compliance_matrix(C: np.ndarray) -> np.ndarray:
    """Calculate the compliance matrix S from the stiffness matrix C."""
    return np.linalg.inv(C)

# -------------------------
# Voigt-Reuss-Hill (VRH) approximation
# -------------------------

def bulk_modulus_voigt(C: np.ndarray) -> float:
    """Calculate bulk modulus (Voigt approximation)."""
    c11, c12, c13, c22, c23, c33, c44, c55, c66 = C[0,0], C[0,1], C[0,2], C[1,1], C[1,2], C[2,2], C[3,3], C[4,4], C[5,5]
    return (c11 + c22 + c33 + 2*(c12 + c13 + c23)) / 9.0

def bulk_modulus_reuss(S: np.ndarray) -> float:
    """Calculate bulk modulus (Reuss approximation)."""
    s11, s12, s13, s22, s23, s33, s44, s55, s66 = S[0,0], S[0,1], S[0,2], S[1,1], S[1,2], S[2,2], S[3,3], S[4,4], S[5,5]
    return 1.0 / ((s11 + s22 + s33) + 2*(s12 + s13 + s23))

def bulk_modulus_vrh(C: np.ndarray, S: np.ndarray) -> float:
    """Calculate bulk modulus (Voigt-Reuss-Hill approximation)."""
    return 0.5 * (bulk_modulus_voigt(C) + bulk_modulus_reuss(S))

def shear_modulus_voigt(C: np.ndarray) -> float:
    """Calculate shear modulus (Voigt approximation)."""
    c11, c12, c13, c22, c23, c33, c44, c55, c66 = C[0,0], C[0,1], C[0,2], C[1,1], C[1,2], C[2,2], C[3,3], C[4,4], C[5,5]
    return (c11 + c22 + c33 - c12 - c13 - c23 + 3*(c44 + c55 + c66)) / 15.0

def shear_modulus_reuss(S: np.ndarray) -> float:
    """Calculate shear modulus (Reuss approximation)."""
    s11, s12, s13, s22, s23, s33, s44, s55, s66 = S[0,0], S[0,1], S[0,2], S[1,1], S[1,2], S[2,2], S[3,3], S[4,4], S[5,5]
    return 15.0 / (4*(s11 + s22 + s33) - 4*(s12 + s13 + s23) + 3*(s44 + s55 + s66))

def shear_modulus_vrh(C: np.ndarray, S: np.ndarray) -> float:
    """Calculate shear modulus (Voigt-Reuss-Hill approximation)."""
    return 0.5 * (shear_modulus_voigt(C) + shear_modulus_reuss(S))

# -------------------------
# Directional properties
# -------------------------

def youngs_modulus_directional(S: np.ndarray, n: np.ndarray) -> float:
    """Calculate Young's modulus in a given direction n using correct tensor notation."""
    if n.shape != (3,):
        raise ValueError("Direction vector n must be a 3-element array.")
    n = n / np.linalg.norm(n)  # Normalize

    # For Young's modulus calculation, we need to convert the direction vector
    # to the proper form for contraction with the 4th order compliance tensor
    # 
    # The correct formula is: E = 1 / (n_i n_j n_k n_l S_ijkl)
    # In Voigt notation: E = 1 / (n_voigt^T S n_voigt)
    # where n_voigt = [n1^2, n2^2, n3^2, n2*n3, n1*n3, n1*n2] (NOT 2*n_i*n_j for shear terms)
    
    n_voigt = np.array([
        n[0]**2,           # n1^2
        n[1]**2,           # n2^2  
        n[2]**2,           # n3^2
        n[1]*n[2],         # n2*n3 (NOT 2*n2*n3)
        n[0]*n[2],         # n1*n3 (NOT 2*n1*n3)
        n[0]*n[1]          # n1*n2 (NOT 2*n1*n2)
    ])

    # E = 1 / (n_voigt^T S n_voigt)
    return 1.0 / np.dot(n_voigt, np.dot(S, n_voigt))

def shear_modulus_directional(S: np.ndarray, n: np.ndarray, return_extrema: bool = False):
    """Calculate shear modulus for direction n.
    
    For a given direction n, calculates the shear modulus by finding the minimum
    and maximum shear moduli in planes perpendicular to n.
    
    The shear modulus in a plane perpendicular to n with shear direction s is:
    G = 1 / (4 * s_i * s_j * s_k * s_l * S_ijkl)
    
    where s is perpendicular to n.
    
    Args:
        S: Compliance matrix (6x6)
        n: Direction vector (3,)
        return_extrema: If True, return (min_G, max_G), else return average
    
    Returns:
        float or tuple: Shear modulus value(s) in GPa
    """
    if n.shape != (3,):
        raise ValueError("Direction vector n must be a 3-element array.")
    n = n / np.linalg.norm(n)  # Normalize
    
    # Find two perpendicular vectors to n to span the plane perpendicular to n
    if abs(n[2]) < 0.9:
        v1 = np.cross(n, np.array([0, 0, 1]))
    else:
        v1 = np.cross(n, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    
    v2 = np.cross(n, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Sample shear directions in the plane perpendicular to n
    # For each angle theta, the shear direction is s = cos(theta)*v1 + sin(theta)*v2
    n_angles = 180
    angles = np.linspace(0, np.pi, n_angles)
    G_values = []
    
    for theta in angles:
        s = np.cos(theta) * v1 + np.sin(theta) * v2
        s = s / np.linalg.norm(s)
        
        # For shear modulus in direction s perpendicular to n:
        # We need to calculate G = 1 / (4 * n_i * s_j * n_k * s_l * S_ijkl)
        # This is the compliance for shear strain in direction s on plane normal to n
        
        # Build the mixed tensor: n⊗s (outer product)
        # In Voigt notation for shear:
        ns_voigt = np.array([
            n[0]*s[0],
            n[1]*s[1],
            n[2]*s[2],
            (n[1]*s[2] + n[2]*s[1])/2,
            (n[0]*s[2] + n[2]*s[0])/2,
            (n[0]*s[1] + n[1]*s[0])/2
        ])
        
        # Calculate shear compliance: γ = 4 * ns_voigt^T * S * ns_voigt
        shear_compliance = 4.0 * np.dot(ns_voigt, np.dot(S, ns_voigt))
        
        if shear_compliance > 1e-12:
            G = 1.0 / shear_compliance
            G_values.append(G)
    
    if len(G_values) == 0:
        if return_extrema:
            return 0.0, 0.0
        else:
            return 0.0
    
    G_min = np.min(G_values)
    G_max = np.max(G_values)
    
    # Clamp to reasonable values
    G_min = max(0.1, min(G_min, 1000.0))
    G_max = max(0.1, min(G_max, 1000.0))
    
    if return_extrema:
        return G_min, G_max
    else:
        # For 2D plots, return average
        return (G_min + G_max) / 2.0

def calculate_shear_modulus_christoffel(C: np.ndarray, n: np.ndarray, s: np.ndarray) -> float:
    """Calculate shear modulus using Christoffel equation.
    
    Args:
        C: Stiffness matrix (6x6) in Voigt notation
        n: Normal vector to the shear plane
        s: Shear direction vector
    
    Returns:
        Shear modulus in GPa
    """
    # Convert Voigt stiffness matrix to 4th order tensor
    C_tensor = np.zeros((3, 3, 3, 3))
    
    # Voigt to tensor mapping
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
    
    for i in range(6):
        for j in range(6):
            p, q = voigt_map[i]
            r, s_idx = voigt_map[j]
            
            C_tensor[p,q,r,s_idx] = C[i,j]
            
            # Ensure full tensor symmetry
            if p != q:
                C_tensor[q,p,r,s_idx] = C_tensor[p,q,r,s_idx]
            if r != s_idx:
                C_tensor[p,q,s_idx,r] = C_tensor[p,q,r,s_idx]
            if p != q and r != s_idx:
                C_tensor[q,p,s_idx,r] = C_tensor[p,q,r,s_idx]
    
    # Calculate Christoffel matrix elements for shear wave
    # Γ_ik = C_ijkl * n_j * n_l
    Gamma = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for l in range(3):
                    Gamma[i,k] += C_tensor[i,j,k,l] * n[j] * n[l]
    
    # For shear wave with polarization s, the shear modulus is
    # G = s^T * Γ * s
    G = np.dot(s, np.dot(Gamma, s))
    
    return max(G, 1e-3)  # Ensure minimum value

def fast_shear_modulus_rotation(S: np.ndarray, R: np.ndarray) -> float:
    """Fast approximation for shear modulus calculation using rotation matrix R.
    
    Uses a more stable approach based on the Christoffel equation for shear waves.
    """
    # For stability, fall back to the full tensor rotation but with optimized implementation
    # This is more accurate than the previous approximation
    
    # Use a simplified 2x2 rotation for the shear plane
    # Extract the in-plane rotation components
    r11, r12 = R[0, 0], R[0, 1]
    r21, r22 = R[1, 0], R[1, 1]
    
    # Calculate the effective shear compliance in the rotated coordinate system
    # Using the 2D rotation of the relevant compliance matrix components
    
    cos_theta = r11
    sin_theta = r12
    
    # For shear modulus, we need the (6,6) component of the rotated compliance matrix
    # Use the transformation rule for the shear compliance component
    
    # S'_66 = S_66*cos^4(θ) + S_11*sin^4(θ) + S_22*cos^4(θ) + 
    #         (2*S_12 + S_66)*sin^2(θ)*cos^2(θ)
    
    cos2 = cos_theta**2
    sin2 = sin_theta**2
    cos4 = cos2**2
    sin4 = sin2**2
    
    s66_eff = (
        S[5,5] * cos4 +
        S[0,0] * sin4 +
        S[1,1] * cos4 +
        (2*S[0,1] + S[5,5]) * sin2 * cos2
    )
    
    # Add contributions from shear-normal coupling terms if they exist
    if abs(S[0,5]) > 1e-12 or abs(S[1,5]) > 1e-12:
        s66_eff += 2 * S[0,5] * sin_theta * cos_theta * (cos2 - sin2)
        s66_eff += 2 * S[1,5] * sin_theta * cos_theta * (sin2 - cos2)
    
    # Ensure reasonable bounds
    s66_eff = max(s66_eff, 1e-6)  # Prevent division by very small numbers
    s66_eff = min(s66_eff, 1.0)   # Prevent unreasonably large moduli
    
    return 1.0 / s66_eff

def rotate_compliance_matrix(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate the 6x6 compliance matrix using Bond transformation.
    
    Args:
        S: 6x6 compliance matrix in Voigt notation
        R: 3x3 rotation matrix
    
    Returns:
        S_rot: Rotated 6x6 compliance matrix
    """
    # Use Bond transformation matrix for Voigt notation
    # This is more direct and accurate than 4th order tensor rotation
    
    # Create Bond transformation matrix
    M = create_bond_matrix(R)
    
    # Transform: S' = M * S * M^T
    S_rot = np.dot(M, np.dot(S, M.T))
    
    return S_rot

def create_bond_matrix(R: np.ndarray) -> np.ndarray:
    """Create Bond transformation matrix for Voigt notation.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        M: 6x6 Bond transformation matrix
    """
    M = np.zeros((6, 6))
    
    # Extract rotation matrix elements
    r = R
    
    # Fill the Bond matrix according to the standard transformation
    # For compliance matrix (strain transformation)
    
    # Normal strain components (1-3)
    M[0,0] = r[0,0]**2
    M[0,1] = r[0,1]**2  
    M[0,2] = r[0,2]**2
    M[0,3] = 2*r[0,1]*r[0,2]
    M[0,4] = 2*r[0,0]*r[0,2]
    M[0,5] = 2*r[0,0]*r[0,1]
    
    M[1,0] = r[1,0]**2
    M[1,1] = r[1,1]**2
    M[1,2] = r[1,2]**2
    M[1,3] = 2*r[1,1]*r[1,2]
    M[1,4] = 2*r[1,0]*r[1,2]
    M[1,5] = 2*r[1,0]*r[1,1]
    
    M[2,0] = r[2,0]**2
    M[2,1] = r[2,1]**2
    M[2,2] = r[2,2]**2
    M[2,3] = 2*r[2,1]*r[2,2]
    M[2,4] = 2*r[2,0]*r[2,2]
    M[2,5] = 2*r[2,0]*r[2,1]
    
    # Shear strain components (4-6)
    M[3,0] = r[1,0]*r[2,0]
    M[3,1] = r[1,1]*r[2,1]
    M[3,2] = r[1,2]*r[2,2]
    M[3,3] = r[1,1]*r[2,2] + r[1,2]*r[2,1]
    M[3,4] = r[1,0]*r[2,2] + r[1,2]*r[2,0]
    M[3,5] = r[1,0]*r[2,1] + r[1,1]*r[2,0]
    
    M[4,0] = r[0,0]*r[2,0]
    M[4,1] = r[0,1]*r[2,1]
    M[4,2] = r[0,2]*r[2,2]
    M[4,3] = r[0,1]*r[2,2] + r[0,2]*r[2,1]
    M[4,4] = r[0,0]*r[2,2] + r[0,2]*r[2,0]
    M[4,5] = r[0,0]*r[2,1] + r[0,1]*r[2,0]
    
    M[5,0] = r[0,0]*r[1,0]
    M[5,1] = r[0,1]*r[1,1]
    M[5,2] = r[0,2]*r[1,2]
    M[5,3] = r[0,1]*r[1,2] + r[0,2]*r[1,1]
    M[5,4] = r[0,0]*r[1,2] + r[0,2]*r[1,0]
    M[5,5] = r[0,0]*r[1,1] + r[0,1]*r[1,0]
    
    return M

def voigt_to_tensor_4th(S: np.ndarray) -> np.ndarray:
    """Convert 6x6 Voigt compliance matrix to 3x3x3x3 tensor using correct factors."""
    S_tensor = np.zeros((3, 3, 3, 3))
    
    # Voigt to tensor index mapping
    voigt_map = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
    
    for i in range(6):
        for j in range(6):
            p, q = voigt_map[i]
            r, s = voigt_map[j]
            
            # Apply correct factors for compliance tensor conversion
            factor = 1.0
            if i >= 3:  # Shear strain component
                factor *= 2.0
            if j >= 3:  # Shear stress component  
                factor *= 2.0
                
            S_tensor[p,q,r,s] = S[i,j] * factor
            
            # Ensure full tensor symmetry
            if p != q:
                S_tensor[q,p,r,s] = S_tensor[p,q,r,s]
            if r != s:
                S_tensor[p,q,s,r] = S_tensor[p,q,r,s]
            if p != q and r != s:
                S_tensor[q,p,s,r] = S_tensor[p,q,r,s]
    
    return S_tensor

def tensor_4th_to_voigt(S_tensor: np.ndarray) -> np.ndarray:
    """Convert 3x3x3x3 tensor to 6x6 Voigt compliance matrix using correct factors."""
    S = np.zeros((6, 6))
    
    # Tensor to Voigt index mapping
    tensor_map = {(0,0): 0, (1,1): 1, (2,2): 2, (1,2): 3, (2,1): 3, 
                  (0,2): 4, (2,0): 4, (0,1): 5, (1,0): 5}
    
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(3):
                    if (p,q) in tensor_map and (r,s) in tensor_map:
                        i = tensor_map[(p,q)]
                        j = tensor_map[(r,s)]
                        
                        # Apply correct factors for Voigt notation
                        factor = 1.0
                        if i >= 3:  # Shear strain component
                            factor /= 2.0
                        if j >= 3:  # Shear stress component
                            factor /= 2.0
                            
                        S[i,j] = S_tensor[p,q,r,s] * factor
    
    return S

def _direction_to_voigt(d: np.ndarray) -> np.ndarray:
    """Convert a unit direction vector to Voigt form for
    contraction with Compliance matrix S (v^T S v).
    
    Returns the Voigt vector corresponding to the tensor n⊗n in STRESS notation 
    (no factor of 2 for shear components).
    """
    return np.array([
        d[0] * d[0],
        d[1] * d[1],
        d[2] * d[2],
        d[1] * d[2],
        d[0] * d[2],
        d[0] * d[1],
    ])

def poisson_ratio_directional(S: np.ndarray, n: np.ndarray, return_extrema: bool = False, m: np.ndarray = None, n_angles: int = 360):
    """Directional Poisson's ratio for loading along n.
    
    ν(n, m) = - (m_i m_j S_ijkl n_k n_l) / (n_i n_j S_ijkl n_k n_l)
    
    Uses engineering strain convention (shear terms ×2 in Voigt notation).
    
    Parameters
    ----------
    S : (6,6) ndarray
        Compliance matrix (engineering Voigt form)
    n : (3,) ndarray
        Loading direction
    return_extrema : bool
        If True return (nu_min, nu_max), else return (nu_min, nu_max)
    m : (3,) ndarray, optional
        Specific lateral direction. If provided, returns ν for this direction only.
    n_angles : int
        Number of angular samples for lateral directions (default: 1440 for high accuracy)
    
    Returns
    -------
    (nu_min, nu_max) or float
        Poisson's ratio extrema or specific value
    """
    if n.shape != (3,):
        raise ValueError("Direction vector n must have shape (3,)")
    n = n / np.linalg.norm(n)
    
    n_voigt = _direction_to_voigt(n)
    axial_compliance = n_voigt @ S @ n_voigt
    
    if axial_compliance <= 0.0:
        raise ValueError("Unstable elastic response along direction n")
    
    # If specific lateral direction is provided
    if m is not None:
        if m.shape != (3,):
            raise ValueError("Lateral direction vector m must have shape (3,)")
        m = m / np.linalg.norm(m)
        
        # Ensure m is perpendicular to n
        if abs(np.dot(m, n)) > 1e-10:
            # Project m onto plane perpendicular to n
            m = m - np.dot(m, n) * n
            m = m / np.linalg.norm(m)
        
        m_voigt = _direction_to_voigt(m)
        lateral_compliance = m_voigt @ S @ n_voigt
        nu = -lateral_compliance / axial_compliance
        
        if return_extrema:
            return nu, nu
        else:
            return nu
    
    # Construct orthonormal basis perpendicular to n
    if abs(n[2]) < 0.9:
        v1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    else:
        v1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    nu_values = []
    
    for theta in angles:
        m = np.cos(theta) * v1 + np.sin(theta) * v2
        m /= np.linalg.norm(m)
        m_voigt = _direction_to_voigt(m)
        lateral_compliance = m_voigt @ S @ n_voigt
        nu = -lateral_compliance / axial_compliance
        nu_values.append(nu)
    
    nu_min = float(np.min(nu_values))
    nu_max = float(np.max(nu_values))
    
    if return_extrema:
        return nu_min, nu_max
    else:
        return nu_min, nu_max

def poisson_ratio_2d_section(S: np.ndarray, n: np.ndarray, plane_normal: np.ndarray, n_angles: int = 360):
    """Poisson's ratio extrema for a 2D section.
    
    Loading direction n lies in the plane.
    Lateral directions are all directions perpendicular to n
    (including out-of-plane components).
    
    Parameters
    ----------
    S : (6,6) ndarray
        Compliance matrix
    n : (3,) ndarray
        Loading direction (in plane)
    plane_normal : (3,) ndarray
        Normal of the section plane
    n_angles : int
        Angular sampling resolution (default: 1440 for high accuracy)
    
    Returns
    -------
    (nu_min, nu_max)
    """
    n = n / np.linalg.norm(n)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    n_voigt = _direction_to_voigt(n)
    axial_compliance = n_voigt @ S @ n_voigt
    
    if axial_compliance <= 0.0:
        raise ValueError("Unstable elastic response along direction n")
    
    # Basis perpendicular to n
    if abs(np.dot(n, plane_normal)) < 0.9:
        v1 = np.cross(n, plane_normal)
    else:
        if abs(n[0]) < 0.9:
            v1 = np.cross(n, np.array([1.0, 0.0, 0.0]))
        else:
            v1 = np.cross(n, np.array([0.0, 1.0, 0.0]))
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    nu_values = []
    
    for theta in angles:
        m = np.cos(theta) * v1 + np.sin(theta) * v2
        m /= np.linalg.norm(m)
        m_voigt = _direction_to_voigt(m)
        lateral_compliance = m_voigt @ S @ n_voigt
        nu = -lateral_compliance / axial_compliance
        nu_values.append(nu)
    
    return float(np.min(nu_values)), float(np.max(nu_values))

def linear_compressibility_directional(S: np.ndarray, n: np.ndarray) -> float:
    """Calculate linear compressibility in a given direction n."""
    if n.shape != (3,):
        raise ValueError("Direction vector n must be a 3-element array.")
    n = n / np.linalg.norm(n)  # Normalize

    # Hydrostatic stress in Voigt notation: [σ1, σ2, σ3, τ23, τ13, τ12]
    hydrostatic_stress_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    
    # Direction vector in Voigt notation for strain calculation
    n_voigt = np.array([
        n[0]**2, n[1]**2, n[2]**2,
        n[1]*n[2], n[0]*n[2], n[0]*n[1]
    ])
    
    return np.dot(n_voigt, np.dot(S, hydrostatic_stress_voigt))

# -------------------------
# 2D and 3D visualization
# -------------------------

def plot_2d_polar(matrix: np.ndarray, prop_type: str, plane: str, n_points: int = 360, filename: str = "plot.png", angle_range: Tuple[float, float] = (0, 2 * np.pi), use_abs: bool = False):
    """Generate a 2D polar plot of a given elastic property in a specified plane.
    
    For Poisson's ratio in 2D sections:
    - Loading direction: rotates in the plane
    - Lateral direction: perpendicular to the plane (out-of-plane direction)
    
    Args:
        matrix: Compliance or stiffness matrix
        prop_type: Property type
        plane: Plane name ('xy', 'xz', 'yz')
        n_points: Number of angular points
        filename: Output filename
        angle_range: Angular range
        use_abs: If True, plot absolute values (for Poisson's ratio)
    """
    theta = np.linspace(angle_range[0], angle_range[1], n_points)
    values = np.zeros_like(theta)
    
    # 对于剪切模量和泊松比，还需要存储极
    # (corrupted comment)
    min_values = np.zeros_like(theta) if prop_type in ['G', 'nu'] else None
    max_values = np.zeros_like(theta) if prop_type in ['G', 'nu'] else None

    for i, t in enumerate(theta):
        if plane == "xy":
            n = np.array([np.cos(t), np.sin(t), 0])
            # For Poisson's ratio: lateral direction is z-axis
            lateral_dir = np.array([0, 0, 1])
        elif plane == "yz":
            n = np.array([0, np.cos(t), np.sin(t)])
            # For Poisson's ratio: lateral direction is x-axis
            lateral_dir = np.array([1, 0, 0])
        elif plane == "xz":
            n = np.array([np.cos(t), 0, np.sin(t)])
            # For Poisson's ratio: lateral direction is y-axis
            lateral_dir = np.array([0, 1, 0])
        else:
            raise ValueError("Plane must be one of \'xy\', \'yz\', or \'xz\'.")

        if prop_type == "E":
            values[i] = youngs_modulus_directional(matrix, n)
        elif prop_type == "L":
            values[i] = linear_compressibility_directional(matrix, n)
        elif prop_type == "G":
            min_g, max_g = shear_modulus_directional(matrix, n, return_extrema=True)
            values[i] = max_g  # 显示最大值作为主
            # (corrupted comment)
            min_values[i] = min_g
            max_values[i] = max_g
        elif prop_type == "nu":
            # For 2D sections, calculate Poisson's ratio extrema
            # Use poisson_ratio_directional to get extrema over all lateral directions
            min_nu, max_nu = poisson_ratio_directional(matrix, n, return_extrema=True)
            if use_abs:
                # 使用绝对
                # (corrupted comment)
                min_values[i] = min(abs(min_nu), abs(max_nu))
                max_values[i] = max(abs(min_nu), abs(max_nu))
                values[i] = max_values[i]
            else:
                values[i] = max_nu  # Display max as main line
                min_values[i] = min_nu
                max_values[i] = max_nu
        elif prop_type == "B":
            # Bulk modulus - directional
            values[i] = bulk_modulus_directional(matrix, n)
        else:
            raise ValueError("Property must be \'E\' (Young\'s modulus), \'L\' (Linear compressibility), \'G\' (Shear Modulus), \'nu\' (Poisson\'s Ratio), or \'B\' (Bulk Modulus).")

    # 获取性质全称和单
    full_name = PROPERTY_NAMES.get(prop_type, prop_type)
    unit = PROPERTY_UNITS.get(prop_type, '')
    unit_str = f' ({unit})' if unit else ''
    unit_with_space = f' {unit}' if unit else ''

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    if prop_type == 'G':
        # 剪切模量：绘制最大值和最小
        # (corrupted comment)
        ax.plot(theta, max_values, linewidth=2, color='red', label=f'Maximum G')
        ax.plot(theta, min_values, linewidth=2, color='blue', label=f'Minimum G')
        ax.fill_between(theta, min_values, max_values, alpha=0.3, color='gray', label='Range')
        
        # 找到全局极
        # (corrupted comment)
        global_max = np.max(max_values)
        global_min = np.min(min_values)
        global_max_idx = np.argmax(max_values)
        global_min_idx = np.argmin(min_values)
        
        # 标记全局极
        # (corrupted comment)
        ax.plot(theta[global_max_idx], global_max, 'ro', markersize=10,
                label=f'Global Max: {global_max:.3f}{unit_with_space}')
        ax.plot(theta[global_min_idx], global_min, 'bo', markersize=10, 
                label=f'Global Min: {global_min:.3f}{unit_with_space}')
    elif prop_type == 'nu':
        # 泊松比：显示最大值和最小值曲
        # (corrupted comment)
        if use_abs:
            # 绝对值图
            ax.plot(theta, max_values, linewidth=2, color='blue', label='|ν_max|', zorder=3)
            ax.plot(theta, min_values, linewidth=2, color='red', label='|ν_min|', zorder=3)
            ax.fill_between(theta, min_values, max_values, alpha=0.3, color='gray', label='Range')
        else:
            # 原始值图（蓝色=最大值，红色=最小值）
            ax.plot(theta, max_values, linewidth=2, color='blue', label='Maximum ν', zorder=3)
            ax.plot(theta, min_values, linewidth=2, color='red', label='Minimum ν', zorder=3)
        
        # 检查是否有负值（auxetic 行为）和正值
        has_negative = np.any(min_values < 0) or np.any(max_values < 0)
        has_positive = np.any(min_values > 0) or np.any(max_values > 0)
        
        if not use_abs and has_negative and has_positive:
            # 有正负值，分区域填
            # 填充负值区域 auxetic 青色
            for i in range(len(theta)-1):
                lower = min(min_values[i], min_values[i+1], 0)
                upper = min(max_values[i], max_values[i+1], 0)
                if lower < upper:
                    ax.fill_between([theta[i], theta[i+1]], 
                                  [lower, lower], [upper, upper], 
                                  alpha=0.4, color='cyan')
            
            # 填充正值区
            # - 黄色
            for i in range(len(theta)-1):
                lower = max(min_values[i], min_values[i+1], 0)
                upper = max(max_values[i], max_values[i+1], 0)
                if lower < upper:
                    ax.fill_between([theta[i], theta[i+1]], 
                                  [lower, lower], [upper, upper], 
                                  alpha=0.3, color='yellow')
            
            # 添加零线
            ax.plot(theta, np.zeros_like(theta), 'k--', linewidth=1.5, alpha=0.7, label='ν = 0', zorder=2)
        elif not use_abs:
            # 全正或全负，正常填充
            color = 'cyan' if has_negative else 'yellow'
            ax.fill_between(theta, min_values, max_values, alpha=0.3, color=color, label='Range')
            if has_negative:
                ax.plot(theta, np.zeros_like(theta), 'k--', linewidth=1, alpha=0.5, label='ν = 0')
        else:
            # 绝对值图，不需要零
            # (corrupted comment)
            pass
        
        # 找到全局极
        # (corrupted comment)
        global_max = np.max(max_values)
        global_min = np.min(min_values)
        global_max_idx = np.argmax(max_values)
        global_min_idx = np.argmin(min_values)
        
        # 标记全局极值（蓝色=最大值，红色=最小值）
        ax.plot(theta[global_max_idx], global_max, 'bo', markersize=10, 
                label=f'Global Max: {global_max:.3f}', zorder=4)
        ax.plot(theta[global_min_idx], global_min, 'ro', markersize=10, 
                label=f'Global Min: {global_min:.3f}', zorder=4)
        
        # 添加负泊松比说明
        if not use_abs and global_min < 0:
            ax.text(0.02, 0.98, 'Auxetic Behavior\n(Negative Poisson\'s Ratio)', 
                   transform=ax.transAxes, fontsize=11, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.6, edgecolor='blue'))
    else:
        # 对于杨氏模量、线性压缩率和体弹性模量，只有一个
        # (corrupted comment)
        ax.plot(theta, values, linewidth=2, color='blue')
        
        # 找到最大值和最小
        # (corrupted comment)
        max_val = np.max(values)
        min_val = np.min(values)
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        
        # 标记最大值和最小
        # (corrupted comment)
        ax.plot(theta[max_idx], max_val, 'ro', markersize=8,
                label=f'Max: {max_val:.3f}{unit_with_space}')
        ax.plot(theta[min_idx], min_val, 'go', markersize=8, 
                label=f'Min: {min_val:.3f}{unit_with_space}')
    
    # 设置标题
    title = f"{full_name}{unit_str} in {plane.upper()} plane"
    ax.set_title(title, fontsize=16, fontfamily=selected_font, pad=20)
    # 设置0度在东方（右方），对应数学定义的+x
    # 逆时针方向，对应标准数学角度定义
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    
    # 设置极坐标图的字
    # (corrupted comment)
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontfamily(selected_font)
    for label in ax.get_yticklabels():
        label.set_fontfamily(selected_font)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 准备返回极值信
    extremal_info = {}
    
    if prop_type in ['G', 'nu']:
        # 找到全局极值的索引
        global_max_idx = np.argmax(max_values)
        global_min_idx = np.argmin(min_values)
        
        # 计算对应的方向向
        # (corrupted comment)
        if plane == "xy":
            max_direction = np.array([np.cos(theta[global_max_idx]), np.sin(theta[global_max_idx]), 0])
            min_direction = np.array([np.cos(theta[global_min_idx]), np.sin(theta[global_min_idx]), 0])
        elif plane == "yz":
            max_direction = np.array([0, np.cos(theta[global_max_idx]), np.sin(theta[global_max_idx])])
            min_direction = np.array([0, np.cos(theta[global_min_idx]), np.sin(theta[global_min_idx])])
        elif plane == "xz":
            max_direction = np.array([np.cos(theta[global_max_idx]), 0, np.sin(theta[global_max_idx])])
            min_direction = np.array([np.cos(theta[global_min_idx]), 0, np.sin(theta[global_min_idx])])
        
        extremal_info = {
            'max_value': np.max(max_values),
            'min_value': np.min(min_values),
            'max_direction': max_direction / np.linalg.norm(max_direction),
            'min_direction': min_direction / np.linalg.norm(min_direction),
            'max_angle_deg': np.degrees(theta[global_max_idx]),
            'min_angle_deg': np.degrees(theta[global_min_idx])
        }
        return theta, max_values, min_values, extremal_info
    else:
        # 找到极值的索引
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        
        # 计算对应的方向向
        # (corrupted comment)
        if plane == "xy":
            max_direction = np.array([np.cos(theta[max_idx]), np.sin(theta[max_idx]), 0])
            min_direction = np.array([np.cos(theta[min_idx]), np.sin(theta[min_idx]), 0])
        elif plane == "yz":
            max_direction = np.array([0, np.cos(theta[max_idx]), np.sin(theta[max_idx])])
            min_direction = np.array([0, np.cos(theta[min_idx]), np.sin(theta[min_idx])])
        elif plane == "xz":
            max_direction = np.array([np.cos(theta[max_idx]), 0, np.sin(theta[max_idx])])
            min_direction = np.array([np.cos(theta[min_idx]), 0, np.sin(theta[min_idx])])
        
        extremal_info = {
            'max_value': np.max(values),
            'min_value': np.min(values),
            'max_direction': max_direction / np.linalg.norm(max_direction),
            'min_direction': min_direction / np.linalg.norm(min_direction),
            'max_angle_deg': np.degrees(theta[max_idx]),
            'min_angle_deg': np.degrees(theta[min_idx])
        }
        return theta, values, extremal_info

def plot_crystallographic_sections(matrix: np.ndarray, prop_type: str, output_dir: str, n_points: int = 360):
    """Generate 2D plots for crystallographic sections (100), (010), and (001)."""
    
    # 获取性质全称和单
    #     full_name = PROPERTY_NAMES.get(prop_type, prop_type)
    unit = PROPERTY_UNITS.get(prop_type, '')
    unit_str = f' ({unit})' if unit else ''
    
    # (100) section: varies in YZ plane
    theta_100 = np.linspace(0, 2*np.pi, n_points)
    values_100 = np.zeros_like(theta_100)
    min_values_100 = np.zeros_like(theta_100) if prop_type in ['G', 'nu'] else None
    max_values_100 = np.zeros_like(theta_100) if prop_type in ['G', 'nu'] else None
    
    # (010) section: varies in XZ plane  
    theta_010 = np.linspace(0, 2*np.pi, n_points)
    values_010 = np.zeros_like(theta_010)
    min_values_010 = np.zeros_like(theta_010) if prop_type in ['G', 'nu'] else None
    max_values_010 = np.zeros_like(theta_010) if prop_type in ['G', 'nu'] else None
    
    # (001) section: varies in XY plane
    theta_001 = np.linspace(0, 2*np.pi, n_points)
    values_001 = np.zeros_like(theta_001)
    min_values_001 = np.zeros_like(theta_001) if prop_type in ['G', 'nu'] else None
    max_values_001 = np.zeros_like(theta_001) if prop_type in ['G', 'nu'] else None
    
    for i, t in enumerate(theta_100):
        # (100) section: x=1, vary in yz
        n_100 = np.array([1, np.cos(t), np.sin(t)])
        n_100 = n_100 / np.linalg.norm(n_100)
        
        # (010) section: y=1, vary in xz
        n_010 = np.array([np.cos(t), 1, np.sin(t)])
        n_010 = n_010 / np.linalg.norm(n_010)
        
        # (001) section: z=1, vary in xy
        n_001 = np.array([np.cos(t), np.sin(t), 1])
        n_001 = n_001 / np.linalg.norm(n_001)
        
        if prop_type == "E":
            values_100[i] = youngs_modulus_directional(matrix, n_100)
            values_010[i] = youngs_modulus_directional(matrix, n_010)
            values_001[i] = youngs_modulus_directional(matrix, n_001)
        elif prop_type == "L":
            values_100[i] = linear_compressibility_directional(matrix, n_100)
            values_010[i] = linear_compressibility_directional(matrix, n_010)
            values_001[i] = linear_compressibility_directional(matrix, n_001)
        elif prop_type == "G":
            min_g, max_g = shear_modulus_directional(matrix, n_100, return_extrema=True)
            values_100[i] = max_g
            min_values_100[i] = min_g
            max_values_100[i] = max_g
            
            min_g, max_g = shear_modulus_directional(matrix, n_010, return_extrema=True)
            values_010[i] = max_g
            min_values_010[i] = min_g
            max_values_010[i] = max_g
            
            min_g, max_g = shear_modulus_directional(matrix, n_001, return_extrema=True)
            values_001[i] = max_g
            min_values_001[i] = min_g
            max_values_001[i] = max_g
        elif prop_type == "nu":
            min_nu, max_nu = poisson_ratio_directional(matrix, n_100, return_extrema=True)
            values_100[i] = max_nu
            min_values_100[i] = min_nu
            max_values_100[i] = max_nu
            
            min_nu, max_nu = poisson_ratio_directional(matrix, n_010, return_extrema=True)
            values_010[i] = max_nu
            min_values_010[i] = min_nu
            max_values_010[i] = max_nu
            
            min_nu, max_nu = poisson_ratio_directional(matrix, n_001, return_extrema=True)
            values_001[i] = max_nu
            min_values_001[i] = min_nu
            max_values_001[i] = max_nu
    
    # 绘制三个截面图
    sections_data = [
        ("100", theta_100, values_100, min_values_100, max_values_100), 
        ("010", theta_010, values_010, min_values_010, max_values_010), 
        ("001", theta_001, values_001, min_values_001, max_values_001)
    ]
    
    for section_name, theta, values, min_vals, max_vals in sections_data:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
        
        if prop_type in ['G', 'nu']:
            # 绘制最大值和最小
            # (corrupted comment)
            ax.plot(theta, max_vals, linewidth=2, color='red', label=f'Maximum {prop_type}')
            ax.plot(theta, min_vals, linewidth=2, color='blue', label=f'Minimum {prop_type}')
            ax.fill_between(theta, min_vals, max_vals, alpha=0.3, color='gray', label='Range')
            
            # 找到全局极
            # (corrupted comment)
            global_max = np.max(max_vals)
            global_min = np.min(min_vals)
            global_max_idx = np.argmax(max_vals)
            global_min_idx = np.argmin(min_vals)
            
            # 标记全局极
            # (corrupted comment)
            ax.plot(theta[global_max_idx], global_max, 'ro', markersize=10,
                    label=f'Global Max: {global_max:.3f}{unit}')
            ax.plot(theta[global_min_idx], global_min, 'bo', markersize=10, 
                    label=f'Global Min: {global_min:.3f}{unit}')
        else:
            ax.plot(theta, values, linewidth=2, color='blue')
            
            # 找到最大值和最小
            # (corrupted comment)
            max_val = np.max(values)
            min_val = np.min(values)
            max_idx = np.argmax(values)
            min_idx = np.argmin(values)
            
            # 标记最大值和最小
            # (corrupted comment)
            ax.plot(theta[max_idx], max_val, 'ro', markersize=8,
                    label=f'Max: {max_val:.3f}{unit}')
            ax.plot(theta[min_idx], min_val, 'go', markersize=8, 
                    label=f'Min: {min_val:.3f}{unit}')
        
        # 设置标题
        title = f"{full_name}{unit_str} in ({section_name}) section"
        ax.set_title(title, fontsize=16, fontfamily=selected_font, pad=20)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        
        # 设置字体
        ax.tick_params(labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontfamily(selected_font)
        for label in ax.get_yticklabels():
            label.set_fontfamily(selected_font)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=10)
        
        filename = f"{output_dir}/{prop_type}_2D_{section_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 保存数据为Origin格式
        data_filename = f"{output_dir}/{prop_type}_2D_{section_name}_data.txt"
        if prop_type in ['G', 'nu']:
            save_2d_extrema_data_for_origin(data_filename, theta, min_vals, max_vals)
        else:
            save_2d_data_for_origin(data_filename, theta, values)
    
    return sections_data

def plot_3d_surface(matrix: np.ndarray, prop_type: str, n_phi: int = 100, n_theta: int = 100, filename: str = "plot.png", phi_range: Tuple[float, float] = (0, 2 * np.pi), theta_range: Tuple[float, float] = (0, np.pi), use_abs: bool = False):
    """Generate a 3D surface plot of a given elastic property.
    
    Args:
        matrix: Compliance or stiffness matrix
        prop_type: Property type ('E', 'G', 'nu', 'L', 'B')
        n_phi: Number of points in phi direction
        n_theta: Number of points in theta direction
        filename: Output filename
        phi_range: Range of phi angles
        theta_range: Range of theta angles
        use_abs: If True, plot absolute values (for Poisson's ratio)
    """
    phi = np.linspace(phi_range[0], phi_range[1], n_phi)
    theta = np.linspace(theta_range[0], theta_range[1], n_theta)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    values = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            n = np.array([x[i,j], y[i,j], z[i,j]])
            if prop_type == "E":
                values[i,j] = youngs_modulus_directional(matrix, n)
            elif prop_type == "L":
                values[i,j] = linear_compressibility_directional(matrix, n)
            elif prop_type == "G":
                # 剪切模量：从旋转方向提取最大值
                _, max_g = shear_modulus_directional(matrix, n, return_extrema=True)
                values[i,j] = max_g
            elif prop_type == "nu":
                # 泊松比：从旋转方向提取最大值
                _, max_nu = poisson_ratio_directional(matrix, n, return_extrema=True)
                values[i,j] = max_nu
            elif prop_type == "B":
                values[i,j] = bulk_modulus_directional(matrix, n) # Assuming matrix is C
            else:
                raise ValueError("Property must be \'E\' (Young\'s modulus), \'L\' (Linear compressibility), \'G\' (Shear Modulus), \'nu\' (Poisson\'s Ratio), or \'B\' (Bulk Modulus).")
    
    # 如果需要绝对
    # (corrupted comment)
    if use_abs and prop_type == 'nu':
        values = np.abs(values)

    # 找到最大值和最小值及其位
    # (corrupted comment)
    max_val = np.max(values)
    min_val = np.min(values)
    max_idx = np.unravel_index(np.argmax(values), values.shape)
    min_idx = np.unravel_index(np.argmin(values), values.shape)
    
    # 最大值和最小值的坐标
    max_x = values[max_idx] * x[max_idx]
    max_y = values[max_idx] * y[max_idx]
    max_z = values[max_idx] * z[max_idx]
    
    min_x = values[min_idx] * x[min_idx]
    min_y = values[min_idx] * y[min_idx]
    min_z = values[min_idx] * z[min_idx]

    # 使用matplotlib创建3D表面
    # (corrupted comment)
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为泊松比选择特殊的颜色映
    # (corrupted comment)
    import matplotlib.colors as mcolors
    
    if prop_type == 'nu' and not use_abs:
        # 使用RdBu_r颜色映射，负值为蓝色，正值为红色
        if min_val < 0 and max_val > 0:
            # 有正负值，使用diverging colormap
            # (corrupted comment)
            norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
            cmap = plt.cm.RdBu_r
        elif max_val <= 0:
            # 全是负值，使用蓝色
            # (corrupted comment)
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
            cmap = plt.cm.Blues_r
        else:
            # 全是正值，使用红色
            # (corrupted comment)
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
            cmap = plt.cm.Reds
        facecolors = cmap(norm(values))
    else:
        # 其他性质使用viridis
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = plt.cm.viridis
        facecolors = cmap(norm(values))
    
    # 绘制3D表面 - 改进视觉效果
    surf = ax.plot_surface(values*x, values*y, values*z, 
                          facecolors=facecolors,
                          alpha=0.9,
                          linewidth=0.05,
                          antialiased=True,
                          shade=True,
                          rcount=n_theta,
                          ccount=n_phi)
    
    # 标记最大值和最小
    # 获取性质全称和单位（提前获取，用于标签）
    full_name = PROPERTY_NAMES.get(prop_type, prop_type)
    unit = PROPERTY_UNITS.get(prop_type, '')
    unit_str = f' ({unit})' if unit else ''
    unit_with_space = f' {unit}' if unit else ''
    
    ax.scatter([max_x], [max_y], [max_z], color='red', s=150, 
              label=f'Max: {max_val:.3f}{unit_with_space}', alpha=1.0, edgecolors='black', linewidths=2)
    ax.scatter([min_x], [min_y], [min_z], color='blue', s=150, 
              label=f'Min: {min_val:.3f}{unit_with_space}', alpha=1.0, edgecolors='black', linewidths=2)
    
    # 设置标题和标
    # (corrupted comment)
    if use_abs and prop_type == 'nu':
        title = f"3D surface of |{full_name}|{unit_str}\n(Max: {max_val:.3f}{unit_with_space}, Min: {min_val:.3f}{unit_with_space})"
    else:
        title = f"3D surface of {full_name}{unit_str}\n(Max: {max_val:.3f}{unit_with_space}, Min: {min_val:.3f}{unit_with_space})"
        if prop_type == 'nu' and min_val < 0 and not use_abs:
            title += "\n(Blue: Negative/Auxetic, Red: Positive)"
    
    ax.set_title(title, fontsize=15, fontfamily=selected_font, pad=25, weight='bold')
    ax.set_xlabel('X', fontsize=13, fontfamily=selected_font, labelpad=10)
    ax.set_ylabel('Y', fontsize=13, fontfamily=selected_font, labelpad=10)
    ax.set_zlabel('Z', fontsize=13, fontfamily=selected_font, labelpad=10)
    
    # 设置坐标轴刻度字
    # (corrupted comment)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontfamily(selected_font)
        label.set_fontsize(10)
    
    # 添加颜色
    # - 修复乱码问题
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(values)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    
    # 设置颜色条标
    # - 使用ASCII字符避免乱码
    if prop_type == 'nu':
        if use_abs:
            cbar_label = f"|Poisson's Ratio|"
        else:
            cbar_label = f"Poisson's Ratio"
    else:
        cbar_label = f"{full_name}{unit_str}"
    
    cbar.set_label(cbar_label, fontsize=12, fontfamily=selected_font, rotation=270, labelpad=20)
    
    # 设置颜色条刻度字
    # (corrupted comment)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(selected_font)
        label.set_fontsize(10)
    
    # 为泊松比添加零线标记（只绘制一次）
    # 注意：TwoSlopeNorm已经
    # 处有分界，不需要额外绘制零?    
    # 改善视角和光
    # (corrupted comment)
    ax.view_init(elev=20, azim=45)
    ax.dist = 10
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置背景颜色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # 添加图例
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 返回极值信息
    extremal_data = {
        'max_value': max_val,
        'min_value': min_val,
        'max_direction': np.array([x[max_idx], y[max_idx], z[max_idx]]),
        'min_direction': np.array([x[min_idx], y[min_idx], z[min_idx]])
    }
    
    return phi, theta, values, extremal_data

# Placeholder functions for now, to be implemented later
def vrh_averages(C: np.ndarray):
    S = compliance_matrix(C)
    return {
        "bulk_modulus_voigt": bulk_modulus_voigt(C),
        "bulk_modulus_reuss": bulk_modulus_reuss(S),
        "bulk_modulus_vrh": bulk_modulus_vrh(C, S),
        "shear_modulus_voigt": shear_modulus_voigt(C),
        "shear_modulus_reuss": shear_modulus_reuss(S),
        "shear_modulus_vrh": shear_modulus_vrh(C, S)
    }

def anisotropy_indices_from_vrh(vrh_data: Dict):
    """Calculate anisotropy indices from VRH data."""
    # Bulk modulus anisotropy index
    B_V = vrh_data["bulk_modulus_voigt"]
    B_R = vrh_data["bulk_modulus_reuss"]
    A_B = (B_V - B_R) / (B_V + B_R) if (B_V + B_R) > 0 else 0.0
    
    # Shear modulus anisotropy index
    G_V = vrh_data["shear_modulus_voigt"]
    G_R = vrh_data["shear_modulus_reuss"]
    A_G = (G_V - G_R) / (G_V + G_R) if (G_V + G_R) > 0 else 0.0
    
    # Universal anisotropy index (Ranganathan and Ostoja-Starzewski)
    A_U = 5 * (G_V / G_R) + (B_V / B_R) - 6 if G_R > 0 and B_R > 0 else 0.0
    
    return {"A_B": A_B, "A_G": A_G, "A_U": A_U}

def calculate_polycrystalline_moduli(C: np.ndarray):
    """Calculate polycrystalline bulk and shear moduli using Voigt-Reuss-Hill averaging."""
    S = compliance_matrix(C)
    
    # Voigt averages (upper bound)
    K_V = bulk_modulus_voigt(C)
    G_V = shear_modulus_voigt(C)
    
    # Reuss averages (lower bound)
    K_R = bulk_modulus_reuss(S)
    G_R = shear_modulus_reuss(S)
    
    # Hill averages (arithmetic mean)
    K_VRH = 0.5 * (K_V + K_R)
    G_VRH = 0.5 * (G_V + G_R)
    
    # Calculate Young's modulus and Poisson's ratio from Hill averages
    E_VRH = 9 * K_VRH * G_VRH / (3 * K_VRH + G_VRH)
    nu_VRH = (3 * K_VRH - 2 * G_VRH) / (2 * (3 * K_VRH + G_VRH))
    
    return {
        "K_V": K_V, "G_V": G_V, "E_V": 9*K_V*G_V/(3*K_V+G_V), "nu_V": (3*K_V-2*G_V)/(2*(3*K_V+G_V)),
        "K_R": K_R, "G_R": G_R, "E_R": 9*K_R*G_R/(3*K_R+G_R), "nu_R": (3*K_R-2*G_R)/(2*(3*K_R+G_R)),
        "K_VRH": K_VRH, "G_VRH": G_VRH, "E_VRH": E_VRH, "nu_VRH": nu_VRH
    }

def calculate_polycrystalline_anisotropy(poly_moduli: Dict):
    """Calculate polycrystalline anisotropy indices."""
    # Bulk modulus anisotropy
    K_V = poly_moduli["K_V"]
    K_R = poly_moduli["K_R"]
    A_K = (K_V - K_R) / (K_V + K_R) if (K_V + K_R) > 0 else 0.0
    
    # Shear modulus anisotropy
    G_V = poly_moduli["G_V"]
    G_R = poly_moduli["G_R"]
    A_G = (G_V - G_R) / (G_V + G_R) if (G_V + G_R) > 0 else 0.0
    
    return {"A_K": A_K, "A_G": A_G}

def compliance(C: np.ndarray):
    return compliance_matrix(C)

def voigt_compliance_to_tensor(S: np.ndarray):
    """Convert 6×6 Voigt compliance matrix S to 3×3×3×3 compliance tensor.

    Voigt → tensor conversion with standard engineering shear strain factors:
    - I,J ∈ {1,2,3}: s_ijkl = S_IJ
    - one of I,J ∈ {4,5,6}: s_ijkl = S_IJ / 2
    - both I,J ∈ {4,5,6}: s_ijkl = S_IJ / 4
    """
    _ij = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    S4 = np.zeros((3, 3, 3, 3))
    for I in range(6):
        i, j = _ij[I]
        f_I = 0.5 if I >= 3 else 1.0
        for J in range(6):
            k, l = _ij[J]
            f_J = 0.5 if J >= 3 else 1.0
            val = S[I, J] / (f_I * f_J)
            S4[i, j, k, l] = val
            S4[j, i, k, l] = val
            S4[i, j, l, k] = val
            S4[j, i, l, k] = val
    return S4


def E_along(S: np.ndarray, n: np.ndarray):
    """Young's modulus along direction n. S is 6×6 Voigt compliance [GPa⁻¹]."""
    return youngs_modulus_directional(S, n)


def shear_modulus(S: np.ndarray, n: np.ndarray):
    """Shear modulus for direction n (S is 6×6 Voigt compliance)."""
    _, max_g = shear_modulus_directional(S, n, return_extrema=True)
    return max_g


def poisson_ratio(S: np.ndarray, n: np.ndarray, m: np.ndarray):
    """Poisson's ratio for direction n with lateral direction m (S is 6×6 compliance)."""
    return poisson_ratio_directional(S, n, m=m)


def linear_compressibility(S: np.ndarray, n: np.ndarray):
    """Linear compressibility along direction n (S is 6×6 compliance)."""
    return linear_compressibility_directional(S, n)


def extremal_shear_and_poisson(S: np.ndarray, n_angles: int = 360):
    """Extremal shear modulus and Poisson's ratio (global search, S is 6×6 compliance).
    Returns (min_G, max_G, min_nu, max_nu).
    """
    _, max_g = shear_modulus_directional(S, np.array([1.0, 0.0, 0.0]), return_extrema=True)
    # Full extremal search over all directions for G and nu
    min_G, max_G = max_g, max_g
    min_nu, max_nu = float('inf'), float('-inf')
    theta_vals = np.linspace(0, 2 * np.pi, n_angles)
    phi_vals = np.linspace(0, np.pi, n_angles // 2 + 1)
    for phi in phi_vals:
        for theta in theta_vals:
            n = np.array([np.sin(phi) * np.cos(theta),
                          np.sin(phi) * np.sin(theta),
                          np.cos(phi)])
            try:
                _, g = shear_modulus_directional(S, n, return_extrema=True)
                min_G = min(min_G, g)
                max_G = max(max_G, g)
                m_perp = np.array([-n[1], n[0], 0.0])
                nm = np.linalg.norm(m_perp)
                if nm > 1e-10:
                    m_perp /= nm
                    nu = poisson_ratio_directional(S, n, m=m_perp)
                    min_nu = min(min_nu, nu)
                    max_nu = max(max_nu, nu)
            except Exception:
                continue
    if min_nu == float('inf'):
        min_nu = 0.0
    if max_nu == float('-inf'):
        max_nu = 0.0
    return min_G, max_G, min_nu, max_nu

def plot_2d_polar_E(S: np.ndarray, plane: str, filename: str, angle_range: Tuple[float, float]):
    return plot_2d_polar(S, "E", plane, filename=filename, angle_range=angle_range)

def plot_3d_surface_E(S: np.ndarray, filename: str, n_phi: int, n_theta: int, phi_range: Tuple[float, float], theta_range: Tuple[float, float]):
    return plot_3d_surface(S, "E", n_phi=n_phi, n_theta=n_theta, filename=filename, phi_range=phi_range, theta_range=theta_range)

def plot_2d_polar_G(S: np.ndarray, plane: str, filename: str, angle_range: Tuple[float, float]):
    return plot_2d_polar(S, "G", plane, filename=filename, angle_range=angle_range)

def plot_3d_surface_G(S: np.ndarray, filename: str, n_phi: int, n_theta: int, phi_range: Tuple[float, float], theta_range: Tuple[float, float]):
    return plot_3d_surface(S, "G", n_phi=n_phi, n_theta=n_theta, filename=filename, phi_range=phi_range, theta_range=theta_range)

def plot_2d_polar_nu(S: np.ndarray, plane: str, filename: str, angle_range: Tuple[float, float]):
    return plot_2d_polar(S, "nu", plane, filename=filename, angle_range=angle_range)

def plot_3d_surface_nu(S: np.ndarray, filename: str, n_phi: int, n_theta: int, phi_range: Tuple[float, float], theta_range: Tuple[float, float]):
    return plot_3d_surface(S, "nu", n_phi=n_phi, n_theta=n_theta, filename=filename, phi_range=phi_range, theta_range=theta_range)

def plot_2d_polar_beta(S: np.ndarray, plane: str, filename: str, angle_range: Tuple[float, float]):
    return plot_2d_polar(S, "L", plane, filename=filename, angle_range=angle_range)

def plot_3d_surface_beta(S: np.ndarray, filename: str, n_phi: int, n_theta: int, phi_range: Tuple[float, float], theta_range: Tuple[float, float]):
    return plot_3d_surface(S, "L", n_phi=n_phi, n_theta=n_theta, filename=filename, phi_range=phi_range, theta_range=theta_range)

def plot_3d_surface_B(C: np.ndarray, filename: str, n_phi: int, n_theta: int, phi_range: Tuple[float, float], theta_range: Tuple[float, float]):
    return plot_3d_surface(C, "B", n_phi=n_phi, n_theta=n_theta, filename=filename, phi_range=phi_range, theta_range=theta_range)

def plot_poisson_ratio_xz_plane(S: np.ndarray, filename: str):
    # Placeholder for Poisson\'s ratio xz plane plot
    pass

def plot_interactive_3d_surface(S: np.ndarray, prop: str, filename: str):
    # Placeholder for interactive 3D surface plot
    pass

def save_xyz_grid(filename: str, phi: np.ndarray, theta: np.ndarray, values: np.ndarray):
    """Save 3D grid data in XYZ format for Origin."""
    # 转换为笛卡尔坐标
    x = values * np.sin(theta) * np.cos(phi)
    y = values * np.sin(theta) * np.sin(phi)
    z = values * np.cos(theta)
    
    # 保存为四列格式：phi, theta, value, 以及笛卡尔坐
    with open(filename, 'w') as f:
        f.write("Phi(rad)\tTheta(rad)\tValue\tX\tY\tZ\n")
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                f.write(f"{phi[i,j]:.6f}\t{theta[i,j]:.6f}\t{values[i,j]:.6f}\t{x[i,j]:.6f}\t{y[i,j]:.6f}\t{z[i,j]:.6f}\n")

def save_matrix(filename: str, values: np.ndarray):
    """Save matrix data for Origin."""
    with open(filename, 'w') as f:
        f.write("# Matrix data for Origin\n")
        f.write("# Rows: theta, Columns: phi\n")
        np.savetxt(f, values, fmt="%.6f", delimiter="\t")

def save_2d_data_for_origin(filename: str, theta: np.ndarray, values: np.ndarray):
    """Save 2D polar data for Origin."""
    with open(filename, 'w') as f:
        f.write("Angle(rad)\tAngle(deg)\tValue\tX\tY\n")
        for i in range(len(theta)):
            angle_deg = theta[i] * 180 / np.pi
            x = values[i] * np.cos(theta[i])
            y = values[i] * np.sin(theta[i])
            f.write(f"{theta[i]:.6f}\t{angle_deg:.6f}\t{values[i]:.6f}\t{x:.6f}\t{y:.6f}\n")

def save_2d_extrema_data_for_origin(filename: str, theta: np.ndarray, min_values: np.ndarray, max_values: np.ndarray):
    """Save 2D polar extrema data for Origin."""
    with open(filename, 'w') as f:
        f.write("Angle(rad)\tAngle(deg)\tMin_Value\tMax_Value\tMin_X\tMin_Y\tMax_X\tMax_Y\n")
        for i in range(len(theta)):
            angle_deg = theta[i] * 180 / np.pi
            min_x = min_values[i] * np.cos(theta[i])
            min_y = min_values[i] * np.sin(theta[i])
            max_x = max_values[i] * np.cos(theta[i])
            max_y = max_values[i] * np.sin(theta[i])
            f.write(f"{theta[i]:.6f}\t{angle_deg:.6f}\t{min_values[i]:.6f}\t{max_values[i]:.6f}\t{min_x:.6f}\t{min_y:.6f}\t{max_x:.6f}\t{max_y:.6f}\n")

def save_extremal_values(filename: str, prop_type: str, extremal_data: Dict):
    """Save extremal values and directions for Origin."""
    with open(filename, 'w') as f:
        f.write(f"# Extremal values for {prop_type}\n")
        f.write("Type\tValue\tDirection_X\tDirection_Y\tDirection_Z\tPhi(rad)\tTheta(rad)\tPhi(deg)\tTheta(deg)\n")
        
        for key, data in extremal_data.items():
            if 'max' in key.lower():
                value, direction = data
                phi = np.arctan2(direction[1], direction[0])
                theta = np.arccos(direction[2])
                phi_deg = phi * 180 / np.pi
                theta_deg = theta * 180 / np.pi
                f.write(f"Maximum\t{value:.6f}\t{direction[0]:.6f}\t{direction[1]:.6f}\t{direction[2]:.6f}\t{phi:.6f}\t{theta:.6f}\t{phi_deg:.6f}\t{theta_deg:.6f}\n")
            elif 'min' in key.lower():
                value, direction = data
                phi = np.arctan2(direction[1], direction[0])
                theta = np.arccos(direction[2])
                phi_deg = phi * 180 / np.pi
                theta_deg = theta * 180 / np.pi
                f.write(f"Minimum\t{value:.6f}\t{direction[0]:.6f}\t{direction[1]:.6f}\t{direction[2]:.6f}\t{phi:.6f}\t{theta:.6f}\t{phi_deg:.6f}\t{theta_deg:.6f}\n")

def report(sym, stab, vrh, poly_moduli, poly_anisotropy, thermo_props, sound_velocities, stressed_C=None):
    # Placeholder for report generation
    report_str = "=== Elastic Properties Report ===\n\n"
    report_str += f"Crystal System: {sym.system}\n"
    report_str += "Symmetry Criteria:\n"
    for k, v in sym.criteria.items():
        report_str += f"  {k}: {v[0]} ({v[1]})\n"
    report_str += f"\nMechanical Stability: {stab.stable}\n"
    report_str += "Stability Criteria:\n"
    for k, v in stab.criteria.items():
        report_str += f"  {k}: {v[0]} ({v[1]})\n"
    report_str += "\nVRH Averages:\n"
    for k, v in vrh.items():
        report_str += f"  {k}: {v:.6f}\n"
    report_str += "\nAnisotropy Indices:\n"
    for k, v in anisotropy_indices_from_vrh(vrh).items(): # Call the function here
        report_str += f"  {k}: {v:.6f}\n"
    report_str += "\nPolycrystalline Moduli:\n"
    for k, v in poly_moduli.items():
        report_str += f"  {k}: {v:.6f}\n"
    report_str += "\nPolycrystalline Anisotropy Indices:\n"
    for k, v in poly_anisotropy.items():
        report_str += f"  {k}: {v:.6f}\n"
    report_str += "\nThermodynamic Properties:\n"
    for k, v in thermo_props.items():
        report_str += f"  {k}: {v:.6f}\n"
    report_str += "\nSound Velocities:\n"
    for k, v in sound_velocities.items():
        report_str += f"  {k}: {v:.6f}\n"
    if stressed_C is not None:
        report_str += "\nStressed Matrix:\n"
        report_str += np.array2string(stressed_C, precision=3, separator=", ") + "\n"
    return report_str

def suggest_matrix_correction(C: np.ndarray, system: str):
    """Suggest matrix correction based on common issues."""
    # Check for C44-C66 swap in hexagonal/trigonal systems
    if system.lower() in ['hexagonal', 'trigonal']:
        c44 = C[3,3]
        c66 = C[5,5]
        c11 = C[0,0]
        c12 = C[0,1]
        
        # In hexagonal systems, C66 should equal (C11-C12)/2
        expected_c66 = (c11 - c12) / 2
        
        # If C44 is closer to expected C66 than current C66, suggest swap
        if abs(c44 - expected_c66) < abs(c66 - expected_c66):
            return True, f"C44 ({c44:.3f}) appears to be swapped with C66 ({c66:.3f}). Expected C66 = {expected_c66:.3f}"
    
    return False, "No correction suggested."

def swap_c44_c66(C: np.ndarray):
    # Placeholder for C44-C66 swap
    C_swapped = C.copy()
    C_swapped[3,3], C_swapped[5,5] = C_swapped[5,5], C_swapped[3,3]
    return C_swapped

def calculate_thermodynamic_properties(C: np.ndarray, density: float, molar_mass: float):
    """Calculate thermodynamic properties including Debye temperature."""
    S = compliance_matrix(C)
    
    # Calculate VRH averages
    K_VRH = bulk_modulus_vrh(C, S)
    G_VRH = shear_modulus_vrh(C, S)
    
    # Convert density from g/cm³ to kg/m³
    rho = density * 1000
    
    # Calculate average sound velocities
    v_l = np.sqrt((K_VRH + 4*G_VRH/3) * 1e9 / rho)  # Longitudinal velocity (m/s)
    v_t = np.sqrt(G_VRH * 1e9 / rho)  # Transverse velocity (m/s)
    
    # Average sound velocity (Debye model)
    v_m = ((1/3) * (1/v_l**3 + 2/v_t**3))**(-1/3)
    
    # Debye temperature
    h = 6.62607015e-34  # Planck constant (J·s)
    hbar = h / (2.0 * np.pi)  # reduced Planck constant (J·s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    N_A = 6.02214076e23  # Avogadro number

    # Number density of atoms (molar_mass 应为“每摩尔原子”的平均原子质量, g/mol)
    n = (rho * N_A) / (molar_mass / 1000)  # atoms/m³

    # Debye temperature:  θ_D = (ħ/k_B) · v_m · (6π²n)^(1/3)
    # 注意: (6π²n)^(1/3) 形式必须搭配约化普朗克常数 ħ (= h/2π), 不能用 h,
    # 否则结果会偏大 2π≈6.28 倍。等价写法为 (h/k_B)·v_m·(3n/4π)^(1/3)。
    theta_D = (hbar / k_B) * v_m * (6 * np.pi**2 * n)**(1/3)
    
    # Minimum thermal conductivity (Clarke model)
    k_min = 0.87 * k_B * n**(2/3) * v_m
    
    return {
        "debye_temp": theta_D,
        "min_thermal_conductivity": k_min,
        "longitudinal_velocity": v_l,
        "transverse_velocity": v_t,
        "average_velocity": v_m
    }

def calculate_sound_velocities(C: np.ndarray, density: float, direction: np.ndarray):
    """Calculate sound velocities in a given direction using Christoffel equation.

    Constructs the 3×3 Christoffel matrix Γ_ik = C_ijkl n_j n_l / ρ
    from the 6×6 Voigt stiffness matrix C [GPa].
    Eigenvalues give ρv² for one quasi-longitudinal and two quasi-transverse modes.

    Parameters
    ----------
    C : (6, 6) stiffness matrix in Voigt notation [GPa]
    density : mass density [g/cm³]
    direction : 3D propagation direction vector

    Returns
    -------
    dict with keys: longitudinal, transverse1, transverse2, average [m/s]
    """
    if direction.shape != (3,):
        raise ValueError("Direction vector must be a 3-element array.")

    n = direction / np.linalg.norm(direction)

    # Convert density from g/cm³ to kg/m³
    rho = density * 1000.0

    # Voigt index → Cartesian (i,j) pair mapping
    _voigt_to_ij = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    # Build 3×3×3×3 stiffness tensor from 6×6 Voigt C
    C4 = np.zeros((3, 3, 3, 3))
    for I in range(6):
        i, j = _voigt_to_ij[I]
        for J in range(6):
            k, l = _voigt_to_ij[J]
            val = C[I, J] * 1e9  # GPa → Pa
            # Voigt symmetry: each off-diagonal pair appears twice
            C4[i, j, k, l] = val
            C4[j, i, k, l] = val
            C4[i, j, l, k] = val
            C4[j, i, l, k] = val

    # Christoffel matrix: Γ_ik = C_ijkl n_j n_l / ρ
    Gamma = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            s = 0.0
            for j in range(3):
                for l in range(3):
                    s += C4[i, j, k, l] * n[j] * n[l]
            Gamma[i, k] = s / rho

    # Eigenvalues of Christoffel matrix = v²
    eigvals = np.linalg.eigvalsh(Gamma)
    eigvals = np.clip(eigvals, 0.0, None)  # ensure non-negative
    velocities = np.sqrt(eigvals)
    velocities = np.sort(velocities)[::-1]  # descending: longitudinal first

    v_l = velocities[0]       # quasi-longitudinal
    v_t1 = velocities[1]      # quasi-transverse 1
    v_t2 = velocities[2]      # quasi-transverse 2

    return {
        "longitudinal": v_l,
        "transverse1": v_t1,
        "transverse2": v_t2,
        "average": ((1.0 / 3.0) * (1.0 / v_l ** 3 + 1.0 / v_t1 ** 3 + 1.0 / v_t2 ** 3)) ** (-1.0 / 3.0),
    }

def apply_isotropic_stress(C: np.ndarray, pressure: float):
    """Apply isotropic stress to the stiffness matrix (simplified approach)."""
    # This is a simplified approach - a full implementation would require
    # knowledge of pressure derivatives of elastic constants
    # For now, we assume a small linear change
    
    # Apply a small perturbation based on pressure
    # This is a very simplified model
    stress_factor = 1.0 + pressure * 0.001  # 0.1% change per GPa (example)
    
    C_stressed = C * stress_factor
    return C_stressed




def bulk_modulus_directional(C: np.ndarray, n: np.ndarray) -> float:
    """Calculate the directional bulk modulus (linear compressibility inverse) from the stiffness matrix C.
    This is essentially 1 / linear_compressibility_directional(S, n).
    """
    S = compliance_matrix(C)
    return 1.0 / linear_compressibility_directional(S, n)






