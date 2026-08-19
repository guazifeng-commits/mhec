"""
等温 → 绝热弹性常数转换模块。

M-HEC 的 NVT 应力-应变法直接给出的是有限温度**等温**弹性常数 C^T
(应力对应变的导数,应变时体系与热浴保持同温)。而超声/布里渊散射等
实验测量的是**绝热**弹性常数 C^S(声波振荡太快、来不及与环境热交换)。
两者由下式关联 (Wallace, Thermodynamics of Crystals):

    C_ij^S = C_ij^T + (T / (ρ c_v)) · λ_i λ_j

其中
    λ_i = −Σ_k C_ik^T α_k          (热应力系数, Voigt 记号, i = 1…6)
    α   = (α_a, α_b, α_c, 0, 0, 0)  (热膨胀张量, 立方各向同性时三者相等)
    ρ c_v                          (单位体积定容热容, J·m⁻³·K⁻¹)
    T                              (温度, K)

物理结论:
  * 只有法向块 (C11, C12, C13, C22 …) 被抬高; 纯剪切分量 (C44, C55, C66)
    因对应的 λ = 0 而**保持不变**。
  * 差值量 C11−C12、剪切模量 G 均为绝热/等温不变量; 只有体模量 B 增大。

热膨胀系数 α 的来源 (按优先级):
  1. 调用方直接提供 (已知值 → 直接用);
  2. 已计算好的 ThermalExpansionResult (本次运行已算过 → 直接读);
  3. 都没有 → 调用 thermal_expansion 从多温度 NPT 轨迹现算, 然后读取。

定容热容 ρc_v 的来源:
  * 默认 Dulong–Petit:  ρc_v = 3 n k_B  (n = 原子数密度 = N_atoms / V),
    对金属在室温及以上是很好的近似;
  * 可选 d⟨E⟩/dT:  由多温度 MD 的平均总能对温度求导得到 (更严格)。
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

# 物理常数
K_B = 1.380649e-23          # J/K
N_A = 6.02214076e23         # 1/mol
EV_TO_J = 1.602176634e-19   # J/eV
A3_TO_M3 = 1.0e-30          # Å³ → m³


@dataclass
class AdiabaticResult:
    """等温→绝热转换结果。"""
    C_T: np.ndarray                 # 等温刚度矩阵 (GPa, 6×6)
    C_S: np.ndarray                 # 绝热刚度矩阵 (GPa, 6×6)
    dC: np.ndarray                  # 修正量 C_S − C_T (GPa, 6×6)
    temperature: float              # K
    alpha_voigt: np.ndarray         # 热膨胀 (1/K, 6,)
    rho_cv: float                   # 单位体积定容热容 (J·m⁻³·K⁻¹)
    cv_source: str = ""             # "Dulong-Petit" 或 "dE/dT"
    natoms: Optional[int] = None
    volume_A3: Optional[float] = None
    alpha_source: str = ""          # α 的来源说明
    report: str = ""


# ============================================================
#  核心公式
# ============================================================

def thermal_stress(C_T: np.ndarray, alpha_voigt: np.ndarray) -> np.ndarray:
    """热应力系数 λ_i = −Σ_k C_ik^T α_k (与 C_T 同单位, 每 K)。"""
    return -np.asarray(C_T, dtype=float) @ np.asarray(alpha_voigt, dtype=float)


def adiabatic_correction(C_T: np.ndarray, T: float, alpha_voigt: np.ndarray,
                         rho_cv: float) -> np.ndarray:
    """
    计算绝热修正量 ΔC = C_S − C_T (GPa)。

    C_T 单位 GPa; alpha 单位 1/K; rho_cv 单位 J·m⁻³·K⁻¹ (= Pa/K); T 单位 K。
    """
    if rho_cv <= 0:
        raise ValueError("ρc_v 必须为正")
    lam_GPa = thermal_stress(C_T, alpha_voigt)      # GPa/K
    lam_Pa = lam_GPa * 1.0e9                          # Pa/K
    dC_Pa = (T / rho_cv) * np.outer(lam_Pa, lam_Pa)   # Pa
    return dC_Pa / 1.0e9                              # GPa


def isothermal_to_adiabatic(C_T: np.ndarray, T: float, alpha_voigt: np.ndarray,
                            rho_cv: float) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (C_S, ΔC), 单位 GPa。"""
    C_T = np.asarray(C_T, dtype=float)
    dC = adiabatic_correction(C_T, T, alpha_voigt, rho_cv)
    return C_T + dC, dC


# ============================================================
#  定容热容 ρc_v
# ============================================================

def dulong_petit_rho_cv(natoms: int, volume_A3: float) -> float:
    """Dulong–Petit 单位体积定容热容: ρc_v = 3 n k_B (J·m⁻³·K⁻¹)。"""
    if natoms <= 0 or volume_A3 <= 0:
        raise ValueError("natoms 和 volume 必须为正")
    n = natoms / (volume_A3 * A3_TO_M3)   # 原子数密度 (m⁻³)
    return 3.0 * n * K_B


def rho_cv_from_density(density_g_cm3: float, molar_mass_g_mol: float) -> float:
    """
    Dulong–Petit + 密度/摩尔质量估计 ρc_v。

    molar_mass 以"每原子平均摩尔质量"计 (单原子如 Al 即原子量;
    化合物需除以化学式原子数)。
    """
    if density_g_cm3 <= 0 or molar_mass_g_mol <= 0:
        raise ValueError("密度和摩尔质量必须为正")
    rho = density_g_cm3 * 1.0e3                 # kg/m³
    M = molar_mass_g_mol * 1.0e-3               # kg/mol (每原子)
    n = rho * N_A / M                           # 原子数密度 (m⁻³)
    return 3.0 * n * K_B


def rho_cv_from_energy(T_arr, E_arr_eV, volume_A3: float, T: float) -> float:
    """
    d⟨E⟩/dT 路线: 由多温度 MD 平均总能 (eV, 每个 cell) 对温度线性/多项式拟合,
    在温度 T 处取导数, 再除以体积得单位体积定容热容 (J·m⁻³·K⁻¹)。
    """
    T_arr = np.asarray(T_arr, dtype=float)
    E_arr = np.asarray(E_arr_eV, dtype=float)
    if len(T_arr) < 2:
        raise ValueError("d⟨E⟩/dT 至少需要 2 个温度点")
    order = min(2, len(T_arr) - 1)
    poly = np.polyfit(T_arr, E_arr, order)
    dEdT_eV = np.polyval(np.polyder(poly), T)   # eV/K (每 cell)
    Cv_cell = dEdT_eV * EV_TO_J                  # J/K (每 cell)
    return Cv_cell / (volume_A3 * A3_TO_M3)      # J·m⁻³·K⁻¹


# ============================================================
#  热膨胀系数获取
# ============================================================

def alpha_at_T(te_result, T: float) -> np.ndarray:
    """
    从 ThermalExpansionResult 在温度 T 处取线膨胀 (α_a, α_b, α_c),
    组装为 Voigt 应变热膨胀向量 (α_a, α_b, α_c, 0, 0, 0)。
    """
    def _alpha(poly):
        val = np.polyval(poly, T)
        dval = np.polyval(np.polyder(poly), T)
        return dval / val if abs(val) > 1e-30 else 0.0
    aa = _alpha(te_result.poly_a)
    ab = _alpha(te_result.poly_b)
    ac = _alpha(te_result.poly_c)
    return np.array([aa, ab, ac, 0.0, 0.0, 0.0])


def volume_at_T(te_result, T: float) -> float:
    """从 ThermalExpansionResult 在温度 T 处取平衡体积 (ų)。"""
    return float(np.polyval(te_result.poly_V, T))


def compute_te_from_npt(npt_base: str, config, skip_frac: float = 0.5):
    """
    从多温度 NPT 轨迹现算热膨胀 (复用 func_801 的数据通路)。

    Returns
    -------
    (te_result, natoms)  —  te_result 可能为 None (温度点不足时)。
    """
    import re
    from .vasp_io import read_poscar
    from .lattice_optimizer import LatticeOptimizer

    if not os.path.isdir(npt_base):
        raise FileNotFoundError(f"NPT 目录不存在: {npt_base}")

    temps = []
    for d in sorted(os.listdir(npt_base)):
        m = re.match(r"npt_(\d+)K$", d)
        if m and os.path.isdir(os.path.join(npt_base, d)):
            temps.append(float(m.group(1)))
    if not temps:
        raise FileNotFoundError(f"在 {npt_base} 中未找到 npt_*K 目录")

    first_dir = os.path.join(npt_base, f"npt_{int(temps[0])}K")
    template = None
    for fname in ["CONTCAR", "POSCAR"]:
        for d in [first_dir, os.path.join(first_dir, "run")]:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                template = read_poscar(p)
                break
        if template:
            break
    if template is None:
        raise FileNotFoundError("无法读取 CONTCAR/POSCAR 模板")

    natoms = int(sum(template["counts"]))

    optimizer = LatticeOptimizer(
        poscar=template,
        temperatures=temps,
        nsw_train=getattr(config, "nsw_train", 1000),
        nsw_refit=getattr(config, "nsw_refit", 1000),
        nsw_run=getattr(config, "nsw_run", 2000),
        potim=getattr(config, "potim", 1.0),
        encut=getattr(config, "encut", None),
        user_overrides=getattr(config, "incar_overrides", None),
        skip_steps=getattr(config, "skip_steps", 0),
    )
    _lattice_data, te_result = optimizer.extract_all_equilibrium(
        npt_base, skip_frac=skip_frac, run_only=True)
    return te_result, natoms


# ============================================================
#  高层编排
# ============================================================

def run_conversion(
    C_T: np.ndarray,
    T: float,
    *,
    alpha_voigt=None,
    te_result=None,
    npt_base=None,
    config=None,
    skip_frac: float = 0.5,
    natoms: Optional[int] = None,
    volume_A3: Optional[float] = None,
    density: Optional[float] = None,
    molar_mass: Optional[float] = None,
    cv_per_volume: Optional[float] = None,
    energy_TE: Optional[Tuple] = None,
) -> AdiabaticResult:
    """
    等温 C^T → 绝热 C^S 的完整转换编排。

    α 来源优先级: alpha_voigt > te_result > 现算(npt_base)。
    ρc_v 来源优先级: cv_per_volume > energy_TE(d⟨E⟩/dT) > Dulong–Petit。
    """
    C_T = np.asarray(C_T, dtype=float)
    alpha_source = ""

    # ---- 1. 获取热膨胀 α ----
    if alpha_voigt is not None:
        alpha_voigt = np.asarray(alpha_voigt, dtype=float)
        if alpha_voigt.size == 3:
            alpha_voigt = np.array([alpha_voigt[0], alpha_voigt[1], alpha_voigt[2], 0, 0, 0])
        alpha_source = "用户提供"
    else:
        if te_result is None:
            if npt_base is None:
                raise ValueError("需要 alpha_voigt、te_result 或 npt_base 之一来获取热膨胀系数")
            te_result, natoms_npt = compute_te_from_npt(npt_base, config, skip_frac)
            if natoms is None:
                natoms = natoms_npt
            alpha_source = f"从 NPT 现算 ({npt_base})"
        else:
            alpha_source = "已有热膨胀结果"
        if te_result is None:
            raise ValueError("热膨胀计算失败 (温度点不足?)")
        alpha_voigt = alpha_at_T(te_result, T)
        if volume_A3 is None:
            volume_A3 = volume_at_T(te_result, T)

    # ---- 2. 获取定容热容 ρc_v ----
    if cv_per_volume is not None:
        rho_cv = float(cv_per_volume)
        cv_source = "外部提供"
    elif energy_TE is not None:
        if volume_A3 is None:
            raise ValueError("d⟨E⟩/dT 需要平衡体积 volume_A3")
        rho_cv = rho_cv_from_energy(energy_TE[0], energy_TE[1], volume_A3, T)
        cv_source = "d⟨E⟩/dT"
    elif natoms is not None and volume_A3 is not None:
        rho_cv = dulong_petit_rho_cv(natoms, volume_A3)
        cv_source = "Dulong-Petit (n=N/V)"
    elif density is not None and molar_mass is not None:
        rho_cv = rho_cv_from_density(density, molar_mass)
        cv_source = "Dulong-Petit (密度/摩尔质量)"
    else:
        raise ValueError("无法确定 ρc_v: 请提供 natoms+volume、density+molar_mass、"
                         "cv_per_volume 或 energy_TE 之一")

    # ---- 3. 转换 ----
    C_S, dC = isothermal_to_adiabatic(C_T, T, alpha_voigt, rho_cv)

    res = AdiabaticResult(
        C_T=C_T, C_S=C_S, dC=dC, temperature=T,
        alpha_voigt=alpha_voigt, rho_cv=rho_cv, cv_source=cv_source,
        natoms=natoms, volume_A3=volume_A3, alpha_source=alpha_source,
    )
    res.report = format_report(res)
    return res


def _fmt_matrix(C: np.ndarray) -> str:
    return "\n".join("  " + "  ".join(f"{v:9.3f}" for v in row) for row in C)


def format_report(res: AdiabaticResult) -> str:
    """生成文本报告。"""
    a = res.alpha_voigt
    lines = []
    lines.append("=" * 64)
    lines.append("  等温 → 绝热弹性常数转换  (Isothermal → Adiabatic)")
    lines.append("=" * 64)
    lines.append(f"  温度 T                : {res.temperature:.1f} K")
    lines.append(f"  热膨胀来源            : {res.alpha_source}")
    lines.append(f"  线膨胀 α_a,α_b,α_c    : "
                 f"{a[0]*1e6:.3f}, {a[1]*1e6:.3f}, {a[2]*1e6:.3f}  (×10⁻⁶/K)")
    if res.natoms is not None and res.volume_A3 is not None:
        lines.append(f"  原子数 / 体积         : {res.natoms} / {res.volume_A3:.3f} ų")
    lines.append(f"  定容热容 ρc_v         : {res.rho_cv:.4e} J·m⁻³·K⁻¹  [{res.cv_source}]")
    lines.append("")
    lines.append("  等温刚度矩阵 C^T (GPa):")
    lines.append(_fmt_matrix(res.C_T))
    lines.append("")
    lines.append("  绝热修正量 ΔC = C^S − C^T (GPa):")
    lines.append(_fmt_matrix(res.dC))
    lines.append("")
    lines.append("  绝热刚度矩阵 C^S (GPa):")
    lines.append(_fmt_matrix(res.C_S))
    lines.append("")
    # 关键标量
    B_T = (res.C_T[0, 0] + res.C_T[1, 1] + res.C_T[2, 2]
           + 2 * (res.C_T[0, 1] + res.C_T[0, 2] + res.C_T[1, 2])) / 9.0
    B_S = (res.C_S[0, 0] + res.C_S[1, 1] + res.C_S[2, 2]
           + 2 * (res.C_S[0, 1] + res.C_S[0, 2] + res.C_S[1, 2])) / 9.0
    lines.append(f"  体模量  B_T = {B_T:.3f} → B_S = {B_S:.3f} GPa  (ΔB = {B_S - B_T:.3f})")
    lines.append("  注: 纯剪切分量 (C44/C55/C66) 与差值量 C11−C12 为绝热/等温不变量。")
    lines.append("=" * 64)
    return "\n".join(lines)


def save_result(res: AdiabaticResult, out_dir: str, prefix: str = "elastic") -> Tuple[str, str]:
    """保存绝热矩阵与报告, 返回 (matrix_path, report_path)。"""
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, f"{prefix}_adiabatic.txt")
    np.savetxt(matrix_path, res.C_S, fmt="%.6f")
    report_path = os.path.join(out_dir, f"{prefix}_adiabatic_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(res.report + "\n")
    return matrix_path, report_path
