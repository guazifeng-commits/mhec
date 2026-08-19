"""
MHEC 交互式命令行界面。

参考 VASPKIT 风格设计，所有功能平铺在主菜单中，无二级菜单。
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List, Dict

from . import __version__
from .config import MHECConfig
from .i18n import T, set_lang, get_lang, save_lang, pref_path
from .crystal_system import (
    CrystalSystem, identify_crystal_system, lattice_params,
    N_INDEPENDENT, CRYSTAL_SYSTEM_NAMES, cs_name, detect_supercell_mismatch,
)
from .strain import (
    get_deform_codes, get_amplitude_labels, generate_strained_structures,
    compare_strain_methods, decode_deform, apply_strain,
)
from .vasp_io import read_poscar, write_poscar, write_incar, write_kpoints_gamma
from .incar_templates import build_incar_params
from .dir_manager import DirManager
from .slurm import SlurmConfig, write_slurm_scripts
from .report import (
    print_cij_matrix, save_elastic_matrix, write_results_file,
    print_comparison_table,
)
from .plugins import PluginRegistry
from .stress_extractor import StressExtractor
from .elastic_fitter import ElasticFitter, robust_slope_fit
from .progress import ProgressBar, track
from .strain_vc import get_vc_modes, generate_vc_structures, VC_SUPPORTED, vc_mode_direction
from .elastic_fitter_vc import VCElasticFitter
from .mlff_workflow import MLFFWorkflow
from .mlff_validator import (
    parse_ml_logfile,
    print_validation_report,
)
from .lattice_optimizer import LatticeOptimizer


# ============================================================
# 界面显示
# ============================================================

# ANSI 颜色码
class _C:
    """终端 ANSI 颜色。"""
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[37m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


def _color_enabled() -> bool:
    """检测终端是否支持颜色。"""
    import sys
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


# 运行时决定是否启用颜色
_USE_COLOR = None

def _c(code: str) -> str:
    """返回颜色码，如果终端不支持则返回空字符串。"""
    global _USE_COLOR
    if _USE_COLOR is None:
        _USE_COLOR = _color_enabled()
    return code if _USE_COLOR else ""


def _dashed_line(color: str, width: int = 60) -> str:
    """生成带颜色的虚线。"""
    return f"{_c(color)}{'─' * width}{_c(_C.RESET)}"


def _disp_w(s: str) -> int:
    """计算字符串显示宽度 (中文/全角字符按 2 计)。"""
    import unicodedata
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _item(num: str, label: str, color: str, width: int = 28) -> str:
    """渲染一个菜单条目单元格 (编号着色 + 标签), 按显示宽度右补空格以便两列对齐。"""
    plain = f"{num:>2}) {label}"
    pad = " " * max(0, width - _disp_w(plain))
    return f"  {_c(color)}{num:>2}){_c(_C.RESET)} {label}{pad}"


def _section(title: str, color: str) -> None:
    """打印分区标题 (上下细线 + 居左标题)。"""
    print(f" {_dashed_line(color)}")
    print(f"  {_c(color)}{_c(_C.BOLD)}{title}{_c(_C.RESET)}")
    print(f" {_dashed_line(color)}")


_LOGO_LINES = [
    "  ███╗   ███╗ ██╗  ██╗ ███████╗  ██████╗ ",
    "  ████╗ ████║ ██║  ██║ ██╔════╝ ██╔════╝ ",
    "  ██╔████╔██║ ████████║ █████╗   ██║      ",
    "  ██║╚██╔╝██║ ██╔══██║ ██╔══╝   ██║      ",
    "  ██║ ╚═╝ ██║ ██║  ██║ ███████╗  ██████╗ ",
    "  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚══════╝  ╚═════╝ ",
]


# ============================================================
# 辅助函数
# ============================================================

def show_banner():
    """显示程序横幅，LOGO + 信息用虚线框包裹。"""
    W = 60
    line = _dashed_line(_C.CYAN, W)
    print()
    print(f" {line}")
    for row in _LOGO_LINES:
        print(f"  {_c(_C.CYAN)}{row}{_c(_C.RESET)}")
    print()
    print(f"  {_c(_C.BOLD)}MLFF-Enhanced High-Temperature Elastic Constants{_c(_C.RESET)}")
    print()
    print(f"  {_c(_C.DIM)}Version   :{_c(_C.RESET)} {_c(_C.GREEN)}{__version__}{_c(_C.RESET)}")
    print(f"  {_c(_C.DIM)}Developer :{_c(_C.RESET)} Dr. Feng Xing")
    print(f"  {_c(_C.DIM)}Email     :{_c(_C.RESET)} fengxing@amgm.ac.cn")
    print(f" {line}")


def _detect_system_label() -> str:
    """尝试从当前目录的 POSCAR 读取体系信息。"""
    if not os.path.isfile("POSCAR"):
        return ""
    try:
        with open("POSCAR", 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if len(lines) < 7:
            return ""
        tok5 = lines[5].split()
        import re as _re
        if all(_re.match(r'^[A-Za-z]{1,2}$', t) for t in tok5):
            elems = tok5
            counts = [int(x) for x in lines[6].split()]
            parts = [f"{e}{c}" for e, c in zip(elems, counts)]
            return " ".join(parts) + f" ({sum(counts)} atoms)"
    except Exception:
        pass
    return ""


def show_status(config: MHECConfig):
    """显示当前配置状态。"""
    # 体系信息：优先从 POSCAR 读取
    sys_label = _detect_system_label()
    if sys_label:
        system_display = sys_label
    else:
        system_display = T("未检测到 POSCAR", "No POSCAR detected")

    # 晶系
    if config.crystal_system:
        from .crystal_system import CrystalSystem, CRYSTAL_SYSTEM_NAMES
        try:
            cs = CrystalSystem(config.crystal_system)
            cs_display = CRYSTAL_SYSTEM_NAMES.get(cs, config.crystal_system)
        except ValueError:
            cs_display = config.crystal_system
    else:
        cs_display = T("自动识别", "auto-detect")

    temps = ", ".join(str(int(t)) for t in config.temperatures)
    mag = config.magnitude
    pts = config.n_points
    from .strain import get_amplitude_labels
    amp = get_amplitude_labels(pts)
    half = len(amp) // 2
    strain_display = T(f"{pts}点 (±{half}δ, δ={mag})", f"{pts} pts (±{half}δ, δ={mag})")
    print()
    print(f" {_dashed_line(_C.DIM)}")
    print(f"  {_c(_C.DIM)}{T('当前体系', 'System')}: {_c(_C.RESET)}{_c(_C.GREEN)}{system_display}{_c(_C.RESET)}    "
          f"{_c(_C.DIM)}{T('晶系', 'Crystal')}: {_c(_C.RESET)}{cs_display}")
    print(f"  {_c(_C.DIM)}{T('计算温度', 'Temperatures')}: {_c(_C.RESET)}[{temps}] K    "
          f"{_c(_C.DIM)}{T('应变设置', 'Strain')}: {_c(_C.RESET)}{strain_display}")
    print(f" {_dashed_line(_C.DIM)}")


def show_menu():
    """显示主菜单 (单列, 分区清晰, 标签简洁清楚, 连续编号)。"""

    def it(num, label, color):
        print(f"  {_c(color)}{num:>2}){_c(_C.RESET)} {label}")

    print()
    _section(T("弹性常数计算", "Elastic Constants"), _C.CYAN)
    it("1", T("SC-SS 单分量应力-应变法 · 生成计算输入",
             "SC-SS Single-Component Stress-Strain · Generate inputs"), _C.CYAN)
    it("2", T("SC-SS 单分量应力-应变法 · 拟合弹性常数",
             "SC-SS Single-Component Stress-Strain · Fit elastic constants"), _C.CYAN)
    it("3", T("SA-SS 对称化应力-应变法 · 生成计算输入",
             "SA-SS Symmetry-Adapted Stress-Strain · Generate inputs"), _C.CYAN)
    it("4", T("SA-SS 对称化应力-应变法 · 拟合弹性常数",
             "SA-SS Symmetry-Adapted Stress-Strain · Fit elastic constants"), _C.CYAN)
    print()
    _section(T("后处理分析", "Post-processing"), _C.GREEN)
    it("5", T("力学性质后处理", "Elastic post-processing"), _C.GREEN)
    it("6", T("AIMD 后处理", "AIMD post-processing"), _C.GREEN)
    print()
    _section(T("机器学习力场", "Machine-Learning Force Field"), _C.BLUE)
    it("7", T("MLFF 训练 (定温)", "MLFF training (isothermal)"), _C.BLUE)
    it("8", T("MLFF 训练 (升温)", "MLFF training (heating ramp)"), _C.BLUE)
    it("9", T("MLFF 精度验证", "MLFF accuracy validation"), _C.BLUE)
    print()
    _section(T("辅助工具", "Utilities"), _C.YELLOW)
    it("10", T("晶系识别与晶格参数", "Crystal system & lattice parameters"), _C.YELLOW)
    it("11", T("变形模式与应变方案", "Deformation modes & strain schemes"), _C.YELLOW)
    it("12", T("平衡晶格与热膨胀系数", "Equilibrium lattice & thermal expansion"), _C.YELLOW)
    it("13", T("等温和绝热弹性常数转换", "Isothermal and adiabatic conversion"), _C.YELLOW)
    print()
    print(f" {_dashed_line(_C.DIM)}")
    print(f"  {_c(_C.MAGENTA)} L){_c(_C.RESET)} {T('界面语言', 'Language')}      "
          f"{_c(_C.MAGENTA)} 0){_c(_C.RESET)} {T('退出', 'Exit')}")
    print(f" {_dashed_line(_C.DIM)}")


def prompt_with_default(prompt: str, default: str) -> str:
    """带默认值的用户输入提示。"""
    user_input = input(f" {prompt} [{default}]: ").strip()
    return user_input if user_input else default


def validate_input(value: str, valid_options: List[str]) -> Optional[str]:
    """验证用户输入。"""
    return value if value in valid_options else None


def sep(title: str = ""):
    """打印分隔线。"""
    if title:
        print(f"\n -->> {title}")
        print(f" {'─' * 60}")
    else:
        print(f" {'─' * 60}")


def ok(msg: str):
    print(f"  +-> {msg}")


def warn(msg: str):
    print(f"  !-> {msg}")


def info(msg: str):
    print(f"  --> {msg}")


# ============================================================
# 通用交互函数
# ============================================================

def _read_poscar_interactive() -> Dict:
    """交互式读取 POSCAR 文件。"""
    while True:
        path = prompt_with_default(T("POSCAR 文件路径", "POSCAR file path"), "POSCAR")
        if os.path.isfile(path):
            try:
                poscar = read_poscar(path)
                ok(T(f"已读取: {path}", f"Loaded: {path}"))
                return poscar
            except Exception as e:
                warn(T(f"读取失败: {e}", f"Failed to read: {e}"))
        else:
            warn(T(f"文件不存在: {path}", f"File not found: {path}"))


def _identify_crystal_interactive(poscar: Dict, tol: float = 0.01,
                                   config: Optional[MHECConfig] = None) -> CrystalSystem:
    """交互式晶系识别与确认。"""

    # 策略0: 配置中指定了空间群 → 最可靠, 直接推导晶系 + Laue class
    if config and getattr(config, "space_group", None):
        try:
            from .crystal_system import space_group_to_crystal_system
            cs_sg = space_group_to_crystal_system(int(config.space_group))
            ok(T(f"由空间群 #{config.space_group} 推导晶系: {cs_name(cs_sg)} "
                 f"({N_INDEPENDENT[cs_sg]} 个独立常数)",
                 f"Crystal system from space group #{config.space_group}: {cs_name(cs_sg)} "
                 f"({N_INDEPENDENT[cs_sg]} independent constants)"))
            return cs_sg
        except (ValueError, TypeError):
            warn(T(f"配置中的空间群 '{config.space_group}' 无效，尝试其他方式。",
                   f"Invalid space group '{config.space_group}' in config; trying other methods."))

    # 策略1: 配置中已指定晶系
    if config and config.crystal_system:
        for cs_opt in CrystalSystem:
            if cs_opt.value == config.crystal_system:
                name = cs_name(cs_opt)
                ok(T(f"配置文件指定晶系: {name} ({N_INDEPENDENT[cs_opt]} 个独立常数)",
                     f"Crystal system from config: {name} ({N_INDEPENDENT[cs_opt]} independent constants)"))
                return cs_opt
        warn(T(f"配置中的晶系 '{config.crystal_system}' 无效，将进行自动识别。",
               f"Invalid crystal system '{config.crystal_system}' in config; auto-detecting."))

    lattice = poscar["lattice"]
    a, b, c, alpha, beta, gamma = lattice_params(lattice)
    print(T("\n  晶格参数:", "\n  Lattice parameters:"))
    print(f"    a = {a:.4f}  b = {b:.4f}  c = {c:.4f}")
    print(f"    α = {alpha:.2f}°  β = {beta:.2f}°  γ = {gamma:.2f}°")

    # 自动识别: 优先 spglib (原子位置 → 空间群 → 晶系, 能区分 Laue class 且抗 MD 噪声),
    # 失败再退回几何法 (只看晶格参数)
    from .crystal_system import spacegroup_number_from_structure, space_group_to_crystal_system
    cs_auto = None
    _sg = spacegroup_number_from_structure(poscar, symprec=max(tol, 1e-3))
    if _sg is not None:
        cs_auto = space_group_to_crystal_system(_sg)
        ok(T(f"spglib 识别空间群 #{_sg} → {cs_name(cs_auto)} ({N_INDEPENDENT[cs_auto]} 个独立常数)",
             f"spglib space group #{_sg} → {cs_name(cs_auto)} ({N_INDEPENDENT[cs_auto]} independent constants)"))
    else:
        cs_auto = identify_crystal_system(lattice, tol)
        info(T("spglib 未识别 (未装或结构信息不全), 改用几何法; 建议 pip install spglib 以获得可靠 Laue class",
               "spglib unavailable; using geometry method; install spglib for reliable Laue class"))

    # 尝试从原胞识别 (仅当 spglib 未给出结果时的补充)
    prim_cs = None
    prim_path = config.primitive_poscar if config else None
    if _sg is None and prim_path and os.path.isfile(prim_path):
        try:
            prim_poscar = read_poscar(prim_path)
            prim_cs = identify_crystal_system(prim_poscar["lattice"], tol)
        except Exception:
            pass

    detected = prim_cs if prim_cs else cs_auto
    detected_name = cs_name(detected)

    # 列出所有晶系，标记自动识别结果
    options = list(CrystalSystem)
    print(T(f"\n  晶系选择 (自动识别: {detected_name}):",
            f"\n  Select crystal system (auto-detected: {detected_name}):"))
    print(f"  {'─' * 50}")
    marker_txt = T(" ◀ 自动识别", " ◀ auto")
    ind_txt = T("个独立常数", "independent constants")
    for i, opt in enumerate(options):
        opt_name = cs_name(opt)
        n_ind = N_INDEPENDENT[opt]
        marker = marker_txt if opt == detected else ""
        print(f"    {i}) {opt_name:14s} ({n_ind:>2d} {ind_txt}){marker}")
    print(f"  {'─' * 50}")

    choice = prompt_with_default(
        T("选择晶系编号 (直接回车确认自动识别)",
          "Enter crystal system number (Enter to accept auto-detected)"), "")
    if choice == "":
        cs = detected
    elif choice.isdigit() and 0 <= int(choice) < len(options):
        cs = options[int(choice)]
    else:
        warn(T(f"无效输入 '{choice}'，使用自动识别结果。",
               f"Invalid input '{choice}'; using auto-detected result."))
        cs = detected

    name = cs_name(cs)
    ok(T(f"晶系: {name} ({N_INDEPENDENT[cs]} 个独立弹性常数)",
         f"Crystal system: {name} ({N_INDEPENDENT[cs]} independent elastic constants)"))

    # 单一事实来源: 把最终确定的晶系 (及 spglib 识别到的空间群) 回写进 config,
    # 由调用方 config.to_file() 持久化到 mhec.yaml, 供后续 NVT 更新与拟合直接复用,
    # 不再各自重新识别 (这是之前 "SA 认 / SS 不认 / 立方被判正交" 的根因)。
    if config is not None:
        config.crystal_system = cs.value
        if _sg is not None and not getattr(config, "space_group", None):
            config.space_group = _sg
    return cs


# ============================================================
# 101) 晶系自动识别
# ============================================================

def func_101(config: MHECConfig) -> None:
    sep(T("晶系识别与晶格参数", "Crystal system & lattice parameters"))
    poscar = _read_poscar_interactive()
    lattice = poscar["lattice"]
    a, b, c, alpha, beta, gamma = lattice_params(lattice)
    vol = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
    print(T("\n  晶格参数:", "\n  Lattice parameters:"))
    print(f"    a = {a:.6f} Å    α = {alpha:.4f}°")
    print(f"    b = {b:.6f} Å    β = {beta:.4f}°")
    print(f"    c = {c:.6f} Å    γ = {gamma:.4f}°")
    print(f"    V = {vol:.4f} ų")

    # 组成 / 摩尔质量 / 密度 (需 POSCAR 含元素符号行)
    from .periodic import composition_info, get_spacegroup
    species = poscar.get("species")
    counts = poscar.get("counts") or []
    comp = composition_info(species, counts, vol)
    print(T("\n  组成与物性:", "\n  Composition & properties:"))
    if comp is not None:
        elem_str = " ".join(f"{s}{c}" for s, c in zip(species, counts))
        print(T(f"    原子组成: {elem_str}  (共 {comp['n_atoms']} 原子)",
                f"    Composition: {elem_str}  ({comp['n_atoms']} atoms)"))
        print(T(f"    化学式:   {comp['formula']}  (Z = {comp['Z']})",
                f"    Formula:   {comp['formula']}  (Z = {comp['Z']})"))
        print(T(f"    摩尔质量: {comp['mass_formula']:.4f} g/mol (每化学式单元)",
                f"    Molar mass: {comp['mass_formula']:.4f} g/mol (per formula unit)"))
        print(T(f"              {comp['mass_cell']:.4f} g/mol (每晶胞) / "
                f"{comp['mass_per_atom']:.4f} g/mol (每原子平均)",
                f"              {comp['mass_cell']:.4f} g/mol (per cell) / "
                f"{comp['mass_per_atom']:.4f} g/mol (per atom)"))
        print(T(f"    密度:     {comp['density']:.4f} g/cm³",
                f"    Density:   {comp['density']:.4f} g/cm³"))
    elif not species:
        warn(T("POSCAR 缺少元素符号行, 无法计算摩尔质量/密度 (请在 POSCAR 第 6 行补上元素符号)。",
               "POSCAR lacks the element-symbol line; cannot compute molar mass/density (add element symbols on line 6)."))
    else:
        warn(T("POSCAR 含未知元素符号, 无法计算摩尔质量/密度。",
               "POSCAR contains unknown element symbols; cannot compute molar mass/density."))

    # 晶系 (mhec 自带) + 空间群 (需 spglib)
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    spg = get_spacegroup(lattice, poscar.get("positions"), poscar.get("coord_type"),
                         species, counts, symprec=max(config.crystal_tol, 1e-3))
    if spg:
        ok(T(f"空间群: {spg}", f"Space group: {spg}"))
    else:
        info(T("空间群: 需安装 spglib 才能识别 (pip install spglib);"
               " 已给出晶系与独立弹性常数数目。",
               "Space group: install spglib to enable detection (pip install spglib);"
               " crystal system and independent-constant count are shown above."))


# ============================================================
# 102) 晶格参数计算
# ============================================================

def func_102(config: MHECConfig) -> None:
    sep("晶格参数计算")
    poscar = _read_poscar_interactive()
    lattice = poscar["lattice"]
    a, b, c, alpha, beta, gamma = lattice_params(lattice)
    vol = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
    print(f"\n  晶格矢量 (Å):")
    for i, label in enumerate(["a", "b", "c"]):
        v = lattice[i]
        print(f"    {label} = ({v[0]:12.6f}, {v[1]:12.6f}, {v[2]:12.6f})")
    print(f"\n  晶格参数:")
    print(f"    a = {a:.6f} Å    α = {alpha:.4f}°")
    print(f"    b = {b:.6f} Å    β = {beta:.4f}°")
    print(f"    c = {c:.6f} Å    γ = {gamma:.4f}°")
    print(f"    V = {vol:.4f} ų")


# ============================================================
# 201) 应变方案生成 (Standard)
# ============================================================

def func_201(config: MHECConfig) -> None:
    sep("应变方案生成 (Standard)")
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    codes = get_deform_codes(cs, "standard")
    _print_deform_codes(codes, config.magnitude, config.n_points)


# ============================================================
# 202) 应变方案生成 (ULICS)
# ============================================================

def func_202(config: MHECConfig) -> None:
    sep("应变方案生成 (ULICS)")
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    codes = get_deform_codes(cs, "ulics")
    _print_deform_codes(codes, config.magnitude, config.n_points)


def _print_deform_codes(codes: List[str], magnitude: float, n_points: int):
    """打印 deform codes 和幅度点信息。"""
    print(f"\n  Deform Codes ({len(codes)}):")
    print(f"  {'─' * 50}")
    for code in codes:
        strain = decode_deform(code, magnitude)
        nonzero = [f"ε{i+1}={strain[i]:+.4f}" for i in range(6) if abs(strain[i]) > 1e-15]
        print(f"    {code}  →  {', '.join(nonzero)}")
    amp_labels = get_amplitude_labels(n_points)
    print(T(f"\n  幅度点 (δ = {magnitude}):", f"\n  Amplitude points (δ = {magnitude}):"))
    for label, mult in amp_labels:
        print(f"    {label}: {mult:+d}δ = {mult * magnitude:+.4f}")
    print(T(f"\n  总计算数: {len(codes)} × {len(amp_labels)} = {len(codes) * len(amp_labels)}",
            f"\n  Total calculations: {len(codes)} × {len(amp_labels)} = {len(codes) * len(amp_labels)}"))


# ============================================================
# 203) 应变方案对比分析
# ============================================================

def func_203(config: MHECConfig) -> None:
    sep(T("查看变形模式", "Deformation modes"))
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)

    # SC-SS 单分量应力-应变法 (通用单分量应变集)
    print(f"\n  {_c(_C.CYAN)}" + T("SC-SS 单分量应力-应变法 变形模式",
                                   "SC-SS Single-Component Stress-Strain deformation modes") + f"{_c(_C.RESET)}")
    codes = get_deform_codes(cs, "standard")
    _print_deform_codes(codes, config.magnitude, config.n_points)

    # SA-SS 对称化应力-应变法
    print(f"\n  {_c(_C.CYAN)}" + T("SA-SS 对称化应力-应变法 变形模式",
                                   "SA-SS Symmetry-Adapted Stress-Strain deformation modes") + f"{_c(_C.RESET)}")
    if cs in VC_SUPPORTED:
        modes = get_vc_modes(cs)
        print(f"  {'─' * 50}")
        for m in modes:
            print(f"    {m['name']:<16} → {m.get('target', '')}")
        print(T(f"\n  共 {len(modes)} 个变形模式 × {len(get_amplitude_labels(config.n_points))} 幅度点",
                f"\n  {len(modes)} deformation modes × {len(get_amplitude_labels(config.n_points))} amplitude points"))
    else:
        print(T(f"  (SA-SS 对称化应力-应变法暂不支持 {cs_name(cs)})",
                f"  (SA-SS Symmetry-Adapted method not yet supported for {cs_name(cs)})"))


# ============================================================
# 204) 生成变形结构文件
# ============================================================

def func_204(config: MHECConfig) -> None:
    sep("生成变形结构文件")
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    method = "standard"
    codes = get_deform_codes(cs, method)
    out_dir = prompt_with_default("输出目录", "deformed_structures")
    os.makedirs(out_dir, exist_ok=True)
    strained = generate_strained_structures(
        poscar["lattice"], codes, config.magnitude, config.n_points
    )
    count = 0
    for code in codes:
        for label, _ in get_amplitude_labels(config.n_points):
            deformed_poscar = dict(poscar)
            deformed_poscar["lattice"] = strained[code][label]
            fname = f"POSCAR_{code}_{label}"
            write_poscar(os.path.join(out_dir, fname), deformed_poscar)
            count += 1
    ok(f"已生成 {count} 个变形结构文件至 {out_dir}/")

    # ---- 防呆: 静态应力-应变必须弛豫内部原子 (钳定离子 vs 弛豫离子) ----
    natoms = int(sum(poscar.get("counts") or [1]))
    if natoms > 1:
        warn(T(
            "重要 — 这些是静态应力-应变变形结构。除单原子布拉菲晶格(如 fcc/bcc 金属)外,含内部原子"
            "自由度的晶体(BeF2、SiO2 等框架/多原子基)必须在固定变形胞下弛豫内部原子,否则得到的是"
            "\u201c钳定离子\u201d弹性常数,可能比真实(弛豫离子)值大数倍,甚至使 C_ij 变号。",
            "Important — these are static stress-strain deformation structures. Except for monatomic "
            "Bravais lattices (e.g. fcc/bcc metals), crystals with internal atomic degrees of freedom "
            "(framework/multi-atom-basis such as BeF2, SiO2) MUST relax the internal ions at the fixed "
            "deformed cell; otherwise the result is the clamped-ion elastic tensor, which can be several "
            "times too stiff and can even flip the sign of C_ij."))
        info(T("推荐 INCAR(变形单点:固定晶胞、只弛豫离子):",
               "Recommended INCAR (deformed single point: fix cell, relax ions only):"))
        _relax_incar = (
            "IBRION = 2      # relax ions\n"
            "ISIF   = 2      # fix cell shape, move ions only (do NOT use 3!)\n"
            "NSW    = 60     # enough relaxation steps\n"
            "EDIFFG = -0.01  # force convergence (eV/Angstrom)\n"
        )
        for _ln in _relax_incar.rstrip().splitlines():
            print("    " + _ln)
        try:
            _ex = os.path.join(out_dir, "INCAR_relax_example")
            with open(_ex, "w", encoding="utf-8") as _f:
                _f.write("# MHEC: relax internal ions at fixed deformed cell (relaxed-ion elastic constants)\n")
                _f.write("# ISIF=2 keeps the imposed strain; ISIF=3 would relax it away.\n")
                _f.write(_relax_incar)
            ok(T(f"示例 INCAR 已写入 {_ex}", f"Example INCAR written to {_ex}"))
        except OSError:
            pass


# ============================================================
# 401) NPT 晶格优化设置
# ============================================================

def func_401(config: MHECConfig) -> None:
    sep("NPT 晶格优化设置")
    poscar = _read_poscar_interactive()
    temp_str = prompt_with_default(
        "温度 (K, 多个用空格分隔)",
        " ".join(str(int(t)) for t in config.temperatures),
    )
    try:
        config.temperatures = [float(t) for t in temp_str.split()]
    except ValueError:
        warn("温度格式错误，使用默认值。")

    # 检测 ML_FF
    run_only = False
    mlff_src = config.mlff_path or "ML_FF"
    if os.path.isfile(mlff_src):
        info(f"检测到预训练 ML_FF: {mlff_src}")
        use_mlff = prompt_with_default("使用预训练 ML_FF? (y/n)", "y")
        if use_mlff.lower() == "y":
            run_only = True
            config.mlff_path = mlff_src
    if not run_only:
        mlff_input = prompt_with_default("ML_FF 路径 (留空=三步模式)", config.mlff_path or "")
        if mlff_input and os.path.isfile(mlff_input):
            run_only = True
            config.mlff_path = mlff_input

    if run_only:
        info("run-only 模式: 跳过 train/refit，直接使用 ML_FF")
    else:
        info("三步模式: train → refit → run")

    optimizer = LatticeOptimizer(
        poscar=poscar,
        temperatures=config.temperatures,
        nsw_train=config.nsw_train,
        nsw_refit=config.nsw_refit,
        nsw_run=config.nsw_run,
        potim=config.potim,
        encut=config.encut,
        user_overrides=config.incar_overrides,
        skip_steps=config.skip_steps,
    )
    npt_base = prompt_with_default("NPT 工作目录", "npt_opt")

    # POTCAR
    potcar_src = config.potcar_path or "POTCAR"
    if not os.path.isfile(potcar_src):
        potcar_src = prompt_with_default("POTCAR 路径", "POTCAR")
        config.potcar_path = potcar_src

    # 提交模板
    submit_tpl = None
    if config.submit_template and os.path.isfile(config.submit_template):
        with open(config.submit_template, 'r') as f:
            submit_tpl = f.read()

    npt_dirs = optimizer.setup_npt_dirs(
        npt_base,
        potcar_path=potcar_src if os.path.isfile(potcar_src) else None,
        submit_template=submit_tpl,
        mlff_path=config.mlff_path if run_only else None,
        run_only=run_only,
        slurm_config=config.slurm,
    )
    for temp, d in npt_dirs.items():
        ok(f"{int(temp)} K: {d}")
    info("提交 NPT 优化任务:")
    print(f"    bash {os.path.join(npt_base, 'submit_npt.sh')}")
    info("完成后运行菜单 13 提取平衡结构并计算热膨胀系数。")


# ============================================================
# 402) NVT 弹性常数计算设置
# ============================================================

def func_402(config: MHECConfig) -> None:
    sep("NVT 弹性常数计算设置")
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    config.strain_method = "standard"
    _setup_work_dir(poscar, cs, config)


# ============================================================
# 403) 完整工作流向导
# ============================================================

def _build_deform_items(method, cs, lattice, config):
    """返回 ([(dirname, deformed_lattice), ...], 变形模式数). 按方法选 SC-SS / SA-SS 变形集。"""
    from .config import normalize_elastic_method
    method = normalize_elastic_method(method)   # 统一为 "sc"/"sa"
    amp_labels = get_amplitude_labels(config.n_points)
    items = []
    if method == "sa":
        modes = get_vc_modes(cs)
        strained = generate_vc_structures(lattice, cs, config.magnitude, config.n_points)
        for m in modes:
            for label, _ in amp_labels:
                items.append((f"{m['name']}_{label}", strained[m['name']][label]))
        return items, len(modes)
    codes = get_deform_codes(cs, "standard")
    strained = generate_strained_structures(lattice, codes, config.magnitude, config.n_points)
    for code in codes:
        for label, _ in amp_labels:
            items.append((f"{code}_{label}", strained[code][label]))
    return items, len(codes)


def func_403(config: MHECConfig, method: str = "sc") -> None:
    from .config import normalize_elastic_method
    method = normalize_elastic_method(method)   # 统一为 "sc"/"sa"
    method_name = (T("SA-SS 对称化应力-应变法", "SA-SS Symmetry-Adapted Stress-Strain") if method == "sa"
                   else T("SC-SS 单分量应力-应变法", "SC-SS Single-Component Stress-Strain"))
    sep(T(f"完整工作流向导 — {method_name}", f"Full workflow wizard — {method_name}"))

    # ---- 1. 读取用户准备的文件 ----
    print(T("\n  请确认当前目录下的输入文件:", "\n  Confirm input files in the current directory:"))
    files_status = {}
    for fname in ["POSCAR", "POTCAR", "KPOINTS", "INCAR", "ML_FF", "run.sh"]:
        exists = os.path.isfile(fname)
        files_status[fname] = exists
        status = "✓" if exists else "✗"
        print(f"    {status} {fname}")

    if not files_status["POSCAR"]:
        warn(T("POSCAR 不存在，无法继续。", "POSCAR not found; cannot continue."))
        return
    if not files_status["INCAR"]:
        warn(T("INCAR 不存在，无法继续。", "INCAR not found; cannot continue."))
        return

    poscar = read_poscar("POSCAR")
    ok(T("已读取 POSCAR", "POSCAR loaded"))

    # 读取用户 INCAR 作为模板
    from .vasp_io import read_incar
    from .incar_templates import adapt_user_incar
    user_incar = read_incar("INCAR")
    ok(T(f"已读取 INCAR ({len(user_incar)} 个参数)", f"INCAR loaded ({len(user_incar)} parameters)"))

    # 显示关键参数
    print(T("\n  INCAR 关键参数:", "\n  Key INCAR parameters:"))
    for key in ["PREC", "EDIFF", "ENCUT", "ISIF", "NSW", "POTIM",
                 "TEBEG", "TEEND", "ML_LMLFF", "ML_MODE", "ML_ISTART"]:
        if key in user_incar:
            print(f"    {key} = {user_incar[key]}")

    # ---- 2. 晶系识别 ----
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    if method == "sa" and cs not in VC_SUPPORTED:
        warn(T(f"SA-SS 对称化应力-应变法暂不支持 {cs_name(cs)}，请使用 SC-SS 单分量应力-应变法 (菜单 1)。",
               f"SA-SS Symmetry-Adapted method not yet supported for {cs_name(cs)}; use SC-SS (menu 1)."))
        return

    # ---- 3. 计算参数确认 ----
    config.strain_method = "standard"
    config.elastic_method = method
    if method == "sa":
        n_modes = len(get_vc_modes(cs))
        info(T(f"SA-SS 变形模式数: {n_modes}", f"SA-SS deformation modes: {n_modes}"))
    else:
        n_modes = len(get_deform_codes(cs, "standard"))
        info(T(f"SC-SS 变形模式数: {n_modes}", f"SC-SS deformation modes: {n_modes}"))

    temp_str = prompt_with_default(
        T("温度 (K, 多个用空格分隔)", "Temperatures (K, space-separated)"),
        " ".join(str(int(t)) for t in config.temperatures),
    )
    try:
        config.temperatures = [float(t) for t in temp_str.split()]
    except ValueError:
        warn(T("温度格式错误，使用默认值。", "Invalid temperature format; using default."))

    # 应变采样: 最大应变 + 点数 (点数按晶系自适应默认), 在 ±max_strain 内均匀分布
    from .config import (default_n_points_for_system, resolve_strain_sampling,
                         strain_amplitude_list)
    ms_default = config.max_strain if config.max_strain else round(
        config.magnitude * (config.n_points // 2), 4)
    ms_str = prompt_with_default(
        T("最大应变 (应变点在 ±此值内均匀分布; 建议 ≤0.02 留在 MLFF 训练域)",
          "Max strain (points evenly spaced within ±this; ≤0.02 to stay in MLFF domain)"),
        str(ms_default))
    try:
        _max_strain = float(ms_str)
    except ValueError:
        _max_strain = ms_default
    np_default = default_n_points_for_system(cs)
    np_str = prompt_with_default(
        T(f"应变点数 (3/5/7/9/11; {cs_name(cs)} 建议 {np_default})",
          f"Strain points (3/5/7/9/11; {np_default} suggested for {cs_name(cs)})"),
        str(np_default))
    try:
        _n_pts = int(np_str)
        if _n_pts not in (3, 5, 7, 9, 11):
            _n_pts = np_default
    except ValueError:
        _n_pts = np_default
    config.max_strain = _max_strain
    config.magnitude, config.n_points = resolve_strain_sampling(
        config.magnitude, _n_pts, max_strain=_max_strain)
    _pts = strain_amplitude_list(config.magnitude, config.n_points)
    _all = _pts[:len(_pts) // 2] + [0.0] + _pts[len(_pts) // 2:]
    info(T(f"应变点: {'  '.join(f'{p:+.4f}' for p in _all)}   (δ={config.magnitude:.4f}, {config.n_points} 点)",
           f"Strain points: {'  '.join(f'{p:+.4f}' for p in _all)}  (delta={config.magnitude:.4f}, {config.n_points} pts)"))

    # NSW 确认
    nsw_npt_default = user_incar.get("NSW", "10000")
    nsw_npt = prompt_with_default(T("NPT NSW (晶格优化步数)", "NPT NSW (lattice optimization steps)"), nsw_npt_default)
    nsw_nvt = prompt_with_default(T("NVT NSW (弹性常数采样步数)", "NVT NSW (elastic sampling steps)"), nsw_npt_default)
    try:
        nsw_npt_int = int(nsw_npt)
        nsw_nvt_int = int(nsw_nvt)
    except ValueError:
        nsw_npt_int = int(nsw_npt_default)
        nsw_nvt_int = int(nsw_npt_default)

    config.nsw_run = nsw_nvt_int

    # ---- 4. 确认摘要 ----
    n_temps = len(config.temperatures)
    n_deform = n_modes
    n_amp = len(get_amplitude_labels(config.n_points))
    n_npt = n_temps
    n_nvt = n_temps * (n_deform * n_amp + 1)
    n_total = n_npt + n_nvt

    print(f"\n  ═══════════════════════════════════════════════════")
    print(T("  工作流摘要:", "  Workflow summary:"))
    print(T(f"    温度: {', '.join(str(int(t)) for t in config.temperatures)} K ({n_temps} 个)",
            f"    Temperatures: {', '.join(str(int(t)) for t in config.temperatures)} K ({n_temps})"))
    print(T(f"    Phase 1 — NPT 晶格优化: {n_npt} 个计算 (NSW={nsw_npt_int})",
            f"    Phase 1 — NPT lattice optimization: {n_npt} runs (NSW={nsw_npt_int})"))
    print(T(f"    Phase 2 — NVT 弹性常数: {n_nvt} 个计算 (NSW={nsw_nvt_int})",
            f"    Phase 2 — NVT elastic sampling: {n_nvt} runs (NSW={nsw_nvt_int})"))
    print(T(f"    总计算数: {n_total}", f"    Total calculations: {n_total}"))
    print(T("    输入文件: ", "    Inputs: ") +
          f"POSCAR{'✓' if files_status['POSCAR'] else '✗'} "
          f"POTCAR{'✓' if files_status['POTCAR'] else '✗'} "
          f"KPOINTS{'✓' if files_status['KPOINTS'] else '✗'} "
          f"ML_FF{'✓' if files_status['ML_FF'] else '✗'} "
          f"run.sh{'✓' if files_status['run.sh'] else '✗'}")
    print(f"  ═══════════════════════════════════════════════════")

    confirm = prompt_with_default(T("确认并生成? (y/n)", "Confirm and generate? (y/n)"), "y")
    if confirm.lower() != "y":
        info(T("已取消。", "Cancelled."))
        return

    # ---- 5. 生成所有目录 ----
    work_dir = prompt_with_default(T("工作目录", "Work directory"), "mhec_work")
    os.makedirs(work_dir, exist_ok=True)

    # 弹性常数计算: 有 ML_FF 时强制 run 模式 (同一力场跑所有变形, 保证一致)
    mlff_run = bool(files_status["ML_FF"])
    if mlff_run:
        info(T("检测到 ML_FF → NPT/NVT 强制 ML_MODE=run (ML_ISTART=2), 剔除训练参数",
               "ML_FF detected → forcing ML_MODE=run (ML_ISTART=2) for NPT/NVT, training params removed"))
    else:
        warn(T("未检测到 ML_FF! 弹性常数计算必须用同一个已训练力场 (ML_MODE=run)。",
               "No ML_FF detected! Elastic-constant calculation must use one trained force field (ML_MODE=run)."))
        warn(T("否则各变形目录会各自在线训练出不一致的力场 → 弹性常数无意义。",
               "Otherwise each deformation directory would train an inconsistent force field → meaningless elastic constants."))
        warn(T("请先训练力场 (菜单 7/8) 得到 ML_FF 再回到本流程, 或确认你的 INCAR 已正确设为 run 模式。",
               "Train a force field first (menu 7/8) to obtain ML_FF, or ensure your INCAR is set to run mode."))

    import shutil

    # 收集所有需要复制文件的目录
    all_dirs = []

    # Phase 1: NPT
    npt_base = os.path.join(work_dir, "phase1_npt")
    os.makedirs(npt_base, exist_ok=True)
    npt_dirs = {}
    for temp in config.temperatures:
        d = os.path.join(npt_base, f"npt_{int(temp)}K")
        os.makedirs(d, exist_ok=True)
        npt_dirs[temp] = d
        all_dirs.append(d)

        # INCAR: 基于用户模板，改 ISIF=3, 温度, NSW (有 ML_FF 则强制 run 模式)
        npt_incar = adapt_user_incar(user_incar, "npt", temp, nsw=nsw_npt_int,
                                     mlff_run=mlff_run,
                                     n_species=len(poscar.get("counts") or [1]))
        write_incar(os.path.join(d, "INCAR"), npt_incar)
        write_poscar(os.path.join(d, "POSCAR"), poscar)

    # Phase 2: NVT
    nvt_base = os.path.join(work_dir, "phase2_nvt")
    os.makedirs(nvt_base, exist_ok=True)
    deform_items, n_modes = _build_deform_items(method, cs, poscar["lattice"], config)

    nvt_dirs_by_temp = {}
    for temp in config.temperatures:
        temp_dirs = []
        # 平衡构型
        eq_dir = os.path.join(nvt_base, f"{int(temp)}K", "equilibrium")
        os.makedirs(eq_dir, exist_ok=True)
        nvt_incar = adapt_user_incar(user_incar, "nvt", temp, nsw=nsw_nvt_int,
                                     mlff_run=mlff_run,
                                     n_species=len(poscar.get("counts") or [1]))
        write_incar(os.path.join(eq_dir, "INCAR"), nvt_incar)
        write_poscar(os.path.join(eq_dir, "POSCAR"), poscar)
        all_dirs.append(eq_dir)
        temp_dirs.append(eq_dir)

        # 变形构型
        for dirname, dlat in deform_items:
            d = os.path.join(nvt_base, f"{int(temp)}K", dirname)
            os.makedirs(d, exist_ok=True)
            deformed_poscar = dict(poscar)
            deformed_poscar["lattice"] = dlat
            write_incar(os.path.join(d, "INCAR"), nvt_incar)
            write_poscar(os.path.join(d, "POSCAR"), deformed_poscar)
            all_dirs.append(d)
            temp_dirs.append(d)

        nvt_dirs_by_temp[temp] = temp_dirs

    # ---- 6. 复制公共文件到所有目录 ----
    for fname in ["POTCAR", "KPOINTS", "ML_FF"]:
        if os.path.isfile(fname):
            n = 0
            for d in all_dirs:
                dst = os.path.join(d, fname)
                if not os.path.isfile(dst):
                    shutil.copy2(fname, dst)
                    n += 1
            ok(T(f"{fname} 已复制到 {n} 个目录", f"{fname} copied to {n} directories"))

    # 复制 run.sh
    run_sh = "run.sh"
    if os.path.isfile(run_sh):
        n = 0
        for d in all_dirs:
            dst = os.path.join(d, "run.sh")
            if not os.path.isfile(dst):
                shutil.copy2(run_sh, dst)
                n += 1
        ok(T(f"run.sh 已复制到 {n} 个目录", f"run.sh copied to {n} directories"))

    # ---- 7. 生成提交脚本 ----
    _write_master_submit(work_dir, npt_dirs, nvt_dirs_by_temp, None, None)

    # 保存配置
    config.to_file(os.path.join(work_dir, "mhec.yaml"))

    print(T(f"\n  模式: {'run-only (ML_FF)' if files_status['ML_FF'] else 'train→refit→run'}",
            f"\n  Mode: {'run-only (ML_FF)' if files_status['ML_FF'] else 'train→refit→run'}"))
    print(T(f"  Phase 1 (NPT): {n_npt} 个计算", f"  Phase 1 (NPT): {n_npt} runs"))
    print(T(f"  Phase 2 (NVT): {n_nvt} 个计算", f"  Phase 2 (NVT): {n_nvt} runs"))
    print(T(f"  总计: {n_total} 个计算目录", f"  Total: {n_total} calculation directories"))
    ok(T(f"工作目录: {os.path.abspath(work_dir)}", f"Work directory: {os.path.abspath(work_dir)}"))
    info(T("一键提交所有任务:", "Submit all jobs with one command:"))
    print(f"    bash {os.path.join(work_dir, 'submit_all.sh')}")


def _write_master_submit(work_dir, npt_dirs, nvt_dirs_by_temp, codes, amp_labels):
    """生成主提交脚本：Phase 1 (NPT) → Phase 2 (NVT)。
    
    支持 sbatch submit_all.sh 和 bash submit_all.sh 两种用法。
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --partition=8358P",
        "#SBATCH --job-name=mhec_submit",
        "#SBATCH --output=submit_all_%j.out",
        "#SBATCH --error=submit_all_%j.err",
        "",
        "# MHEC 一键提交脚本",
        "# Phase 1: NPT 晶格优化 (所有温度并行)",
        "# Phase 2: NVT 弹性常数 (等待 NPT 完成后提交)",
        "# 用法: sbatch submit_all.sh  或  bash submit_all.sh",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        'fi',
        "",
        "# ===== Phase 1: NPT 晶格优化 =====",
        "echo '>>> Phase 1: NPT 晶格优化'",
        "NPT_JOBS=()",
        "",
    ]

    for temp, d in sorted(npt_dirs.items()):
        rel = os.path.relpath(d, work_dir)
        lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
        lines.append('JOB_ID=$(sbatch --parsable run.sh)')
        lines.append('NPT_JOBS+=($JOB_ID)')
        lines.append(f'echo "  {int(temp)}K NPT: $JOB_ID"')
        lines.append('popd > /dev/null')

    lines.extend([
        "",
        "# 构建 NPT 依赖字符串",
        'NPT_DEPS=$(IFS=:; echo "${NPT_JOBS[*]}")',
        "",
        "# ===== Phase 1.5: 用 NPT CONTCAR 更新 NVT POSCAR =====",
        "# NPT 完成后，读取各温度 CONTCAR，重新生成 NVT 变形 POSCAR",
        'echo ">>> 等待 NPT 完成后更新 NVT 结构..."',
        'UPDATE_JOB=$(sbatch --parsable --dependency=afterok:$NPT_DEPS '
        '--wrap="python3 -m mhec.nvt_generator $WORK_ROOT" '
        '-N1 -n1 --job-name=mhec_update '
        '--output=$WORK_ROOT/update_%j.out --error=$WORK_ROOT/update_%j.err)',
        'echo "  POSCAR 更新任务: $UPDATE_JOB"',
        "",
        "# ===== Phase 2: NVT 弹性常数 (等待更新完成) =====",
        "echo '>>> Phase 2: NVT 弹性常数'",
        "NVT_JOBS=()",
        "",
    ])

    for temp, dirs in sorted(nvt_dirs_by_temp.items()):
        for d in dirs:
            rel = os.path.relpath(d, work_dir)
            lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
            lines.append('JOB_ID=$(sbatch --parsable --dependency=afterok:$UPDATE_JOB run.sh)')
            lines.append('NVT_JOBS+=($JOB_ID)')
            lines.append(f'echo "  {rel}: $JOB_ID"')
            lines.append('popd > /dev/null')

    lines.extend([
        "",
        'echo ">>> 所有任务已提交。"',
        f'echo ">>> NPT: ${{#NPT_JOBS[@]}} 个任务"',
        f'echo ">>> NVT: ${{#NVT_JOBS[@]}} 个任务"',
    ])

    path = os.path.join(work_dir, "submit_all.sh")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  +-> {path}")


# ============================================================
# _setup_work_dir (内部函数)
# ============================================================

def _setup_work_dir(poscar: Dict, cs: CrystalSystem, config: MHECConfig) -> str:
    """创建工作目录并生成所有计算输入文件。

    支持两种模式:
    - 三步模式 (train→refit→run): config.mlff_path 为空
    - run-only 模式: config.mlff_path 指向预训练的 ML_FF
    """
    import shutil

    run_only = bool(config.mlff_path and os.path.isfile(config.mlff_path))
    stages = ["run"] if run_only else ["train", "refit", "run"]
    if run_only:
        info(f"使用预训练 ML_FF: {config.mlff_path}")
        info(f"仅生成 run 目录 (跳过 train/refit)")

    deform_codes = get_deform_codes(cs, config.strain_method)
    amp_labels = get_amplitude_labels(config.n_points)
    amp_label_strs = [label for label, _ in amp_labels]

    work_dir = prompt_with_default("工作目录", "mhec_work")
    os.makedirs(work_dir, exist_ok=True)

    info("生成计算目录...")
    dm = DirManager(work_dir)
    all_dirs = dm.create_deform_dirs(deform_codes, amp_label_strs)
    eq_dirs = dm.create_equilibrium_dir()
    lattice = poscar["lattice"]
    strained = generate_strained_structures(
        lattice, deform_codes, config.magnitude, config.n_points
    )

    all_calc_dirs = []

    for temp in config.temperatures:
        info(f"温度: {temp} K")
        for stage in stages:
            stage_dir = eq_dirs[stage]
            write_poscar(os.path.join(stage_dir, "POSCAR"), poscar)
            params = build_incar_params(
                ensemble="nvt", mlff_stage=stage, temperature=temp,
                nsw=getattr(config, f"nsw_{stage}"),
                potim=config.potim, encut=config.encut,
                user_overrides=config.incar_overrides,
            )
            write_incar(os.path.join(stage_dir, "INCAR"), params)
            write_kpoints_gamma(os.path.join(stage_dir, "KPOINTS"))
            all_calc_dirs.append(stage_dir)

        for code in deform_codes:
            for label, _ in amp_labels:
                deformed_lattice = strained[code][label]
                deformed_poscar = dict(poscar)
                deformed_poscar["lattice"] = deformed_lattice
                for stage in stages:
                    stage_dir = all_dirs[code][label][stage]
                    write_poscar(os.path.join(stage_dir, "POSCAR"), deformed_poscar)
                    params = build_incar_params(
                        ensemble="nvt", mlff_stage=stage, temperature=temp,
                        nsw=getattr(config, f"nsw_{stage}"),
                        potim=config.potim, encut=config.encut,
                        user_overrides=config.incar_overrides,
                    )
                    write_incar(os.path.join(stage_dir, "INCAR"), params)
                    write_kpoints_gamma(os.path.join(stage_dir, "KPOINTS"))
                    all_calc_dirs.append(stage_dir)

    write_slurm_scripts(work_dir, all_dirs, eq_dirs, config.slurm)
    config.to_file(os.path.join(work_dir, "mhec.yaml"))

    # 自动复制 POTCAR
    potcar_src = config.potcar_path or "POTCAR"
    if os.path.isfile(potcar_src):
        from .slurm import copy_potcar_to_dirs
        n_copied = copy_potcar_to_dirs(potcar_src, all_calc_dirs)
        ok(f"POTCAR 已复制到 {n_copied} 个目录")
    else:
        warn(f"POTCAR 未找到 ({potcar_src})，请手动复制")

    # 自动复制 ML_FF (run-only 模式)
    if run_only:
        n_mlff = 0
        for d in all_calc_dirs:
            dst = os.path.join(d, "ML_FF")
            if not os.path.isfile(dst):
                shutil.copy2(config.mlff_path, dst)
                n_mlff += 1
        ok(f"ML_FF 已复制到 {n_mlff} 个目录")

    # 使用用户提交模板（如果有）
    submit_template = None
    tpl_path = config.submit_template
    if tpl_path and os.path.isfile(tpl_path):
        from .slurm import read_submit_template, write_slurm_with_template
        submit_template = read_submit_template(tpl_path)
        write_slurm_with_template(work_dir, all_dirs, eq_dirs, config.slurm, submit_template)
        ok(f"使用提交模板: {tpl_path}")

    n_deform = len(deform_codes)
    n_amp = len(amp_labels)
    n_stages = len(stages)
    n_total = (n_deform * n_amp + 1) * n_stages
    mode_str = "run-only (ML_FF)" if run_only else "train→refit→run"
    print(f"\n  模式: {mode_str}")
    print(f"  变形模式: {n_deform}    幅度点: {n_amp}    总计算目录: {n_total}")
    ok(f"工作目录: {os.path.abspath(work_dir)}")
    info("提交所有任务:")
    print(f"    bash {os.path.join(work_dir, 'submit_all.sh')}")
    return work_dir


# ============================================================
# 404) 生成 SLURM 提交脚本
# ============================================================

def func_404(config: MHECConfig) -> None:
    sep("生成 SLURM 提交脚本")
    work_dir = prompt_with_default("工作目录", "mhec_work")
    if not os.path.isdir(work_dir):
        warn(f"目录不存在: {work_dir}")
        return
    info("重新生成 SLURM 脚本...")
    # 重建目录结构信息
    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    deform_codes = get_deform_codes(cs, config.strain_method)
    amp_label_strs = [label for label, _ in get_amplitude_labels(config.n_points)]
    dm = DirManager(work_dir)
    all_dirs = dm.create_deform_dirs(deform_codes, amp_label_strs)
    eq_dirs = dm.create_equilibrium_dir()
    write_slurm_scripts(work_dir, all_dirs, eq_dirs, config.slurm)
    ok(f"SLURM 脚本已生成至 {work_dir}/")


# ============================================================
# 501) 提取时间平均应力
# ============================================================

def func_501(config: MHECConfig) -> None:
    sep("提取时间平均应力")
    work_dir = prompt_with_default("计算目录路径", ".")
    skip = int(prompt_with_default("跳过初始步数", str(config.skip_steps)))
    extractor = StressExtractor()
    try:
        mean, stderr, meta = extractor.extract_and_average(work_dir, skip)
        print(f"\n  数据来源: {meta['source']}")
        print(f"  总步数: {meta['n_total_steps']}    使用步数: {meta['n_used_steps']}")
        labels = ["σ_xx", "σ_yy", "σ_zz", "σ_yz", "σ_xz", "σ_xy"]
        print(f"\n  {'分量':>6}  {'平均 (kBar)':>12}  {'标准误 (kBar)':>14}")
        print(f"  {'─' * 38}")
        for i, lbl in enumerate(labels):
            print(f"  {lbl:>6}  {mean[i]:12.4f}  {stderr[i]:14.4f}")
    except Exception as e:
        warn(f"提取失败: {e}")


# ============================================================
# 拟合诊断图
# ============================================================

_VOIGT_LABELS = ["s1(xx)", "s2(yy)", "s3(zz)", "s4(yz)", "s5(xz)", "s6(xy)"]

# 诊断图/热图/数据导出只保留 R² ≥ 此阈值的「强通道」; 弱通道(低或负 R² 的
# 横向读数)不再画出/导出, 也在热图里标灰。注意: 内部联立加权求解仍使用全部
# 通道(按 R² 加权), 这里只影响输出显示, 不改变最终弹性常数矩阵。
_R2_STRONG = 0.9


def _curve_from_fit(res):
    """由 robust_slope_fit 的结果构造用于绘图的曲线函数。"""
    if res.get("kind") == "cubic" and res.get("coef") is not None:
        c = res["coef"]
        return lambda xx, c=c: c[0] * xx + c[1] * xx ** 3
    s = res.get("slope", 0.0)
    return lambda xx, s=s: s * xx


def _narrow_note(res):
    """缩窗拟合时给出简短标注 (用于图标题/报告)。"""
    if res.get("kind") == "linear_narrowed":
        return " [narrowed, %d pts]" % int(res.get("n_used", 0))
    return ""


def _strong_components(meaningful, r2_vals):
    """从对称性允许分量中挑出强通道(R² ≥ _R2_STRONG);
    若一个都没有, 退回该模式中 R² 最高的单个分量, 避免出现空图。"""
    strong = [i for i in meaningful if float(r2_vals[i]) >= _R2_STRONG]
    if not strong and meaningful:
        strong = [max(meaningful, key=lambda i: float(r2_vals[i]))]
    return strong


def _write_cij_solution_report(strong_rows, cij, Nmat, out_dir, fname, title, cij_std=None):
    """把「每个 Cij 由哪些强通道方程解出」写成文字, 打印并存文件。

    strong_rows : [(mode_name, comp_i, combo_str, slope_GPa, r2), ...] 仅强通道
    cij         : 最终(对称化)6×6 矩阵, 用于给出各常数的解值 (可为 None)
    Nmat        : 弹性常数符号名矩阵 (来自 _cij_name_matrix)
    cij_std     : 6×6 拟合标准误矩阵 (可为 None); 有则显示 值 ± 标准误
    """
    import re as _re

    # 兼容 5 元组 (无 kind) 与 6 元组 (含拟合方式)。
    def _kind_of(row):
        return row[5] if len(row) > 5 else "linear"

    def _fit_tag(row):
        return "  [narrowed to small strain]" if _kind_of(row) == "linear_narrowed" else ""

    # 把通道分成「真正的强通道」(R² ≥ 阈值) 与「回退通道」(R² < 阈值, 仅因该常数
    # 没有任何强通道可用而被保留)。回退通道单独列出并明确标注约束很弱, 避免误导。
    truly_strong = [row for row in strong_rows if float(row[4]) >= _R2_STRONG]
    fallback = [row for row in strong_rows if float(row[4]) < _R2_STRONG]

    lines = [title, "", "Strong stress-slope equations used (R2 >= %.2f):" % _R2_STRONG]
    if truly_strong:
        for row in truly_strong:
            mode, i, combo, slope, r2 = row[0], row[1], row[2], row[3], row[4]
            lines.append("  %-10s %-7s :  %-16s = %9.2f GPa   (R2=%.3f)%s"
                         % (mode, _VOIGT_LABELS[i], combo, slope, r2, _fit_tag(row)))
    else:
        lines.append("  (none reached the R2 threshold)")
    if fallback:
        lines.append("")
        lines.append("Fallback channels below threshold (R2 < %.2f, poorly constrained, "
                     "retained only because no strong channel is available):" % _R2_STRONG)
        for row in fallback:
            mode, i, combo, slope, r2 = row[0], row[1], row[2], row[3], row[4]
            lines.append("  %-10s %-7s :  %-16s = %9.2f GPa   (R2=%.3f)  [WEAK]%s"
                         % (mode, _VOIGT_LABELS[i], combo, slope, r2, _fit_tag(row)))
    # 缩窗通道单独提示 (即便 R² 已达标, 也说明用的是小应变窗口而非全部数据点)
    narrowed = [row for row in strong_rows if _kind_of(row) == "linear_narrowed"]
    if narrowed:
        lines.append("")
        lines.append("Auto-narrowed channels (odd-polynomial/linear over the full strain range "
                     "was poor; refitted linearly on the small-strain window only):")
        for row in narrowed:
            mode, i, combo, slope, r2 = row[0], row[1], row[2], row[3], row[4]
            lines.append("  %-10s %-7s :  %-16s = %9.2f GPa   (R2=%.3f, narrowed)"
                         % (mode, _VOIGT_LABELS[i], combo, slope, r2))
    lines.append("")
    lines.append("Each independent Cij (value +/- fit standard error from the R2-weighted "
                 "joint solve) and the channels that constrain it:")
    seen, order = {}, []
    if Nmat is not None:
        for i in range(6):
            for j in range(i, 6):
                nmj = Nmat[i][j]
                if nmj == "0" or nmj in seen:
                    continue
                seen[nmj] = (i, j)
                order.append(nmj)
    warnings = []   # 收集约束很弱/误差棒过大的常数, 末尾统一提醒
    for nmj in order:
        ii, jj = seen[nmj]
        val = (cij[ii][jj] if cij is not None else float("nan"))
        std = (cij_std[ii][jj] if cij_std is not None else None)
        # 该常数的来源通道 (区分强/弱/缩窗)
        src_rows = [(row[0], row[1], row[2], row[4], _kind_of(row)) for row in strong_rows
                    if nmj in set(_re.findall(r"C\d\d", row[2]))]
        srcs = ["%s %s(%s)%s%s" % (mode, _VOIGT_LABELS[i2], combo,
                                   "" if float(r2) >= _R2_STRONG else " [WEAK]",
                                   " [narrowed]" if kind == "linear_narrowed" else "")
                for (mode, i2, combo, r2, kind) in src_rows]
        src_str = "; ".join(srcs) if srcs else "(fixed by symmetry / derived relation)"
        # 判定是否约束不足: (a) 所有来源通道都低于阈值, 或 (b) 误差棒过大
        only_weak = bool(src_rows) and all(float(r2) < _R2_STRONG for (_, _, _, r2, _k) in src_rows)
        big_err = (std is not None and np.isfinite(std)
                   and std > 2.0 and (abs(val) < 1e-6 or std > 0.25 * abs(val)))
        flag = " (!)" if (only_weak or big_err) else ""
        if only_weak or big_err:
            reason = ("no strong channel" if only_weak else "large fit uncertainty")
            warnings.append((nmj, reason))
        if std is not None:
            lines.append("  %-6s = %9.2f +/- %6.2f GPa%s   <-  %s"
                         % (nmj, val, std, flag, src_str))
        else:
            lines.append("  %-6s = %9.2f GPa%s   <-  %s" % (nmj, val, flag, src_str))
    if warnings:
        lines.append("")
        lines.append("WARNING: the following constants are poorly constrained and should NOT "
                     "be trusted as-is (marked (!) above):")
        for (nmj, reason) in warnings:
            lines.append("  %-6s : %s" % (nmj, reason))
        lines.append("  Suggestions: raise the validation temperature so soft modes are thermally "
                     "activated, add more amplitude points and longer NVT sampling for the weak "
                     "channel, and check c/a and the force-field shear accuracy.")
    text = "\n".join(lines)
    try:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(text + "\n")
    except OSError:
        pass
    return text


def _cij_name_matrix(cs):
    """构造 6×6 弹性常数符号名矩阵 (按晶系对称性: 等价项同名, 零项为 '0')。"""
    from .symmetry import get_symmetry_rules

    def nm(i, j):
        a, b = sorted((i + 1, j + 1))
        return f"C{a}{b}"

    N = [[nm(i, j) for j in range(6)] for i in range(6)]
    rules = get_symmetry_rules(cs)
    for group in rules.equivalent_groups:
        i0, j0 = group[0]
        rep = nm(i0, j0)
        for (i, j) in group:
            N[i][j] = rep
            N[j][i] = rep
    for (i, j) in rules.zero_entries:
        N[i][j] = "0"
        N[j][i] = "0"
    return N


def _cij_combo_label(N, d, i):
    """给定应变方向 d, 返回应力分量 σ_i 斜率对应的弹性常数组合字符串,
    即 (C·d)_i 的符号表达式, 如 'C11-C12' / '2·C44' / 'C11+2·C12' / '0'。"""
    coef, order = {}, []
    for j in range(6):
        if abs(d[j]) < 1e-3:          # 丢弃体积守恒的二阶补偿等微小分量
            continue
        nmj = N[i][j]
        if nmj == "0":
            continue
        if nmj not in coef:
            order.append(nmj)
        coef[nmj] = coef.get(nmj, 0.0) + d[j]
    terms = []
    for nmj in order:
        c = round(coef[nmj], 3)
        if abs(c) < 1e-6:
            continue
        if abs(c - 1) < 1e-6:
            terms.append(f"+{nmj}")
        elif abs(c + 1) < 1e-6:
            terms.append(f"-{nmj}")
        else:
            cc = int(round(c)) if abs(c - round(c)) < 1e-6 else c
            terms.append(f"{'+' if cc >= 0 else '-'}{abs(cc):g}·{nmj}")
    if not terms:
        return "0"
    s = "".join(terms)
    return s[1:] if s.startswith("+") else s


def _meaningful_components(N, d):
    """返回该应变方向下对应非零弹性常数的应力分量索引 (适配所有晶系)。"""
    if N is None:
        return list(range(6))
    out = [i for i in range(6) if _cij_combo_label(N, d, i) != "0"]
    return out if out else list(range(6))


def _grid_for(n):
    """根据子图数选择合适的网格 (行, 列)。"""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3
    if n == 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    return 3, 3


def _plot_fitting_diagnostics(
    deform_codes, all_strains, all_stresses, baseline,
    results, magnitude, out_dir=".", fit_order=3, cs=None,
):
    """为每个 deform code 生成应力-应变拟合诊断图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warn(T("matplotlib 不可用，跳过拟合诊断图。", "matplotlib unavailable; skipping fitting diagnostic plots."))
        return

    def _fit_curve(x, y, order):
        """返回 (线性斜率 C1, 曲线函数, 拟合方式)。与 fit_single_mode 用同一稳健判据:
        奇多项式 / 过原点线性 / 小应变缩窗自动选择, 保证图与最终矩阵一致。"""
        res = robust_slope_fit(x, y, allow_cubic=(order >= 3))
        return res["slope"], _curve_from_fit(res), res

    os.makedirs(out_dir, exist_ok=True)
    fit_info = results["fit_info_per_mode"]
    cij_raw = results["cij_raw"]
    Nmat = _cij_name_matrix(cs) if cs is not None else None
    strong_rows = []          # (mode, comp_i, combo, slope, r2) 仅强通道, 供求解来源报告

    for code in deform_codes:
        if code not in all_strains:
            continue
        strains = all_strains[code]
        stresses = all_stresses[code]
        delta_stress = stresses - baseline[np.newaxis, :]

        # 找出该 deform code 对应的应变分量与方向 (用于标注每条曲线对应的弹性常数)
        from .strain import decode_deform as _decode
        base_strain = _decode(code, magnitude)
        nonzero = np.where(np.abs(base_strain) > 1e-15)[0]
        strain_label = ", ".join(f"e{j+1}" for j in nonzero)
        direction = base_strain / magnitude if magnitude else base_strain

        r2 = fit_info[code]["r_squared"]

        # 只画「强通道」: 对称性允许 且 R² ≥ 阈值 (弱/噪声横向读数不显示)
        meaningful = _meaningful_components(Nmat, direction)
        strong = _strong_components(meaningful, r2)
        nr, ncols = _grid_for(len(strong))
        fig, axes = plt.subplots(nr, ncols, figsize=(ncols * 5.0, nr * 4.2),
                                 squeeze=False)
        axes_list = axes.flatten()
        fig.suptitle(f"{code}  (strain: {strain_label})", fontsize=14, fontweight="bold")

        for ax_idx, i in enumerate(strong):
            ax = axes_list[ax_idx]
            y = delta_stress[:, i] * (-0.1)  # kBar → GPa, 取负号与 Cij 一致
            x = strains

            ax.scatter(x, y, s=60, zorder=5, color="#0173B2", edgecolors="black", linewidth=0.5)

            slope = 0.0
            res_i = None
            if np.dot(x, x) > 0:
                slope, curve, res_i = _fit_curve(x, y, fit_order)
                x_fit = np.linspace(min(x) * 1.2, max(x) * 1.2, 100)
                ax.plot(x_fit, curve(x_fit), "r-", linewidth=1.5,
                        label=f"slope={slope:.1f} GPa")

            combo = _cij_combo_label(Nmat, direction, i) if Nmat is not None else "?"
            kind_i = res_i["kind"] if res_i else "linear"
            strong_rows.append((code, i, combo, slope, float(r2[i]), kind_i))
            note = _narrow_note(res_i) if res_i else ""
            ax.set_title(f"{_VOIGT_LABELS[i]}  →  {combo} = {slope:.1f} GPa{note}\nR2={r2[i]:.4f}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("strain", fontsize=9)
            ax.set_ylabel("d_stress (GPa)", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        # 隐藏多余的空子图
        for k in range(len(strong), len(axes_list)):
            axes_list[k].axis("off")

        plt.tight_layout()
        png_path = os.path.join(out_dir, f"fit_{code}.png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        ok(f"Fitting plot: {png_path}")

    # 汇总图：R² 热力图 (对称性零项标灰, 不参与好坏判断)
    n_codes = len(deform_codes)
    r2_matrix = np.full((n_codes, 6), np.nan)
    valid_codes = []
    from .strain import decode_deform as _decode2
    for k, code in enumerate(deform_codes):
        if code in fit_info:
            row = np.array(fit_info[code]["r_squared"], dtype=float)
            if cs is not None:
                Nm = _cij_name_matrix(cs)
                d = _decode2(code, magnitude) / magnitude if magnitude else _decode2(code, magnitude)
                for j in range(6):
                    if _cij_combo_label(Nm, d, j) == "0":
                        row[j] = np.nan          # 对称性零项 → 不显示 R²
            # 弱通道 (R² < 阈值) 也标灰排除, 使热图只反映强通道
            with np.errstate(invalid="ignore"):
                row = np.where(row < _R2_STRONG, np.nan, row)
            r2_matrix[k, :] = row
            valid_codes.append(code)

    if valid_codes:
        import matplotlib.cm as _cm
        cmap = _cm.get_cmap("RdYlGn").copy()
        cmap.set_bad("#D9D9D9")                  # 零项/弱通道显示为灰色
        fig, ax = plt.subplots(figsize=(8, max(3, n_codes * 0.6 + 1)))
        im = ax.imshow(r2_matrix[:len(valid_codes)], aspect="auto", cmap=cmap,
                       vmin=-1, vmax=1)
        ax.set_yticks(range(len(valid_codes)))
        ax.set_yticklabels(valid_codes, fontsize=9)
        ax.set_xticks(range(6))
        ax.set_xticklabels(_VOIGT_LABELS, fontsize=9)
        ax.set_title(f"R2 Heatmap (only strong channels R2 >= {_R2_STRONG}; "
                     f"gray = symmetry-zero or weak, excluded)", fontsize=11)
        for k in range(len(valid_codes)):
            for j in range(6):
                val = r2_matrix[k, j]
                if np.isnan(val):
                    ax.text(j, k, "·", ha="center", va="center", fontsize=9, color="#777777")
                else:
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, k, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)
        plt.colorbar(im, ax=ax, label="R2")
        plt.tight_layout()
        heatmap_path = os.path.join(out_dir, "fit_r2_heatmap.png")
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        ok(f"R2 heatmap: {heatmap_path}")

    # 输出 Origin 兼容数据文件
    for code in deform_codes:
        if code not in all_strains:
            continue
        strains = all_strains[code]
        stresses = all_stresses[code]
        delta_stress = stresses - baseline[np.newaxis, :]

        # 原始数据 (kBar)
        dat_path = os.path.join(out_dir, f"fit_{code}_data.dat")
        header_lines = [
            f"# MHEC Stress-Strain Fitting Data: {code}",
            f"# Units: strain (dimensionless), stress change (kBar)",
            f"# Baseline stress subtracted",
            "strain\tdelta_s1_xx_kBar\tdelta_s2_yy_kBar\tdelta_s3_zz_kBar\tdelta_s4_yz_kBar\tdelta_s5_xz_kBar\tdelta_s6_xy_kBar",
        ]
        data = np.column_stack([strains, delta_stress])
        with open(dat_path, 'w') as f:
            for h in header_lines:
                f.write(h + "\n")
            for row in data:
                f.write("\t".join(f"{v:.6e}" for v in row) + "\n")

        # GPa 版本: 只导出有意义(非零)分量, 每列列名标注对应的弹性常数 (Origin 可直接读)
        from .strain import decode_deform as _decode3
        direction = (_decode3(code, magnitude) / magnitude) if magnitude else _decode3(code, magnitude)
        meaningful = _meaningful_components(Nmat, direction)
        strong_c = _strong_components(meaningful, fit_info[code]["r_squared"])
        gpa_path = os.path.join(out_dir, f"fit_{code}_GPa.dat")
        delta_gpa = delta_stress * (-0.1)  # kBar → GPa, 取负号与 Cij 一致
        cols = [np.asarray(strains, dtype=float)]
        names = ["strain"]
        for i in strong_c:
            s, curve, _res = _fit_curve(strains, delta_gpa[:, i], fit_order)
            combo = (_cij_combo_label(Nmat, direction, i) if Nmat is not None else f"s{i+1}")
            tag = combo.replace("·", "*")
            cols.append(delta_gpa[:, i]); names.append(f"{_VOIGT_LABELS[i]}[{tag}]_meas")
            cols.append(curve(strains)); names.append(f"{tag}_fit")
        data_gpa = np.column_stack(cols)
        with open(gpa_path, 'w') as f:
            f.write(f"# MHEC Stress-Strain Fitting Data: {code}\n")
            f.write("# Units: strain (dimensionless), Cij-convention stress (GPa)\n")
            f.write("# Only symmetry-allowed components; each column maps to its elastic constant\n")
            f.write("\t".join(names) + "\n")
            for row in data_gpa:
                f.write("\t".join(f"{v:.6e}" for v in row) + "\n")

    if strong_rows:
        rpt = _write_cij_solution_report(
            strong_rows, results.get("cij_symmetrized"), Nmat, out_dir, "fit_solution.txt",
            "=== SC-SS: how each Cij is solved (strong channels, R2 >= %.2f) ===" % _R2_STRONG,
            cij_std=results.get("cij_std"))
        ok(f"Solution provenance: {os.path.join(out_dir, 'fit_solution.txt')}")
        print(rpt)
    ok(f"Fitting data (Origin): fit_*_data.dat, fit_*_GPa.dat")


def _plot_vc_fitting_diagnostics(modes, all_strains, all_stresses, baseline,
                                 magnitude, out_dir=".", fit_order=3, cs=None, cij=None,
                                 cij_std=None):
    """为每个 VC (体积守恒) 变形模式生成应力-应变拟合诊断图 + Origin 数据。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warn(T("matplotlib 不可用，跳过 VC 拟合诊断图。", "matplotlib unavailable; skipping SA-SS fitting diagnostic plots."))
        return

    os.makedirs(out_dir, exist_ok=True)

    def _fit(x, y):
        """返回 (线性斜率 C1, R2, 曲线函数, 拟合结果 dict)。与 fit_single_mode 用同一
        稳健判据 (奇多项式 / 过原点线性 / 小应变缩窗), 保证图与最终矩阵一致。"""
        if np.dot(x, x) <= 0:
            return 0.0, 1.0, (lambda xx: np.zeros_like(xx)), {"kind": "linear", "n_used": 0}
        res = robust_slope_fit(x, y, allow_cubic=(fit_order >= 3))
        return res["slope"], res["r2"], _curve_from_fit(res), res

    r2_by_mode = {}
    strong_rows = []          # (mode, comp_i, combo, slope, r2) 仅强通道, 供求解来源报告
    Nmat = _cij_name_matrix(cs) if cs is not None else None
    for m in modes:
        nm = m["name"]
        if nm not in all_strains:
            continue
        strains = all_strains[nm]
        delta_gpa = (all_stresses[nm] - baseline[np.newaxis, :]) * (-0.1)  # kBar→GPa, Cij 约定
        target = m.get("target", "")
        direction = vc_mode_direction(m)   # 应变方向, 用于标注每条曲线对应的弹性常数 (适配所有晶系)

        # 先对 6 个分量都拟合 (供热图/数据), 再只画对应非零弹性常数的分量
        fits = [_fit(strains, delta_gpa[:, i]) for i in range(6)]
        r2_row = np.array([fits[i][1] for i in range(6)])
        meaningful = _meaningful_components(Nmat, direction)
        strong = _strong_components(meaningful, r2_row)   # 只画/导出强通道
        for i in strong:
            combo_i = _cij_combo_label(Nmat, direction, i) if Nmat is not None else f"s{i+1}"
            strong_rows.append((nm, i, combo_i, fits[i][0], fits[i][1], fits[i][3]["kind"]))
        nr, ncols = _grid_for(len(strong))
        fig, axes = plt.subplots(nr, ncols, figsize=(ncols * 5.0, nr * 4.2), squeeze=False)
        axes_list = axes.flatten()
        fig.suptitle(f"SA-SS mode: {nm}   (target: {target})", fontsize=14, fontweight="bold")
        for ax_idx, i in enumerate(strong):
            ax = axes_list[ax_idx]
            x, y = strains, delta_gpa[:, i]
            slope, r2, curve, res_i = fits[i]
            ax.scatter(x, y, s=60, zorder=5, color="#159484", edgecolors="black", linewidth=0.5)
            if np.dot(x, x) > 0:
                xf = np.linspace(min(x) * 1.2, max(x) * 1.2, 100)
                ax.plot(xf, curve(xf), "r-", linewidth=1.5, label=f"slope={slope:.1f} GPa")
            combo = _cij_combo_label(Nmat, direction, i) if Nmat is not None else f"s{i+1}"
            note = _narrow_note(res_i)
            ax.set_title(f"{_VOIGT_LABELS[i]}  →  {combo} = {slope:.1f} GPa{note}\nR2={r2:.4f}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("strain", fontsize=9)
            ax.set_ylabel("d_stress (GPa)", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
        for k in range(len(strong), len(axes_list)):
            axes_list[k].axis("off")
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"vcfit_{nm}.png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        r2_by_mode[nm] = r2_row
        ok(f"VC fitting plot: {png_path}")

        # Origin 数据: 原始 (kBar) + GPa(measured+fitted)
        delta_kbar = all_stresses[nm] - baseline[np.newaxis, :]
        dat = os.path.join(out_dir, f"vcfit_{nm}_data.dat")
        with open(dat, "w") as f:
            f.write(f"# MHEC VC Stress-Strain Data: mode {nm} (target {target})\n")
            f.write("# Units: strain (dimensionless), stress change (kBar), baseline subtracted\n")
            f.write("strain\tds1_xx\tds2_yy\tds3_zz\tds4_yz\tds5_xz\tds6_xy\n")
            for row in np.column_stack([strains, delta_kbar]):
                f.write("\t".join(f"{v:.6e}" for v in row) + "\n")

        slopes = np.zeros(6)
        fitted = np.zeros_like(delta_gpa)
        for i in range(6):
            s, _r2, curve, _res = _fit(strains, delta_gpa[:, i])
            slopes[i] = s
            fitted[:, i] = curve(strains)
        # 只导出有意义(非零)分量, 每列列名标注对应弹性常数 (Origin 可直接读)
        gpa = os.path.join(out_dir, f"vcfit_{nm}_GPa.dat")
        cols = [np.asarray(strains, dtype=float)]
        cnames = ["strain"]
        for i in strong:
            combo = (_cij_combo_label(Nmat, direction, i) if Nmat is not None else f"s{i+1}")
            tag = combo.replace("·", "*")
            cols.append(delta_gpa[:, i]); cnames.append(f"{_VOIGT_LABELS[i]}[{tag}]_meas")
            cols.append(fitted[:, i]); cnames.append(f"{tag}_fit")
        with open(gpa, "w") as f:
            f.write(f"# MHEC VC Stress-Strain Data: mode {nm} (target {target})\n")
            f.write("# Units: strain (dimensionless), Cij-convention stress (GPa)\n")
            f.write("# Only symmetry-allowed components; each column maps to its elastic constant\n")
            f.write("\t".join(cnames) + "\n")
            for row in np.column_stack(cols):
                f.write("\t".join(f"{v:.6e}" for v in row) + "\n")

    # R² 热力图 (对称性零项标灰, 不参与好坏判断)
    names = [m["name"] for m in modes if m["name"] in r2_by_mode]
    if names:
        mat = np.array([r2_by_mode[n] for n in names], dtype=float)
        if Nmat is not None:
            mode_by_name = {m["name"]: m for m in modes}
            for r, nmn in enumerate(names):
                d = vc_mode_direction(mode_by_name[nmn])
                for j in range(6):
                    if _cij_combo_label(Nmat, d, j) == "0":
                        mat[r, j] = np.nan
        # 弱通道 (R² < 阈值) 也标灰排除, 使热图只反映强通道
        with np.errstate(invalid="ignore"):
            mat = np.where(mat < _R2_STRONG, np.nan, mat)
        import matplotlib.cm as _cm
        cmap = _cm.get_cmap("RdYlGn").copy()
        cmap.set_bad("#D9D9D9")
        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6 + 1)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(range(6))
        ax.set_xticklabels(_VOIGT_LABELS, fontsize=9)
        ax.set_title(f"VC Fit R2 Heatmap (only strong channels R2 >= {_R2_STRONG}; "
                     f"gray = symmetry-zero or weak, excluded)", fontsize=11)
        for k in range(len(names)):
            for j in range(6):
                v = mat[k, j]
                if np.isnan(v):
                    ax.text(j, k, "·", ha="center", va="center", fontsize=9, color="#777777")
                else:
                    ax.text(j, k, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="R2")
        plt.tight_layout()
        hp = os.path.join(out_dir, "vcfit_r2_heatmap.png")
        fig.savefig(hp, dpi=200, bbox_inches="tight")
        plt.close(fig)
        ok(f"VC R2 heatmap: {hp}")

    if strong_rows:
        rpt = _write_cij_solution_report(
            strong_rows, cij, Nmat, out_dir, "vcfit_solution.txt",
            "=== SA-SS: how each Cij is solved (strong channels, R2 >= %.2f) ===" % _R2_STRONG,
            cij_std=cij_std)
        ok(f"VC solution provenance: {os.path.join(out_dir, 'vcfit_solution.txt')}")
        print(rpt)
    ok("VC fitting data (Origin): vcfit_*_data.dat, vcfit_*_GPa.dat")


# ============================================================
# SS / SA 共用的拟合收尾
# ============================================================

def _finalize_elastic_fit(cij_raw, cij_sym, fit_info, baseline, cs, config,
                          out_dir, born_stable, born_warnings, temperature,
                          method):
    """SC-SS 与 SA-SS 共用的拟合收尾流程 (适配所有 9 个晶系):

    1) Wallace 残余压力诊断 + Brugger 修正;
    2) 写完整报告 elastic_constants.txt;
    3) 保存 elastic.txt (|P|>0.5 GPa 用 Brugger, 否则 Birch) + elastic_birch.txt;
    4) 提示是否立即进行力学性质分析 → func_601。

    fit_info 可为空 dict (SA-SS 的 VCElasticFitter 无逐模式 R² 信息)。
    """
    from .wallace import (compute_residual_pressure, apply_wallace_correction,
                          format_wallace_diagnostic)
    P_GPa, aniso_GPa = compute_residual_pressure(baseline)
    cij_brugger = apply_wallace_correction(cij_sym, P_GPa)
    print(format_wallace_diagnostic(cij_sym, cij_brugger, P_GPa, aniso_GPa))
    if abs(P_GPa) > 0.5:
        print_cij_matrix(cij_brugger,
                         T("Wallace 修正后弹性常数矩阵 (Brugger, 推荐值, GPa)",
                           "Wallace-corrected elastic matrix (Brugger, recommended, GPa)"))

    if born_stable:
        ok(T("Born 稳定性: 满足 ✓", "Born stability: satisfied ✓"))
    else:
        warn(T("Born 稳定性: 不满足 ✗", "Born stability: not satisfied ✗"))
        for w in (born_warnings or []):
            print(f"    {w}")

    write_results_file(
        os.path.join(out_dir, "elastic_constants.txt"),
        cij_raw=cij_raw,
        cij_sym=cij_sym,
        fit_info=fit_info or {},
        crystal_system=cs,
        temperature=temperature,
        magnitude=config.magnitude,
        method=method,
        born_stable=born_stable,
        born_warnings=born_warnings,
        baseline_stress=baseline,
        apply_wallace=True,
    )
    # elastic.txt 保存最终推荐矩阵: 残余压力显著时用 Brugger (Wallace 修正后),
    # 否则用 Birch (与修正后等价)。下游工具 (elasticpost 等) 直接读 elastic.txt
    # 即获得物理正确的弹性常数。
    elastic_txt_path = os.path.join(out_dir, "elastic.txt")
    if abs(P_GPa) > 0.5:
        save_elastic_matrix(elastic_txt_path, cij_brugger)
        save_elastic_matrix(os.path.join(out_dir, "elastic_birch.txt"), cij_sym)
        ok(T(f"elastic_constants.txt (完整报告) -> {out_dir}/",
             f"elastic_constants.txt (full report) -> {out_dir}/"))
        ok(T(f"elastic.txt (Brugger, Wallace 修正后 6×6 矩阵) -> {out_dir}/",
             f"elastic.txt (Brugger, Wallace-corrected 6×6 matrix) -> {out_dir}/"))
        info(T(f"elastic_birch.txt (原始 Birch 矩阵, 仅供参考) -> {out_dir}/",
               f"elastic_birch.txt (raw Birch matrix, for reference) -> {out_dir}/"))
    else:
        save_elastic_matrix(elastic_txt_path, cij_sym)
        ok(T(f"elastic_constants.txt (完整报告) -> {out_dir}/",
             f"elastic_constants.txt (full report) -> {out_dir}/"))
        ok(T(f"elastic.txt (6×6 矩阵, |P|<0.5 GPa Wallace 修正可忽略) -> {out_dir}/",
             f"elastic.txt (6×6 matrix; |P|<0.5 GPa, Wallace correction negligible) -> {out_dir}/"))

    do_mech = prompt_with_default(
        T("是否立即进行力学性质分析? (y/n)", "Run mechanical-property analysis now? (y/n)"), "y")
    if do_mech.lower() == "y":
        # 完整分析: VRH + 3D/2D 各向异性曲面图 + Origin 可直接读取的数据
        func_602(config,
                 default_matrix=elastic_txt_path,
                 default_outdir=os.path.join(out_dir, "elasticpost"))


# ============================================================
# 502) 弹性常数线性拟合
# ============================================================

def func_502(config: MHECConfig) -> None:
    sep(T("SC-SS 单分量应力-应变法 · 弹性常数拟合",
          "SC-SS Single-Component Stress-Strain · Fit elastic constants"))
    work_dir = prompt_with_default(T("工作目录", "Work directory"), ".")

    # 加载保存的配置: mhec.yaml 可能在 work_dir 或其上级 (工作根目录), 逐级向上查找
    # (与 SA-SS 拟合一致), 以确保 magnitude/n_points/strain_method/crystal_system 等
    # 参数与生成计算输入时保持一致
    saved_config_path = None
    _search = os.path.abspath(work_dir)
    for _ in range(4):
        _cand = os.path.join(_search, "mhec.yaml")
        if os.path.isfile(_cand):
            saved_config_path = _cand
            break
        _parent = os.path.dirname(_search)
        if _parent == _search:
            break
        _search = _parent
    if saved_config_path:
        try:
            saved = MHECConfig.from_file(saved_config_path)
            # 同步关键拟合参数
            if abs(saved.magnitude - config.magnitude) > 1e-12:
                info(T(f"从 {saved_config_path} 读取应变幅度 δ = {saved.magnitude} (覆盖当前默认值 {config.magnitude})",
                       f"Read strain amplitude δ = {saved.magnitude} from {saved_config_path} (overrides default {config.magnitude})"))
            config.magnitude = saved.magnitude
            config.n_points = saved.n_points
            config.strain_method = saved.strain_method
            if saved.crystal_system:
                config.crystal_system = saved.crystal_system
            if getattr(saved, "space_group", None):
                config.space_group = saved.space_group
            if saved.primitive_poscar:
                config.primitive_poscar = saved.primitive_poscar
            ok(T(f"已加载配置 ({saved_config_path}): δ={config.magnitude}, n_points={config.n_points}, method={config.strain_method}",
                 f"Loaded config ({saved_config_path}): δ={config.magnitude}, n_points={config.n_points}, method={config.strain_method}"))
        except Exception as e:
            warn(T(f"读取 {saved_config_path} 失败: {e}，使用当前会话配置",
                   f"Failed to read {saved_config_path}: {e}; using current session config"))
    else:
        # 没有保存的配置，明确提示当前将使用的 magnitude，给用户改正机会
        info(T("未找到 mhec.yaml (已向上查找 4 层)，将使用当前会话参数",
               "mhec.yaml not found (searched up to 4 parent levels); using current session parameters"))
        mag_str = prompt_with_default(
            T("请确认应变幅度 δ (生成计算时的值)", "Confirm strain amplitude δ (value used when generating)"),
            str(config.magnitude))
        try:
            config.magnitude = float(mag_str)
        except ValueError:
            warn(T(f"无效的应变幅度 '{mag_str}'，使用 {config.magnitude}",
                   f"Invalid strain amplitude '{mag_str}'; using {config.magnitude}"))

    skip = int(prompt_with_default(T("跳过初始步数", "Steps to skip at start"), str(config.skip_steps)))

    # 自动探测平衡构型 POSCAR：优先 equilibrium/POSCAR，其次 equilibrium/run/POSCAR
    eq_poscar_path = None
    for candidate in [
        os.path.join(work_dir, "equilibrium", "POSCAR"),
        os.path.join(work_dir, "equilibrium", "run", "POSCAR"),
        os.path.join(work_dir, "POSCAR"),
    ]:
        if os.path.isfile(candidate):
            eq_poscar_path = candidate
            break
    if eq_poscar_path is None:
        eq_poscar_path = prompt_with_default(T("平衡构型 POSCAR 路径", "Equilibrium POSCAR path"), "POSCAR")
    ok(T(f"平衡构型 POSCAR: {eq_poscar_path}", f"Equilibrium POSCAR: {eq_poscar_path}"))
    poscar = read_poscar(eq_poscar_path)

    cs = None
    if getattr(config, "space_group", None):
        from .crystal_system import space_group_to_crystal_system
        try:
            cs = space_group_to_crystal_system(int(config.space_group))
            info(T(f"由空间群 #{config.space_group} 推导晶系",
                   f"Crystal system derived from space group #{config.space_group}"))
        except (ValueError, TypeError):
            cs = None
    if cs is None and config.crystal_system:
        try:
            cs = CrystalSystem(config.crystal_system)
        except ValueError:
            cs = None
    if cs is None:
        # spglib 自动识别 (原子位置, 抗噪声/分 Laue class), 失败退回几何法
        from .crystal_system import crystal_system_from_structure
        cs = crystal_system_from_structure(poscar, symprec=max(config.crystal_tol, 1e-3))
        if cs is not None:
            info(T("由 spglib 从结构自动识别晶系", "Crystal system auto-detected from structure by spglib"))
        else:
            cs = identify_crystal_system(poscar["lattice"], config.crystal_tol)
    # 结果处理阶段: 始终保留一道人工确认/覆盖
    cs = _confirm_or_override_cs(cs)
    name = cs_name(cs)

    deform_codes = get_deform_codes(cs, config.strain_method)
    amp_labels = get_amplitude_labels(config.n_points)
    extractor = StressExtractor()

    # 自动探测平衡构型应力目录：优先 equilibrium/，其次 equilibrium/run/
    eq_run_dir = None
    for candidate in [
        os.path.join(work_dir, "equilibrium"),
        os.path.join(work_dir, "equilibrium", "run"),
    ]:
        if os.path.isdir(candidate) and (
            os.path.isfile(os.path.join(candidate, "vasprun.xml")) or
            os.path.isfile(os.path.join(candidate, "OUTCAR"))
        ):
            eq_run_dir = candidate
            break
    if eq_run_dir is None:
        eq_run_dir = prompt_with_default(T("平衡构型计算目录", "Equilibrium calculation directory"),
                                         os.path.join(work_dir, "equilibrium"))
    ok(T(f"平衡构型目录: {eq_run_dir}", f"Equilibrium directory: {eq_run_dir}"))

    try:
        baseline, _, eq_meta = extractor.extract_and_average(eq_run_dir, skip)
        ok(T(f"平衡构型应力: {eq_meta['n_used_steps']} 步 (来源: {eq_meta['source']})",
             f"Equilibrium stress: {eq_meta['n_used_steps']} steps (source: {eq_meta['source']})"))
    except Exception as e:
        warn(T(f"提取平衡构型应力失败: {e}", f"Failed to extract equilibrium stress: {e}"))
        return

    all_strains = {}
    all_stresses = {}
    _bar = ProgressBar(len(deform_codes) * len(amp_labels), desc=T("读取应力数据", "Reading stress data"))
    for code in deform_codes:
        strain_vals = []
        stress_vals = []
        base_strain = decode_deform(code, config.magnitude)
        for label, mult in amp_labels:
            # 自动探测: 优先 {code}_{label}/，其次 {code}_{label}/run/
            run_dir = None
            for candidate in [
                os.path.join(work_dir, f"{code}_{label}"),
                os.path.join(work_dir, f"{code}_{label}", "run"),
            ]:
                if os.path.isdir(candidate) and (
                    os.path.isfile(os.path.join(candidate, "vasprun.xml")) or
                    os.path.isfile(os.path.join(candidate, "OUTCAR"))
                ):
                    run_dir = candidate
                    break
            if run_dir is None:
                run_dir = os.path.join(work_dir, f"{code}_{label}")
            try:
                mean, _, _ = extractor.extract_and_average(run_dir, skip)
                strain_vals.append(config.magnitude * mult)
                stress_vals.append(mean)
            except Exception as e:
                warn(T(f"{code}_{label} 提取失败: {e}", f"{code}_{label} extraction failed: {e}"))
            _bar.update(info=f"{code}_{label}")
        if strain_vals:
            all_strains[code] = np.array(strain_vals)
            all_stresses[code] = np.array(stress_vals)
    _bar.close()

    if not all_strains:
        warn(T("无有效数据，无法拟合 (未找到任何 SC-SS deform* 变形目录)。",
               "No valid data (no SC-SS deform* directories found)."))
        # 防呆: 若目录其实是 SA-SS 模式命名 (uni_x/ortho/shear_*), 提示改用 SA-SS 拟合
        _sa_names = ("uni_x", "uni_y", "uni_z", "ortho", "shear_")
        _has_sa = any(any(d.startswith(p) for p in _sa_names)
                      for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)))
        if _has_sa:
            warn(T("检测到 SA-SS 变形目录 (uni_*/ortho/shear_*) —— 请改用 SA-SS 拟合(菜单 7)。",
                   "Found SA-SS deformation directories — use the SA-SS fit (menu 7) instead."))
        return

    fitter = ElasticFitter(cs)
    results = fitter.fit_all(
        deform_codes, all_strains, all_stresses, baseline, config.magnitude,
        fit_order=config.fit_poly_order,
    )

    # 屏幕输出流程与 SA-SS 完全一致: 打印矩阵 → 询问输出目录 → 诊断图 → 收尾
    print_cij_matrix(results["cij_raw"],
                     T("原始弹性常数矩阵 (GPa)", "Raw elastic constant matrix (GPa)"))
    print_cij_matrix(results["cij_symmetrized"],
                     T("对称性修正后弹性常数矩阵 (Birch, GPa)",
                       "Symmetrized elastic constant matrix (Birch, GPa)"))

    out_dir = prompt_with_default(T("结果输出目录", "Result output directory"), work_dir)
    # 生成拟合诊断图 (输出到用户指定目录, 与 SA-SS 一致)
    _plot_fitting_diagnostics(
        deform_codes, all_strains, all_stresses, baseline,
        results, config.magnitude, out_dir=out_dir,
        fit_order=config.fit_poly_order, cs=cs,
    )
    _finalize_elastic_fit(
        results["cij_raw"], results["cij_symmetrized"],
        results["fit_info_per_mode"], baseline, cs, config, out_dir,
        results["born_stable"], results["born_warnings"],
        config.temperatures[0], config.strain_method,
    )


# ============================================================
# 503) 对称性修正
# ============================================================

def func_503(config: MHECConfig) -> None:
    sep("对称性修正")
    matrix_path = prompt_with_default("弹性矩阵文件路径 (6×6)", "elastic.txt")
    if not os.path.isfile(matrix_path):
        warn(f"文件不存在: {matrix_path}")
        return
    cij_raw = np.loadtxt(matrix_path)
    if cij_raw.shape != (6, 6):
        warn(f"矩阵维度错误: {cij_raw.shape}，需要 (6, 6)")
        return

    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)

    fitter = ElasticFitter(cs)
    cij_sym, corrections = fitter.symmetrize(cij_raw)

    print_cij_matrix(cij_raw, "原始矩阵 (GPa)")
    print_cij_matrix(cij_sym, "对称性修正后矩阵 (GPa)")
    print(f"  最大修正量: {corrections['max_correction']:.4f} GPa")
    if corrections["warnings"]:
        for w in corrections["warnings"]:
            warn(w)

    out_path = prompt_with_default("保存修正后矩阵", "elastic_sym.txt")
    save_elastic_matrix(out_path, cij_sym)
    ok(f"已保存: {out_path}")


# ============================================================
# 504) Born 稳定性检验
# ============================================================

def func_504(config: MHECConfig) -> None:
    sep("Born 稳定性检验")
    from .born_stability import check_born_stability

    matrix_path = prompt_with_default("弹性矩阵文件路径 (6×6)", "elastic.txt")
    if not os.path.isfile(matrix_path):
        warn(f"文件不存在: {matrix_path}")
        return
    cij = np.loadtxt(matrix_path)
    if cij.shape != (6, 6):
        warn(f"矩阵维度错误: {cij.shape}，需要 (6, 6)")
        return

    poscar = _read_poscar_interactive()
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)

    eigenvalues = np.linalg.eigvalsh(cij)
    print(T("\n  弹性常数矩阵特征值:", "\n  Eigenvalues of the elastic matrix:"))
    for i, ev in enumerate(eigenvalues):
        status = "✓" if ev > 0 else "✗"
        print(f"    λ{i+1} = {ev:12.4f} GPa  {status}")

    stable, violations = check_born_stability(cij, cs)
    if stable:
        ok(T("Born 稳定性: 满足 ✓", "Born stability: satisfied ✓"))
    else:
        warn(T("Born 稳定性: 不满足 ✗", "Born stability: not satisfied ✗"))
        for v in violations:
            print(f"    {v}")


# ============================================================
# 601) 力学性质快速计算 (VRH)
# ============================================================

def _auto_material_props(matrix_path: str):
    """从弹性矩阵附近的 POSCAR 自动推算 (密度 g/cm³, 每原子平均摩尔质量 g/mol, 来源串)。

    找不到 POSCAR 或元素不全时返回 (None, None, None)。
    每原子平均摩尔质量正是德拜温度公式所需 (原子数密度 n = ρ·N_A / M_atom)。
    """
    from .periodic import composition_info
    mdir = os.path.dirname(os.path.abspath(matrix_path))
    candidates = [
        os.path.join(mdir, "POSCAR"),
        os.path.join(mdir, "CONTCAR"),
        os.path.join(mdir, "equilibrium", "POSCAR"),
        os.path.join(mdir, "equilibrium", "run", "POSCAR"),
        os.path.join(os.path.dirname(mdir), "POSCAR"),
        "POSCAR",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                pc = read_poscar(p)
                lat = pc["lattice"]
                vol = abs(np.dot(lat[0], np.cross(lat[1], lat[2])))
                comp = composition_info(pc.get("species"), pc.get("counts") or [], vol)
                if comp:
                    return comp["density"], comp["mass_per_atom"], f"{comp['formula']} ({os.path.relpath(p)})"
            except Exception:
                pass
    return None, None, None


def func_601(config: MHECConfig) -> None:
    sep(T("力学性质快速计算 (VRH)", "Quick mechanical properties (VRH)"))
    from .adapters import ElasticPostAdapter

    matrix_path = prompt_with_default(T("弹性矩阵文件路径", "Elastic matrix file path"), "elastic.txt")
    if not os.path.isfile(matrix_path):
        warn(T(f"文件不存在: {matrix_path}", f"File not found: {matrix_path}"))
        return
    cij = np.loadtxt(matrix_path)
    cs_str = prompt_with_default(T("晶系", "Crystal system"), "cubic")
    auto_rho, auto_m, auto_src = _auto_material_props(matrix_path)
    if auto_rho:
        ok(T(f"从 POSCAR 自动推算 {auto_src}: 密度={auto_rho:.4f} g/cm³, 每原子摩尔质量={auto_m:.4f} g/mol",
             f"Auto from POSCAR {auto_src}: density={auto_rho:.4f} g/cm³, molar mass/atom={auto_m:.4f} g/mol"))
    density_str = prompt_with_default(
        T("材料密度 (g/cm³)", "Density (g/cm³)"), f"{auto_rho:.4f}" if auto_rho else "5.0")
    molar_mass_str = prompt_with_default(
        T("摩尔质量 (g/mol, 每原子平均)", "Molar mass (g/mol, per atom)"), f"{auto_m:.4f}" if auto_m else "50.0")

    adapter = ElasticPostAdapter(
        density=float(density_str),
        molar_mass=float(molar_mass_str),
    )
    result = adapter.calculate(cij, cs_str)
    if result:
        for k, v in result.items():
            if k == "report":
                print(f"\n{v}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        print(f"    {kk}: {vv:.4f}")
                    elif not isinstance(vv, np.ndarray):
                        print(f"    {kk}: {vv}")
            elif not isinstance(v, np.ndarray):
                print(f"  {k}: {v}")


# ============================================================
# 602) 力学性质完整分析
# ============================================================

def func_602(config: MHECConfig, default_matrix: str = "elastic.txt",
             default_outdir: str = "elasticpost_results") -> None:
    sep(T("力学性质完整分析", "Full mechanical-property analysis"))
    from .adapters import ElasticPostAdapter

    matrix_path = prompt_with_default(T("弹性矩阵文件路径", "Elastic matrix file path"), default_matrix)
    output_dir = prompt_with_default(T("输出目录", "Output directory"), default_outdir)
    auto_rho, auto_m, auto_src = _auto_material_props(matrix_path)
    if auto_rho:
        ok(T(f"从 POSCAR 自动推算 {auto_src}: 密度={auto_rho:.4f} g/cm³, 每原子摩尔质量={auto_m:.4f} g/mol",
             f"Auto from POSCAR {auto_src}: density={auto_rho:.4f} g/cm³, molar mass/atom={auto_m:.4f} g/mol"))
    density_str = prompt_with_default(
        T("材料密度 (g/cm³)", "Density (g/cm³)"), f"{auto_rho:.4f}" if auto_rho else "5.0")
    molar_mass_str = prompt_with_default(
        T("摩尔质量 (g/mol, 每原子平均)", "Molar mass (g/mol, per atom)"), f"{auto_m:.4f}" if auto_m else "50.0")

    adapter = ElasticPostAdapter(
        density=float(density_str),
        molar_mass=float(molar_mass_str),
        output_dir=output_dir,
    )
    result = adapter.run_full_analysis(matrix_path, output_dir)
    print(T(f"\n  分析状态: {result.get('status', 'unknown')}",
            f"\n  Analysis status: {result.get('status', 'unknown')}"))
    if result.get("output_dir"):
        ok(T(f"输出目录: {result['output_dir']}", f"Output directory: {result['output_dir']}"))


# ============================================================
# 603) AIMD 后处理 (完整分析)
# ============================================================

def func_603(config: MHECConfig) -> None:
    sep(T("AIMD 后处理 (完整分析)", "AIMD post-processing (full analysis)"))
    from .adapters import AIMDPostAdapter

    print(T("  将执行完整 AIMD 后处理分析，包括:", "  Full AIMD post-processing will include:"))
    print(T("    RDF / MSD / VACF+PDOS / Green-Kubo 粘度", "    RDF / MSD / VACF+PDOS / Green-Kubo viscosity"))
    print(T("    结构因子 S(q) / 配位数 / 键长键角分布", "    Structure factor S(q) / coordination / bond length & angle"))
    print(T("    Bootstrap 置信区间 / 分子摩擦系数", "    Bootstrap CI / molecular friction coefficient"))
    print()

    work_dir = prompt_with_default(T("计算目录路径", "Calculation directory path"), ".")
    strain_label = prompt_with_default(T("应变标签", "Strain label"), "equilibrium")
    stage = prompt_with_default(T("MLFF 阶段", "MLFF stage"), "run")
    temp_str = prompt_with_default(T("温度 (K)", "Temperature (K)"), str(config.temperatures[0]))
    potim_str = prompt_with_default("POTIM (fs)", str(config.potim))
    n_blocks_str = prompt_with_default(T("块平均块数", "Number of blocks"), "5")
    stable_frac_str = prompt_with_default(T("丢弃初始比例", "Initial fraction to discard"), "0.3")
    r_eff_str = prompt_with_default(T("有效粒子半径 (Å, 留空自动)", "Effective particle radius (Å, empty=auto)"), "")

    extra_kwargs = {}
    if r_eff_str:
        extra_kwargs["r_eff"] = float(r_eff_str)

    do_friction = prompt_with_default(T("是否计算宏观摩擦系数? (y/n)", "Compute macroscopic friction coefficient? (y/n)"), "n")
    if do_friction.lower() == "y":
        friction_U = prompt_with_default(T("滑动速度 (m/s, 空格分隔)", "Sliding velocities (m/s, space-separated)"), "0.01 0.1 1.0")
        friction_p = prompt_with_default(T("载荷压力 (Pa)", "Load pressure (Pa)"), "1e6")
        friction_h = prompt_with_default(T("润滑膜厚度 (m)", "Lubricant film thickness (m)"), "1e-9")
        extra_kwargs["friction_U"] = friction_U
        extra_kwargs["friction_p"] = float(friction_p)
        extra_kwargs["friction_h"] = float(friction_h)

    adapter = AIMDPostAdapter(
        temperature=float(temp_str),
        potim=float(potim_str),
        n_blocks=int(n_blocks_str),
        stable_frac=float(stable_frac_str),
    )
    result = adapter.process(work_dir, strain_label, stage, **extra_kwargs)
    if result:
        print(T(f"\n  后处理状态: {result.get('status', 'unknown')}",
                f"\n  Post-processing status: {result.get('status', 'unknown')}"))
        if result.get("output_prefix"):
            ok(T(f"输出文件前缀: {result['output_prefix']}", f"Output file prefix: {result['output_prefix']}"))


# ============================================================
# 701) MLFF 力场精度验证
# ============================================================

def func_701(config: MHECConfig) -> None:
    sep(T("MLFF 力场精度验证", "MLFF accuracy validation"))
    print(T("  1) 分析 ML_LOGFILE", "  1) Analyze ML_LOGFILE"))
    print(T("  2) 生成 ML_AB 评估目录", "  2) Generate ML_AB evaluation directories"))
    print(T("  3) 收集 ML_AB 评估结果", "  3) Collect ML_AB evaluation results"))
    choice = prompt_with_default(T("选择", "Select"), "1")

    if choice == "1":
        path = prompt_with_default(T("ML_LOGFILE 路径", "ML_LOGFILE path"), "ML_LOGFILE")
        if not os.path.isfile(path):
            warn(T(f"文件不存在: {path}", f"File not found: {path}"))
            return
        try:
            result = parse_ml_logfile(path)
            print_validation_report(result, path)
        except Exception as e:
            warn(T(f"解析失败: {e}", f"Parsing failed: {e}"))

    elif choice == "2":
        _func_ab_eval_generate(config)

    elif choice == "3":
        _func_ab_eval_collect(config)

    else:
        warn(T(f"无效选择: {choice}", f"Invalid choice: {choice}"))


def _func_ab_eval_generate(config: MHECConfig) -> None:
    """生成 ML_AB 全数据集 MLFF 评估目录 (eval_mlff 方法)。"""
    sep(T("生成 ML_AB 评估目录 (ML_MODE=run)", "Generate ML_AB evaluation directories (ML_MODE=run)"))
    from .mlff_ab_eval import generate_ab_eval_dirs

    print(T("  方法: 以 ML_AB 中存储的 DFT 参考值为基准, 对每个训练构型用 ML_FF",
            "  Method: using DFT references stored in ML_AB, re-predict each training"))
    print(T("        重新预测 (ML_MODE=run), 得到覆盖整个训练集的 E/F/S parity。",
            "        configuration with ML_FF (ML_MODE=run) → E/F/S parity over the whole training set."))
    print(T("  前提: 已完成 MLFF 训练 (train→refit), 得到 ML_FF 和 ML_AB。",
            "  Prerequisite: MLFF training (train→refit) completed, with ML_FF and ML_AB available."))
    print()

    ml_ab = prompt_with_default(T("ML_AB 文件路径", "ML_AB file path"), "ML_AB")
    if not os.path.isfile(ml_ab):
        warn(T(f"文件不存在: {ml_ab}", f"File not found: {ml_ab}"))
        return
    ml_ff = prompt_with_default(T("ML_FF 力场路径", "ML_FF force field path"), "ML_FF")
    potcar = prompt_with_default(T("POTCAR 路径", "POTCAR path"), "POTCAR")
    kpoints = prompt_with_default(T("KPOINTS 路径", "KPOINTS path"), "KPOINTS")
    for fp, nm in [(ml_ff, "ML_FF"), (potcar, "POTCAR"), (kpoints, "KPOINTS")]:
        if not os.path.isfile(fp):
            warn(T(f"文件不存在: {nm} ({fp})", f"File not found: {nm} ({fp})"))
            return

    out_dir = prompt_with_default(T("评估目录输出路径", "Evaluation output directory"), "mlff_ab_eval")
    user_incar = prompt_with_default(
        T("本地参考 INCAR (存在则自动改写为评估用, 否则用内置默认)",
          "Local reference INCAR (rewritten for evaluation if present, else built-in default)"), "INCAR")
    max_str = prompt_with_default(
        T("评估构型数 (0=全部, >0=随机抽取, <0=取前N个)",
          "Number of configurations (0=all, >0=random sample, <0=first N)"), "0")
    try:
        max_structures = int(max_str)
    except ValueError:
        max_structures = 0

    sys_label = _detect_system_label() or "Material"
    sys_name = sys_label.split(" (")[0].replace(" ", "")

    if os.path.isfile(user_incar):
        ok(T(f"将基于本地 INCAR 改写为评估用: {user_incar}",
             f"Will adapt local INCAR for evaluation: {user_incar}"))
    else:
        info(T(f"未找到本地 INCAR ({user_incar}), 将使用内置默认模板",
               f"Local INCAR not found ({user_incar}); using built-in default template"))

    # 直接读取用户的提交脚本模板 (其 #SBATCH / conda / oneAPI / 核数 / vasp 命令
    # 都按用户集群配置好), 无需在此重复询问集群参数。留空则用内置默认。
    tpl_default = config.submit_template or ("run.sh" if os.path.isfile("run.sh") else "")
    submit_tpl = prompt_with_default(
        T("提交脚本模板 (复用其 #SBATCH/环境/核数/VASP 命令; 留空=内置默认)",
          "Submit-script template (reuses its #SBATCH/env/cores/VASP command; empty=built-in default)"),
        tpl_default)
    submit_template = submit_tpl if (submit_tpl and os.path.isfile(submit_tpl)) else None
    if submit_tpl and submit_template is None:
        warn(T(f"模板不存在: {submit_tpl}, 改用内置默认脚本",
               f"Template not found: {submit_tpl}; using built-in default script"))
    try:
        result = generate_ab_eval_dirs(
            ml_ab=ml_ab, ml_ff=ml_ff, potcar=potcar, kpoints=kpoints,
            out_dir=out_dir, system=sys_name, encut=config.encut or 500,
            max_structures=max_structures, user_incar=user_incar,
            submit_template=submit_template,
        )
    except Exception as e:
        warn(T(f"生成失败: {e}", f"Generation failed: {e}"))
        return

    ok(T(f"已生成 {result['n_structures']} 个构型评估目录 → {result['out_dir']}",
         f"Generated {result['n_structures']} configuration directories → {result['out_dir']}"))
    info(T(f"提交: cd {out_dir} && sbatch submit_ab_eval.sh  (或 bash submit_ab_eval.sh)",
           f"Submit: cd {out_dir} && sbatch submit_ab_eval.sh  (or bash submit_ab_eval.sh)"))
    info(T("完成后回到本菜单选择 7) 收集结果并出图",
           "After completion, return to this menu and choose 3) to collect results and plot"))


def _func_ab_eval_collect(config: MHECConfig) -> None:
    """收集 ML_AB 评估结果并出 parity 图。"""
    sep(T("收集 ML_AB 评估结果 → parity 图", "Collect ML_AB evaluation results → parity plots"))
    from .mlff_ab_eval import collect_ab_parity
    from .mlff_validator import plot_parity, write_parity_report

    out_dir = prompt_with_default(T("评估目录路径", "Evaluation directory path"), "mlff_ab_eval")
    if not os.path.isdir(out_dir):
        warn(T(f"目录不存在: {out_dir}", f"Directory not found: {out_dir}"))
        return

    ml_ab = None
    if not os.path.isfile(os.path.join(out_dir, "ML_AB")):
        ml_ab = prompt_with_default(T("ML_AB 路径 (目录内未找到副本)", "ML_AB path (no copy found in directory)"), "ML_AB")
        if not os.path.isfile(ml_ab):
            warn(T(f"文件不存在: {ml_ab}", f"File not found: {ml_ab}"))
            return

    info(T("正在收集 struct_*/OUTCAR 并配对 ML_AB 的 DFT 参考 ...",
           "Collecting struct_*/OUTCAR and pairing with ML_AB DFT references ..."))
    try:
        parity = collect_ab_parity(out_dir, ml_ab)
    except Exception as e:
        warn(T(f"收集失败: {e}", f"Collection failed: {e}"))
        return

    if parity.n_frames == 0:
        warn(T("没有可配对的构型, 请确认 ML_MODE=run 任务是否完成。",
               "No pairable configurations; check whether the ML_MODE=run jobs finished."))
        return
    ok(T(f"成功配对 {parity.n_frames} 个构型", f"Paired {parity.n_frames} configurations"))

    metrics = parity.metrics()
    hdr = T("物理量", "Quantity"); unit_h = T("单位", "Unit")
    print(f"\n  {hdr:>8}  {'RMSE':>12}  {'MAE':>12}  {'R²':>8}  {'N':>8}  {unit_h}")
    print("  " + "─" * 62)
    name_map = {"energy": T("能量", "Energy"), "force": T("力", "Force"), "stress": T("应力", "Stress")}
    for name in ["energy", "force", "stress"]:
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"  {name_map[name]:>8}  {m['rmse']:>12.4f}  {m['mae']:>12.4f}  "
              f"{m['r2']:>8.4f}  {m['n']:>8}  {m['unit']}")

    plot_dir = os.path.join(out_dir, "parity")
    os.makedirs(plot_dir, exist_ok=True)
    info(T("正在生成 parity 图 ...", "Generating parity plots ..."))
    try:
        plot_paths = plot_parity(parity, plot_dir)
        for name, p in plot_paths.items():
            ok(f"{name}_parity.svg  → {p}")
    except Exception as e:
        warn(T(f"绘图失败: {e}", f"Plotting failed: {e}"))
        plot_paths = None

    report_path = write_parity_report(parity, plot_dir, plot_paths)
    ok(T(f"完整报告 → {report_path}", f"Full report → {report_path}"))


# ============================================================
# 801) 提取平衡晶格 + 热膨胀系数
# ============================================================

def func_801(config: MHECConfig) -> None:
    sep(T("提取平衡晶格 + 热膨胀系数", "Equilibrium lattice & thermal expansion"))

    # 自动探测 NPT 目录
    default_npt = "phase1_npt" if os.path.isdir("phase1_npt") else "npt_opt"
    npt_base = prompt_with_default(T("NPT 工作目录", "NPT work directory"), default_npt)
    if not os.path.isdir(npt_base):
        warn(T(f"目录不存在: {npt_base}", f"Directory not found: {npt_base}"))
        return

    skip_frac = float(prompt_with_default(T("丢弃初始比例", "Initial fraction to discard"), "0.5"))

    # 自动探测温度目录
    detected_temps = []
    import re as _re
    for d in sorted(os.listdir(npt_base)):
        m = _re.match(r"npt_(\d+)K$", d)
        if m and os.path.isdir(os.path.join(npt_base, d)):
            detected_temps.append(float(m.group(1)))
    if not detected_temps:
        warn(T(f"在 {npt_base} 中未找到 npt_*K 目录", f"No npt_*K directories found in {npt_base}"))
        return
    info(T(f"检测到 {len(detected_temps)} 个温度: {', '.join(str(int(t)) for t in detected_temps)} K",
           f"Detected {len(detected_temps)} temperatures: {', '.join(str(int(t)) for t in detected_temps)} K"))
    config.temperatures = detected_temps

    # 读取第一个温度的 CONTCAR/POSCAR 作为模板（获取原子信息）
    first_temp_dir = os.path.join(npt_base, f"npt_{int(detected_temps[0])}K")
    template_poscar = None
    for fname in ["CONTCAR", "POSCAR"]:
        for d in [first_temp_dir, os.path.join(first_temp_dir, "run")]:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                template_poscar = read_poscar(p)
                break
        if template_poscar:
            break
    if template_poscar is None:
        warn(T("无法读取任何 CONTCAR 或 POSCAR 作为模板", "Could not read any CONTCAR or POSCAR as template"))
        return

    optimizer = LatticeOptimizer(
        poscar=template_poscar,
        temperatures=config.temperatures,
        nsw_train=config.nsw_train,
        nsw_refit=config.nsw_refit,
        nsw_run=config.nsw_run,
        potim=config.potim,
        encut=config.encut,
        user_overrides=config.incar_overrides,
        skip_steps=config.skip_steps,
    )

    info(T("从 NPT 轨迹提取平衡晶格...", "Extracting equilibrium lattice from NPT trajectories..."))
    lattice_data, te_result = optimizer.extract_all_equilibrium(
        npt_base, skip_frac=skip_frac, run_only=True)

    if not lattice_data:
        warn(T("未能提取任何温度的平衡晶格。", "Failed to extract equilibrium lattice for any temperature."))
        return

    # 保存各温度的平衡 POSCAR
    out_dir = prompt_with_default(T("平衡结构输出目录", "Equilibrium structure output directory"), "equilibrium_structures")
    os.makedirs(out_dir, exist_ok=True)
    for temp, (avg_lat, stats) in sorted(lattice_data.items()):
        eq_poscar = {
            "comment": template_poscar["comment"],
            "scale": template_poscar["scale"],
            "lattice": avg_lat,
            "species": template_poscar.get("species"),
            "counts": template_poscar["counts"],
            "selective": template_poscar.get("selective", False),
            "coord_type": template_poscar["coord_type"],
            "positions": template_poscar["positions"],
        }
        fname = os.path.join(out_dir, f"POSCAR_{int(temp)}K")
        write_poscar(fname, eq_poscar)
        ok(f"{int(temp)} K → {fname}")

    ok(T(f"平衡结构已保存到 {out_dir}/", f"Equilibrium structures saved to {out_dir}/"))
    if te_result:
        info(T("热膨胀系数数据已保存到 NPT 工作目录。",
               "Thermal-expansion coefficient data saved to the NPT work directory."))


def _write_nvt_submit(nvt_base, lattice_data, codes, amp_labels):
    """生成 NVT 弹性常数批量提交脚本。"""
    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --partition=8358P",
        "#SBATCH --job-name=mhec_nvt",
        "#SBATCH --output=submit_nvt_%j.out",
        "#SBATCH --error=submit_nvt_%j.err",
        "",
        "# MHEC NVT 弹性常数批量提交脚本",
        "# 用法: sbatch submit_nvt.sh  或  bash submit_nvt.sh",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        'fi',
        "",
        "NVT_JOBS=()",
        "",
    ]

    for temp in sorted(lattice_data.keys()):
        temp_label = f"{int(temp)}K"
        lines.append(f"echo '>>> {temp_label} NVT 弹性常数'")

        # equilibrium
        lines.append(f'pushd "$WORK_ROOT/{temp_label}/equilibrium" > /dev/null')
        lines.append('JOB_ID=$(sbatch --parsable run.sh)')
        lines.append('NVT_JOBS+=($JOB_ID)')
        lines.append(f'echo "  {temp_label}/equilibrium: $JOB_ID"')
        lines.append('popd > /dev/null')

        # deformations
        for code in codes:
            for label, _ in amp_labels:
                rel = f"{temp_label}/{code}_{label}"
                lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
                lines.append('JOB_ID=$(sbatch --parsable run.sh)')
                lines.append('NVT_JOBS+=($JOB_ID)')
                lines.append(f'echo "  {rel}: $JOB_ID"')
                lines.append('popd > /dev/null')
        lines.append("")

    lines.append('echo ">>> 所有 NVT 任务已提交: ${#NVT_JOBS[@]} 个"')

    path = os.path.join(nvt_base, "submit_nvt.sh")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  +-> {path}")


def _default_training_strains(config: MHECConfig, cs=None) -> list:
    """MLFF 训练默认应变幅度: 与弹性计算的应变采样单一来源保持一致。

    取 config 解析后的正向应变点 (覆盖到用户设定的最大应变), 这样训练域
    正好覆盖后续弹性拟合会探测的应变范围; 若解析失败则回退到 DEFAULT_STRAIN_MAGNITUDES。
    """
    from .config import (resolve_strain_sampling, strain_amplitude_list,
                         default_n_points_for_system)
    from .run_mlff_pipeline import DEFAULT_STRAIN_MAGNITUDES
    try:
        n_pts = getattr(config, "n_points", None)
        if not n_pts and cs is not None:
            n_pts = default_n_points_for_system(cs)
        mag, n_pts = resolve_strain_sampling(
            magnitude=getattr(config, "magnitude", 0.005),
            n_points=n_pts or 9,
            max_strain=getattr(config, "max_strain", None),
            strain_points=getattr(config, "strain_points", None),
        )
        pos = sorted(a for a in strain_amplitude_list(mag, n_pts) if a > 1e-9)
        return [round(a, 8) for a in pos] if pos else list(DEFAULT_STRAIN_MAGNITUDES)
    except Exception:
        return list(DEFAULT_STRAIN_MAGNITUDES)


# ============================================================
# 901) MLFF 高精度训练
# ============================================================

def func_901(config: MHECConfig) -> None:
    sep(T("MLFF 高精度训练", "MLFF high-accuracy training"))

    # 检查必需文件
    missing = []
    for f in ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]:
        if not os.path.isfile(f):
            missing.append(f)
    if missing:
        warn(T(f"缺少文件: {', '.join(missing)}", f"Missing files: {', '.join(missing)}"))
        info(T("请在当前目录准备: POSCAR, POTCAR, KPOINTS, run.sh",
               "Prepare in the current directory: POSCAR, POTCAR, KPOINTS, run.sh"))
        return

    from .run_mlff_pipeline import (
        generate_mlff_training,
        DEFAULT_TRAIN_TEMPERATURES_K,
        DEFAULT_STRAIN_MAGNITUDES,
    )
    from .vasp_io import read_poscar
    from .crystal_system import identify_crystal_system, CrystalSystem

    # 晶系识别: spglib 读空间群分 Laue class (4/mmm vs 4/m) + 人工确认
    poscar = read_poscar("POSCAR")
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    print()

    temp_str = prompt_with_default(
        T("训练温度 (K, 空格分隔)", "Training temperatures (K, space-separated)"),
        " ".join(str(int(t)) for t in DEFAULT_TRAIN_TEMPERATURES_K))
    try:
        temps = [float(t) for t in temp_str.split()]
    except ValueError:
        temps = list(DEFAULT_TRAIN_TEMPERATURES_K)

    _default_strains = _default_training_strains(config, cs)
    strain_str = prompt_with_default(
        T("应变幅度 (空格分隔)", "Strain amplitudes (space-separated)"),
        " ".join(str(s) for s in _default_strains))
    try:
        strains = [float(s) for s in strain_str.split()]
    except ValueError:
        strains = list(_default_strains)

    mode_str = prompt_with_default(T("弹性训练模式 (auto / general)", "Elastic training mode (auto / general)"),
                                   "auto").strip().lower()
    if mode_str not in ("auto", "general"):
        warn(T(f"未知模式 {mode_str!r}，使用 auto", f"Unknown mode {mode_str!r}; using auto"))
        mode_str = "auto"

    nsw = int(prompt_with_default(T("每个训练任务 MD 步数", "MD steps per training job"), "5000"))
    out_dir = prompt_with_default(T("训练工作目录", "Training work directory"), "mlff_train")

    result = generate_mlff_training(
        work_dir=out_dir,
        temperatures=temps,
        strain_magnitudes=strains,
        nsw_train=nsw,
        potim=config.potim,
        encut=config.encut,
        elastic_training_mode=mode_str,
    )

    if result:
        info(T(f"提交训练: sbatch {out_dir}/submit_train.sh", f"Submit training: sbatch {out_dir}/submit_train.sh"))
        info(T("串联+refit: submit_train.sh 会在全部训练作业成功后自动提交（或手动 --refit --auto-run）",
               "Chained + refit: submit_train.sh auto-submits refit after all training jobs succeed (or use --refit --auto-run)"))


# ============================================================
# 902) MLFF 升温 ramp 训练
# ============================================================

def func_902(config: MHECConfig) -> None:
    sep(T("MLFF 升温训练", "MLFF heating-ramp training"))

    missing = [f for f in ["POSCAR", "POTCAR", "KPOINTS", "run.sh"] if not os.path.isfile(f)]
    if missing:
        warn(T(f"缺少文件: {', '.join(missing)}", f"Missing files: {', '.join(missing)}"))
        info(T("请在当前目录准备: POSCAR, POTCAR, KPOINTS, run.sh",
               "Prepare in the current directory: POSCAR, POTCAR, KPOINTS, run.sh"))
        return

    from .run_mlff_pipeline import (
        generate_mlff_training_ramp, DEFAULT_STRAIN_MAGNITUDES,
    )
    from .crystal_system import identify_crystal_system, CrystalSystem

    # 晶系识别: spglib 读空间群分 Laue class (4/mmm vs 4/m) + 人工确认
    poscar = read_poscar("POSCAR")
    cs = _identify_crystal_interactive(poscar, config.crystal_tol, config)
    print()

    t_start = prompt_with_default(T("起始温度 (K)", "Start temperature (K)"), "300")
    t_end = prompt_with_default(T("结束温度 (K)", "End temperature (K)"), "1500")
    try:
        t_start = float(t_start); t_end = float(t_end)
    except ValueError:
        t_start, t_end = 300.0, 1500.0

    _default_strains = _default_training_strains(config, cs)
    strain_str = prompt_with_default(
        T("应变幅度 (空格分隔)", "Strain amplitudes (space-separated)"),
        " ".join(str(s) for s in _default_strains))
    try:
        strains = [float(s) for s in strain_str.split()]
    except ValueError:
        strains = list(_default_strains)

    mode_str = prompt_with_default(T("弹性训练模式 (auto / general)", "Elastic training mode (auto / general)"),
                                   "auto").strip().lower()
    if mode_str not in ("auto", "general"):
        warn(T(f"未知模式 {mode_str!r}, 使用 auto", f"Unknown mode {mode_str!r}; using auto"))
        mode_str = "auto"

    nsw = int(prompt_with_default(T("每段升温 MD 步数 (要够大以采满温区)", "MD steps per ramp segment (large enough to cover the temperature range)"), "30000"))
    out_dir = prompt_with_default(T("训练工作目录", "Training work directory"), "mlff_train_ramp")

    result = generate_mlff_training_ramp(
        work_dir=out_dir, t_start=t_start, t_end=t_end,
        strain_magnitudes=strains, nsw_ramp=nsw,
        potim=config.potim, encut=config.encut,
        elastic_training_mode=mode_str,
    )
    if result:
        info(T(f"提交训练: sbatch {out_dir}/submit_train.sh  (顺序串行: A→B…→refit)",
               f"Submit training: sbatch {out_dir}/submit_train.sh  (sequential: A→B…→refit)"))
        info(T(f"完成后: cp {out_dir}/refit_final/ML_FFN  ML_FF",
               f"After completion: cp {out_dir}/refit_final/ML_FFN  ML_FF"))


# ============================================================
# SA-SS) 对称化应力-应变法 (对称化组合应变): 生成与拟合
# ============================================================

def func_vc_generate(config: MHECConfig) -> None:
    # SA-SS 与 SC-SS 共用完整工作流 (NPT 优化各温度晶格 → 提取平衡晶格 → NVT 变形采样)，
    # 仅变形集不同 (SA-SS 用对称化组合应变)。委托给统一的 func_403，传入 method="sa"。
    func_403(config, method="sa")


def _confirm_or_override_cs(cs: CrystalSystem) -> CrystalSystem:
    """打印当前晶系并让用户确认或按编号覆盖 (拟合命令共用)。

    几何自动识别分不清 4/mmm 与 4/m 这类 Laue class, 且 MD 噪声可能把
    立方误判为正交/四方, 因此拟合前必须给用户一次确认/纠正的机会。
    """
    options = list(CrystalSystem)
    name = cs_name(cs)
    ok(T(f"晶系: {name} ({N_INDEPENDENT[cs]} 个独立弹性常数)",
         f"Crystal system: {name} ({N_INDEPENDENT[cs]} independent elastic constants)"))
    ind_txt = T("个独立常数", "independent constants")
    print(T("\n  可选晶系 (回车=接受当前, 或输入编号覆盖):",
            "\n  Crystal systems (Enter=accept current, or type a number to override):"))
    for i, opt in enumerate(options):
        marker = " <=" if opt == cs else ""
        print(f"    {i}) {cs_name(opt)} ({N_INDEPENDENT[opt]} {ind_txt}){marker}")
    idx = prompt_with_default(T("确认晶系", "Confirm crystal system"), "")
    if idx.strip().isdigit() and 0 <= int(idx) < len(options):
        cs = options[int(idx)]
        ok(T(f"已选择: {cs_name(cs)} ({N_INDEPENDENT[cs]} 个独立弹性常数)",
             f"Selected: {cs_name(cs)} ({N_INDEPENDENT[cs]} independent elastic constants)"))
    return cs


def func_vc_fit(config: MHECConfig) -> None:
    sep(T("SA-SS 对称化应力-应变法 · 弹性常数拟合",
          "SA-SS Symmetry-Adapted Stress-Strain · Fit elastic constants"))
    work_dir = prompt_with_default(T("工作目录", "Work directory"), ".")
    if not os.path.isdir(work_dir):
        warn(T(f"目录不存在: {work_dir}", f"Directory not found: {work_dir}"))
        return

    # 加载保存的配置: mhec.yaml 位于 work_dir 的上级 (工作根目录)，逐级向上查找
    saved_config_path = None
    search = os.path.abspath(work_dir)
    for _ in range(4):
        cand = os.path.join(search, "mhec.yaml")
        if os.path.isfile(cand):
            saved_config_path = cand
            break
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    if saved_config_path:
        try:
            saved = MHECConfig.from_file(saved_config_path)
            config.magnitude = saved.magnitude
            config.n_points = saved.n_points
            if saved.crystal_system:
                config.crystal_system = saved.crystal_system
            if getattr(saved, "space_group", None):
                config.space_group = saved.space_group
            ok(T(f"已加载配置 ({saved_config_path}): δ={config.magnitude}, n_points={config.n_points}",
                 f"Loaded config ({saved_config_path}): δ={config.magnitude}, n_points={config.n_points}"))
        except Exception as e:
            warn(T(f"读取配置失败: {e}", f"Failed to read config: {e}"))

    skip = int(prompt_with_default(T("跳过初始步数", "Steps to skip at start"), str(config.skip_steps)))

    # 识别晶系: 空间群 > 配置晶系 > .mhec_vc 标记 > 平衡结构自动识别
    cs = None
    flag = os.path.join(work_dir, ".mhec_vc")
    if getattr(config, "space_group", None):
        from .crystal_system import space_group_to_crystal_system
        try:
            cs = space_group_to_crystal_system(int(config.space_group))
            info(T(f"由空间群 #{config.space_group} 推导晶系",
                   f"Crystal system derived from space group #{config.space_group}"))
        except (ValueError, TypeError):
            cs = None
    if cs is None and config.crystal_system:
        try:
            cs = CrystalSystem(config.crystal_system)
        except ValueError:
            cs = None
    if cs is None and os.path.isfile(flag):
        try:
            with open(flag) as f:
                cs = CrystalSystem(f.read().strip())
        except ValueError:
            cs = None
    if cs is None:
        eq_p = os.path.join(work_dir, "equilibrium", "POSCAR")
        if not os.path.isfile(eq_p):
            eq_p = os.path.join(work_dir, "equilibrium", "run", "POSCAR")
        if os.path.isfile(eq_p):
            _eqp = read_poscar(eq_p)
            from .crystal_system import crystal_system_from_structure
            cs = crystal_system_from_structure(_eqp, symprec=max(config.crystal_tol, 1e-3))
            if cs is not None:
                info(T("由 spglib 从结构自动识别晶系", "Crystal system auto-detected from structure by spglib"))
            else:
                cs = identify_crystal_system(_eqp["lattice"], config.crystal_tol)
    if cs is None:
        warn(T("无法识别晶系。", "Could not identify crystal system."))
        return
    # 结果处理阶段: 始终保留一道人工确认/覆盖 (纯铝立方曾因 MD 噪声被判成正交)
    cs = _confirm_or_override_cs(cs)
    name = cs_name(cs)
    if cs not in VC_SUPPORTED:
        warn(T(f"VC 法不支持 {name}。", f"SA-SS method does not support {name}."))
        return

    modes = get_vc_modes(cs)
    amp_labels = get_amplitude_labels(config.n_points)
    extractor = StressExtractor()

    def _find_run_dir(subname):
        for cand in [os.path.join(work_dir, subname),
                     os.path.join(work_dir, subname, "run")]:
            if os.path.isdir(cand) and (
                os.path.isfile(os.path.join(cand, "vasprun.xml")) or
                os.path.isfile(os.path.join(cand, "OUTCAR"))):
                return cand
        return None

    eq_dir = _find_run_dir("equilibrium")
    if eq_dir is None:
        warn(T("未找到 equilibrium 应力数据。", "No equilibrium stress data found."))
        return
    baseline, _, _ = extractor.extract_and_average(eq_dir, skip)
    ok(T(f"参考构型应力已读取 ({eq_dir})", f"Reference stress loaded ({eq_dir})"))

    all_strains, all_stresses = {}, {}
    _bar = ProgressBar(len(modes) * len(amp_labels), desc=T("读取应力数据 (SA-SS)", "Reading stress data (SA-SS)"))
    for mode in modes:
        nm = mode["name"]
        sv, tv = [], []
        for label, mult in amp_labels:
            d = _find_run_dir(f"{nm}_{label}")
            if d is None:
                _bar.update(info=f"{nm}_{label} " + T("(缺)", "(missing)"))
                continue
            try:
                mean, _, _ = extractor.extract_and_average(d, skip)
                sv.append(config.magnitude * mult)
                tv.append(mean)
            except Exception as e:
                warn(T(f"{nm}_{label} 提取失败: {e}", f"{nm}_{label} extraction failed: {e}"))
            _bar.update(info=f"{nm}_{label}")
        if sv:
            all_strains[nm] = np.array(sv)
            all_stresses[nm] = np.array(tv)
    _bar.close()

    if not all_strains:
        warn(T("无有效 SA-SS 数据 (未找到任何 SA-SS 变形目录)。",
               "No valid SA-SS data (no SA-SS deformation directories found)."))
        # 防呆: 若目录其实是 SC-SS 的 deform* 命名, 提示改用 SC-SS 拟合
        _has_deform = any(d.startswith("deform") for d in os.listdir(work_dir)
                          if os.path.isdir(os.path.join(work_dir, d)))
        if _has_deform:
            warn(T("检测到 deform* 目录 —— 这是 SC-SS(单分量)数据! 请改用 SC-SS 拟合(菜单 5)。",
                   "Found deform* directories — this is SC-SS (single-component) data! "
                   "Use the SC-SS fit (menu 5) instead."))
        else:
            info(T(f"当前晶系 {name} 期望的 SA-SS 模式目录: "
                   f"{', '.join(m['name'] for m in modes)} (× 各幅度点); 请确认目录名与晶系一致。",
                   f"Expected SA-SS mode dirs for {name}: "
                   f"{', '.join(m['name'] for m in modes)} (× amplitudes); check dir names vs crystal system."))
        return

    results = VCElasticFitter(cs).fit_all(all_strains, all_stresses, baseline,
                                          fit_order=config.fit_poly_order)
    # 屏幕输出流程/措辞与 SC-SS 完全一致
    print_cij_matrix(results["cij_raw"],
                     T("原始弹性常数矩阵 (GPa)", "Raw elastic constant matrix (GPa)"))
    print_cij_matrix(results["cij_symmetrized"],
                     T("对称性修正后弹性常数矩阵 (Birch, GPa)",
                       "Symmetrized elastic constant matrix (Birch, GPa)"))

    out_dir = prompt_with_default(T("结果输出目录", "Result output directory"), work_dir)
    # 拟合诊断图 + Origin 数据导出
    _plot_vc_fitting_diagnostics(modes, all_strains, all_stresses, baseline,
                                 config.magnitude, out_dir=out_dir,
                                 fit_order=config.fit_poly_order, cs=cs,
                                 cij=results["cij_symmetrized"],
                                 cij_std=results.get("cij_std"))
    # 与 SC-SS 统一的收尾: Wallace 修正 + 完整报告 + elastic.txt + 力学分析提示
    _finalize_elastic_fit(
        results["cij_raw"], results["cij_symmetrized"],
        results.get("fit_info_per_mode") or {}, baseline, cs, config, out_dir,
        results["born_stable"], results["born_warnings"],
        config.temperatures[0], "SA-SS (对称化应力-应变法)",
    )


# ============================================================
# 13) 等温 → 绝热弹性常数转换
# ============================================================

def func_adiabatic(config: MHECConfig) -> None:
    sep(T("等温和绝热弹性常数转换", "Isothermal and adiabatic elastic constants"))
    import numpy as np
    from . import adiabatic as adia

    matrix_path = prompt_with_default(
        T("等温弹性矩阵文件 (6×6)", "Isothermal matrix file (6×6)"), "elastic.txt")
    if not os.path.isfile(matrix_path):
        warn(T(f"文件不存在: {matrix_path}", f"File not found: {matrix_path}"))
        return
    try:
        C_T = np.asarray(np.loadtxt(matrix_path), dtype=float).reshape(6, 6)
    except Exception as e:
        warn(T(f"读取矩阵失败: {e}", f"Failed to read matrix: {e}"))
        return

    default_T = (f"{config.temperatures[0]:.0f}"
                 if getattr(config, "temperatures", None) else "300")
    T_val = float(prompt_with_default(T("温度 T (K)", "Temperature T (K)"), default_T))

    # ---- 热膨胀系数 α: 有则直接读, 无则从 NPT 现算 ----
    alpha_in = prompt_with_default(
        T("线膨胀 α_a,α_b,α_c (×10⁻⁶/K, 逗号分隔; 留空→从 NPT 自动计算)",
          "Linear CTE α_a,α_b,α_c (×10⁻⁶/K, comma; blank→auto from NPT)"), "")

    alpha_voigt = None
    npt_base = None
    natoms = None
    volume_A3 = None
    density = None
    molar_mass = None

    if alpha_in.strip():
        try:
            parts = [float(x) for x in alpha_in.replace(" ", "").split(",") if x]
            if len(parts) == 1:
                parts = parts * 3
            alpha_voigt = np.array([parts[0], parts[1], parts[2], 0, 0, 0]) * 1e-6
        except Exception:
            warn(T("α 解析失败", "Failed to parse α"))
            return
    else:
        default_npt = "phase1_npt" if os.path.isdir("phase1_npt") else "npt_opt"
        npt_base = prompt_with_default(T("NPT 工作目录", "NPT directory"), default_npt)

    # ---- 定容热容 ρc_v: Dulong-Petit (默认) 或 d⟨E⟩/dT ----
    cv_method = prompt_with_default(
        T("c_v 方法: 1=Dulong-Petit  2=d⟨E⟩/dT", "c_v: 1=Dulong-Petit  2=dE/dT"), "1")
    energy_TE = None
    if cv_method.strip() == "2":
        etxt = prompt_with_default(
            T("能量-温度文件 (两列: T[K]  E[eV/cell])", "Energy-T file (T[K]  E[eV/cell])"),
            "energy_vs_T.dat")
        if os.path.isfile(etxt):
            arr = np.loadtxt(etxt)
            energy_TE = (arr[:, 0], arr[:, 1])
        else:
            warn(T("能量文件不存在, 回退 Dulong-Petit",
                   "Energy file missing; fall back to Dulong-Petit"))

    # Dulong-Petit 需要 natoms+volume 或 密度/摩尔质量。
    # NPT 现算路径会自动提供 natoms 与平衡体积; 手动 α 路径需用户给密度/摩尔质量。
    need_dp = (energy_TE is None)
    if need_dp and alpha_voigt is not None:
        nat = prompt_with_default(
            T("原子数 (留空→用密度/摩尔质量)", "N atoms (blank→use density/molar)"), "")
        if nat.strip():
            natoms = int(float(nat))
            volume_A3 = float(prompt_with_default(
                T("平衡体积 (ų)", "Equilibrium volume (Å³)"), "0"))
            if volume_A3 <= 0:
                warn(T("体积无效", "Invalid volume"))
                return
        else:
            density = float(prompt_with_default(
                T("密度 (g/cm³)", "Density (g/cm³)"), "2.70"))
            molar_mass = float(prompt_with_default(
                T("摩尔质量 (g/mol, 每原子平均)", "Molar mass (g/mol, per atom)"), "26.98"))
    elif energy_TE is not None and alpha_voigt is not None:
        # d⟨E⟩/dT 需要体积
        volume_A3 = float(prompt_with_default(
            T("平衡体积 (ų)", "Equilibrium volume (Å³)"), "0")) or None

    try:
        res = adia.run_conversion(
            C_T, T_val,
            alpha_voigt=alpha_voigt,
            npt_base=npt_base,
            config=config,
            natoms=natoms,
            volume_A3=volume_A3,
            density=density,
            molar_mass=molar_mass,
            energy_TE=energy_TE,
        )
    except Exception as e:
        warn(T(f"转换失败: {e}", f"Conversion failed: {e}"))
        return

    print()
    print(res.report)
    out_dir = os.path.dirname(os.path.abspath(matrix_path)) or "."
    mpath, rpath = adia.save_result(res, out_dir)
    ok(T(f"绝热矩阵 → {mpath}", f"Adiabatic matrix → {mpath}"))
    ok(T(f"报告 → {rpath}", f"Report → {rpath}"))


# ============================================================
# 功能分发表
# ============================================================

_DISPATCH = {
    1: func_403,          # SC-SS 单分量应力-应变法 · 生成计算输入 (完整工作流)
    2: func_502,          # SC-SS · 拟合弹性常数
    3: func_vc_generate,  # SA-SS 对称化应力-应变法 · 生成计算输入
    4: func_vc_fit,       # SA-SS · 拟合弹性常数
    5: func_602,          # 力学性质完整分析 (出图 + Origin 数据导出)
    6: func_603,          # AIMD 后处理
    7: func_901,          # MLFF 训练 (定温 grid)
    8: func_902,          # MLFF 训练 (升温)
    9: func_701,          # MLFF 力场精度验证
    10: func_101,         # 晶系识别与晶格参数
    11: func_203,         # 查看变形模式与应变方案
    12: func_801,         # 提取平衡晶格 + 热膨胀系数
    13: func_adiabatic,   # 等温 → 绝热弹性常数转换
}


# ============================================================
# 命令行参数解析
# ============================================================

def parse_cli_args() -> Optional[argparse.Namespace]:
    """解析命令行参数用于非交互式批量运行。"""
    parser = argparse.ArgumentParser(
        prog="mhec",
        description="MHEC: MLFF-accelerated High-temperature Elastic Constants",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--poscar", type=str, help="POSCAR 文件路径")
    parser.add_argument("--primitive", type=str, help="原胞 POSCAR 路径")
    parser.add_argument(
        "--crystal-system", type=str,
        choices=[cs.value for cs in CrystalSystem],
        help="手动指定晶系",
    )
    parser.add_argument("--temp", type=float, nargs="+", help="温度 (K)")
    parser.add_argument("--strain", type=float, help="应变幅度 δ")
    parser.add_argument(
        "--method", choices=["standard", "ulics"], help="应变方案 (高级, 默认 standard)"
    )
    parser.add_argument("--batch", action="store_true", help="批量模式")
    parser.add_argument("--config", type=str, help="配置文件路径 (YAML/JSON)")

    args = parser.parse_args()
    if not args.batch and not args.poscar and not args.config:
        return None
    return args


# ============================================================
# 主入口
# ============================================================

def _func_language(config: MHECConfig) -> None:
    """切换界面语言 (立即生效并永久保存, 单步即可返回)。"""
    sep(T("界面语言", "Language"))
    cur_lang = get_lang()
    print(f"  {T('当前语言', 'Current language')}: {cur_lang}   (zh = 中文, en = English)")
    lang_in = prompt_with_default("zh/en", cur_lang)
    new_lang = set_lang(lang_in)
    config.lang = new_lang
    if save_lang(new_lang):
        ok(T(f"语言已设为 {new_lang} 并永久保存到 {pref_path()}",
             f"Language set to {new_lang} and saved permanently to {pref_path()}"))
    else:
        ok(T(f"语言已设为 {new_lang}", f"Language set to {new_lang}"))


# _func_settings 已移除: 所有参数都在各功能流程中就地询问, 或从用户 INCAR / 当前目录 /
# mhec.yaml 读取; 独立的全局设置菜单冗余且易造成 "改了不生效" 的困惑。


def _func_compare_info(config: MHECConfig) -> None:
    """方法对比说明。"""
    sep("方法对比说明")
    print("""
  MHEC 提供两种弹性常数计算方法:

  MHEC 提供两种弹性常数计算方法 (本质都是应力-应变法, 区别在变形基):

  SC-SS 单分量应力-应变法 (菜单 1/2)
    - 施加单分量有限应变 (ε1, ε4, ...), 读应力, 拟合 σ = C·ε
    - 按晶系对称性自适应取最小独立应变集 (通用)
    - 与 VASPKIT 同源, 适合大多数体系

  SA-SS 对称化应力-应变法 (菜单 3/4)
    - 使用对称化组合应变 (部分保体积: 正交 [δ,-δ,0]、保体积剪切;
      静水/单轴模式不守恒)
    - 去偏模式直接测 C11-C12、C44 等组合, 消除体积/压强污染,
      对剪切常数与立方 C11-C12 信噪比更好

  MLFF 三步流程:
    train  → 从头算训练机器学习力场
    refit  → 用已有数据重新拟合力场
    run    → 使用力场进行长时间 MD 采样
""")


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    """主入口。"""
    args = parse_cli_args()

    config = MHECConfig()
    if args and args.config:
        try:
            config = MHECConfig.from_file(args.config)
            print(T(f" 已加载配置: {args.config}", f" Loaded config: {args.config}"))
        except Exception as e:
            print(T(f" 配置加载失败: {e}，使用默认配置。", f" Failed to load config: {e}; using defaults."))

    # 应用界面语言(config.lang 优先于环境变量默认)
    if getattr(config, "lang", None):
        set_lang(config.lang)

    if args:
        if args.temp:
            config.temperatures = args.temp
        if args.strain:
            config.magnitude = args.strain
        if args.method:
            config.strain_method = args.method
        if args.primitive:
            config.primitive_poscar = args.primitive
        if args.crystal_system:
            config.crystal_system = args.crystal_system

    # 批量模式
    if args and args.batch:
        if not args.poscar:
            print(T(" 错误: 批量模式需要 --poscar 参数。", " Error: batch mode requires --poscar."))
            sys.exit(1)
        poscar = read_poscar(args.poscar)
        if config.crystal_system:
            cs = CrystalSystem(config.crystal_system)
        elif config.primitive_poscar and os.path.isfile(config.primitive_poscar):
            prim = read_poscar(config.primitive_poscar)
            cs = identify_crystal_system(prim["lattice"], config.crystal_tol)
        else:
            cs = identify_crystal_system(poscar["lattice"], config.crystal_tol)
            warn(T("未提供原胞，从超胞自动识别晶系。", "No primitive cell provided; identifying crystal system from the supercell."))
        name = cs_name(cs)
        ok(T(f"晶系: {name} ({N_INDEPENDENT[cs]} 个独立常数)",
             f"Crystal system: {name} ({N_INDEPENDENT[cs]} independent constants)"))
        _setup_work_dir(poscar, cs, config)
        return

    # 交互模式
    while True:
        show_banner()
        show_status(config)
        show_menu()
        try:
            choice_str = input(T(" 请输入功能编号: ", " Enter option: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(T("\n 再见。", "\n Goodbye."))
            break

        if not choice_str:
            continue

        # 处理字符串命令
        if choice_str.lower() == "l":
            try:
                _func_language(config)
            except Exception as e:
                warn(T(f"执行出错: {e}", f"Error: {e}"))
            continue
        if choice_str == "0":
            print(T(" 再见。", " Goodbye."))
            break

        try:
            choice = int(choice_str)
        except ValueError:
            warn(T(f"无效输入: '{choice_str}'", f"Invalid input: '{choice_str}'"))
            continue

        if choice == 0:
            print(T(" 再见。", " Goodbye."))
            break

        func = _DISPATCH.get(choice)
        if func is None:
            warn(T(f"未知功能编号: {choice}", f"Unknown option: {choice}"))
            continue

        try:
            func(config)
        except KeyboardInterrupt:
            print("\n  操作已取消。")
        except Exception as e:
            warn(f"执行出错: {e}")

        sep()


if __name__ == "__main__":
    main()
