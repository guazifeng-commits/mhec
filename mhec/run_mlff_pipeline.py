#!/usr/bin/env python3
"""
MHEC MLFF 高精度训练 Pipeline

一键生成覆盖弹性常数计算所需场景的 MLFF 训练目录：
  - 多温度 NPT（晶格热膨胀，高温弹性需正确平衡胞）
  - 每一训练温度 NVT 应变 MD（避免只在端点温度采样导致中间温度外推）
  - auto (默认)：自动识别晶系，按对称性裁剪应变集
  - general：完整 Voigt 基（xx/yy/zz 单轴 ±，yz/xz/xy 剪切 ±，体积 ±）适合任意晶系

用法:
  1. 准备: 当前目录放 POSCAR, POTCAR, KPOINTS, run.sh
  2. 生成: mhec → 菜单 10 (或 python -m mhec.run_mlff_pipeline)
  3. 提交: sbatch mlff_train/submit_train.sh
  4. 自动串联 + 自动 refit（由依赖任务触发）
  5. 得到: mlff_train/refit_final/ML_FF

设计原则:
  - DFT 参考数据高精度: PREC=Accurate, EDIFF=1E-6; LREAL=Auto + NCORE=4 (默认省内存, 大体系/高熵合金友好; 追求极致力/应力精度可改 LREAL=.FALSE.)
  - 高温弹性：温度网格 + 小应变采样，减轻力/应力噪声对曲率的影响
  - ML_IWEIGHT=3（VASP Wiki 推荐归一化）
  - 最少的用户输入: 只需 POSCAR + POTCAR + KPOINTS + run.sh
"""

import os
import shutil
import argparse
import subprocess
import numpy as np
from typing import List, Dict, Optional, Tuple

from .crystal_system import CrystalSystem, identify_crystal_system, CRYSTAL_SYSTEM_NAMES, cs_name
from .i18n import T

# 默认面向「宽温区高温弹性」：与常见 DFT-MD 训练尺度一致，可用 CLI 覆盖
DEFAULT_TRAIN_TEMPERATURES_K = [300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0]
# 含小应变以贴近弹性线性区；可仅传 0.01 0.02 减少任务数
DEFAULT_STRAIN_MAGNITUDES = [0.005, 0.01, 0.02]

# ============================================================
# 按晶系对称性裁剪的 Phase B 训练应变集
# ============================================================
# 每个晶系只需覆盖其独立 Voigt 分量即可完整学习应变-应力响应，
# 无需全部 14 个应变方向，大幅减少高对称晶系的训练目录数。
#
# 格式: { CrystalSystem: {"uniax": [(voigt_idx, name, [signs])],
#                        "shear": [(voigt_idx, name, [signs])],
#                        "vol": [signs]} }
#
# 约定: 单轴应变使用双符号 (+/-) 以学习非谐效应；
#       剪切和体积只需正号（响应通常对称或正应变训练已覆盖）。

def _get_training_strain_set(crystal_system: CrystalSystem) -> Dict:
    """
    返回该晶系在 MLFF Phase B 训练中应采样的应变方向集合。

    Returns
    -------
    {"uniax": [(voigt_idx, name, [signs])], "shear": [...], "vol": [signs]}

    规则:
    - 立方:      只需要 xx 单轴 + yz 剪切 + 体积膨胀 = 4 个
    - 六方:      需要 xx, zz + yz, xy + 体积 = 7 个
    - 三方(-3m): 需要 xx, zz + yz + 体积 = 7 个
    - 三方(-3):  需要 xx, zz + yz, xz + 体积 = 9 个
    - 四方(4/mmm):需要 xx, zz + yz, xy + 体积 = 7 个
    - 四方(4/m): 需要 xx, zz + yz, xy(双号) + 体积 = 8 个
    - 正交:      需要全部三单轴 + 三剪切 + 体积 = 10 个
    - 单斜:      需要全部分量 + 体积(双号) = 14 个（全 Voigt）
    - 三斜:      14 个（全 Voigt，无缩减空间）
    """
    PLUS_MINUS = [1.0, -1.0]
    PLUS_ONLY = [1.0]

    sets = {
        CrystalSystem.CUBIC: {
            "uniax": [(0, "xx", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_ONLY)],
            "vol": PLUS_ONLY,
        },
        CrystalSystem.HEXAGONAL: {
            "uniax": [(0, "xx", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_ONLY), (5, "xy", PLUS_ONLY)],
            "vol": PLUS_ONLY,
        },
        CrystalSystem.TRIGONAL_3M: {
            "uniax": [(0, "xx", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_MINUS)],  # C14 耦合需要双号
            "vol": PLUS_ONLY,
        },
        CrystalSystem.TRIGONAL_3: {
            "uniax": [(0, "xx", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_MINUS), (4, "xz", PLUS_MINUS)],
            "vol": PLUS_ONLY,
        },
        CrystalSystem.TETRAGONAL_4MMM: {
            "uniax": [(0, "xx", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_ONLY), (5, "xy", PLUS_ONLY)],
            "vol": PLUS_ONLY,
        },
        CrystalSystem.TETRAGONAL_4M: {
            "uniax": [(0, "xx", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_ONLY), (5, "xy", PLUS_MINUS)],  # C16 耦合需要 xy 双号
            "vol": PLUS_ONLY,
        },
        CrystalSystem.ORTHORHOMBIC: {
            "uniax": [(0, "xx", PLUS_MINUS), (1, "yy", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_ONLY), (4, "xz", PLUS_ONLY), (5, "xy", PLUS_ONLY)],
            "vol": PLUS_ONLY,
        },
        CrystalSystem.MONOCLINIC: {
            "uniax": [(0, "xx", PLUS_MINUS), (1, "yy", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_MINUS), (4, "xz", PLUS_MINUS), (5, "xy", PLUS_MINUS)],
            "vol": PLUS_MINUS,
        },
        CrystalSystem.TRICLINIC: {
            "uniax": [(0, "xx", PLUS_MINUS), (1, "yy", PLUS_MINUS), (2, "zz", PLUS_MINUS)],
            "shear": [(3, "yz", PLUS_MINUS), (4, "xz", PLUS_MINUS), (5, "xy", PLUS_MINUS)],
            "vol": PLUS_MINUS,
        },
    }
    return sets[crystal_system]


def _count_training_strains(crystal_system: CrystalSystem) -> int:
    """计算给定晶系每 (温度, 幅度) 的训练应变数。"""
    s = _get_training_strain_set(crystal_system)
    count = sum(len(signs) for _, _, signs in s["uniax"])
    count += sum(len(signs) for _, _, signs in s["shear"])
    count += len(s["vol"])
    return count


def _lattice_from_voigt(lattice0: np.ndarray, voigt6: np.ndarray, voigt_strain_to_tensor) -> np.ndarray:
    """F = I + ε，新格矢 lattice = lattice0 @ F.T（与现有 Phase B 一致）。"""
    eps = voigt_strain_to_tensor(voigt6)
    F = np.eye(3) + eps
    return lattice0 @ F.T


def generate_mlff_training(
    work_dir: str = "mlff_train",
    temperatures: List[float] = None,
    strain_temperatures: List[float] = None,
    strain_magnitudes: List[float] = None,
    nsw_train: int = 5000,
    potim: float = 1.0,
    encut: Optional[float] = 500.0,
    poscar_path: str = "POSCAR",
    potcar_path: str = "POTCAR",
    kpoints_path: str = "KPOINTS",
    run_sh_path: str = "run.sh",
    elastic_training_mode: str = "auto",
    concat_schedule: str = "high_to_low",
    b_lattice_source: str = "npt_equilibrated",
    b_skip_frac: float = 0.5,
    incar_path: str = "INCAR",
) -> Dict:
    """
    生成 MLFF 训练所需的全部计算目录。

    训练策略:
    Phase A: 多温度 NPT train（学习热膨胀 + 基本力/应力）
    Phase B: 多应变 NVT train（学习应变-应力响应；auto 按晶系对称性自动裁剪）
    Phase C: refit 合并所有 ML_AB 得到最终 ML_FF

    Parameters
    ----------
    work_dir : 训练工作目录
    temperatures : 训练温度列表，None 则自动设置
    strain_temperatures : 进行应变训练的温度列表（默认等于 temperatures）。可用稀疏温度点减少任务量
    strain_magnitudes : 应变幅度列表
    nsw_train : 每个训练任务的 MD 步数
    potim : 时间步长 (fs)
    encut : 截断能，None 则 500
    elastic_training_mode : "auto" 按晶系对称性自动裁剪；"general" 完整 Voigt 应变基
    concat_schedule : 串联训练顺序（写入 training_dirs.txt 的顺序）
        - "high_to_low": 先高温后低温（推荐宽温区 < Tmax：先覆盖大构型空间，再精修低温曲率）
        - "as_generated": 按生成顺序
    """
    from .vasp_io import read_poscar, write_poscar, write_incar
    from .strain import voigt_strain_to_tensor
    from .lattice_optimizer import LatticeOptimizer

    mode = (elastic_training_mode or "auto").strip().lower()
    if mode not in ("auto", "general"):
        print(f"Error: elastic_training_mode must be 'auto' or 'general', got {elastic_training_mode!r}")
        return {}

    # 读取 POSCAR 并识别晶系
    poscar = read_poscar(poscar_path)
    lattice0 = poscar["lattice"]
    crystal_system = identify_crystal_system(lattice0)
    cs_label = cs_name(crystal_system)

    # 解析训练模式
    if mode == "auto":
        strain_set = _get_training_strain_set(crystal_system)
        n_strains = _count_training_strains(crystal_system)
        mode_label = f"auto → {cs_label}({n_strains} 应变/温度/幅度)"
    else:  # general
        # 全 Voigt 基：所有 6 个分量 ± + 体积 ±
        strain_set = _get_training_strain_set(CrystalSystem.TRICLINIC)
        n_strains = _count_training_strains(CrystalSystem.TRICLINIC)
        mode_label = f"general 全 Voigt({n_strains} 应变/温度/幅度)"

    print(T(f"  → 晶系: {cs_label}  |  训练模式: {mode_label}",
            f"  → Crystal system: {cs_label}  |  training mode: {mode_label}"))

    # 默认参数
    if temperatures is None:
        temperatures = list(DEFAULT_TRAIN_TEMPERATURES_K)
    if strain_magnitudes is None:
        strain_magnitudes = list(DEFAULT_STRAIN_MAGNITUDES)
    if strain_temperatures is None:
        strain_temperatures = list(temperatures)

    schedule = (concat_schedule or "high_to_low").strip().lower()
    if schedule not in ("high_to_low", "as_generated"):
        print(f"Error: concat_schedule must be 'high_to_low' or 'as_generated', got {concat_schedule!r}")
        return {}

    # 检查必需文件
    for f, name in [(poscar_path, "POSCAR"), (potcar_path, "POTCAR"),
                     (kpoints_path, "KPOINTS"), (run_sh_path, "run.sh")]:
        if not os.path.isfile(f):
            print(f"Error: {name} not found: {f}")
            return {}

    os.makedirs(work_dir, exist_ok=True)

    b_src = (b_lattice_source or "initial").strip().lower()
    if b_src not in ("initial", "npt_equilibrated"):
        print(f"Error: b_lattice_source must be 'initial' or 'npt_equilibrated', got {b_lattice_source!r}")
        return {}
    if not (0.0 <= float(b_skip_frac) < 1.0):
        print(f"Error: b_skip_frac must be in [0,1), got {b_skip_frac!r}")
        return {}
    npt_optimizer = LatticeOptimizer(poscar=poscar, temperatures=[float(t) for t in (temperatures or [])],
                                     crystal_system=crystal_system)

    # 高精度 DFT + MLFF train 的 INCAR 基础参数
    base_params = {
        "PREC": "Accurate",
        "EDIFF": "1E-6",
        "EDIFFG": "-1E-3",
        "ADDGRID": ".TRUE.",
        "LREAL": "Auto",
        "NCORE": "4",
        "ALGO": "Normal",
        "ISMEAR": "0",
        "SIGMA": "0.05",
        "NELM": "200",
        "ISYM": "0",
        "IBRION": "0",
        "POTIM": str(potim),
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
        # MLFF train 参数
        "ML_LMLFF": ".TRUE.",
        "ML_MODE": "train",
        "ML_ISTART": "0",
        "ML_IALGO_LINREG": "4",
        "ML_CX": "0.1",
        "ML_MB": "3000",
        "ML_MCONF_NEW": "1",
        "ML_LBASIS_DISCARD": ".FALSE.",
        "ML_IWEIGHT": "3",
        # 弹性常数对应力精度要求高: 保能量/力基准, 加大应力权重
        "ML_WTOTEN": "1.0",
        "ML_WTIFOR": "1.0",
        "ML_WTSIF": "4.0",
    }
    base_params["ENCUT"] = str(int(encut if encut is not None else 500))

    # 继承用户 INCAR 中的泛函/色散/Hubbard 等"物理"设置 (如 GGA=PS, IVDW, VDW_*, LDAU*)。
    # 这些决定 DFT 参考数据的物理, 必须带入训练 INCAR, 否则 MLFF 学到的是默认 PBE 的力场!
    # 管线自身管理的 MD/精度/ML 控制参数不被覆盖。
    # 只强制"会破坏 MD/训练工作流"的控制参数; 其余(ENCUT/PREC/EDIFF/ALGO/LREAL/
    # NELM/ISMEAR/SIGMA/GGA/IVDW/VDW_*/LDAU*/ISPIN/MAGMOM 等电子结构参数)一律从
    # 用户 INCAR 继承, 由用户按材料类型自行设定; base_params 仅提供缺省值。
    _MANAGED = {
        "IBRION", "MDALGO", "ISIF", "NSW", "TEBEG", "TEEND", "POTIM",
        "SMASS", "PMASS", "PSTRESS", "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L",
        "ISYM", "LWAVE", "LCHARG",
    }
    if incar_path and os.path.isfile(incar_path):
        try:
            from .vasp_io import read_incar
            user_inc = read_incar(incar_path)
            carried = {}
            for k, v in user_inc.items():
                ku = k.upper()
                if ku in _MANAGED or ku.startswith("ML_"):
                    continue
                carried[k] = v
            if carried:
                base_params.update(carried)
                print(T(f"  → 继承用户 INCAR 物理设置 ({incar_path}): {', '.join(sorted(carried))}",
                        f"  → Inherited physical settings from user INCAR ({incar_path}): {', '.join(sorted(carried))}"))
        except Exception as e:
            print(T(f"  ! 读取用户 INCAR ({incar_path}) 失败, 未继承额外参数: {e}",
                    f"  ! Failed to read user INCAR ({incar_path}); no extra parameters inherited: {e}"))

    all_dirs = []
    dir_info = []  # (dir_path, description, phase)

    def _make_dir(subdir, desc, phase, lattice, ensemble, temp):
        """创建一个训练子目录。"""
        d = os.path.join(work_dir, subdir)
        os.makedirs(d, exist_ok=True)

        # POSCAR
        p = dict(poscar)
        p["lattice"] = lattice
        write_poscar(os.path.join(d, "POSCAR"), p)

        # INCAR
        params = dict(base_params)
        params["NSW"] = str(nsw_train)
        params["TEBEG"] = str(int(temp))
        params["TEEND"] = str(int(temp))
        if ensemble == "npt":
            params["ISIF"] = "3"
            params["MDALGO"] = "3"
            # LANGEVIN_GAMMA 需按元素种类给 N 个值 (每种元素一个)
            _nsp = len(poscar.get("counts") or [1]) or 1
            params["LANGEVIN_GAMMA"] = " ".join(["10.0"] * _nsp)
            params["LANGEVIN_GAMMA_L"] = "1.0"
            params["PMASS"] = "1000"
            params["PSTRESS"] = "0"
        else:
            params["ISIF"] = "2"
            params["MDALGO"] = "2"
            params["SMASS"] = "2"
        write_incar(os.path.join(d, "INCAR"), params)

        # POTCAR, KPOINTS, run.sh
        for src in [potcar_path, kpoints_path, run_sh_path]:
            dst = os.path.join(d, os.path.basename(src))
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)

        all_dirs.append(d)
        dir_info.append((d, desc, phase))

    # ================================================================
    # Phase A: 多温度 NPT train（学习热膨胀、晶格响应）
    # ================================================================
    print("Phase A: NPT training (thermal expansion)")
    for temp in temperatures:
        _make_dir(f"A_npt_{int(temp)}K",
                  f"NPT {int(temp)}K equilibrium",
                  "A", lattice0, "npt", temp)

    # ================================================================
    # Phase B: 每一训练温度 NVT + 应变（高温弹性：避免中间温度未采样）
    # Voigt: 1=xx,2=yy,3=zz,4=yz,5=xz,6=xy（与 mhec.strain 一致）
    # ================================================================
    temps_for_strain = list(strain_temperatures)
    print(
        f"Phase B: Strained NVT at {len(temps_for_strain)} temperatures (mode={mode}, schedule={schedule})"
    )

    def _strain_tag(m: float, sign_char: str) -> str:
        """目录名用：正号 e0.010，负号 em0.010（避免歧义）。"""
        if sign_char == "+":
            return f"e{m:.3f}"
        return f"em{m:.3f}"

    # 串联建议：先高温覆盖，再低温精修（影响 training_dirs.txt 顺序，进而影响 concat_train 顺序）
    if schedule == "high_to_low":
        train_temps = sorted(temps_for_strain, reverse=True)
    else:
        train_temps = list(temps_for_strain)

    for temp in train_temps:
        # Phase B 参考晶格：优先从 Phase A NPT 平衡结构提取；未完成时回退到初始 POSCAR
        lattice_ref = lattice0
        if b_src == "npt_equilibrated":
            a_dir = os.path.join(work_dir, f"A_npt_{int(temp)}K")
            if os.path.isdir(a_dir):
                try:
                    lattice_ref, stats = npt_optimizer.extract_equilibrium_lattice(a_dir, skip_frac=float(b_skip_frac))
                    src = stats.get("source", "unknown")
                    print(f"  +-> Using NPT equilibrated lattice for {int(temp)}K ({src})")
                except Exception as e:
                    print(f"  !-> NPT lattice extraction failed for {int(temp)}K: {e}")
                    print(f"      Falling back to initial POSCAR lattice for {int(temp)}K")
            else:
                print(f"  !-> Phase A dir not found for {int(temp)}K: {a_dir}")
                print(f"      Falling back to initial POSCAR lattice for {int(temp)}K")

        for mag in strain_magnitudes:
            # 统一使用 strain_set 生成应变目录
            # 单轴应变
            for idx, name, signs in strain_set["uniax"]:
                for sgn in signs:
                    v = np.zeros(6)
                    v[idx] = sgn * mag
                    lat = _lattice_from_voigt(lattice_ref, v, voigt_strain_to_tensor)
                    tpart = _strain_tag(mag, "+" if sgn > 0 else "-")
                    _make_dir(
                        f"B_uniax_{name}_{int(temp)}K_{tpart}",
                        f"Uniaxial {name} {sgn * mag:+.4f} at {int(temp)}K",
                        "B", lat, "nvt", temp,
                    )
            # 剪切应变
            for idx, name, signs in strain_set["shear"]:
                for sgn in signs:
                    v = np.zeros(6)
                    v[idx] = sgn * mag
                    lat = _lattice_from_voigt(lattice_ref, v, voigt_strain_to_tensor)
                    tpart = _strain_tag(mag, "+" if sgn > 0 else "-")
                    _make_dir(
                        f"B_shear_{name}_{int(temp)}K_{tpart}",
                        f"Shear {name} {sgn * mag:+.4f} at {int(temp)}K",
                        "B", lat, "nvt", temp,
                    )
            # 体积应变
            for sgn in strain_set["vol"]:
                factor = (1.0 + sgn * mag) ** (1.0 / 3.0)
                if factor <= 0:
                    continue
                lat_vol = lattice_ref * factor
                tpart = _strain_tag(mag, "+" if sgn > 0 else "-")
                _make_dir(
                    f"B_vol_{int(temp)}K_{tpart}",
                    f"Volume {sgn * mag:+.4f} (hydrostatic) at {int(temp)}K",
                    "B", lat_vol, "nvt", temp,
                )

        _make_dir(f"B_eq_{int(temp)}K",
                  f"Equilibrium NVT at {int(temp)}K",
                  "B", lattice_ref, "nvt", temp)

    # ================================================================
    # 生成提交脚本
    # ================================================================
    _write_submit_script(work_dir, all_dirs)

    # 保存目录信息
    info_path = os.path.join(work_dir, "training_dirs.txt")
    with open(info_path, 'w') as f:
        f.write(f"# MHEC MLFF Training Pipeline\n")
        f.write(f"# {len(all_dirs)} training directories\n")
        f.write(f"# Temperatures: {temperatures}\n")
        f.write(f"# Strain temperatures: {temps_for_strain}\n")
        f.write(f"# Strain magnitudes: {strain_magnitudes}\n")
        f.write(f"# NSW per dir: {nsw_train}\n")
        f.write(f"# Elastic training mode: {mode}\n\n")
        for d, desc, phase in dir_info:
            f.write(f"{phase}\t{os.path.basename(d)}\t{desc}\n")

    print(f"\nGenerated {len(all_dirs)} training directories in {work_dir}/")
    print(f"  Phase A (NPT): {sum(1 for _,_,p in dir_info if p=='A')} dirs")
    print(f"  Phase B (strained NVT): {sum(1 for _,_,p in dir_info if p=='B')} dirs")
    print(f"\nNext steps:")
    print(f"  1. Submit: sbatch {work_dir}/submit_train.sh")
    print(f"  2. Wait for all jobs to finish")
    print(f"  3. Concat+refit: " + T("由 submit_train.sh 末尾依赖任务自动执行（或手动 --refit --auto-run）",
                                      "run automatically by the dependency job at the end of submit_train.sh (or manually --refit --auto-run)"))

    return {"work_dir": work_dir, "n_dirs": len(all_dirs), "dirs": all_dirs}


def _write_submit_script(work_dir: str, all_dirs: List[str]):
    """生成批量提交脚本，含 Phase A → lattice update → Phase B → concat+refit 依赖链。"""
    phase_a_dirs = [d for d in all_dirs if os.path.basename(d).startswith("A_")]
    phase_b_dirs = [d for d in all_dirs if os.path.basename(d).startswith("B_")]

    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --partition=8358P",
        "#SBATCH --job-name=mhec_mlff_train",
        "#SBATCH --output=submit_train_%j.out",
        "",
        "# MHEC MLFF Training Pipeline",
        "# Phase A (NPT) -> Lattice Update -> Phase B (NVT strain) -> Concat+Refit",
        "# 用法: sbatch submit_train.sh  或  bash submit_train.sh",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        'fi',
        "",
    ]

    # ================================================================
    # Phase A: NPT training (all temperatures in parallel)
    # ================================================================
    lines.extend([
        "echo '>>> Phase A: NPT training'",
        "A_JOBS=()",
        "",
    ])

    for d in phase_a_dirs:
        rel = os.path.relpath(d, work_dir)
        lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
        lines.append('JOB_ID=$(sbatch --parsable run.sh)')
        lines.append('A_JOBS+=($JOB_ID)')
        lines.append(f'echo "  {rel}: $JOB_ID"')
        lines.append('popd > /dev/null')

    # ================================================================
    # Phase 1.5: Update Phase B POSCARs from Phase A NPT results
    # ================================================================
    lines.extend([
        "",
        'A_DEPS=$(IFS=:; echo "${A_JOBS[*]}")',
        "",
        "echo '>>> Lattice update: extracting NPT equilibrium lattices for Phase B'",
        'UPDATE_JOB=$(sbatch --parsable --dependency=afterok:$A_DEPS '
        '--wrap "python3 -m mhec.run_mlff_pipeline --update-phase-b $WORK_ROOT" '
        '-N1 -n1 --partition=8358P --job-name=mhec_phase_b_update '
        '--output=$WORK_ROOT/update_phase_b_%j.out)',
        'echo "  Phase B POSCAR update job: $UPDATE_JOB"',
        "",
    ])

    # ================================================================
    # Phase B: NVT strain training (depends on POSCAR update)
    # ================================================================
    lines.extend([
        "echo '>>> Phase B: NVT strain training'",
        "B_JOBS=()",
        "",
    ])

    for d in phase_b_dirs:
        rel = os.path.relpath(d, work_dir)
        lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
        lines.append('JOB_ID=$(sbatch --parsable --dependency=afterok:$UPDATE_JOB run.sh)')
        lines.append('B_JOBS+=($JOB_ID)')
        lines.append(f'echo "  {rel}: $JOB_ID"')
        lines.append('popd > /dev/null')

    # ================================================================
    # Phase C: Concat + Refit (depends on Phase B)
    # ================================================================
    lines.extend([
        "",
        'echo ">>> Phase C: Concat + Refit follow-up"',
        'echo "  (will be submitted after Phase B completes)"',
        'B_DEPS=$(IFS=:; echo "${B_JOBS[*]}")',
        'REFIT_JOB=$(sbatch --parsable --dependency=afterok:$B_DEPS '
        '--wrap "python3 -m mhec.run_mlff_pipeline --refit $WORK_ROOT --auto-run" '
        '-N1 -n1 --partition=8358P --job-name=mhec_mlff_refit '
        '--output=$WORK_ROOT/refit_followup_%j.out)',
        'echo "  Auto concat+refit job: $REFIT_JOB"',
        "",
    ])

    lines.extend([
        'echo ""',
        'echo "=== Pipeline summary ==="',
        f'echo "  Phase A (NPT):      ${{#A_JOBS[@]}} jobs"',
        f'echo "  Phase B (NVT strain): ${{#B_JOBS[@]}} jobs"',
        'echo "  Phase C (concat+refit): $REFIT_JOB"',
        'echo ""',
        'echo "Submission order: Phase A -> Update POSCARs -> Phase B -> Concat+Refit"',
    ])

    path = os.path.join(work_dir, "submit_train.sh")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  +-> {path}")


def _update_phase_b_poscars(work_dir: str = "mlff_train", b_skip_frac: float = 0.5) -> None:
    """
    从 Phase A NPT 结果提取各温度平衡晶格，更新所有 Phase B POSCAR。

    解析 Phase B 目录名确定应变参数，用 NPT 平衡晶格重新计算变形晶格。
    用于 submit 脚本中 Phase A 完成后、Phase B 运行前的中间步骤。
    """
    import re
    from .vasp_io import read_poscar, write_poscar
    from .lattice_optimizer import LatticeOptimizer
    from .strain import voigt_strain_to_tensor

    if not os.path.isdir(work_dir):
        print(f"Error: {work_dir} not found")
        return

    # 扫描 Phase A 温度目录
    a_dirs = {}  # {temp_K: dir_path}
    for d in sorted(os.listdir(work_dir)):
        m = re.match(r"A_npt_(\d+)K$", d)
        if m and os.path.isdir(os.path.join(work_dir, d)):
            a_dirs[int(m.group(1))] = os.path.join(work_dir, d)

    if not a_dirs:
        print("Error: no A_npt_*K directories found")
        return

    # 扫描 Phase B 目录（按温度分组）
    b_dirs_by_temp = {}  # {temp_K: [(dir_name, strain_type, strain_sub, strain_tag)]}
    _b_uniax_shear_re = re.compile(r"B_(uniax|shear)_(\w+)_(\d+)K_(e[m\d\.]+)$")
    _b_vol_re = re.compile(r"B_vol_(\d+)K_(e[m\d\.]+)$")
    _b_eq_re = re.compile(r"B_eq_(\d+)K$")
    for d in sorted(os.listdir(work_dir)):
        dpath = os.path.join(work_dir, d)
        if not os.path.isdir(dpath):
            continue
        # 单轴 / 剪切
        m = _b_uniax_shear_re.match(d)
        if m:
            temp_k = int(m.group(3))
            b_dirs_by_temp.setdefault(temp_k, []).append((d, m.group(1), m.group(2), m.group(4)))
            continue
        # 体积
        m = _b_vol_re.match(d)
        if m:
            temp_k = int(m.group(1))
            b_dirs_by_temp.setdefault(temp_k, []).append((d, "vol", None, m.group(2)))
            continue
        # 平衡
        m = _b_eq_re.match(d)
        if m:
            temp_k = int(m.group(1))
            b_dirs_by_temp.setdefault(temp_k, []).append((d, "eq", None, None))
            continue

    if not b_dirs_by_temp:
        print("Error: no B_* directories found")
        return

    # 读取模板 POSCAR（取第一个 Phase A 目录中的 CONTCAR 或初始 POSCAR）
    template_poscar = None
    for temp_dir in a_dirs.values():
        for fname in ["CONTCAR", "POSCAR"]:
            for d in [temp_dir, os.path.join(temp_dir, "run")]:
                p = os.path.join(d, fname)
                if os.path.isfile(p):
                    template_poscar = read_poscar(p)
                    break
            if template_poscar:
                break
        if template_poscar:
            break

    if template_poscar is None:
        print("Error: no POSCAR found in Phase A dirs for template")
        return

    # 初始化 LatticeOptimizer 用于提取平衡晶格
    # 晶系从模板结构识别一次 (而非从每温度的噪声平均晶格重复识别)
    _tmpl_cs = identify_crystal_system(template_poscar["lattice"])
    optimizer = LatticeOptimizer(
        poscar=template_poscar,
        temperatures=sorted([float(t) for t in a_dirs.keys()]),
        crystal_system=_tmpl_cs,
    )

    updated = 0
    for temp in sorted(a_dirs.keys()):
        npt_dir = a_dirs[temp]
        try:
            avg_lattice, stats = optimizer.extract_equilibrium_lattice(
                npt_dir, skip_frac=b_skip_frac)
            src = stats.get("source", "unknown")
            print(f"  {temp}K: a={stats['a']:.6f} ({src})")
        except Exception as e:
            print(f"  {temp}K: NPT lattice extraction failed: {e}, skipping")
            continue

        lattice_ref = avg_lattice

        if temp not in b_dirs_by_temp:
            print(f"  {temp}K: no Phase B directories, skip")
            continue

        for bname, strain_type, strain_sub, strain_tag in b_dirs_by_temp[temp]:
            bpath = os.path.join(work_dir, bname)
            poscar_path = os.path.join(bpath, "POSCAR")
            try:
                poscar = read_poscar(poscar_path)
            except Exception:
                print(f"    Skip {bname}: cannot read POSCAR")
                continue

            # 计算变形晶格
            if strain_type == "eq":
                new_lattice = lattice_ref
            elif strain_type == "vol":
                sign = -1.0 if strain_tag.startswith("em") else 1.0
                mag = float(strain_tag[1:] if sign == 1.0 else strain_tag[2:])
                factor = (1.0 + sign * mag) ** (1.0 / 3.0)
                if factor <= 0:
                    continue
                new_lattice = lattice_ref * factor
            elif strain_type == "uniax":
                axis_map = {"xx": 0, "yy": 1, "zz": 2}
                if strain_sub not in axis_map:
                    continue
                sign = -1.0 if strain_tag.startswith("em") else 1.0
                mag = float(strain_tag[1:] if sign == 1.0 else strain_tag[2:])
                v = np.zeros(6)
                v[axis_map[strain_sub]] = sign * mag
                new_lattice = _lattice_from_voigt(lattice_ref, v, voigt_strain_to_tensor)
            elif strain_type == "shear":
                comp_map = {"yz": 3, "xz": 4, "xy": 5}
                if strain_sub not in comp_map:
                    continue
                sign = -1.0 if strain_tag.startswith("em") else 1.0
                mag = float(strain_tag[1:] if sign == 1.0 else strain_tag[2:])
                v = np.zeros(6)
                v[comp_map[strain_sub]] = sign * mag
                new_lattice = _lattice_from_voigt(lattice_ref, v, voigt_strain_to_tensor)
            else:
                continue

            poscar["lattice"] = new_lattice
            write_poscar(poscar_path, poscar)
            updated += 1

    print(f"\nUpdated {updated} Phase B POSCAR files.")


def refit_mlff(work_dir: str = "mlff_train", auto_run: bool = False):
    """
    基于 VASP 官方推荐执行串联训练，再准备最终 refit 目录。

    官方推荐（ML_MODE=train）:
    - 将上一次训练产物 ML_ABN 复制为下一次计算输入 ML_AB
    - 反复继续 train，实现训练集安全融合（避免手工拼接 ML_AB 头信息错误）
    """
    from .vasp_io import write_incar

    if not os.path.isdir(work_dir):
        print(f"Error: {work_dir} not found")
        return

    # 优先读取我们自己写出的目录顺序，保证 Phase A/B 串联顺序稳定
    ordered_train_dirs: List[str] = []
    info_path = os.path.join(work_dir, "training_dirs.txt")
    if os.path.isfile(info_path):
        with open(info_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                dname = parts[1]
                dpath = os.path.join(work_dir, dname)
                if os.path.isdir(dpath):
                    ordered_train_dirs.append(dpath)

    # 兜底：没找到 training_dirs.txt 时，按目录名排序
    if not ordered_train_dirs:
        for name in sorted(os.listdir(work_dir)):
            d = os.path.join(work_dir, name)
            if os.path.isdir(d) and name.startswith(("A_", "B_")):
                ordered_train_dirs.append(d)

    if not ordered_train_dirs:
        print("Error: no training directories found (A_*/B_*).")
        return

    # 收集各目录训练产物（优先 ML_ABN）
    dataset_entries = []
    for d in ordered_train_dirs:
        ml_abn = os.path.join(d, "ML_ABN")
        ml_ab = os.path.join(d, "ML_AB")
        if os.path.isfile(ml_abn):
            dataset_entries.append((d, ml_abn, "ML_ABN"))
        elif os.path.isfile(ml_ab):
            dataset_entries.append((d, ml_ab, "ML_AB"))

    if not dataset_entries:
        print("Error: no ML_ABN/ML_AB found. Training may not have completed.")
        return

    print(f"Found {len(dataset_entries)} completed training datasets for concatenation.")

    # ================================================================
    # 1) 创建串联训练目录（concat_train），每一步都用上一步 ML_ABN -> 本步 ML_AB
    # ================================================================
    concat_root = os.path.join(work_dir, "concat_train")
    os.makedirs(concat_root, exist_ok=True)

    stage_dirs: List[str] = []
    stage_sources: List[str] = []
    for i, (src_dir, _, _) in enumerate(dataset_entries):
        stage_name = f"stage_{i+1:03d}_{os.path.basename(src_dir)}"
        stage_dir = os.path.join(concat_root, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        stage_dirs.append(stage_dir)
        stage_sources.append(src_dir)

    # 初始化第 1 步 ML_AB：直接使用第 1 个数据集作为起点
    _, seed_file, _ = dataset_entries[0]
    shutil.copy2(seed_file, os.path.join(stage_dirs[0], "ML_AB"))

    # 每个 stage 复制其对应源目录的计算输入，并写 train 模式 INCAR
    for i, stage_dir in enumerate(stage_dirs):
        src_dir = stage_sources[i]

        for fname in ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]:
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(stage_dir, fname))

        # 尽量继承源 INCAR 的动力学设置；关键 ML 标签强制覆盖为 train continuation
        src_incar = os.path.join(src_dir, "INCAR")
        incar_params = {
            "PREC": "Accurate",
            "EDIFF": "1E-6",
            "EDIFFG": "-1E-3",
            "ADDGRID": ".TRUE.",
            "LREAL": "Auto",
            "NCORE": "4",
            "ALGO": "Normal",
            "ISMEAR": "0",
            "SIGMA": "0.05",
            "NELM": "200",
            "ISYM": "0",
            "IBRION": "0",
            "NSW": "2000",
            "POTIM": "1",
            "ISIF": "2",
            "MDALGO": "2",
            "SMASS": "2",
            "TEBEG": "300",
            "TEEND": "300",
            "LWAVE": ".FALSE.",
            "LCHARG": ".FALSE.",
            "ML_LMLFF": ".TRUE.",
            "ML_MODE": "train",
            "ML_ISTART": "1",
            "ML_IALGO_LINREG": "4",
            "ENCUT": "500",
        }
        if os.path.isfile(src_incar):
            with open(src_incar, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    incar_params[k.strip()] = v.strip()

        # 强制官方推荐的 continuation train（防止源 INCAR 覆盖关键 ML 参数）
        incar_params["ML_LMLFF"] = ".TRUE."
        incar_params["ML_MODE"] = "train"
        incar_params["ML_ISTART"] = "1"
        incar_params["ML_IALGO_LINREG"] = "4"
        incar_params["ML_IWEIGHT"] = "3"
        incar_params["ML_WTOTEN"] = "1.0"
        incar_params["ML_WTIFOR"] = "1.0"
        incar_params["ML_WTSIF"] = "4.0"   # 加大应力权重 (保能量/力)
        incar_params["ML_MB"] = "3000"
        incar_params["ML_LBASIS_DISCARD"] = ".FALSE."
        write_incar(os.path.join(stage_dir, "INCAR"), incar_params)

        # stage_i+1 的 ML_AB 在串联脚本运行时由 stage_i 的 ML_ABN 提供

    # 生成串联提交脚本（顺序串行，避免并发覆盖/错序）
    concat_submit = os.path.join(concat_root, "submit_concat.sh")
    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --partition=8358P",
        "#SBATCH --job-name=mhec_mlff_concat",
        "#SBATCH --output=submit_concat_%j.out",
        "",
        "# Sequential concatenation training:",
        "# stage_i/ML_ABN -> stage_(i+1)/ML_AB",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        "fi",
        "",
        "set -e",
        "",
    ]

    for i, stage_dir in enumerate(stage_dirs):
        rel = os.path.relpath(stage_dir, concat_root)
        lines.append(f'echo "=== Running {rel} ==="')
        if i > 0:
            prev_rel = os.path.relpath(stage_dirs[i - 1], concat_root)
            lines.append(f'cp "$WORK_ROOT/{prev_rel}/ML_ABN" "$WORK_ROOT/{rel}/ML_AB"')
        lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
        lines.append("sbatch --wait run.sh")
        lines.append("popd > /dev/null")
        lines.append("")

    lines.extend([
        'echo "Concatenation training finished."',
        'echo "Final merged database: '
        + os.path.relpath(stage_dirs[-1], concat_root)
        + '/ML_ABN"',
        "",
    ])

    with open(concat_submit, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ================================================================
    # 2) 创建最终 refit 目录（以串联后的最后一个 ML_ABN 为输入）
    # ================================================================
    refit_dir = os.path.join(work_dir, "refit_final")
    os.makedirs(refit_dir, exist_ok=True)

    last_stage = stage_dirs[-1]
    last_src = stage_sources[-1]

    # 注意：真正执行 refit 前，请先完成 concat_train 串联并确保 last_stage/ML_ABN 存在
    if os.path.isfile(os.path.join(last_stage, "ML_ABN")):
        shutil.copy2(os.path.join(last_stage, "ML_ABN"), os.path.join(refit_dir, "ML_AB"))

    for fname in ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]:
        src = os.path.join(last_src, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(refit_dir, fname))

    # refit：结构相关项 + ML_MODE=refit；其余 ML 回归参数走 VASP 默认（见 Wiki ML_MODE / ML_IWEIGHT）
    # - ML_MODE=refit 自动设 ML_LFAST、NSW=1、ML_IALGO_LINREG=4、ML_SIGW0/ML_SIGV0、ML_EPS_LOW 等
    # - ML_IWEIGHT=3 为 Wiki 推荐（子集标准差归一化）
    # - 弹性常数对应力精度敏感: 保能量/力基准(1.0), 加大应力权重 ML_WTSIF
    refit_params = {
        "ENCUT": "500",
        "PREC": "Accurate",
        "LREAL": "Auto",
        "NCORE": "4",
        "IBRION": "-1",
        "ML_LMLFF": ".TRUE.",
        "ML_MODE": "refit",
        "ML_IWEIGHT": "3",
        "ML_WTOTEN": "1.0",
        "ML_WTIFOR": "1.0",
        "ML_WTSIF": "8.0",
        "NSW": "1",
    }
    write_incar(os.path.join(refit_dir, "INCAR"), refit_params)

    print(f"\nConcatenation root: {concat_root}/")
    print(f"  submit script: {concat_submit}")
    print(f"\nPlease run concatenation first:")
    print(f"  cd {concat_root}")
    print(f"  sbatch submit_concat.sh")
    print(f"\nAfter concat finishes, run final refit in:")
    print(f"  cd {refit_dir}")
    print(f"  cp ../concat_train/{os.path.basename(last_stage)}/ML_ABN ML_AB")
    print(f"  sbatch run.sh")
    print(f"\nAfter completion, final ML_FF will be in {refit_dir}/ML_FF")

    if auto_run:
        print("\nAuto mode enabled: running concatenation and final refit now.")
        try:
            subprocess.run(["sbatch", "--wait", "submit_concat.sh"], cwd=concat_root, check=True)
            shutil.copy2(os.path.join(last_stage, "ML_ABN"), os.path.join(refit_dir, "ML_AB"))
            subprocess.run(["sbatch", "--wait", "run.sh"], cwd=refit_dir, check=True)
            print(f"\nAuto pipeline finished. Final ML_FF: {refit_dir}/ML_FF")
        except Exception as e:
            print(f"\nAuto pipeline failed: {e}")
            print("You can rerun manually with the commands shown above.")


def main():
    parser = argparse.ArgumentParser(
        description="MHEC MLFF Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect crystal system and use symmetry-adapted strain set (recommended)
  python -m mhec.run_mlff_pipeline

  # Force full Voigt basis (backward compat / triclinic)
  python -m mhec.run_mlff_pipeline --elastic-mode general

  # Physically consistent: build Phase B from Phase A equilibrated lattices (rerun after NPT finished)
  python -m mhec.run_mlff_pipeline --b-lattice-source npt_equilibrated --b-skip-frac 0.5

  # After training completes, refit to get ML_FF
  python -m mhec.run_mlff_pipeline --refit mlff_train
""")
    parser.add_argument("--refit", type=str, metavar="DIR",
                        help="Collect ML_AB and refit (after training completes)")
    parser.add_argument("--auto-run", action="store_true",
                        help="Run concatenation and final refit automatically")
    parser.add_argument("--temps", type=float, nargs="+",
                        help="Training temperatures (K)")
    parser.add_argument("--strain-temps", type=float, nargs="+",
                        help="Temperatures (K) used for strained NVT training (default: same as --temps)")
    parser.add_argument("--strains", type=float, nargs="+",
                        help="Strain magnitudes for training")
    parser.add_argument("--elastic-mode", type=str, default="auto",
                        choices=["auto", "general"],
                        help="auto=symmetry-adapted; general=full Voigt")
    parser.add_argument("--nsw", type=int, default=5000,
                        help="MD steps per training dir (default: 5000)")
    parser.add_argument("--potim", type=float, default=1.0,
                        help="Time step in fs (default: 1.0)")
    parser.add_argument("--encut", type=float, default=500.0,
                        help="ENCUT (default: 500)")
    parser.add_argument("--outdir", type=str, default="mlff_train",
                        help="Output directory (default: mlff_train)")
    parser.add_argument("--concat-schedule", type=str, default="high_to_low",
                        choices=["high_to_low", "as_generated"],
                        help="Ordering of strained temps in training_dirs (affects concatenation order)")
    parser.add_argument("--b-lattice-source", type=str, default="npt_equilibrated",
                        choices=["initial", "npt_equilibrated"],
                        help="Phase B lattice source: initial POSCAR or Phase A NPT equilibrated lattice")
    parser.add_argument("--b-skip-frac", type=float, default=0.5,
                        help="Discard fraction of early NPT frames when averaging lattice (vasprun.xml). Default 0.5")
    parser.add_argument("--update-phase-b", type=str, metavar="DIR",
                        help="Update Phase B POSCAR files from Phase A NPT equilibrium lattices, then exit")
    args = parser.parse_args()

    if args.update_phase_b:
        _update_phase_b_poscars(args.update_phase_b, b_skip_frac=args.b_skip_frac)
        return

    if args.refit:
        refit_mlff(args.refit, auto_run=args.auto_run)
    else:
        generate_mlff_training(
            work_dir=args.outdir,
            temperatures=args.temps,
            strain_temperatures=args.strain_temps,
            strain_magnitudes=args.strains,
            nsw_train=args.nsw,
            potim=args.potim,
            encut=args.encut,
            elastic_training_mode=args.elastic_mode,
            concat_schedule=args.concat_schedule,
            b_lattice_source=args.b_lattice_source,
            b_skip_frac=args.b_skip_frac,
        )


if __name__ == "__main__":
    main()


# ============================================================
# 升温斜坡 (ramp) 训练模式
# ============================================================

def generate_mlff_training_ramp(
    work_dir: str = "mlff_train_ramp",
    t_start: float = 300.0,
    t_end: float = 1500.0,
    strain_magnitudes: List[float] = None,
    nsw_ramp: int = 30000,
    potim: float = 1.0,
    encut: Optional[float] = 500.0,
    poscar_path: str = "POSCAR",
    potcar_path: str = "POTCAR",
    kpoints_path: str = "KPOINTS",
    run_sh_path: str = "run.sh",
    elastic_training_mode: str = "auto",
    incar_path: str = "INCAR",
) -> Dict:
    """
    升温斜坡 MLFF 训练 (顺序续训)。

    Phase A: 单个 NPT 升温训练 (TEBEG=t_start, TEEND=t_end), 从零训练,
             一条轨迹覆盖整个温区 + 热膨胀。
    Phase B: 各应变方向 NVT 升温训练 (同样 t_start→t_end), 以 ML_ISTART=1
             在上一阶段力场基础上 **续训**, 只补充应变-应力响应。
    Phase C: refit 得到最终 ML_FF (加大应力权重)。

    全程严格顺序: A → B1 → B2 → ... → refit, 每一步用上一阶段的 ML_ABN
    作为本阶段的 ML_AB 输入 (VASP 官方推荐的安全数据融合方式, 无重复构型)。
    """
    from .vasp_io import read_poscar, write_poscar, write_incar
    from .strain import voigt_strain_to_tensor

    mode = (elastic_training_mode or "auto").strip().lower()
    if mode not in ("auto", "general"):
        print(T(f"Error: elastic_training_mode 必须是 auto/general, 收到 {elastic_training_mode!r}",
                f"Error: elastic_training_mode must be auto/general, got {elastic_training_mode!r}"))
        return {}

    for f, name in [(poscar_path, "POSCAR"), (potcar_path, "POTCAR"),
                     (kpoints_path, "KPOINTS"), (run_sh_path, "run.sh")]:
        if not os.path.isfile(f):
            print(T(f"Error: 缺少 {name}: {f}", f"Error: missing {name}: {f}"))
            return {}

    if strain_magnitudes is None:
        strain_magnitudes = [0.01, 0.02]

    poscar = read_poscar(poscar_path)
    lattice0 = poscar["lattice"]
    crystal_system = identify_crystal_system(lattice0)
    cs_label = cs_name(crystal_system)

    if mode == "auto":
        strain_set = _get_training_strain_set(crystal_system)
    else:
        strain_set = _get_training_strain_set(CrystalSystem.TRICLINIC)

    t_lo, t_hi = float(min(t_start, t_end)), float(max(t_start, t_end))
    print(T(f"  → 晶系: {cs_label} | 升温 ramp {int(t_lo)}→{int(t_hi)}K | 模式: {mode}",
            f"  → Crystal system: {cs_label} | heating ramp {int(t_lo)}→{int(t_hi)}K | mode: {mode}"))

    os.makedirs(work_dir, exist_ok=True)

    base_params = {
        "PREC": "Accurate", "EDIFF": "1E-6", "EDIFFG": "-1E-3",
        "ADDGRID": ".TRUE.", "LREAL": "Auto", "NCORE": "4", "ALGO": "Normal",
        "ISMEAR": "0", "SIGMA": "0.05", "NELM": "200", "ISYM": "0",
        "IBRION": "0", "POTIM": str(potim), "LWAVE": ".FALSE.", "LCHARG": ".FALSE.",
        "ML_LMLFF": ".TRUE.", "ML_MODE": "train",
        "ML_IALGO_LINREG": "4", "ML_CX": "0.1", "ML_MB": "3000",
        "ML_MCONF_NEW": "1", "ML_LBASIS_DISCARD": ".FALSE.",
        # 保能量/力, 加大应力权重 (弹性常数关键)
        "ML_IWEIGHT": "3", "ML_WTOTEN": "1.0", "ML_WTIFOR": "1.0", "ML_WTSIF": "4.0",
        "ENCUT": str(int(encut if encut is not None else 500)),
    }

    # 继承用户 INCAR 的泛函/色散/Hubbard 物理设置 (GGA=PS, IVDW, VDW_*, LDAU* 等),
    # 否则升温训练学到的是默认 PBE 力场。管线管理的 MD/ML 控制参数不被覆盖。
    # 只强制 MD/训练控制参数, 其余电子结构参数从用户 INCAR 继承 (同定温训练)
    _MANAGED_R = {
        "IBRION", "MDALGO", "ISIF", "NSW", "TEBEG", "TEEND", "POTIM",
        "SMASS", "PMASS", "PSTRESS", "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L",
        "ISYM", "LWAVE", "LCHARG",
    }
    if incar_path and os.path.isfile(incar_path):
        try:
            from .vasp_io import read_incar
            user_inc = read_incar(incar_path)
            carried = {k: v for k, v in user_inc.items()
                       if k.upper() not in _MANAGED_R and not k.upper().startswith("ML_")}
            if carried:
                base_params.update(carried)
                print(T(f"  → 继承用户 INCAR 物理设置 ({incar_path}): {', '.join(sorted(carried))}",
                        f"  → Inherited physical settings from user INCAR ({incar_path}): {', '.join(sorted(carried))}"))
        except Exception as e:
            print(T(f"  ! 读取用户 INCAR ({incar_path}) 失败, 未继承额外参数: {e}",
                    f"  ! Failed to read user INCAR ({incar_path}); no extra parameters inherited: {e}"))

    ordered = []   # (dirname, desc)

    def _make(subdir, desc, lattice, ensemble, istart):
        d = os.path.join(work_dir, subdir)
        os.makedirs(d, exist_ok=True)
        p = dict(poscar)
        p["lattice"] = lattice
        write_poscar(os.path.join(d, "POSCAR"), p)
        params = dict(base_params)
        params["NSW"] = str(nsw_ramp)
        params["TEBEG"] = str(int(t_lo))
        params["TEEND"] = str(int(t_hi))     # 线性升温
        params["ML_ISTART"] = str(istart)
        if ensemble == "npt":
            _nsp = len(poscar.get("counts") or [1]) or 1
            params.update({"ISIF": "3", "MDALGO": "3",
                           "LANGEVIN_GAMMA": " ".join(["10.0"] * _nsp),
                           "LANGEVIN_GAMMA_L": "1.0", "PMASS": "1000", "PSTRESS": "0"})
        else:
            params.update({"ISIF": "2", "MDALGO": "2", "SMASS": "2"})
        write_incar(os.path.join(d, "INCAR"), params)
        for src in [potcar_path, kpoints_path, run_sh_path]:
            shutil.copy2(src, os.path.join(d, os.path.basename(src)))
        ordered.append((subdir, desc))

    # Phase A: 升温 NPT, 从零训练
    _make("A_ramp_npt", f"NPT 升温 {int(t_lo)}->{int(t_hi)}K (从零训练)",
          lattice0, "npt", 0)

    # Phase B: 各应变升温 NVT, 续训 (ML_ISTART=1)
    for mag in strain_magnitudes:
        for idx, name, signs in strain_set["uniax"]:
            for sgn in signs:
                v = np.zeros(6); v[idx] = sgn * mag
                lat = _lattice_from_voigt(lattice0, v, voigt_strain_to_tensor)
                tag = f"e{mag:.3f}" if sgn > 0 else f"em{mag:.3f}"
                _make(f"B_uniax_{name}_{tag}",
                      f"单轴 {name} {sgn*mag:+.4f} 升温续训", lat, "nvt", 1)
        for idx, name, signs in strain_set["shear"]:
            for sgn in signs:
                v = np.zeros(6); v[idx] = sgn * mag
                lat = _lattice_from_voigt(lattice0, v, voigt_strain_to_tensor)
                tag = f"e{mag:.3f}" if sgn > 0 else f"em{mag:.3f}"
                _make(f"B_shear_{name}_{tag}",
                      f"剪切 {name} {sgn*mag:+.4f} 升温续训", lat, "nvt", 1)
        for sgn in strain_set["vol"]:
            factor = (1.0 + sgn * mag) ** (1.0 / 3.0)
            if factor <= 0:
                continue
            tag = f"e{mag:.3f}" if sgn > 0 else f"em{mag:.3f}"
            _make(f"B_vol_{tag}", f"体积 {sgn*mag:+.4f} 升温续训",
                  lattice0 * factor, "nvt", 1)

    # Phase C: refit_final
    refit_dir = os.path.join(work_dir, "refit_final")
    os.makedirs(refit_dir, exist_ok=True)
    for src in [poscar_path, potcar_path, kpoints_path, run_sh_path]:
        shutil.copy2(src, os.path.join(refit_dir, os.path.basename(src)))
    refit_params = {
        "ENCUT": str(int(encut if encut is not None else 500)),
        "PREC": "Accurate", "LREAL": "Auto", "NCORE": "4",
        # refit 为静态 ML 重拟合, 无 MD: IBRION=-1 (否则 NSW>0 时 VASP 默认 IBRION=0
        # 会要求 POTIM 而报错); POTIM 一并给出以防万一
        "IBRION": "-1", "POTIM": str(potim),
        "ML_LMLFF": ".TRUE.", "ML_MODE": "refit",
        "ML_IWEIGHT": "3", "ML_WTOTEN": "1.0", "ML_WTIFOR": "1.0", "ML_WTSIF": "8.0",
        "NSW": "1",
    }
    write_incar(os.path.join(refit_dir, "INCAR"), refit_params)

    # 顺序续训提交脚本: A -> B1 -> B2 -> ... -> refit_final
    _write_ramp_submit_script(work_dir, [d for d, _ in ordered])

    # 记录
    with open(os.path.join(work_dir, "training_dirs.txt"), "w") as f:
        f.write(f"# MHEC MLFF 升温 ramp 训练\n# {int(t_lo)}->{int(t_hi)}K\n")
        f.write(f"# 应变幅度: {strain_magnitudes}\n# NSW/段: {nsw_ramp}\n")
        f.write(f"# 顺序续训: A -> B... -> refit_final\n\n")
        for i, (d, desc) in enumerate(ordered):
            f.write(f"{i}\t{d}\t{desc}\n")

    print(T(f"\n生成 {len(ordered)} 个训练段 + refit_final 于 {work_dir}/",
            f"\nGenerated {len(ordered)} training segments + refit_final in {work_dir}/"))
    print(T(f"  Phase A 升温 NPT: 1 段", f"  Phase A heating NPT: 1 segment"))
    print(T(f"  Phase B 应变升温续训: {len(ordered)-1} 段",
            f"  Phase B strained heating continuation: {len(ordered)-1} segments"))
    print(T(f"\n下一步:", f"\nNext steps:"))
    print(T(f"  提交: sbatch {work_dir}/submit_train.sh  (或 bash, 顺序串行)",
            f"  Submit: sbatch {work_dir}/submit_train.sh  (or bash; sequential)"))
    print(T(f"  完成后最终力场: {work_dir}/refit_final/ML_FFN  → cp 为 ML_FF 使用",
            f"  Final force field after completion: {work_dir}/refit_final/ML_FFN  → cp to ML_FF"))
    return {"work_dir": work_dir, "n_stages": len(ordered)}


def _write_ramp_submit_script(work_dir: str, ordered_dirs: List[str]) -> None:
    """生成升温 ramp 的顺序续训提交脚本。

    严格串行: 每段用 sbatch --wait 等待完成, 并把上一段 ML_ABN 复制为本段 ML_AB,
    最后 refit_final 用最末段 ML_ABN 拟合出 ML_FFN。
    """
    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --partition=8358P",
        "#SBATCH --job-name=mhec_mlff_ramp",
        "#SBATCH --output=submit_train_%j.out",
        "",
        "# MHEC MLFF 升温 ramp 顺序续训",
        "# A(升温NPT,从零) -> B(各应变升温,续训) -> refit_final",
        "# 用法: sbatch submit_train.sh  或  bash submit_train.sh",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        "else",
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        "fi",
        "",
        "set -e",
        "",
    ]
    for i, d in enumerate(ordered_dirs):
        lines.append(f'echo "=== [{i+1}/{len(ordered_dirs)}] {d} ==="')
        if i > 0:
            prev = ordered_dirs[i - 1]
            lines.append(f'cp "$WORK_ROOT/{prev}/ML_ABN" "$WORK_ROOT/{d}/ML_AB"')
        lines.append(f'pushd "$WORK_ROOT/{d}" > /dev/null')
        lines.append("sbatch --wait run.sh")
        lines.append("popd > /dev/null")
        lines.append("")
    # refit_final
    last = ordered_dirs[-1]
    lines.extend([
        'echo "=== refit_final ==="',
        f'cp "$WORK_ROOT/{last}/ML_ABN" "$WORK_ROOT/refit_final/ML_AB"',
        'pushd "$WORK_ROOT/refit_final" > /dev/null',
        "sbatch --wait run.sh",
        "popd > /dev/null",
        "",
        'echo "完成! 最终力场: refit_final/ML_FFN (cp 为 ML_FF 使用)"',
        "",
    ])
    path = os.path.join(work_dir, "submit_train.sh")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass
    print(f"  +-> {path}")
