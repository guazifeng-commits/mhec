"""
NVT 变形目录生成器。

从 NPT 计算结果提取各温度平衡结构（优先 vasprun.xml 时间平均，回退 CONTCAR），
生成 NVT 弹性常数计算所需的变形目录和 POSCAR。
可作为独立命令调用: python -m mhec.nvt_generator <work_dir>
"""

import os
import sys
import json
import re
import numpy as np
from .vasp_io import read_poscar, write_poscar
from .strain import get_deform_codes, get_amplitude_labels, generate_strained_structures
from .strain_vc import get_vc_modes, generate_vc_structures, VC_SUPPORTED
from .crystal_system import (
    CrystalSystem, identify_crystal_system, enforce_lattice_symmetry,
    resolve_crystal_system,
)


def update_nvt_from_npt(work_dir: str, skip_frac: float = 0.5) -> None:
    """
    从 NPT 结果提取平衡晶格，更新 NVT 目录的所有 POSCAR。

    提取策略（按优先级）：
    1. vasprun.xml 时间平均（丢弃前 skip_frac）+ 对称性约束
    2. CONTCAR + 对称性约束

    目录结构:
        work_dir/
            phase1_npt/npt_100K/vasprun.xml (或 CONTCAR)
            phase2_nvt/100K/equilibrium/POSCAR
            phase2_nvt/100K/deform100000_n3/POSCAR
            mhec.yaml
    """
    npt_base = os.path.join(work_dir, "phase1_npt")
    nvt_base = os.path.join(work_dir, "phase2_nvt")
    config_path = os.path.join(work_dir, "mhec.yaml")

    if not os.path.isdir(npt_base):
        print(f"Error: NPT dir not found: {npt_base}")
        return
    if not os.path.isdir(nvt_base):
        print(f"Error: NVT dir not found: {nvt_base}")
        return

    # 读取配置 (默认与 config.py 一致: mag=0.005, n_points=9)
    magnitude = 0.005
    n_points = 9
    strain_method = "standard"
    elastic_method = "ss"
    crystal_system_str = None
    space_group = None

    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        magnitude = cfg.get("magnitude", 0.005)
        n_points = cfg.get("n_points", 9)
        # 若 yaml 里给了 max_strain / strain_points, 据此反推 magnitude & n_points
        from .config import normalize_elastic_method, resolve_strain_sampling
        if cfg.get("strain_points") or cfg.get("max_strain") is not None:
            try:
                magnitude, n_points = resolve_strain_sampling(
                    magnitude, n_points,
                    max_strain=cfg.get("max_strain"), strain_points=cfg.get("strain_points"))
            except ValueError as e:
                print(f"Warning: strain sampling resolve failed ({e}); using magnitude/n_points as-is.")
        strain_method = cfg.get("strain_method", "standard")
        elastic_method = normalize_elastic_method(cfg.get("elastic_method", "sc"))
        crystal_system_str = cfg.get("crystal_system")
        space_group = cfg.get("space_group")
        _cfg_encut = cfg.get("encut")
        _cfg_potim = cfg.get("potim")
        _cfg_overrides = cfg.get("incar_overrides")
    else:
        _cfg_encut = None
        _cfg_potim = None
        _cfg_overrides = None

    # 从配置解析已知晶系一次 (空间群优先, 其次晶系字符串); 无则留 None 由几何识别兜底
    cs_known = None
    if crystal_system_str or space_group:
        try:
            cs_known = resolve_crystal_system(
                crystal_system=crystal_system_str, space_group=space_group)
        except ValueError as e:
            print(f"Warning: crystal system resolve failed ({e}); will auto-detect.")

    # 扫描 NPT 温度目录
    npt_temps = {}
    for d in sorted(os.listdir(npt_base)):
        m = re.match(r"npt_(\d+)K$", d)
        if m and os.path.isdir(os.path.join(npt_base, d)):
            temp = int(m.group(1))
            npt_temps[temp] = os.path.join(npt_base, d)

    if not npt_temps:
        print(f"Error: no npt_*K dirs in {npt_base}")
        return

    print(f"Found {len(npt_temps)} temperatures: {', '.join(str(t) for t in sorted(npt_temps))}")
    print(f"Strain: elastic_method={elastic_method}, magnitude={magnitude}, n_points={n_points}")

    # 使用 LatticeOptimizer 的提取逻辑（vasprun 时间平均 + 对称性约束）
    from .lattice_optimizer import LatticeOptimizer

    # 读取模板 POSCAR（获取原子信息）
    template_poscar = None
    for temp_dir in npt_temps.values():
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
        print("Error: no CONTCAR or POSCAR found in any NPT dir")
        return

    optimizer = LatticeOptimizer(
        poscar=template_poscar,
        temperatures=sorted([float(t) for t in npt_temps.keys()]),
        encut=_cfg_encut,
        potim=_cfg_potim or 1.0,
        user_overrides=_cfg_overrides or {},
        crystal_system=cs_known,
    )

    updated = 0
    for temp in sorted(npt_temps.keys()):
        npt_dir = npt_temps[temp]
        nvt_temp_dir = os.path.join(nvt_base, f"{temp}K")

        if not os.path.isdir(nvt_temp_dir):
            print(f"  Skip {temp}K: NVT dir not found")
            continue

        # 提取平衡晶格（vasprun 时间平均 → CONTCAR 回退，含对称性约束）
        try:
            avg_lattice, stats = optimizer.extract_equilibrium_lattice(
                npt_dir, skip_frac=skip_frac)
            src = stats.get("source", "unknown")
            print(f"  {temp}K: a={stats['a']:.6f} ({src})")
        except Exception as e:
            print(f"  {temp}K: extraction failed: {e}")
            continue

        lattice = avg_lattice

        # 晶系: 优先用配置解析出的已知晶系, 否则从(已按对称约束的)晶格识别
        cs = cs_known if cs_known is not None else identify_crystal_system(lattice)

        amp_labels = get_amplitude_labels(n_points)

        # 构建平衡 POSCAR
        eq_poscar = {
            "comment": template_poscar["comment"],
            "scale": template_poscar["scale"],
            "lattice": lattice,
            "species": template_poscar.get("species"),
            "counts": template_poscar["counts"],
            "selective": template_poscar.get("selective", False),
            "coord_type": template_poscar["coord_type"],
            "positions": template_poscar["positions"],
        }

        # 更新 equilibrium
        eq_dir = os.path.join(nvt_temp_dir, "equilibrium")
        if os.path.isdir(eq_dir):
            write_poscar(os.path.join(eq_dir, "POSCAR"), eq_poscar)
            updated += 1

        # 按弹性方法生成变形结构并更新 (elastic_method 已归一为 "sc"/"sa")
        if elastic_method == "sa":
            modes = get_vc_modes(cs)
            strained = generate_vc_structures(lattice, cs, magnitude, n_points)
            dir_names = [m["name"] for m in modes]
            strain_map = {m["name"]: strained[m["name"]] for m in modes}
            n_modes = len(modes)
        else:
            codes = get_deform_codes(cs, strain_method)
            strained = generate_strained_structures(lattice, codes, magnitude, n_points)
            dir_names = list(codes)
            strain_map = {c: strained[c] for c in codes}
            n_modes = len(codes)

        for name in dir_names:
            for label, _ in amp_labels:
                d = os.path.join(nvt_temp_dir, f"{name}_{label}")
                if os.path.isdir(d):
                    deformed_poscar = dict(eq_poscar)
                    deformed_poscar["lattice"] = strain_map[name][label]
                    write_poscar(os.path.join(d, "POSCAR"), deformed_poscar)
                    updated += 1

        n_dirs = 1 + n_modes * len(amp_labels)
        print(f"  {temp}K: updated {n_dirs} POSCAR files")

    print(f"\nDone: {updated} POSCAR files updated")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mhec.nvt_generator <work_dir> [skip_frac]")
        print("  Reads phase1_npt/npt_*K/ equilibrium lattices")
        print("  Updates phase2_nvt/*/POSCAR")
        sys.exit(1)
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    update_nvt_from_npt(sys.argv[1], skip_frac=skip)
