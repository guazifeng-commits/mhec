"""
VASP 输入输出处理模块。

处理 POSCAR/CONTCAR 读写、INCAR 读写、KPOINTS 生成。
"""

import numpy as np
import os
import re
from typing import Dict, Optional


def read_poscar(filepath: str) -> Dict:
    """
    读取 POSCAR/CONTCAR 文件。

    Returns
    -------
    dict : comment, scale, lattice(3×3), species, counts,
           selective, coord_type, positions(N×3)
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    comment = lines[0].strip()
    scale = float(lines[1].strip())

    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = [float(x) for x in lines[2 + i].split()]
    lattice *= scale

    # 第6行可能是元素名或原子数
    tokens = lines[5].split()
    try:
        counts = [int(x) for x in tokens]
        species = None
        count_line = 5
    except ValueError:
        species = tokens
        counts = [int(x) for x in lines[6].split()]
        count_line = 6

    n_atoms = sum(counts)

    # Selective dynamics?
    next_line = count_line + 1
    selective = False
    stripped = lines[next_line].strip() if next_line < len(lines) else ""
    if stripped and stripped[0].lower() == "s":
        selective = True
        next_line += 1

    coord_type = lines[next_line].strip()
    next_line += 1

    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        positions[i] = [float(x) for x in lines[next_line + i].split()[:3]]

    return {
        "comment": comment,
        "scale": 1.0,  # 已乘入 lattice
        "lattice": lattice,
        "species": species,
        "counts": counts,
        "selective": selective,
        "coord_type": coord_type,
        "positions": positions,
    }


def write_poscar(filepath: str, poscar: Dict) -> None:
    """写入 POSCAR 文件。缩放因子归一化为 1.0。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(poscar["comment"] + "\n")
        f.write("1.0\n")
        for i in range(3):
            lat = poscar["lattice"][i]
            f.write(f"  {lat[0]:20.14f}  {lat[1]:20.14f}  {lat[2]:20.14f}\n")
        if poscar.get("species"):
            f.write("  " + "  ".join(poscar["species"]) + "\n")
        f.write("  " + "  ".join(str(c) for c in poscar["counts"]) + "\n")
        if poscar.get("selective"):
            f.write("Selective dynamics\n")
        f.write(poscar["coord_type"] + "\n")
        for pos in poscar["positions"]:
            f.write(f"  {pos[0]:20.14f}  {pos[1]:20.14f}  {pos[2]:20.14f}\n")


def read_incar(filepath: str) -> Dict[str, str]:
    """读取 INCAR 文件为键值对字典。忽略注释行和空行。"""
    params = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.split("#")[0].split("!")[0].strip()
            if not line or "=" not in line:
                continue
            # 处理同一行多个参数（分号分隔）
            for part in line.split(";"):
                part = part.strip()
                if "=" in part:
                    key, val = part.split("=", 1)
                    params[key.strip().upper()] = val.strip()
    return params


def write_incar(filepath: str, params: Dict[str, str]) -> None:
    """将参数字典写入 INCAR 文件，每行一个 KEY = VALUE。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        for key, val in params.items():
            f.write(f"{key} = {val}\n")


def write_kpoints_gamma(filepath: str) -> None:
    """写入 Gamma 点 KPOINTS 文件。"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write("Automatic mesh\n")
        f.write("0\n")
        f.write("Gamma\n")
        f.write("1 1 1\n")
        f.write("0 0 0\n")
