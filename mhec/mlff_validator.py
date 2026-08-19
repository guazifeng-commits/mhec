"""
MLFF 力场精度验证模块。

解析 VASP ML_LOGFILE 文件，提取训练/验证过程中的误差指标:
  - ERR:  能量/力/应力的 RMSE (训练集)
  - BEEF: 贝叶斯误差估计 (预测模式)
  - STDAB: 训练数据标准差

参考: https://vasp.at/wiki/ML_LOGFILE
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from .i18n import T


# ============================================================
# 数据结构
# ============================================================

class MLFFErrorRecord:
    """单步误差记录。"""
    __slots__ = ("step", "energy", "force", "stress")

    def __init__(self, step: int, energy: float, force: float, stress: float):
        self.step = step
        self.energy = energy
        self.force = force
        self.stress = stress

    def __repr__(self):
        return (f"MLFFErrorRecord(step={self.step}, "
                f"E={self.energy:.6e}, F={self.force:.6e}, S={self.stress:.6e})")


class MLFFValidationResult:
    """MLFF 验证结果汇总。"""

    def __init__(self):
        self.err_records: List[MLFFErrorRecord] = []
        self.beef_records: List[MLFFErrorRecord] = []
        self.stdab_records: List[MLFFErrorRecord] = []
        self.status_counts: Dict[str, int] = {}
        self.total_steps: int = 0

    # ---------- 统计量 ----------

    @property
    def has_err(self) -> bool:
        return len(self.err_records) > 0

    @property
    def has_beef(self) -> bool:
        return len(self.beef_records) > 0

    def final_err(self) -> Optional[MLFFErrorRecord]:
        return self.err_records[-1] if self.err_records else None

    def mean_err(self) -> Optional[Tuple[float, float, float]]:
        if not self.err_records:
            return None
        e = np.mean([r.energy for r in self.err_records])
        f = np.mean([r.force for r in self.err_records])
        s = np.mean([r.stress for r in self.err_records])
        return (e, f, s)

    def final_beef(self) -> Optional[MLFFErrorRecord]:
        return self.beef_records[-1] if self.beef_records else None

    def mean_beef(self) -> Optional[Tuple[float, float, float]]:
        if not self.beef_records:
            return None
        e = np.mean([r.energy for r in self.beef_records])
        f = np.mean([r.force for r in self.beef_records])
        s = np.mean([r.stress for r in self.beef_records])
        return (e, f, s)

    def final_stdab(self) -> Optional[MLFFErrorRecord]:
        return self.stdab_records[-1] if self.stdab_records else None

    # ---------- 精度评估 ----------

    def assess_quality(self) -> Dict[str, str]:
        """
        评估 MLFF 精度等级。

        对于弹性常数计算，应力精度至关重要:
          - 应力 RMSE < 2 kBar: 优秀
          - 应力 RMSE < 5 kBar: 良好
          - 应力 RMSE < 10 kBar: 可接受
          - 应力 RMSE >= 10 kBar: 需要改进

        力的精度:
          - 力 RMSE < 0.05 eV/Å: 优秀
          - 力 RMSE < 0.10 eV/Å: 良好
          - 力 RMSE < 0.20 eV/Å: 可接受
          - 力 RMSE >= 0.20 eV/Å: 需要改进

        能量精度:
          - 能量 RMSE < 1 meV/atom: 优秀
          - 能量 RMSE < 5 meV/atom: 良好
          - 能量 RMSE < 10 meV/atom: 可接受
          - 能量 RMSE >= 10 meV/atom: 需要改进
        """
        result = {}

        # 优先使用 ERR (训练集 RMSE)，其次 BEEF (贝叶斯估计)
        rec = self.final_err() or self.final_beef()
        if rec is None:
            return {"overall": "无数据"}

        # 能量 (eV/atom → meV/atom)
        e_mev = rec.energy * 1000
        if e_mev < 1:
            result["energy"] = "优秀"
        elif e_mev < 5:
            result["energy"] = "良好"
        elif e_mev < 10:
            result["energy"] = "可接受"
        else:
            result["energy"] = "需要改进"

        # 力 (eV/Å)
        if rec.force < 0.05:
            result["force"] = "优秀"
        elif rec.force < 0.10:
            result["force"] = "良好"
        elif rec.force < 0.20:
            result["force"] = "可接受"
        else:
            result["force"] = "需要改进"

        # 应力 (kBar)
        if rec.stress < 2:
            result["stress"] = "优秀"
        elif rec.stress < 5:
            result["stress"] = "良好"
        elif rec.stress < 10:
            result["stress"] = "可接受"
        else:
            result["stress"] = "需要改进"

        # 综合评估
        levels = {"优秀": 0, "良好": 1, "可接受": 2, "需要改进": 3}
        worst = max(levels.get(v, 3) for v in result.values())
        level_names = {0: "优秀", 1: "良好", 2: "可接受", 3: "需要改进"}
        result["overall"] = level_names[worst]

        return result


# ============================================================
# 解析器
# ============================================================

# 匹配浮点数 (科学计数法 + 普通小数)
_FLOAT = r"[+-]?(?:\d+\.\d+[Ee][+-]?\d+|\d+\.\d+)"

# ERR 行: ERR  step  rmse_energy  rmse_force  rmse_stress
_RE_ERR = re.compile(
    rf"^ERR\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})"
)

# BEEF 行: BEEF  step  bee_energy  bee_force  bee_stress  ...
# 后续列 (bayesian_error_*) 按需扩展
_RE_BEEF = re.compile(
    rf"^BEEF\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})"
)

# STDAB 行: STDAB  step  std_energy  std_force  std_stress
_RE_STDAB = re.compile(
    rf"^STDAB\s+(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})"
)

# STATUS 行: STATUS  step  state  ...
_RE_STATUS = re.compile(
    r"^STATUS\s+(\d+)\s+(\w+)"
)


def parse_ml_logfile(filepath: str) -> MLFFValidationResult:
    """
    解析 ML_LOGFILE 文件。

    Parameters
    ----------
    filepath : ML_LOGFILE 文件路径

    Returns
    -------
    MLFFValidationResult 包含所有解析到的误差数据
    """
    result = MLFFValidationResult()

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ML_LOGFILE 不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # 跳过注释行
            if line.startswith("#"):
                continue

            m = _RE_ERR.match(line)
            if m:
                result.err_records.append(MLFFErrorRecord(
                    step=int(m.group(1)),
                    energy=float(m.group(2)),
                    force=float(m.group(3)),
                    stress=float(m.group(4)),
                ))
                continue

            m = _RE_BEEF.match(line)
            if m:
                result.beef_records.append(MLFFErrorRecord(
                    step=int(m.group(1)),
                    energy=float(m.group(2)),
                    force=float(m.group(3)),
                    stress=float(m.group(4)),
                ))
                continue

            m = _RE_STDAB.match(line)
            if m:
                result.stdab_records.append(MLFFErrorRecord(
                    step=int(m.group(1)),
                    energy=float(m.group(2)),
                    force=float(m.group(3)),
                    stress=float(m.group(4)),
                ))
                continue

            m = _RE_STATUS.match(line)
            if m:
                state = m.group(2)
                result.status_counts[state] = result.status_counts.get(state, 0) + 1
                result.total_steps = max(result.total_steps, int(m.group(1)))
                continue

    return result


# ============================================================
# 报告输出
# ============================================================

def print_validation_report(result: MLFFValidationResult, label: str = "") -> None:
    """打印单个 ML_LOGFILE 的验证报告。"""
    _QMAP = {"优秀": "Excellent", "良好": "Good", "可接受": "Acceptable",
             "需要改进": "Needs improvement", "无数据": "No data", "-": "-"}
    def _q(v):
        return _QMAP.get(v, v) if T("zh", "en") == "en" else v
    if label:
        print(f"\n  ── {label} ──")

    print(T(f"  总步数: {result.total_steps}", f"  Total steps: {result.total_steps}"))
    if result.status_counts:
        parts = [f"{k}: {v}" for k, v in sorted(result.status_counts.items())]
        print(T(f"  状态统计: {', '.join(parts)}", f"  Status counts: {', '.join(parts)}"))

    # ERR (训练集 RMSE)
    if result.has_err:
        final = result.final_err()
        mean = result.mean_err()
        print(T("\n  训练集 RMSE (ERR):", "\n  Training-set RMSE (ERR):"))
        print(f"    {'':>8}  {T('能量 (meV/atom)','Energy (meV/atom)'):>16}  {T('力 (eV/Å)','Force (eV/Å)'):>12}  {T('应力 (kBar)','Stress (kBar)'):>12}")
        print(f"    {T('最终值','Final'):>8}  {final.energy*1000:>16.4f}  {final.force:>12.6f}  {final.stress:>12.4f}")
        print(f"    {T('平均值','Mean'):>8}  {mean[0]*1000:>16.4f}  {mean[1]:>12.6f}  {mean[2]:>12.4f}")
        print(T(f"    数据点数: {len(result.err_records)}", f"    Data points: {len(result.err_records)}"))

    # BEEF (贝叶斯误差估计)
    if result.has_beef:
        final = result.final_beef()
        mean = result.mean_beef()
        print(T("\n  贝叶斯误差估计 (BEEF):", "\n  Bayesian error estimate (BEEF):"))
        print(f"    {'':>8}  {T('能量 (meV/atom)','Energy (meV/atom)'):>16}  {T('力 (eV/Å)','Force (eV/Å)'):>12}  {T('应力 (kBar)','Stress (kBar)'):>12}")
        print(f"    {T('最终值','Final'):>8}  {final.energy*1000:>16.4f}  {final.force:>12.6f}  {final.stress:>12.4f}")
        print(f"    {T('平均值','Mean'):>8}  {mean[0]*1000:>16.4f}  {mean[1]:>12.6f}  {mean[2]:>12.4f}")
        print(T(f"    数据点数: {len(result.beef_records)}", f"    Data points: {len(result.beef_records)}"))

    # STDAB (训练数据标准差)
    if result.final_stdab():
        std = result.final_stdab()
        print(T("\n  训练数据标准差 (STDAB):", "\n  Training-data std. dev. (STDAB):"))
        print(T(f"    能量: {std.energy*1000:.4f} meV/atom", f"    Energy: {std.energy*1000:.4f} meV/atom"))
        print(T(f"    力:   {std.force:.6f} eV/Å", f"    Force:  {std.force:.6f} eV/Å"))
        print(T(f"    应力: {std.stress:.4f} kBar", f"    Stress: {std.stress:.4f} kBar"))

    # 精度评估
    quality = result.assess_quality()
    if quality.get("overall") != "无数据":
        print(T("\n  精度评估:", "\n  Accuracy assessment:"))
        print(T(f"    能量: {_q(quality.get('energy', '-'))}", f"    Energy: {_q(quality.get('energy', '-'))}"))
        print(T(f"    力:   {_q(quality.get('force', '-'))}", f"    Force:  {_q(quality.get('force', '-'))}"))
        print(T(f"    应力: {_q(quality.get('stress', '-'))}", f"    Stress: {_q(quality.get('stress', '-'))}"))
        print(T(f"    综合: {_q(quality['overall'])}", f"    Overall: {_q(quality['overall'])}"))

        if quality["overall"] == "需要改进":
            print(T("\n  建议:", "\n  Suggestions:"))
            if quality.get("stress") == "需要改进":
                print(T("    - 应力精度不足，弹性常数计算结果可能不可靠",
                        "    - Stress accuracy insufficient; elastic constants may be unreliable"))
                print(T("    - 尝试增加训练步数 (NSW) 或降低 ML_CX",
                        "    - Try increasing training steps (NSW) or lowering ML_CX"))
                print(T("    - 确保 PREC=Accurate, EDIFF=1E-8; 精度不足可尝试 LREAL=.FALSE. (更精确但更慢/更耗内存)",
                        "    - Ensure PREC=Accurate, EDIFF=1E-8; if inaccurate try LREAL=.FALSE. (more accurate but slower/heavier)"))
            if quality.get("force") == "需要改进":
                print(T("    - 力的精度不足，可能影响 MD 轨迹质量",
                        "    - Force accuracy insufficient; may affect MD trajectory quality"))
                print(T("    - 考虑增加训练数据量 (ML_MB)",
                        "    - Consider increasing training data (ML_MB)"))


# ============================================================
# DFT-MLFF 单点抽样验证 (Parity Plot)
# ============================================================
"""
单点抽样验证工具 — 从已完成的 MLFF run 轨迹中等间隔抽帧,
对每帧做一次纯 DFT 单点 (关闭 MLFF), 然后比较:
  1) 能量 (eV/atom)
  2) 力   (eV/Å)  — 每个原子每个分量
  3) 应力 (kBar)  — 6 个 Voigt 分量

输出 3 张 parity plot (能量 / 力 / 应力), 每张图横轴为 DFT, 纵轴为 MLFF,
附带 RMSE、MAE、R²、y=x 参考线和线性拟合。
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import os
import re
import shutil


# ============================================================
# 轨迹抽帧 + 输入生成
# ============================================================

def generate_validation_dirs(
    run_dir: str,
    out_dir: str,
    n_frames: int = 50,
    skip_initial: int = 500,
    template_incar: Optional[str] = None,
) -> Dict:
    """
    从完成的 MLFF run 轨迹中等间隔抽 N 帧, 为每帧生成一个 DFT 单点目录。

    Parameters
    ----------
    run_dir : 已完成的 MLFF run 目录 (包含 XDATCAR, OUTCAR, INCAR, POSCAR, POTCAR)
    out_dir : 验证目录根, 会在其下创建 frame_0001/ ... frame_NNNN/
    n_frames : 抽帧数量, 默认 50 (更多点能让 parity plot 更有说服力)
    skip_initial : 跳过初始 MD 步数 (默认 500, 避开平衡段)
    template_incar : 可选, 用于单点的 INCAR 模板路径; 若为 None 则基于 run_dir/INCAR 改造

    Returns
    -------
    dict: {
        'out_dir': 验证根目录,
        'n_frames': 实际生成的帧数,
        'frame_steps': [每帧对应原轨迹的 MD step 索引],
        'frame_dirs': [每帧的绝对路径],
    }
    """
    from .vasp_io import read_poscar, write_poscar, read_incar, write_incar, write_kpoints_gamma

    run_dir = os.path.abspath(run_dir)
    xdatcar = os.path.join(run_dir, "XDATCAR")
    incar_src = os.path.join(run_dir, "INCAR")
    poscar_src = os.path.join(run_dir, "POSCAR")
    potcar_src = os.path.join(run_dir, "POTCAR")

    if not os.path.isfile(xdatcar):
        raise FileNotFoundError(f"XDATCAR 不存在: {xdatcar}")
    if not os.path.isfile(poscar_src):
        raise FileNotFoundError(f"POSCAR 不存在: {poscar_src}")

    # 读取参考 POSCAR 以获得元素/数目/晶格 (用于重建每帧 POSCAR)
    ref_poscar = read_poscar(poscar_src)

    # 读取 XDATCAR 帧 (使用 aimdpost 里的鲁棒解析器)
    from .aimdpost.aimd_postproc import read_xdatcar_frames
    frac_coords, lattice_xd, elems_xd, counts_xd = read_xdatcar_frames(xdatcar)
    nframes_total, natoms, _ = frac_coords.shape

    if nframes_total <= skip_initial:
        raise ValueError(
            f"XDATCAR 仅有 {nframes_total} 帧, 不足以跳过 {skip_initial} 步"
        )

    # 等间隔抽帧
    usable_start = skip_initial
    usable_end = nframes_total
    stride = max(1, (usable_end - usable_start) // n_frames)
    sampled_idx = list(range(usable_start, usable_end, stride))[:n_frames]

    # 构建 DFT 单点 INCAR (在原 INCAR 基础上修改)
    if template_incar and os.path.isfile(template_incar):
        params = read_incar(template_incar)
    elif os.path.isfile(incar_src):
        params = read_incar(incar_src)
    else:
        params = {}

    # DFT 单点必需参数 (覆盖任何已有值)
    dft_single_point = {
        "NSW": "0",
        "IBRION": "-1",
        "ML_LMLFF": ".FALSE.",
        "LWAVE": ".FALSE.",
        "LCHARG": ".FALSE.",
        "ISIF": "2",            # 需要应力
        "NELM": "200",
        "PREC": "Accurate",
        "LREAL": "Auto",
        "NCORE": "4",
        "ADDGRID": ".TRUE.",
    }
    # 移除所有 MLFF 相关键
    mlff_keys = [k for k in params if k.upper().startswith("ML_")]
    for k in mlff_keys:
        del params[k]
    params.update(dft_single_point)

    # 创建各帧目录
    os.makedirs(out_dir, exist_ok=True)
    frame_dirs = []
    for i, step in enumerate(sampled_idx):
        fdir = os.path.join(out_dir, f"frame_{i+1:04d}")
        os.makedirs(fdir, exist_ok=True)

        # 写 POSCAR: 使用此帧的分数坐标 + XDATCAR 的晶格
        frame_poscar = dict(ref_poscar)
        # XDATCAR 的 lattice 是固定的 (NVT), 与 POSCAR 一致
        frame_poscar["lattice"] = lattice_xd
        frame_poscar["coord_type"] = "Direct"
        frame_poscar["positions"] = frac_coords[step].copy()
        write_poscar(os.path.join(fdir, "POSCAR"), frame_poscar)

        # 写 INCAR
        write_incar(os.path.join(fdir, "INCAR"), params)

        # 写 KPOINTS (Gamma 点, 与原计算保持一致)
        write_kpoints_gamma(os.path.join(fdir, "KPOINTS"))

        # 复制 POTCAR
        if os.path.isfile(potcar_src):
            shutil.copy2(potcar_src, os.path.join(fdir, "POTCAR"))

        frame_dirs.append(fdir)

    # 生成 SLURM 提交脚本
    _write_validation_submit_script(out_dir, frame_dirs)

    # 保存 manifest (便于后续 collect_parity_data 无需重新指定 frame_steps)
    import json
    manifest_path = os.path.join(out_dir, "frame_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir,
            "n_frames": len(frame_dirs),
            "frame_steps": sampled_idx,
            "skip_initial": skip_initial,
        }, f, indent=2)

    return {
        "out_dir": os.path.abspath(out_dir),
        "n_frames": len(frame_dirs),
        "frame_steps": sampled_idx,
        "frame_dirs": frame_dirs,
    }


def _write_validation_submit_script(out_dir: str, frame_dirs: List[str]) -> None:
    """生成批量提交脚本 submit_validation.sh。"""
    path = os.path.join(out_dir, "submit_validation.sh")
    lines = ["#!/bin/bash", "# MHEC MLFF validation single-point jobs",
             "# 每帧一个独立的 DFT 单点计算 (NSW=0, ML_LMLFF=.FALSE.)", ""]
    for fdir in frame_dirs:
        rel = os.path.relpath(fdir, out_dir)
        lines.append(f"(cd {rel} && sbatch ../run_single.sh)")
    lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

    # 单任务模板 (用户按需修改)
    run_path = os.path.join(out_dir, "run_single.sh")
    run_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=mhec_val",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=64",
        "#SBATCH --partition=8358P",
        "#SBATCH --output=slurm-%j.out",
        "",
        "module load intel/2021.4",
        "mpirun -np $SLURM_NTASKS vasp_gam",
        "",
    ]
    with open(run_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(run_lines))
    try:
        os.chmod(run_path, 0o755)
    except Exception:
        pass


# ============================================================
# OUTCAR 解析 (能量, 力, 应力)
# ============================================================

def extract_single_point_outcar(outcar_path: str) -> Optional[Dict]:
    """
    从单点计算的 OUTCAR 中提取:
      - total_energy (eV): free  energy TOTEN
      - energy_per_atom (eV/atom)
      - forces (natoms, 3): 原子力 (eV/Å)
      - stress (6,) Voigt: σ_xx σ_yy σ_zz σ_yz σ_xz σ_xy  (kBar)
      - natoms

    如果 OUTCAR 不完整或计算未收敛, 返回 None。
    """
    if not os.path.isfile(outcar_path):
        return None
    with open(outcar_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # --- natoms ---
    m = re.search(r"NIONS\s*=\s*(\d+)", content)
    if not m:
        return None
    natoms = int(m.group(1))

    # --- Energy (取最后一次 "free  energy TOTEN") ---
    energies = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([\-\d.Ee+]+)", content)
    if not energies:
        return None
    total_energy = float(energies[-1])

    # --- 力 (取最后一次 "TOTAL-FORCE" 块) ---
    force_blocks = list(re.finditer(
        r"POSITION\s+TOTAL-FORCE[^\n]*\n\s*-+\n(.*?)\n\s*-+",
        content, re.DOTALL))
    if not force_blocks:
        return None
    block = force_blocks[-1].group(1)
    forces = []
    for line in block.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 6:
            # 格式: x y z Fx Fy Fz
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
            forces.append([fx, fy, fz])
    if len(forces) != natoms:
        return None
    forces = np.array(forces)

    # --- 应力 (取最后一次 "in kB" 行) ---
    stress_matches = re.findall(
        r"in kB\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
        r"\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
        content)
    if not stress_matches:
        return None
    last = stress_matches[-1]
    vals = [float(x) for x in last]
    # OUTCAR 顺序: XX YY ZZ XY YZ ZX
    # 转为 Voigt: xx, yy, zz, yz, xz, xy
    stress_voigt = np.array([
        vals[0], vals[1], vals[2],
        vals[4], vals[5], vals[3],
    ])

    return {
        "natoms": natoms,
        "total_energy": total_energy,
        "energy_per_atom": total_energy / natoms,
        "forces": forces,
        "stress": stress_voigt,
    }


def extract_mlff_trajectory(run_dir: str) -> Optional[Dict]:
    """
    从 MLFF run 目录的 OUTCAR 提取**每一步**的能量、力、应力。

    Returns
    -------
    dict 包含:
      'step_energies':  (nsteps,) eV
      'step_forces':    (nsteps, natoms, 3)  eV/Å
      'step_stresses':  (nsteps, 6)  kBar  (Voigt)
      'natoms':         int
    """
    outcar = os.path.join(run_dir, "OUTCAR")
    if not os.path.isfile(outcar):
        return None

    with open(outcar, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    m = re.search(r"NIONS\s*=\s*(\d+)", content)
    if not m:
        return None
    natoms = int(m.group(1))

    energies = [float(x) for x in re.findall(
        r"free\s+energy\s+TOTEN\s*=\s*([\-\d.Ee+]+)", content)]

    # 所有力块
    force_blocks = re.findall(
        r"POSITION\s+TOTAL-FORCE[^\n]*\n\s*-+\n(.*?)\n\s*-+",
        content, re.DOTALL)
    all_forces = []
    for blk in force_blocks:
        f_frame = []
        for line in blk.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 6:
                f_frame.append([float(parts[3]), float(parts[4]), float(parts[5])])
        if len(f_frame) == natoms:
            all_forces.append(f_frame)

    # 所有应力
    stress_matches = re.findall(
        r"in kB\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
        r"\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
        content)
    all_stresses = []
    for match in stress_matches:
        vals = [float(x) for x in match]
        all_stresses.append([vals[0], vals[1], vals[2],
                             vals[4], vals[5], vals[3]])

    # 取共同长度 (三者应等长, 若不等以最小为准)
    n_common = min(len(energies), len(all_forces), len(all_stresses))
    if n_common == 0:
        return None

    return {
        "natoms": natoms,
        "step_energies": np.array(energies[:n_common]),
        "step_forces":   np.array(all_forces[:n_common]),
        "step_stresses": np.array(all_stresses[:n_common]),
    }


# ============================================================
# 成对 (MLFF vs DFT) 数据收集
# ============================================================

class ParityData:
    """MLFF vs DFT 配对数据容器。"""

    def __init__(self):
        self.e_mlff: List[float] = []  # eV/atom
        self.e_dft:  List[float] = []
        self.f_mlff: List[float] = []  # 每原子分量展开的力 (eV/Å)
        self.f_dft:  List[float] = []
        self.s_mlff: List[float] = []  # 6 个 Voigt 分量 (kBar)
        self.s_dft:  List[float] = []
        self.n_frames = 0

    def add_frame(self, mlff_e: float, mlff_f: np.ndarray, mlff_s: np.ndarray,
                  dft_e: float, dft_f: np.ndarray, dft_s: np.ndarray,
                  natoms: int):
        self.e_mlff.append(mlff_e)
        self.e_dft.append(dft_e)
        self.f_mlff.extend(mlff_f.flatten().tolist())
        self.f_dft.extend(dft_f.flatten().tolist())
        self.s_mlff.extend(mlff_s.tolist())
        self.s_dft.extend(dft_s.tolist())
        self.n_frames += 1

    def metrics(self) -> Dict:
        """计算 RMSE, MAE, R² 对三个分量。"""
        out = {}
        for name, x_dft_list, x_mlff_list, unit in [
            ("energy", self.e_dft, self.e_mlff, "eV/atom"),
            ("force",  self.f_dft, self.f_mlff, "eV/Å"),
            ("stress", self.s_dft, self.s_mlff, "kBar"),
        ]:
            x_dft = np.array(x_dft_list)
            x_mlff = np.array(x_mlff_list)
            if len(x_dft) == 0:
                continue
            diff = x_mlff - x_dft
            rmse = float(np.sqrt(np.mean(diff**2)))
            mae = float(np.mean(np.abs(diff)))
            if np.std(x_dft) > 1e-12:
                ss_tot = np.sum((x_dft - np.mean(x_dft))**2)
                ss_res = np.sum(diff**2)
                r2 = 1.0 - ss_res / ss_tot
            else:
                r2 = float("nan")
            out[name] = {"rmse": rmse, "mae": mae, "r2": r2,
                         "n": len(x_dft), "unit": unit}
        return out


def collect_parity_data(
    validation_dir: str,
    mlff_run_dir: str,
    frame_steps: Optional[List[int]] = None,
) -> ParityData:
    """
    收集已完成的验证目录下所有帧的 DFT 结果, 与 MLFF run 对应步的结果配对。

    Parameters
    ----------
    validation_dir : generate_validation_dirs 创建的根目录
    mlff_run_dir   : 原 MLFF run 目录 (提供 MLFF 预测的应力/力/能量)
    frame_steps    : 每个帧对应的 MD step 索引 (来自 generate_validation_dirs 的返回),
                     若为 None 则尝试从 validation_dir/frame_manifest.json 读取

    Returns
    -------
    ParityData
    """
    import json

    # 如果没传 frame_steps, 尝试从 manifest 读取
    manifest = os.path.join(validation_dir, "frame_manifest.json")
    if frame_steps is None and os.path.isfile(manifest):
        with open(manifest, "r") as f:
            frame_steps = json.load(f)["frame_steps"]

    if frame_steps is None:
        raise ValueError(
            "必须提供 frame_steps, 或在 validation_dir 下保留 frame_manifest.json")

    # 读 MLFF 轨迹 (一次性, 避免多次打开 OUTCAR)
    mlff = extract_mlff_trajectory(mlff_run_dir)
    if mlff is None:
        raise RuntimeError(f"无法从 {mlff_run_dir}/OUTCAR 读取 MLFF 轨迹")

    natoms = mlff["natoms"]
    parity = ParityData()

    for i, step in enumerate(frame_steps):
        frame_dir = os.path.join(validation_dir, f"frame_{i+1:04d}")
        outcar = os.path.join(frame_dir, "OUTCAR")
        dft = extract_single_point_outcar(outcar)
        if dft is None:
            # 该帧还未完成或失败, 跳过
            continue
        if dft["natoms"] != natoms:
            continue

        # 取 MLFF 在同一步的预测
        if step >= len(mlff["step_energies"]):
            continue

        parity.add_frame(
            mlff_e=mlff["step_energies"][step] / natoms,
            mlff_f=mlff["step_forces"][step],
            mlff_s=mlff["step_stresses"][step],
            dft_e=dft["energy_per_atom"],
            dft_f=dft["forces"],
            dft_s=dft["stress"],
            natoms=natoms,
        )

    return parity


# ============================================================
# Parity Plots (3 张独立图)
# ============================================================

def plot_parity(parity: ParityData, out_dir: str) -> Dict[str, str]:
    """
    生成三张 parity 图:
      energy_parity.svg, force_parity.svg, stress_parity.svg
    全部 300 dpi, SVG 矢量 + PNG, 可直接用于论文 / 报告。

    Returns
    -------
    dict: {quantity: svg_path}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    metrics = parity.metrics()

    cfg = [
        ("energy", parity.e_dft, parity.e_mlff,
         "DFT energy (eV/atom)", "MLFF energy (eV/atom)", "#4C72B0"),
        ("force", parity.f_dft, parity.f_mlff,
         "DFT force component (eV/Å)", "MLFF force component (eV/Å)", "#55A868"),
        ("stress", parity.s_dft, parity.s_mlff,
         "DFT stress (kBar)", "MLFF stress (kBar)", "#DD8452"),
    ]

    out_paths = {}
    for name, xd, ym, xlabel, ylabel, color in cfg:
        if len(xd) == 0:
            continue
        x = np.array(xd); y = np.array(ym)

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
        # parity diagonal
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        lo -= pad; hi += pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")
        # linear fit
        if np.std(x) > 1e-12:
            coef = np.polyfit(x, y, 1)
            xf = np.linspace(lo, hi, 100)
            ax.plot(xf, np.polyval(coef, xf), "-", color=color,
                    lw=1.2, alpha=0.45,
                    label=f"fit: slope = {coef[0]:.4f}")
        # scatter
        ax.scatter(x, y, s=16, alpha=0.65, color=color,
                   edgecolors="white", linewidth=0.4, zorder=5,
                   label=f"n = {len(x)}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)

        # metrics box
        m = metrics.get(name, {})
        txt = (f"RMSE = {m.get('rmse', 0):.4f} {m.get('unit', '')}\n"
               f"MAE  = {m.get('mae', 0):.4f} {m.get('unit', '')}\n"
               f"R²   = {m.get('r2', 0):.4f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                fontsize=9, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#BBB", linewidth=0.5, alpha=0.95))
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_title(f"MLFF vs DFT parity — {name}", fontsize=10, pad=6)

        svg_path = os.path.join(out_dir, f"{name}_parity.svg")
        png_path = os.path.join(out_dir, f"{name}_parity.png")
        fig.tight_layout()
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths[name] = svg_path

    return out_paths


# ============================================================
# 报告输出
# ============================================================

def write_parity_report(parity: ParityData, out_dir: str,
                        plot_paths: Optional[Dict[str, str]] = None) -> str:
    """生成 parity_report.txt, 包含三个量的统计信息和评级。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "parity_report.txt")
    metrics = parity.metrics()

    lines = []
    lines.append("=" * 64)
    lines.append("MHEC MLFF 验证报告 (DFT vs MLFF 单点抽样)")
    lines.append("=" * 64)
    lines.append(f"  有效帧数: {parity.n_frames}")
    lines.append("")
    lines.append(f"  {'物理量':>8}  {'RMSE':>12}  {'MAE':>12}  {'R²':>8}  {'N':>8}  单位")
    lines.append("  " + "─" * 62)

    thresh = {
        "energy": [(0.001, "优秀"), (0.005, "良好"), (0.010, "可接受")],
        "force":  [(0.05,  "优秀"), (0.10,  "良好"), (0.20,  "可接受")],
        "stress": [(2.0,   "优秀"), (5.0,   "良好"), (10.0,  "可接受")],
    }
    name_zh = {"energy": "能量", "force": "力", "stress": "应力"}

    overall_grades = []
    for name in ["energy", "force", "stress"]:
        if name not in metrics:
            continue
        m = metrics[name]
        lines.append(
            f"  {name_zh[name]:>8}  {m['rmse']:>12.4f}  {m['mae']:>12.4f}  "
            f"{m['r2']:>8.4f}  {m['n']:>8}  {m['unit']}"
        )
        # 分级
        rmse = m["rmse"]
        grade = "需要改进"
        for thr, g in thresh[name]:
            if rmse < thr:
                grade = g; break
        overall_grades.append(grade)
    lines.append("")

    # 综合评级
    if overall_grades:
        levels = {"优秀": 0, "良好": 1, "可接受": 2, "需要改进": 3}
        worst_idx = max(levels[g] for g in overall_grades)
        worst_name = [k for k, v in levels.items() if v == worst_idx][0]
        lines.append(f"  综合评级: {worst_name}")

        # 对 C_ij 的间接影响估计 (经验关系: ΔC ≈ 3-5 × stress RMSE)
        if "stress" in metrics:
            s_rmse = metrics["stress"]["rmse"]
            # 粗略估计: 单点应力 RMSE 通过拟合衰减 ~ sqrt(n_points), 但系统偏差不衰减
            # 保守给出 C_ij 量级不确定度
            cij_sigma = s_rmse * 0.1  # kBar → GPa
            lines.append("")
            lines.append(f"  对弹性常数 Cᵢⱼ 的影响估计:")
            lines.append(f"    应力 RMSE = {s_rmse:.3f} kBar")
            lines.append(f"    量级 σ(Cᵢⱼ) ≈ {cij_sigma:.2f} GPa (95% 置信区间约 ±{2*cij_sigma:.2f} GPa)")

    lines.append("")
    if plot_paths:
        lines.append("  生成的 parity 图:")
        for name, p in plot_paths.items():
            lines.append(f"    {name}: {p}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 同时输出纯数据 CSV (便于用户自己绘图)
    csv_path = os.path.join(out_dir, "parity_data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("# frame,quantity,dft_value,mlff_value,unit\n")
        for i, (xd, xm) in enumerate(zip(parity.e_dft, parity.e_mlff)):
            f.write(f"{i+1},energy,{xd:.8f},{xm:.8f},eV/atom\n")
        for i, (xd, xm) in enumerate(zip(parity.f_dft, parity.f_mlff)):
            f.write(f"{i+1},force,{xd:.6f},{xm:.6f},eV/A\n")
        for i, (xd, xm) in enumerate(zip(parity.s_dft, parity.s_mlff)):
            f.write(f"{i+1},stress,{xd:.4f},{xm:.4f},kBar\n")

    return path
