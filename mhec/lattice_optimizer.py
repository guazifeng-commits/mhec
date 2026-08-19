"""
温度依赖晶格优化模块。

通过 NPT 系综 AIMD 获取目标温度下的平衡晶格参数。
支持两种模式:
  1. 三步模式 (train → refit → run): 从头训练 MLFF
  2. run-only 模式: 使用预训练的 ML_FF 直接 run
"""

import os
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from .vasp_io import read_poscar, write_poscar, write_kpoints_gamma
from .incar_templates import build_incar_params
from .crystal_system import lattice_params
from .thermal_expansion import (
    LatticeAtTemp, compute_thermal_expansion,
    print_thermal_expansion_report, save_thermal_expansion, plot_thermal_expansion,
)


class LatticeOptimizer:
    """温度依赖晶格优化模块。"""

    def __init__(
        self,
        poscar: Dict,
        temperatures: List[float],
        nsw_train: int = 2000,
        nsw_refit: int = 1,      # ML_MODE=refit 为静态重拟合, 不跑 MD, NSW=1
        nsw_run: int = 10000,
        potim: float = 1.0,
        encut: Optional[float] = None,
        user_overrides: Optional[Dict[str, str]] = None,
        skip_steps: int = 500,
        crystal_system=None,
    ):
        self.poscar = poscar
        self.temperatures = temperatures
        self.nsw_train = nsw_train
        self.nsw_refit = nsw_refit
        self.nsw_run = nsw_run
        self.potim = potim
        self.encut = encut
        self.user_overrides = user_overrides or {}
        self.skip_steps = skip_steps
        # 已知晶系 (来自空间群/配置/输入结构), 用于约束 NPT 平均晶格的对称性。
        # 绝不从带噪声的时间平均晶格重新识别 (否则立方会被误判为四方)。
        self.crystal_system = crystal_system

    def setup_npt_dirs(self, base_dir: str, potcar_path: str = None,
                       submit_template: str = None,
                       mlff_path: str = None,
                       run_only: bool = False,
                       slurm_config=None) -> Dict[float, str]:
        """为每个温度创建 NPT 优化目录。

        Parameters
        ----------
        base_dir : 基础目录
        potcar_path : POTCAR 文件路径
        submit_template : 提交脚本模板内容
        mlff_path : 预训练 ML_FF 文件路径 (run-only 模式)
        run_only : 是否只生成 run 目录 (跳过 train/refit)
        slurm_config : SlurmConfig 实例

        Returns
        -------
        {温度: 目录路径}
        """
        from .slurm import SlurmConfig, generate_run_slurm, copy_potcar_to_dirs

        if slurm_config is None:
            slurm_config = SlurmConfig()

        result = {}
        all_calc_dirs = []

        stages = ["run"] if run_only else ["train", "refit", "run"]

        for temp in self.temperatures:
            temp_dir = os.path.join(base_dir, f"npt_{int(temp)}K")

            for stage in stages:
                stage_dir = os.path.join(temp_dir, stage)
                os.makedirs(stage_dir, exist_ok=True)

                # POSCAR
                write_poscar(os.path.join(stage_dir, "POSCAR"), self.poscar)

                # INCAR
                nsw = getattr(self, f"nsw_{stage}", self.nsw_run)
                params = build_incar_params(
                    ensemble="npt", mlff_stage=stage, temperature=temp,
                    nsw=nsw, potim=self.potim, encut=self.encut,
                    user_overrides=self.user_overrides,
                )
                from .vasp_io import write_incar
                write_incar(os.path.join(stage_dir, "INCAR"), params)

                # KPOINTS
                write_kpoints_gamma(os.path.join(stage_dir, "KPOINTS"))

                # ML_FF (run-only 模式)
                if mlff_path and os.path.isfile(mlff_path):
                    dst = os.path.join(stage_dir, "ML_FF")
                    if not os.path.isfile(dst):
                        shutil.copy2(mlff_path, dst)

                # run.sh
                slurm_path = os.path.join(stage_dir, "run.sh")
                if submit_template:
                    with open(slurm_path, 'w') as f:
                        f.write(submit_template)
                else:
                    with open(slurm_path, 'w') as f:
                        f.write(generate_run_slurm(slurm_config))

                all_calc_dirs.append(stage_dir)

            result[temp] = temp_dir

        # 复制 POTCAR
        if potcar_path and os.path.isfile(potcar_path):
            n = copy_potcar_to_dirs(potcar_path, all_calc_dirs)
            if n > 0:
                print(f"  +-> POTCAR 已复制到 {n} 个 NPT 目录")

        # 生成 submit_npt.sh
        self._write_npt_submit_script(base_dir, stages)

        return result

    def _write_npt_submit_script(self, base_dir: str, stages: List[str]) -> None:
        """生成 NPT 批量提交脚本。支持 sbatch 和 bash 两种用法。"""
        lines = [
            "#!/bin/bash",
            "#SBATCH -N 1",
            "#SBATCH -n 1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --partition=8358P",
            "#SBATCH --job-name=mhec_npt",
            "#SBATCH --output=submit_npt_%j.out",
            "#SBATCH --error=submit_npt_%j.err",
            "",
            "# MHEC NPT 晶格优化批量提交脚本",
            f"# 阶段: {' → '.join(stages)}",
            "# 用法: sbatch submit_npt.sh  或  bash submit_npt.sh",
            "",
            'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
            '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
            'else',
            '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
            'fi',
            "",
        ]
        for temp in self.temperatures:
            temp_dir = f"npt_{int(temp)}K"
            lines.append(f"echo '>>> {int(temp)} K NPT 优化'")
            prev_var = None
            for stage in stages:
                rel = f"{temp_dir}/{stage}"
                lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')
                if prev_var:
                    lines.append(
                        f'JOB_ID=$(sbatch --parsable --dependency=afterok:${prev_var} run.sh)')
                else:
                    lines.append('JOB_ID=$(sbatch --parsable run.sh)')
                lines.append(f'echo "  {rel}: $JOB_ID"')
                var_name = f'{stage.upper()}_{int(temp)}'
                lines.append(f'{var_name}=$JOB_ID')
                lines.append("popd > /dev/null")
                prev_var = var_name
            lines.append("")

        lines.append('echo ">>> 所有 NPT 任务已提交。"')
        path = os.path.join(base_dir, "submit_npt.sh")
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")

    def extract_equilibrium_lattice(
        self,
        temp_dir: str,
        skip_frac: float = 0.5,
        skip_steps: Optional[int] = None,
        run_only: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        从 NPT AIMD 轨迹提取平衡晶格。

        提取策略（按优先级）：
        1. vasprun.xml: 从所有离子步提取晶格，时间平均（最精确）
        2. XDATCAR: 从轨迹帧提取晶格（如果 vasprun 无晶格数据）
        3. CONTCAR: 直接读取最终结构（最简单，无统计信息）

        Parameters
        ----------
        temp_dir : 温度目录 (如 phase1_npt/npt_300K)
        skip_frac : 丢弃初始比例
        skip_steps : 丢弃初始步数 (优先于 skip_frac)
        run_only : 是否只有 run 目录

        Returns
        -------
        (平均晶格矩阵, 统计信息字典)
        """
        # 自动探测文件位置: 优先 temp_dir/ 其次 temp_dir/run/
        def _find_file(filename):
            for d in [temp_dir, os.path.join(temp_dir, "run")]:
                p = os.path.join(d, filename)
                if os.path.isfile(p):
                    return p
            return None

        # ---- 策略1: 从 vasprun.xml 提取时间平均晶格 ----
        vasprun_path = _find_file("vasprun.xml")
        if vasprun_path:
            try:
                lattices = self._parse_lattices_from_vasprun(vasprun_path)
                if len(lattices) > 0:
                    return self._average_lattices(lattices, skip_frac, skip_steps,
                                                   source=f"vasprun.xml ({len(lattices)} frames)")
                else:
                    print(f"    vasprun.xml 存在但未解析到晶格数据，尝试 CONTCAR...")
            except Exception as e:
                print(f"    vasprun.xml 解析异常: {e}，尝试 CONTCAR...")

        # ---- 策略2: 从 CONTCAR 直接读取最终结构 ----
        contcar_path = _find_file("CONTCAR")
        if contcar_path:
            poscar = read_poscar(contcar_path)
            lattice = self._enforce_symmetry(poscar["lattice"])
            a, b, c, alpha, beta, gamma = lattice_params(lattice)
            volume = abs(np.linalg.det(lattice))
            stats = {
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "volume": volume,
                "n_frames": 1,
                "n_total": 1,
                "skip": 0,
                "source": f"CONTCAR (最终结构)",
            }
            return lattice, stats

        # ---- 策略3: 从 POSCAR 读取（最后回退） ----
        poscar_path = _find_file("POSCAR")
        if poscar_path:
            poscar = read_poscar(poscar_path)
            lattice = poscar["lattice"]
            a, b, c, alpha, beta, gamma = lattice_params(lattice)
            volume = abs(np.linalg.det(lattice))
            stats = {
                "a": a, "b": b, "c": c,
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "volume": volume,
                "n_frames": 1,
                "n_total": 1,
                "skip": 0,
                "source": f"POSCAR (初始结构, 未优化)",
            }
            return lattice, stats

        raise FileNotFoundError(
            f"在 {temp_dir} 中未找到 vasprun.xml、CONTCAR 或 POSCAR")

    def _parse_lattices_from_vasprun(self, vasprun_path: str) -> np.ndarray:
        """从 vasprun.xml 解析所有离子步的晶格矩阵。
        
        支持两种 vasprun.xml 格式:
        1. 标准 DFT: <calculation><structure><crystal><varray name="basis">
        2. MLFF run: <structure><crystal><varray name="basis"> (无 calculation 包裹)
        
        跳过 name="primitive_cell" 和 name="initialpos" 的初始结构。
        """
        from xml.etree import ElementTree as ET

        lattices = []
        in_structure = False
        in_crystal = False
        in_basis = False
        basis_rows = []
        structure_name = None
        n_skipped = 0

        try:
            for event, elem in ET.iterparse(vasprun_path, events=('start', 'end')):
                if event == 'start':
                    if elem.tag == 'structure':
                        in_structure = True
                        structure_name = elem.get('name', '')
                    elif elem.tag == 'crystal' and in_structure:
                        in_crystal = True
                    elif (elem.tag == 'varray' and elem.get('name') == 'basis'
                          and in_crystal):
                        in_basis = True
                        basis_rows = []

                elif event == 'end':
                    if elem.tag == 'v' and in_basis:
                        if elem.text:
                            try:
                                row = [float(x) for x in elem.text.split()]
                                if len(row) == 3:
                                    basis_rows.append(row)
                            except ValueError:
                                pass
                    elif elem.tag == 'varray' and in_basis:
                        in_basis = False
                        if len(basis_rows) == 3:
                            # 跳过初始结构 (primitive_cell, initialpos)
                            if structure_name in ('primitive_cell', 'initialpos'):
                                n_skipped += 1
                            else:
                                lattices.append(np.array(basis_rows))
                    elif elem.tag == 'crystal':
                        in_crystal = False
                    elif elem.tag == 'structure':
                        in_structure = False
                        structure_name = None
                        elem.clear()  # 释放内存
                    elif elem.tag == 'calculation':
                        elem.clear()

        except ET.ParseError as e:
            print(f"    警告: vasprun.xml 解析错误 (可能文件不完整): {e}")

        if lattices:
            print(f"    vasprun.xml: 解析到 {len(lattices)} 个晶格帧 (跳过 {n_skipped} 个初始结构)")
        return np.array(lattices) if lattices else np.array([])

    def _average_lattices(self, lattices: np.ndarray, skip_frac: float,
                          skip_steps: Optional[int], source: str = "") -> Tuple[np.ndarray, Dict]:
        """对晶格序列做时间平均，并根据晶系对称性约束晶格参数。"""
        n_total = len(lattices)
        if skip_steps is not None:
            skip = min(skip_steps, n_total - 1)
        else:
            skip = int(n_total * skip_frac)

        eq_lattices = lattices[skip:]
        avg_lattice = np.mean(eq_lattices, axis=0)

        # 根据晶系对称性约束晶格
        avg_lattice = self._enforce_symmetry(avg_lattice)

        a, b, c, alpha, beta, gamma = lattice_params(avg_lattice)
        volume = abs(np.linalg.det(avg_lattice))

        stats = {
            "a": a, "b": b, "c": c,
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "volume": volume,
            "n_frames": len(eq_lattices),
            "n_total": n_total,
            "skip": skip,
            "source": source,
        }
        return avg_lattice, stats

    def _enforce_symmetry(self, lattice: np.ndarray) -> np.ndarray:
        """根据晶系对称性约束晶格矩阵 (使用已知晶系, 不从噪声晶格重新识别)。"""
        from .crystal_system import enforce_lattice_symmetry
        return enforce_lattice_symmetry(lattice, crystal_system=self.crystal_system)

    def use_provided_poscar(self, temperature: float, poscar_path: str) -> Dict:
        """用户直接提供平衡结构，跳过 NPT 优化。"""
        return read_poscar(poscar_path)

    def extract_all_equilibrium(
        self,
        base_dir: str,
        skip_frac: float = 0.5,
        run_only: bool = False,
    ) -> Tuple[Dict[float, Tuple[np.ndarray, Dict]], Optional["ThermalExpansionResult"]]:
        """
        从所有温度的 NPT 计算中提取平衡晶格，并计算热膨胀系数。

        Parameters
        ----------
        base_dir : NPT 工作目录
        skip_frac : 丢弃初始比例 (默认 0.5)
        run_only : 是否只有 run 目录

        Returns
        -------
        (lattice_dict, thermal_result)
        lattice_dict: {温度: (平均晶格矩阵, 统计信息)}
        thermal_result: ThermalExpansionResult 或 None
        """
        lattice_data = {}
        te_data = []

        stage = "run"
        for temp in self.temperatures:
            temp_dir = os.path.join(base_dir, f"npt_{int(temp)}K")
            run_dir = os.path.join(temp_dir, stage)

            try:
                avg_lattice, stats = self.extract_equilibrium_lattice(
                    temp_dir, skip_frac=skip_frac, run_only=run_only)
                lattice_data[temp] = (avg_lattice, stats)

                te_data.append(LatticeAtTemp(
                    temperature=temp,
                    a=stats["a"], b=stats["b"], c=stats["c"],
                    alpha_deg=stats["alpha"], beta_deg=stats["beta"],
                    gamma_deg=stats["gamma"],
                    volume=stats["volume"],
                    lattice=avg_lattice,
                    n_frames=stats["n_frames"],
                ))
                src = stats.get('source', f"{stats['n_frames']} frames")
                print(f"  +-> {int(temp)} K: a={stats['a']:.4f} b={stats['b']:.4f} "
                      f"c={stats['c']:.4f} V={stats['volume']:.2f} ({src})")
            except Exception as e:
                print(f"  !-> {int(temp)} K: 提取失败 — {e}")

        # 计算热膨胀系数
        te_result = None
        if len(te_data) >= 2:
            try:
                te_result = compute_thermal_expansion(te_data)
                print_thermal_expansion_report(te_result)
                save_thermal_expansion(te_result, os.path.join(base_dir, "thermal"))
                plot_thermal_expansion(te_result, os.path.join(base_dir, "thermal"))
            except Exception as e:
                print(f"  !-> 热膨胀系数计算失败: {e}")

        return lattice_data, te_result
