"""
插件适配器：将 elasticpost 和 aimdpost 子包桥接到 MHEC 插件接口。
"""

import os
import sys
import numpy as np
from typing import Dict

from .plugins import MechanicalPropertiesPlugin, AIMDPostProcessorPlugin
from .i18n import T


class ElasticPostAdapter(MechanicalPropertiesPlugin):
    """
    将 elasticpost 子包适配为 MechanicalPropertiesPlugin。

    读取弹性矩阵后，调用 elasticpost 的完整分析流程：
    VRH 平均、各向异性指数、方向性性质、稳定性判据、热力学性质等。
    """

    def __init__(self, density: float = 5.0, molar_mass: float = 50.0,
                 output_dir: str = "elasticpost_results", auto_mode: bool = True):
        self.density = density
        self.molar_mass = molar_mass
        self.output_dir = output_dir
        self.auto_mode = auto_mode

    def calculate(self, cij_matrix: np.ndarray, crystal_system: str) -> Dict:
        """
        计算力学性质。

        Parameters
        ----------
        cij_matrix : (6, 6) 弹性常数矩阵 (GPa)
        crystal_system : 晶系名称字符串

        Returns
        -------
        dict : 力学性质键值对
        """
        from .elasticpost.elastic_tools import (
            identify_symmetry,
            stability_criteria,
            vrh_averages,
            anisotropy_indices_from_vrh,
            compliance_matrix,
            calculate_thermodynamic_properties,
            calculate_sound_velocities,
            calculate_polycrystalline_moduli,
            calculate_polycrystalline_anisotropy,
            report,
        )

        results = {}

        try:
            # 对称性识别
            sym = identify_symmetry(cij_matrix)
            results["symmetry"] = sym.system

            # 稳定性判据
            stab = stability_criteria(cij_matrix, sym.system)
            results["stable"] = stab.stable

            # VRH 平均 (vrh_averages 返回 dict)
            vrh = vrh_averages(cij_matrix)
            K_V = vrh["bulk_modulus_voigt"]
            K_R = vrh["bulk_modulus_reuss"]
            K_H = vrh["bulk_modulus_vrh"]
            G_V = vrh["shear_modulus_voigt"]
            G_R = vrh["shear_modulus_reuss"]
            G_H = vrh["shear_modulus_vrh"]
            denom = 3 * K_H + G_H
            if denom > 1e-30:
                E_H = 9 * K_H * G_H / denom
                nu_H = (3 * K_H - 2 * G_H) / (2 * denom)
            else:
                E_H = 0.0
                nu_H = 0.5  # 不可压缩极限

            results["bulk_modulus_voigt"] = K_V
            results["bulk_modulus_reuss"] = K_R
            results["bulk_modulus_hill"] = K_H
            results["shear_modulus_voigt"] = G_V
            results["shear_modulus_reuss"] = G_R
            results["shear_modulus_hill"] = G_H
            results["youngs_modulus_hill"] = E_H
            results["poisson_ratio_hill"] = nu_H

            # 各向异性指数
            aniso = anisotropy_indices_from_vrh(vrh)
            results["anisotropy_universal"] = aniso["A_U"]
            # Zener 各向异性指数 A_Z = 2*C44/(C11-C12)，仅立方体系有意义
            if sym.system == "Cubic":
                try:
                    A_Z = 2 * cij_matrix[3, 3] / (cij_matrix[0, 0] - cij_matrix[0, 1])
                    results["anisotropy_zener"] = A_Z
                except (ZeroDivisionError, IndexError):
                    results["anisotropy_zener"] = None
            else:
                results["anisotropy_zener"] = None

            # 多晶模量
            poly = calculate_polycrystalline_moduli(cij_matrix)
            results["polycrystalline"] = poly

            # 多晶各向异性
            poly_aniso = calculate_polycrystalline_anisotropy(poly)
            results["polycrystalline_anisotropy"] = poly_aniso

            # 柔度矩阵
            S = compliance_matrix(cij_matrix)
            results["compliance_matrix"] = S

            # 热力学性质
            thermo = calculate_thermodynamic_properties(
                cij_matrix, self.density, self.molar_mass
            )
            results["thermodynamic"] = thermo

            # 声速
            direction = np.array([1.0, 0.0, 0.0])
            sound = calculate_sound_velocities(cij_matrix, self.density, direction)
            results["sound_velocities_100"] = sound

            # 生成报告
            poly_aniso_for_report = calculate_polycrystalline_anisotropy(poly)
            report_str = report(
                sym, stab, vrh, poly, poly_aniso_for_report,
                thermo, sound,
            )
            results["report"] = report_str

            print(T(f"力学性质计算完成 (晶系: {sym.system})",
                    f"Mechanical-property calculation done (crystal system: {sym.system})"))

        except Exception as e:
            print(T(f"力学性质计算出错: {e}", f"Mechanical-property calculation error: {e}"))
            import traceback
            traceback.print_exc()

        return results

    def run_full_analysis(self, matrix_file: str = "elastic.txt",
                          output_dir: str = None) -> Dict:
        """
        运行 elasticpost 完整分析流程（含绘图和文件输出）。

        Parameters
        ----------
        matrix_file : 弹性矩阵文件路径
        output_dir : 输出目录

        Returns
        -------
        dict : 分析结果
        """
        from .elasticpost.elastic_calculator import main as run_elastic

        out = output_dir or self.output_dir
        try:
            run_elastic(matrix_file, out, auto_mode=self.auto_mode)
            return {"status": "success", "output_dir": out}
        except Exception as e:
            print(T(f"elasticpost 完整分析出错: {e}", f"elasticpost full-analysis error: {e}"))
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e), "output_dir": out}


class AIMDPostAdapter(AIMDPostProcessorPlugin):
    """
    将 aimdpost 子包适配为 AIMDPostProcessorPlugin。

    在指定计算目录中运行完整的 AIMD 后处理分析，
    默认开启所有高级功能（partial RDF、结构因子、配位分析、
    MSD 高阶矩、键长键角分析、bootstrap 置信区间）。
    """

    def __init__(self, temperature: float = 300.0, potim: float = 1.0,
                 n_blocks: int = 5, stable_frac: float = 0.3,
                 out_prefix: str = "aimd_postproc", out_subdir: str = "aimdpost_results"):
        self.temperature = temperature
        self.potim = potim
        self.n_blocks = n_blocks
        self.stable_frac = stable_frac
        self.out_prefix = out_prefix
        self.out_subdir = out_subdir

    def process(self, work_dir: str, strain_label: str,
                mlff_stage: str, **kwargs) -> Dict:
        """
        执行完整 AIMD 后处理分析（含所有高级功能）。

        Parameters
        ----------
        work_dir : 包含 VASP 输出文件的计算目录
        strain_label : 应变标签
        mlff_stage : MLFF 阶段
        **kwargs : 可选参数覆盖
            temperature, potim, n_blocks, stable_frac,
            r_eff, tmax_ps, no_plots,
            fit_start_frac, fit_end_frac,
            friction_U, friction_p, friction_h,
            qmax, bond_cutoff

        Returns
        -------
        dict : 后处理结果
        """
        from .aimdpost.aimd_postproc import build_parser, main as aimd_main

        prefix = f"{self.out_prefix}_{strain_label}_{mlff_stage}"

        # 保存当前目录，切换到工作目录执行
        original_dir = os.getcwd()
        try:
            os.chdir(work_dir)

            # 所有输出归入专用结果子文件夹 (aimdpost_results)
            out_subdir = kwargs.get("out_subdir", self.out_subdir)
            os.makedirs(out_subdir, exist_ok=True)
            prefix_path = os.path.join(out_subdir, prefix)

            # 构建命令行参数（使用相对路径）
            local_args = [
                "--xdatcar", "XDATCAR",
                "--outcar", "OUTCAR",
                "--poscar", "POSCAR",
                "--T", str(kwargs.get("temperature", self.temperature)),
                "--potim", str(kwargs.get("potim", self.potim)),
                "--n_blocks", str(kwargs.get("n_blocks", self.n_blocks)),
                "--stable_frac", str(kwargs.get("stable_frac", self.stable_frac)),
                "--out_prefix", prefix_path,
                # 默认开启所有高级分析
                "--advanced_analysis",
                "--bootstrap",
            ]

            # 可选参数
            if kwargs.get("r_eff"):
                local_args.extend(["--r_eff", str(kwargs["r_eff"])])
            if kwargs.get("tmax_ps"):
                local_args.extend(["--truncation", "tmax",
                                   "--tmax_ps", str(kwargs["tmax_ps"])])
            if kwargs.get("fit_start_frac"):
                local_args.extend(["--fit_start_frac", str(kwargs["fit_start_frac"])])
            if kwargs.get("fit_end_frac"):
                local_args.extend(["--fit_end_frac", str(kwargs["fit_end_frac"])])
            if kwargs.get("no_plots"):
                local_args.append("--no_plots")
            if kwargs.get("qmax"):
                local_args.extend(["--qmax", str(kwargs["qmax"])])
            if kwargs.get("bond_cutoff"):
                local_args.extend(["--bond_cutoff", str(kwargs["bond_cutoff"])])

            # 摩擦系数参数
            if kwargs.get("friction_U"):
                local_args.extend(["--friction_U", str(kwargs["friction_U"])])
            if kwargs.get("friction_p"):
                local_args.extend(["--friction_p", str(kwargs["friction_p"])])
            if kwargs.get("friction_h"):
                local_args.extend(["--friction_h", str(kwargs["friction_h"])])

            # 通过 sys.argv 模拟运行
            old_argv = sys.argv
            sys.argv = ["aimd_postproc"] + local_args
            try:
                aimd_main()
                return {
                    "status": "success",
                    "work_dir": work_dir,
                    "strain_label": strain_label,
                    "mlff_stage": mlff_stage,
                    "output_prefix": prefix,
                }
            finally:
                sys.argv = old_argv
        except SystemExit:
            # argparse 可能调用 sys.exit
            return {
                "status": "completed",
                "work_dir": work_dir,
                "strain_label": strain_label,
                "mlff_stage": mlff_stage,
                "output_prefix": prefix,
            }
        except Exception as e:
            print(T(f"AIMD 后处理出错: {e}", f"AIMD post-processing error: {e}"))
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "work_dir": work_dir,
            }
        finally:
            os.chdir(original_dir)
