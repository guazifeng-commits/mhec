"""
体积守恒 (VC) 变形的弹性常数拟合。

思路:
  1. 每个 VC 模式有若干幅度点, 用 ElasticFitter.fit_single_mode 拟合得到
     斜率向量 slope = dσ/d(amplitude) (GPa, 已含 kBar→GPa 与符号修正)。
     由线性弹性 σ = C ε, 对方向 d (单位幅度应变) 有 slope = C @ d。
  2. 用晶系对称性把 6×6 的 C 参数化为少数独立常数 (复用 symmetry.py 规则),
     对 {slope_k = C @ d_k} 做线性最小二乘求解这些独立常数。
  3. 再走 ElasticFitter.symmetrize 做一次规范化 + Born 稳定性检查。

该求解器对所有晶系通用 (只要给定足够的 VC 模式张成独立常数空间)。
"""

import numpy as np
from typing import Dict, List, Tuple
from .crystal_system import CrystalSystem
from .elastic_fitter import ElasticFitter, solve_cij_weighted, _quality_weights
from .strain_vc import get_vc_modes, vc_mode_direction


class VCElasticFitter:
    """体积守恒变形弹性常数拟合器。"""

    def __init__(self, crystal_system: CrystalSystem):
        self.crystal_system = crystal_system
        self._base = ElasticFitter(crystal_system)

    def fit_all(
        self,
        all_strains: Dict[str, np.ndarray],
        all_stresses: Dict[str, np.ndarray],
        baseline_stress: np.ndarray,
        fit_order: int = 3,
    ) -> Dict:
        """
        Parameters
        ----------
        all_strains : {mode_name: (N_pts,) 幅度数组}
        all_stresses : {mode_name: (N_pts, 6) 平均应力 (kBar)}
        baseline_stress : (6,) 参考构型平均应力 (kBar)

        Returns
        -------
        dict : {cij_raw, cij_symmetrized, slopes_per_mode, born_stable, born_warnings}
        """
        from .born_stability import check_born_stability

        modes = get_vc_modes(self.crystal_system)
        mode_by_name = {m["name"]: m for m in modes}

        # 1) 每个模式拟合斜率向量 + 通道质量权重 (R²)
        slopes_per_mode = {}
        mode_data = []
        for name, mode in mode_by_name.items():
            if name not in all_strains:
                continue
            slopes, info = self._base.fit_single_mode(
                all_strains[name], all_stresses[name], baseline_stress,
                fit_order=fit_order,
            )
            slopes_per_mode[name] = slopes
            d = vc_mode_direction(mode)
            weights = _quality_weights(info["r_squared"])
            mode_data.append((d, slopes, weights))

        if not mode_data:
            raise ValueError("没有可用的 VC 模式数据")

        # 2) 对称性参数化 + 「按 R² 加权」的联立最小二乘 (与 SC 法共用求解器)。
        #    同一常数 (如 C13) 常被多个方程约束: 强差分/主轴通道干净 (R²≈1)、弱横向
        #    通道信号≈噪声 (R² 低甚至为负)。等权 lstsq 会被弱通道拖偏, 这正是同一常数
        #    出现多个矛盾估计 (C13 出 4-5 个值) 的根源。加权后干净通道主导、噪声通道
        #    权重→地板, 既是文献式联立求解又等效于「只用最靠谱的变形」。
        cij_raw, cij_std = solve_cij_weighted(self.crystal_system, mode_data)

        # 3) 规范化 (零项/等价/派生/对称) + Born 稳定性
        cij_sym, _corr = self._base.symmetrize(cij_raw)
        born_stable, born_warnings = check_born_stability(cij_sym, self.crystal_system)

        return {
            "cij_raw": cij_raw,
            "cij_symmetrized": cij_sym,
            "cij_std": cij_std,
            "slopes_per_mode": slopes_per_mode,
            "born_stable": born_stable,
            "born_warnings": born_warnings,
        }
