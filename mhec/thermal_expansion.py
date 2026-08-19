"""
热膨胀系数计算模块。

从多温度 NPT AIMD 轨迹提取平衡晶格参数，
拟合 a(T), b(T), c(T), V(T) 曲线，
计算线热膨胀系数 α_L(T) 和体热膨胀系数 α_V(T)。

物理公式:
  线热膨胀系数: α_a(T) = (1/a)(da/dT)
  体热膨胀系数: α_V(T) = (1/V)(dV/dT)
  对于各向同性材料: α_V ≈ α_a + α_b + α_c
  对于立方体系: α_V = 3·α_L

拟合方法:
  使用多项式拟合 (默认 2 阶) 或样条插值。
  α(T) = (1/X) · dX/dT，其中 X = a, b, c 或 V。
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LatticeAtTemp:
    """单个温度下的平衡晶格数据。"""
    temperature: float          # K
    a: float                    # Å
    b: float                    # Å
    c: float                    # Å
    alpha_deg: float            # degree
    beta_deg: float             # degree
    gamma_deg: float            # degree
    volume: float               # ų
    lattice: Optional[np.ndarray] = None  # 3×3 矩阵
    n_frames: int = 0           # 用于平均的帧数
    a_std: float = 0.0          # a 的标准差
    b_std: float = 0.0
    c_std: float = 0.0
    volume_std: float = 0.0


@dataclass
class ThermalExpansionResult:
    """热膨胀系数计算结果。"""
    temperatures: np.ndarray           # K
    # 晶格参数
    a_values: np.ndarray               # Å
    b_values: np.ndarray
    c_values: np.ndarray
    volume_values: np.ndarray          # ų
    # 拟合曲线 (密集温度网格)
    T_fit: np.ndarray                  # K
    a_fit: np.ndarray
    b_fit: np.ndarray
    c_fit: np.ndarray
    V_fit: np.ndarray
    # 热膨胀系数 (密集温度网格)
    alpha_a: np.ndarray                # K⁻¹
    alpha_b: np.ndarray
    alpha_c: np.ndarray
    alpha_V: np.ndarray                # K⁻¹
    # 拟合多项式系数
    poly_a: np.ndarray
    poly_b: np.ndarray
    poly_c: np.ndarray
    poly_V: np.ndarray
    # 拟合阶数
    poly_order: int = 2


def compute_thermal_expansion(
    data: List[LatticeAtTemp],
    poly_order: int = 2,
    T_grid_points: int = 200,
) -> ThermalExpansionResult:
    """
    从多温度晶格数据计算热膨胀系数。

    Parameters
    ----------
    data : 各温度下的晶格数据列表（至少 3 个温度点）
    poly_order : 多项式拟合阶数 (默认 2)
    T_grid_points : 输出温度网格点数

    Returns
    -------
    ThermalExpansionResult
    """
    if len(data) < 2:
        raise ValueError("至少需要 2 个温度点来计算热膨胀系数")

    # 按温度排序
    data_sorted = sorted(data, key=lambda x: x.temperature)

    T = np.array([d.temperature for d in data_sorted])
    a = np.array([d.a for d in data_sorted])
    b = np.array([d.b for d in data_sorted])
    c = np.array([d.c for d in data_sorted])
    V = np.array([d.volume for d in data_sorted])

    # 限制多项式阶数不超过数据点数-1
    order = min(poly_order, len(T) - 1)

    # 多项式拟合
    poly_a = np.polyfit(T, a, order)
    poly_b = np.polyfit(T, b, order)
    poly_c = np.polyfit(T, c, order)
    poly_V = np.polyfit(T, V, order)

    # 密集温度网格
    T_fit = np.linspace(T.min(), T.max(), T_grid_points)

    # 拟合值
    a_fit = np.polyval(poly_a, T_fit)
    b_fit = np.polyval(poly_b, T_fit)
    c_fit = np.polyval(poly_c, T_fit)
    V_fit = np.polyval(poly_V, T_fit)

    # 导数多项式
    dpoly_a = np.polyder(poly_a)
    dpoly_b = np.polyder(poly_b)
    dpoly_c = np.polyder(poly_c)
    dpoly_V = np.polyder(poly_V)

    # 热膨胀系数: α(T) = (1/X) · dX/dT
    da_dT = np.polyval(dpoly_a, T_fit)
    db_dT = np.polyval(dpoly_b, T_fit)
    dc_dT = np.polyval(dpoly_c, T_fit)
    dV_dT = np.polyval(dpoly_V, T_fit)

    alpha_a = da_dT / a_fit
    alpha_b = db_dT / b_fit
    alpha_c = dc_dT / c_fit
    alpha_V = dV_dT / V_fit

    return ThermalExpansionResult(
        temperatures=T,
        a_values=a, b_values=b, c_values=c, volume_values=V,
        T_fit=T_fit,
        a_fit=a_fit, b_fit=b_fit, c_fit=c_fit, V_fit=V_fit,
        alpha_a=alpha_a, alpha_b=alpha_b, alpha_c=alpha_c, alpha_V=alpha_V,
        poly_a=poly_a, poly_b=poly_b, poly_c=poly_c, poly_V=poly_V,
        poly_order=order,
    )


def print_thermal_expansion_report(result: ThermalExpansionResult) -> None:
    """打印热膨胀系数报告。"""
    print("\n  ═══════════════════════════════════════════════════")
    print("  热膨胀系数分析结果")
    print("  ═══════════════════════════════════════════════════")

    # 晶格参数表
    print(f"\n  {'T (K)':>8}  {'a (Å)':>10}  {'b (Å)':>10}  {'c (Å)':>10}  {'V (ų)':>12}")
    print(f"  {'─' * 56}")
    for i in range(len(result.temperatures)):
        print(f"  {result.temperatures[i]:>8.0f}  "
              f"{result.a_values[i]:>10.4f}  {result.b_values[i]:>10.4f}  "
              f"{result.c_values[i]:>10.4f}  {result.volume_values[i]:>12.4f}")

    # 热膨胀系数（在数据温度点处插值）
    print(f"\n  热膨胀系数 (×10⁻⁶ K⁻¹):")
    print(f"  {'T (K)':>8}  {'α_a':>10}  {'α_b':>10}  {'α_c':>10}  {'α_V':>10}")
    print(f"  {'─' * 52}")
    for T_val in result.temperatures:
        idx = np.argmin(np.abs(result.T_fit - T_val))
        print(f"  {T_val:>8.0f}  "
              f"{result.alpha_a[idx]*1e6:>10.2f}  {result.alpha_b[idx]*1e6:>10.2f}  "
              f"{result.alpha_c[idx]*1e6:>10.2f}  {result.alpha_V[idx]*1e6:>10.2f}")

    print(f"\n  拟合多项式阶数: {result.poly_order}")
    print(f"  温度范围: {result.temperatures.min():.0f} — {result.temperatures.max():.0f} K")


def save_thermal_expansion(result: ThermalExpansionResult, prefix: str) -> None:
    """保存热膨胀数据到文件。"""
    # 1. 晶格参数 vs 温度 (原始数据点)
    lattice_dat = f"{prefix}_lattice_vs_T.dat"
    header = "T_K\ta_A\tb_A\tc_A\tV_A3"
    np.savetxt(lattice_dat,
               np.column_stack((result.temperatures, result.a_values,
                                result.b_values, result.c_values,
                                result.volume_values)),
               header=header, comments='', delimiter='\t', fmt='%.6f')
    print(f"  +-> {lattice_dat}")

    # 2. 拟合曲线 (密集网格)
    fit_dat = f"{prefix}_lattice_fit.dat"
    header = "T_K\ta_fit\tb_fit\tc_fit\tV_fit"
    np.savetxt(fit_dat,
               np.column_stack((result.T_fit, result.a_fit,
                                result.b_fit, result.c_fit, result.V_fit)),
               header=header, comments='', delimiter='\t', fmt='%.6f')
    print(f"  +-> {fit_dat}")

    # 3. 热膨胀系数 (密集网格)
    alpha_dat = f"{prefix}_thermal_expansion.dat"
    header = "T_K\talpha_a_1_K\talpha_b_1_K\talpha_c_1_K\talpha_V_1_K"
    np.savetxt(alpha_dat,
               np.column_stack((result.T_fit, result.alpha_a,
                                result.alpha_b, result.alpha_c, result.alpha_V)),
               header=header, comments='', delimiter='\t', fmt='%.6e')
    print(f"  +-> {alpha_dat}")

    # 4. JSON 摘要
    summary = {
        "temperatures_K": result.temperatures.tolist(),
        "a_A": result.a_values.tolist(),
        "b_A": result.b_values.tolist(),
        "c_A": result.c_values.tolist(),
        "V_A3": result.volume_values.tolist(),
        "poly_order": result.poly_order,
        "poly_a": result.poly_a.tolist(),
        "poly_b": result.poly_b.tolist(),
        "poly_c": result.poly_c.tolist(),
        "poly_V": result.poly_V.tolist(),
        "alpha_V_at_data_points_1_K": [],
    }
    for T_val in result.temperatures:
        idx = np.argmin(np.abs(result.T_fit - T_val))
        summary["alpha_V_at_data_points_1_K"].append(float(result.alpha_V[idx]))

    json_path = f"{prefix}_thermal_expansion.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  +-> {json_path}")


def plot_thermal_expansion(result: ThermalExpansionResult, prefix: str) -> None:
    """绘制热膨胀系数图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  !-> matplotlib 不可用，跳过绘图")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) 晶格参数 a, b, c vs T
    ax = axes[0, 0]
    ax.plot(result.temperatures, result.a_values, 'ro', markersize=8, label='a (data)')
    ax.plot(result.temperatures, result.b_values, 'bs', markersize=8, label='b (data)')
    ax.plot(result.temperatures, result.c_values, 'g^', markersize=8, label='c (data)')
    ax.plot(result.T_fit, result.a_fit, 'r-', alpha=0.7)
    ax.plot(result.T_fit, result.b_fit, 'b-', alpha=0.7)
    ax.plot(result.T_fit, result.c_fit, 'g-', alpha=0.7)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Lattice parameter (Å)')
    ax.set_title('Lattice Parameters vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) 体积 vs T
    ax = axes[0, 1]
    ax.plot(result.temperatures, result.volume_values, 'ko', markersize=8, label='V (data)')
    ax.plot(result.T_fit, result.V_fit, 'k-', alpha=0.7, label='V (fit)')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Volume (ų)')
    ax.set_title('Volume vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) 线热膨胀系数 vs T
    ax = axes[1, 0]
    ax.plot(result.T_fit, result.alpha_a * 1e6, 'r-', label='α_a')
    ax.plot(result.T_fit, result.alpha_b * 1e6, 'b-', label='α_b')
    ax.plot(result.T_fit, result.alpha_c * 1e6, 'g-', label='α_c')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Linear CTE (×10⁻⁶ K⁻¹)')
    ax.set_title('Linear Thermal Expansion Coefficients')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) 体热膨胀系数 vs T
    ax = axes[1, 1]
    ax.plot(result.T_fit, result.alpha_V * 1e6, 'k-', linewidth=2, label='α_V')
    # 标注数据点处的值
    for T_val in result.temperatures:
        idx = np.argmin(np.abs(result.T_fit - T_val))
        ax.plot(T_val, result.alpha_V[idx] * 1e6, 'ko', markersize=8)
        ax.annotate(f'{result.alpha_V[idx]*1e6:.1f}',
                    (T_val, result.alpha_V[idx] * 1e6),
                    textcoords="offset points", xytext=(5, 10), fontsize=9)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Volumetric CTE (×10⁻⁶ K⁻¹)')
    ax.set_title('Volumetric Thermal Expansion Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = f"{prefix}_thermal_expansion.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  +-> {out_png}")
