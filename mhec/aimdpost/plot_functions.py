#!/usr/bin/env python3
"""
AIMD Analysis Suite - Enhanced Plot Functions
改进的绘图函数，使用统一的样式配置
"""
import numpy as np
from typing import Dict, List, Optional
try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能出图
    import matplotlib.pyplot as plt
    from .plot_config import get_plot_config, format_label
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def _convert_dpi(dpi):
    """
    转换DPI值为数字
    
    Parameters:
    -----------
    dpi : int or str
        DPI值或预设名称
    
    Returns:
    --------
    int : DPI数值
    """
    if dpi is None:
        return 300  # 默认值
    
    if isinstance(dpi, (int, float)):
        return int(dpi)
    
    # 字符串转换
    dpi_map = {
        'screen': 100,
        'standard': 300,
        'high': 600,
        'publication': 1200
    }
    
    return dpi_map.get(str(dpi).lower(), 300)

def save_plot_rdf(r: np.ndarray, g_r: np.ndarray, out_png: str, dpi=None):
    """
    保存径向分布函数图
    
    Parameters:
    -----------
    r : np.ndarray
        距离数组 (Angstrom)
    g_r : np.ndarray
        径向分布函数值
    out_png : str
        输出文件名
    dpi : int or str, optional
        图片DPI (可以是数字或'standard'等字符串)
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping RDF plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)  # 转换DPI为数字
    
    fig, ax = config.create_figure(size='medium')
    
    ax.plot(r, g_r, linewidth=2.5, color='#0173B2', label='$g(r)$')
    
    ax.set_xlabel(format_label('r', 'Å'), fontsize=14)
    ax.set_ylabel(format_label('g(r)', ''), fontsize=14)
    ax.set_title('Radial Distribution Function', fontsize=16, pad=15)
    
    ax.legend(loc='best', frameon=True, shadow=True)
    config.add_grid(ax)
    
    # 设置坐标轴范围
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_msd(times: np.ndarray, msd: np.ndarray, D: float, out_png: str, dpi=None):
    """
    保存均方位移图
    
    Parameters:
    -----------
    times : np.ndarray
        时间数组 (s)
    msd : np.ndarray
        均方位移 (m^2)
    D : float
        扩散系数 (m^2/s)
    out_png : str
        输出文件名
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping MSD plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax = config.create_figure(size='medium')
    
    # 转换时间单位到ps
    times_ps = times * 1e12
    
    # 绘制MSD数据
    ax.plot(times_ps, msd, linewidth=2.5, color='#0173B2', 
            label='MSD', marker='o', markersize=4, markevery=max(1, len(times)//20))
    
    # 绘制线性拟合
    if D > 0:
        slope = 6 * D
        fit = slope * times + (msd[0] if len(msd) > 0 else 0)
        ax.plot(times_ps, fit, '--', linewidth=2, color='#DE8F05',
                label=f'Linear fit: $D$ = {D:.3e} m²/s')
    
    ax.set_xlabel(format_label('t', 'ps'), fontsize=14)
    ax.set_ylabel(format_label('\\mathrm{MSD}', 'm²'), fontsize=14)
    ax.set_title('Mean Squared Displacement', fontsize=16, pad=15)
    
    ax.legend(loc='best', frameon=True, shadow=True)
    config.add_grid(ax)
    
    # 设置坐标轴范围
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_msd_by_species(times: np.ndarray, msd_species: Dict[str, np.ndarray], 
                             out_png: str, species_diffusion: Dict = None, dpi=None):
    """
    保存按物种分类的均方位移图
    
    Parameters:
    -----------
    times : np.ndarray
        时间数组 (s)
    msd_species : Dict[str, np.ndarray]
        各物种的MSD数据
    out_png : str
        输出文件名
    species_diffusion : Dict, optional
        各物种的扩散系数
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping MSD-by-species plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax = config.create_figure(size='large')
    
    times_ps = times * 1e12
    
    # 颜色列表
    colors = config.COLOR_SCHEMES['scientific']
    
    for idx, (sname, msd) in enumerate(msd_species.items()):
        color = colors[idx % len(colors)]
        marker = ['o', 's', '^', 'v', 'D'][idx % 5]
        
        ax.plot(times_ps, msd, linewidth=2.5, color=color, 
                label=f'{sname}', marker=marker, markersize=5, 
                markevery=max(1, len(times)//15))
        
        # 绘制拟合线
        if species_diffusion and sname in species_diffusion:
            D = species_diffusion[sname]
            if D > 0:
                fit = 6 * D * times + (msd[0] if len(msd) > 0 else 0)
                ax.plot(times_ps, fit, '--', linewidth=1.5, color=color, alpha=0.7,
                        label=f'{sname} fit: $D$ = {D:.3e} m²/s')
    
    ax.set_xlabel(format_label('t', 'ps'), fontsize=14)
    ax.set_ylabel(format_label('\\mathrm{MSD}', 'm²'), fontsize=14)
    ax.set_title('Mean Squared Displacement by Species', fontsize=16, pad=15)
    
    ax.legend(loc='best', frameon=True, shadow=True, ncol=2 if len(msd_species) > 4 else 1)
    config.add_grid(ax)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_vacf(times: np.ndarray, vacf: np.ndarray, out_png: str, dpi=None):
    """
    保存速度自相关函数图
    
    Parameters:
    -----------
    times : np.ndarray
        时间数组 (s)
    vacf : np.ndarray
        速度自相关函数
    out_png : str
        输出文件名
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping VACF plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax = config.create_figure(size='medium')
    
    times_ps = times * 1e12
    
    ax.plot(times_ps, vacf, linewidth=2.5, color='#029E73', label='VACF')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(format_label('t', 'ps'), fontsize=14)
    ax.set_ylabel(format_label('\\mathrm{VACF}', 'm²/s²'), fontsize=14)
    ax.set_title('Velocity Autocorrelation Function', fontsize=16, pad=15)
    
    ax.legend(loc='best', frameon=True, shadow=True)
    config.add_grid(ax)
    
    ax.set_xlim(left=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_molecular_friction(friction_data: Dict, out_png: str, dpi=None):
    """
    保存分子摩擦系数图
    
    Parameters:
    -----------
    friction_data : Dict
        摩擦系数数据
    out_png : str
        输出文件名
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping molecular friction plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax = config.create_figure(size='large')
    
    species = list(friction_data.keys())
    friction_values = [friction_data[s] for s in species]
    
    colors = config.COLOR_SCHEMES['scientific']
    bars = ax.bar(range(len(species)), friction_values, 
                   color=colors[:len(species)], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Species', fontsize=14)
    ax.set_ylabel(format_label('\\xi', 'kg/s'), fontsize=14)
    ax.set_title('Molecular Friction Coefficients', fontsize=16, pad=15)
    
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(species, fontsize=12)
    
    # 在柱子上添加数值标签
    for i, (bar, val) in enumerate(zip(bars, friction_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=10)
    
    config.add_grid(ax, axis='y')
    ax.set_ylim(bottom=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_molecular_friction_coefficients(species_diffusion: Dict, out_png: str, dpi=None):
    """
    保存分子摩擦系数和扩散系数组合图
    
    Parameters:
    -----------
    species_diffusion : Dict
        包含每个物种的扩散系数和摩擦系数数据
        格式: {species_name: {'D_m2_s': float, 'friction_coeff_kg_s': float}}
    out_png : str
        输出文件名
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping molecular friction coefficient plot to {out_png}")
        return
    
    # 提取有效数据
    species_names = []
    diffusion_coeffs = []
    molecular_friction_coeffs = []
    
    for sname, data in species_diffusion.items():
        D = data.get('D_m2_s', float('nan'))
        xi = data.get('friction_coeff_kg_s', float('nan'))
        if not np.isnan(D) and not np.isnan(xi):
            species_names.append(sname)
            diffusion_coeffs.append(D)
            molecular_friction_coeffs.append(xi * 1e12)  # 转换为 pN*s/m 便于显示
    
    if not species_names:
        print("Warning: No valid molecular friction coefficient data to plot")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = config.COLOR_SCHEMES['scientific']
    
    # 绘制扩散系数
    bars1 = ax1.bar(range(len(species_names)), diffusion_coeffs, 
                    color=colors[:len(species_names)], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel(format_label('D', 'm²/s'), fontsize=14)
    ax1.set_title('Diffusion Coefficients by Species', fontsize=16, pad=15)
    ax1.set_xticks(range(len(species_names)))
    ax1.set_xticklabels(species_names, fontsize=12)
    ax1.set_yscale('log')
    config.add_grid(ax1, axis='y')
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars1, diffusion_coeffs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 绘制分子摩擦系数
    bars2 = ax2.bar(range(len(species_names)), molecular_friction_coeffs, 
                    color=colors[:len(species_names)], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel(format_label('\\xi', 'pN·s/m'), fontsize=14)
    ax2.set_xlabel('Species', fontsize=14)
    ax2.set_title('Molecular Friction Coefficients by Species (ξ = k_BT/D)', fontsize=16, pad=15)
    ax2.set_xticks(range(len(species_names)))
    ax2.set_xticklabels(species_names, fontsize=12)
    ax2.set_yscale('log')
    config.add_grid(ax2, axis='y')
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars2, molecular_friction_coeffs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)
    print(f"[PLOT] 分子摩擦系数图像已保存到 {out_png}")


def save_plot_viscosity(times_list: List[np.ndarray], visc_list: List[np.ndarray], 
                       mean_curve: np.ndarray, out_png: str, 
                       labels: Optional[List[str]] = None, dpi=None):
    """
    保存粘度图
    
    Parameters:
    -----------
    times_list : List[np.ndarray]
        时间数组列表
    visc_list : List[np.ndarray]
        粘度数据列表
    mean_curve : np.ndarray
        平均粘度曲线
    out_png : str
        输出文件名
    labels : List[str], optional
        标签列表
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping Viscosity plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax = config.create_figure(size='medium')
    
    colors = config.COLOR_SCHEMES['scientific']
    
    # 绘制各条曲线
    for i, visc in enumerate(visc_list):
        tps = times_list[i] * 1e12
        label = labels[i] if labels and i < len(labels) else f'Run {i+1}'
        ax.plot(tps, visc, linewidth=1.5, alpha=0.5, color=colors[i % len(colors)])
    
    # 绘制平均曲线
    if len(times_list) > 0:
        tps_mean = times_list[0] * 1e12
        ax.plot(tps_mean, mean_curve, linewidth=3, color='#D62728', 
                label='Mean', linestyle='-', marker='o', markersize=4, 
                markevery=max(1, len(tps_mean)//20))
    
    ax.set_xlabel(format_label('t', 'ps'), fontsize=14)
    ax.set_ylabel(format_label('\\eta', 'Pa·s'), fontsize=14)
    ax.set_title('Green-Kubo Viscosity', fontsize=16, pad=15)
    
    ax.legend(loc='best', frameon=True, shadow=True)
    config.add_grid(ax)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_energy_time(en_times: np.ndarray, energies: List[float], 
                         out_png: str, temps: Optional[List[float]] = None, dpi=None):
    """
    保存能量-时间图
    
    Parameters:
    -----------
    en_times : np.ndarray
        时间数组
    energies : List[float]
        能量数据
    out_png : str
        输出文件名
    temps : List[float], optional
        温度数据
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping Energy vs Time plot to {out_png}")
        return
    
    config = get_plot_config()
    dpi = _convert_dpi(dpi)
    fig, ax1 = config.create_figure(size='medium')
    
    # 绘制能量
    color1 = '#0173B2'
    ax1.plot(en_times, energies, linewidth=2, color=color1, label='Energy')
    ax1.set_xlabel(format_label('t', 'fs'), fontsize=14)
    ax1.set_ylabel(format_label('E', 'eV'), fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    config.add_grid(ax1)
    
    # 绘制温度（如果有）
    if temps:
        ax2 = ax1.twinx()
        color2 = '#DE8F05'
        ax2.plot(en_times, temps, linewidth=2, color=color2, label='Temperature', alpha=0.7)
        ax2.set_ylabel(format_label('T', 'K'), fontsize=14, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('Energy and Temperature vs Time', fontsize=16, pad=15)
    
    fig.tight_layout()
    config.save_figure(out_png, fig=fig, dpi=dpi)
    plt.close(fig)


def save_plot_viscosity_from_sacf(data_file: str, out_png: str, dpi=None):
    """
    从SACF积分数据绘制粘度图
    
    Parameters:
    -----------
    data_file : str
        SACF粘度数据文件路径
    out_png : str
        输出图片文件路径
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping viscosity from SACF plot to {out_png}")
        return
    
    try:
        # 读取数据文件
        data = np.loadtxt(data_file, skiprows=1)
        if data.size == 0:
            print(f"Warning: Empty data file {data_file}")
            return
        
        times = data[:, 0]  # time_ps
        viscosity = data[:, 1]  # viscosity_from_SACF_Pa_s
        
        config = get_plot_config()
        dpi = _convert_dpi(dpi)
        fig, ax = config.create_figure(size='large')
        
        ax.plot(times, viscosity, linewidth=2.5, color='#CC78BC', 
                label='Viscosity from SACF integration', marker='o', markersize=4,
                markevery=max(1, len(times)//20))
        
        ax.set_xlabel(format_label('t', 'ps'), fontsize=14)
        ax.set_ylabel(format_label('\\eta', 'Pa·s'), fontsize=14)
        ax.set_title('Viscosity from SACF Integration', fontsize=16, pad=15)
        
        ax.legend(loc='best', frameon=True, shadow=True)
        config.add_grid(ax)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        config.save_figure(out_png, fig=fig, dpi=dpi)
        plt.close(fig)
        print(f"[PLOT] SACF积分粘度图像已保存到 {out_png}")
        
    except Exception as e:
        print(f"Error plotting viscosity from SACF: {e}")
        return


def save_plot_pdos(frequencies_THz: np.ndarray, pdos: np.ndarray, out_png: str, dpi=None):
    """
    绘制声子态密度（PDOS）图
    
    Parameters:
    -----------
    frequencies_THz : np.ndarray
        频率数组（THz）
    pdos : np.ndarray
        归一化的声子态密度
    out_png : str
        输出图片文件路径
    dpi : int or str, optional
        图片DPI
    """
    if not HAS_MPL:
        print(f"Warning: Matplotlib not available, skipping PDOS plot to {out_png}")
        return
    
    try:
        config = get_plot_config()
        dpi = _convert_dpi(dpi)
        
        # 创建双 x 轴图：上方 THz，下方 cm^-1
        fig, ax1 = config.create_figure(size='large')
        
        # 主图：THz
        ax1.plot(frequencies_THz, pdos, linewidth=2.5, color='#1F77B4', alpha=0.8)
        ax1.fill_between(frequencies_THz, 0, pdos, alpha=0.3, color='#1F77B4')
        
        ax1.set_xlabel(format_label('\\nu', 'THz'), fontsize=14)
        ax1.set_ylabel('Phonon Density of States (normalized)', fontsize=14)
        ax1.set_title('Phonon Density of States from VACF', fontsize=16, pad=15)
        
        # 找到峰值并标注
        if len(pdos) > 0:
            peak_idx = np.argmax(pdos)
            peak_freq = frequencies_THz[peak_idx]
            peak_value = pdos[peak_idx]
            
            ax1.plot(peak_freq, peak_value, 'ro', markersize=8, label=f'Peak: {peak_freq:.2f} THz')
            ax1.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax1.legend(loc='best', frameon=True, shadow=True)
        
        # 添加次坐标轴（cm^-1）
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim()[0] * 33.356, ax1.get_xlim()[1] * 33.356)
        ax2.set_xlabel(format_label('\\nu', 'cm^{-1}'), fontsize=14)
        
        config.add_grid(ax1)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        
        config.save_figure(out_png, fig=fig, dpi=dpi)
        plt.close(fig)
        print(f"[PLOT] PDOS图像已保存到 {out_png}")
        
    except Exception as e:
        print(f"Error plotting PDOS: {e}")
        import traceback
        traceback.print_exc()
        return


# 导出函数列表
__all__ = [
    'save_plot_rdf',
    'save_plot_msd',
    'save_plot_msd_by_species',
    'save_plot_vacf',
    'save_plot_molecular_friction',
    'save_plot_molecular_friction_coefficients',
    'save_plot_viscosity',
    'save_plot_energy_time',
    'save_plot_viscosity_from_sacf',
    'save_plot_pdos',
]
