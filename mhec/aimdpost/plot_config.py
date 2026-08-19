#!/usr/bin/env python3
"""
AIMD Analysis Suite - Plot Configuration
统一管理图表样式和导出设置
"""
import matplotlib as mpl
mpl.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

class PlotConfig:
    """图表配置管理器"""
    
    # 默认DPI设置
    DPI_OPTIONS = {
        'screen': 100,      # 屏幕显示
        'standard': 300,    # 标准打印质量
        'high': 600,        # 高质量打印
        'publication': 1200 # 出版物质量
    }
    
    # 图表尺寸（英寸）
    FIGURE_SIZES = {
        'small': (6, 4),
        'medium': (8, 6),
        'large': (10, 7),
        'wide': (12, 4),
        'square': (8, 8)
    }
    
    # 颜色方案
    COLOR_SCHEMES = {
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'scientific': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                      '#949494', '#ECE133', '#56B4E9', '#F0E442', '#D55E00'],
        'colorblind': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
                      '#949494', '#ECE133', '#56B4E9'],
    }
    
    def __init__(self, dpi='standard', font='Times New Roman', color_scheme='scientific'):
        """
        初始化绘图配置
        
        Parameters:
        -----------
        dpi : str or int
            DPI设置，可以是预设名称或具体数值
        font : str
            字体名称
        color_scheme : str
            颜色方案名称
        """
        self.dpi = self._get_dpi(dpi)
        self.font = font
        self.color_scheme = color_scheme
        self._apply_config()
    
    def _get_dpi(self, dpi):
        """获取DPI值"""
        if isinstance(dpi, int):
            return dpi
        return self.DPI_OPTIONS.get(dpi, 300)
    
    def _apply_config(self):
        """应用配置到matplotlib"""
        # 设置字体
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = [self.font, 'DejaVu Serif', 'Times']
        rcParams['font.size'] = 12
        rcParams['axes.labelsize'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 11
        
        # 设置线条和标记
        rcParams['lines.linewidth'] = 2.0
        rcParams['lines.markersize'] = 6
        rcParams['axes.linewidth'] = 1.2
        
        # 设置网格
        rcParams['grid.alpha'] = 0.3
        rcParams['grid.linewidth'] = 0.8
        
        # 设置图例
        rcParams['legend.frameon'] = True
        rcParams['legend.framealpha'] = 0.9
        rcParams['legend.edgecolor'] = 'gray'
        
        # 设置刻度
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        rcParams['xtick.major.size'] = 6
        rcParams['ytick.major.size'] = 6
        rcParams['xtick.minor.size'] = 3
        rcParams['ytick.minor.size'] = 3
        
        # 设置颜色循环
        if self.color_scheme in self.COLOR_SCHEMES:
            rcParams['axes.prop_cycle'] = mpl.cycler(color=self.COLOR_SCHEMES[self.color_scheme])
    
    def create_figure(self, size='medium', **kwargs):
        """
        创建图表
        
        Parameters:
        -----------
        size : str or tuple
            图表尺寸
        **kwargs : dict
            传递给plt.figure的其他参数
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        if isinstance(size, str):
            figsize = self.FIGURE_SIZES.get(size, (8, 6))
        else:
            figsize = size
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100, **kwargs)
        return fig, ax
    
    def save_figure(self, filename, fig=None, dpi=None, **kwargs):
        """
        保存图表
        
        Parameters:
        -----------
        filename : str
            输出文件名
        fig : matplotlib.figure.Figure, optional
            要保存的图表，默认为当前图表
        dpi : int or str, optional
            DPI设置，默认使用配置的DPI
        **kwargs : dict
            传递给savefig的其他参数
        """
        if dpi is None:
            dpi = self.dpi
        
        # 确保DPI是数字
        dpi = self._get_dpi(dpi)
        
        # 默认参数
        save_params = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'facecolor': 'white',
            'edgecolor': 'none',
            'transparent': False
        }
        save_params.update(kwargs)
        
        if fig is None:
            plt.savefig(filename, **save_params)
        else:
            fig.savefig(filename, **save_params)
        
        print(f"[PLOT] 图表已保存: {filename} (DPI={dpi})")
    
    def format_axis_label(self, label, unit=None):
        """
        格式化坐标轴标签
        
        Parameters:
        -----------
        label : str
            标签文本
        unit : str, optional
            单位
        
        Returns:
        --------
        str : 格式化的标签
        """
        if unit:
            # 使用斜体表示变量，正体表示单位
            return f"${label}$ ({unit})"
        return f"${label}$"
    
    def add_grid(self, ax=None, which='major', **kwargs):
        """添加网格"""
        if ax is None:
            ax = plt.gca()
        
        grid_params = {
            'which': which,
            'alpha': 0.3,
            'linestyle': '--',
            'linewidth': 0.8
        }
        grid_params.update(kwargs)
        ax.grid(True, **grid_params)
    
    def set_scientific_notation(self, ax=None, axis='both'):
        """设置科学计数法"""
        if ax is None:
            ax = plt.gca()
        
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        
        if axis in ['x', 'both']:
            ax.xaxis.set_major_formatter(formatter)
        if axis in ['y', 'both']:
            ax.yaxis.set_major_formatter(formatter)

# 全局配置实例
_plot_config = None

def get_plot_config(dpi='standard', font='Times New Roman', color_scheme='scientific'):
    """获取绘图配置单例"""
    global _plot_config
    if _plot_config is None:
        _plot_config = PlotConfig(dpi=dpi, font=font, color_scheme=color_scheme)
    return _plot_config

def set_plot_config(dpi='standard', font='Times New Roman', color_scheme='scientific'):
    """设置绘图配置"""
    global _plot_config
    _plot_config = PlotConfig(dpi=dpi, font=font, color_scheme=color_scheme)
    return _plot_config

# 便捷函数
def create_figure(size='medium', **kwargs):
    """创建图表"""
    return get_plot_config().create_figure(size=size, **kwargs)

def save_figure(filename, fig=None, dpi=None, **kwargs):
    """保存图表"""
    get_plot_config().save_figure(filename, fig=fig, dpi=dpi, **kwargs)

def format_label(label, unit=None):
    """格式化标签"""
    return get_plot_config().format_axis_label(label, unit=unit)

if __name__ == "__main__":
    # 测试绘图配置
    print("AIMD Plot Configuration Test")
    print("=" * 50)
    
    # 创建测试图表
    config = PlotConfig(dpi='high', font='Times New Roman')
    
    fig, ax = config.create_figure(size='medium')
    
    # 测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='sin(x)')
    ax.plot(x, y2, label='cos(x)')
    
    ax.set_xlabel(config.format_axis_label('x', 'rad'))
    ax.set_ylabel(config.format_axis_label('y', 'a.u.'))
    ax.set_title('Test Plot')
    ax.legend()
    config.add_grid(ax)
    
    config.save_figure('test_plot.png', fig=fig)
    plt.close()
    
    print("\nTest plot saved as 'test_plot.png'")
    print(f"DPI: {config.dpi}")
    print(f"Font: {config.font}")
    print(f"Color scheme: {config.color_scheme}")
