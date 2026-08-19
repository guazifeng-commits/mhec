"""
aimdpost: AIMD 后处理子包。

提供径向分布函数 (RDF)、均方位移 (MSD)、速度自相关函数 (VACF)、
Green-Kubo 粘度、扩散系数、摩擦系数等 AIMD 后处理分析功能。
"""

from .config_manager import ConfigManager, get_config_manager
from .plot_config import PlotConfig, get_plot_config, set_plot_config
