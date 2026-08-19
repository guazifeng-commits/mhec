#!/usr/bin/env python3
"""
AIMD Analysis Suite - Configuration Manager
管理安装配置和用户设置
"""
import os
import sys
import configparser
from pathlib import Path

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_file = self._find_config_file()
        self.load_config()
    
    def _find_config_file(self):
        """查找配置文件"""
        # 尝试多个位置
        possible_locations = [
            # 1. 程序安装目录
            os.path.join(self._get_install_dir(), 'config.ini'),
            # 2. 用户文档目录
            os.path.join(os.path.expanduser('~'), 'Documents', 'AIMD_Suite', 'config.ini'),
            # 3. 当前目录
            'config.ini',
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                return location
        
        # 如果都不存在，返回默认位置
        return possible_locations[0]
    
    def _get_install_dir(self):
        """获取程序安装目录"""
        if getattr(sys, 'frozen', False):
            # 打包后的环境
            return os.path.dirname(sys.executable)
        else:
            # 开发环境
            return os.path.dirname(os.path.abspath(__file__))
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config['Paths'] = {
            'InstallDir': self._get_install_dir(),
            'DataDir': os.path.join(os.path.expanduser('~'), 'Documents', 'AIMD_Data'),
        }
        
        self.config['Settings'] = {
            'Version': '2.0',
            'Language': 'zh_CN',
        }
        
        self.config['PlotSettings'] = {
            'DPI': 'standard',
            'Font': 'Times New Roman',
            'ColorScheme': 'scientific',
        }
        
        self.config['Recent'] = {
            'LastWorkDir': '',
            'LastOutputDir': '',
        }
    
    def save_config(self):
        """保存配置"""
        try:
            # 确保配置文件目录存在
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            return True
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
            return False
    
    def get_data_dir(self):
        """获取数据目录"""
        try:
            data_dir = self.config.get('Paths', 'DataDir')
            # 确保目录存在
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            return data_dir
        except:
            # 返回默认值
            default_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'AIMD_Data')
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
            return default_dir
    
    def get_install_dir(self):
        """获取安装目录"""
        try:
            return self.config.get('Paths', 'InstallDir')
        except:
            return self._get_install_dir()
    
    def set_data_dir(self, path):
        """设置数据目录"""
        if not self.config.has_section('Paths'):
            self.config.add_section('Paths')
        self.config.set('Paths', 'DataDir', path)
        self.save_config()
    
    def get_last_work_dir(self):
        """获取上次工作目录"""
        try:
            last_dir = self.config.get('Recent', 'LastWorkDir')
            if last_dir and os.path.exists(last_dir):
                return last_dir
        except:
            pass
        return self.get_data_dir()
    
    def set_last_work_dir(self, path):
        """设置上次工作目录"""
        if not self.config.has_section('Recent'):
            self.config.add_section('Recent')
        self.config.set('Recent', 'LastWorkDir', path)
        self.save_config()
    
    def get_last_output_dir(self):
        """获取上次输出目录"""
        try:
            last_dir = self.config.get('Recent', 'LastOutputDir')
            if last_dir and os.path.exists(last_dir):
                return last_dir
        except:
            pass
        return self.get_data_dir()
    
    def set_last_output_dir(self, path):
        """设置上次输出目录"""
        if not self.config.has_section('Recent'):
            self.config.add_section('Recent')
        self.config.set('Recent', 'LastOutputDir', path)
        self.save_config()
    
    def get_plot_settings(self):
        """获取绘图设置"""
        try:
            dpi = self.config.get('PlotSettings', 'DPI')
            font = self.config.get('PlotSettings', 'Font')
            color_scheme = self.config.get('PlotSettings', 'ColorScheme')
            return {
                'dpi': dpi,
                'font': font,
                'color_scheme': color_scheme
            }
        except:
            # 返回默认值
            return {
                'dpi': 'standard',
                'font': 'Times New Roman',
                'color_scheme': 'scientific'
            }
    
    def set_plot_settings(self, dpi, font, color_scheme):
        """设置绘图配置"""
        if not self.config.has_section('PlotSettings'):
            self.config.add_section('PlotSettings')
        
        # 如果dpi是整数，转换为字符串
        dpi_str = str(dpi) if isinstance(dpi, int) else dpi
        
        self.config.set('PlotSettings', 'DPI', dpi_str)
        self.config.set('PlotSettings', 'Font', font)
        self.config.set('PlotSettings', 'ColorScheme', color_scheme)
        self.save_config()

# 全局配置管理器实例
_config_manager = None

def get_config_manager():
    """获取配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# 便捷函数
def get_data_dir():
    """获取数据目录"""
    return get_config_manager().get_data_dir()

def get_install_dir():
    """获取安装目录"""
    return get_config_manager().get_install_dir()

def get_last_work_dir():
    """获取上次工作目录"""
    return get_config_manager().get_last_work_dir()

def set_last_work_dir(path):
    """设置上次工作目录"""
    get_config_manager().set_last_work_dir(path)

def get_last_output_dir():
    """获取上次输出目录"""
    return get_config_manager().get_last_output_dir()

def set_last_output_dir(path):
    """设置上次输出目录"""
    get_config_manager().set_last_output_dir(path)

def get_plot_settings():
    """获取绘图设置"""
    return get_config_manager().get_plot_settings()

def set_plot_settings(dpi, font, color_scheme):
    """设置绘图配置"""
    get_config_manager().set_plot_settings(dpi, font, color_scheme)

if __name__ == "__main__":
    # 测试配置管理器
    print("AIMD Configuration Manager Test")
    print("=" * 50)
    
    cm = get_config_manager()
    
    print(f"Config file: {cm.config_file}")
    print(f"Install dir: {cm.get_install_dir()}")
    print(f"Data dir: {cm.get_data_dir()}")
    print(f"Last work dir: {cm.get_last_work_dir()}")
    print(f"Last output dir: {cm.get_last_output_dir()}")
    
    print("\nConfiguration loaded successfully!")
