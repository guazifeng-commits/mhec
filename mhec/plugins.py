"""
插件抽象接口与注册机制。

提供力学性质计算和 AIMD 后处理的标准接口。
"""

import importlib
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


class MechanicalPropertiesPlugin(ABC):
    """力学性质计算插件抽象接口。"""

    @abstractmethod
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


class AIMDPostProcessorPlugin(ABC):
    """AIMD 后处理插件抽象接口。"""

    @abstractmethod
    def process(
        self, work_dir: str, strain_label: str, mlff_stage: str, **kwargs
    ) -> Dict:
        """
        执行 AIMD 后处理分析。

        Parameters
        ----------
        work_dir : 包含 VASP 输出文件的计算目录
        strain_label : 应变标签
        mlff_stage : MLFF 阶段
        """


class DefaultMechanicalProperties(MechanicalPropertiesPlugin):
    """默认占位实现。"""

    def calculate(self, cij_matrix, crystal_system):
        print("提示：未注册力学性质计算插件，请安装或注册自定义实现。")
        return {}


class DefaultAIMDPostProcessor(AIMDPostProcessorPlugin):
    """默认占位实现。"""

    def process(self, work_dir, strain_label, mlff_stage, **kwargs):
        print("提示：未注册 AIMD 后处理插件，请安装或注册自定义实现。")
        return {}


class PluginRegistry:
    """插件注册中心。"""

    def __init__(self):
        self._mech: MechanicalPropertiesPlugin = DefaultMechanicalProperties()
        self._aimd: AIMDPostProcessorPlugin = DefaultAIMDPostProcessor()

    def register_mechanical(self, plugin: MechanicalPropertiesPlugin) -> None:
        self._mech = plugin

    def register_aimd_postprocessor(self, plugin: AIMDPostProcessorPlugin) -> None:
        self._aimd = plugin

    def get_mechanical(self) -> MechanicalPropertiesPlugin:
        return self._mech

    def get_aimd_postprocessor(self) -> AIMDPostProcessorPlugin:
        return self._aimd

    def load_from_config(self, config: Dict) -> None:
        """从配置字典加载插件。"""
        if config.get("mechanical"):
            self._load_plugin(config["mechanical"], "mechanical")
        if config.get("aimd_postprocessor"):
            self._load_plugin(config["aimd_postprocessor"], "aimd")

    def _load_plugin(self, class_path: str, plugin_type: str) -> None:
        """动态加载插件类。"""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls()
            if plugin_type == "mechanical":
                self.register_mechanical(instance)
            else:
                self.register_aimd_postprocessor(instance)
        except Exception as e:
            warnings.warn(f"加载插件 {class_path} 失败: {e}", UserWarning)

    def load_from_entry_points(self) -> None:
        """从 Python entry_points 自动发现并加载插件。"""
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            # Python 3.12+ returns a dict, older returns SelectableGroups
            mech_eps = eps.get("mhec.mechanical", [])
            for ep in mech_eps:
                try:
                    cls = ep.load()
                    self.register_mechanical(cls())
                except Exception as e:
                    warnings.warn(f"加载插件 {ep.name} 失败: {e}", UserWarning)

            aimd_eps = eps.get("mhec.aimd_postprocessor", [])
            for ep in aimd_eps:
                try:
                    cls = ep.load()
                    self.register_aimd_postprocessor(cls())
                except Exception as e:
                    warnings.warn(f"加载插件 {ep.name} 失败: {e}", UserWarning)
        except Exception:
            pass
