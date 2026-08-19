"""
MLFF 三步工作流管理器。

管理 train → refit → run 三个阶段的 INCAR 生成和状态验证。
"""

import os
from enum import Enum
from typing import Dict, List, Optional
from .incar_templates import build_incar_params
from .vasp_io import write_incar


class MLFFStage(Enum):
    TRAIN = "train"
    REFIT = "refit"
    RUN = "run"

    @property
    def expected_outputs(self) -> List[str]:
        """各阶段的关键输出文件列表。"""
        if self == MLFFStage.TRAIN:
            return ["ML_ABN"]
        elif self == MLFFStage.REFIT:
            return ["ML_FF"]
        else:
            return ["vasprun.xml"]


STAGE_ORDER = [MLFFStage.TRAIN, MLFFStage.REFIT, MLFFStage.RUN]


class MLFFWorkflow:
    """MLFF 三步工作流管理器。"""

    def __init__(
        self,
        work_dir: str,
        ensemble: str,
        temperature: float,
        nsw_train: int = 2000,
        nsw_refit: int = 1,      # ML_MODE=refit 为静态重拟合, 不跑 MD, NSW=1
        nsw_run: int = 10000,
        potim: float = 1.0,
        encut: Optional[float] = None,
        user_overrides: Optional[Dict[str, str]] = None,
    ):
        self.work_dir = work_dir
        self.ensemble = ensemble
        self.temperature = temperature
        self.nsw = {
            "train": nsw_train,
            "refit": nsw_refit,
            "run": nsw_run,
        }
        self.potim = potim
        self.encut = encut
        self.user_overrides = user_overrides or {}

    def setup_stage(self, stage: MLFFStage) -> str:
        """为指定阶段创建子目录并生成 INCAR。返回阶段目录路径。"""
        stage_dir = os.path.join(self.work_dir, stage.value)
        os.makedirs(stage_dir, exist_ok=True)

        params = build_incar_params(
            ensemble=self.ensemble,
            mlff_stage=stage.value,
            temperature=self.temperature,
            nsw=self.nsw[stage.value],
            potim=self.potim,
            encut=self.encut,
            user_overrides=self.user_overrides,
        )
        write_incar(os.path.join(stage_dir, "INCAR"), params)
        return stage_dir

    def validate_stage(self, stage: MLFFStage) -> bool:
        """验证指定阶段的关键输出文件是否存在。"""
        stage_dir = os.path.join(self.work_dir, stage.value)
        for output_file in stage.expected_outputs:
            if not os.path.exists(os.path.join(stage_dir, output_file)):
                return False
        return True

    def get_resume_stage(self) -> Optional[MLFFStage]:
        """检测已完成的阶段，返回下一个需要执行的阶段。全部完成返回 None。"""
        for stage in STAGE_ORDER:
            if not self.validate_stage(stage):
                return stage
        return None

    def setup_all(self, skip_completed: bool = True) -> List[str]:
        """设置所有未完成阶段的输入文件。返回需要执行的阶段目录列表。"""
        dirs = []
        for stage in STAGE_ORDER:
            if skip_completed and self.validate_stage(stage):
                continue
            stage_dir = self.setup_stage(stage)
            dirs.append(stage_dir)
        return dirs
