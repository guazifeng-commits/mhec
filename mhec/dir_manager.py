"""
计算目录结构管理器。

管理 work_dir/{deform_code}_{amplitude}/{mlff_stage}/ 层级目录。
"""

import os
from typing import Dict, List, Optional


class DirManager:
    """计算目录结构管理器。"""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir

    def create_deform_dirs(
        self,
        deform_codes: List[str],
        amplitude_labels: List[str],
        mlff_stages: List[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        创建完整的目录层级结构。

        Returns
        -------
        {deform_code: {amplitude: {stage: dir_path}}}
        """
        if mlff_stages is None:
            mlff_stages = ["train", "refit", "run"]

        result = {}
        for code in deform_codes:
            result[code] = {}
            for amp in amplitude_labels:
                result[code][amp] = {}
                for stage in mlff_stages:
                    dir_path = os.path.join(
                        self.work_dir, f"{code}_{amp}", stage
                    )
                    os.makedirs(dir_path, exist_ok=True)
                    result[code][amp][stage] = dir_path
        return result

    def create_equilibrium_dir(
        self,
        mlff_stages: List[str] = None,
    ) -> Dict[str, str]:
        """
        创建未变形参考构型目录。

        Returns
        -------
        {stage: dir_path}
        """
        if mlff_stages is None:
            mlff_stages = ["train", "refit", "run"]

        result = {}
        for stage in mlff_stages:
            dir_path = os.path.join(self.work_dir, "equilibrium", stage)
            os.makedirs(dir_path, exist_ok=True)
            result[stage] = dir_path
        return result

    def detect_completed(self) -> Dict[str, List[str]]:
        """检测已完成的计算任务。"""
        completed = {}
        stage_outputs = {
            "train": "ML_ABN",
            "refit": "ML_FF",
            "run": "vasprun.xml",
        }
        for root, dirs, files in os.walk(self.work_dir):
            for stage, output_file in stage_outputs.items():
                if os.path.basename(root) == stage and output_file in files:
                    parent = os.path.dirname(root)
                    if parent not in completed:
                        completed[parent] = []
                    completed[parent].append(stage)
        return completed

    def get_all_dirs(self) -> List[str]:
        """返回所有计算目录的平铺列表。"""
        dirs = []
        for root, subdirs, files in os.walk(self.work_dir):
            if any(f in files for f in ["INCAR", "POSCAR"]):
                dirs.append(root)
        return dirs
