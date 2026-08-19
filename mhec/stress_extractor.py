"""
应力提取与统计。

从 VASP 输出文件 (vasprun.xml / OUTCAR) 提取时间平均应力张量。
"""

import os
import re
import numpy as np
from typing import Tuple, Dict, Optional
from xml.etree import ElementTree as ET


class StressExtractor:
    """从 VASP 输出提取时间平均应力。"""

    def extract_from_vasprun(self, filepath: str) -> np.ndarray:
        """
        从 vasprun.xml 提取每个离子步的应力张量。

        Returns
        -------
        (N_steps, 6) : Voigt 记号 (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)
                       单位: kBar
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        stresses = []
        for calc in root.iter("calculation"):
            stress_elem = calc.find(".//varray[@name='stress']")
            if stress_elem is None:
                continue
            rows = []
            for v in stress_elem.findall("v"):
                rows.append([float(x) for x in v.text.split()])
            s = np.array(rows)  # 3×3 应力张量
            # 转为 Voigt: xx, yy, zz, yz, xz, xy
            voigt = np.array([
                s[0, 0], s[1, 1], s[2, 2],
                s[1, 2], s[0, 2], s[0, 1]
            ])
            stresses.append(voigt)

        if not stresses:
            raise ValueError(f"未在 {filepath} 中找到应力数据")
        return np.array(stresses)

    def extract_from_outcar(self, filepath: str) -> np.ndarray:
        """从 OUTCAR 提取应力张量。"""
        stresses = []
        pattern = re.compile(
            r"in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
            r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
        )
        with open(filepath, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    # OUTCAR 顺序: XX YY ZZ XY YZ ZX
                    vals = [float(m.group(i)) for i in range(1, 7)]
                    # 转为 Voigt: xx, yy, zz, yz, xz, xy
                    voigt = np.array([
                        vals[0], vals[1], vals[2],
                        vals[4], vals[5], vals[3]
                    ])
                    stresses.append(voigt)

        if not stresses:
            raise ValueError(f"未在 {filepath} 中找到应力数据")
        return np.array(stresses)

    def compute_average(
        self,
        stresses: np.ndarray,
        skip_steps: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算时间平均应力和标准误差。

        Parameters
        ----------
        stresses : (N_steps, 6)
        skip_steps : 跳过的初始弛豫步数

        Returns
        -------
        (mean_stress(6,), stderr(6,))
        """
        if skip_steps >= len(stresses):
            # MD 平均: skip 超过总步数时, 自动退回到后半段而不是直接报错
            new_skip = len(stresses) // 2
            print(f"[stress] 警告: skip_steps ({skip_steps}) >= 可用帧数 "
                  f"({len(stresses)})，自动改用后半段 (skip={new_skip})。"
                  f"若可用帧数异常偏少, 请检查 VASP 输出频率 (见下文提示)。")
            skip_steps = new_skip
        data = stresses[skip_steps:]
        n = len(data)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1) if n > 1 else np.zeros(6)
        stderr = std / np.sqrt(n)
        return mean, stderr

    def extract_and_average(
        self,
        work_dir: str,
        skip_steps: int = 0,
        prefer_mlff: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        从计算目录提取并计算平均应力。

        Returns
        -------
        (mean_stress, stderr, metadata)
        """
        vasprun_path = os.path.join(work_dir, "vasprun.xml")
        outcar_path = os.path.join(work_dir, "OUTCAR")

        # 同时尝试 vasprun.xml 与 OUTCAR, 选帧数更多的来源。
        # (MLFF run 时 vasprun.xml 常只稀疏写入若干帧, 而 OUTCAR 的 'in kB'
        #  通常每步都有, 因此优先用帧数多的那个以获得完整的时间平均。)
        candidates = []
        if os.path.exists(vasprun_path):
            try:
                candidates.append((self.extract_from_vasprun(vasprun_path), "vasprun.xml"))
            except Exception:
                pass
        if os.path.exists(outcar_path):
            try:
                candidates.append((self.extract_from_outcar(outcar_path), "OUTCAR"))
            except Exception:
                pass

        if not candidates:
            raise FileNotFoundError(
                f"在 {work_dir} 中未找到可解析的 vasprun.xml 或 OUTCAR"
            )

        stresses, source = max(candidates, key=lambda cs: len(cs[0]))

        mean, stderr = self.compute_average(stresses, skip_steps)
        metadata = {
            "n_total_steps": len(stresses),
            "n_used_steps": len(stresses) - skip_steps,
            "source": source,
        }
        return mean, stderr, metadata
