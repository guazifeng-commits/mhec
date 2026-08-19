"""
配置文件解析与默认值。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .slurm import SlurmConfig


def normalize_elastic_method(value: str) -> str:
    """把弹性常数方法名归一到统一规范值 (与菜单显示前缀一致)。

    统一后只有两个规范值:
      "sc" —— SC-SS  单分量应力-应变法 (Single-Component Stress-Strain, strain.py)
      "sa" —— SA-SS  对称化应力-应变法 (Symmetry-Adapted Stress-Strain, strain_vc.py)

    兼容别名 (旧配置/旧叫法照常可用):
      sc / ss / SC-SS / single-component  → "sc"
      sa / vc / SA-SS / symmetry-adapted / volume-conserving → "sa"
    """
    if not value:
        return "sc"
    v = str(value).strip().lower().replace("-", "").replace("_", "")
    if v in ("sa", "vc", "sass", "vcss", "symmetryadapted", "symmetrized", "volumeconserving"):
        return "sa"
    if v in ("sc", "ss", "scss", "singlecomponent", "stressstrain"):
        return "sc"
    return v  # 未知值原样返回, 由下游报错


# 允许的采样点数 (get_amplitude_labels 支持的值)
_VALID_NPOINTS = (3, 5, 7, 9, 11)


def default_n_points_for_system(crystal_system) -> int:
    """按晶系对称性给默认采样点数: 高对称少点省机时, 低对称多点保拟合稳定。

    立方/六方 → 5;  四方/三方/正交 → 7;  单斜/三斜 → 9。
    crystal_system 可为 CrystalSystem 枚举、其 .value 字符串, 或 None(默认 7)。
    """
    val = getattr(crystal_system, "value", crystal_system)
    if val in ("cubic", "hexagonal"):
        return 5
    if val in ("monoclinic", "triclinic"):
        return 9
    if val in ("tetragonal_4mmm", "tetragonal_4m", "trigonal_3m",
               "trigonal_3", "orthorhombic"):
        return 7
    return 7


def resolve_strain_sampling(magnitude=0.005, n_points=9,
                            max_strain=None, strain_points=None):
    """把三种指定方式统一解析为 (magnitude, n_points), 复用现有整数倍布点机制。

    优先级:
      1. strain_points 显式列表 (关于 0 对称、均匀) → 反推 magnitude=最小间隔, n_points=2*半数+1
      2. max_strain + n_points → magnitude = max_strain / (n_points//2)
      3. magnitude + n_points (原样)
    返回 (magnitude, n_points), 均已校验 n_points ∈ {3,5,7,9,11}。
    """
    if strain_points:
        pts = sorted({round(abs(float(p)), 8) for p in strain_points if abs(float(p)) > 1e-9})
        if not pts:
            raise ValueError("strain_points 全为 0")
        half = len(pts)
        n_pts = 2 * half + 1
        if n_pts not in _VALID_NPOINTS:
            raise ValueError(f"strain_points 数目 {half}(半支) 对应 n_points={n_pts}, "
                             f"需为 {[ (k-1)//2 for k in _VALID_NPOINTS ]} 之一(半支)")
        mag = pts[0]
        # 校验均匀: 每个点应为 mag 的整数倍 1..half
        for k, p in enumerate(pts, start=1):
            if abs(p - k * mag) > 1e-6:
                raise ValueError(f"strain_points 必须均匀等间隔 (如 0.01,0.02,...); 收到 {pts}")
        return mag, n_pts
    if max_strain is not None and float(max_strain) > 0:
        if n_points not in _VALID_NPOINTS:
            n_points = 9
        half = n_points // 2
        return float(max_strain) / half, n_points
    return magnitude, n_points


def strain_amplitude_list(magnitude, n_points):
    """返回该采样对应的显式非零应变点列表 (供界面展示), 如 [-0.02,-0.01,0.01,0.02]。"""
    half = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5}.get(n_points)
    if half is None:
        return []
    neg = [-magnitude * k for k in range(half, 0, -1)]
    pos = [magnitude * k for k in range(1, half + 1)]
    return neg + pos


@dataclass
class MHECConfig:
    """MHEC 全局配置。"""
    # 计算参数
    temperatures: List[float] = field(default_factory=lambda: [300.0])
    magnitude: float = 0.005      # 基础幅度 δ; 配合 n_points=9 → ±0.005/0.01/0.015/0.02
    n_points: int = 9             # 采样点数: 9 → ±0.005,±0.01,±0.015,±0.02 (8 个非零点,
                                  # 最大应变仍 0.02 与 MLFF 训练域一致, 密度翻倍以稳定拟合)
    strain_method: str = "standard"
    elastic_method: str = "sc"   # 弹性常数方法(统一命名): "sc"=SC-SS 单分量 / "sa"=SA-SS 对称化
                                 # 兼容别名: 旧值 "ss"→sc, "vc"→sa 仍可读
    # --- 应变采样 (二选一, 优先级: strain_points > max_strain+n_points > magnitude+n_points) ---
    max_strain: Optional[float] = None      # 若设, 应变点在 [-max_strain,+max_strain] 均匀分布,
                                            # magnitude 自动 = max_strain / (n_points//2); 保证最大应变=max_strain
    strain_points: Optional[List[float]] = None  # 显式非零应变点列表 (如 [0.01,0.02] 或 [-0.02,-0.01,0.01,0.02]);
                                            # 设了则据此反推 magnitude/n_points (要求关于 0 对称、均匀)
    fit_poly_order: int = 3      # 拟合阶数: 3=奇多项式(去非谐偏置,推荐) / 1=线性
    skip_steps: int = 500
    lang: Optional[str] = None   # 界面语言: "zh" / "en";None=按 MHEC_LANG 环境变量或默认中文

    # VASP 参数
    encut: Optional[float] = None
    nsw_train: int = 2000
    nsw_refit: int = 1      # ML_MODE=refit 为静态重拟合, 不跑 MD, NSW=1
    nsw_run: int = 10000
    potim: float = 1.0

    # SLURM 参数
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # INCAR 覆盖
    incar_overrides: Dict[str, str] = field(default_factory=dict)

    # 插件
    mechanical_plugin: Optional[str] = None
    aimd_plugin: Optional[str] = None

    # 容差
    crystal_tol: float = 0.01
    r2_threshold: float = 0.95
    zero_threshold: float = 0.10

    # 原胞路径（用于晶系识别，避免超胞误判）
    primitive_poscar: Optional[str] = None
    # 用户手动指定晶系（覆盖自动识别）。取值见 CrystalSystem 枚举, 如
    # "cubic" / "tetragonal_4mmm" / "tetragonal_4m" / "hexagonal" /
    # "trigonal_3m" / "trigonal_3" / "orthorhombic" / "monoclinic" / "triclinic"
    crystal_system: Optional[str] = None
    # 空间群号 (1-230); 若指定则据此推导晶系与 Laue class (优先于 crystal_system 字符串,
    # 最可靠)。例: BeF2 P4₁2₁2 填 92 → 自动得 tetragonal_4mmm (6 常数, C16=0)
    space_group: Optional[int] = None

    # POTCAR 路径（自动复制到所有计算目录）
    potcar_path: Optional[str] = None
    # 用户提交脚本模板路径（用于生成 submit.sh）
    submit_template: Optional[str] = None
    # 预训练 ML_FF 路径（run-only 模式）
    mlff_path: Optional[str] = None

    @classmethod
    def from_file(cls, filepath: str) -> "MHECConfig":
        """从 YAML 或 JSON 配置文件加载。"""
        with open(filepath, "r") as f:
            if filepath.endswith((".yaml", ".yml")):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("需要安装 pyyaml: pip install pyyaml")
            else:
                data = json.load(f)

        config = cls()
        if not data:
            return config

        # 简单字段
        for key in ["temperatures", "magnitude", "n_points", "strain_method",
                     "skip_steps", "crystal_tol", "r2_threshold", "zero_threshold",
                     "primitive_poscar", "crystal_system", "space_group", "potcar_path",
                     "submit_template", "mlff_path", "elastic_method", "fit_poly_order",
                     "max_strain", "strain_points", "lang"]:
            if key in data:
                setattr(config, key, data[key])

        # 归一化弹性常数方法别名 (SA->sa, SC->sc)
        config.elastic_method = normalize_elastic_method(config.elastic_method)

        # 采样解析: strain_points / max_strain 优先, 反推 magnitude & n_points
        if config.strain_points or config.max_strain is not None:
            try:
                config.magnitude, config.n_points = resolve_strain_sampling(
                    config.magnitude, config.n_points,
                    max_strain=config.max_strain, strain_points=config.strain_points)
            except ValueError as e:
                import warnings
                warnings.warn(f"应变采样解析失败, 用 magnitude/n_points 原值: {e}")

        # VASP 参数
        vasp = data.get("vasp", {})
        for key in ["encut", "nsw_train", "nsw_refit", "nsw_run", "potim"]:
            if key in vasp:
                setattr(config, key, vasp[key])

        # SLURM 参数
        slurm_data = data.get("slurm", {})
        if slurm_data:
            config.slurm = SlurmConfig(**{
                k: v for k, v in slurm_data.items()
                if k in SlurmConfig.__dataclass_fields__
            })

        # INCAR 覆盖
        config.incar_overrides = data.get("incar_overrides", {})

        # 插件
        plugins = data.get("plugins", {})
        config.mechanical_plugin = plugins.get("mechanical")
        config.aimd_plugin = plugins.get("aimd_postprocessor")

        return config

    def to_file(self, filepath: str) -> None:
        """保存配置到 JSON 文件。"""
        data = {
            "temperatures": self.temperatures,
            "magnitude": self.magnitude,
            "n_points": self.n_points,
            "strain_method": self.strain_method,
            "elastic_method": self.elastic_method,
            "fit_poly_order": self.fit_poly_order,
            "lang": self.lang,
            "skip_steps": self.skip_steps,
            "vasp": {
                "encut": self.encut,
                "nsw_train": self.nsw_train,
                "nsw_refit": self.nsw_refit,
                "nsw_run": self.nsw_run,
                "potim": self.potim,
            },
            "slurm": {
                k: getattr(self.slurm, k)
                for k in SlurmConfig.__dataclass_fields__
            },
            "incar_overrides": self.incar_overrides,
            "plugins": {
                "mechanical": self.mechanical_plugin,
                "aimd_postprocessor": self.aimd_plugin,
            },
            "crystal_tol": self.crystal_tol,
            "r2_threshold": self.r2_threshold,
            "zero_threshold": self.zero_threshold,
            "primitive_poscar": self.primitive_poscar,
            "crystal_system": self.crystal_system,
            "space_group": self.space_group,
            "max_strain": self.max_strain,
            "strain_points": self.strain_points,
            "potcar_path": self.potcar_path,
            "submit_template": self.submit_template,
            "mlff_path": self.mlff_path,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
