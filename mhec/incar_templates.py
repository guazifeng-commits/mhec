"""
INCAR 模板与参数管理。

提供 MLFF 三步工作流各阶段的 INCAR 参数模板，
支持 NPT/NVT 系综和用户覆盖。

MLFF 精度要求:
  弹性常数计算对应力精度要求极高，MLFF 训练必须使用高精度 DFT 参数。
  - PREC=Accurate, EDIFF=1E-8: 确保 DFT 参考数据精度
  - LREAL=Auto, NCORE=4: 默认省内存 (大体系/高熵合金友好); 追求极致力/应力精度可改 LREAL=.FALSE.
  - ADDGRID=.TRUE.: 增强 FFT 网格精度
  - ML_CX=0.1: 控制 MLFF 训练的贝叶斯误差阈值
  - ML_MB=3000: 训练集容量 (默认 3000 平衡精度与内存; 8000 等过大值在多核节点上易触发 OOM)
  - ML_IWEIGHT=3, ML_WTSIF=4.0: 在保证能量/力的前提下加大应力权重,
    提升弹性常数计算所需的应力拟合精度 (ML_WTOTEN=ML_WTIFOR=1.0 为基准)
"""

from typing import Dict, Optional


# ============================================================
# 高精度 DFT 基础参数（所有阶段共用）
# ============================================================

BASE_HIGH_PRECISION = {
    # 电子自洽精度
    "PREC": "Accurate",
    "EDIFF": "1E-8",
    "EDIFFG": "-1E-4",
    "ALGO": "Normal",
    "NELM": "200",
    # 精度增强
    "ADDGRID": ".TRUE.",
    "LREAL": "Auto",
    "NCORE": "4",
    # 展宽
    "ISMEAR": "0",
    "SIGMA": "0.05",
}


# ============================================================
# 系综参数
# ============================================================

NPT_MD_BASE = {
    "IBRION": "0",
    "ISIF": "3",          # 允许晶格变化
    "MDALGO": "3",        # Langevin 恒温器
    "ISYM": "0",
    "PSTRESS": "0",
    "SMASS": "-1",
    "PMASS": "1000",
}

NVT_MD_BASE = {
    "IBRION": "0",
    "ISIF": "2",          # 固定晶格，计算应力
    "MDALGO": "2",        # Nosé-Hoover 恒温器
    "ISYM": "0",
    "SMASS": "2",
}


# ============================================================
# MLFF 各阶段参数
# ============================================================

MLFF_PARAMS = {
    "train": {
        "ML_LMLFF": ".TRUE.",
        "ML_MODE": "train",
        "ML_ISTART": "0",        # 从头训练
        "ML_IALGO_LINREG": "4",  # 贝叶斯线性回归
        "ML_CX": "0.1",          # 贝叶斯误差阈值 (越小越严格)
        "ML_MB": "3000",         # 最大局部参考构型数 (默认 3000 控内存; 多元素/大体系可上调, 过大易 OOM)
        "ML_MCONF_NEW": "1",     # 每步最多加入1个新构型
        "ML_LBASIS_DISCARD": ".FALSE.",  # 不丢弃基函数
        # 训练数据权重 (弹性常数对应力精度要求高, 在保证能量/力的前提下加大应力权重)
        "ML_IWEIGHT": "3",       # 归一化方式 (VASP 推荐, 权重为无量纲乘子)
        "ML_WTOTEN": "1.0",      # 能量权重 (基准)
        "ML_WTIFOR": "1.0",      # 力权重 (基准)
        "ML_WTSIF": "4.0",       # 应力权重 (加大, 提升弹性常数所需的应力精度)
    },
    "refit": {
        "ML_LMLFF": ".TRUE.",
        "ML_MODE": "refit",
        "IBRION": "-1",          # refit 不移动离子, 避免 IBRION=0 缺 POTIM 报错
        "ML_ISTART": "1",        # 从已有力场继续
        "ML_IALGO_LINREG": "4",
        "ML_IWEIGHT": "3",
        "ML_WTOTEN": "1.0",
        "ML_WTIFOR": "1.0",
        "ML_WTSIF": "8.0",       # refit 阶段加倍应力权重 (优先保证弹性常数拟合精度)
    },
    "run": {
        "ML_LMLFF": ".TRUE.",
        "ML_MODE": "run",
        "ML_ISTART": "2",        # 使用已有力场 (快速模式)
    },
}

# 验证模式: 在 run 阶段同时做 DFT 对比
MLFF_VALIDATE_PARAMS = {
    "ML_LMLFF": ".TRUE.",
    "ML_MODE": "run",
    "ML_ISTART": "1",
    "ML_OUTPUT": "2",            # 输出详细误差信息到 ML_LOGFILE
    "ML_NMDINT": "50",           # 每 50 步做一次 DFT 对比
}


# ============================================================
# 默认 NSW
# ============================================================

DEFAULT_NSW = {
    "train": 2000,
    "refit": 1,        # ML_MODE=refit 为静态重拟合, 不跑 MD, NSW=1
    "run": 10000,
    "validate": 1000,  # 验证模式
}


def build_incar_params(
    ensemble: str,
    mlff_stage: str,
    temperature: float,
    nsw: Optional[int] = None,
    potim: float = 1.0,
    encut: Optional[float] = None,
    user_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    组装完整的 INCAR 参数字典。

    合并顺序: 高精度基础 → 系综参数 → MLFF 阶段参数 → 温度/步数 → 用户覆盖
    后合并的参数覆盖先前的同名参数。

    Parameters
    ----------
    ensemble : "npt" | "nvt"
    mlff_stage : "train" | "refit" | "run" | "validate"
    temperature : 目标温度 (K)
    nsw : MD 步数，None 则使用默认值
    potim : 时间步长 (fs)
    encut : 截断能 (eV)，None 则不设置（使用 VASP 默认）
    user_overrides : 用户自定义参数覆盖
    """
    params = dict(BASE_HIGH_PRECISION)

    # 系综参数
    if ensemble.lower() == "npt":
        params.update(NPT_MD_BASE)
    elif ensemble.lower() == "nvt":
        params.update(NVT_MD_BASE)
    else:
        raise ValueError(f"未知系综: {ensemble}，可选: npt, nvt")

    # MLFF 阶段参数
    if mlff_stage == "validate":
        params.update(MLFF_VALIDATE_PARAMS)
    elif mlff_stage in MLFF_PARAMS:
        params.update(MLFF_PARAMS[mlff_stage])
    else:
        raise ValueError(f"未知 MLFF 阶段: {mlff_stage}，可选: train, refit, run, validate")

    # 温度和步数
    params["TEBEG"] = str(int(temperature))
    params["TEEND"] = str(int(temperature))
    params["POTIM"] = str(potim)
    params["NSW"] = str(nsw if nsw is not None else DEFAULT_NSW.get(mlff_stage, 2000))

    if encut is not None:
        params["ENCUT"] = str(encut)

    # 用户覆盖（最高优先级）
    if user_overrides:
        params.update(user_overrides)

    return params


def adapt_user_incar(
    user_params: Dict[str, str],
    ensemble: str,
    temperature: float,
    nsw: Optional[int] = None,
    mlff_run: bool = False,
    n_species: Optional[int] = None,
    potim: float = 1.0,
) -> Dict[str, str]:
    """
    基于用户提供的 INCAR 参数，强制 MD 控制参数生成 NPT/NVT INCAR。

    继承用户的电子结构参数（PREC, EDIFF, ENCUT, GGA, ISMEAR, ALGO 等），
    但**强制**分子动力学所必需的控制参数。这样即使用户提供的是几何弛豫型
    INCAR (IBRION=2, 无 MDALGO/POTIM/热浴)，也能正确生成有限温 NPT/NVT MD，
    而不会退化成 0 K 几何弛豫。

    强制覆盖的 MD 控制参数:
      - IBRION=0        (MD; 覆盖用户的 IBRION=2 弛豫)
      - ISYM=0          (MD 关闭对称)
      - ISIF            (NPT=3 允许晶格变化 / NVT=2 固定晶格算应力)
      - MDALGO          (NPT=3 Langevin / NVT 默认 2 Nosé-Hoover, 用户已指定则尊重)
      - POTIM           (缺失则给默认时间步长)
      - 热浴参数         (NPT: LANGEVIN_GAMMA/LANGEVIN_GAMMA_L/PMASS;
                          NVT Nosé-Hoover: SMASS; 缺失则补默认值)

    Parameters
    ----------
    user_params : 用户 INCAR 的键值对字典
    ensemble : "npt" | "nvt"
    temperature : 目标温度 (K)
    nsw : MD 步数，None 则保留用户原值
    mlff_run : True 时强制 MLFF run 模式 (ML_MODE=run, ML_ISTART=2),
        并剔除训练专用参数。弹性常数计算必须用同一已训练力场跑 run 模式,
        否则每个变形目录会重新在线训练、力场不一致 → 弹性常数无意义。
    n_species : 元素种类数 (用于展开 LANGEVIN_GAMMA)
    potim : 缺省时间步长 (fs)，仅当用户 INCAR 未提供 POTIM 时使用

    Returns
    -------
    修改后的参数字典（不改变原字典）
    """
    params = dict(user_params)
    ens = ensemble.lower()
    ns = n_species if (n_species and n_species > 0) else 1

    # ---- 强制 MD 控制参数 (覆盖用户的弛豫型设置) ----
    # MD 必须 IBRION=0; 用户若给弛豫型 (IBRION=2/1/-1) 会在此被强制覆盖,
    # 否则生成的是几何能量最小化而非有限温分子动力学。
    params["IBRION"] = "0"
    params["ISYM"] = "0"
    # POTIM: 保留用户值, 否则给默认时间步长 (IBRION=0 缺 POTIM 会导致 VASP 报错)
    if "POTIM" not in params:
        params["POTIM"] = str(potim)

    if ens == "npt":
        params["ISIF"] = "3"            # 允许晶格变化 (热膨胀)
        params["MDALGO"] = "3"          # NPT 必须用 Langevin 恒温/恒压器
        if "PSTRESS" not in params:
            params["PSTRESS"] = "0"
        # Langevin NPT 热浴 + 晶格质量参数, 缺失则给合理默认
        if "LANGEVIN_GAMMA" not in params:
            params["LANGEVIN_GAMMA"] = " ".join(["10.0"] * ns)
        if "LANGEVIN_GAMMA_L" not in params:
            params["LANGEVIN_GAMMA_L"] = "1.0"
        if "PMASS" not in params:
            params["PMASS"] = "1000"
        # Langevin 下 SMASS 无意义, 移除以免误导
        params.pop("SMASS", None)
    elif ens == "nvt":
        params["ISIF"] = "2"            # 固定晶格, 计算应力
        # 热浴: 用户已显式指定 MDALGO 则尊重, 否则默认 Nosé-Hoover
        mdalgo = str(params.get("MDALGO", "2"))
        params["MDALGO"] = mdalgo
        if mdalgo == "3":               # Langevin NVT
            if "LANGEVIN_GAMMA" not in params:
                params["LANGEVIN_GAMMA"] = " ".join(["10.0"] * ns)
        else:                           # Nosé-Hoover 需要 SMASS
            if "SMASS" not in params:
                params["SMASS"] = "2"
        # NVT 不改变晶格, 移除 NPT 专用参数以免误导
        for k in ("PSTRESS", "LANGEVIN_GAMMA_L", "PMASS"):
            params.pop(k, None)

    # 温度
    params["TEBEG"] = str(int(temperature))
    params["TEEND"] = str(int(temperature))

    # NSW（如果指定）
    if nsw is not None:
        params["NSW"] = str(nsw)

    # LANGEVIN_GAMMA (Langevin 恒温器) 需按元素种类给 N 个值。
    # 若用户只给了 1 个值 (或不足 N 个) 而体系是多元素, 自动按最后一个值补齐。
    if n_species and n_species > 1 and "LANGEVIN_GAMMA" in params:
        toks = str(params["LANGEVIN_GAMMA"]).split()
        if 0 < len(toks) < n_species:
            toks = (toks + [toks[-1]] * n_species)[:n_species]
            params["LANGEVIN_GAMMA"] = " ".join(toks)

    # MLFF run 模式: 用已训练 ML_FF 跑, 不再训练 (弹性常数计算必须如此)
    if mlff_run:
        params["ML_LMLFF"] = ".TRUE."
        params["ML_MODE"] = "run"
        params["ML_ISTART"] = "2"
        # 剔除训练专用参数 (run 模式无意义且易误导)
        for k in ("ML_CX", "ML_MB", "ML_MCONF_NEW", "ML_LBASIS_DISCARD",
                  "ML_IALGO_LINREG", "ML_IWEIGHT", "ML_WTOTEN", "ML_WTIFOR",
                  "ML_WTSIF", "ML_EPS_LOW", "ML_SION1", "ML_SION2"):
            params.pop(k, None)

    return params
