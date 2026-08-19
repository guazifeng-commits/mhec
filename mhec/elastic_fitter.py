"""
弹性常数多点线性回归拟合与对称性修正。
"""

import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from .crystal_system import CrystalSystem
from .symmetry import get_symmetry_rules
from .strain import decode_deform


# ============================================================
#  单通道稳健斜率拟合 (自动判据: 奇多项式 vs 线性 vs 小应变缩窗)
# ============================================================
# 软框架 (SiO2/BeF2 等) 的最软方向在大应变或低温下会强烈非简谐甚至非单调,
# 此时过原点奇多项式 (σ=a·ε+b·ε³) 会过拟合出乱跳的 R²(甚至负值), 而全程线性
# 又被大应变坏点带偏。下面的判据在每个应力通道上自动选择最稳健的模型:
#   1. 全范围: 奇多项式 R² 若不优于过原点线性, 就退回线性 (避免三次过拟合)。
#   2. 若所选全范围模型 R² 仍 < R2_GOOD_FIT, 按 |ε| 从大到小逐层剔除外圈点,
#      取仍能达到 R² ≥ R2_GOOD_FIT 的最大小应变窗口 (过原点线性), 标记为
#      "linear_narrowed"。这样能在软通道自动回到线性区, 恢复物理斜率。
R2_GOOD_FIT = 0.9


def _lin_through_origin(x: np.ndarray, y: np.ndarray):
    """过原点线性 y=slope·x, 返回 (slope, r2, stderr)。"""
    sxx = float(np.dot(x, x))
    if sxx <= 0:
        return 0.0, 1.0, 0.0
    slope = float(np.dot(x, y) / sxx)
    y_pred = slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    dof = max(1, len(x) - 1)
    stderr = float(np.sqrt(ss_res / dof) / np.sqrt(sxx))
    return slope, r2, stderr


def _odd_cubic_through_origin(x: np.ndarray, y: np.ndarray):
    """过原点奇多项式 y=a·x+b·x³, 返回 (a, r2, stderr, coef)。"""
    A = np.column_stack([x, x ** 3])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    y_pred = A @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    sxx = float(np.dot(x, x))
    dof = max(1, len(x) - 2)
    stderr = float(np.sqrt(ss_res / dof) / np.sqrt(sxx)) if sxx > 0 else 0.0
    return slope, r2, stderr, coef


def _odd_even_slope(x, y, allow_cubic: bool = True,
                    tol: float = 1e-9) -> Optional[Dict]:
    """由奇偶分离提取 eps->0 斜率; 幅度不对称或档数不足时返回 None。

    为什么这是正确做法
    ------------------
    sigma(eps) = delta + C eps + (1/2) C3 eps^2 + (1/6) C4 eps^3 + ...
      奇部: C eps + (1/6) C4 eps^3     <- 只含二阶与四阶
      偶部: delta + (1/2) C3 eps^2     <- 残余应力与三阶
    取奇部即**精确**除去残余应力与三阶贡献, 不依赖"它们足够小"。
    过原点直接拟合做不到这一点: 偶部无处可去, 只能变成残差压低 R2,
    进而触发丢点 (narrow), 反而损失统计量。

    返回值里额外给出 even_max, 即偶部的最大幅度, 作为残余应力/非谐的诊断。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mags = sorted({round(float(abs(v)), 12) for v in x if abs(v) > 1e-15})
    if not mags:
        return None

    xs, ys, evens = [], [], []
    for m in mags:
        ip = np.where(np.abs(x - m) <= tol + 1e-12 * m)[0]
        im = np.where(np.abs(x + m) <= tol + 1e-12 * m)[0]
        if ip.size == 0 or im.size == 0:
            return None                     # 该幅度没有成对的正负点 -> 不对称
        yp = float(np.mean(y[ip])); ym = float(np.mean(y[im]))
        xs.append(m)
        ys.append(0.5 * (yp - ym))          # 奇部
        evens.append(0.5 * (yp + ym))       # 偶部 (诊断用)
    xs = np.asarray(xs); ys = np.asarray(ys)
    if xs.size < 1:
        return None

    even_max = float(np.max(np.abs(evens))) if evens else 0.0

    # 奇部再拟合: 点数够就用 a1*x + a3*x^3, 否则过原点线性
    if allow_cubic and xs.size >= 3:
        slope, r2, se, coef = _odd_cubic_through_origin(xs, ys)
        kind = "odd_cubic"
        if not np.isfinite(slope):
            slope, r2, se = _lin_through_origin(xs, ys)
            kind, coef = "odd_linear", None
    else:
        slope, r2, se = _lin_through_origin(xs, ys)
        kind, coef = "odd_linear", None

    return {"slope": float(slope), "r2": float(r2), "stderr": float(se),
            "kind": kind, "n_used": int(x.size), "coef": coef,
            "even_max": even_max, "n_amplitudes": int(xs.size)}


def robust_slope_fit(x, y, allow_cubic: bool = True, r2_good: float = R2_GOOD_FIT) -> Dict:
    """对单个应力通道 (x=应变, y=应力变化) 做稳健的 ε→0 斜率拟合。

    返回 dict:
      slope  : ε→0 斜率 (未做 kBar→GPa / 符号变换, 与旧 lstsq 斜率同量纲)
      r2     : 最终所选模型的 R²
      stderr : 斜率标准误
      kind   : 'cubic' | 'linear' | 'linear_narrowed'
      n_used : 参与最终拟合的点数
      coef   : 奇多项式系数 (仅 kind=='cubic' 时给出, 供画曲线), 否则 None
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {"slope": 0.0, "r2": 1.0, "stderr": 0.0,
           "kind": "linear", "n_used": int(len(x)), "coef": None}
    sxx = float(np.dot(x, x))
    if sxx <= 0:
        return out

    # ---- 优先用奇偶分离 (物理上最干净的 eps->0 斜率估计) ----
    # 应力对应变的展开: sigma = delta + C eps + (1/2)C3 eps^2 + (1/6)C4 eps^3
    # 二阶常数 C 住在**奇部**, 而残余应力 delta 与三阶项 C3 都住在**偶部**。
    # 若幅度关于 0 对称, 取
    #     odd(eps) = [sigma(+eps) - sigma(-eps)] / 2 = C eps + (1/6)C4 eps^3
    # 就**精确**消掉 delta 与 C3, 不需要它们足够小的假设。
    # 这解决了实测中"逐点表观斜率乱跳"的现象: 跳动全在偶部, 奇部是稳的。
    odd = _odd_even_slope(x, y, allow_cubic=allow_cubic)
    if odd is not None:
        return odd

    n_nonzero = int(np.count_nonzero(np.abs(x) > 1e-15))
    use_cubic = bool(allow_cubic) and (n_nonzero >= 6)

    slope_lin, r2_lin, se_lin = _lin_through_origin(x, y)
    if use_cubic:
        slope_cub, r2_cub, se_cub, coef_cub = _odd_cubic_through_origin(x, y)
        # 奇多项式只有在 R² 不劣于线性时才采用, 否则说明它在过拟合
        if r2_cub >= r2_lin - 1e-9:
            slope, r2, se, kind, coef = slope_cub, r2_cub, se_cub, "cubic", coef_cub
        else:
            slope, r2, se, kind, coef = slope_lin, r2_lin, se_lin, "linear", None
    else:
        slope, r2, se, kind, coef = slope_lin, r2_lin, se_lin, "linear", None
    n_used = int(len(x))

    # 若仍拟合很差, 尝试缩到小应变线性窗口
    if r2 < r2_good and abs(slope) > 1e-12:
        mags = np.abs(x)
        tiers = sorted({round(float(m), 12) for m in mags if m > 1e-15})
        # 从「次大窗口」向「最小窗口」搜索, 取仍能达到阈值的最大窗口
        for k in range(len(tiers) - 2, -1, -1):
            thr = tiers[k]
            mask = mags <= thr + 1e-15
            if int(np.count_nonzero(mags[mask] > 1e-15)) < 2:
                continue
            sw, r2w, sew = _lin_through_origin(x[mask], y[mask])
            if r2w >= r2_good and r2w > r2 + 1e-9:
                slope, r2, se = sw, r2w, sew
                kind = "linear_narrowed"
                coef = None
                n_used = int(np.count_nonzero(mask))
                break

    out.update(slope=slope, r2=r2, stderr=se, kind=kind, n_used=n_used, coef=coef)
    return out


# ============================================================
#  对称性参数化的联立最小二乘求解器 (SC 与 VC 共用)
# ============================================================
# 思路 (与文献 stress-strain 法一致, 如 Le Page & Saxe 2002、de Jong 2015):
#   不把「一个变形 → 一个/一列常数」直接填表, 而是把所有变形的所有应力分量
#   σ_i = Σ_j C_ij ε_j 拼成一个超定线性方程组, 按晶系对称性把 C 参数化为少数
#   独立常数, 一次性最小二乘解出全部。每个方程 (变形 × 应力分量) 再按其线性
#   拟合质量 R² 加权, 使强信号通道 (如正交偏量测 C11-C12、a-c 偏量测 C13)
#   主导, 弱横向 / 高噪声通道 (低或负 R²) 权重趋零, 从而消除「同一常数出现
#   多个互相矛盾的估计」的问题。

def _param_layout(crystal_system: CrystalSystem):
    """按对称性建立 C 矩阵的独立参数布局。返回 (params, pos_to_param, rules, zeros)。"""
    rules = get_symmetry_rules(crystal_system)

    zeros = set()
    for (i, j) in rules.zero_entries:
        zeros.add((i, j))
        zeros.add((j, i))

    derived_targets = set()
    for rel in rules.derived_relations:
        ti, tj = rel["target"]
        derived_targets.add((ti, tj))
        derived_targets.add((tj, ti))

    params: List[Tuple[int, int]] = []
    pos_to_param: Dict[Tuple[int, int], int] = {}

    for group in rules.equivalent_groups:
        pidx = len(params)
        params.append(group[0])
        for (i, j) in group:
            pos_to_param[(i, j)] = pidx
            pos_to_param[(j, i)] = pidx

    for i in range(6):
        for j in range(i, 6):
            if (i, j) in zeros or (i, j) in derived_targets or (i, j) in pos_to_param:
                continue
            pidx = len(params)
            params.append((i, j))
            pos_to_param[(i, j)] = pidx
            pos_to_param[(j, i)] = pidx

    return params, pos_to_param, rules, zeros


def _basis_matrix(p_index: int, pos_to_param, rules) -> np.ndarray:
    """第 p_index 个独立参数=1 (其余=0) 对应的 6×6 C 矩阵 (含派生关系)。"""
    C = np.zeros((6, 6))
    for (i, j), pidx in pos_to_param.items():
        if pidx == p_index:
            C[i, j] = 1.0
    for rel in rules.derived_relations:
        ti, tj = rel["target"]
        v = rel["func"](C)
        C[ti, tj] = v
        C[tj, ti] = v
    return C


def solve_cij_weighted(crystal_system: CrystalSystem, mode_data: List[Tuple]) -> np.ndarray:
    """
    加权联立最小二乘求解 6×6 弹性常数矩阵。

    Parameters
    ----------
    crystal_system : 晶系
    mode_data : [(direction d (6,), slopes (6,), weights (6,)), ...]
        d       : 单位幅度应变方向 (线性), 使 slope = C @ d
        slopes  : 该变形拟合得到的 6 个应力斜率 (GPa)
        weights : 每个应力分量方程的权重 (>=0, 建议用 clip(R²,0,1)²)

    Returns
    -------
    cij_raw : (6,6) 未对称化的弹性常数矩阵
    """
    params, pos_to_param, rules, _zeros = _param_layout(crystal_system)
    nparam = len(params)
    Bs = [_basis_matrix(p, pos_to_param, rules) for p in range(nparam)]

    rows, rhs, wts = [], [], []
    for d, slopes, weights in mode_data:
        Bd = [Bs[p] @ d for p in range(nparam)]   # 预算 B_p·d
        for i in range(6):
            w = float(weights[i])
            if w <= 0.0:
                continue
            rows.append([float(Bd[p][i]) for p in range(nparam)])
            rhs.append(float(slopes[i]))
            wts.append(w)

    if not rows:
        raise ValueError("没有有效 (权重>0) 的拟合方程")

    A = np.array(rows)
    b = np.array(rhs)
    wsq = np.sqrt(np.array(wts))
    Aw = A * wsq[:, None]
    bw = b * wsq
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    cij = np.zeros((6, 6))
    for p in range(nparam):
        cij += x[p] * Bs[p]

    # ---- 不确定度: 由加权残差估参数协方差, 再传播到每个 Cij ----
    #   加权最小二乘残差 rw = Aw·x − bw; 约化方差 s² = rwᵀrw / (m − nparam)
    #   参数协方差 cov(x) = s² (AwᵀAw)⁻¹; Cij = Σ_p x_p B_p[i,j] 为线性组合,
    #   故 var(Cij) = gᵀ cov(x) g, g_p = B_p[i,j]。反映"各强通道彼此吻合程度"。
    cij_std = np.zeros((6, 6))
    m = Aw.shape[0]
    dof = max(1, m - nparam)
    rw = Aw @ x - bw
    s2 = float(rw @ rw) / dof
    AtA = Aw.T @ Aw
    try:
        cov_x = s2 * np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        cov_x = s2 * np.linalg.pinv(AtA)
    Bmat = np.array([[Bs[p][i, j] for p in range(nparam)]
                     for i in range(6) for j in range(6)])  # (36, nparam)
    for k in range(36):
        g = Bmat[k]
        var = float(g @ cov_x @ g)
        cij_std[k // 6, k % 6] = np.sqrt(var) if var > 0 else 0.0
    return cij, cij_std


_WEIGHT_FLOOR = 1.0e-3


def _quality_weights(r_squared: np.ndarray) -> np.ndarray:
    """
    由每分量 R² 生成通道权重: w = max(clip(R²,0,1)², floor)。
    干净通道 (R²≈1) 权重≈1 主导求解; 噪声/负 R² 通道权重降到 floor (≈1e-3),
    基本不参与但保留极小地板, 防止某常数所有通道都弱时线性系统欠定。
    """
    q = np.clip(np.asarray(r_squared, dtype=float), 0.0, 1.0)
    return np.maximum(q * q, _WEIGHT_FLOOR)


class ElasticFitter:
    """弹性常数多点线性回归拟合。"""

    def __init__(self, crystal_system: CrystalSystem):
        self.crystal_system = crystal_system

    def fit_single_mode(
        self,
        strains: np.ndarray,
        stresses: np.ndarray,
        baseline_stress: np.ndarray,
        fit_order: int = 3,
    ) -> Tuple[np.ndarray, Dict]:
        """
        对单个 deform code 的多点数据进行回归, 提取 ε→0 线性弹性斜率。

        fit_order=3 (默认): 过原点奇多项式 σ = C1·ε + C3·ε³, 取 C1。
            单轴拉/压的应力-应变曲线因非谐性并不对称, 直接线性拟合会被
            有限应变下的奇次非谐项 (D·ε²) 偏置 (如立方 C12 偏高)。奇多项式
            显式分离立方项, 给出真正的 ε→0 弹性常数, 且天然忽略偶部 (体积项)。
        fit_order=1: 过原点线性拟合 (旧行为)。
        数据点不足时自动退回线性。

        Parameters
        ----------
        strains : (N_points,) 应变幅度数组
        stresses : (N_points, 6) 对应的时间平均应力
        baseline_stress : (6,) 未变形参考构型的平均应力

        Returns
        -------
        slopes : (6,) 斜率，即 C 矩阵对应列的分量 (GPa)
        fit_info : {r_squared, stderr, fit_order}
        """
        # 减去基线
        delta_stress = stresses - baseline_stress[np.newaxis, :]

        slopes = np.zeros(6)
        r_squared = np.zeros(6)
        stderr = np.zeros(6)
        fit_kind = ["linear"] * 6
        n_used = np.zeros(6, dtype=int)

        x = np.asarray(strains, dtype=float)
        n_nonzero = int(np.count_nonzero(np.abs(x) > 1e-15))
        # 奇多项式 (a·x+b·x³) 有 2 个参数, 且 x³ 列与 x 列量级相差极大 (条件数差),
        # 用 4 个点拟合会过参数化, 三次项乱跳并把线性斜率带偏 (BeF2 案例中即此)。
        # 要求至少 6 个非零点 (默认 n_points=9 → 8 点) 才启用三次, 否则退回过原点线性。
        # 具体的「奇多项式 / 线性 / 小应变缩窗」自动判据见 robust_slope_fit。
        allow_cubic = (fit_order >= 3)

        for i in range(6):
            if np.all(x == 0):
                continue
            res = robust_slope_fit(x, delta_stress[:, i], allow_cubic=allow_cubic)
            slopes[i] = res["slope"]
            r_squared[i] = res["r2"]
            stderr[i] = res["stderr"]
            fit_kind[i] = res["kind"]
            n_used[i] = res["n_used"]

        # kBar -> GPa 转换 (VASP 输出应力单位为 kBar, 1 kBar = 0.1 GPa)
        # 符号修正: VASP 输出的是压力（正值=压缩），而弹性常数定义为
        # σ_i = C_ij * ε_j（正值=拉伸），因此 P_i = -σ_i = -C_ij * ε_j
        # 斜率 dP/dε = -C_ij，需要取负号得到 C_ij
        slopes *= -0.1
        stderr *= 0.1

        fit_info = {
            "r_squared": r_squared,
            "stderr": stderr,
            "fit_order": 3 if (allow_cubic and n_nonzero >= 6) else 1,
            "fit_kind": fit_kind,     # 每个应力通道最终采用的拟合方式
            "n_used": n_used,         # 每个通道参与拟合的点数 (缩窗后 < 总点数)
        }
        return slopes, fit_info

    def assemble_cij_matrix(
        self,
        deform_codes: List[str],
        slopes_dict: Dict[str, np.ndarray],
        magnitude: float,
    ) -> np.ndarray:
        """
        从各 deform code 的线性回归斜率组装 6×6 弹性常数矩阵。

        对于 standard 方案（单分量变形）：
            deform code 的非零位指示 C 矩阵的列索引，斜率直接填入对应列。

        对于 ULICS 方案（多分量变形）：
            构建应变设计矩阵 A (N_modes × 6)，通过最小二乘求解 C 矩阵。
            关系: Δσ_i = Σ_j C_ij * ε_j
            对每个应力分量 i，有 slopes_i = C_i,: @ (ε_normalized)
            即 slopes_dict[code][i] = Σ_j C_ij * (base_strain_j / magnitude)
        """
        # 检查是否所有 deform code 都是单分量
        all_single = True
        for code in deform_codes:
            base_strain = decode_deform(code, magnitude)
            nonzero_idx = np.where(np.abs(base_strain) > 1e-15)[0]
            if len(nonzero_idx) != 1:
                all_single = False
                break

        cij = np.zeros((6, 6))
        # 记录哪些 (i,j) 元素已被直接测量
        measured = np.zeros((6, 6), dtype=bool)

        if all_single:
            # Standard 方案：直接填入已测量列
            for code in deform_codes:
                base_strain = decode_deform(code, magnitude)
                j = np.where(np.abs(base_strain) > 1e-15)[0][0]
                cij[:, j] = slopes_dict[code]
                measured[:, j] = True

            # 利用 Cij = Cji 对称性传播已测量值到转置位置
            for i in range(6):
                for j in range(6):
                    if measured[i, j] and not measured[j, i]:
                        cij[j, i] = cij[i, j]
                        measured[j, i] = True

            # 对于高对称晶系（deform codes < 6），部分对角元和非对角元
            # 仍未被测量。利用对称性等价关系传播已测量值。
            # 例如立方: C11 已测量，C22=C11, C33=C11 未测量 → 复制
            rules = get_symmetry_rules(self.crystal_system)
            for group in rules.equivalent_groups:
                # 找出组内已测量的元素值
                measured_vals = [cij[i, j] for i, j in group if measured[i, j]]
                if measured_vals:
                    avg_val = np.mean(measured_vals)
                    for i, j in group:
                        if not measured[i, j]:
                            cij[i, j] = avg_val
                            cij[j, i] = avg_val
                            measured[i, j] = True
                            measured[j, i] = True
        else:
            # ULICS/多分量方案：构建设计矩阵，最小二乘求解
            # slopes_dict[code][i] 是 Δσ_i 对 ε_applied 的斜率
            # 实际关系: slopes[i] * magnitude = Σ_j C_ij * base_strain_j
            # 即: slopes[i] = Σ_j C_ij * (base_strain_j / magnitude)
            # 设 A[k, j] = base_strain_j / magnitude (归一化应变系数)
            # 则 slopes[k, i] = Σ_j A[k, j] * C_ij
            # 对每个应力分量 i: slopes[:, i] = A @ C[i, :]
            # 最小二乘: C[i, :] = (A^T A)^{-1} A^T slopes[:, i]

            n_modes = len(deform_codes)
            A = np.zeros((n_modes, 6))
            S = np.zeros((n_modes, 6))  # slopes matrix

            for k, code in enumerate(deform_codes):
                base_strain = decode_deform(code, magnitude)
                A[k, :] = base_strain / magnitude
                S[k, :] = slopes_dict[code]

            # 最小二乘求解每个应力分量
            ATA = A.T @ A
            try:
                ATA_inv = np.linalg.inv(ATA)
                for i in range(6):
                    cij[i, :] = ATA_inv @ A.T @ S[:, i]
            except np.linalg.LinAlgError:
                # 矩阵奇异，使用伪逆
                for i in range(6):
                    cij[i, :], _, _, _ = np.linalg.lstsq(A, S[:, i], rcond=None)

        return cij

    def symmetrize(self, cij_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对称性修正弹性常数矩阵。

        步骤:
            (a) 零项强制置零
            (b) 等价项取平均
            (c) 派生关系重新计算
            (d) 强制 Cij = Cji
        """
        rules = get_symmetry_rules(self.crystal_system)
        cij = cij_raw.copy()
        corrections = {
            "zero_corrections": {},
            "average_corrections": {},
            "derived_corrections": {},
            "warnings": [],
        }

        # 计算非零项平均值（用于诊断）
        nonzero_vals = []
        for i in range(6):
            for j in range(i, 6):
                if (i, j) not in rules.zero_entries and (j, i) not in rules.zero_entries:
                    if abs(cij[i, j]) > 1e-10:
                        nonzero_vals.append(abs(cij[i, j]))
        avg_nonzero = np.mean(nonzero_vals) if nonzero_vals else 1.0

        # (a) 零项强制置零
        for i, j in rules.zero_entries:
            old_val = cij[i, j]
            if abs(old_val) > 1e-10:
                corrections["zero_corrections"][(i, j)] = old_val
                if abs(old_val) > 0.10 * avg_nonzero:
                    msg = (f"C{i+1}{j+1} 应为零但原始值为 {old_val:.2f} GPa "
                           f"(>{avg_nonzero*0.10:.2f})，采样可能不足")
                    corrections["warnings"].append(msg)
                    warnings.warn(msg, UserWarning)
            cij[i, j] = 0.0
            cij[j, i] = 0.0

        # (b) 等价项取平均
        for group in rules.equivalent_groups:
            vals = [cij[i, j] for i, j in group]
            avg = np.mean(vals)
            for i, j in group:
                old_val = cij[i, j]
                corrections["average_corrections"][(i, j)] = avg - old_val
                cij[i, j] = avg
                cij[j, i] = avg

        # (c) 派生关系
        for rel in rules.derived_relations:
            ti, tj = rel["target"]
            new_val = rel["func"](cij)
            old_val = cij[ti, tj]
            corrections["derived_corrections"][(ti, tj)] = new_val - old_val
            cij[ti, tj] = new_val
            cij[tj, ti] = new_val

        # (d) 强制对称
        cij = (cij + cij.T) / 2

        corrections["max_correction"] = 0.0
        for d in [corrections["zero_corrections"],
                  corrections["average_corrections"],
                  corrections["derived_corrections"]]:
            if d:
                corrections["max_correction"] = max(
                    corrections["max_correction"],
                    max(abs(v) for v in d.values())
                )

        return cij, corrections

    def fit_all(
        self,
        deform_codes: List[str],
        all_strains: Dict[str, np.ndarray],
        all_stresses: Dict[str, np.ndarray],
        baseline_stress: np.ndarray,
        magnitude: float,
        fit_order: int = 3,
    ) -> Dict:
        """
        完整拟合流程。

        Parameters
        ----------
        deform_codes : deform code 列表
        all_strains : {deform_code: (N_points,) 应变幅度数组}
        all_stresses : {deform_code: (N_points, 6) 应力数组}
        baseline_stress : (6,) 基线应力
        magnitude : 基础应变幅度

        Returns
        -------
        dict : {cij_raw, cij_symmetrized, fit_info_per_mode,
                correction_info, born_stable, born_warnings}
        """
        from .born_stability import check_born_stability

        slopes_dict = {}
        fit_info_per_mode = {}
        mode_data = []

        for code in deform_codes:
            slopes, info = self.fit_single_mode(
                all_strains[code], all_stresses[code], baseline_stress,
                fit_order=fit_order,
            )
            slopes_dict[code] = slopes
            fit_info_per_mode[code] = info

            # 单位幅度应变方向 (线性), slope = C @ d
            d = decode_deform(code, magnitude) / magnitude
            weights = _quality_weights(info["r_squared"])
            mode_data.append((d, slopes, weights))
            # 注: 不再逐通道打印低 R² 警告 — 与 SA(VC) 流程保持一致, 弱通道的取舍
            # 统一由联立加权求解 + 求解来源报告(仅强通道)处理, 屏幕不再刷 R² 警告。

        # 按文献 stress-strain 法: 全部 (变形 × 应力分量) 联立、按 R² 加权最小二乘,
        # 一次解出所有独立常数 (取代旧的「单轴直接填列 + 转置平均」, 避免同一常数
        # 出现多个矛盾估计、且弱噪声通道拖偏)。
        cij_raw, cij_std = solve_cij_weighted(self.crystal_system, mode_data)
        cij_sym, correction_info = self.symmetrize(cij_raw)
        born_stable, born_warnings = check_born_stability(cij_sym, self.crystal_system)

        return {
            "cij_raw": cij_raw,
            "cij_symmetrized": cij_sym,
            "cij_std": cij_std,
            "fit_info_per_mode": fit_info_per_mode,
            "correction_info": correction_info,
            "born_stable": born_stable,
            "born_warnings": born_warnings,
        }
