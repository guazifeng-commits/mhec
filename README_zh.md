# M-HEC

**MLFF-Enhanced High-Temperature Elastic Constants Calculations**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](https://github.com/mhec)

M-HEC 是一个计算**高温弹性常数**的工具包，基于 VASP 的从头算分子动力学（AIMD）与机器学习力场（MLFF）加速，并集成了 AIMD 轨迹后处理与弹性矩阵后处理分析。

---

## 主要功能

### 弹性常数计算（两种方法）

- **应力-应变法（SS）**：按晶系对称性自适应施加单分量有限应变，读取 MD 时间平均应力，线性拟合 σ = C·ε。与 VASPKIT 同源，但给出**有限温度**弹性常数。
- **体积守恒法（VC / 变形元胞法）**：使用体积守恒变形（正交 [δ,−δ,0]、体积守恒剪切等），消除体积变化对应力的影响，对剪切常数与立方 C11−C12 信噪比更好。
- 支持全部 7 大晶系，自动识别晶系。
- 自动施加晶系对称性约束、Born 力学稳定性判据、Wallace 残余压力（Birch↔Brugger）诊断。

### 机器学习力场（MLFF）

- **定温训练**：每个温度一条 NPT 轨迹（grid 模式）。
- **升温 ramp 训练**：一条 NPT 升温轨迹（TEBEG→TEEND）覆盖整个温区，再在该力场基础上对各应变方向**顺序续训**，任务数大幅减少。
- 训练 INCAR 默认在保证能量/力的前提下**加大应力权重**（`ML_IWEIGHT=3`，`ML_WTSIF=4.0`），以满足弹性常数对应力精度的要求。
- **精度验证（三种方式）**：
  - 解析 `ML_LOGFILE` 训练误差（ERR/BEEF/STDAB）；
  - 从 MLFF 轨迹抽帧做 DFT 单点的 parity 验证；
  - 基于 `ML_AB` 全数据集的 parity 验证（对每个训练构型用 `ML_MODE=run` 重新预测）。

### 后处理分析

- **AIMD 后处理**：RDF g(r)、配位数、MSD 与扩散系数（爱因斯坦关系，单位 m²/s）、VACF/PDOS、Stokes-Einstein 与 Green-Kubo 粘度、分子摩擦系数等。
- **弹性矩阵后处理**：VRH 平均、各向异性指数、方向性杨氏/剪切模量与泊松比、2D/3D 出图、Origin 兼容数据导出、德拜温度与声速等。

---

## 安装

```bash
# 从源码包安装
pip install dist/mhec-1.0.0-py3-none-any.whl
# 或
pip install dist/mhec-1.0.0.tar.gz

# 开发模式
pip install -e .
```

安装后命令行入口为 `mhec`。依赖：Python ≥ 3.8、numpy、scipy；绘图需 matplotlib（可选）。

HPC 集群上也可不安装直接运行：
```bash
export PYTHONPATH="/path/to/mhec:$PYTHONPATH"
python -m mhec   # 或 python /path/to/mhec/mhec/cli.py
```

---

## 快速开始

直接运行 `mhec` 进入交互菜单（VASPKIT 风格）：

```
 ────────────────────────────────────────
  弹性常数计算
 ────────────────────────────────────────
   1) 应力-应变法 · 生成计算输入
   2) 应力-应变法 · 拟合弹性常数
   3) 体积守恒法 · 生成计算输入
   4) 体积守恒法 · 拟合弹性常数

 ────────────────────────────────────────
  后处理分析
 ────────────────────────────────────────
   5) 力学性质后处理
   6) AIMD 后处理

 ────────────────────────────────────────
  机器学习力场
 ────────────────────────────────────────
   7) MLFF 训练 (定温)
   8) MLFF 训练 (升温 ramp)
   9) MLFF 精度验证

 ────────────────────────────────────────
  辅助工具
 ────────────────────────────────────────
  10) 晶系识别与晶格参数
  11) 变形模式与应变方案
  12) 生成 INCAR / KPOINTS
  13) 平衡晶格与热膨胀系数

  s) 修改设置    0) 退出
```

---

## 典型工作流（MLFF 加速）

```bash
# 1. 准备 POSCAR, POTCAR, KPOINTS, run.sh

# 2. 训练 MLFF（菜单 7 定温 或 8 升温 ramp）
#    完成后得到力场: <训练目录>/refit_final/ML_FFN
cp <训练目录>/refit_final/ML_FFN ML_FF

# 3. 验证力场精度（菜单 9）—— 重点关注应力 parity

# 4. 生成弹性常数计算输入（菜单 1 应力-应变法 / 3 体积守恒法）

# 5. 提交计算后，拟合弹性常数（菜单 2 / 4）

# 6. 力学性质后处理与出图（菜单 5）
```

---

## 两种弹性方法对比

| 特性 | 应力-应变法 (SS) | 体积守恒法 (VC) |
|------|-----------------|-----------------|
| 变形 | 单分量单轴/剪切 | 体积守恒变形 |
| 体积变化 | ΔV/V ≈ δ | ΔV/V ≈ 0 |
| 剪切/C11−C12 精度 | 良好 | 更好 |
| 适用 | 通用 | 难算的剪切常数、立方 C12 |

两种方法都按晶系对称性确定最小独立变形集，并输出该晶系全部独立弹性常数。

---

## 配置

设置保存在工作目录的 `mhec.yaml`（生成计算输入时自动写出），拟合时自动读取以保证应变幅度等参数一致。常用参数：温度列表、应变幅度 δ、应变点数、`skip_steps`、ENCUT、各阶段 NSW、POTCAR/ML_FF 路径等。

---

## 依赖

- Python ≥ 3.8
- NumPy ≥ 1.18.0
- SciPy ≥ 1.4.0
- matplotlib ≥ 3.0.0（可选，绘图）

---

## 作者

- Xing Feng
- Zhongkan Ren
- Yuan Yu
- Huaguo Tang
- Zhijian Wu
- Zhuhui Qiao（通讯作者：zhqiao@licp.cas.cn）

## 许可

MIT License

## 引用

If you use M-HEC in your research, please cite:

```
Feng X, Ren Z, Yu Y, Tang H, Wu Z, Qiao Z. M-HEC: A Python Package for
MLFF-Enhanced High-Temperature Elastic Constants Calculations.
Computer Physics Communications, 2026.
```

## 参考文献

- Wen et al., J. Appl. Phys. 113, 103501 (2013) — 体积守恒变形法
- Zhao et al., Phys. Rev. B 75, 094105 (2007) — 弹性常数计算
- Liu et al., arXiv:2002.00005 — OHESS/ULICS 应变矩阵集
- Allen & Tildesley, Computer Simulation of Liquids (1987) — RDF/MSD/VACF
- Green, J. Chem. Phys. 22, 398 (1954) — Green-Kubo 粘度
