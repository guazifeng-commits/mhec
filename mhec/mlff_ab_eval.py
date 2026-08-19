"""
基于 ML_AB 的 MLFF 全数据集精度验证 (eval_mlff 方法)。

与 mlff_validator 中"抽帧做 DFT 单点"的方法互补：
  - mlff_validator: 从 MLFF run 轨迹抽帧 → 重新做纯 DFT 单点 → 对比。
  - 本模块 (mlff_ab_eval): 直接以 ML_AB 中存储的 DFT 参考值为基准，
    对其中每个构型用训练好的力场 (ML_MODE=run) 重新预测 → 对比。
    等价于 VASP 的 ML_MODE=select，但可控、可断点、可直接出 E/F/S parity。

使用前提（从哪一步开始可用）:
  完成 MLFF 训练 (train → refit) 后，得到:
    - ML_FF  : 训练好的力场
    - ML_AB  : 训练数据集 (含每个构型的 DFT 能量/力/应力)
  即可用本模块验证整个训练集上的精度。

工作流:
  1) generate_ab_eval_dirs(...)  生成 struct_*/ 评估目录 + 提交脚本
  2) 提交 submit_ab_eval.sh       对每个构型做 ML_MODE=run 单点
  3) collect_ab_parity(...)       收集结果 → ParityData → parity 图
"""

import os
import re
import shutil
import numpy as np
from typing import Dict, List, Optional

# 复用 mlff_validator 的成对数据容器与绘图/报告, 保持输出一致
from .mlff_validator import ParityData, plot_parity, write_parity_report


# ============================================================
#  ML_AB 解析 (DFT 参考: energy / forces / stress + 生成 POSCAR)
# ============================================================

def _is_separator(s: str) -> bool:
    s = s.strip()
    return (not s) or set(s) <= set("*-=")


def _floats(tokens: List[str]) -> List[float]:
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def parse_ml_ab(filename: str) -> List[Dict]:
    """
    解析标准 VASP ML_AB 文件。每个 'Configuration num.' 块提取:
        natoms, 元素与数量, 晶格矢量, 笛卡尔坐标, 总能量, 力, 应力(6 分量)
    返回结构字典列表, 每个含:
        idx, natoms, dft_energy(eV), dft_forces((natoms,3) eV/Å),
        dft_stress((6,) Voigt[xx,yy,zz,yz,xz,xy] kBar), poscar(list[str])
    """
    with open(filename, "r", errors="ignore") as f:
        lines = f.readlines()
    n = len(lines)

    def first_data_line(header_idx):
        j = header_idx + 1
        while j < n:
            if not _is_separator(lines[j]):
                return j
            j += 1
        return None

    config_starts = [i for i, l in enumerate(lines) if "Configuration num." in l]
    if not config_starts:
        return []

    structures = []
    idx = 0
    for c, start in enumerate(config_starts):
        end = config_starts[c + 1] if c + 1 < len(config_starts) else n

        natoms = None
        elements, counts = [], []
        latt_vecs, positions = [], []
        energy = None
        forces, stress = [], []

        i = start
        while i < end:
            line = lines[i]

            if "The number of atoms" in line:
                vi = first_data_line(i)
                if vi is not None:
                    natoms = int(lines[vi].split()[0])
                i = (vi + 1) if vi is not None else i + 1
                continue

            if "Atom types and atom numbers" in line:
                vi = first_data_line(i)
                j = vi if vi is not None else i + 1
                while j < end and not _is_separator(lines[j]):
                    parts = lines[j].split()
                    if len(parts) >= 2:
                        elements.append(parts[0])
                        try:
                            counts.append(int(parts[1]))
                        except ValueError:
                            break
                    j += 1
                i = j
                continue

            if "Primitive lattice vectors" in line:
                vi = first_data_line(i)
                if vi is not None and vi + 2 < n:
                    latt_vecs = [lines[vi].split()[:3],
                                 lines[vi + 1].split()[:3],
                                 lines[vi + 2].split()[:3]]
                    i = vi + 3
                else:
                    i += 1
                continue

            if "Atomic positions" in line:
                vi = first_data_line(i)
                j = vi if vi is not None else i + 1
                positions = []
                limit = natoms if natoms else 10 ** 9
                while j < end and len(positions) < limit and not _is_separator(lines[j]):
                    positions.append(lines[j].split()[:3])
                    j += 1
                i = j
                continue

            if "Total energy" in line:
                vi = first_data_line(i)
                if vi is not None:
                    vals = _floats(lines[vi].split())
                    if vals:
                        energy = vals[0]
                i = (vi + 1) if vi is not None else i + 1
                continue

            if "Forces" in line:
                vi = first_data_line(i)
                j = vi if vi is not None else i + 1
                forces = []
                limit = natoms if natoms else 10 ** 9
                while j < end and len(forces) < limit and not _is_separator(lines[j]):
                    vals = _floats(lines[j].split())
                    if len(vals) >= 3:
                        forces.append(vals[:3])
                        j += 1
                    else:
                        break
                i = j
                continue

            if "Stress" in line:
                # 应力可能拆成 (XX YY ZZ) 与 (XY YZ ZX) 两个带标签小节,
                # 收集后续纯数值行的浮点数, 凑满 6 个。
                j = i + 1
                vals = []
                while j < end and len(vals) < 6:
                    if "Configuration num." in lines[j]:
                        break
                    toks = lines[j].split()
                    nums = _floats(toks)
                    if toks and len(nums) == len(toks):
                        vals.extend(nums)
                    j += 1
                stress = vals[:6]
                i = j
                continue

            i += 1

        if (natoms is None or natoms <= 0 or len(latt_vecs) != 3 or
                len(positions) != natoms or energy is None or
                not elements or sum(counts) != natoms):
            idx += 1
            continue

        # ML_AB 应力顺序 XX YY ZZ XY YZ ZX → Voigt [xx,yy,zz,yz,xz,xy]
        stress_voigt = []
        if len(stress) == 6:
            xx, yy, zz, xy, yz, zx = stress
            stress_voigt = [xx, yy, zz, yz, zx, xy]

        poscar = [f"Configuration {idx}\n", "1.0\n"]
        for vec in latt_vecs:
            poscar.append("  " + "  ".join(vec) + "\n")
        poscar.append("  " + "  ".join(elements) + "\n")
        poscar.append("  " + "  ".join(str(x) for x in counts) + "\n")
        poscar.append("Cartesian\n")
        for pos in positions:
            poscar.append("  " + "  ".join(pos) + "\n")

        structures.append({
            "idx": idx,
            "natoms": natoms,
            "dft_energy": energy,
            "dft_forces": np.array(forces) if len(forces) == natoms else None,
            "dft_stress": np.array(stress_voigt) if stress_voigt else None,
            "poscar": poscar,
        })
        idx += 1

    return structures


# ============================================================
#  OUTCAR 提取 (MLFF 预测; 兼容 ML_MODE=run 的 "ML TOTEN" 写法)
# ============================================================

def extract_mlff_outcar(outcar_path: str) -> Optional[Dict]:
    """从 ML_MODE=run 的 OUTCAR 提取能量/力/应力 (取最后一帧)。"""
    if not os.path.isfile(outcar_path):
        return None
    with open(outcar_path, "r", errors="ignore") as f:
        content = f.read()
        lines = content.splitlines()

    m = re.search(r"NIONS\s*=\s*(\d+)", content)
    natoms = int(m.group(1)) if m else None

    # 能量: 兼容 "free energy TOTEN" 与 MLFF 的 "free energy ML TOTEN"
    energy = None
    for line in lines:
        if "TOTEN" in line and "energy" in line:
            for t in reversed(line.split()):
                try:
                    energy = float(t)
                    break
                except ValueError:
                    continue
        elif "ML energy" in line:
            for t in line.replace("=", " ").split():
                try:
                    energy = float(t)
                except ValueError:
                    pass
    if energy is None:
        return None

    # 力: 最后一个 TOTAL-FORCE 块
    force_blocks = re.findall(
        r"POSITION\s+TOTAL-FORCE[^\n]*\n\s*-+\n(.*?)\n\s*-+",
        content, re.DOTALL)
    forces = []
    if force_blocks:
        for line in force_blocks[-1].strip().split("\n"):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    pass
    forces = np.array(forces) if forces else None
    if natoms is None and forces is not None:
        natoms = len(forces)

    # 应力: 最后一个 "in kB" 行 → Voigt [xx,yy,zz,yz,xz,xy]
    stress = None
    sm = re.findall(
        r"in kB\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
        r"\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)", content)
    if sm:
        v = [float(x) for x in sm[-1]]    # OUTCAR 顺序 XX YY ZZ XY YZ ZX
        stress = np.array([v[0], v[1], v[2], v[4], v[5], v[3]])

    return {"natoms": natoms, "energy": energy, "forces": forces, "stress": stress}


# ============================================================
#  生成评估输入 + 提交脚本
# ============================================================

_INCAR_RUN = """# MHEC MLFF 全数据集评估 (ML_MODE=run 单点预测)
SYSTEM = {system}
ENCUT  = {encut}
PREC   = Accurate
ISMEAR = 1
SIGMA  = 0.1
EDIFF  = 1E-5
LREAL  = Auto
LASPH  = .TRUE.
ISYM   = 0
NSW    = 0
IBRION = -1
ML_LMLFF  = .TRUE.
ML_MODE   = run
LWAVE  = .FALSE.
LCHARG = .FALSE.
NWRITE = 2
NCORE  = 1
"""

# 评估必需项 (强制覆盖) 与 MD/系综专用项 (评估单点应剔除)
_EVAL_FORCE = {
    "NSW": "0", "IBRION": "-1",
    "ML_LMLFF": ".TRUE.", "ML_MODE": "run", "ML_ISTART": "2",
    "LWAVE": ".FALSE.", "LCHARG": ".FALSE.",
}
_EVAL_DROP = {"MDALGO", "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L", "PMASS", "PSTRESS",
              "SMASS", "TEBEG", "TEEND", "POTIM", "ANDERSEN_PROB", "NBLOCK", "KBLOCK",
              "ML_WTOTEN", "ML_WTIFOR", "ML_WTSIF", "ML_IWEIGHT", "ISIF"}


def build_eval_incar(system: str = "Material", encut: float = 500,
                     user_incar: str = "INCAR") -> str:
    """
    生成 ML_AB 评估用 INCAR 文本。

    若本地存在 user_incar, 在其基础上改写为 MLFF 静态评估 (ML_MODE=run):
      - 保留用户电子结构参数 (ENCUT/PREC/EDIFF/ISMEAR/ALGO/LREAL/NCORE ...)
      - 强制评估必需项 (_EVAL_FORCE), 剔除 MD/系综专用项 (_EVAL_DROP)
    否则使用内置默认模板。
    """
    params: Dict[str, str] = {}
    order = []
    if user_incar and os.path.isfile(user_incar):
        with open(user_incar) as f:
            for line in f:
                s = line.split("#", 1)[0].split("!", 1)[0].strip()
                if not s or "=" not in s:
                    continue
                for part in s.split(";"):
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    k, v = k.strip().upper(), v.strip()
                    if not k or k in _EVAL_DROP:
                        continue
                    if k not in params:
                        order.append(k)
                    params[k] = v
        note = f"# 基于本地 INCAR 改写 ({user_incar})"
    else:
        for k, v in [("SYSTEM", str(system)), ("ENCUT", str(int(encut))),
                     ("PREC", "Accurate"), ("ISMEAR", "1"), ("SIGMA", "0.1"),
                     ("EDIFF", "1E-5"), ("LREAL", "Auto"), ("LASPH", ".TRUE."),
                     ("ISYM", "0"), ("NWRITE", "2"), ("NCORE", "1")]:
            order.append(k)
            params[k] = v
        note = "# 内置默认模板 (未找到本地 INCAR)"
    for k, v in _EVAL_FORCE.items():
        if k not in params:
            order.append(k)
        params[k] = v
    lines = ["# MHEC ML_AB 全数据集评估 INCAR (ML_MODE=run)", note]
    lines += [f"{k} = {params[k]}" for k in order]
    return "\n".join(lines) + "\n"


# 并行评估驱动 (与经过验证的 eval_mlff.sh 相同: 预检 + Pool 并行 + 断点续跑 + 诊断)
_EVAL_DRIVER_PY = r"""
python3 << 'PYEOF'
import os, sys, glob, subprocess
from multiprocessing import Pool

work = os.environ["WORK_ROOT"]
vasp = os.environ["VASP_CMD"]
npar = int(os.environ.get("NPAR_EVAL", "1") or "1")
dirs = sorted(glob.glob(os.path.join(work, "struct_*")))
if not dirs:
    print("错误: 未找到 struct_* 目录"); sys.exit(1)

def _done(d):
    o = os.path.join(d, "OUTCAR")
    if not os.path.exists(o):
        return False
    try:
        with open(o, errors="ignore") as f:
            t = f.read()
        return ("ML TOTEN" in t) or ("TOTEN" in t)
    except Exception:
        return False

def run_one(d):
    if _done(d):
        return (d, True, "skip")
    with open(os.path.join(d, "vasp.log"), "w") as log:
        subprocess.run(vasp, shell=True, cwd=d, stdout=log, stderr=subprocess.STDOUT)
    return (d, _done(d), "run")

print("预检: 串行运行 " + os.path.basename(dirs[0]) + " 验证 VASP/环境 ...", flush=True)
d0, ok0, _ = run_one(dirs[0])
if not ok0:
    print("=" * 60)
    print("预检失败: VASP 未生成有效 OUTCAR。请排查:")
    print("  VASP 命令: " + vasp)
    log = os.path.join(d0, "vasp.log")
    if os.path.exists(log):
        with open(log, errors="ignore") as f:
            tail = f.readlines()[-40:]
        print("  ----- vasp.log 末尾 -----"); print("".join(tail))
    print("  常见原因: 1) 无 vasp_gam -> 改用 vasp_std; 2) POTCAR 元素顺序;")
    print("            3) ML_FF 无效或与体系不匹配; 4) 环境未正确加载 (conda/oneAPI)")
    print("=" * 60)
    sys.exit(1)
print("预检通过。", flush=True)

rest = dirs[1:]
if rest:
    with Pool(processes=max(1, npar)) as pool:
        results = pool.map(run_one, rest)
else:
    results = []
results = [(d0, ok0, "run")] + results
nok = sum(1 for _, ok, _ in results if ok)
print("评估完成: %d/%d 个结构生成有效 OUTCAR" % (nok, len(results)))
if nok < len(results):
    print("部分结构失败; 可重新提交本脚本 (已完成的会自动跳过)。")
PYEOF
"""


def _eval_driver_block(vasp_cmd: str, npar_expr: str) -> str:
    """评估驱动的可注入片段: 设定环境变量 + 调用并行驱动。"""
    return "\n".join([
        "",
        "# MHEC: ML_AB 全数据集 MLFF 评估 (每个构型 ML_MODE=run 单点, 并行)",
        "# 若集群无 gamma 版 vasp_gam, 请把 VASP_CMD 改为 vasp_std",
        "# WORK_ROOT 解析: 依次尝试 SLURM 提交目录 / 脚本所在目录 / 当前目录,",
        "# 取第一个真正含 struct_* 的目录 (SLURM 常把脚本拷到 spool, 故 $0 不可靠)。",
        'unset WORK_ROOT',
        'for _cand in "$SLURM_SUBMIT_DIR" "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)" "$PWD"; do',
        '    if [ -n "$_cand" ] && ls -d "$_cand"/struct_* >/dev/null 2>&1; then',
        '        export WORK_ROOT="$_cand"; break',
        '    fi',
        'done',
        'export WORK_ROOT="${WORK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"',
        'echo "WORK_ROOT = $WORK_ROOT"',
        f'export VASP_CMD="{vasp_cmd}"',
        f'export NPAR_EVAL={npar_expr}',
        _EVAL_DRIVER_PY,
        'echo "全部完成。回到 mhec 菜单选择 \'收集 ML_AB 评估结果\' 出 parity 图。"',
        "",
    ])


def _template_preamble(template_path: str):
    """
    读取用户提交模板, 返回 (环境前导文本, 探测到的 vasp 可执行名)。

    保留其 #SBATCH 头与环境设置 (conda/module/source/export/ulimit), 剔除真正
    启动 vasp 的执行行 (mpirun/srun/... vasp_*), 并从中探测 vasp 可执行名。
    """
    import re
    with open(template_path, "r", errors="ignore") as f:
        raw = f.read().splitlines()
    preamble, vasp_exe = [], None
    for ln in raw:
        s = ln.strip()
        low = s.lower()
        is_exec = (
            "vasp" in low and not s.startswith("#")
            and ("mpirun" in low or "srun" in low or "mpiexec" in low
                 or low.startswith("./") or low.startswith("vasp"))
        )
        if is_exec:
            m = re.search(r"(vasp[\w.\-]*)", s)
            if m and vasp_exe is None:
                vasp_exe = m.group(1)
            continue  # 丢弃执行行, 用评估驱动替代
        preamble.append(ln)
    return "\n".join(preamble).rstrip() + "\n", vasp_exe


def generate_ab_eval_dirs(
    ml_ab: str = "ML_AB",
    ml_ff: str = "ML_FF",
    potcar: str = "POTCAR",
    kpoints: str = "KPOINTS",
    out_dir: str = "mlff_ab_eval",
    system: str = "Material",
    encut: float = 500,
    max_structures: int = 0,
    vasp_cmd: str = "mpirun -np 1 vasp_gam",
    partition: str = "8358P",
    user_incar: str = "INCAR",
    ncores: int = 64,
    submit_template: Optional[str] = None,
) -> Dict:
    """
    为 ML_AB 中每个构型生成 ML_MODE=run 单点评估目录 + 批量提交脚本。

    max_structures: 0=全部, >0=随机抽取, <0=取前 N 个
    user_incar: 本地参考 INCAR (存在则自动改写为评估用; 否则用内置默认)
    ncores: 每节点核数 = 同时并行评估的构型数 (每个构型单核 MLFF 单点很快)
    """
    ncores = max(1, int(ncores))
    missing = [f for f in [ml_ab, ml_ff, potcar, kpoints] if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"缺少文件: {', '.join(missing)}")

    structures = parse_ml_ab(ml_ab)
    if not structures:
        raise ValueError("未从 ML_AB 解析到有效构型 (文件格式可能不是标准 ML_AB)")

    total = len(structures)
    if max_structures > 0 and max_structures < total:
        import random
        structures = random.sample(structures, max_structures)
    elif max_structures < 0:
        structures = structures[:min(-max_structures, total)]

    os.makedirs(out_dir, exist_ok=True)
    incar_text = build_eval_incar(system, encut, user_incar)

    frame_idx = []
    for s in structures:
        d = os.path.join(out_dir, f"struct_{s['idx']:06d}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "POSCAR"), "w") as f:
            f.writelines(s["poscar"])
        with open(os.path.join(d, "INCAR"), "w") as f:
            f.write(incar_text)
        for src, name in [(ml_ff, "ML_FF"), (potcar, "POTCAR"), (kpoints, "KPOINTS")]:
            shutil.copy2(src, os.path.join(d, name))
        frame_idx.append(s["idx"])

    # 保留一份 ML_AB 供 collect 阶段读取 DFT 参考
    if os.path.abspath(ml_ab) != os.path.abspath(os.path.join(out_dir, "ML_AB")):
        shutil.copy2(ml_ab, os.path.join(out_dir, "ML_AB"))

    # 提交脚本: 优先复用用户提交模板 (其 #SBATCH/conda/oneAPI/核数/VASP 命令按
    # 用户集群配置好), 只把其中启动 vasp 的执行行替换为并行评估驱动; 核数自动取
    # 自 $SLURM_NTASKS。未提供模板时回退到内置默认 (含 conda init, 可配 ncores)。
    if submit_template and os.path.isfile(submit_template):
        preamble, exe = _template_preamble(submit_template)
        per_struct = f"mpirun -np 1 {exe}" if exe else vasp_cmd
        npar_expr = "${SLURM_NTASKS:-%d}" % ncores  # 自动取自模板分配, 无则回退
        submit_text = preamble + _eval_driver_block(per_struct, npar_expr)
    else:
        header = "\n".join([
            "#!/bin/bash",
            "#SBATCH -N 1",
            f"#SBATCH -n {ncores}",
            f"#SBATCH --ntasks-per-node={ncores}",
            f"#SBATCH --partition={partition}",
            "#SBATCH --output=ab_eval_%j.out",
            "#SBATCH --error=ab_eval_%j.err",
            "",
            "# 初始化 conda (未安装 conda 时自动跳过, 不会报错)",
            "if command -v conda &> /dev/null; then",
            '    eval "$(conda shell.bash hook 2>/dev/null)"',
            "fi",
            "",
            "# 环境 (内置默认; 建议改为复用你自己的 run.sh 模板)",
            "source /data/app/intel/oneapi-2021.4/setvars.sh",
            "export PATH=/data/app/vasp/vasp.6.4.2/bin:$PATH",
            "ulimit -s unlimited",
            "ulimit -l unlimited",
        ])
        submit_text = header + _eval_driver_block(vasp_cmd, str(ncores))

    submit_path = os.path.join(out_dir, "submit_ab_eval.sh")
    with open(submit_path, "w", newline="\n") as f:
        f.write(submit_text)
    try:
        os.chmod(submit_path, 0o755)
    except Exception:
        pass

    return {"out_dir": os.path.abspath(out_dir), "n_structures": len(structures),
            "frame_idx": frame_idx, "submit": submit_path}


# ============================================================
#  收集结果 → ParityData
# ============================================================

def collect_ab_parity(out_dir: str, ml_ab: Optional[str] = None) -> ParityData:
    """
    从已算完的 struct_*/OUTCAR 收集 MLFF 预测, 与 ML_AB 的 DFT 参考配对。
    """
    import glob

    if ml_ab is None:
        ml_ab = os.path.join(out_dir, "ML_AB")
        if not os.path.isfile(ml_ab):
            raise FileNotFoundError(f"找不到 ML_AB: {ml_ab}")

    ref = {s["idx"]: s for s in parse_ml_ab(ml_ab)}
    parity = ParityData()

    for d in sorted(glob.glob(os.path.join(out_dir, "struct_*"))):
        base = os.path.basename(d)
        try:
            idx = int(base.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx not in ref:
            continue
        s = ref[idx]
        mlff = extract_mlff_outcar(os.path.join(d, "OUTCAR"))
        if mlff is None or mlff["energy"] is None:
            continue
        natoms = s["natoms"]

        # 能量、力、应力都要齐全才配对（缺哪个跳过哪个由 add_frame 处理需对齐, 这里要求齐全）
        if (s["dft_forces"] is None or s["dft_stress"] is None or
                mlff["forces"] is None or mlff["stress"] is None or
                len(mlff["forces"]) != natoms or len(s["dft_forces"]) != natoms):
            # 至少配对能量
            parity.e_mlff.append(mlff["energy"] / natoms)
            parity.e_dft.append(s["dft_energy"] / natoms)
            parity.n_frames += 1
            continue

        parity.add_frame(
            mlff_e=mlff["energy"] / natoms,
            mlff_f=mlff["forces"],
            mlff_s=mlff["stress"],
            dft_e=s["dft_energy"] / natoms,
            dft_f=s["dft_forces"],
            dft_s=s["dft_stress"],
            natoms=natoms,
        )

    return parity
