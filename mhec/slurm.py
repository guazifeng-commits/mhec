"""
SLURM 脚本生成。
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SlurmConfig:
    """SLURM 提交参数配置。"""
    nodes: int = 1
    ntasks: int = 64
    ntasks_per_node: int = 64
    partition: str = "8358P"
    nodelist: Optional[str] = None
    oneapi_path: str = "/data/app/intel/oneapi-2021.4"
    vasp_path: str = "/data/app/vasp/vasp.6.4.2/bin"
    vasp_exec: str = "vasp_gam"
    mpirun_cmd: str = "mpirun"
    script_name: str = "run.sh"  # SLURM 脚本文件名


def generate_run_slurm(
    config: SlurmConfig,
    job_name: str = "mhec",
) -> str:
    """生成单个计算目录的 run.sh 脚本内容。"""
    lines = [
        "#!/bin/sh",
        f"#SBATCH -N {config.nodes}",
        f"#SBATCH -n {config.ntasks}",
        f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
        f"#SBATCH --partition={config.partition}",
    ]
    if config.nodelist:
        lines.append(f"#SBATCH -w {config.nodelist}")
    lines.extend([
        "#SBATCH --output=%j.out",
        "#SBATCH --error=%j.err",
        "",
        f"source {config.oneapi_path}/setvars.sh",
        f"export PATH={config.vasp_path}:$PATH",
        "",
        "ulimit -s unlimited",
        "ulimit -l unlimited",
        "",
        f"{config.mpirun_cmd} -np $SLURM_NPROCS {config.vasp_exec}",
    ])
    return "\n".join(lines) + "\n"


def generate_submit_all(
    dirs_by_stage: Dict[str, List[str]],
    config: SlurmConfig,
    work_dir: str = ".",
) -> str:
    """
    生成一键批量提交脚本 submit_all.sh。

    按 MLFF 阶段顺序提交，通过 --dependency=afterok 实现阶段间串行。
    使用相对路径，脚本在工作目录下执行即可。
    """
    sn = config.script_name  # run.sh

    lines = [
        "#!/bin/bash",
        "#SBATCH -N 1",
        "#SBATCH -n 1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --partition={config.partition}",
        "#SBATCH --job-name=mhec_submit",
        "#SBATCH --output=submit_all_%j.out",
        "#SBATCH --error=submit_all_%j.err",
        "",
        "# MHEC 批量提交脚本",
        "# 按 MLFF 阶段顺序提交: train → refit → run",
        "# 用法: sbatch submit_all.sh  或  bash submit_all.sh",
        "",
        "# 获取工作根目录:",
        "#   sbatch 模式: SLURM_SUBMIT_DIR 是提交时所在目录",
        "#   bash 模式: 脚本自身所在目录",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"',
        'fi',
        "",
    ]

    for stage in ["train", "refit", "run"]:
        stage_dirs = dirs_by_stage.get(stage, [])
        if not stage_dirs:
            continue

        lines.append(f"# ===== {stage.upper()} 阶段 ({len(stage_dirs)} 个任务) =====")
        lines.append(f"echo '>>> 提交 {stage.upper()} 阶段 ({len(stage_dirs)} 个任务)...'")
        lines.append(f"{stage.upper()}_JOBS=()")
        lines.append("")

        for d in stage_dirs:
            rel = os.path.relpath(d, work_dir)
            lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')

            if stage == "train":
                lines.append(f"JOB_ID=$(sbatch --parsable {sn})")
            else:
                prev = "TRAIN" if stage == "refit" else "REFIT"
                lines.append(f'if [ ${{#{prev}_JOBS[@]}} -gt 0 ]; then')
                lines.append(f'  DEPS=$(IFS=:; echo "${{{prev}_JOBS[*]}}")')
                lines.append(f'  JOB_ID=$(sbatch --parsable --dependency=afterok:$DEPS {sn})')
                lines.append(f'else')
                lines.append(f'  JOB_ID=$(sbatch --parsable {sn})')
                lines.append(f'fi')

            lines.append(f"{stage.upper()}_JOBS+=($JOB_ID)")
            lines.append(f'echo "  {rel}: $JOB_ID"')
            lines.append("popd > /dev/null")
            lines.append("")

    lines.append('echo ">>> 所有任务已提交。"')
    return "\n".join(lines) + "\n"


def write_slurm_scripts(
    work_dir: str,
    all_dirs: Dict[str, Dict[str, Dict[str, str]]],
    eq_dirs: Dict[str, str],
    config: SlurmConfig,
) -> None:
    """为所有计算目录写入 run.sh 和 submit_all.sh。"""
    dirs_by_stage: Dict[str, List[str]] = {"train": [], "refit": [], "run": []}
    sn = config.script_name  # run.sh

    # 平衡构型目录
    for stage, path in eq_dirs.items():
        slurm_content = generate_run_slurm(config, job_name="eq_" + stage)
        with open(os.path.join(path, sn), "w") as f:
            f.write(slurm_content)
        dirs_by_stage[stage].append(path)

    # 变形构型目录
    for code, amps in all_dirs.items():
        for amp, stages in amps.items():
            for stage, path in stages.items():
                slurm_content = generate_run_slurm(
                    config, job_name=f"{code}_{amp}_{stage}"
                )
                with open(os.path.join(path, sn), "w") as f:
                    f.write(slurm_content)
                dirs_by_stage[stage].append(path)

    # submit_all.sh
    submit_content = generate_submit_all(dirs_by_stage, config, work_dir)
    with open(os.path.join(work_dir, "submit_all.sh"), "w") as f:
        f.write(submit_content)


def copy_potcar_to_dirs(potcar_path: str, dirs: List[str]) -> int:
    """将 POTCAR 复制到所有计算目录。返回成功复制的数量。"""
    if not os.path.isfile(potcar_path):
        print(f"  !-> POTCAR 不存在: {potcar_path}")
        return 0
    count = 0
    for d in dirs:
        dst = os.path.join(d, "POTCAR")
        if not os.path.isfile(dst):
            try:
                shutil.copy2(potcar_path, dst)
                count += 1
            except Exception as e:
                print(f"  !-> 复制 POTCAR 到 {d} 失败: {e}")
    return count


def read_submit_template(template_path: str) -> str:
    """读取用户提供的提交脚本模板。"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_submit_from_template(
    template: str,
    dirs_by_stage: Dict[str, List[str]],
    work_dir: str,
    script_name: str = "run.sh",
) -> str:
    """
    基于用户模板生成批量提交脚本。

    模板中的 VASP 执行命令行会被自动识别（包含 vasp 的行），
    脚本会为每个目录 cd 进去执行模板中的命令。

    生成的脚本按 train → refit → run 顺序，
    每个阶段内的任务并行提交，阶段间串行等待。
    """
    lines = [
        "#!/bin/bash",
        "# MHEC 批量提交脚本 (基于用户模板)",
        "# 按 MLFF 阶段顺序: train → refit → run",
        f"# 工作目录: {os.path.abspath(work_dir)}",
        "",
        'if [ -n "$SLURM_SUBMIT_DIR" ]; then',
        '  WORK_ROOT="$SLURM_SUBMIT_DIR"',
        'else',
        '  WORK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)',
        'fi',
        "",
    ]

    for stage in ["train", "refit", "run"]:
        stage_dirs = dirs_by_stage.get(stage, [])
        if not stage_dirs:
            continue

        lines.append(f"# ===== {stage.upper()} 阶段 ({len(stage_dirs)} 个任务) =====")
        lines.append(f"echo '>>> 提交 {stage.upper()} 阶段 ({len(stage_dirs)} 个任务)...'")
        lines.append(f"{stage.upper()}_JOBS=()")
        lines.append("")

        for d in stage_dirs:
            rel = os.path.relpath(d, work_dir)
            lines.append(f'pushd "$WORK_ROOT/{rel}" > /dev/null')

            if stage == "train":
                lines.append(f"JOB_ID=$(sbatch --parsable {script_name})")
            else:
                prev = "TRAIN" if stage == "refit" else "REFIT"
                lines.append(f'if [ ${{#{prev}_JOBS[@]}} -gt 0 ]; then')
                lines.append(f'  DEPS=$(IFS=:; echo "${{{prev}_JOBS[*]}}")')
                lines.append(f'  JOB_ID=$(sbatch --parsable --dependency=afterok:$DEPS {script_name})')
                lines.append(f'else')
                lines.append(f'  JOB_ID=$(sbatch --parsable {script_name})')
                lines.append(f'fi')

            lines.append(f"{stage.upper()}_JOBS+=($JOB_ID)")
            lines.append(f'echo "  {rel}: $JOB_ID"')
            lines.append("popd > /dev/null")
            lines.append("")

    lines.append('echo ">>> 所有任务已提交。"')
    return "\n".join(lines) + "\n"


def write_slurm_with_template(
    work_dir: str,
    all_dirs: Dict[str, Dict[str, Dict[str, str]]],
    eq_dirs: Dict[str, str],
    config: "SlurmConfig",
    template: Optional[str] = None,
) -> None:
    """为所有计算目录写入 run.sh 和 submit_all.sh。

    如果提供了 template，则用模板内容作为 run.sh；
    否则使用默认生成的 run.sh。
    """
    dirs_by_stage: Dict[str, List[str]] = {"train": [], "refit": [], "run": []}
    sn = config.script_name  # run.sh

    def _write_run_script(path: str, job_name: str):
        if template:
            content = template
        else:
            content = generate_run_slurm(config, job_name=job_name)
        with open(os.path.join(path, sn), "w") as f:
            f.write(content)

    # 平衡构型目录
    for stage, path in eq_dirs.items():
        _write_run_script(path, "eq_" + stage)
        dirs_by_stage[stage].append(path)

    # 变形构型目录
    for code, amps in all_dirs.items():
        for amp, stages in amps.items():
            for stage, path in stages.items():
                _write_run_script(path, f"{code}_{amp}_{stage}")
                dirs_by_stage[stage].append(path)

    # submit_all.sh
    if template:
        submit_content = generate_submit_from_template(
            template, dirs_by_stage, work_dir, script_name=sn)
    else:
        submit_content = generate_submit_all(dirs_by_stage, config, work_dir)
    with open(os.path.join(work_dir, "submit_all.sh"), "w") as f:
        f.write(submit_content)
