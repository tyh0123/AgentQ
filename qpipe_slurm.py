"""
SLURM submission script generation.
"""
from __future__ import annotations

import os

from qpipe_config import Config


TEMPLATE = """#!/bin/bash
#SBATCH -N {nodes}
#SBATCH -C gpu
#SBATCH -G {gpus}
#SBATCH -q {queue}
#SBATCH -J {job_name}
#SBATCH -A {account}
#SBATCH -t {wall_time}

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n {gpus} -c 32 --cpu_bind=cores -G {gpus} {binary} {input_file}
"""


def write_slurm(prefix: str, input_file: str, cfg: Config) -> str:
    S = cfg.slurm
    job_name = (S.job_name_prefix + prefix.replace("qubit_", ""))[:8]
    content = TEMPLATE.format(
        nodes=S.nodes,
        gpus=S.gpus,
        queue=S.queue,
        job_name=job_name,
        account=S.account,
        wall_time=S.wall_time,
        binary=S.binary,
        input_file=os.path.basename(input_file),
    )
    out_file = f"perl_{prefix}.run"
    with open(out_file, "w") as f:
        f.write(content)
    return out_file
