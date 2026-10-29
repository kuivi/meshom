#!/bin/bash
#SBATCH --job-name=meshom_run
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --output=meshom-run-%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD MPLBACKEND=Agg
PY=${PY:-python}                   # the python of the environment active at submission (sbatch passes it on)
echo "== $(date -Iseconds) host $(hostname) commit $(git rev-parse --short HEAD)"
# usage: sbatch -A <project> scripts/run_example.sbatch docs/examples/global_150km_m14.toml [more configs]
for C in "$@"; do
  echo "== run $C $(date -Iseconds)"; $PY -m meshom.cli run "$C"; echo "== rc $? $(date -Iseconds)"
done
$PY -m meshom.cli runs .tmp/runs
echo "== end $(date -Iseconds)"
