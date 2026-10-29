#!/bin/bash
#SBATCH --job-name=meshom-ui
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=meshom-ui-%j.log
# The meshom web UI on a Levante compute node, viewed in the browser of your own computer through an ssh tunnel.
#   sbatch -A <project> scripts/ui.sbatch [meshom ui options: --runs DIR, --refs DIR, --config FILE, --port N]
# Submit from the repository with the meshom environment active (the job inherits it). The log meshom-ui-<jobid>.log
# prints the URL with its token and the tunnel command to run on your computer. Stop the server with scancel <jobid>.
set -u
cd "$SLURM_SUBMIT_DIR"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -m meshom.cli ui --host "$(hostname)" --port 8765 "$@"
