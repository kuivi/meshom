#!/bin/bash
# Run one command on a Levante compute node and wait for it (blocking srun, output streamed here).
# Nothing heavier than a few-second single-process task runs on the login node: Levante's login nodes are for short
# tasks only.
#   scripts/on_node.sh [-c CPUS] [-t HH:MM:SS] [-m MEM] [-p PARTITION] [-J NAME] -- cmd args...
# Defaults: 8 CPUs, 2 h, 32G, partition shared (1 task on a shared node; use -p compute -c 256 for a whole node).
set -u
CPUS=8; TIME=02:00:00; MEM=7G; PART=shared; NAME=meshom
while [ $# -gt 0 ]; do
  case "$1" in
    -c) CPUS=$2; shift 2;; -t) TIME=$2; shift 2;; -m) MEM=$2; shift 2;; -p) PART=$2; shift 2;; -J) NAME=$2; shift 2;;
    --) shift; break;; *) break;;
  esac
done
export NUMBA_NUM_THREADS=$CPUS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec srun --account=ab0995 --partition=$PART --ntasks=1 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
     --job-name=$NAME --chdir="$PWD" --export=ALL --quiet "$@"
