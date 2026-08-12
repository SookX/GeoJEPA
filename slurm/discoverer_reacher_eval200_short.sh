#!/bin/bash
# Short Discoverer/Slurm wrapper for one resumable 200-start Reacher evaluation.
#
# The account is low on remaining billing-hours, so this wrapper requests a
# short wall time and does not exclude dgx1. The underlying eval loop skips
# completed chunks when FORCE is unset/0, so rerunning this script resumes.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=lewm_eval20m
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o logs/lewm_eval20m.%j.out
#SBATCH -e logs/lewm_eval20m.%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
cd "${PROJECT_DIR}"

exec bash slurm/discoverer_reacher_eval200_loop.sh
