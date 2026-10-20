#!/bin/bash
#SBATCH --job-name=ela-forensics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
# Sesuaikan partition/account dengan pusat HPC yang dipakai:
# #SBATCH --partition=compute
# #SBATCH --account=PROJECT_ID

module load python 2>/dev/null || true
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/generate_dataset.py --out data --n 2000 --size 1024
python src/benchmark.py --data data --workers 1,2,4,8,16
python src/detect.py --data data --workers ${SLURM_CPUS_PER_TASK}
