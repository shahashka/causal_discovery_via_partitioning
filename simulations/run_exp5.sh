#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l walltime=72:00:00
#PBS -l filesystems=home
#PBS -q preemptable
#PBS -A FoundEpidem
module use /soft/modulefiles
module load conda
conda activate /home/shahashka/miniconda3/envs/cd_part2
cd /home/shahashka/causal_discovery_via_partitioning 
python simulations/experiment_5_num_nodes_sweep.py 