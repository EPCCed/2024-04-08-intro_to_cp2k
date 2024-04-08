#!/bin/bash

#SBATCH --job-name=CP2K_scaling_test

# number of nodes - each with 128 cores
#SBATCH --nodes=1

# number of MPI processes on a node should be 128/OMP_NUM_THREADS
#SBATCH --ntasks-per-node=128

# number of OMP threads per MPI process 
# you probably want ntasks-per-node * cpus-per-task = 128 
# make sure OMP_NUM_THREADS further down is consistent
#SBATCH --cpus-per-task=1

# wall time
#SBATCH --time=00:20:00

##SBATCH --reservation=XXXreservationIDXXX
##SBATCH --qos=reservation

#SBATCH --account=ta154
#SBATCH --partition=standard
#SBATCH --qos=short

# Load the relevent CP2K module
module load cp2k

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun --hint=nomultithread --distribution=block:block cp2k.psmp -i H2O-32xX.inp > H2O-32xX.out
#srun --hint=nomultithread --distribution=block:block cp2k.psmp -i H2O-gga.inp > H2O-gga.out
#srun --hint=nomultithread --distribution=block:block cp2k.psmp -i H2O-hyb.inp > H2O-hyb.out
#srun --hint=nomultithread --distribution=block:block cp2k.psmp -i diag_cu144_broy.inp > diag_cu144_broy.out

