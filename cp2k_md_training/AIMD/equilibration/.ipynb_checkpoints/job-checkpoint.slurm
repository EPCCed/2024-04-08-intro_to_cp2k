#!/bin/bash -l
#SBATCH --job-name=cp2k
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:12:00
#SBATCH --account=ta154
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread

# Load cp2k
module load cp2k/cp2k-2023.2

#Ensure that no libraries are inadvertently using threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_SUBMIT_DIR=$(readlink -f $SLURM_SUBMIT_DIR)

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

# Launch the parallel job
srun cp2k.psmp -i sys.inp -o sys.out