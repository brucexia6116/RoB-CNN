#!/bin/bash
#SBATCH -J CNN_RoB           # job name
#SBATCH -o CNN_RoB.o%j       # output and error file name (%j expands to jobID)
#SBATCH -n 16              # total number of mpi tasks requested
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -t 12:00:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=byron.wallace@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
module load cuda
module load python
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python RoB_CNN_2.py            
