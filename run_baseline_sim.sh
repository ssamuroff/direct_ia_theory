#!/bin/bash 
#SBATCH --job-name=y6ia_nla
#SBATCH -A des
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH -o /pscratch/sd/s/sws/logs/y6ia_nla_sim.log 

source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

export OMP_NUM_THREADS=1

export COSMOSIS_LIB=$SCRATCH/y6_cosmosis/cosmosis-standard-library/
export IA_LIB=$HOME/direct_ia_theory/
export DATA_DIR=$HOME/y6_ia/data

cd $IA_LIB

# settings, names and paths
export ZBIN=4
export LBIN=3
export DVNAME=simulated
export CHAIN_NAME=nt_z${ZBIN}L${LBIN}_nla_${DVNAME}_v2
export OUTPUT_DIR=chain_outputs

# finally run cosmosis
#srun -n 128 cosmosis --mpi y6ia/y6ia-nla-slow.ini -p runtime.sampler=nautilus nautilus.resume=T output.lock=F -v intrinsic_alignment_parameters.a2=0.

cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F -v intrinsic_alignment_parameters.a2=0. intrinsic_alignment_parameters.a1=1.0
