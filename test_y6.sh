#!/bin/bash 


source /dvs_ro/cfs/projectdirs/des/zuntz/cosmosis-global/setup-cosmosis3

export OMP_NUM_THREADS=1

export COSMOSIS_LIB=$SCRATCH/y6_cosmosis/cosmosis-standard-library/
export IA_LIB=$HOME/direct_ia_theory/
export DATA_DIR=$HOME/y6_ia/data

cd $IA_LIB

# settings, names and paths
export ZBIN=1
export LBIN=3
export CHAIN_NAME=nt_z${ZBIN}L${LBIN}_tatt
export OUTPUT_DIR=chain_outputs

# finally run cosmosis
#mpirun -n 64 cosmosis --mpi examples/params-tatt-fast.ini -p runtime.sampler=nautilus nautilus.resume=T output.lock=F

#cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F
#cosmosis y6ia/y6ia-tatt-slow-eemu.ini -p runtime.sampler=test output.lock=F
cosmosis y6ia/y6ia-tatt-slow-camb.ini -p runtime.sampler=test output.lock=F camb.halofit_version=mead2020
