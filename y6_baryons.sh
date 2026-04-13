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
#
#cosmosis y6ia/y6ia-tatt-slow-camb.ini -p runtime.sampler=test output.lock=F -v halo_model_parameters.logt_agn=8.1
#cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F read_pk.pk_loc=$IA_LIB/output/pk_fid/ read_pk_lin.pk_loc=$IA_LIB/output/pk_fid/ test.save_dir=output/redmagic_z${ZBIN}_L${LBIN}_slow_baseline

#cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F read_pk.pk_loc=$IA_LIB/output/pk_fid_even_more_baryons/ read_pk_lin.pk_loc=$IA_LIB/output/pk_fid_even_more_baryons/ test.save_dir=output/redmagic_z${ZBIN}_L${LBIN}_slow_baryons

#cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F read_pk.pk_loc=$IA_LIB/output/pk_eemu/ read_pk_lin.pk_loc=$IA_LIB/output/pk_eemu/ test.save_dir=output/redmagic_z${ZBIN}_L${LBIN}_slow_eemu


cosmosis y6ia/y6ia-tatt-slow.ini -p runtime.sampler=test output.lock=F read_pk.pk_loc=$IA_LIB/output/pk_dmo/ read_pk_lin.pk_loc=$IA_LIB/output/pk_dmo/ test.save_dir=output/redmagic_z${ZBIN}_L${LBIN}_slow_dmo
