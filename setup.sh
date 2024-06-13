# source cosmosis
cd $SHARED
conda activate ./env
source cosmosis-configure
cd -

# you need this line unless you want camb to be very slow
export OMP_NUM_THREADS=1

# replace these lines with wherever the repos live in your setup
export COSMOSIS_LIB=$SHARED/y6_cosmosis/cosmosis-standard-library/
export IA_LIB=$HOME/direct_ia_theory/
export DATA_DIR=$HOME/IAmeasurementsStore/
