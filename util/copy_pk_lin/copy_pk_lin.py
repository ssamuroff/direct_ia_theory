from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi

def setup(options):
    return 0

def execute(block, config):

    z,k,pk = block.get_grid('matter_power_lin', 'z', 'k_h', 'p_k')

    # interpolate so the linear matter power spectrum is on the
    # same k,z grid as the nonlinear version it's replacing
    z_nl,k_nl,pk_nl = block.get_grid('matter_power_nl', 'z', 'k_h', 'p_k')
    interp = spi.interp2d(np.log10(k),z,np.log10(pk))

    block['matter_power_nl', 'p_k'] = 10**interp(np.log10(k_nl),z_nl)

    return 0
