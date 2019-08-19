from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi

def setup(options):
    mode = options.get_string(option_section, 'mode', default='w')
    sample_a = options.get_string(option_section, "sample_a", default="source")
    sample_b = options.get_string(option_section, "sample_b", default="lens")
    return mode, sample_a, sample_b

def execute(block, config):

    mode, sample_a, sample_b = config

    if mode=='w':
        # one halo term(s)
        gi_1h = block['galaxy_intrinsic_w_1h', 'w_rp_1h_1_1_%s_%s'%(sample_a,sample_b)]

        # two halo term(s)
        gi_2h = block['galaxy_intrinsic_w', 'w_rp_1_1_%s_%s'%(sample_a,sample_b)]

        # add them together and save to the data block
        # the sampling should be the same, so we shouldn't need to interpolate
        gi = gi_1h + gi_2h

        block.replace_double_array_1d('galaxy_intrinsic_w', 'w_rp_1_1_%s_%s'%(sample_a,sample_b), gi)


    elif mode=='pk':
        # one halo term(s)
        k, z, gi_1h = block.get_grid('galaxy_intrinsic_power_1h', 'k_h','z','p_k')
        k, z, mi_1h = block.get_grid('matter_intrinsic_power_1h', 'k_h','z','p_k')
        k, z, ii_1h = block.get_grid('intrinsic_power_1h', 'k_h','z','p_k')

        # two halo term(s)
        k, z, gi_2h = block.get_grid('galaxy_intrinsic_power', 'k_h','z','p_k')
        k, z, mi_2h = block.get_grid('matter_intrinsic_power', 'k_h','z','p_k')
        k, z, ii_2h = block.get_grid('intrinsic_power', 'k_h','z','p_k')

        # add them together and save to the data block
        gi = gi_1h + gi_2h
        mi = mi_1h + mi_2h
        ii = ii_1h + ii_2h

        block.replace_grid('galaxy_intrinsic_power', 'k_h', k,'z', z,'p_k', gi)
        block.replace_grid('matter_intrinsic_power', 'k_h', k,'z', z,'p_k', mi)
        block.replace_grid('intrinsic_power', 'k_h', k,'z', z,'p_k', ii)

    return 0