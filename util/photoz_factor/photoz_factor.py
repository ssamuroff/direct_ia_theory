from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi


def setup(options):

    do_gp = options.get_bool(option_section, "do_gp", default=False)
    do_gg = options.get_bool(option_section, "do_gg", default=False)
    do_pp = options.get_bool(option_section, "do_pp", default=False)

    mu = options.get_double(option_section, "mu", default=-1.)

    sample_a = options.get_string(option_section, "sample_a")
    sample_b = options.get_string(option_section, "sample_b")

    return do_gp, do_gg, do_pp, sample_a, sample_b, mu


def execute(block, config):

    do_gp, do_gg, do_pp, sample_a, sample_b, mu = config
   # import pdb ; pdb.set_trace()

    if (mu==-1):
        mu = block['photoz_errors', 'mu']
    print('mu = %f'%mu)

    if do_gp:
        name = 'w_rp_1_1_%s_%s'%(sample_a,sample_b)
        section_name = 'galaxy_intrinsic_w'
        w0 = block[section_name, name]
        block.replace_double_array_1d(section_name, name, mu * w0)

    if do_gg:
        name = 'w_rp_1_1_%s_%s'%(sample_a,sample_a)
        section_name = 'galaxy_w'
        w0 = block[section_name, name]
        block.replace_double_array_1d(section_name, name, mu * w0)

    if do_pp:
        name = 'w_rp_1_1_%s_%s'%(sample_b,sample_b)
        section_name = 'intrinsic_w'
        w0 = block[section_name, name]
        block.replace_double_array_1d(section_name, name, mu * w0)

    
    return 0
