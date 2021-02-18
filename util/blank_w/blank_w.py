from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def setup(options):
    section_name = options.get_string(option_section, "section", default='galaxy_w')
    sa = options.get_string(option_section, "sample_a", default='redmagic_high_density')
    sb = options.get_string(option_section, "sample_b", default='redmagic_high_density')

    return section_name, sa, sb


def execute(block, config):
    section_name, sa, sb = config

    try:
        rp = block[section_name,'r_p']
    except:
        rp = np.logspace(-2,3,500)
        block.put_double_array_1d(section_name,'r_p',rp)
    out_name = "w_rp_1_1_%s_%s"%(sa,sb)

    w = np.zeros_like(rp)
    block.put_double_array_1d(section_name,out_name,w)

    #import pdb ; pdb.set_trace()

    
    return 0
