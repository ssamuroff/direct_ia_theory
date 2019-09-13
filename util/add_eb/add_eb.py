from cosmosis.datablock import option_section, names
import sys
import os
import numpy as np
from cosmosis.datablock import names, option_section
import scipy.interpolate as interp

def setup(options):
    ee_section = options.get_string(option_section, "ee_section", "intrinsic_w")
    bb_section=options.get_string(option_section, "bb_section", "intrinsic_w_bb")
    redshifts = np.atleast_1d(options[option_section, "redshift"])
    return ee_section, bb_section, redshifts

def execute(block, config):

    
    ee_section, bb_section, redshifts = config

    x = block[ee_section, "r_p"]
    x_bb = block[bb_section, "r_p"]

    for i,z in enumerate(redshifts):
        suffix = '_%1.3f'%(z)

        ee = block[ee_section,'w_rp_limber'+suffix]
        bb = block[bb_section,'w_rp_limber'+suffix]

        # preserve the EE part in case it's needed at some point
        block[ee_section+'_ee', 'w_rp_limber'+suffix] = ee
        # overwrite with the EE+BB total signal
        block[ee_section, 'w_rp_limber'+suffix] = ee+bb


    return 0
