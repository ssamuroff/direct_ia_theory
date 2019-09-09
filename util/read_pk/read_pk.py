from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def setup(options):

    path = options.get_string(option_section, "pk_loc")
    section_name = options.get_string(option_section, "pk_type", default='matter_power_nl')

    pk = np.loadtxt('%s/%s/p_k.txt'%(path,section_name))
    k = np.loadtxt('%s/%s/k_h.txt'%(path,section_name))
    z = np.loadtxt('%s/%s/z.txt'%(path,section_name))

    vals = open('%s/%s/values.txt'%(path,section_name)).readlines()

    print('Reading power spectrum from file: %s'%path)
 

    return pk, k, z, vals, section_name


def execute(block, config):
    pk, k, z, vals, section_name = config

    block[section_name,'p_k'] = pk
    block[section_name,'k_h'] = k
    block[section_name,'z'] = z

    for v in vals:
        name,val = v.split(' = ')
        comm = "block['%s', '"%(section_name) + v.replace(" =", "'] =").replace('\n','')
        exec(comm)

    
    return 0
