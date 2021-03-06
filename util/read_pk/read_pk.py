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
    distances = options.get_bool(option_section, "distances", default=False)

    pk = np.loadtxt('%s/%s/p_k.txt'%(path,section_name))
    k = np.loadtxt('%s/%s/k_h.txt'%(path,section_name))
    z = np.loadtxt('%s/%s/z.txt'%(path,section_name))
    if distances:
        Dm = np.loadtxt('%s/distances/d_m.txt'%(path))
        a = np.loadtxt('%s/distances/a.txt'%(path))
        zm = np.loadtxt('%s/distances/z.txt'%(path))
    else:
        Dm,zm,a = None,None,None

    vals = open('%s/%s/values.txt'%(path,section_name)).readlines()

    print('Reading power spectrum from file: %s'%path)
 

    return pk, k, z, Dm, a, zm, vals, section_name


def execute(block, config):
    pk, k, z, Dm, a, zm, vals, section_name = config

    block[section_name,'p_k'] = pk
    block[section_name,'k_h'] = k
    block[section_name,'z'] = z
    if Dm is not None:
        block['distances','z'] = zm
        block['distances','d_m'] = Dm
        block['distances','a'] = a

    for v in vals:
        name,val = v.split(' = ')
        comm = "block['%s', '"%(section_name) + v.replace(" =", "'] =").replace('\n','')
        exec(comm)

    
    return 0
