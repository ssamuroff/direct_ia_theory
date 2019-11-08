from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def compute_c1_baseline():
    C1_M_sun = 5e-14  
    M_sun = 1.9891e30  
    Mpc_in_m = 3.0857e22  
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m)**3  
    
    G = 6.67384e-11  
    H = 100  
    H_SI = H * 1000.0 / Mpc_in_m 
    rho_crit_0 = 3 * H_SI**2 / (8 * np.pi * G)
    f = C1_SI * rho_crit_0
    return f

C1_RHOCRIT = compute_c1_baseline()
#snapshot = 99
#CONST = -0.005616
CONST = -0.005307
CONST2 = 0.033832


def setup(options):

    loc = options.get_string(option_section, "tensor_dir")
    snapshot = options.get_int(option_section, "snapshot")
    resolution = options.get_int(option_section, "resolution")

    gamma_I = fi.FITS(loc+'/stellar_shape_vects_0%d_%d.fits'%(snapshot, resolution))[-1].read()
    sij = -1. * fi.FITS(loc+'/dm_tidal_vects_0%d_0.25_%d.fits'%(snapshot, resolution))[-1].read()
    dsij = -1. * fi.FITS(loc+'/star_tidal_vects_0%d_0.25_%d.fits'%(snapshot, resolution))[-1].read()

    #svals = fi.FITS(loc+'/dm_tidal_vals_0%d_0.25.fits'%snapshot)[-1].read()


    s2 = np.linalg.det(sij)
    s2 = s2 * s2
    # cut out the cells with no galaxies
    mask = (gamma_I!=0)
    mask2 = (gamma_I[:,:,:,0,0]!=0)
    s2 = s2[mask2]

    npix = len(s2)
    sij = sij[mask].reshape((npix,3,3))
    dsij = dsij[mask].reshape((npix,3,3))
    gamma_I = gamma_I[mask].reshape((npix,3,3))

    sigma = np.zeros_like(gamma_I)
    for i in range(3):
        sigma[:,i,:] = np.std(gamma_I[:,i,:])

    #import pdb ; pdb.set_trace()

    S = np.array([np.diag([np.linalg.det(s)]*3) for s in sij]).flatten()
    S2 = 1./3*S*S

    sumsij = np.array([ [ [[np.sum(s0[axis1,:]*s0[:,axis2]) for axis2 in range(3)] for axis1 in range(3)]] for s0 in sij])
    sumsij = sumsij.flatten()

    return gamma_I, sigma, sij, dsij, S2, sumsij

def delta(i,j):
    if (i==j):
        return 1
    else:
        return 0

#CONST = -0.005616

def execute(block, config):
    gamma_I, sigma, sij, dsij, S2, sumsij = config

    A1 = block['intrinsic_alignment_parameters', 'A1']
    A2 = block['intrinsic_alignment_parameters', 'A2']
    bg = block['intrinsic_alignment_parameters', 'bias_ta']


    print(A1,A2,bg)

    C1 = A1 * CONST 
    C2 = A2 * CONST2

    T = C1 * sij.flatten() + bg * C1 * dsij.flatten() + C2 * (sumsij - S2)
    chi2 = (gamma_I.flatten() - T )**2 
    chi2 = chi2 / sigma.flatten() / sigma.flatten()
    chi2 = np.sum(chi2)

    like = -0.5 * chi2

    print('chi2 = %3.3f'%chi2)
    print('(reduced = %3.3f)'%(chi2/(len(T) - 3)))

    block[names.data_vector, 'tatt_direct'+"_CHI2"] = chi2
    block[names.likelihoods, 'tatt_direct'+"_LIKE"] = like
    import pdb ; pdb.set_trace()


    return 0





















