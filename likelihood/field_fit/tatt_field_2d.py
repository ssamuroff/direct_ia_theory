from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import numpy.fft as npf
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
    nx = options.get_int(option_section, "resolution") # pixel resolution 128, 64, 32, 16

    base='/home/rmandelb.proj/ssamurof/mb2_tidal/'


    nxyz = fi.FITS('%s/density/dm_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()
    gxyz = fi.FITS('%s/density/star_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()
    n0 = int(nxyz.shape[0]/2)

    # now compute the tidal tensor
    k  = npf.fftfreq(nx)[np.mgrid[0:nx,0:nx,0:nx]]
    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    stellar_tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)

    # overdensity field
    K = np.mean(nxyz)
    d = nxyz/K -1 
    g = gxyz/np.mean(gxyz) -1 

    # FFT the box
    fft_dens = npf.fftn(d) 
    stellar_fft_dens = npf.fftn(g) 

    A=(2.*np.pi)**3/nx 

    for i in range(3):
        for j in range(3):
            print(i,j)
            # k[i], k[j] are 3D matrices
            temp = fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)
            stellar_temp = stellar_fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)

            # subtract off the trace...
            if (i==j):
                temp -= 1./3 * fft_dens
                stellar_temp -= 1./3 * stellar_fft_dens

            temp[0,0,0] = 0
            stellar_temp[0,0,0] = 0
            tidal_tensor[:,:,:,i,j] = - A * npf.ifftn(temp).real
            stellar_tidal_tensor[:,:,:,i,j] = - A * npf.ifftn(stellar_temp).real


    gammaI = fi.FITS(base+'tidal/raw/star_tidal_traceless_0%d_0.25_%d.fits'%(snapshot,nx))[-1].read()

    dgammaI = np.zeros_like(gammaI)
    for i in range(3):
        for j in range(3):
            dgammaI[:,:,:,i,j] = np.std(np.unique(gammaI[:,:,:,i,j]))

    dx = tidal_tensor.std()
    xc = tidal_tensor.mean()
    x = np.linspace(xc-4*dx, xc+4*dx, 21)
    x0 = (x[:-1]+x[1:])/2

    stellar_dx = stellar_tidal_tensor[stellar_tidal_tensor!=-9999.].std()
    stellar_xc = stellar_tidal_tensor[stellar_tidal_tensor!=-9999.].mean()
    stellar_x = np.linspace(stellar_xc-4*stellar_dx, stellar_xc+4*stellar_dx, 21)
    stellar_x0 = (stellar_x[:-1]+stellar_x[1:])/2

    y00 = np.zeros((20,20))-9999.
    dy00 = np.zeros((20,20))-9999.

    for i,(lower,upper) in enumerate(zip(x[:-1],x[1:])):
        for j,(slower,supper) in enumerate(zip(stellar_x[:-1],stellar_x[1:])):
            mask00 = (tidal_tensor[:,:,:,0,0]>lower) & (tidal_tensor[:,:,:,0,0]<upper)
            smask00 = (stellar_tidal_tensor[:,:,:,0,0]>slower) & (stellar_tidal_tensor[:,:,:,0,0]<supper)
            mask = mask00 & smask00

            y, dy = get_binned(0, gammaI, mask)
            y00[i,j] = y
            dy00[i,j] = dy

    y00[np.isnan(y00)] = -9999.
    dy00[np.isnan(dy00)] = -9999.

    xx,sxx = np.meshgrid(x0,stellar_x0)
    #import pdb ; pdb.set_trace()

    return gammaI, tidal_tensor, dgammaI, xx, sxx, y00, dy00



def execute(block, config):
    gammaI, tidal_tensor, dgammaI, x0, stellar_x0, y00, dy00 = config

    A1 = block['intrinsic_alignment_parameters', 'A1']
    A2 = block['intrinsic_alignment_parameters', 'A2']

    C1 = A1 * CONST 
    C2 = A2 * CONST2

    bias_ta = block['intrinsic_alignment_parameters', 'bias_ta']
    C1d = bias_ta * C1 

    #print(A1,A2,C1d/C1)
   # import pdb ; pdb.set_trace()

    mask = (y00!=-9999.)
    dy = gammaI.std() * np.ones_like(y00) #np.ones_like(y00) * dy00[mask & (dy00!=0.)].mean()

    T = C1 * x0 + C1d * stellar_x0 #+ C2 * (sumsij - S2)
    chi2 = (y00[mask] - T[mask] )**2 
    chi2 = chi2 / dy[mask] / dy[mask]
    chi2 = np.sum(chi2)

    like = -0.5 * chi2

    print('chi2 = %3.3f'%chi2)
    print('(reduced = %3.3f)'%(chi2/(len(T) - 3)))

    block[names.data_vector, 'tatt_direct'+"_CHI2"] = chi2
    block[names.likelihoods, 'tatt_direct'+"_LIKE"] = like
    

    #chis = chi2_dist(50, gamma_I, T, sigma)
    #import pdb ; pdb.set_trace()


    return 0

def chi2_dist(dof, D, T, dD):
    nsub = D.flatten().shape[0]/dof
    cvec = []
    T = T.reshape(D.shape)
    for i in range(nsub):
        #import pdb ; pdb.set_trace()
        res = D - T
        X = np.sum(( res.flatten()[i*dof:(i+1)*dof] )**2  / dD.flatten()[i*dof:(i+1)*dof]  / dD.flatten()[i*dof:(i+1)*dof] )
        cvec.append(X)

    return np.array(cvec)




def get_binned(i, gamma,mask):
    y = gamma[:,:,:,i,i][mask].mean()
    dy = gamma[:,:,:,i,i][mask].std()
    n = len(gamma[:,:,:,i,i][mask])
    
    #yvec.append(y)
    #dyvec.append(dy)

    return y, dy











