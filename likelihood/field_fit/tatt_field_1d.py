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


def load_gamma(nx):
    # load and plot the smoothed galaxy shape data
    cat=fi.FITS('/home/ssamurof/hydro_ias/data/cats/fits/TNG300-1_99_non-reduced_galaxy_shapes.fits')[-1].read()

    def get_trace_free_matrix(I):
        ndim = len(I[0])   
        T = np.matrix.trace(I)
        K = 1./float(ndim) * T * np.identity(ndim)
        I_TF = I - K
        return I_TF

    ngal = len(cat['x'])
    gammaI = np.zeros((nx,nx,nx,3,3))
    num = np.zeros((nx,nx,nx,3,3))


    X,Y,Z = np.meshgrid(np.arange(0,nx,1),np.arange(0,nx,1),np.arange(0,nx,1))
    B = np.linspace(0,300*0.69,nx+1)
    lower,upper = B[:-1],B[1:]

    #print(cat['x'].max())

    # Compute the shape cube by adding galaxies one at a time to cells
    for i in range(ngal):
        #print ("Processing galaxy %d/%d"%(i,ngal))

        # put this galaxy into a cell
        info = cat[i]
        ix = np.argwhere((cat['x'][i]>=lower) & (cat['x'][i]<upper))[0,0]
        iy = np.argwhere((cat['y'][i]>=lower) & (cat['y'][i]<upper))[0,0]
        iz = np.argwhere((cat['z'][i]>=lower) & (cat['z'][i]<upper))[0,0]

        # reconstruct the 3x3 shape matrix for this galaxy
        a0 = np.array([cat['av_x'][i],cat['av_y'][i],cat['av_z'][i]])
        b0 = np.array([cat['bv_x'][i],cat['bv_y'][i],cat['bv_z'][i]])
        c0 = np.array([cat['cv_x'][i],cat['cv_y'][i],cat['cv_z'][i]])
        V = np.diag(np.array([cat['a'][i]**2, cat['b'][i]**2, cat['c'][i]**2]))
        v = np.array([a0,b0,c0])
        v0=np.linalg.inv(v)
        I = np.dot(v0,np.dot(V,v))

        I_TF = get_trace_free_matrix(I)

        # add it to the correct cell
        gammaI[ix,iy,iz,:,:] += I_TF
        num[ix,iy,iz,:,:] += 1

    gammaI /= num
    gammaI[np.isinf(gammaI)]=-9999.
    gammaI[np.isnan(gammaI)]=-9999.
    gammaI[(num==1)]=-9999.



    dgammaI = np.zeros((nx,nx,nx,3,3))

    for i in range(ngal):
        #print ("Processing galaxy %d/%d"%(i,ngal))

        # put this galaxy into a cell
        info = cat[i]
        ix = np.argwhere((cat['x'][i]>=lower) & (cat['x'][i]<upper))[0,0]
        iy = np.argwhere((cat['y'][i]>=lower) & (cat['y'][i]<upper))[0,0]
        iz = np.argwhere((cat['z'][i]>=lower) & (cat['z'][i]<upper))[0,0]

        # reconstruct the 3x3 shape matrix for this galaxy
        a0 = np.array([cat['av_x'][i],cat['av_y'][i],cat['av_z'][i]])
        b0 = np.array([cat['bv_x'][i],cat['bv_y'][i],cat['bv_z'][i]])
        c0 = np.array([cat['cv_x'][i],cat['cv_y'][i],cat['cv_z'][i]])
        V = np.diag(np.array([cat['a'][i]**2, cat['b'][i]**2, cat['c'][i]**2]))
        v = np.array([a0,b0,c0])
        v0=np.linalg.inv(v)
        I = np.dot(v0,np.dot(V,v))

        I_TF = get_trace_free_matrix(I)

        # add it to the correct cell
        dgammaI[ix,iy,iz,:,:] += (I_TF - gammaI[ix,iy,iz,:,:])**2

    #import pdb ; pdb.set_trace()

    dgammaI/=num

    dgammaI[num<2] = dgammaI[np.isfinite(dgammaI)].max()
    dgammaI = np.sqrt(dgammaI)

    return gammaI, dgammaI


def setup(options):

    loc = options.get_string(option_section, "tensor_dir")
    snapshot = options.get_int(option_section, "snapshot")
    nx = options.get_int(option_section, "resolution") # pixel resolution 128, 64, 32, 16
    use_binned = options.get_bool(option_section,"use_binned", default=False)

    base='/home/rmandelb.proj/ssamurof/mb2_tidal/'


    nxyz = fi.FITS('%s/density/dm_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()
    gxyz = fi.FITS('%s/density/star_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()
    n0 = int(nxyz.shape[0]/2)

    # now compute the tidal tensor
    k  = npf.fftfreq(nx)[np.mgrid[0:nx,0:nx,0:nx]]
    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    galaxy_tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)

    # overdensity field
    K = np.mean(nxyz)
    d = nxyz/K -1 
    g = gxyz/np.mean(gxyz) -1 

    # FFT the box
    fft_dens = npf.fftn(d) 
    galaxy_fft_dens = npf.fftn(g)

    F = 2.85
    A = 1./np.pi/np.pi/np.pi/2/2/2.
  #  A=1.
  #  A = F**(128./nx) /np.pi/np.pi/np.pi/2/2/300.



    for i in range(3):
        for j in range(3):
            print(i,j)
            # k[i], k[j] are 3D matrices
            temp = fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)
            galaxy_temp = galaxy_fft_dens * k[i]*k[j]/(k[0]**2 + k[1]**2 + k[2]**2)

            # subtract off the trace...
            if (i==j):
                temp -= 1./3 * fft_dens
                galaxy_temp -= 1./3 * galaxy_fft_dens

            temp[0,0,0] = 0
            tidal_tensor[:,:,:,i,j] = A * npf.ifftn(temp).real
            galaxy_temp[0,0,0] = 0
            galaxy_tidal_tensor[:,:,:,i,j] = A * npf.ifftn(galaxy_temp).real

    print('loading shapes')
    gammaI, dgammaI  = load_gamma(nx)
   # import pdb ; pdb.set_trace()

#    A = F**(64./nx) /np.pi/np.pi/np.pi/2/2/2.
#    s0 = np.mean(tidal_tensor)
#    tidal_tensor = A*tidal_tensor
#    s1 = np.mean(tidal_tensor)
#    tidal_tensor=tidal_tensor-s1+s0
#
    S = tidal_tensor.reshape(int(tidal_tensor.size/3./3.),3,3)
    S2 = np.zeros_like(S)
    delta_tidal = np.zeros_like(S)

    for l,s in enumerate(S):
        M = np.zeros((3,3))

        # the density weighting term
        # just rescale the tidal tensor by the normalised matter overdensity 
        #import pdb ; pdb.set_trace()
        delta_tidal[l,:,:] = s * d.flatten()[l]

        for i in range(3):
            for j in range(3):
                M[i,j] = np.sum(s[i,:]*s[:,j])
                if (i==j):
                    M[i,j]-=(1./3)*np.linalg.det(s)**2

        S2[l] = M
        #print(l)

    import pdb ; pdb.set_trace()

   # S2=np.array(S2)



    #S2 = np.array([np.dot(s,s) for s in S])
#
#    s0 = np.mean(galaxy_tidal_tensor)
#    galaxy_tidal_tensor = A*galaxy_tidal_tensor
#    s1 = np.mean(galaxy_tidal_tensor)
#    galaxy_tidal_tensor=galaxy_tidal_tensor-s1+s0

   # import pdb ; pdb.set_trace()


    #fi.FITS(base+'tidal/raw/star_tidal_traceless_0%d_0.25_%d.fits'%(snapshot,nx))[-1].read()

   # tidal_tensor*=np.std(tidal_tensor[tidal_tensor!=-9999.])**2
   # galaxy_tidal_tensor*=np.std(galaxy_tidal_tensor[galaxy_tidal_tensor!=-9999.])**2

    #dgammaI= np.zeros_like(gammaI)
    #for i in range(3):
    #    for j in range(3):
    #        dgammaI[:,:,:,i,j] = np.std(np.unique(gammaI[:,:,:,i,j]))

    dx = tidal_tensor.std()
    xc = tidal_tensor.mean()
    x = np.linspace(xc-2*dx, xc+2*dx, 20)
    x0 = (x[:-1]+x[1:])/2

    y00, dy00 = [], []
    y11, dy11 = [], []
    y22, dy22 = [], []

    if use_binned:
        for i,(lower,upper) in enumerate(zip(x[:-1],x[1:])):
            mask00 = (tidal_tensor[:,:,:,0,0]>lower) & (tidal_tensor[:,:,:,0,0]<upper)
            mask11 = (tidal_tensor[:,:,:,1,1]>lower) & (tidal_tensor[:,:,:,1,1]<upper)
            mask22 = (tidal_tensor[:,:,:,2,2]>lower) & (tidal_tensor[:,:,:,2,2]<upper)

            y00, dy00 = get_binned(0, gammaI, mask00, y00, dy00)
            y11, dy11 = get_binned(1, gammaI, mask11, y11, dy11)
            y22, dy22 = get_binned(2, gammaI, mask22, y22, dy22)

    
    return delta_tidal, gammaI, tidal_tensor, galaxy_tidal_tensor, S2, dgammaI, x0, y00, y11, y22, dy00, dy11, dy22, use_binned



def execute(block, config):
    delta_tidal, gammaI, tidal_tensor, galaxy_tidal_tensor, S2, dgammaI, x0, y00, y11, y22, dy00, dy11, dy22, use_binned = config

    A1 = block['intrinsic_alignment_parameters', 'A1']
    A2 = block['intrinsic_alignment_parameters', 'A2']

    C1 = A1 * CONST 
    C2 = A2 * CONST2

    bias_ta = block['intrinsic_alignment_parameters', 'bias_ta']
    C1d = bias_ta * C1 

    print(A1)


    if use_binned:
        T = C1 * x0 #+ C1d * dsij.flatten() + C2 * (sumsij - S2)
        chi2 = (y00 - T )**2 
        chi2 = chi2 / dy00 / dy00

    else:
        D = gammaI[:,:,:,:,:].flatten()
        dy=np.ones_like(D) * np.std(D[(D!=-9999.)]) * 2
        mask = (D!=-9999)  

        x = tidal_tensor[:,:,:].flatten()
        x2 = S2[:,:,:].flatten()
        deltaSij = delta_tidal[:,:,:].flatten()

        T = (C1 * x) + (C1d*deltaSij) + (C2 * x2)
        #dy = dgammaI[:,:,:,0,0].flatten()
          #(CONST*x>-0.95) & (CONST*x<0.95)
        #np.array([gammaI[:,:,:,0,0].std()]*len(gammaI[:,:,:,0,0].flatten()))
        chi2 = (D[mask] - T[mask] )**2 
        chi2 = chi2 / dy[mask] / dy[mask]
        #import pdb ; pdb.set_trace()

    chi2 = np.sum(chi2)


    like = -0.5 * chi2

    print('chi2 = %3.3f'%chi2)
    print('(reduced = %3.3f)'%(chi2/(len(T) - 1)))

    block[names.data_vector, 'tatt_direct'+"_CHI2"] = chi2
    block[names.likelihoods, 'tatt_direct'+"_LIKE"] = like
    

    #chis = chi2_dist(50, gamma_I, T, sigma)
  #  import pdb ; pdb.set_trace()


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




def get_binned(i, gamma,mask, yvec, dyvec):
    y = gamma[:,:,:,i,i][mask].mean()
    dy = gamma[:,:,:,i,i][mask].std()
    n = len(gamma[:,:,:,i,i][mask])
    
    yvec.append(y)
    dyvec.append(dy)

    return yvec, dyvec











