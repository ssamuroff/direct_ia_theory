from cosmosis.datablock import names, option_section
import fitsio as fi
import numpy as np
import numpy.fft as npf


# these are computed at the TNG cosmology
CONST = -0.005307
CONST2 = 0.033832


def setup(options):

    base = options.get_string(option_section, "tensor_dir")
    snapshot = options.get_int(option_section, "snapshot")
    nx = options.get_int(option_section, "resolution") # pixel resolution 128, 64, 32, 16

  #  base='/home/rmandelb.proj/ssamurof/tng_tidal/'

    # gridded particle density data
    nxyz = fi.FITS('%s/density/dm_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()
    gxyz = fi.FITS('%s/density/star_density_0%d_%d.fits'%(base,snapshot,nx))[-1].read()

    

    # overdensity field
    d = nxyz/np.mean(nxyz) -1 
    g = gxyz/np.mean(gxyz) -1 

    tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)
    galaxy_tidal_tensor = np.zeros((nx,nx,nx,3,3),dtype=np.float32)

    # FFT the box
    fft_dens = npf.fftn(d) 
    galaxy_fft_dens = npf.fftn(g) 

    A=1. #/2/np.pi #/(2.*np.pi)**3 

    # now compute the tidal tensor
    k  = npf.fftfreq(nx)[np.mgrid[0:nx,0:nx,0:nx]]

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
    gammaI = load_gamma(nx)

    
    return gammaI, tidal_tensor, galaxy_tidal_tensor



def execute(block, config):
    gammaI, tidal_tensor, galaxy_tidal_tensor = config

    A1 = block['intrinsic_alignment_parameters', 'A1']
    A2 = block['intrinsic_alignment_parameters', 'A2']

    C1 = A1 * CONST 
    C2 = A2 * CONST2

    bias_ta = block['intrinsic_alignment_parameters', 'bias_ta']
    C1d = bias_ta * C1 

    print(A1)

    D = gammaI[:,:,:,0,0].flatten()
    x = tidal_tensor[:,:,:,0,0].flatten()
    gx = galaxy_tidal_tensor[:,:,:,0,0].flatten()
    T = (C1 * x) + (C1d*gx)

    mask = (D!=-9999) # mask out cells with no galaxies

    # assume just shape noise for the covariance
    # this is a bit hacked, but will do for the moment
    dy = np.ones_like(D) * np.std(D[mask])

    chi2 = (D[mask] - T[mask] )**2
    chi2 = chi2 / dy[mask] / dy[mask]
    chi2 = np.sum(chi2)


    like = -0.5 * chi2

    print('chi2 = %3.3f'%chi2)
    print('(reduced = %3.3f)'%(chi2/(len(T) - 1)))

    block[names.data_vector, 'tatt_direct'+"_CHI2"] = chi2
    block[names.likelihoods, 'tatt_direct'+"_LIKE"] = like
    

    return 0

def get_trace_free_matrix(I):
    ndim = len(I[0])   
    T = np.matrix.trace(I)
    K = 1./float(ndim) * T * np.identity(ndim)
    I_TF = I - K
    return I_TF


def load_gamma(nx):
    # this is a subhalo shape catalogue
    # one entry per galaxy
    cat=fi.FITS('/home/ssamurof/hydro_ias/data/cats/fits/TNG300-1_99_non-reduced_galaxy_shapes.fits')[-1].read()

    ngal = len(cat['x'])
    gammaI = np.zeros((nx,nx,nx,3,3))
    num = np.zeros((nx,nx,nx,3,3))


    X,Y,Z = np.meshgrid(np.arange(0,nx,1),np.arange(0,nx,1),np.arange(0,nx,1))
    B = np.linspace(0,300*0.69,nx+1)
    lower,upper = B[:-1],B[1:]

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

    return gammaI




