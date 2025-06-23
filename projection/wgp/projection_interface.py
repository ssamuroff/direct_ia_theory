from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from hankl import FFTLog

# constants
clight = 299792.4580 # kms^-1
sigmaz_a=0.02
sigmaz_b=0.02
alpha=1.719

def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="lens")
    sample_b = options.get_string(option_section, "sample_b", default="lens")

    include_lensing = options.get_bool(option_section, "include_lensing", default=True)
    include_magnification = options.get_bool(option_section, "include_magnification", default=True)

    constant_sigmaz = options.get_bool(option_section, "constant_sigmaz", default=True)
    if not constant_sigmaz: 
        print('Using a linear approximation for sigma_z(z)')
    else:
        print('Using constant sigmaz=%2.2f'%sigmaz_a)

    timing = options.get_bool(option_section, "timing", default=True)

    return (sample_a,sample_b,timing,include_lensing,include_magnification,constant_sigmaz)

def execute(block, config):
    sample_a, sample_b, timing, include_lensing, include_magnification, constant_sigmaz = config

    if timing:
        from time import time
        T0 = time()

    # binning for integrals
    # choose a set of bins for line-of-sight separation 
    Npi = 80
    Nz0 = 50
    Pi = np.linspace(-500,500,Npi)
    z_low = np.linspace(0.01,3.00,Nz0)

    z_distance = block['distances', 'z']
    chi_distance = block['distances', 'd_m']*block['cosmological_parameters', 'h0']
    a_distance = 1./(1+z_distance)
    chi_of_z_spline = interp1d(z_distance, chi_distance)

    zf = np.linspace(0.0,3.00,5000)
    chi = chi_of_z_spline(zf)

    # if the redshift error has some z dependence, that gets set here
    # otherwise sigmaz is a constant
    if not constant_sigmaz:
        sigmaz_a = get_sigma_z(block, zf, sample_a)
        sigmaz_b = get_sigma_z(block, zf, sample_b)
    else:
        sigmaz_a=0.02
        sigmaz_b=0.011

    P_gI = block['galaxy_intrinsic_power','p_k']
    k_power = block['galaxy_intrinsic_power','k_h']
    z_power = block['galaxy_intrinsic_power','z']
    chi_power = chi_of_z_spline(z_power)
    P_gI_interpolator = interp2d(k_power,chi_power,P_gI)

    Nell = 300
    ell = np.logspace(-6,np.log10(19000),Nell) 
    Cell_all = np.zeros((Nz0, Npi, Nell))

    P_gI_2d=[]
    for i, l in enumerate(ell):
        P1d = [P_gI_interpolator((l+0.5)/x, x) for x in chi]
        #P1d = np.diag(np.fliplr(P_gI_interpolator((l+0.5)/chi, chi))) 
        P_gI_2d.append(P1d)
        #import pdb ; pdb.set_trace()
    P_gI_2d = np.array(P_gI_2d)



    if include_lensing:
        P_gG = block['matter_galaxy_power','p_k']
        P_gG_interpolator = interp2d(k_power,chi_power,P_gG)

        P_mG = 2*(alpha-1)*block['matter_power_nl','p_k']
        P_mG_interpolator = interp2d(k_power,chi_power,P_mG)

        P_gG_2d=[]
        P_mG_2d=[]
        for i, l in enumerate(ell):
            P1d_gG = [P_gG_interpolator((l+0.5)/x, x) for x in chi]
            #P1d_gG = np.diag(np.fliplr(P_gG_interpolator((l+0.5)/chi, chi)))
            P_gG_2d.append(P1d_gG)

            P1d_mG = [P_mG_interpolator((l+0.5)/x, x) for x in chi]
            #P1d_mG = np.diag(np.fliplr(P_mG_interpolator((l+0.5)/chi, chi)))
            P_mG_2d.append(P1d_mG)

            #import pdb ; pdb.set_trace()

        P_gG_2d = np.array(P_gG_2d)
        P_mG_2d = np.array(P_mG_2d)


    # first bit: Limber integrals
    if timing:
        T1 = time()
        print('Starting loop')

    # we loop over a grid of los separation Pi and mean z z0
    for i, z_l in enumerate(z_low):
        for j,pi in enumerate(Pi):
            
            # coordinate transform
            Hz = 100 * np.sqrt(block['cosmological_parameters','omega_m']*(1+z_l)**3 + block['cosmological_parameters', 'omega_lambda']) # no h because Pi is in units h^-1 Mpc
            z1 = z_l
            z2 = z_l + (1./clight * Hz * pi)
            if z2<0: 
                continue

            Pz1 = gaussian(zf, sigmaz_a, z1)
            Pz1 = Pz1/np.trapz(Pz1,chi)

            Pz2 = gaussian(zf, sigmaz_b, z2)
            Pz2 = Pz2/np.trapz(Pz2,chi)

            gz1 = get_approximate_lensing_kernel(block, chi_of_z_spline(z1), chi, 1./(1+zf))
            gz2 = get_approximate_lensing_kernel(block, chi_of_z_spline(z2), chi, 1./(1+zf))

            C_gI = do_limber_integral(ell, P_gI_2d, Pz1, Pz2, chi)
            Cell_all[i,j,:]=C_gI

            if include_lensing:
                C_gG = do_limber_integral(ell, P_gG_2d, Pz1, gz2, chi)
                Cell_all[i,j,:]+=C_gG
            if include_magnification:
                C_mG = do_limber_integral(ell, P_mG_2d, gz1, gz2, chi)
                Cell_all[i,j,:]+=C_mG

            #import pdb ; pdb.set_trace()
   
    # Next do the Hankel transform
    xi_all = np.zeros_like(Cell_all)-9999.
    rp = np.logspace(np.log10(0.1), np.log10(300), xi_all.shape[2])

    if timing:
        T2 = time()
        print('Hankel transform...')

    for i, z in enumerate(z_low):
        x0 =  chi_of_z_spline(z)
        # do the coordinate transform to convert theta to rp at given redshift
        theta_radians = np.arctan(rp/x0)
        theta_degrees = theta_radians * 180./np.pi
        for j,pi in enumerate(Pi):
            Cell = Cell_all[i,j,:]

            theta_new,xi_new = FFTLog(ell, Cell*ell, 0, 2, lowring=True)
            xi_new = -xi_new/theta_new/2/np.pi
            xi_interpolated = interp1d(theta_new,xi_new,fill_value="extrapolate")(theta_radians)
            xi_all[i,j,:]=xi_interpolated


    # integrate over los separation, between +/-Pi_max
    Pi_max=100.
    mask = ((Pi<Pi_max) & (Pi>-Pi_max))
    xi_rp = np.trapz(xi_all[:,mask,:], x=Pi[mask], axis=1)

    # and then over redshift
    za, W = get_redshift_kernel(block, 0, 0, zf, chi, sample_a, sample_b)
    W/=np.trapz(W,zf)
    Wofz = interp1d(zf,W)
    K = np.array([Wofz(z_low)]*len(rp)).T
    w_rp = np.trapz(xi_rp*K, x=z_low, axis=0) #/np.trapz(K, Zm, axis=0)


    if timing:
        T3 = time()
        print('time 1:',T1-T0)
        print('time 2:',T2-T1)
        print('time 3:',T3-T2)

    # finally save wg+ to the data block
    block.put_double_array_1d('galaxy_intrinsic_w', 'w_rp_1_1_%s_%s'%(sample_a,sample_b), w_rp)
    block['galaxy_intrinsic_w', 'r_p'] = rp

    return 0

def gaussian(x,s,m):
    return np.exp(-(x-m)*(x-m)/2/s/s)


def get_approximate_lensing_kernel(block, X0, chi, az):
    H0 = block['cosmological_parameters', 'h0']*100
    omega_m = block['cosmological_parameters', 'omega_m']
    gz = 3/2 * H0 * H0 * omega_m / clight / clight * (chi/az) * (X0-chi) / X0
    gz[gz<0] = 0.
    return gz


def get_lensing_kernel(block, chi, pz, az):
    H0 = block['cosmological_parameters', 'h0']*100
    omega_m = block['cosmological_parameters', 'omega_m']
    gz=[]
    for i,x in enumerate(chi):
        z = 1/az - 1
        dz_dX = np.gradient(z,chi)
        coeff = 3/2 * H0 * H0 * omega_m / clight / clight * (x/az[i])
        #I = np.trapz(pz*dz_dX*(X-x)/X, X)
        I = np.trapz(pz*(chi-x)/chi, chi)
        gz.append(coeff*I)
    gz = np.array(gz)
    gz[gz<0] = 0.
    return gz

def do_limber_integral(ell, P, p1, p2, X):

    I1 = interp1d(X,p1)
    I2 = interp1d(X,p2)
    cl = [] 
    Az = 1./X/X*I1(X)*I2(X)
    Az[np.isinf(Az)]=0
    Az[np.isnan(Az)]=0

    Az_reshaped = np.array([Az]*P.shape[0])
    cl = np.trapz(Az_reshaped*P[:,:,0],X,axis=1)

 #   cl = np.sum(Az_reshaped*P[:,:,0],axis=1)*(X[1]-X[0])

#    from time import time

 #   T0 = time()
    
#    for i, l in enumerate(ell):

        #P1d = [P((l+0.5)/x, x) for x in X]
#        P1d = P[i,:]
#        P1d[np.isinf(P1d)]=0
 #       P1d[np.isnan(P1d)]=0

  #      K = np.trapz(Az*np.array(P1d)[:,0],X)
  #      K = np.sum(Az*np.array(P1d)[:,0])*(X[1]-X[0])
 #       cl.append(K)
#
  #  import pdb ; pdb.set_trace()

 #   cl2 = [] 
 #   dX = X[1]-X[0]

 #   T1 = time()
    
 #   for i, l in enumerate(ell):
 #       K = 0
 #       for x in X:
 #           Az = 1./x/x*I1(x)*I2(x)
 #           if not np.isfinite(Az):
 #               Az = 0.

 #           Pkz = P((l+0.5)/x, x)
#
#            K+=Az*Pkz
#
 #       cl2.append(K*dX)
 #   T2 = time()

 #   print(T1-T0)


    return np.array(cl)

def get_redshift_kernel(block, i, j, z0, x, sample_a, sample_b):


    dz = z0[1]-z0[0]
    dxdz = np.gradient(x,dz)
    #interp_dchi = spi.interp1d(z,Dchi)

    na = block['nz_%s'%sample_a, 'nbin']
    nb = block['nz_%s'%sample_b, 'nbin']
    zmin = 0.01

    nz_b = block['nz_%s'%sample_b, 'bin_%d'%(j+1)]
    zb = block['nz_%s'%sample_b, 'z']
    nz_a = block['nz_%s'%sample_a, 'bin_%d'%(i+1)]
    za = block['nz_%s'%sample_a, 'z']

    interp_nz = interp1d(zb, nz_b, fill_value='extrapolate')
    nz_b = interp_nz(z0)
    interp_nz = interp1d(za, nz_a, fill_value='extrapolate')
    nz_a = interp_nz(z0)


    X = nz_a * nz_b/x/x/dxdz
    X[0]=0
    interp_X = interp1d(z0, X, fill_value='extrapolate')

    # Inner integral over redshift
    V,Verr = quad(interp_X, zmin, z0.max())
    W = nz_a*nz_b/x/x/dxdz/np.trapz(X,x=z0)
    #V
    W[0]=0

    return z0,W

def get_sigma_z(block, z, sample):

    #import pdb ; pdb.set_trace()
    az = block['photoz_errors','a_%s'%sample]
    bz = block['photoz_errors','b_%s'%sample]
    sigmaz = az*z + bz

    # we don't want to extrapolate the linear sigmaz to very extreme redshifts
    # so set some limits here
    nofz = block['nz_redmagic_density','bin_1']
    z_nofz = block['nz_redmagic_density','z']
    N0 = nofz.max()/100

    z1 = z_nofz[nofz>N0][0]
    z2 = z_nofz[nofz>N0][-1]

    sigmaz[(z<z1)]=0.01
    sigmaz[(z>z2)]=0.01

    return sigmaz


