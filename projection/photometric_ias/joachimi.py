from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import limber_lib as limber
from gsl_wrappers_local import GSLSpline, GSLSpline2d, NullSplineError, BICUBIC, BILINEAR
import hankel_transform as sukhdeep
import sys
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as sint
import mcfit
from mcfit import P2xi
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.special import eval_legendre as legendre
from scipy import integrate
from scipy.integrate import cumtrapz
from scipy.integrate import quad
import glob
import time
from pyccl import Cosmology
from pyccl import correlation
import pylab as plt
plt.switch_backend('pdf')
#plt.style.use('y1a1')


# An implementation of the theory prediction for direct IA measurements with photometric samples
# see Joachimi et al 2010 for details https://arxiv.org/abs/1008.3491
# The module has two modes: the first does the Limber integral to give C(ell|z1,z2), using an
# estimate for the per-galaxy redshift PDFs
# The second mode reads back Hankel transformed C(ells) and integrates over redshift and 
# line-of-sight separation to give the projected correlation function w(r_p) 



clight = 299792.4580 # kms^-1
relative_tolerance = 1e-3
absolute_tolerance = 0.

def setup(options):
    # sample info
    sample_a = options.get_string(option_section, "sample_a", default="lens")
    sigma_a = options.get_double(option_section, "sigma_a", default=0.01)
    sample_b = options.get_string(option_section, "sample_b", default="lens")
    sigma_b = options.get_double(option_section, "sigma_b", default=0.01)

    pzmethod = options.get_string(option_section, "pzmethod", default="analytic")
    sigmaz_file = options.get_string(option_section, "sigmaz_file", default="")

    if pzmethod=='interpolate':
        print("using interpolated empirical sigma_z from %s"%sigmaz_file)
        z0,sz = np.loadtxt(sigmaz_file).T
        Sz_interpolator = interp1d(z0,sz) 
    else:
        print("using analytic Gaussian error sigma_z = (1+z)x%1.4f"%sigma_a)
        Sz_interpolator = None


    do_lensing = options.get_bool(option_section, "include_lensing", default=False)
    do_magnification = options.get_bool(option_section, "include_magnification", default=False)

    use_precomputed_cls = options.get_bool(option_section, "use_precomputed_cls", default=False)



    if do_lensing:
        print('Will include lensing contributions')
    else:
        print('Will not include lensing :(')
    if do_magnification:
        print('Will include magnification contributions')
    else:
        print('Will not include magnification :(')

    ell_max = options.get_double(option_section, "ell_max", default=10000)
    nell = options.get_double(option_section, "nell", default=60)

    # binning
    pimax = options.get_double(option_section, "pimax", default=68.) # in h^-1 Mpc

    # order of the Bessel function used in the Hankel transform
    nu = options.get_int(option_section, "bessel_type", default=2)

    if nu==-1:
        transform = sukhdeep.hankel_transform(rmin=0.05, rmax=205., kmin=1e-2, kmax=80., j_nu=[2], n_zeros=91000)
    else:
        transform=None

    # where to read from and save to
    input_name = options.get_string(option_section, "input_name", default="galaxy_intrinsic_power")
    output_name = options.get_string(option_section, "output_name", default="galaxy_intrinsic_w")

    if use_precomputed_cls:
        cl_dict = get_cls_from_file(input_name, do_lensing, do_magnification, sample_a, sample_b)
    else:
        cl_dict = None


    return sample_a, sigma_a, sample_b, sigma_b, pimax, nu, input_name, output_name, ell_max, nell, transform, do_lensing, do_magnification, use_precomputed_cls, cl_dict, Sz_interpolator

def growth_from_power(chi, k, p, k_growth):
    "Get D(chi) from power spectrum"
    growth_ind=np.where(k>k_growth)[0][0]
    growth_array = np.sqrt(np.divide(p[:,growth_ind], p[0,growth_ind], out=np.zeros_like(p[:,growth_ind]), where=p[:,growth_ind]!=0.))
    return GSLSpline(chi, growth_array)

def load_power_growth_chi(block, chi_of_z, section, k_name, z_name, p_name, k_growth=1.e-3):
    z,k,p = block.get_grid(section, z_name, k_name, p_name)
    chi = chi_of_z(z)
    growth_spline = growth_from_power(chi, k, p, k_growth)
    power_spline = GSLSpline2d(chi, np.log(k), p.T, spline_type=BICUBIC)
    return power_spline, growth_spline


def get_approximate_lensing_kernel(block, X, pz, X0, az):
    #X0 = np.trapz(pz*X,X)
    H0 = block['cosmological_parameters', 'h0']*100
    omega_m = block['cosmological_parameters', 'omega_m']

    gz = 3/2 * H0 * H0 * omega_m / clight / clight / az * X * (X0-X) / X0
    gz[gz<0] = 0.
    #if sum(gz)>0:
    #    gz/=np.trapz(gz,X)
    
    return gz



def get_lensing_terms(block, input_name, do_lensing, do_magnification, ell, P_flat, pz1, pz2, X, X01, X02, z1, z2, az, sample_a, sample_b, save_cls=False):

    if (not do_magnification) and (not do_lensing):
        return 0

    gz1 = get_approximate_lensing_kernel(block, X, pz1, X01, az)
    gz2 = get_approximate_lensing_kernel(block, X, pz2, X02, az)

    vec = np.zeros_like(ell)

    if input_name=='galaxy_intrinsic_power':

        # gI --------------------
        P_gI = P_flat['galaxy_intrinsic_power']
        K_gI = block['bias_parameters', 'b_%s'%sample_a]

        if (sum(pz1*pz2)==0):
            C_gI = np.zeros_like(ell)
        else:
            C_gI = K_gI * do_limber_integral(ell, P_gI, pz1, pz2, X)

            

        if do_lensing:
            # gG --------------------
            P_GG = P_flat['matter_power_nl']
            K_gG = block['bias_parameters', 'b_%s'%sample_a]

            if (sum(pz1*gz2)==0):
                C_gG = np.zeros_like(ell)
            else:
                C_gG = do_limber_integral(ell, P_GG, pz1, gz2, X)

            vec += K_gG * C_gG


        if do_magnification and do_lensing:
            # mG --------------------

            P_GI = P_flat['matter_intrinsic_power']

            K_mG = 2 * (block['galaxy_luminosity_function', 'alpha_%s'%sample_a] -1)

            if (sum(gz1*gz2)==0):
                C_mG = np.zeros_like(ell)
            else:
                C_mG = do_limber_integral(ell, P_GG, gz1, gz2, X)

            vec += K_mG * C_mG

        if do_magnification:
            # mI --------------------

            P_GI = P_flat['matter_intrinsic_power']

            K_mI = 2 * (block['galaxy_luminosity_function', 'alpha_%s'%sample_a] -1)

            if (sum(gz1*pz2)==0):
                C_mI = np.zeros_like(ell)
            else:
                C_mI = do_limber_integral(ell, P_GI, gz1, pz2, X)

            vec += K_mI * C_mI

        if save_cls:
            base='cls/cl_gI-'+'%s_%s-z1_%3.3f-z2_%3.3f.txt'%(sample_a, sample_b, z1,z2)
            np.savetxt(base, np.vstack((ell, K_gI*C_gI, K_gG*C_gG, K_mI*C_mI, K_mG*C_mG)).T)



    if input_name=='galaxy_power':
        # gg --------------------

        P_GG = P_flat['matter_power_nl']

        K_gg = block['bias_parameters', 'b_%s'%sample_a] * block['bias_parameters', 'b_%s'%sample_b]
#
        if (sum(pz1*pz2)==0):
            C_gg = np.zeros_like(ell)
        else:
            C_gg = do_limber_integral(ell, P_GG, pz1, pz2, X)

        #vec += K_gg * C_gg

        if do_magnification:
            # mm --------------------
            K_mm = 4. * (block['galaxy_luminosity_function', 'alpha_%s'%sample_a] -1) * (block['galaxy_luminosity_function', 'alpha_%s'%sample_b] -1)

            if (sum(gz1*gz2)==0):
                C_mm = np.zeros_like(ell)
            else:
                C_mm = do_limber_integral(ell, P_GG, gz1, gz2, X)

            vec += K_mm * C_mm

            # mg --------------------

            K_mg = 2. * (block['galaxy_luminosity_function', 'alpha_%s'%sample_a] -1) * block['bias_parameters', 'b_%s'%sample_b]

            if (sum(gz1*pz2)==0):
                C_mg = np.zeros_like(ell)
            else:
                C_mg = do_limber_integral(ell, P_GG, gz1, pz2, X)

            vec += K_mg * C_mg

            # gm --------------------

            K_gm = 2. * (block['galaxy_luminosity_function', 'alpha_%s'%sample_b] -1) * block['bias_parameters', 'b_%s'%sample_a]

            if (sum(pz1*gz2)==0):
                C_gm = np.zeros_like(ell)
            else:
                C_gm = do_limber_integral(ell, P_GG, pz1, gz2, X)

            vec += K_gm * C_gm

        if save_cls:
            base='cls/cl_gg-'+'%s_%s-z1_%3.3f-z2_%3.3f.txt'%(sample_a, sample_b, z1,z2)
            np.savetxt(base, np.vstack((ell, K_gg * C_gg, K_mm * C_mm, K_mg * C_mg, K_gm * C_gm)).T)

    if input_name=='intrinsic_power':

        # GG --------------------

        if do_lensing:
            P_GG = P_flat['matter_power_nl']

            K_GG = 1.

            if (sum(gz1*gz2)==0):
                C_GG = np.zeros_like(ell)
            else:
                C_GG = do_limber_integral(ell, P_GG, gz1, gz2, X)

            vec += K_GG * C_GG 

        # II --------------------

        P_II = P_flat['intrinsic_power']

        K_II = 1.

        if (sum(pz1*pz2)==0):
            C_II = np.zeros_like(ell)
        else:
            C_II = do_limber_integral(ell, P_II, pz1, pz2, X)

        #vec += K_II * C_II

        # GI --------------------

        if do_lensing:
            P_GI = P_flat['matter_intrinsic_power']

            K_GI = 1.

            if (sum(gz1*pz2)==0):
                C_GI = np.zeros_like(ell)
            else:
                C_GI = do_limber_integral(ell, P_GI, gz1, pz2, X)

            vec += K_GI * C_GI

            # IG --------------------
            K_IG = 1.

            if (sum(pz1*gz2)==0):
                C_IG = np.zeros_like(ell)
            else:
                C_IG = do_limber_integral(ell, P_GI, pz1, gz2, X)

            vec += K_IG * C_IG

            #if sum(C_GG)!=0: import pdb ; pdb.set_trace() 


        if save_cls:
            base='cls/cl_II-'+'%s_%s-z1_%3.3f-z2_%3.3f.txt'%(sample_a, sample_b, z1,z2)
            np.savetxt(base, np.vstack((ell, K_II * C_II, K_GG * C_GG, K_GI * C_GI, K_IG * C_IG)).T)

      #import pdb ; pdb.set_trace()
    


    return vec


def execute(block, config):
    sample_a, sigma_a, sample_b, sigma_b, pimax, nu, input_name, output_name, ell_max, nell, transform, do_lensing, do_magnification, use_precomputed_cls, cl_dict, Sz_interpolator = config
    


    H0 = block['cosmological_parameters', 'h0']*100
    h0 = block['cosmological_parameters', 'h0']
    omega_m = block['cosmological_parameters', 'omega_m']
    omega_b = block['cosmological_parameters', 'omega_b']
    omega_de = block['cosmological_parameters', 'omega_lambda']
    sigma_8 = 0.8234379064365687 # this is hard coded for now...
    ns = block['cosmological_parameters', 'n_s']

    cosmology = Cosmology(Omega_c=omega_m-omega_b, Omega_b=omega_b, h=h0, sigma8=sigma_8, n_s=ns, matter_power_spectrum='halofit', transfer_function='boltzmann_camb')

    # choose a set of bins for line-of-sight separation 
    npi = 20
    nzm = 50
    Pi = np.hstack((np.linspace(-500,0,npi), np.linspace(0,500,npi)[1:] ))# h^-1 Mpc
    npi = len(Pi)
    Zm = np.linspace(0.05,2.15,nzm)


    z_distance = block['distances', 'z']
    chi_distance = block['distances', 'd_m']
    a_distance = 1./(1+z_distance)
    chi_of_z_spline = interp1d(z_distance, chi_distance)



    ell = np.logspace(-1,np.log10(8000000),100)
    cl_vec = np.zeros((nzm, npi, len(ell)))

    print('initialising arrays...')
    x1 = np.linspace(0,3,100)
    X = chi_of_z_spline(x1)
    az = 1./(1+x1)

    if not use_precomputed_cls:
        P_flat = load_power_all(block, input_name, chi_of_z_spline, ell, X, do_lensing, do_magnification)

    # first bit: Limber integrals
    print('Starting loop')
    #import pdb ; pdb.set_trace()

    for i, zm in enumerate(Zm):
        for j,p in enumerate(Pi):
            
            # coordinate transform
            Hz = 100 * np.sqrt(omega_m*(1+zm)**3 + omega_de) # no h because Pi is in units h^-1 Mpc
            z1 = zm - (0.5/clight * Hz * p)
            z2 = zm + (0.5/clight * Hz * p)


            if (z1<0) or (z2<0) :
                continue

            if use_precomputed_cls:
                Cell = extract_cls(cl_dict, block, input_name, do_lensing, do_magnification, z1, z2, sample_a, sample_b)
                #get_cls_from_file(block, input_name, do_lensing, do_magnification, z1, z2, sample_a, sample_b)
            else:
                # evaluate the per-galaxy PDFs at z1 and z2
                x1,pz1 = choose_pdf(z1, sigma=sigma_a, interpolator=Sz_interpolator) #gaussian(z1, sigma=sigma_a)
                pz1 /=np.trapz(pz1,X) #pz1.max() 

                x2,pz2 = choose_pdf(z2, sigma=sigma_b, interpolator=Sz_interpolator) #gaussian(z2, sigma=sigma_b)
                pz2 /= np.trapz(pz2,X)
                
                Cell = coeff(block, sample_a, sample_b,  input_name) * do_limber_integral(ell, P_flat[input_name], pz1, pz2, X)
                Cell += get_lensing_terms(block, input_name, do_lensing, do_magnification, ell, P_flat, pz1, pz2, X, chi_of_z_spline(z1), chi_of_z_spline(z2), z1, z2, az, sample_a, sample_b)
                #import pdb ; pdb.set_trace()

            
            cl_vec[i,j,:] = Cell

       #     import pdb ; pdb.set_trace()

            #print(i,j)

    #import pdb ; pdb.set_trace()


    # Next do the Hankel transform
    xi_vec = np.zeros_like(cl_vec)-9999.
    rp_vec = np.logspace(np.log10(0.01), np.log10(500), xi_vec.shape[2])
    theta = 2*np.pi/np.flipud(ell) * (180/np.pi)

    print('Hankel transform...')

    for i, zm in enumerate(Zm):
        x0 =  chi_of_z_spline(zm)
        # do the coordinate transform to convert theta to rp at given redshift
        theta_radians = rp_vec/x0
        theta_degrees = theta_radians * 180./np.pi

        for j,p in enumerate(Pi):
            # select a Cell, at fixed Pi, zm, and Hankel transform it
            j_flipped = len(Pi)-1-j

            if not (xi_vec[i,j,:][0]==-9999.):
                continue

            C = cl_vec[i,j,:]
            #import pdb ; pdb.set_trace()
            if (nu==0):
               # import pdb ; pdb.set_trace() 
                #xi = - (np.pi/np.sqrt(1.04)*np.sqrt(np.pi/2))/1.77 * correlation(cosmology, ell, C, theta_degrees, type='NG', method='FFTLog')
                if (abs(C)<1e-10).all():
                    xi = np.zeros(len(ell))
                else:
                    xi = -(np.pi/np.sqrt(1.04) * np.sqrt(np.pi)/2)/1.77 * correlation(cosmology, ell, C, theta_degrees, type='NG', method='FFTLog')

                #rp, xi = transform.projected_correlation(ell, C, j_nu=2, taper=True)
                #xi = 10**interp1d(np.log10(rp), np.log10(-xi))(np.log10(rp_vec))
            elif (nu==1):
                if (abs(C)<1e-40).all():
                    xi = np.zeros(len(ell))
                else:
                    xi = (np.pi/2) * np.sqrt(1.02)* correlation(cosmology, ell, C, theta_degrees, type='NN', method='FFTLog')
            elif (nu==2):
                if (abs(C)<1e-40).all():
                    xi = np.zeros(len(ell))
                else:
                    xi_0 = correlation(cosmology, ell, C, theta_degrees, type='GG+', method='FFTLog')
                    xi_4 = correlation(cosmology, ell, C, theta_degrees, type='GG-', method='FFTLog')
                    #xi = (1./2/1.08/np.sqrt(np.pi/2))*(xi_0 + xi_4) #/np.sqrt(2)
                    #xi = (1./1.08/np.sqrt(2.*np.pi))*(xi_0 + xi_4) #; import pdb ; pdb.set_trace()
                    xi = 1.0279*np.sqrt(2)*(xi_0 + xi_4)/np.pi
                    #xi = (xi_0 + xi_4)/2/np.pi
                #(np.pi/2/1.08)*(xi_0 + xi_4)

            xi_vec[i,j,:] = xi
            xi_vec[i,j_flipped,:] = xi # by symmetry
            #if (p==0) & (i==10):
            #if sum(xi)!=0.:
            #    import pdb ; pdb.set_trace()

    #rp_vec*=h0

    print('saving')

#    import pdb ; pdb.set_trace()

    xi_vec[np.isnan(xi_vec)]=0.
    xi_vec[np.isinf(xi_vec)]=0.

    # integrate over line of sight separation
    
    mask = ((Pi<pimax) & (Pi>-pimax))
    print('pi_max=%3.3f'%pimax)
    xi_pi_rp = np.trapz(xi_vec[:,mask,:], Pi[mask], axis=1)

    # and then over redshift
    za, W = get_redshift_kernel(block, 0, 0, x1, X, sample_a, sample_b)
    #W[np.isnan(W)] = 0
    #import pdb ; pdb.set_trace() 
    Wofz = interp1d(W,za)
    K = np.array([Wofz(Zm)]*len(rp_vec)).T
    #K = np.array([W]*len(rp_vec)).T
    w_rp = np.trapz(xi_pi_rp*K, Zm, axis=0)/np.trapz(K, Zm, axis=0)

    #bg = block['bias_parameters','b_%s'%sample_a]
    #w_rp*=bg
    #import pdb ; pdb.set_trace()  

    block.put_double_array_1d(output_name, 'w_rp_1_1_%s_%s'%(sample_a,sample_b), w_rp)
    block[output_name, 'r_p'] = rp_vec

    return 0

def extract_cls(cl_dict, block, input_name, do_lensing, do_magnification, z1, z2, sample_a, sample_b):

    #import pdb ; pdb.set_trace()
   # t0=time.time()
    z1 = float('%3.3f'%z1)
    z2 = float('%3.3f'%z2)

    #print(time.time()-t0)


    if (input_name=='galaxy_intrinsic_power'):
        A = block['intrinsic_alignment_parameters', 'A']
        ba = block['bias_parameters', 'b_%s'%sample_a]

       # print(time.time()-t0)

        C = A * ba * cl_dict[('gI', z1, z2, sample_a, sample_b)]

        #print(time.time()-t0)

        if do_lensing:
            C += ba * cl_dict[('gG', z1, z2, sample_a, sample_b)]
            #print(time.time()-t0)
        if do_magnification:
            C += cl_dict[('mI', z1, z2, sample_a, sample_b)]
           # print(time.time()-t0)
        if do_lensing and do_magnification:
            C += cl_dict[('mG', z1, z2, sample_a, sample_b)]
            #print(time.time()-t0)

    elif (input_name=='galaxy_power'):
        ba = block['bias_parameters', 'b_%s'%sample_a]
        bb = block['bias_parameters', 'b_%s'%sample_b]

        C = ba * bb * cl_dict[('gg', z1, z2, sample_a, sample_b)]

        if do_magnification:
            C += cl_dict[('mm', z1, z2, sample_a, sample_b)]
            C += bb * cl_dict[('mg', z1, z2, sample_a, sample_b)]
            C += ba * cl_dict[('gm', z1, z2, sample_a, sample_b)]


    elif (input_name=='intrinsic_power'):
        A = block['intrinsic_alignment_parameters', 'A']

        C = A * A * cl_dict[('II', z1, z2, sample_a, sample_b)]

        if do_lensing:
            C += A * cl_dict[('GI', z1, z2, sample_a, sample_b)]
            C += A * A * cl_dict[('II', z1, z2, sample_a, sample_b)]
            C += cl_dict[('GG', z1, z2, sample_a, sample_b)]

   # print(time.time()-t0)


    return C


def get_cls_from_file(input_name, do_lensing, do_magnification, sample_a, sample_b):

    #A = block['intrinsic_alignment_parameters', 'A']
    #ba = block['bias_parameters', 'b_%s'%sample_a]

    

    cl_dict = {}
    base = glob.glob('cls/cl_*-'+'%s_%s-*.txt'%(sample_a, sample_b))

    for f in base:
        z1 = float(f.split('z1_')[-1].split('-z2_')[0])
        z2 = float(f.split('z1_')[-1].split('-z2_')[1].replace('.txt',''))
        ctype = f.split('cl_')[1].split('-')[0]

        if (input_name=='galaxy_intrinsic_power') and (ctype=='gI'):
            ell, C_gI, C_gG, C_mI, C_mG = np.loadtxt(f).T
            cl_dict[('gI', z1, z2, sample_a, sample_b)] =  C_gI
            cl_dict['ell'] = ell

            if do_lensing:
                cl_dict[('gG', z1, z2, sample_a, sample_b)] = C_gG 
            if do_magnification:
                cl_dict[('mI', z1, z2, sample_a, sample_b)] = C_mI 
            if do_lensing and do_magnification:
                cl_dict[('mG', z1, z2, sample_a, sample_b)] = C_mG 

        elif (input_name=='galaxy_power') and (ctype=='gg'):
            ell, C_gg, C_mm, C_mg, C_gm = np.loadtxt(f).T

            cl_dict[('gg', z1, z2, sample_a, sample_b)] = C_gg 
            cl_dict['ell'] = ell

            if do_magnification:
                cl_dict[('mm', z1, z2, sample_a, sample_b)] = C_mm
                cl_dict[('mg', z1, z2, sample_a, sample_b)] = C_mg  
                cl_dict[('gm', z1, z2, sample_a, sample_b)] = C_gm


        elif (input_name=='intrinsic_power') and (ctype=='II'):
            ell, C_II, C_GG, C_GI, C_IG = np.loadtxt(f).T

            cl_dict[('II', z1, z2, sample_a, sample_b)] = C_II
            cl_dict['ell'] = ell

            if do_lensing:
                cl_dict[('GG', z1, z2, sample_a, sample_b)] = C_GG 
                cl_dict[('GI', z1, z2, sample_a, sample_b)] = C_GI
                cl_dict[('IG', z1, z2, sample_a, sample_b)] = C_IG 

    return cl_dict


def interpolate_power(k,z,p, chi_of_z_spline):
    #import pdb ; pdb.set_trace()
    if (p>0).all():
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), np.log10(p))
        loglog=True
        mloglog=False

    elif (p<0).all():
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), np.log10(-p))
        mloglog=True
        loglog=True
    else:
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), p)
        loglog=False
        mloglog=False

    return loglog, mloglog, P_interp

def load_power_all(block, input_name, chi_of_z_spline, ell, X, do_lensing, do_magnification):

    Pk_flat_dict = {}

    # get the primary power spectrum and turn it into a spline
    Pk = block[input_name, 'p_k']
    k = block[input_name, 'k_h']
    z = block[input_name, 'z']

    loglog, mloglog, P_interp = interpolate_power(k,z,Pk,chi_of_z_spline)
    #P_interp = interp2d(np.log10(k), chi_of_z_spline(z), Pk) ; loglog=False

    if loglog:
        P_flat = np.array([[10**P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])
        if mloglog:
            P_flat*=-1
    else:
        P_flat = np.array([[P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])

    Pk_flat_dict[input_name] = P_flat

    # also some extra ones if needed
    if do_lensing or do_magnification:
        Pk = block['matter_power_nl', 'p_k']
        k = block['matter_power_nl', 'k_h']
        z = block['matter_power_nl', 'z']

        loglog, mloglog, P_interp = interpolate_power(k,z,Pk,chi_of_z_spline)

        P_flat = np.array([[10**P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])

        Pk_flat_dict['matter_power_nl'] = P_flat

        Pk = block['matter_intrinsic_power', 'p_k']
        k = block['matter_intrinsic_power', 'k_h']
        z = block['matter_intrinsic_power', 'z']

        loglog, mloglog, P_interp = interpolate_power(k,z,Pk,chi_of_z_spline)

        if loglog:
            P_flat = np.array([[10**P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])
        if mloglog:
            P_flat*=-1
        else:
            P_flat = np.array([[P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])

        Pk_flat_dict['matter_intrinsic_power'] = P_flat

    if do_lensing and (input_name=='intrinsic_power'):
        Pk = block['galaxy_intrinsic_power', 'p_k']
        k = block['galaxy_intrinsic_power', 'k_h']
        z = block['galaxy_intrinsic_power', 'z']

        loglog, mloglog, P_interp = interpolate_power(k,z,Pk,chi_of_z_spline)

        if loglog:
            P_flat = np.array([[10**P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])
        if mloglog:
            P_flat*=-1
        else:
            P_flat = np.array([[P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])

        Pk_flat_dict['galaxy_intrinsic_power'] = P_flat

    return Pk_flat_dict



def do_limber_integral(ell, P_flat, p1, p2, X):
    outvec = [] 
    
#    for i, l in enumerate(ell):
#        K = 1./X/X*p1*p2*P_flat[i]
#
#        K[np.isinf(K)] = 0.
#        K[np.isnan(K)] = 0.
#
#       # interp = interp1d(X,K)
#
#        I = np.trapz(K, X, axis=0)
#       # I = cumtrapz(K, x=X, initial=0)
#        #I,_ = quad(interp, X[0], X[-1])
#        outvec.append(I)

    K0 = np.array([1./X/X*p1*p2*P_flat[i] for i in range(len(ell))])
    K0[np.isinf(K0)] = 0.
    K0[np.isnan(K0)] = 0.
    outvec = cumtrapz(K0, x=X, initial=0, axis=1)[:,-1]
    # this is about twice as fast as the commented out method above using a loop
    # python... :) 


 #   import pdb ; pdb.set_trace()      

    return np.array(outvec)


def choose_pdf(z, sigma=None, interpolator=None):
    if Sz_interpolator is None:
        return gaussian(z, sigma=sigmaz)
    else:
        Sz = Sz_interpolator(z)
        return gaussian(z, sigma=Sz)



def gaussian(z0,sigma=0.017):
    x = np.linspace(0.0,3,100)
    sigz = sigma * (1+z0)
    return x, np.exp(-(x-z0) * (x-z0) /2 /sigz /sigz)



def get_p_of_chi_spline(block, z, pz):
    # Extract some useful distance splines
    # have to copy these to get into C ordering (because we reverse them)
    z_distance = block[names.distances, 'z']
    a_distance = block[names.distances, 'a']
    chi_distance = block[names.distances, 'd_m']

    h0 = block[names.cosmological_parameters, "h0"]

    interp = interp1d(z_distance, chi_distance)
    # convert Mpc to Mpc/h
    X = interp(z) * h0

    chi_max = X.max()
    #pz/=np.trapz(pz,z)
    #import pdb ; pdb.set_trace()
    pz_of_chi = GSLSpline(X,pz)

    return pz_of_chi


def load_distance_splines(block):
    # Extract some useful distance splines
    # have to copy these to get into C ordering (because we reverse them)
    z_distance = block[names.distances, 'z']
    a_distance = block[names.distances, 'a']
    chi_distance = block[names.distances, 'd_m']
    if z_distance[1] < z_distance[0]:
        z_distance = z_distance[::-1].copy()
        a_distance = a_distance[::-1].copy()
        chi_distance = chi_distance[::-1].copy()

    h0 = block[names.cosmological_parameters, "h0"]

    # convert Mpc to Mpc/h
    chi_distance *= h0

    if block.has_value(names.distances, 'CHISTAR'):
        chi_star = block[names.distances, 'CHISTAR'] * h0
    else:
        chi_star = None

    chi_max = chi_distance.max()
    a_of_chi = GSLSpline(chi_distance, a_distance)
    chi_of_z = GSLSpline(z_distance, chi_distance)

    return a_of_chi, chi_of_z



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

    interp_nz = spi.interp1d(zb, nz_b, fill_value='extrapolate')
    nz_b = interp_nz(z0)
    interp_nz = spi.interp1d(za, nz_a, fill_value='extrapolate')
    nz_a = interp_nz(z0)


    X = nz_a * nz_b/x/x/dxdz
    X[0]=0
    interp_X = spi.interp1d(z0, X, fill_value='extrapolate')

    # Inner integral over redshift
    V,Verr = sint.quad(interp_X, zmin, z0.max())
    W = nz_a*nz_b/x/x/dxdz/V
    W[0]=0

    return z0,W


def coeff(block, sample_a, sample_b,  input_name) :
    if (input_name=='galaxy_intrinsic_power'):
        return block['bias_parameters','b_%s'%sample_a]
    elif (input_name=='galaxy_power'):
        return block['bias_parameters','b_%s'%sample_a]*block['bias_parameters','b_%s'%sample_b]
    elif (input_name=='intrinsic_power'):
        return 1.

