from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import limber_lib as limber
from gsl_wrappers_local import GSLSpline, GSLSpline2d, NullSplineError, BICUBIC, BILINEAR
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
import pyccl as ccl
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')


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

    ell_max = options.get_double(option_section, "ell_max", default=4e6)
    nell = options.get_double(option_section, "nell", default=200)

    # binning
    pimax = options.get_double(option_section, "pimax", default=100.) # in h^-1 Mpc

    # order of the Bessel function used in the Hankel transform
    nu = options.get_int(option_section, "bessel_type", default=2)

    # where to read from and save to
    input_name = options.get_string(option_section, "input_name", default="galaxy_intrinsic_power")
    output_name = options.get_string(option_section, "output_name", default="galaxy_intrinsic_w")


    return sample_a, sigma_a, sample_b, sigma_b, pimax, nu, input_name, output_name, ell_max, nell

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

def execute(block, config):
    sample_a, sigma_a, sample_b, sigma_b, pimax, nu, input_name, output_name, ell_max, nell = config


    H0 = block['cosmological_parameters', 'h0']*100
    h0 = block['cosmological_parameters', 'h0']
    omega_m = block['cosmological_parameters', 'omega_m']
    omega_b = block['cosmological_parameters', 'omega_b']
    omega_de = block['cosmological_parameters', 'omega_lambda']
    sigma_8 = 0.8234379064365687 # this is hard coded for now...
    ns = block['cosmological_parameters', 'n_s']

    cosmology = ccl.Cosmology(Omega_c=omega_m-omega_b, Omega_b=omega_b, h=h0, sigma8=sigma_8, n_s=ns, matter_power_spectrum='halofit', transfer_function='boltzmann_camb')

    # choose a set of bins for line-of-sight separation 
    npi = 20
    nzm = 40
    Pi = np.hstack((np.linspace(-600,0,npi), np.linspace(0,600,npi)[1:] ))# h^-1 Mpc
    npi = len(Pi)
    Zm = np.linspace(0.05,1.15,nzm)

    # get the power spectrum and turn it into a spline
    Pk = block[input_name, 'p_k']
    k = block[input_name, 'k_h']
    z = block[input_name, 'z']

    z_distance = block['distances', 'z']
    chi_distance = block['distances', 'd_m']
    a_distance = 1./(1+z_distance)
    chi_of_z_spline = interp1d(z_distance, chi_distance)

    #P_interp = GSLSpline2d(chi_of_z_spline(z), np.log10(k), Pk, spline_type=BICUBIC)

    if (Pk>0).all():
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), np.log10(Pk))
        loglog=True
        mloglog=False
    elif (Pk<0).all():
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), np.log10(-Pk))
        mloglog=True
        loglog=True
    else:
        P_interp = interp2d(np.log10(k), chi_of_z_spline(z), Pk)
        loglog=False
        mloglog=False


    # first bit: Limber integrals

    ell = np.logspace(0,np.log10(ell_max),nell)
    cl_vec = np.zeros((nzm, npi, len(ell))) - 9999.

    print('initialising arrays...')
    x1 = np.linspace(0,3,600)
    X = chi_of_z_spline(x1)

    if loglog:
        P_flat = np.array([[10**P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])
        if mloglog:
            P_flat*=-1
    else:
        P_flat = np.array([[P_interp(np.log10(l/x), x)[0] for x in X] for l in ell])

    print('Starting loop')

    for i, zm in enumerate(Zm):
        for j,p in enumerate(Pi):
            # coordinate transform
            Hz = 100 * np.sqrt(omega_m*(1+zm)**3 + omega_de) # no h because Pi is in units h^-1 Mpc
            z1 = zm - (0.5/clight * Hz * p)
            z2 = zm + (0.5/clight * Hz * p)

            # evaluate the per-galaxy PDFs at z1 and z2
            x1,pz1 = gaussian(z1, sigma=sigma_a)
            pz1 /=np.trapz(pz1,X) #pz1.max() 

            x2,pz2 = gaussian(z2, sigma=sigma_b)
            pz2 /= np.trapz(pz2,X)

            Cell = do_limber_integral(ell, P_flat, pz1, pz2, X)
            cl_vec[i,j,:] = Cell

            #print(i,j)


    # Next do the Hankel transform
    xi_vec = np.zeros_like(cl_vec)
    rp_vec = np.logspace(np.log10(0.1), np.log10(200), xi_vec.shape[2])
    theta = 2*np.pi/np.flipud(ell) * (180/np.pi)

    print('Hankel transform...')

    for i, zm in enumerate(Zm):
        x0 =  chi_of_z_spline(zm)
        # do the coordinate transform to convert theta to rp at given redshift
        theta_radians = rp_vec/x0
        theta_degrees = theta_radians * 180./np.pi

        for j,p in enumerate(Pi):
            # select a Cell, at fixed Pi, zm, and Hankel transform it
            C = cl_vec[i,j,:]
            if (nu==0):
                xi = ccl.correlation(cosmology, ell, C, theta_degrees, corr_type='GL', method='FFTLog')
            elif (nu==1):
                xi = ccl.correlation(cosmology, ell, C, theta_degrees, corr_type='gg', method='FFTLog')
            elif (nu==2):
                xi_0 = ccl.correlation(cosmology, ell, C, theta_degrees, corr_type='L+', method='FFTLog')
                xi_4 = ccl.correlation(cosmology, ell, C, theta_degrees, corr_type='L-', method='FFTLog')
                xi = xi_0 + xi_4

            xi_vec[i,j,:] = xi

    rp_vec*=h0

  #  import pdb ; pdb.set_trace()

    xi_vec[np.isnan(xi_vec)]=0.
    xi_vec[np.isinf(xi_vec)]=0.

    # integrate over line of sight separation
    xi_pi_rp = np.trapz(xi_vec, Pi, axis=1)

    # and then over redshift
    za, W = get_redshift_kernel(block, 0, 0, Zm, sample_a, sample_b)
    K = np.array([W]*len(rp_vec)).T
    w_rp = np.trapz(xi_pi_rp*K, Zm, axis=0)/np.trapz(K, Zm, axis=0)

    #bg = block['bias_parameters','b_%s'%sample_a]
    #w_rp*=bg       

    block.put_double_array_1d(output_name, 'w_rp_1_1_%s_%s'%(sample_a,sample_b), w_rp)
    block[output_name, 'r_p'] = rp_vec

    return 0


def do_limber_integral(ell, P_flat, p1, p2, X):
    outvec = [] 
    
    for i, l in enumerate(ell):
        K = 1./X/X*p1*p2*P_flat[i]
        K[np.isinf(K)] = 0.
        K[np.isnan(K)] = 0.
        #print(K)
        I = np.trapz(K, X, axis=0)
        outvec.append(I)
        

    return np.array(outvec)



def gaussian(z0,sigma=0.017):
    x = np.linspace(0,3,600)
    sigz = sigma / (1+z0)
    return x, np.exp(-(x-z0) * (x-z0) /2 /sigma /sigma)



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



def get_redshift_kernel(block, i, j, z0, sample_a, sample_b):

    chi = block['distances','d_m']
    z = block['distances','z']
    interp_chi = spi.interp1d(z,chi)
    dz = z[1]-z[0]
    Dchi = np.gradient(chi,dz)
    interp_dchi = spi.interp1d(z,Dchi)

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

    x = interp_chi(z0)
    dxdz = interp_dchi(z0)

    X = nz_a * nz_b/x/x/dxdz
    interp_X = spi.interp1d(z0, X, fill_value='extrapolate')

    # Inner integral over redshift
    V,Verr = sint.quad(interp_X, zmin, z0.max())
    W = nz_a*nz_b/x/x/dxdz/V

    # Now do the power spectrum integration
    #W2d,_ = np.meshgrid(W,rp)
    #W2d[np.invert(np.isfinite(W2d))] = 1e-30

    return za,W


class Projected_Corr_RSD():
    def __init__(self,rp=None,pi=None,pi_max=100,l=[0,2,4],k=None, lowring=True):
        self.rp=rp
        self.pi=pi
        if rp is None:
            self.rp=np.logspace(-1,np.log10(200),100)
        if pi is None:
            self.pi=np.logspace(-3,np.log10(pi_max),250)
#            self.pi=np.append(0,self.pi)
        self.dpi=np.gradient(self.pi)
        self.piG,self.rpG=np.meshgrid(self.pi,self.rp)
        self.rG=np.sqrt(self.rpG**2+self.piG**2)
        self.muG=self.piG/self.rG
        self.L={}
        self.j={}
        for i in l:
            self.L[i]=legendre(i,self.muG)
            self.j[i]=P2xi(k,l=i, lowring=lowring)
        
    def alpha(self,l,beta1,beta2):
        if l==0:
            return 1+1./3.*(beta1+beta2)+1./5*(beta1*beta2)
        elif l==2:
            return (2./3.*(beta1+beta2)+4./7.*(beta1*beta2))
        elif l==4:
            return 8./35.*(beta1*beta2)

    def w_to_DS(self,rp=[],w=[]):
        DS0=2*w[0]*rp[0]**2
        return 2.*integrate.cumtrapz(w*rp,x=rp,initial=0)/rp**2-w+DS0/rp**2

    def get_xi(self, pk=[], l=[0,2,4]):
        xi={}
        for i in l:
            ri, xi_i = self.j[i](pk, extrap=True)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            xi[i] = xi_intp(self.rG) * self.L[i]
            #np.dot((xi_intp(self.rG)*self.L[i]),self.dpi)
            #if (self.nu==2):
            #    xi[i]*=(-1)**(i/2)
            #xi[i]*=2#one sided pi
        return xi

    def wgg_calc(self,f=0,bg=0,bg2=None,pk=[],xi=None,l=[0,2,4]):
        bg1=bg
        if bg2 is None:
            bg2=bg
        beta1=f/bg1
        beta2=f/bg2
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[xi.keys()[0]])
        
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg1*bg2).T

        return W

    def wgm_calc(self,f=0,bg=0,beta2=0,pk=[],xi=None,l=[0,2,4],do_DS=True):
        beta1=f/bg
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W=np.zeros_like(xi[xi.keys()[0]])
        y=[]
        for i in l:
            W+=(xi[i].T*self.alpha(i,beta1,beta2)*bg).T

      #  import pdb ; pdb.set_trace()
        if do_DS:
            W=self.w_to_DS(rp=self.rp,w=W)
        return W
