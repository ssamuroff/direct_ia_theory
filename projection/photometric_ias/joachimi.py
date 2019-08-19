from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import limber
from gsl_wrappers import GSLSpline, NullSplineError
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
import pylab as plt
plt.switch_backend('agg')


# An implementation of the theory prediction for direct IA measurements with photometric samples
# see Joachimi et al 2010 for details https://arxiv.org/abs/1008.3491
# The module has two modes: the first does the Limber integral to give C(ell|z1,z2), using an
# estimate for the per-galaxy redshift PDFs
# The second mode reads back Hankel transformed C(ells) and integrates over redshift and 
# line-of-sight separation to give the projected correlation function w(r_p) 



clight = 299792.4580 # kms^-1
relative_tolerance = 1e-6
absolute_tolerance = 0.

def setup(options):
    # sample info
    sample_a = options.get_string(option_section, "sample_a", default="lens")
    error_model_a = options.get_string(option_section, "error_model_a", default="delta")
    sample_b = options.get_string(option_section, "sample_b", default="lens")
    error_model_b = options.get_string(option_section, "error_model_b", default="delta")

    # binning
    pimax = options.get_double(option_section, "pimax", default=100.) # in h^-1 Mpc

    # order of the Bessel function used in the Hankel transform
    nu = options.get_int(option_section, "bessel_type", default=2)

    # where to read from and save to
    input_name = options.get_string(option_section, "input_name", default="galaxy_intrinsic_power")
    output_name = options.get_string(option_section, "output_name", default="galaxy_intrinsic_w")

    models = ['gaussian', 'delta']

    mode = options.get_string(option_section, "mode", default="limber")

    if (error_model_a.lower() not in models) or (error_model_b.lower() not in models):
        raise ValueError('One or more of the photo-z error models is not recognised.')

    return sample_a, error_model_a, sample_b, error_model_b, pimax, nu, input_name, output_name, mode

def execute(block, config):
    sample_a, error_model_a, sample_b, error_model_b, pimax, nu, input_name, output_name, mode = config


    H0 = block['cosmological_parameters', 'h0']*100
    h0 = block['cosmological_parameters', 'h0']
    omega_m = block['cosmological_parameters', 'omega_m']
    omega_de = block['cosmological_parameters', 'omega_lambda']   

    # choose a set of bins for line-of-sight separation 
    npi = 30
    nzm = 500
    Pi = np.hstack((np.linspace(-pimax,0,npi), np.linspace(0,pimax,npi)[1:] ))# h^-1 Mpc
    npi = len(Pi)
    Zm = np.linspace(0.05,1.4,nzm)
    
    if mode.lower()=='limber':
        # initialise some splines
        a_of_chi, chi_of_z = load_distance_splines(block)
        P,D = limber.load_power_growth_chi(block, chi_of_z, input_name, "k_h", "z", "p_k")
        
        ell = np.logspace(np.log10(0.1), np.log10(16000), 90000)
        nell = len(ell)
        C_ell_pi_zm = np.zeros((nzm,npi,nell))-9999.

        for i, zm in enumerate(Zm):
            for j,p in enumerate(Pi):
                # coordinate transform
                Hz = 100 * np.sqrt(omega_m*(1+zm)**3 + omega_de) # no h because Pi is in units h^-1 Mpc
                z1 = zm - (0.5/clight * Hz * p)
                z2 = zm + (0.5/clight * Hz * p)

                # evaluate the per-galaxy PDFs at z1 and z2
                x1,pz1 = gaussian(z1,sigma=0.01)
                pz1 /= pz1.max() #np.trapz(pz1,x1)

                x2,pz2 = gaussian(z2,sigma=0.01)
                pz2 /= pz2.max() #np.trapz(pz2,x2)


                if (z1<0) or (z2<0):
                    c_ell = np.zeros_like(ell) 
                    zneg = True
                else:
                    # turn them into a spline
                    pz1_chi_spline = get_p_of_chi_spline(block,x1,pz1)
                    pz2_chi_spline = get_p_of_chi_spline(block,x2,pz2)
                    zneg = False
                

                if zneg:
                    print('Unphysical redshift - skipping')
                # in the case where at least one of the samples is spectroscopic, 
                # the integral simplifies to an analytic expression
                elif (error_model_a=='delta') & (error_model_b=='delta'):
                    if abs(z1-z2)<0.001:
                        K = chi_of_z(z1)**-2
                        #import pdb ; pdb.set_trace()
                        c_ell = np.concatenate([K*P(chi_of_z(z1), np.log(l/chi_of_z(z1))) for l in ell])
                       # import pdb ; pdb.set_trace()
                        
                    else:
                        c_ell = np.zeros_like(ell)

                elif (error_model_a=='delta'):
                    K = pz2_chi_spline(chi_of_z(z1))*chi_of_z(z1)**-2
                    c_ell = np.concatenate([K*P(chi_of_z(z1), np.log(l/chi_of_z(z1))) for l in ell])

                elif (error_model_b=='delta'):
                    K = pz2_chi_spline(chi_of_z(z2))*chi_of_z(z2)**-2
                    c_ell = np.concatenate([K*P(chi_of_z(z2), np.log(l/chi_of_z(z2))) for l in ell])

                # otherwise do the Limber integral
                else:
                    c_ell = limber.limber(pz1_chi_spline, pz2_chi_spline, P, D, ell.astype(float), 1., rel_tol=relative_tolerance, abs_tol=absolute_tolerance )

                C_ell_pi_zm[i,j,:] = c_ell

                #if abs(np.sum(c_ell))>0:
                #    import pdb ; pdb.set_trace()

                block.put_double_array_1d(output_name, 'c_ell_%s_%s_%d_%d'%(sample_a,sample_b,i,j), c_ell) 
                try:
                    block.put_double_array_1d(output_name, 'ell', ell) 
                except:
                    pass

        block.put_int(output_name, 'npi', npi)
        block.put_int(output_name, 'nzm', nzm)
       # import pdb ; pdb.set_trace()

    elif mode.lower()=='collect':

        theta = block[output_name,'theta']
        nr = 200
        a_of_chi, chi_of_z = load_distance_splines(block)
        rp = np.logspace(np.log10(0.1), np.log10(2000.), nr)
        xi_rp_pi_zm = np.zeros((nzm,npi,nr))
        logrp = np.log10(rp)


        for i, zm in enumerate(Zm):
            for j,p in enumerate(Pi):
                #import pdb ; pdb.set_trace()
                try:
                    xi = block[output_name,'w_rp_%d_%d_%s_%s'%(i,j,sample_a,sample_b)]
                except:
                    import pdb ; pdb.set_trace()

                # translate theta into rp at given zm
                # see Joachimi et al 2011 eq A10
                rp_theta = chi_of_z(zm) * theta * h0

                # now interpolate so xi is on a consistent rp grid
                interp = interp1d(np.log10(rp_theta), xi, fill_value='extrapolate', bounds_error=False)
                xi_rp_pi_zm[i,j,:] = interp(logrp)

        xi_rp_pi_zm[np.isnan(xi_rp_pi_zm)] = 0.


        x0, W = get_redshift_kernel(block, 0, 0, rp, Zm, sample_a, sample_b)
        W[np.invert(np.isfinite(W))] = 1e-30

        #integrate over Pi
        xi = np.trapz(xi_rp_pi_zm, Pi, axis=1)

        # and over redshift
        integrand = W.T * xi
        w = np.trapz(integrand,Zm,axis=0) / np.trapz(W.T,Zm,axis=0)

        import pdb ; pdb.set_trace()

        block.put_double_array_1d(output_name, 'w_rp_%s_%s'%(sample_a,sample_b), w)

        try:
            block.put_double_array_1d(output_name, 'r_p', rp)
        except:
            pass

        

        





#    # integrate over Pi
#    C = np.trapz(C_ell_pi_zm, Pi, axis=1)
#
#    # and over redshift
#    integrand = W.T * C
#    Pw = sint.trapz(integrand,Zm,axis=0) / sint.trapz(W.T,Zm,axis=0)
#
#    import pdb ; pdb.set_trace()
#
#    block.put_double_array_1d(output_name, 'w_rp_1_1_%s_%s'%(sample_a,sample_b), Pw) # except it isn't, quite
#    try:
#        block.put_double_array_1d(output_name, 'r_p', X.rp)
#    except:
#        pass 

    return 0

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



def get_redshift_kernel(block, i, j, rp, z0, sample_a, sample_b):

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
    W2d,_ = np.meshgrid(W,rp)
    W2d[np.invert(np.isfinite(W2d))] = 1e-30

    return za,W2d


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
