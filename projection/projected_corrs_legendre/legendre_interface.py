from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
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

def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="lens lens").split()
    sample_b = options.get_string(option_section, "sample_b", default="lens source").split()
    rmin = options.get_double(option_section, "rpmin", default=0.0001)
    rmax = options.get_double(option_section, "rpmax", default=300.)
    nr = options.get_int(option_section, "nr", default=2000)
    nk = options.get_int(option_section, "nk", default=2000)

    rp = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    pimax = options.get_double(option_section, "pimax", default=100.) # in h^-1 Mpc

    corrs = options.get_string(option_section, "correlations", default="wgp").split()

    do_rsd = options.get_bool(option_section, "include_rsd", default=False)

    if do_rsd:
        print('will include RSDs (Pi_max = %3.1f)'%pimax)

    return sample_a, sample_b, rp, pimax, nk, corrs, do_rsd

def execute(block, config):
    sample_a,sample_b,rp,pimax,nk,corrs,do_rsd = config

    k = block['galaxy_power', 'k_h']
    knew = np.logspace(np.log10(k.min()), np.log10(k.max()), nk)
    X = Projected_Corr_RSD(rp=rp, pi_max=pimax, k=knew, lowring=True)

    # bookkeeping
    pknames = {'wgg':'galaxy_power', 'wgp':'galaxy_intrinsic_power'}

    if do_rsd:
        fz = block['growth_parameters', 'f_z']
        z = block['growth_parameters', 'z']
        #interp = interp1d(z,f0)
        #fz = interp(0.27)
        beta2 = -1
    else:
        fz = 0.
        beta2 = 0.
        z = block['growth_parameters', 'z']

    print (corrs)

    for c,s1,s2 in zip(corrs,sample_a,sample_b):
        if ('bias_parameters','b_%s'%s1) in block.keys():
            ba = block['bias_parameters', 'b_%s'%s1]
        else:
            ba = 1.

        if ('bias_parameters','b_%s'%s2) in block.keys():
            bb = block['bias_parameters', 'b_%s'%s2]
        else:
            bb = 1.

        P = block[pknames[c],'p_k']
        if (P>0).all():
            inter = interp2d(np.log10(k), z, np.log10(P))
            Pnew = 10**inter(np.log10(knew), z)
        else:
            inter = interp2d(np.log10(k), z, P)
            Pnew = inter(np.log10(knew), z)

        #import pdb ; pdb.set_trace()
        if (c=='wgg'):
            W = X.wgg_calc(f=fz, bg=ba, bg2=bb, pk=Pnew, xi=None, l=[0,2,4]) #  * ba * bb
            za, K = get_redshift_kernel(block, 0, 0, X.rp, z, s1, s2)
        elif (c=='wgp'):
            W = X.wgm_calc(f=fz, bg=ba, beta2=-1., pk=-Pnew, xi=None, l=[0,2,4])  * ba
            za, K = get_redshift_kernel(block, 0, 0, X.rp, z, s1, s2)
        
        integrand = K.T * W
        W_flat = sint.trapz(integrand,z,axis=0) / sint.trapz(K.T,z,axis=0)
        #import pdb ; pdb.set_trace()

        section = pknames[c].replace('_power','_w')
        block.put_double_array_1d(section, 'w_rp_1_1_%s_%s'%(s1,s2), W_flat)
        try:
        	block.put_double_array_1d(section, 'r_p', X.rp)
        except:
        	pass 


    return 0


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
    interp_X = spi.interp1d(z0, X)

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
            xi[i] = np.dot((xi_intp(self.rG)*self.L[i]),self.dpi)
            #if (self.nu==2):
            #    xi[i]*=(-1)**(i/2)
            xi[i]*=2#one sided pi
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
