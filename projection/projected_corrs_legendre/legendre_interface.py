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
from astropy.cosmology import FlatLambdaCDM

y3fid_cosmology = FlatLambdaCDM(H0=69., Om0=0.30, Ob0=0.048)

def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="lens lens").split()
    sample_b = options.get_string(option_section, "sample_b", default="lens source").split()
    rmin = options.get_double(option_section, "rpmin", default=0.01)
    rmax = options.get_double(option_section, "rpmax", default=500.)
    nr = options.get_int(option_section, "nr", default=1024)
    nk = options.get_int(option_section, "nk", default=200)

    rp = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    pimax = options.get_double(option_section, "pimax", default=100.) # in h^-1 Mpc

    corrs = options.get_string(option_section, "correlations", default="wgp").split()

    do_rsd = options.get_bool(option_section, "include_rsd", default=False)
    do_lensing = options.get_bool(option_section, "include_lensing", default=False)
    do_magnification = options.get_bool(option_section, "include_magnification", default=False)

    cl_dir = options.get_string(option_section, "cl_loc", default="")

    if do_rsd:
        print('will include RSDs (Pi_max = %3.1f)'%pimax)
    else:
        print('will not include RSDs :( ')
        print("redshift space will not be distorted, and it's your fault...")

 
    return sample_a, sample_b, rp, pimax, nk, corrs, do_rsd, do_lensing, do_magnification, cl_dir



def execute(block, config):
    sample_a,sample_b,rp,pimax,nk,corrs,do_rsd, do_lensing, do_magnification, cl_dir = config

    k = block['galaxy_power', 'k_h']
    print(k.min(),k.max())
    knew = np.logspace(np.log10(0.001), np.log10(k.max()), nk)
    X = Projected_Corr_RSD(rp=rp, pi_max=pimax, k=knew, lowring=True)

    # bookkeeping
    pknames = {'wgg':'galaxy_power', 'wgp':'galaxy_intrinsic_power'}

    if do_rsd:
        fz = block['growth_parameters', 'f_z']
        z1 = block['growth_parameters', 'z']
        #interp = interp1d(z,f0)
        #fz = interp(0.27)
        beta2 = -1

        Dz=block['growth_parameters', 'd_z']/block['growth_parameters', 'd_z'][0]
        lnD=np.log(Dz)
        lna=np.log(block['growth_parameters', 'a'])
        fz = np.gradient(lnD,lna)
    else:
        fz = 0.
        beta2 = 0.
        z1 = block['growth_parameters', 'z']

    print (corrs)

    for c,s1,s2 in zip(corrs,sample_a,sample_b):
        print(c,s1,s2)

        if ('bias_parameters','b_%s'%s1) in block.keys():
            ba = block['bias_parameters', 'b_%s'%s1]
        else:
            ba = 1.

        if ('bias_parameters','b_%s'%s2) in block.keys():
            bb = block['bias_parameters', 'b_%s'%s2]
        else:
            bb = 1.

        P = block[pknames[c],'p_k']
        z = block[pknames[c],'z']
        #import pdb ; pdb.set_trace()

        if (P>0).all():
            inter = interp2d(np.log10(k), z, np.log10(P))
            #import pdb ; pdb.set_trace()
            Pnew = 10**inter(np.log10(knew), z1)
        else:
            #import pdb ; pdb.set_trace()
            inter = interp2d(np.log10(k), z, np.log10(-P))
            Pnew = -10**inter(np.log10(knew), z1)


        if (c=='wgg'):
            za, W = get_redshift_kernel(block, 0, 0, z1, block['distances','d_m'], s1, s2)
            #import pdb ; pdb.set_trace()
            #Wofz = interp1d(W,za)
            K = np.array([W]*len(X.rp))

            z0 = np.trapz(za*W,za)

            if do_magnification:
                Pnew = add_gg_mag_terms(block, Pnew, za, knew, z0, s1, s2, cl_dir=cl_dir)
            bb = ba
           # print('BIAS : %f %f'%(ba,bb))
            W = X.wgg_calc(f=fz, bg=ba, bg2=bb, pk=Pnew, xi=None, l=[0,2,4]) #  * ba * bb
            #import pdb ; pdb.set_trace()
            

        elif (c=='wgp'):
            za, W = get_redshift_kernel(block, 0, 0, z1, block['distances','d_m'], s1, s2)
           # Wofz = interp1d(W,za)
            K = np.array([W]*len(X.rp))

            z0 = np.trapz(za*W,za)
            #import pdb ; pdb.set_trace()

            W = X.wgm_calc(f=fz, bg=ba, beta2=beta2, pk=-Pnew, xi=None, l=[0,2,4]) 


            #W*=np.sqrt(2.) 
                      

    #    import pdb ; pdb.set_trace()
        
        integrand = K.T * W
        W_flat = sint.trapz(integrand,z1,axis=0) / sint.trapz(K.T,z1,axis=0)
        

        section = pknames[c].replace('_power','_w')
        block.put_double_array_1d(section, 'w_rp_1_1_%s_%s'%(s1,s2), W_flat)
        try:
        	block.put_double_array_1d(section, 'r_p', X.rp)
        except:
        	block.replace_double_array_1d(section, 'r_p', X.rp) 

        #if (c=='wgp'):
        #    import pdb ; pdb.set_trace()


    return 0




def add_gg_mag_terms(block, Pnew, z, k, z0, s1,s2, cl_dir=""):

    h0 = y3fid_cosmology.h


    if (len(cl_dir)==0):
        c_mm = block['magnification_cl', 'bin_1_1']
        c_gm = block['magnification_galaxy_cl', 'bin_1_1']
        c_mI = block['magnification_intrinsic_cl', 'bin_1_1']
        ell = block['magnification_cl', 'ell']
    else:
        dsample1 = s1.replace('_density', '')
        dsample2 = s2.replace('_density', '')


        c_mm = np.loadtxt("%s/magnification_cl_%s_%s.txt"%(cl_dir, dsample1, dsample2))
        c_mg = np.loadtxt("%s/magnification_galaxy_cl_%s_%s.txt"%(cl_dir, dsample1, dsample2))

        ell = np.loadtxt("%s/ell.txt"%cl_dir)

    p_mm = c_mm * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2 * h0 * h0 * h0 #this is an approximation
    k_mm = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_mg = c_mg * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_mg = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0



    p_mm_int = interp1d(np.log10(k_mm), p_mm, bounds_error=False, fill_value=0)
    p_mm = p_mm_int(np.log10(k))

    p_mg_int = interp1d(np.log10(k_mg), p_mg, bounds_error=False, fill_value=0)
    p_mg = p_mg_int(np.log10(k))


    Pmm = np.array([p_mm]*len(z))
    Pmg = np.array([p_mg]*len(z))

    print('Adding magnification....')


    return Pnew + Pmm + 2*Pmg



def add_gp_lensmag_terms(block, Pnew, z, k, z0, s1, s2, cl_dir=None, do_lensing=True, do_magnification=True):

    # lensing contribution to g+
    # this is a back-of-the-envelope thing Sukhdeep came up with
    # assuming the contribution is small 

    h0 = y3fid_cosmology.h


    if (len(cl_dir)==0):
        c_mm = block['magnification_cl', 'bin_1_1']
        c_gm = block['magnification_galaxy_cl', 'bin_1_1']
        c_mI = block['magnification_intrinsic_cl', 'bin_1_1']
        ell = block['magnification_cl', 'ell']
    else:
        dsample = s1.replace('_density', '')
        ssample = s2.replace('_shape', '')


        c_mI = np.loadtxt("%s/magnification_intrinsic_cl_%s_%s.txt"%(cl_dir, dsample, ssample))
        c_mG = np.loadtxt("%s/magnification_shear_cl_%s_%s.txt"%(cl_dir, dsample, ssample))
        c_gG = np.loadtxt("%s/galaxy_shear_cl_%s_%s.txt"%(cl_dir, dsample, ssample))

        ell = np.loadtxt("%s/ell.txt"%cl_dir)

    p_mI = c_mI * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2 * h0 * h0 * h0 #this is an approximation
    k_mI = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_mG = c_mG * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_mG = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0

    p_gG = c_gG * y3fid_cosmology.comoving_transverse_distance(z0).value**3 / 2* h0 * h0 * h0 #this is an approximation
    k_gG = ell / y3fid_cosmology.comoving_transverse_distance(z0).value/ h0



    p_mI_int = interp1d(np.log10(k_mI), p_mI, bounds_error=False, fill_value=0)
    p_mI = p_mI_int(np.log10(k))

    p_mG_int = interp1d(np.log10(k_mG), p_mG, bounds_error=False, fill_value=0)
    p_mG = p_mG_int(np.log10(k))

    p_gG_int = interp1d(np.log10(k_gG), p_gG, bounds_error=False, fill_value=0)
    p_gG = p_gG_int(np.log10(k))

    PmI = np.array([p_mI]*len(z))
    PmG = np.array([p_mG]*len(z))
    PgG = np.array([p_gG]*len(z))

    #import pdb ; pdb.set_trace()

    if do_lensing and do_magnification:
        return Pnew + PmI + PmG + PgG
    elif do_lensing and (not do_magnification):
        return Pnew + PgG
    else:
        return Pnew + PmI + PmG

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

    #import pdb ; pdb.set_trace()

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


class Projected_Corr_RSD():
    def __init__(self,rp=None,pi=None,pi_max=100,l=[0,2,4],k=None, lowring=True):
        self.rp=rp
        self.pi=pi
        if rp is None:
            self.rp=np.logspace(-1,np.log10(200),60)
        if pi is None:
            self.pi=np.logspace(-3,np.log10(pi_max),125)
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
        
    def alpha(self, l, beta1, beta2):
        if l==0:
            return 1 + 1./3.*(beta1+beta2) + 1./5*(beta1*beta2)
        elif l==2:
            return (2./3.*(beta1+beta2) + 4./7.*(beta1*beta2))
        elif l==4:
            return 8./35.*(beta1*beta2)

    def w_to_DS(self,rp=[],w=[]):
        DS0=2*w[0]*rp[0]**2
        return 2.*integrate.cumtrapz(w*rp,x=rp,initial=0)/rp**2-w+DS0/rp**2

    def get_xi(self, pk=[], l=[0,2,4]):
        xi={}
        for i in l:
            #
            ri, xi_i = self.j[i](pk, extrap=True)
            xi_intp=interp1d(ri,xi_i,bounds_error=False,fill_value=0)
            xi[i] = np.dot((xi_intp(self.rG)*self.L[i]),self.dpi)
            #if (self.nu==2):
            #    xi[i]*=(-1)**(i/2)
            xi[i]*=2#one sided pi
            #import pdb ; pdb.set_trace()
        return xi

    def wgg_calc(self,f=0,bg=0,bg2=None,pk=[],xi=None,l=[0,2,4]):
        bg1=bg
        if bg2 is None:
            bg2=bg
        beta1=f/bg1
        beta2=f/bg2
        if xi is None:
            xi=self.get_xi(pk=pk,l=l)
        W = np.zeros_like(xi[xi.keys()[0]])

       # import pdb ; pdb.set_trace()
        
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

        #import pdb ; pdb.set_trace()
        if do_DS:
            W=self.w_to_DS(rp=self.rp,w=W)
        return W
