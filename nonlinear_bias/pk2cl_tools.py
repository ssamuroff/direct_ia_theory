#python code for P(k) to C(l) calculations including
#Limber and exact ("non-Limber"). If this is too slow
#should be straightforward to convert to c.
from __future__ import print_function
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import quad
import sys
from LOG_HT import fft_log
import time
from fftlog import Fftlog

inv_sqrt2pi = 1./np.sqrt(2*np.pi)

def nearest_power_of_2(x):
    return 2**int(np.ceil(np.log2(x)))
    #Find nearest, greater power of 2 to x. 
    # return 1<<(int(x)-1).bit_length()

def get_dlogchi(dchi, chimax):
    """
    Since our chi values need to be log-spaced (for the fftlog), this dchi
    will correspond to the last chi increment. Hence our dlogchi is given by
    dlogchi = log(chi_N) - log(chi_{N-1}) = log(chi_N / chi_{N-1})
    We can substitute chi_N = chimax and chi_{N-1} = chi_N - dchi to get
    dlogchi = log(chimax / (chimax - dchi))

    Parameters
    ----------
    dchi: float
        chi spacing 
    chimax: float
        maximum chi 

    Returns
    -------
    dlogchi: float
        corresponding spacing in log(chi)
    """
    dlogchi = np.log(chimax / (chimax-dchi))
    return dlogchi

def exact_integral(ells, kernel1_interp, kernel2_interp,
    pk0_interp_logk, growth_interp, chimin, chimax, dlogchi, 
    do_rsd=False, b1_1=None, b1_2=None, f_interp=None, 
    chi_pad_upper=2., chi_pad_lower=2., 
    verbose=False ):
    """
    Do the exact P(k,chi)->C_l projection integral
    The full integral is 
    \int_0^\inf dk k P(k,0) I_1(k) I_2(k)
    where I_x(k) = \int_0^{\inf} dr W_x(r) r^{-0.5} D(r) J_{l+0.5}(kr),
    and W_x(r_1) is the radial kernel for tracer x and D(r) is the growth factor

    Here we use Joe's fftlog function for the I_x(k) calculation.
    This assumes the form 
    F(k) = \int_0^{\inf} k dr w(r) J_{mu}(kr). 
    So if we set w(r) = F_x(r) r^{-0.5} D(r), we need
    to divide out by a two factors (one for each of 
    I_1 and I_2) of k at the end i.e. F(k) = kI(k)

    So we do the integral as 
    \int_0^\inf k^{-1} dk P(k,0) * (k*I_1(k)) * (k*I_2(k)).
    We actually do it on logk i.e.
    \int_0^\inf dlogk P(k,0) * (k*I_1(k)) * (k*I_2(k)).

    We also optionally do RSD. In this case, I_x(k)=I_x(k,l)
    I_x(k,l) = \int_0^{\inf} k dr f_x(r) * [ J_{mu}(kr)
        + f(r)( L_0 * J_{mu} + L_m2 * J_{mu-2} + L_p2 * J_{mu+2} )]
    See below for definition of L_0s, L_m2s and L_p2s

    Parameters
    ----------
    ells : np.array
        np.array of ell values to compute C(l) for.
    kernel1_interp: spline
        Spline of F_1(chi)
    kernel2_interp: spline
        Spline of F_2(chi)
    pk0_interp_logk: spline
        Spline of P(log(k), z=0)
    growth_interp: spline
        Spline of growth(chi)
    chimin: float
        minimum chi for integral over chi
    chimax: float
        maximum chi for integral over chi
    chi_pad_lower: float
        extend the integral over log(chi) lower limit by
        this factor (maybe required for good fftlog behaviour)
    chi_pad_upper: float
        extend the integral over log(chi) upper limit by
        this factor (maybe required for good fftlog behaviour)
    dlogchi: float
        spacing to use in integral over log(chi)
    do_rsd: bool
        Include RSD
    b1_1: float
        Value of linear bias for sample 1 (required for RSD calculation)
    b1_2: float
        Value of linear bias for sample 2 (required for RSD calculation)
    f_interp: spline
        Spline of growth function dlog(D)/dloga as a function of chi.

    Returns
    -------
    cell: float array
        np.array of C(l) values.
    """

    #Assert that chimin>0 since we're going to do the integral
    #in log(chi)
    assert chimin>0.
    log_chimin, log_chimax = np.log(chimin), np.log(chimax)
    if verbose:
        print("""padding chi values by e^%.2f/%.2f at 
            lower/upper ends"""%(chi_pad_lower,chi_pad_upper))
    log_chimin_padded, log_chimax_padded = log_chimin-chi_pad_lower, log_chimax+chi_pad_upper
    nchi_orig = np.ceil((log_chimax-log_chimin)/dlogchi).astype(int)
    nchi = nearest_power_of_2(nchi_orig) #use nchi that is a power of 2 for fast fft.
    log_chi_vals = np.linspace(log_chimin_padded, log_chimax_padded, nchi)
    chi_vals = np.exp(log_chi_vals)
    if verbose:
        print("chimin padded, chimax padded, nchi padded:", 
            chi_vals[0], chi_vals[-1], len(chi_vals))
    growth_vals = growth_interp(chi_vals)
    kernel1_vals = kernel1_interp(chi_vals)
    auto=False
    if kernel2_interp is kernel1_interp:
        auto = True
        assert b1_1==b1_2

    #Only implemented case where both samples have rsd
    do_rsd_1 = do_rsd_2 = do_rsd
    if do_rsd_1 or do_rsd_2:
        #Get Legendre coefficients
        L_0s = (2*ells*ells+2*ells-1)/(2*ells+3)/(2*ells-1)
        L_m2s = -ells*(ells-1)/(2*ells-1)/(2*ells+1)
        L_p2s = -(ells+1)*(ells+2)/(2*ells+1)/(2*ells+3)
        assert f_interp is not None
        f_vals = f_interp(chi_vals) #dlnD/dlna at each chi value

    w1_vals = kernel1_vals * growth_vals * np.power(chi_vals, -0.5)
    if do_rsd_1:
        w1_rsd_vals = w1_vals * f_vals
        try:
            assert b1_1 is not None
        except AssertionError as e:
            print("do_rsd_1 is true, but b1_1=None.")
            raise(e)

    if not auto:
        kernel2_vals = kernel2_interp(chi_vals)
        w2_vals = kernel2_vals * growth_vals * np.power(chi_vals, -0.5)
        if do_rsd_2:
            w2_rsd_vals = w2_vals * f_vals
            try:
                assert b1_2 is not None
            except AssertionError as e:
                print("do_rsd_2 is true, but b1_2=None")
                raise(e)

    cell = np.zeros_like(ells)

    for i_ell, ell in enumerate(ells):
        k_vals, F_1 = fft_log(chi_vals, w1_vals, 0, ell+0.5)

        #multiply this term by bias 
        if b1_1 is not None:
            F_1 *= b1_1

        #Now rsd part.
        if do_rsd_1:
            k_vals_check, F_1_0 = fft_log(chi_vals, 
                w1_rsd_vals, 0, ell+0.5)
            if ell>1:
                k_vals_check, F_1_m2 = fft_log(chi_vals, 
                    w1_rsd_vals, 0, ell-1.5, kr=ell+1)
            else:
                F_1_m2 = 0.
            assert np.allclose(k_vals_check, k_vals)
            k_vals_check, F_1_p2 = fft_log(chi_vals, 
                w1_rsd_vals, 0, ell+2.5, kr=ell+1)
            assert np.allclose(k_vals_check, k_vals)
            F_1_rsd = L_0s[i_ell]*F_1_0 + L_m2s[i_ell]*F_1_m2 + L_p2s[i_ell]*F_1_p2
            F_1 += F_1_rsd

        if auto:
            F_2 = F_1
        else:
            _, F_2 = fft_log(chi_vals, w2_vals, 0, ell+0.5)

            #multiply normal term by bias
            if b1_2 is not None:
                F_2 *= b1_2
            
            if do_rsd_2:
                #Now rsd part
                k_vals_check, F_2_0 = fft_log(chi_vals, 
                    w2_rsd_vals, 0, ell+0.5)
                if ell>1:
                    k_vals_check, F_2_m2 = fft_log(chi_vals, 
                        w2_rsd_vals, 0, ell-1.5, kr=ell+1)
                else:
                    F_2_m2 = 0.
                k_vals_check, F_2_p2 = fft_log(chi_vals, 
                    w2_rsd_vals, 0, ell+2.5, kr=ell+1)
                F_2_rsd = (L_0s[i_ell]*F_2_0 + L_m2s[i_ell]*F_2_m2 
                    + L_p2s[i_ell]*F_2_p2)
                F_2 += F_2_rsd

        logk_vals = np.log(k_vals)
        pk_vals = pk0_interp_logk(logk_vals)
        #Now we can compute the full integral \int_0^{\inf} k^{-1} dk P(k,0) F_1(k) F_2(k)
        #We are values logspaced in k, so calculate as \int_0^{inf} dlog(k) P(k,0) F_1(k) F_2(k)
        integrand_vals = pk_vals * F_1 * F_2
        #Spline and integrate the integrand.
        integrand_interp = IUS(logk_vals, integrand_vals)
        integral = integrand_interp.integral(logk_vals.min(), 
            logk_vals.max())
        cell[i_ell] = integral
    return cell

def exact_integral_fftlogxiao(ells, kernel1_interp, kernel2_interp,
    pk0_interp_logk, growth_interp, chimin, chimax, dlogchi, 
    do_rsd=False, b1_1=None, b1_2=None, f_interp=None, 
    chi_pad_upper=1., chi_pad_lower=1., chi_extrap_upper=1.,
    chi_extrap_lower=1., verbose=True):
    """
    Do the exact P(k,chi)->C_l projection integral
    The full integral is 
    2/pi \int_0^\inf dk k^2 P(k,0) I_1(k) I_2(k)
    where I_x(k,r) = \int_0^{\inf} dr W_x(r) D(r) j_{l}(kr),
    and W_x(r) is the radial kernel for tracer x 
    and D(r) is the growth factor

    We want to use an fftlog for the I_x(k) calculation. Xiao's
    code does F(k) = \int_0^\infty dr / r * q(r) * j_\ell(kr)
    so in our case q(r)/r = W_x(r) D(r) so q(r) = r W_x(r) D(r).

    We actually do the integral in log(k), so calculate 
    \sqrt(2/pi) \int_0^\inf k^3 d(logk) P(k,0) I_1(k) I_2(k).

    We also optionally do RSD. In this case, 
    I_x(k)=I_x(k,l) = \int_0^\infty dr/r (q(r)j_l(kr) -
    n(r)f(r)D(r)j_l''(kr))
    where f(r) is [dlnD/dlna](r) the logarithmic growth rate.

    Parameters
    ----------
    ells : np.array
        np.array of ell values to compute C(l) for.
    kernel1_interp: spline
        Spline of F_1(chi)
    kernel2_interp: spline
        Spline of F_2(chi)
    pk0_interp_logk: spline
        Spline of P(log(k), z=0)
    growth_interp: spline
        Spline of growth(chi)
    chimin: float
        minimum chi for integral over chi
    chimax: float
        maximum chi for integral over chi
    chi_pad_lower: float
        Pad f(chi) with zeros up to this factor
        times chimax
    chi_pad_upper: float
        Pad f(chi) with zeros down to this factor
        times chimax
    chi_extrap_upper: float
    chi_extrap_lower: float
    dlogchi: float
        spacing to use in integral over log(chi)
    do_rsd: boolN_
        Include RSD
    b1_1: float
        Value of linear bias for sample 1 (required for RSD calculation)
    b1_2: float
        Value of linear bias for sample 2 (required for RSD calculation)
    f_interp: spline
        Spline of growth function dlog(D)/dloga as a function of chi.

    Returns
    -------
    cell: float array
        np.array of C(l) values.
    """
    chi_pad_upper=chi_pad_lower=1.
    #chi_extrap_upper=chi_extrap_lower=2.
    assert chimin>0.
    log_chimin, log_chimax = np.log(chimin), np.log(chimax)
    nchi = np.ceil((log_chimax-log_chimin)/dlogchi).astype(int)
    #nchi = nearest_power_of_2(nchi_orig) #use nchi that is a power of 2 for fast fft.
    log_chi_vals = np.linspace(log_chimin, log_chimax, nchi)

    if chi_pad_upper>0.:
        assert chi_pad_lower==chi_pad_upper
        N_pad = (np.ceil(float(chi_pad_upper)/dlogchi)).astype(int)
    else:
        N_pad = 0
    if chi_extrap_upper>0.:
        N_extrap_upper = (np.ceil(float(chi_extrap_upper)/dlogchi)).astype(int)
    else:
        N_extrap_upper = 0
    if chi_extrap_lower>0.:
        N_extrap_lower = (np.ceil(float(chi_extrap_lower)/dlogchi)).astype(int)
    else:
        N_extrap_lower = 0    
    print("chmin, chimax:", chimin, chimax)
    print("nchi:", nchi, N_pad)
    pad_and_extrap_kwargs = {"N_pad": N_pad, "N_extrap_low": N_extrap_lower,
                             "N_extrap_high": N_extrap_upper}
    print("pad and extrap kwargs:", pad_and_extrap_kwargs)

    chi_vals = np.exp(log_chi_vals)
    growth_vals = growth_interp(chi_vals)
    growth_vals = growth_interp(chi_vals)
    kernel1_vals = kernel1_interp(chi_vals)
    auto=False
    if kernel2_interp is kernel1_interp:
        auto = True
        assert b1_1==b1_2

    #Only implemented case where both samples have rsd
    do_rsd_1 = do_rsd_2 = do_rsd
    if do_rsd_1 or do_rsd_2:
        assert f_interp is not None
        f_vals = f_interp(chi_vals) #dlnD/dlna at each chi value

    w1_vals = kernel1_vals * growth_vals * chi_vals
    if do_rsd_1:
        w1_rsd_vals = w1_vals * f_vals

    if not auto:
        kernel2_vals = kernel2_interp(chi_vals)
        w2_vals = kernel2_vals * growth_vals * chi_vals
        if do_rsd_2:
            w2_rsd_vals = w2_vals * f_vals

    #output array for C(l)s
    cell = np.zeros_like(ells)
    #Set up fftlog instances
    fftlog_1 = Fftlog(chi_vals, w1_vals, **pad_and_extrap_kwargs)
    if do_rsd_1:
        fftlog_rsd_1 = Fftlog(chi_vals, w1_rsd_vals, nu=1.1, **pad_and_extrap_kwargs)

    if not auto:
        fftlog_2 = Fftlog(chi_vals, w2_vals, **pad_and_extrap_kwargs)
        if do_rsd_2:
            fftlog_rsd_2 = Fftlog(chi_vals, w2_rsd_vals, nu=1.1, **pad_and_extrap_kwargs)

    for i_ell, ell in enumerate(ells):
        k_vals, I_1 = fftlog_1.fftlog(ell)
        #multiply this term by bias 
        if b1_1 is not None:
            I_1 *= b1_1

        #Now rsd part.
        if do_rsd:
            k_vals_check, I_1_rsd = fftlog_rsd_1.fftlog_ddj(
                ell)
            assert np.allclose(k_vals_check, k_vals)
            I_1 -= I_1_rsd

        if auto:
            I_2 = I_1
        else:
            k_vals, I_2 = fftlog_2.fftlog(ell)

            #multiply normal term by bias
            if b1_2 is not None:
                I_2 *= b1_2
            
            if do_rsd:
                #Now rsd part
                k_vals_check, I_2_rsd = fftlog_rsd_2.fftlog_ddj(
                    ell)
                assert np.allclose(k_vals_check, k_vals)
                I_2 -= I_2_rsd

        logk_vals = np.log(k_vals)
        pk_vals = pk0_interp_logk(logk_vals)
        #Now we can compute the full integral 
        #2/pi * \int_0^{\inf} k^2 dk P(k,0) I_1(k) I_2(k)
        #We are values logspaced in k, so calculate as 
        #2/pi * \int_0^{inf} k^3 dlog(k) P(k,0) I_1(k) I_2(k)
        integrand_vals =k_vals * k_vals * k_vals * pk_vals * I_1 * I_2
        #Spline and integrate the integrand.
        integrand_interp = IUS(logk_vals, integrand_vals)
        integral = integrand_interp.integral(logk_vals.min(), 
            logk_vals.max())
        integral *= 2./np.pi
        cell[i_ell] = integral
    return cell

def limber_integral(ells, kernel1, kernel2, pk_interp_logk, chimin, chimax, dchi,
    method="trapz", verbose=False):
    """
    Do the Limber integral 
    C(l) = \int dchi K_1(chi) K_2(chi) P((ell+0.5)/chi, chi) / chi^2
    Can do this via two methods, 'spline' or 'trapz'. 'trapz' should 
    be faster but a little less accurate for a given chimin, chimax, dchi.

    Parameters
    ----------
    ells : np.array
        np.array of ell values to compute C(l) for.
    kernel1_interp: spline
        Spline of F_1(chi)
    kernel2_interp: spline
        Spline of F_2(chi)
    pk_interp_logk: scipy.interpolate.RectBivariateSpline instance
        Spline of P(log(k), chi)
    chimin: float
        minimum chi for integral over chi
    chimax: float
        maximum chi for integral over chi
    dchi: float
        chi spacing for integral over chi

    Returns
    -------
    c_ells: float array
        np.array of C(l) values.
    c_ell_errs: float array
        np.array of error values, nans if method=="trapz"
    """
    if verbose:
        print("""Doing Limber integral with method %s between 
            chi_min: %.2e and chi_max: %.2e with step size %.2e"""%(method, chimin, chimax, dchi))
    try:
        assert chimin>=0.
    except AssertionError as e:
        print("found chimin = %f"%chimin)
        raise(e)

    #Initialize c_ell and error arrays.
    c_ells, c_ell_errs = np.zeros_like(ells), np.nan * np.ones_like(ells)

    #Get chi values and evaluate kernels
    chi_vals = np.arange(chimin, chimax+dchi, dchi)
    kernel1_vals = kernel1(chi_vals)
    kernel2_vals = kernel2(chi_vals)
    k1k2 = kernel1_vals * kernel2_vals

    #Go ahead and do integral
    if method == "trapz":
        #Trapz method uses the trapezium rule to do all
        #ell simulatenously
        K_VALS = (ells[:, np.newaxis]+0.5) / chi_vals
        CHI_VALS = (chi_vals[:, np.newaxis]).T * np.ones((ells.shape[0], chi_vals.shape[0]))
        K1K2 = (k1k2[:, np.newaxis]).T * np.ones_like(CHI_VALS)
        PK_VALS = pk_interp_logk(CHI_VALS, np.log(K_VALS), grid=False)
        #compute integral via trapezium rule
        integrands = K1K2 * PK_VALS / CHI_VALS / CHI_VALS
        DCHI = CHI_VALS[:,1:] - CHI_VALS[:,:-1]
        integrands = DCHI * 0.5 * (integrands[:,1:] + integrands[:,:-1])
        c_ells = np.sum(integrands, axis=1)
        c_ell_errs = np.nan * np.ones_like(c_ells)

    elif method == "spline":
        #Spline method loops through ells, splining the integrand 
        #and then integrating, so is probably more accurate but definitely
        #slower than trapz method with all else equal.
        for i,ell in enumerate(ells):
            nu = ell+0.5
            k_vals = nu / chi_vals
            pk_vals = pk_interp_logk(chi_vals, np.log(k_vals), grid=False)
            integrand = k1k2 * pk_vals / chi_vals / chi_vals
            int_spline = IUS(chi_vals, integrand)
            c_ells[i], c_ell_errs[i] = int_spline.integral(chimin, chimax), np.nan
    return c_ells, c_ell_errs
