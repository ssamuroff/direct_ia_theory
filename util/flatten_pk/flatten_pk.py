from __future__ import print_function
import limber_fn as limber
from gsl_wrap import GSLSpline
from builtins import range
from cosmosis.datablock import names, option_section
#from gsl_wrappers import GSLSpline, NullSplineError, GSLSpline2d, BICUBIC
import sys
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as sint
import scipy.special as sps


kernel_dict = {'galaxy_intrinsic_power': ('N','N'),
               'galaxy_intrinsic_power_1h': ('N','N'),
               'magnification_intrinsic_power': ('W', 'N'), 
               'magnification_shear_power' : ('W','W'),
               'matter_galaxy_power': ('N', 'W'),
               'magnification_power' : ('W','W'),
               'galaxy_power' : ('N','N'),
               'nlgal_power' : ('N','N'),
               'magnification_galaxy_power' : ('W', 'N'),
               'intrinsic_power' : ('N', 'N'),
               'shear_power' : ('W', 'W'),
               'matter_intrinsic_power' : ('W', 'N')}

def setup(options):

    power_spectrum_name = options.get_string(option_section, "pk_name")
    sample_a = options.get_string(option_section, "sample_a", default="source")
    sample_b = options.get_string(option_section, "sample_b", default="lens")
    window_function = options.get_bool(option_section, "window_function", default=False)
    if not window_function:
    	redshift = options[option_section, "redshift"]
    	if isinstance(redshift, float):
    		redshift = [redshift]
    else:
    	redshift = None
    add_bias = options.get_bool(option_section, "add_bias", default=False)
    add_rsd = options.get_bool(option_section, "add_rsd", default=False)
    add_ia = options.get_bool(option_section, "add_intrinsic_alignments", default=False)

    return power_spectrum_name, redshift, add_bias, add_ia, add_rsd, window_function, sample_a, sample_b

def apply_ia(block, pkname, redshift, pk, add_ia):
	if (not add_ia) or (pkname=='galaxy_power'):
		return pk

	zname = '%3.3f'%(redshift)
	zname = zname.replace('.','_')
	A = block['intrinsic_alignment_parameters', 'a_%s'%zname]

	#print('Applying bg=%3.3f to %s'%(bg, pkname))

	if (pkname=='intrinsic_power') or (pkname=='intrinsic_power_bb'):
		return A * A * pk
	elif ('galaxy_intrinsic_power' in pkname) or ('matter_intrinsic_power' in pkname):
		return A * pk
	else:
		raise ValueError('Unknown power spectrum type: %s'%pkname)

def apply_bias(block, pkname, sample_a, sample_b, redshift, pk, add_bias):
	if (not add_bias) or (pkname=='intrinsic_power') or  (pkname=='intrinsic_power_bb'):
		return pk

	if not (redshift==None):
		zname = '%3.3f'%(redshift)
		zname = zname.replace('.','_')
		b1 = block['bias_parameters', 'b_%s'%zname]
		# only look for the second bias coefficient if we actually need it
		if (pkname=='galaxy_power'):
			b2 = block['bias_parameters', 'b_%s'%zname]

	else:
		b1 = block['bias_parameters', 'b_%s'%sample_a]
		if (pkname=='galaxy_power'):
			b2 = block['bias_parameters', 'b_%s'%sample_b]

	#print('Applying bg=%3.3f to %s'%(bg, pkname))

	if (pkname=='galaxy_power'):
		return b1 * b2 * pk
	elif ('galaxy_intrinsic_power' in pkname):
		return b1 * pk
	else:
		raise ValueError('Unknown power spectrum type: %s'%pkname)

def magnification_prefactor(block, n):
	alpha = block['galaxy_luminosity_function', 'alpha_binned']
	K = 2 * (alpha - 1)**n
	return K 


def get_pk(block, power_spectrum_name):
	if (power_spectrum_name, 'p_k') in block.keys():
		name = power_spectrum_name
		K = 1.

	elif (power_spectrum_name=='magnification_power'):
		name = 'matter_power_nl'
		K = magnification_prefactor(block, 2)

	elif (power_spectrum_name=='magnification_intrinsic_power'):
		name = 'matter_intrinsic_power'
		K = magnification_prefactor(block, 1)

	elif (power_spectrum_name=='magnification_shear_power'):
		name = 'matter_power_nl'
		K = magnification_prefactor(block, 1)

	elif (power_spectrum_name=='magnification_galaxy_power'):
		name = 'matter_galaxy_power'
		K = magnification_prefactor(block, 1)

	elif (power_spectrum_name=='matter_galaxy_power'):
		name = 'matter_galaxy_power'
		K = 1.
	elif (power_spectrum_name=='galaxy_intrinsic_power'):
		name = 'galaxy_intrinsic_power'
		K = 1.

	elif (power_spectrum_name=='shear_power'):
		name = 'matter_galaxy_power'
		K = 1.

	elif (power_spectrum_name=='matter_intrinsic_power'):
		name = 'matter_intrinsic_power'
		K = 1.

	elif (power_spectrum_name=='intrinsic_power'):
		name = 'shear_power_ii'
		K = 1.

	else:
		print('Unrecognised power spectrum: %s'%power_spectrum_name)
		import pdb ; pdb.set_trace()

	z,k,P =block.get_grid(name, 'z', 'k_h', 'p_k')
	P*=K

	return z, k, P



def execute(block, config):
	power_spectrum_name, redshift, add_bias, add_ia, add_rsd, window_function, sample_a, sample_b = config

	z,k,pk = get_pk(block, power_spectrum_name) 
	#block.get_grid(power_spectrum_name, 'z', 'k_h', 'p_k')
	#

	if len(pk[pk>0])==len(pk):
		logint = True
		lnpk = np.log(pk)
	else:
		print('Negative Pk values - will interpolate in linear space.')
		logint = False
		lnpk = pk

	interp = spi.interp2d(np.log(k), z, lnpk, kind='linear')

	# Simplest case: take the 2D grid P(k,z) and interpolate to the desired redshift
	# Effectively assuming the redshift distribution is a delta fn.
	# Store the linearised power spectrum to the block with the suffix identifying z
	# Only do this if window_function=F
	redshift = np.atleast_1d(redshift)
	if (not window_function):
		for z0 in redshift:
			pk_interpolated = interp(np.log(k), [z0])
			if logint:
				pk_interpolated = np.exp(pk_interpolated)

			pk_interpolated = apply_ia(block, power_spectrum_name, z0, pk_interpolated, add_ia)
			pk_interpolated = apply_bias(block, power_spectrum_name, None, None, z0, pk_interpolated, add_bias)

			try:
				block.put_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'p_k', pk_interpolated)
				block.put_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'k_h', k)

			except:
				block.replace_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'p_k', pk_interpolated)
				block.replace_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'k_h', k)
			#import pdb ; pdb.set_trace()

	# Slightly more complicated case: project P(k,z) along the redshift axis,
	# with a window function defined by a specific pair of n(z)
	# This is a bit more fiddly as it involves looping over bin pairs.
	else:

		# we'll need chi(z) and dchi/dz
		# (at the same z sampling as the n(z) )
		chi = block['distances','d_m']
		z = block['distances','z']
		interp_chi = spi.interp1d(z,chi)
		dz = z[1]-z[0]
		Dchi = np.gradient(chi,dz)
		interp_dchi = spi.interp1d(z,Dchi)

		# Now do the kernel calculation for each bin pair in turn
		# see, for example, eq 8 from Singh et al arXiv:1411.1755
		na = block['nz_%s'%sample_a, 'nbin']
		nb = block['nz_%s'%sample_b, 'nbin']
		zmin = 0.01

		# Do this only once per power spectrum
		output_section_name = str(power_spectrum_name+'_projected')
		try:
			block.put_double_array_1d(output_section_name, 'k_h', k)
			block.put_int(output_section_name, 'nbin_a', na)
			block.put_int(output_section_name, 'nbin_b', nb)
		except:
			pass

		kernel1, kernel2 = kernel_dict[power_spectrum_name]

		#Now loop over bin pairs
		for i in range(na):
			for j in range(nb):

				za, nz_a = choose_kernel(block, kernel1, sample_a, i)
				zb, nz_b = choose_kernel(block, kernel2, sample_b, j)

				x = interp_chi(za)
				dxdz = interp_dchi(za)

				#import pdb ; pdb.set_trace()

				X = nz_a * nz_b/x/x/dxdz
				interp_X = spi.interp1d(za, X)

				# Inner integral over redshift
				V,Verr = sint.quad(interp_X, zmin, za.max())
				W = nz_a*nz_b/x/x/dxdz/V

				# Now do the power spectrum integration
				W2d,_ = np.meshgrid(W,k)
				W2d[np.invert(np.isfinite(W2d))] = 1e-30

				pk_interpolated = interp(np.log(k), za)
				if logint:
					pk_interpolated = np.exp(pk_interpolated)

				pk_interpolated = apply_bias(block, power_spectrum_name, sample_a, sample_b, None, pk_interpolated, add_bias)
				#pk_interpolated = apply_rsd(block, power_spectrum_name, za, pk_interpolated, add_rsd)

				# Outer integral with kernel calculated above
				# nb the normalisation matters!!!
				integrand = W2d.T * pk_interpolated
				Pw = sint.trapz(integrand,za,axis=0) / sint.trapz(W2d.T,za,axis=0)

				block.put_double_array_1d(output_section_name, 'p_k_%d_%d_%s_%s'%(i+1,j+1,sample_a,sample_b), Pw)
        #import pdb ; pdb.set_trace()
	return 0




def choose_kernel(block, kernel_type, sample, i):


	nz = block['nz_%s'%sample, 'bin_%d'%(i+1)]
	z = block['nz_%s'%sample, 'z']

	if (kernel_type=='N'):
		return z, nz
	elif (kernel_type=='W'):

		chi_distance = block['distances', 'd_m']
		a_distance = block['distances', 'a']
		z_distance = block['distances', 'z']

		if (z_distance[1] < z_distance[0]):
			z_distance = z_distance[::-1].copy()
			a_distance = a_distance[::-1].copy()
			chi_distance = chi_distance[::-1].copy()

		h0 = block[names.cosmological_parameters, "h0"]
		# convert Mpc to Mpc/h
		chi_distance *= h0
		chi_max = chi_distance.max()

		a_of_chi = GSLSpline(chi_distance, a_distance)
		chi_of_z = GSLSpline(z_distance, chi_distance)

		W = limber.get_named_w_spline(block, 'nz_%s'%sample, i+1, z, chi_max, a_of_chi)
		X = chi_of_z(z)

		return z, W(X)/np.trapz(W(X),z)



#
#	if len(za)!=len(zb):
#		interp_nz = spi.interp1d(zb, nz_b)
#		nz_b = interp_nz(za)
					

