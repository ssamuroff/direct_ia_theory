from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as sint
import scipy.special as sps

def setup(options):

    power_spectrum_name = options.get_string(option_section, "pk_name")
    sample_a = options.get_string(option_section, "sample_a", default="source")
    sample_b = options.get_string(option_section, "sample_b", default="lens")
    window_function = options.get_bool(option_section, "window_function", default=False)

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


def execute(block, config):
	power_spectrum_name, redshift, add_bias, add_ia, add_rsd, window_function, sample_a, sample_b = config

	z,k,pk = block.get_grid(power_spectrum_name, 'z', 'k_h', 'p_k')

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

			block.put_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'p_k', pk_interpolated)
			block.put_double_array_1d(power_spectrum_name+'_%2.3f'%z0, 'k_h', k)
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

		#Now loop over bin pairs
		for i in range(na):
			nz_a = block['nz_%s'%sample_a, 'bin_%d'%(i+1)]
			za = block['nz_%s'%sample_a, 'z']
			for j in range(nb):
				nz_b = block['nz_%s'%sample_b, 'bin_%d'%(j+1)]
				zb = block['nz_%s'%sample_b, 'z']

				if len(za)!=len(zb):
					interp_nz = spi.interp1d(zb, nz_b)
					nz_b = interp_nz(za)
					#raise ValueError('Redshift sampling does not match!')

				x = interp_chi(za)
				dxdz = interp_dchi(za)

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

	return 0





