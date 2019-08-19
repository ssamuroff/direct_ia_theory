from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi


# see tables 1 & 2 in arXiv:0903.3870
#giparams = {'q11' : 0.01867, 'q12' : 6.924, 'q13' : 0.3725, 
#            'q21' : 1.989, 'q22' : 1.081, 'q23' : 0.6816, 
#            'q31' : 4.232, 'q32' : 0.1748, 'q33' : 0.481}


giparams = {'q11' : 0.005, 'q12' : 5.909, 'q13' : 0.3798, 
            'q21' : 0.6, 'q22' : 1.087, 'q23' : 0.6655, 
            'q31' : 3.1, 'q32' : 0.1912, 'q33' : 0.4368}

iiparams = {'q11' : 0.09939, 'q12' : 3.718, 'q13' : 0.3475, 
            'q21' : 1.931, 'q22' : 1.061, 'q23' : 0.7484, 
            'q31' : 6.082, 'q32' : 0.1045, 'q33' : 0.613}

def setup(options):
	do_II = options.get_bool(option_section, "do_II", default=True)
	do_GI = options.get_bool(option_section, "do_GI", default=True)
	return do_II, do_GI


def execute(block, config):

	do_II, do_GI = config
	
	z,k,pk = block.get_grid('matter_power_nl', 'z', 'k_h', 'p_k')

	z2d, k2d = np.meshgrid(z, k)

	#ah = 0.084 * 2 * np.pi #0.21
	ah = block['intrinsic_alignment_parameters', 'a_halo_model']

	# First do the 1h GI term
	p1 = giparams['q11'] * np.exp(giparams['q12'] * z2d**giparams['q13'])
	p2 = giparams['q21'] * np.exp(giparams['q22'] * z2d**giparams['q23'])
	p3 = giparams['q31'] * np.exp(giparams['q32'] * z2d**giparams['q33'])
	P_GI = -ah * (k2d/p1) * (k2d/p1) / (1 + (k2d/p2)**p3)

	# And II
	p1 = iiparams['q11'] * np.exp(iiparams['q12'] * z2d**iiparams['q13'])
	p2 = iiparams['q21'] * np.exp(iiparams['q22'] * z2d**iiparams['q23'])
	p3 = iiparams['q31'] * np.exp(iiparams['q32'] * z2d**iiparams['q33'])
	P_II = ah * ah * (k2d/p1) * (k2d/p1) * (k2d/p1) * (k2d/p1) / (1 + (k2d/p2)**p3)

	block.put_grid('galaxy_intrinsic_power_1h', 'k_h', k, 'z', z, 'p_k', P_GI)
	block.put_grid('matter_intrinsic_power_1h', 'k_h', k, 'z', z, 'p_k', P_GI)
	block.put_grid('intrinsic_power_1h', 'k_h', k, 'z', z, 'p_k', P_II)


	return 0





