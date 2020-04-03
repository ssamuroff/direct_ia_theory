from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as sint
import scipy.special as sps
import pylab as plt
plt.switch_backend('agg')
plt.style.use('y1a1')

def setup(options):
	sample_a = options.get_string(option_section, 'sample_a', default="")
	sample_b = options.get_string(option_section, 'sample_b', default="")
	corr_type = options.get_string(option_section, 'corr_type', default="gp")

	return sample_a, sample_b, corr_type
def execute(block, config):
	sample_a,sample_b,corr_type = config

	if corr_type=='gp':
		PgI = block['galaxy_intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		k = block['galaxy_intrinsic_power_projected', 'k_h']

		# additional lensing & magnification terms
		PmI = block['magnification_intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		PgG = block['matter_galaxy_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		PmG = block['magnification_shear_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]

		P0 = PgI + PgG + PmI + PmG 

		block['galaxy_intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)] =  P0
		
		#plot_gp_terms(k, PgI, PmI, PgG, PmG)

	if corr_type=='gg':
		Pgg = block['galaxy_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		k = block['galaxy_power_projected', 'k_h']

		# additional magnification terms
		Pmm = block['magnification_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		Pgm = block['magnification_galaxy_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]

		P0 = Pgg + Pmm + 2*Pgm 

		block['galaxy_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)] =  P0

	if corr_type=='pp':
		PII = block['intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		k = block['intrinsic_power_projected', 'k_h']

		# additional lensing terms
		PIG = block['matter_intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]
		PGG = block['shear_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)]

		P0 = PII + PGG + 2*PIG 

		block['intrinsic_power_projected', 'p_k_1_1_%s_%s'%(sample_a,sample_b)] =  P0

	return 0

def plot_gp_terms(k, PgI, PmI, PgG, PmG):
	plt.plot(k,-PgI, color='darkmagenta', lw=1.5, label='$-gI$')
	plt.plot(k,-PmI, color='royalblue', lw=1.5, label='$-mI$')
	plt.plot(k,PgG, color='plum', lw=1.5, label='$gG$')
	plt.plot(k,PmG, color='hotpink', lw=1.5, label='$mG$')
	plt.xlabel('$k$ / $h$ Mpc$^{-1}$', fontsize=16)
	plt.ylabel('$P(k)$', fontsize=16)
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.subplots_adjust(bottom=0.15,left=0.15)
	plt.savefig('/home/ssamurof/tmp/pk_theory_terms.png')
	plt.close()
