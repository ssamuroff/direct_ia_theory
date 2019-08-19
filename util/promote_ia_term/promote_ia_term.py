from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import numpy as np
import scipy.interpolate as spi

def setup(options):

    power_spectrum_name = options.get_string(option_section, "pk_name")
    section_name = options.get_string(option_section, "section", default="fastpt")
    output_name = options.get_string(option_section, "save_name", default="")

    return power_spectrum_name, section_name, output_name


def execute(block, config):
	power_spectrum_name, section_name, output_name = config

	z,k,pk = block.get_grid(section_name, 'z', 'k_h', power_spectrum_name)
	
	block[output_name, 'p_k'] = pk
	block[output_name, 'k_h'] = k
	block[output_name, 'z'] = z

	print('Saved %s as %s'%(power_spectrum_name,output_name))

	#import pdb ; pdb.set_trace()

	return 0





