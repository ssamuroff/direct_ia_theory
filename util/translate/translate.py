from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi
import pylab as plt
plt.switch_backend('pdf')
plt.style.use('y1a1')


def setup(options):

    index = options.get_int(option_section, "index")
    redshift = options.get_double(option_section, "redshift")

    return index, redshift

def execute(block, config):
	index,redshift = config
	#import pdb ; pdb.set_trace()
	print('comverting z=%f --> i=%d'%(redshift,index))
	gg = block['galaxy_w','w_rp_limber_%1.3f'%redshift]
	block.put_double_array_1d('galaxy_w','w_rp_%d'%index,gg)

	gp = block['galaxy_intrinsic_w','w_rp_limber_%1.3f'%redshift]
	block.put_double_array_1d('galaxy_intrinsic_w','w_rp_%d'%index,gp)

	pp = block['intrinsic_w','w_rp_limber_%1.3f'%redshift]
	block.put_double_array_1d('intrinsic_w','w_rp_%d'%index,pp)
	return 0