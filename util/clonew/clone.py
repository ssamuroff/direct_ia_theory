from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi


def setup(options):

    source_a = options.get_string(option_section, "source_a")
    source_b = options.get_string(option_section, "source_b")

    sample_a = options.get_string(option_section, "sample_a")
    sample_b = options.get_string(option_section, "sample_b")

    output_name = options.get_string(option_section, "output_name")

    return source_a, source_b, sample_a, sample_b, output_name


def execute(block, config):

    source_a, source_b, sample_a, sample_b, output_name = config

    source_label = 'w_rp_1_1_%s_%s'%(source_a,source_b)
    out_label = 'w_rp_1_1_%s_%s'%(sample_a,sample_b)

    w0 = block[output_name, source_label]
    block.put_double_array_1d(output_name, out_label, w0)

    return 0
