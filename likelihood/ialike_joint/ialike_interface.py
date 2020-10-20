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

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def setup(options):


    apply_hartlap = options.get_bool(option_section, "apply_hartlap", default=True)

    filename = options.get_string(option_section, "datafile")
    corrs_to_use = options[option_section, "ctypes"].split()
    samples_to_use = options[option_section, "samples"].split()
    indices = options[option_section, "indices"]

    cov_slices = options[option_section, "covariance_blocks"].split()


    indices = indices.split(',')
    indices = [int(i) for i in indices]


    # convert samples to a list of tuples
    samples_to_use = [( str(s.split(',')[0].replace('(','')), str(s.split(',')[1].replace(')','')) ) for s in samples_to_use] 

    # Unpack the 2pt data into one continuous array
    data = []
    R = []
    data_all = fi.FITS(filename)
    done2=[]

    

    
    for corr in corrs_to_use:
        dvec = data_all[corr].read()
        for i in indices:

            #import pdb ; pdb.set_trace()
            done = []
            npt = 0

            mask = (dvec['BIN']==i)

            data.append(dvec['VALUE'][mask])
            R.append(dvec['SEP'][mask])
            npt+=len(dvec['SEP'][mask])

            print("%s, index %d: found %d points"%(corr, i, npt))
            done2.append(corr)

    mask1d = parse_cuts(indices, corrs_to_use, R, options)

    data = np.concatenate(data)
    #import pdb ; pdb.set_trace()
    print('Flattened data vector contains %d points'%len(data))
    print('or %d after scale cuts'%len(data[mask1d]))

    # Read and invert the covariance matrix
    # we need to apply scale cuts at this point
    # if we don't want to have to invert the covariance matrix
    # at every step in parameter space
    cov0 = fi.FITS(filename)['COVMAT'].read()

    COV = np.zeros((len(R)*len(R[0]), len(R)*len(R[0]))) + 1e-8

    #k0 = 0 
    nr = len(R[0])

    for k0,thing in enumerate(cov_slices):
        a0,b0=thing.replace('(','').replace(')','').split(',')
        a0 = int(a0)
        b0 = int(b0)
        COV[k0*nr:(k0+1)*nr,k0*nr:(k0+1)*nr] = cov0[a0:b0,a0:b0]

      #  k0+=1


    cov = np.array([row[mask1d] for row in COV])
    cov = np.array([col[mask1d] for col in cov.T])

    #import pdb ; pdb.set_trace()

    invcov = np.linalg.inv(cov)

    # Apply a correction for bias due to the fact that the
    # covariance matrix contains noise
    # See e.g. Friedrich et al 2015 arXiv:1508.00895
    if apply_hartlap:
        hdr = fi.FITS(filename)['COVMAT'].read_header()
        print('Applying Hartlap correction for noise-induced bias.')
        N = hdr['NREAL'] * 1.0
        d = len(cov[0]) * 1.0
        alpha = (N-1)/(N-d-2)
        print('alpha = %3.3f'%alpha)
        invcov = invcov / alpha
        #import pdb ; pdb.set_trace()

    return indices, corrs_to_use, samples_to_use, R, data, invcov, mask1d

def parse_cuts(redshifts, corrs_to_use, R, options):
    rmin = np.atleast_1d(options[option_section, 'rmin'])
    rmax = np.atleast_1d(options[option_section, 'rmax'])

    #corrs_to_use*=len(redshifts)

    mask1d = []
    count = 0
    
    for corr in corrs_to_use:
        for z0 in redshifts:
            # Now assess which of the bins passes the chosen set of scale cuts
            rlower,rupper = rmin[count], rmax[count]

            scale_window = (R[count]>rlower) & (R[count]<rupper)
            
            mask1d.append(scale_window)
            count+=1
            print(rlower,rupper,corr,z0,len(scale_window[scale_window]))
    #import pdb ; pdb.set_trace()

    # Flatten into 1D
    mask1d = np.concatenate(mask1d)

    return mask1d

def execute(block, config):
    redshifts, corrs_to_use, samples_to_use, R, data, invcov, mask1d = config


    # Now construct the theory data vector
    y = []
    count=0
    
    
    for corr, (s1,s2) in zip(corrs_to_use,samples_to_use):
        for z0 in redshifts:

            #import pdb ; pdb.set_trace()
            section = block_names[corr]

            Y = block[section, 'w_rp_%d'%(z0)]

            xf = block[section, 'r_p']
            if np.all(Y>0):
                interpolator = spi.interp1d(np.log10(xf), np.log10(Y))
                ylog = True
                y_resampled = 10**(interpolator(np.log10(R[count])))
            else:
                try:
                    interpolator = spi.interp1d(np.log10(xf), Y)
                except: 
                    import pdb ; pdb.set_trace()
                ylog = False
                try:
                    y_resampled = interpolator(np.log10(R[count]))
                except:
                    import pdb ; pdb.set_trace()

            y.append(y_resampled)
            count+=1

    y = np.concatenate(y)
   # import pdb ; pdb.set_trace()

    # Evaluate the likelihood and save it 
    res = (y-data)
    res = res[mask1d]
    chi2 = np.dot( res, np.dot(invcov, res) )
    chi2 = float(chi2)
    like = -0.5*chi2

 #   import pdb ; pdb.set_trace()


    block[names.data_vector, 'iacorr'+"_CHI2"] = chi2
    block[names.likelihoods, 'iacorr'+"_LIKE"] = like
    #import pdb ; pdb.set_trace()
    
    return 0



