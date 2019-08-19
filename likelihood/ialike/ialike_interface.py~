from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def setup(options):

    use_limber = options.get_bool(option_section, "limber", default=True)
    apply_hartlap = options.get_bool(option_section, "apply_hartlap", default=True)

    filename = options.get_string(option_section, "datafile")
    corrs_to_use = options[option_section, "ctypes"].split()
    samples_to_use = options[option_section, "samples"].split()
    redshifts = options[option_section, "redshifts"]

    if isinstance(redshifts, float):
        redshifts = np.atleast_1d(redshifts)

    if isinstance(redshifts, unicode):
        redshifts = list(np.atleast_1d(redshifts))
        for i,z in enumerate(redshifts):
            a,b = z.split(',')
            redshifts[i] = (int(a),int(b))

    # convert samples to a list of tuples
    samples_to_use = [( str(s.split(',')[0].replace('(','')), str(s.split(',')[1].replace(')','')) ) for s in samples_to_use] 

    # Unpack the 2pt data into one continuous array
    data = []
    R = []
    data_all = fi.FITS(filename)
    done2=[]

    for i,z0 in enumerate(redshifts):
        for corr in corrs_to_use:
            if (corr in done2): #and (isinstance(redshifts,tuple)):
                continue
 
            dvec = data_all[corr].read()
            if 'SAMPLE1' in dvec.dtype.names:
                sample1 = dvec['SAMPLE1']
                sample2 = dvec['SAMPLE2']
            else:
                sample1 = [0]
                sample2 = [0]


            done = []
            npt = 0
            for s1,s2 in zip(sample1,sample2):
                if ('%d%d'%(s1,s2) in done):
                    continue

                if 'BIN' in dvec.dtype.names:
                    mask = (dvec['BIN']==i)
                else:
                    mask = (dvec['BIN1']==z0[0]) & (dvec['BIN2']==z0[1]) & (dvec['SAMPLE1']==s1) & (dvec['SAMPLE2']==s2) 

                data.append(dvec['VALUE'][mask])
                R.append(dvec['SEP'][mask])
                done.append('%d%d'%(s1,s2))
                npt+=len(dvec['SEP'][mask])


            print("%s, redshift %d: found %d points"%(corr, i, npt))
            done2.append(corr)

    mask1d = parse_cuts(redshifts, corrs_to_use, R, options)

    data = np.concatenate(data)
    #import pdb ; pdb.set_trace()
    print('Flattened data vector contains %d points'%len(data))
    print('or %d after scale cuts'%len(data[mask1d]))

    # Read and invert the covariance matrix
    # we need to apply scale cuts at this point
    # if we don't want to have to invert the covariance matrix
    # at every step in parameter space
    cov0 = fi.FITS(filename)['COVMAT'].read()
    #import pdb ; pdb.set_trace()
    cov = np.array([row[mask1d] for row in cov0])
    cov = np.array([col[mask1d] for col in cov.T])

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

    return redshifts, corrs_to_use, samples_to_use, R, data, invcov, mask1d, use_limber

def parse_cuts(redshifts, corrs_to_use, R, options):
    rmin = np.atleast_1d(options[option_section, 'rmin'])
    rmax = np.atleast_1d(options[option_section, 'rmax'])

    #corrs_to_use*=len(redshifts)

    mask1d = []
    count = 0
    for z0 in redshifts:
        for corr in corrs_to_use:
            # Now assess which of the bins passes the chosen set of scale cuts
            #import pdb ; pdb.set_trace()
            rlower,rupper = rmin[count], rmax[count]
            scale_window = (R[count]>rlower) & (R[count]<rupper)
            mask1d.append(scale_window)
            count+=1

    # Flatten into 1D
    mask1d = np.concatenate(mask1d)

    return mask1d

def execute(block, config):
    redshifts, corrs_to_use, samples_to_use, R, data, invcov, mask1d, use_limber = config

    if use_limber:
        suffix = '_limber'
    else:
        suffix = ''

    # Now construct the theory data vector
    y = []
    count=0
    for z0 in redshifts:
        for corr, (s1,s2) in zip(corrs_to_use,samples_to_use):
            section = block_names[corr]

            if isinstance(z0,tuple):
                Y = block[section, 'w_rp_%d_%d_%s_%s'%(z0[0],z0[1],s1,s2)]
            else:
                Y = block[section, 'w_rp%s_%3.3f'%(suffix,z0)]

            xf = block[section, 'r_p']
            if np.all(Y>0):
                interpolator = spi.interp1d(np.log10(xf), np.log10(Y))
                ylog = True
                y_resampled = 10**(interpolator(np.log10(R[count])))
            else:
                interpolator = spi.interp1d(np.log10(xf), Y)
                ylog = False
                y_resampled = interpolator(np.log10(R[count]))

            y.append(y_resampled)
            count+=1

    y = np.concatenate(y)

    # Evaluate the likelihood and save it 
    res = (y-data)
    res = res[mask1d]
    chi2 = np.dot( res, np.dot(invcov, res) )
    chi2 = float(chi2)
    like = -0.5*chi2

    #import pdb ; pdb.set_trace()


    block[names.data_vector, 'iacorr'+"_CHI2"] = chi2
    block[names.likelihoods, 'iacorr'+"_LIKE"] = like
    #import pdb ; pdb.set_trace()
    
    return 0



