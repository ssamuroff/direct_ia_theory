from __future__ import print_function
from builtins import range
from cosmosis.datablock import names, option_section
import sys
import fitsio as fi
import numpy as np
import scipy.interpolate as spi


#ordering: gg, gp, pp
pimax_corrections={'tng': {0:{'gg':0.974641,'gp':0.982287,'pp':0.985378},1:{'gg':0.981684,'gp':0.986832,'pp':0.988855},2:{'gg':0.979082,'gp':0.985709,'pp':0.988479},3:{'gg':0.978361,'gp':0.984973,'pp':0.987730}},
                   'illustris': {0:{'gg':0.390854,'gp':0.393996,'pp':0.396007},1:{'gg':0.404012,'gp':0.404616,'pp':0.404172},2:{'gg':0.409620,'gp':0.411010,'pp':0.410383},3:{'gg':0.402978,'gp':0.405009,'pp':0.405595}},
                   'mbii': {0:{'gg':0.666008,'gp':0.683459,'pp':0.691134},1:{'gg':0.686631,'gp':0.699460,'pp':0.703857},2:{'gg':0.690361,'gp':0.706124,'pp':0.710931},3:{'gg':0.682522,'gp':0.698526,'pp':0.704644}}}

block_names = {'wgp':'galaxy_intrinsic_w', 'wpp':'intrinsic_w', 'wgg':'galaxy_w'}

def setup(options):

    use_limber = options.get_bool(option_section, "limber", default=True)
    apply_hartlap = options.get_bool(option_section, "apply_hartlap", default=True)

    filename = options.get_string(option_section, "datafile")
    corrs_to_use = options[option_section, "ctypes"].split()
    samples_to_use = options[option_section, "samples"].split()
    redshifts = options[option_section, "redshifts"]

    apply_pimax = options.get_bool(option_section, "apply_pimax", default=True) 
    if apply_pimax:
        print('will apply Pi_max correction.')

    if isinstance(redshifts, float):
        redshifts = np.atleast_1d(redshifts)
    else:
    #if isinstance(redshifts, np.unicode):
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

    
    for corr in corrs_to_use:
        for i,z0 in enumerate(redshifts):
            if (corr in done2 and (i==0)):
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
           # import pdb ; pdb.set_trace()
            sind = np.argsort(sample1)
            if len(sind)>1:
                sample1 = sample1[sind]
                sample2 = sample2[sind]
            for s1,s2 in zip(sample1,sample2):
                if ('%d%d'%(s1,s2) in done):
                    continue

                if 'BIN' in dvec.dtype.names:
                    mask = (dvec['BIN']==i)
                elif 'BIN1' in dvec.dtype.names:
                    mask = (dvec['BIN1']==z0[0]) & (dvec['BIN2']==z0[1]) & (dvec['SAMPLE1']==s1) & (dvec['SAMPLE2']==s2)
                else:
                    mask = np.ones_like(dvec['VALUE']).astype(bool)

                data.append(dvec['VALUE'][mask])
                R.append(dvec['SEP'][mask])
                done.append('%d%d'%(s1,s2))
                npt+=len(dvec['SEP'][mask])


            print("%s, redshift %d: found %d points"%(corr, i, npt))
            done2.append(corr)

    #import pdb ; pdb.set_trace()

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

    return redshifts, corrs_to_use, samples_to_use, R, data, invcov, mask1d, use_limber, apply_pimax

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
            try:
                scale_window = (R[count]>rlower) & (R[count]<rupper)
            except:
                import pdb ; pdb.set_trace()
            mask1d.append(scale_window)
            count+=1
            #print(count)

    # Flatten into 1D
    mask1d = np.concatenate(mask1d)

    return mask1d

def execute(block, config):
    redshifts, corrs_to_use, samples_to_use, R, data, invcov, mask1d, use_limber, apply_pimax = config

    if use_limber:
        suffix = '_limber'
    else:
        suffix = ''

    # Now construct the theory data vector
    y = []
    count=0
    
    for corr, (s1,s2) in zip(corrs_to_use,samples_to_use):
        for z0 in redshifts:
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
                try:
                    interpolator = spi.interp1d(np.log10(xf), Y)
                except: 
                    import pdb ; pdb.set_trace()
                ylog = False
                try:
                    y_resampled = interpolator(np.log10(R[count]))
                except:
                    import pdb ; pdb.set_trace()

            if apply_pimax:
                M = pimax_corrections[s1][int(z0)][corr.replace('w','')]
                y_resampled*=M
                #import pdb ; pdb.set_trace()

            y.append(y_resampled)
            count+=1

    y = np.concatenate(y)
    block[names.data_vector,'iacorr_theory'] = y
    block[names.data_vector,'iacorr_data'] = data

    # Evaluate the likelihood and save it 
    res = (y-data)
    res = res[mask1d]
    chi2 = np.dot( res, np.dot(invcov, res) )
    chi2 = float(chi2)
    like = -0.5*chi2

   # import pdb ; pdb.set_trace()


    block[names.data_vector, 'iacorr'+"_CHI2"] = chi2
    block[names.likelihoods, 'iacorr'+"_LIKE"] = like
    #import pdb ; pdb.set_trace()
    
    return 0



