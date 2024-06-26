# Direct IA Theory
Code for IA modelling. The modules here are designed to be plugged into a CosmoSIS pipeline.

Contact: Simon Samuroff (s.samuroff@northeastern.edu)


## Contents:

### util

add_1h_ia:
<ul>
    <li><strong>Summary:</strong> Reads pre-computed one and two halo power spectra from the block, and add them together.</li>
    <li><strong>Language:</strong> python</li>
    <li><strong>Inputs:</strong> 1h and 2h IA power spectra P^1h_GI(k,z), P^2h_GI(k,z), P^1h_II(k,z), P^2h_II(k,z)</li>
    <li><strong>Outputs:</strong> Combined 1h+2h IA power spectrum P_GI(k,z), P_II(k,z)</li>

</ul>


flatten_pk:
<ul>
    <li><strong>Summary:</strong> Reads a pre-computed power spectrum and integrates along the redshift direction. Also applies galaxy bias and IA amplitudes if desired.</li>
    <li><strong>Language:</strong> python</li>
    <li><strong>Inputs:</strong> IA power spectra P_GI(k,z),P_II(k,z); redshift distributions p_1(z),p_2(z)</li>
    <li><strong>Outputs:</strong> Flattened spectra P_GI(k),P_II(k)</li>
</ul>

promote_ia_term:
<ul>
    <li><strong>Summary:</strong> Reads a power spectrum, renames it.</li>
    <li><strong>Language:</strong> python</li>
    <li><strong>Inputs:</strong> Generic power spectrum P(k,z)</li>
    <li><strong>Outputs:</strong> The same power spectrum, but now called something else in the data block P(k,z)</li>
</ul>


### likelihood

add_1h_ia:
<ul>
    <li><strong>Summary:</strong> Computes a likelihood for some combination of wgg, wg+ and w++ 2pt data. Scale cuts and ordering specified in the params file.</li>
    <li><strong>Language:</strong> python</li>
    <li><strong>Inputs:</strong> Theory correlations wgg, wg+, w++; data vector wgg', wg+', w++'; covariance matrix C.</li>
    <li><strong>Outputs:</strong> A likelihood and a chi2.</li>
</ul>


### power_spectra

schneider_bridle:
<ul>
    <li><strong>Summary:</strong> Computes IA power spectra using the fitting fuctions of https://arxiv.org/abs/0903.3870.</li>
    <li><strong>Language:</strong> python</li>
    <li><strong>Inputs:</strong> None</li>
    <li><strong>Outputs:</strong> One halo intrinsic alignment power spectra P^1h_GI(k,z), P^1h_II(k,z) </li>
</ul>

### projection

projected_corrs_limber:
<ul>
    <li><strong>Summary:</strong> Reads a flattened power spectrum and performs a Hankel transform with the appropriate Bessel function to generate wg+, wgg, w++ etc.</li>
    <li><strong>Language:</strong> C </li>
    <li><strong>Inputs:</strong> Flattened power spectra P(k)</li>
    <li><strong>Outputs:</strong> Real space projected correlations as a function of perpendicular separation wg+(r_p), w++(r_p), wgg(r_p).</li>
</ul>

projected_corrs_legendre:
<ul>
    <li><strong>Summary:</strong> Reads an IA power spectrum computes the line-of-sight projected correlations using Legendre polynomials.</li>
    <li><strong>Language:</strong> python </li>
    <li><strong>Inputs:</strong> IA power spectra P_GI(k,z), P_II(k,z).</li>
    <li><strong>Outputs:</strong> Real space projected correlations as a function of perpendicular separation wg+(r_p), w++(r_p), wgg(r_p).</li>
</ul>

photometric_ias:
<ul>
    <li><strong>Summary:</strong> Computes projected IA correlations in the case where one or both of the samples have finite redshift uncertainty (i.e. for photometric samples) using the prescription of https://arxiv.org/abs/1008.3491. Note that this process involves (a) computing C_ells via a series of Limber integrals, (b) a Hankel transform, and then (c) projecting in Pi and redshift. All of these steps are done internally (might take a few seconds, depending on settings), making use of CCL for (b). </li>
    <li><strong>Language:</strong> python </li>
    <li><strong>Inputs:</strong> IA power spectra P_GI(k,z), P_II(k,z).</li>
    <li><strong>Outputs:</strong> real space projected correlations as a function of perpendicular separation wg+(r_p), w++(r_p), wgg(r_p).</li>
</ul>

