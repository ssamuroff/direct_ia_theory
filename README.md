# direct_ia_theory
Code for IA modelling. The modules here are designed to be plugged into a CosmoSIS pipeline.



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
