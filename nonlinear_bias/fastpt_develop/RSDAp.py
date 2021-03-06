''' This is the script to make figure 1. of the paper
	This script is a a good exampel of how to implement 1-loop calculations.
	See line 24 (or around line 24 ) for the call to FAST-PT
	J. E. McEwen
	email: jmcewen314@gmail.com
	
'''

import numpy as np 
from matter_power_spt import one_loop
import FASTPTII 
from time import time 

# load the input power spectrum data 
d=np.loadtxt('PT_1loop_RSD_Aterms_planck15.dat')

k=d[:,0]
P=d[:,1]

# use if you want to interpolate data 
#from scipy.interpolate import interp1d 
#power=interp1d(k,P)
#k=np.logspace(np.log10(k[0]),np.log10(k[-1]),3000)
#P=power(k)
#print d[:,0]-k


P_window=np.array([.2,.2])  
C_window=.65	
n_pad=1000
# initialize the FASTPT class		
fastpt=FASTPTII.FASTPT(k,to_do=['RSD'],low_extrap=-6,high_extrap=4,n_pad=n_pad) 
# if you want to use the high and low extrapolation uncomment below, comment above 
#fastpt=FASTPT.FASTPT(k,nu,low_extrap=-5,high_extrap=5,n_pad=n_pad) 
	
	
t1=time()	
f=1.
A1,A3,A5,B0,B2,B4,B6,P1,P3,P5=fastpt.RSD_components(P,f,C_window=C_window) 
t2=time()
print('time'), t2-t1 


print('To make a one-loop power spectrum for ', k.size, ' grid points, using FAST-PT takes ', t2-t1, 'seconds.')

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

fig=plt.figure(figsize=(16,10))

x1=10**(-2.5)
x2=10
ax1=fig.add_subplot(231)
ax1.set_ylim(1e2,1e5)
ax1.set_xlim(x1,x2)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r'RSD $A_i^{\rm II}(k)$', size=30)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', width=2, length=10)
ax1.tick_params(axis='both', which='minor', width=1, length=5)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax1.xaxis.labelpad = 20
ax1.set_xticklabels([])

ax1.plot(k,-(P1), lw=4, color='black', label=r'-$A_1^{\rm II}(k)$, $f=%2.1f$'%(f))


plt.legend(loc=3, fontsize=25)
plt.grid()

ax2=fig.add_subplot(234)
ax2.set_xscale('log')
ax2.set_xlabel(r'$k$ [$h$/Mpc]', size=30)
# ax2.set_ylabel('fractional difference',size=30)
ax2.set_ylim(-.0005,0.0005)
ax2.set_xlim(x1,x2)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', width=2, length=10)
ax2.tick_params(axis='both', which='minor', width=1, length=5)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax2.xaxis.labelpad = 20


ax2.plot(k,P1/((d[:,2]+f*d[:,3])*d[:,1])-1,lw=2, color='black', alpha=.5, label='fractional difference')

plt.legend(loc=3,fontsize=20)
plt.grid()

ax3=fig.add_subplot(232)
ax3.set_ylim(1e2,1e5)
ax3.set_xlim(x1,x2)
ax3.set_xscale('log')
ax3.set_yscale('log')
# ax3.set_ylabel(r'$A_i$', size=30)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.tick_params(axis='both', width=2, length=10)
ax3.tick_params(axis='both', which='minor', width=1, length=5)
ax3.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax3.xaxis.labelpad = 20
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax3.plot(k,-P3, lw=4, color='black', label=r'-$A_3^{\rm II}(k)$, $f=%2.1f$'%(f))


plt.legend(loc=3,fontsize=25)
plt.grid()

ax4=fig.add_subplot(235)
ax4.set_xscale('log')
ax4.set_xlabel(r'$k$ [$h$/Mpc]', size=30)
# ax4.set_ylabel('fractional difference',size=30)
ax4.set_ylim(-.0005,0.0005)
ax4.set_xlim(x1,x2)
ax4.tick_params(axis='both', which='major', labelsize=20)
ax4.tick_params(axis='both', width=2, length=10)
ax4.tick_params(axis='both', which='minor', width=1, length=5)
ax4.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax4.xaxis.labelpad = 20
ax4.set_yticklabels([])

ax4.plot(k,P3/((f*d[:,4]+f**2 *d[:,5])*d[:,1])-1,lw=2, color='black', alpha=.5, label='fractional difference')

plt.legend(loc=3,fontsize=20)
plt.grid()


ax5=fig.add_subplot(233)
ax5.set_ylim(1e2,1e5)
ax5.set_xlim(x1,x2)
ax5.set_xscale('log')
ax5.set_yscale('log')
# ax5.set_ylabel(r'$A_i$', size=30)
ax5.tick_params(axis='both', which='major', labelsize=20)
ax5.tick_params(axis='both', width=2, length=10)
ax5.tick_params(axis='both', which='minor', width=1, length=5)
ax5.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax5.xaxis.labelpad = 20
ax5.set_xticklabels([])
ax5.set_yticklabels([])

ax5.plot(k,-P5, lw=4, color='black', label=r'-$A_5^{\rm II}(k)$, $f=%2.1f$'%(f))

plt.legend(loc=3,fontsize=25)
plt.grid()

ax6=fig.add_subplot(236)
ax6.set_xscale('log')
ax6.set_xlabel(r'$k$ [$h$/Mpc]', size=30)
# ax6.set_ylabel('fractional difference',size=30)
ax6.set_ylim(-.0005,0.0005)
ax6.set_xlim(x1,x2)
ax6.tick_params(axis='both', which='major', labelsize=20)
ax6.tick_params(axis='both', width=2, length=10)
ax6.tick_params(axis='both', which='minor', width=1, length=5)
ax6.xaxis.set_major_formatter(FormatStrFormatter('%2.2f'))
ax6.xaxis.labelpad = 20
ax6.set_yticklabels([])

ax6.plot(k,P5/(f**2 *d[:,6]*d[:,1])-1,lw=2,color='black', alpha=.5, label='fractional difference')

plt.legend(loc=3,fontsize=20)
plt.grid()


plt.tight_layout()
plt.show()
fig.savefig('RSDAp_plot.pdf')