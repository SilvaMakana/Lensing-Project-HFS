import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from kTriangle import *
from distance import *
from dust_optical_depth import *
from perturb_matter_bispec import * 
from CLASS_matter_powerspec import *
from halo_model_bispec import *
from halo_model_dust_bispec import *
from global_variables import * 

#z_alpha = float(sys.argv[1])
#z_beta = float(sys.argv[2])
#l_mag = float(sys.argv[3])
nl = 50
nz = 4

log_l_array = np.logspace(1,4,nl)
z_array = np.array([0.35,0.5,0.7,0.95])

#defining C_l^ij as in Eq. (15) arXiv:0910.3786v2 as an integral over redshift
print(Clz_iz_j_array,np.shape(Clz_iz_j_array))
def Clz_iz_j(z_alpha,z_beta,l_mag):
	Clij = 0
	D_alpha = distance(0,z_alpha)
	D_beta = distance(0,z_beta)
	for i in range(n):
		delta_z = z_alpha/n
		zi = i*delta_z
		zmid = 1/2*(zi + (i+1)*delta_z) 
		D_mid = distance(0,zmid)
		window_alpha = window_distance(D_mid,D_alpha)
		window_beta = window_distance(D_mid,D_beta)
		Clij += l_mag**4 * (window_alpha * window_beta)/(D_mid**2) * (3*Omega_m * d_h**(-2))/(2 *1/(1+zmid)) * PSetLin.P_interp(zmid,l_mag/D_mid)[0,0] * d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
	return(Clij)
#print(Clz_iz_j(z_alpha,z_beta,l_mag))

#Using a counter to parse the data vector in the right order
"""counter = 0
Clz_iz_j_array = np.zeros(int(nl*nz*(nz+1)/2))
for il in range(nl):
	for izi in range(nz):
		for izj in range(izi+1):
			Clz_iz_j_array[counter] = Clz_iz_j(z_array[izi],z_array[izj],log_l_array[il])
			print(counter,il,izi,izj,Clz_iz_j_array[counter])
			counter += 1
print(Clz_iz_j_array)"""

#Using lists to build the data vector
Clz_iz_j_array = []
for il in range(nl):
	for izi in range(nz):
		for izj in range(izi+1): 
			Clz_iz_j_array.append(Clz_iz_j(z_array[izi],z_array[izj],log_l_array[il]))
			print(il,izi,izj,Clz_iz_j_array)
Clz_iz_j_array = np.array(Clz_iz_j_array)
print(Clz_iz_j_array)


