##Building reduced shear power spectrum with Matter-Matter-Dust Bispectrum model
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

z_step = 4
l_tripleprime_step = 8
phi_step = 8
def reduced_shear1(z_ini,l_tripleprime_max,z_alpha,z_beta,l_mag,l_phi):
	sigma_galaxy = np.pi * r_virial**2
	shear = 0
	D_alpha = distance(z_ini,z_alpha)
	D_beta = distance(z_ini,z_beta)
	#redshift integral from 0 to Chi(z_alpha)
	delta_z = (z_alpha-z_ini)/z_step
	for i in range (z_step):
		zi = (z_ini + i*delta_z)
		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
		D_mid = distance(z_ini,zmid)
		#l_tripleprime magnitude integral from 0 to some max
		delta_l_tripleprime = (l_tripleprime_max)/l_tripleprime_step
		for j in range (l_tripleprime_step):
			l_tripleprime_j = j*delta_l_tripleprime
			l_tripleprime_mid = 1/2*(l_tripleprime_j + (j+1)*delta_l_tripleprime)
			#angular integral for l_tripleprime from 0 to pi FOR THE SPECIAL CASE l_phi = 0 rad!!
			delta_phi = np.pi/phi_step
			for k in range (phi_step):
				phi_k = k*delta_phi
				phi_mid = 1/2*(phi_k + (k+1)*delta_phi)
				shear += 2*window_distance(D_mid,D_alpha) * window_distance(D_mid,D_beta) * sigma_galaxy*numberdensity_galaxy*tau_g(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * np.cos(2*l_phi - 2*phi_mid) * (9*Omega_m**2*d_h**(-4))/(4 *1/(1+zmid)**2) * total_halo_dust_bispectrum(zmid,kTriangle(l_mag/D_mid,l_tripleprime_mid/D_mid,l_phi - phi_mid))[0,0] * 1/(2*np.pi)**2 * delta_z * delta_phi * l_tripleprime_mid * delta_l_tripleprime
				print(i,j,k,shear)
	return (shear)