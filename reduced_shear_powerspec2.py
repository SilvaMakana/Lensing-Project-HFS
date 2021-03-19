##Building reduced shear power spectrum with Matter-Matter-Dust Bispectrum model in Elliptical coordinates
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

#coordinate conversion for (x,y) --> (mu,nu)
# x(mu,nu) = l_mag * cosh(mu) * cos(nu)
# y(mu,nu) = l_mag * sinh(mu) * sin(nu)
# mu_max = arcCosh(l_tripleprime_max/l_mag + cos(np.pi))
# for l_tripleprime_max = 10000, mu_max = 3.63

z_step = 4 #steps in redshift
mu_step = 8 #steps in mu
nu_step = 8 #steps in nu
def reduced_shear2(z_ini,mu_max,z_alpha,z_beta,l_mag,l_phi):
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
		window_alpha = window_distance(D_mid,D_alpha)
		window_beta = window_distance(D_mid,D_beta)
		halo_data = halo_info(zmid,M_halo_min,M_halo_max,n_halo_integral_step)
		factor = 2*window_alpha * window_beta * sigma_galaxy*numberdensity_galaxy*tau_g(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * (9*Omega_m**2*d_h**(-4))/(4 *1/(1+zmid)**2)
		#mu parameter integral from 0 to some max
		delta_mu = (mu_max)/mu_step
		for j in range (mu_step):
			mu_j = j*delta_mu
			mu_mid = 1/2*(mu_j + (j+1)*delta_mu)
			#nu parameter integral from 0 to pi FOR THE SPECIAL CASE l_phi = 0 rad!!
			delta_nu = np.pi/nu_step
			for k in range (nu_step):
				nu_k = k*delta_nu
				nu_mid = 1/2*(nu_k + (k+1)*delta_nu)
				psi = np.arccos((-np.cosh(mu_mid) * np.cos(nu_mid) + 1)/(2*(np.cosh(mu_mid) - np.cos(nu_mid))))
				my_tri = kTriangle(l_mag/D_mid,l_mag * (np.cosh(mu_mid) - np.cos(nu_mid))/D_mid,l_phi - (np.pi - psi))
				shear_ijk = factor * np.cos(2*l_phi - 2*(np.pi - psi))  * total_halo_dust_bispectrum(zmid,my_tri,halo_data)[0,0] * 1/(2*np.pi)**2 * 1/4 * l_mag**2 * np.abs((np.cosh(2 * mu_mid) - np.cos(2 * nu_mid))) * delta_z * delta_mu * delta_nu
				#shear_ijk = factor * np.cos(2*l_phi - 2*(np.arccos((np.cosh(mu_mid) * np.cos(nu_mid) + 1)/(2*(np.cosh(mu_mid) - np.cos(nu_mid))))))  * total_halo_dust_bispectrum(zmid,kTriangle(l_mag/D_mid,l_mag * (np.cosh(mu_mid) - np.cos(nu_mid))/D_mid,l_phi - (np.arccos((np.cosh(mu_mid) * np.cos(nu_mid) + 1)/(2*(np.cosh(mu_mid) - np.cos(nu_mid)))))),halo_data)[0,0] * 1/(2*np.pi)**2 * 1/2 * l_mag**2 * (np.cosh(2 * mu_mid) - np.cos(2 * nu_mid)) * delta_z * delta_mu * delta_nu
				shear += shear_ijk
				print(i,j,k,zmid,mu_mid,nu_mid,shear_ijk)
				#sys.stdout.flush()
	return (shear)