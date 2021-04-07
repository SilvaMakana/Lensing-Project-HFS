##Building Dust Model from arXiv:0902.4240v1
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 
wavelength_V = 5.50*10**(-7) #in units of meters
r_virial = 0.110 #in units of h^-1 Mpc
numberdensity_galaxy = 0.037 #comoving number density of galaxies in units of h^3 Mpc^-3

#calling data in from kext_albedo_WD_MW_3.1_60_D03.all found in the website https://www.astro.princeton.edu/~draine/dust/dustmix.html
cols = ['lambda', 'albedo', '<cos>', 'C_ext/H' , 'K_abs', '<cos^2>', 'comment']

table = pd.read_csv('kext_albedo_WD_MW_3.1_60_D03.all', names=cols, skiprows=78, delim_whitespace=True)


wavelength = table['lambda']
extinction_per_H = table['C_ext/H']

dust_interp = interp1d(wavelength, extinction_per_H, kind='linear')

tau_g_Vband = 0.005871

#wavelength in meters
def tau_g(z,wavelength_obs):
	return(tau_g_Vband*dust_interp(wavelength_obs*10**(6)/(1+z))/dust_interp(wavelength_V*10**(6)/(1.36)))

#Defining dust model integral from arXiv:0902.4240v1 Eq. (50)
def tau_meandust(wavelength_obs,n,z_ini,z_f):
	sigma_galaxy = np.pi * r_virial**2
	tau_dust = 0
	delta_z = (z_f-z_ini)/n
	#for i in range (n):
		#zi = (z_ini + i*delta_z)
		#zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
	zmid = np.linspace(0.5*delta_z,(n-1/2)*delta_z,n)
	tau_dust = np.sum(sigma_galaxy*numberdensity_galaxy*tau_g(zmid,wavelength_obs)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L)) * delta_z
	return (tau_dust)