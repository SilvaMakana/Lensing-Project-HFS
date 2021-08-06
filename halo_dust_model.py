##Building Dust Model from Halo Model dust density profile
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 
from halo_model_dust_bispec import *
wavelength_V = 5.50*10**(-7) #in units of meters

#calling data in from kext_albedo_WD_MW_3.1_60_D03.all found in the website https://www.astro.princeton.edu/~draine/dust/dustmix.html
cols = ['lambda', 'albedo', '<cos>', 'C_ext/H' , 'K_abs', '<cos^2>', 'comment']

table = pd.read_csv('kext_albedo_WD_MW_3.1_60_D03.all', names=cols, skiprows=78, delim_whitespace=True)

dust_mass_per_H = 1.870*10**(-26)

wavelength = table['lambda'] #in microns
extinction_per_H = table['C_ext/H'] #in cm^2/H

dust_interp = interp1d(wavelength, extinction_per_H, kind='linear')

#wavelength must be entered as meters
def dust_opacity(z,wavelength_obs):
	return(dust_interp(wavelength_obs*10**6/(1+z))/dust_mass_per_H)

#wavelength must be entered as meters
def tau_meandust_halo(wavelength_obs,n,z_ini,z_f):
	tau_dust = 0
	delta_z = (z_f-z_ini)/n
	conversion = 1.989*10**33 * (3.086*10**24)**(-2) * h_cosmo
	for i in range (n):
		zi = (z_ini + i*delta_z)
		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
	#zmid = np.linspace(0.5*delta_z,(n-1/2)*delta_z,n)
		tau_dust += dust_opacity(zmid,wavelength_obs)* conversion *rho_bar_dust(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
	return (tau_dust)