import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from CLASS_matter_powerspec import *
from distance import *
from dust_mass_extinction_measurement import *
from dust_optical_depth import *
from halo_model_bispec import *
from halo_model_dust_bispec import *
from infer_dust_mass import *
from kTriangle import *
from perturb_matter_bispec import *
from reduced_shear_powerspec1 import *
from global_variables import * 

#Plotting Nonlinear and Linear Power Spectrum
#plt.loglog(PSetNL.k_array,PSetNL.P_grid.T)
#plt.loglog(PSetLin.k_array,PSetLin.P_grid.T)
#plt.show()

#Calculating a distance between redshift z_i to z_f
#print(distance(0,2))

#Calculating dust mass from extinction measurement of dust
#print(Menard_value(0.177)/10**7)

#Plotting mean dust optical depth of Universe
#z_f = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
#ax = plt.gca()
#tau_f = []
#for j in range(len(z_f)): # range(20):
	# plt.scatter(z_f[j],1.086*tau_meandust(wavelength_V,n,0,z_f[j]))
	#print(z_f[j],1.086*tau_meandust(wavelength_V,n,0,z_f[j]))
	#tau_f.append(1.086*tau_meandust(wavelength_V,n,0,z_f[j]))
#plt.plot(z_f, tau_f)
#plt.yscale('log')
#plt.show()

#Calculating total halo model matter-matter-matter bispectrum
#mytri = kTriangle(0.01,0.01,2*np.pi/3)
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#print(total_halo_bispectrum(0,mytri,halo_data))

#Calulating total halo model matter-matter-dust bispectrum
#mytri1 = kTriangle(0.01,0.01,2*np.pi/3)
#mytri2 = kTriangle(100,100,2*np.pi/3)
#stellar_info = parameters_stellarMvshaloM(0)
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#total_halo_dust_bispectrum_value_1 = total_halo_dust_bispectrum(0,mytri1,halo_data,stellar_info)
#total_halo_dust_bispectrum_value_2 = total_halo_dust_bispectrum(0,mytri2,halo_data,stellar_info)
#print((0.01)**3/(2*np.pi**2)*(total_halo_dust_bispectrum_value_1)**(1/2),(100)**3/(2*np.pi**2)*(total_halo_dust_bispectrum_value_2)**(1/2))

#Calulating the infered dust mass from a scaling relation with the power law from dust extinction
#print(infer_dust_mass(4*10**7,10**(11.61),0.0209))

#Calulating reduced shear power Spectrum
#print(window_distance(1,2),window_distance(2,1))
print(reduced_shear(0,10000,1,1,10,0))