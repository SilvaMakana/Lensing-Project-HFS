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
from reduced_shear_powerspec import *
from reduced_shear_powerspec1 import *
from reduced_shear_powerspec2 import *
#from dust_bias_func import *
from global_variables import * 

#Plotting Nonlinear and Linear Power Spectrum
#plt.loglog(PSetNL.k_array,PSetNL.P_grid.T)
#plt.loglog(PSetLin.k_array,PSetLin.P_grid.T)
#plt.show()

#Calculating a distance between redshift z_i to z_f
#print(distance(0,2))

#Calculating dust mass from extinction measurement of dust
#print(r_halo_virial(4.1*10**11))
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
#print(reduced_shear(0,10000,1,1,10,0))

#Calculating extinction measurement
#print(extinction_measurement(1))

#Calculating bias parameter amplitude from 2-halo term from dust extinction measurement
#print(two_halo_term_amplitude_measurement(0,0.177,1000))

#Testing if the class in dust_bias_func.py is working
#dustmodel = dust_mass_model(0,1)
#stellar_info = parameters_stellarMvshaloM(0)
#print(dustmodel.dust_model_test(1,2))
#print(dustmodel.dust_model_1(0,10**12),M_dust_optimistic(10**12,stellar_info)/2)
#print(n_halo_distribution(0,10**14))
#print(two_halo_term_amplitude_def(1,10**14,dustmodel)) ##NOT WORKING
#print(alpha_parameter(parameters_stellarMvshaloM(0.36),13102297747878.25))


#M_array = np.logspace(-3,16,1000)
#difference = two_halo_term_amplitude_measurement(2,0.177,1000) - two_halo_term_amplitude_def(2,10**50,parameter_array)
#plt.title("Dust Model")
#plt.ylabel("M_{dust} [h^{-1} M_{solar}]")
#plt.xlabel("M_{halo} [h^{-1} M_{solar}]")
#plt.semilogx(M_array,dust_model_1(0,M_array,find_parameter_cutoff(0.36)).T)
#plt.show()
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#print(y_halo_parameter2(100,10**10,halo_data,0))

#print(reduced_shear2(620*10**(-9),0,4.41,0.35,0.7,500,0)) #mu_max is about 4.41 corresponding to l_tripleprime_max of 10000 and nu_max = 0

#print(y_halo_parameter3(100,10**10,halo_data,0))
"""np_1f2 = np.frompyfunc(hyp1f2,4,1)
np_1f2_array1 = np.zeros(100)
np_1f2_array2 = np.zeros(100)
x_array = np.linspace(1,50,100)
for i in range (100):
	np_1f2_array1[i] = np_1f2(0.58,1.5,1.58,-1/4 * x_array[i]**2)
	np_1f2_array2[i] = np_1f2(0.5,1.5,1.5,-1/4 * x_array[i]**2)
plt.plot(x_array,np_1f2_array1)
plt.plot(x_array,np_1f2_array2)
plt.xlabel("x")
plt.ylabel("1F2")
plt.show()"""
#sys.exit()
##Checking if the total_halo_dust_bispectrum works
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
"""log_halo_k_array = np.logspace(-2,2,100)
for j in range (100):
	#reduced_shear_value = reduced_shear(0,10000,1,1,log_l_array[j],0)[0,0]
	#tri = kTriangle(log_halo_k_array[i],log_halo_k_array[i],2/3*np.pi)
	B1_dust = I_03_dust(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi),halo_data)
	B2_dust = double_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi),halo_data)[0,0]
	B3_dust = triple_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi),halo_data)[0,0]
	total_halo_dust_bispectrum_value = total_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi))[0,0]
	total_halo_bispectrum_value = total_halo_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi),halo_data)[0,0]
	matter_bispectrum_perturb = B_matterspec(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],5/6*np.pi))[0,0]
	#plt.scatter(log_halo_k_array[j],total_halo_bispectrum_value)
	#print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(B1)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B2)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B3)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))
	print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(B1_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B2_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B3_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_dust_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2))
"""

log_l_array = np.logspace(1,4,50)
reduced_shear_value = np.zeros(50)
for j in range (50):
	reduced_shear_value[j] = reduced_shear2(620*10**(-9),0,4.41,0.35,0.7,log_l_array[j],0)
	#plt.scatter(log_l_array[j],reduced_shear_value)
	print(log_l_array[j],reduced_shear_value[j])
	sys.stdout.flush()
	#plt.loglog(log_l_array[j],reduced_shear(0,10000,1,1,log_l_array[j],0))
plt.loglog(log_l_array,reduced_shear_value)
plt.xlabel('k [Mpc/h]')
plt.yscale('Reduced Shear Power Spectrum')
plt.show()



