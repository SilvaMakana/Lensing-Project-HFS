##Using 2-halo term (which contains the dust bias) from of the measurement in arXiv:0902.4240v1 in Eq. (30) to find a model for the dust mass as a function of halo mass
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from kTriangle import *
from perturb_matter_bispec import * 
from CLASS_matter_powerspec import *
from halo_model_bispec import *
from halo_model_dust_bispec import *
from global_variables import * 

#Relate Eq. (30) in arXiv:0902.4240v1 to the 2-halo term and solving for the amplitude
K_ext_V = 3.21689961*10**(-12) / h_cosmo # dust opacity  in units of (Mpc*h^-1)^2/(M_solar*h^-1), values quoted in arXiv:0902.4240v1 Eq.(43) is 1.54*10^(4) cm^2/g

#Defining extinction measurement power law, Eq. (30)
def extinction_measurement(r_perp):
	return(4.14*10**(-3) * 0.1**(0.84) * (r_perp)**(-0.84))

#Amplitude of the two halo term from measurement profile
def two_halo_term_amplitude_measurement(z_ini,r_perp,k_max):
	galaxy_bias = 1 #galaxy bias parameter
	integral_matter_powerspec = 0
	delta_k = k_max/n
	k_mid = np.linspace(0.5*delta_k,(n-1/2)*delta_k,n)
	integral_matter_powerspec = 1/(2*np.pi) * (2.5/np.log(10)) * np.sum(k_mid * PSetNL.P_interp(z_ini,k_mid) * np.sin(k_mid * r_perp)/(k_mid * r_perp)) * delta_k
	return(extinction_measurement(r_perp)/(galaxy_bias * integral_matter_powerspec))

#A class that contains different models for the dust mass as a function of halo mass
class dust_mass_model(object):
	def __init__(self,in1,in2):
		#self.z = in1
		#self.M_halo = in2
		self.parameter_1 = in1
		self.parameter_2 = in2

		#stellar_info = parameters_stellarMvshaloM(self.z)

	#Dust Mass Model
	def dust_model_1(self,z,M_halo):
		stellar_info = parameters_stellarMvshaloM(z)
		return(M_dust_optimistic(M_halo,stellar_info)/(1 + (M_halo/self.parameter_2)**(self.parameter_1)))

	#def dust_model_test(self,z,M):
	#	return(z*M*self.parameter_1*self.parameter_2)


#Integrate dn/dM (halo distribution function) with repsect to M
def n_halo_distribution(z,M_max):
	n_halo_dist = 0
	delta_M = (M_max)/n
	M_mid = np.linspace(0.5*delta_M,(n-1/2)*delta_M,n)
	n_halo_dist = np.sum(halo_distribution_function(z,M_mid)) * delta_M
	return(n_halo_dist)


#Definition of the two halo amplitude, DOES NOT WORK USING PYTHON FUNCTIONS TO EXECUTE INTEGRALS
#def two_halo_term_amplitude_def(z,M_halo_max,dust_model):
#	amplitude_def = 0
#	delta_M = (M_halo_max)/n
#	M_mid = np.linspace(0.5*delta_M,(n-1/2)*delta_M,n)
#	amplitude_def = K_ext_V * np.sum(bias_parameter_1(z,M_mid) * n_halo_distribution(z,M_mid) * dust_model.dust_model_1(z,M_mid)) * (1+z)**(2) * delta_M
#	return(amplitude_def)

#Definition of the two halo amplitude, WORKS USING FOR LOOPS
def two_halo_term_amplitude_def(z,M_halo_max,dust_model):
	amplitude_def = 0
	delta_M = (M_halo_max)/n
	for i in range(n):
		M_i = i*delta_M
		M_mid = 1/2*(M_i + (i+1)*delta_M)
		amplitude_def += bias_parameter_1(z,M_mid) * n_halo_distribution(z,M_mid) * dust_model.dust_model_1(z,M_mid) * (1+z)**(2) * delta_M
	return(K_ext_V * amplitude_def)





