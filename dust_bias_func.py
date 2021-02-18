##Using 2-halo term (which contains the dust bias) from of the measurement in arXiv:0902.4240v1 in Eq. (30) to find a model for the dust mass as a function of halo mass
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy import special
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

#Amplitude of the two halo term from measurement profile, FIX BESSEL FUNCTIONS, IT SHOULD BE REGULAR BESSEL NOT SPHERICAL
def two_halo_term_amplitude_measurement(z_ini,r_perp,k_max):
	#galaxy bias parameter, REPLACE WITH THE FACT THAT THE SIMPLEST CASE IS THAT THE GALAXY IS BIASED THE SAME AS THE HALO BIAS
	Menard_halo_mass = 4.1 * 10**11
	integral_matter_powerspec = 0
	delta_k = k_max/n
	k_mid = np.linspace(0.5*delta_k,(n-1/2)*delta_k,n)
	integral_matter_powerspec = bias_parameter_1(z_ini,Menard_halo_mass) * 1/(2*np.pi) * (2.5/np.log(10)) * np.sum(k_mid * PSetNL.P_interp(z_ini,k_mid) * special.jv(0,k_mid*r_perp)) * delta_k
	return(extinction_measurement(r_perp)/(integral_matter_powerspec))

#A class that contains different models for the dust mass as a function of halo mass
#class dust_mass_model(object):
#	def __init__(self,in1):
#		#self.z = in1
#		#self.M_halo = in2
#		self.parameter_1 = in1
#
#	def alpha_parameter(self,stellarstuff):
#		Menard_halo_mass = 4.1 * 10**11 #
#		Menard_dust_value = 4.109 * 10**7 #mass of dust in a halo with halo mass Menard_halo_mass
#		return(np.log(M_dust_optimistic(Menard_halo_mass,stellar_info)/Menard_dust_value - 1)/np.log(Menard_halo_mass/self.parameter_1))
#
#	#Dust Mass Model
#	def dust_model_1(self,z,M_halo):
#		stellar_info = parameters_stellarMvshaloM(z)
#		#alpha_parameter = np.log(M_dust_optimistic(M_halo,stellar_info)/Menardvalue - 1)/np.log(M_halo/self.parameter_1)
#		alpha = alpha_parameter(stellar_info)
#		return(M_dust_optimistic(M_halo,stellar_info)/(1 + (M_halo/self.parameter_1)**(alpha_parameter)))
#
#	#def dust_model_test(self,z,M):
#	#	return(z*M*self.parameter_1*self.parameter_2)


#Definition of the two halo amplitude, DOES NOT WORK USING PYTHON FUNCTIONS TO EXECUTE INTEGRALS
#def two_halo_term_amplitude_def(z,M_halo_max,dust_model):
#	amplitude_def = 0
#	delta_M = (M_halo_max)/n
#	M_mid = np.linspace(0.5*delta_M,(n-1/2)*delta_M,n)
#	amplitude_def = K_ext_V * np.sum(bias_parameter_1(z,M_mid) * n_halo_distribution(z,M_mid) * dust_model.dust_model_1(z,M_mid)) * (1+z)**(2) * delta_M
#	return(amplitude_def)


def alpha_parameter(stellarstuff,parameter_cutoff):
	Menard_halo_mass = 4.1 * 10**11 
	Menard_dust_value = 4.109 * 10**7 #mass of dust in a halo with halo mass Menard_halo_mass
	return(np.log(M_dust_optimistic(Menard_halo_mass,stellarstuff)/Menard_dust_value - 1)/np.log(Menard_halo_mass/parameter_cutoff))

def dust_model_1(z,M_halo,parameter_cutoff):
	stellar_info = parameters_stellarMvshaloM(z)
	return(M_dust_optimistic(M_halo,stellar_info)/(1 + (M_halo/parameter_cutoff)**(alpha_parameter(stellar_info,parameter_cutoff))))

#Definition of the two halo amplitude, WORKS USING FOR LOOPS, log space for mass integration
def two_halo_term_amplitude_def(z,M_halo_max,parameter_cutoff):
	amplitude_def = 0
	for i in range(n):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		amplitude_def += bias_parameter_1(z,M_mid) * halo_distribution_function(z,M_mid) * dust_model_1(z,M_mid,parameter_cutoff) * (1+z)**(2) * delta_M
	return(K_ext_V * amplitude_def)

def find_parameter_cutoff(z):
	something = optimize.root_scalar(lambda parameter_cutoff: (two_halo_term_amplitude_measurement(z,0.177,1000) - two_halo_term_amplitude_def(z,10**20,parameter_cutoff)),bracket=[10**(-4),10**15],method ='brentq')
	return something.root



