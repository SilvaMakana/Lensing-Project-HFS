##Replicate Stellar Mass vs Halo Mass curve from arXiv:1207.6105v2 Eq(3) and Intrinsic Parameters from Results section
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from perturb_matter_bispec import * 
from CLASS_matter_powerspec import *
from halo_model_bispec import *
from global_variables import * 

#Defining a class to define the Intrinsic Parameters that are functions of redshift
class parameters_stellarMvshaloM(object):
	def __init__(self,in1):
		self.z = in1 #redshift


		self.scale_factor = 1/(1+self.z)
		self.parameter_nu = np.exp(-4*self.scale_factor**2)
		self.log_parameter_M1 = 11.514 + (-1.793*(self.scale_factor- 1) - 0.251*self.z) * self.parameter_nu
		self.log_parameter_epsilon = -1.777 + (-0.006*(self.scale_factor - 1) - 0.000*self.z) * self.parameter_nu - 0.119*(self.scale_factor - 1)
		self.parameter_alpha = -1.412 + (0.731*(self.scale_factor - 1))*self.parameter_nu
		self.parameter_delta = 3.508 + (2.608*(self.scale_factor - 1) - 0.043*self.z)*self.parameter_nu
		self.parameter_gamma = 0.316 + (1.319*(self.scale_factor -1) + 0.279*self.z)*self.parameter_nu


def function_f(x,stellarstuff):
	return(-np.log10(10**(stellarstuff.parameter_alpha*x)+1) + stellarstuff.parameter_delta*(np.log10(1 + np.exp(x)))**stellarstuff.parameter_gamma/(1 + np.exp(10**(-x))))

#logarithmic function of stella mass as a function of halo mass
def logM_stellar(M_halo,stellarstuff):
	return(stellarstuff.log_parameter_epsilon + stellarstuff.log_parameter_M1 + function_f((np.log10(M_halo) - stellarstuff.log_parameter_M1),stellarstuff) - function_f(0,stellarstuff))

#root equation to find the halo mass given a stellar mass, since the Behroozi et al. paper gives the stellar mass as a function of halo mass
def root_M_halo_from_M_stellar(M_halo,M_stellar,stellarstuff):
	return(10**(logM_stellar(M_halo,stellarstuff)) - M_stellar)

#solving the root equation above, the limits of the halo mass range are from arXiv:1207.6105v2 in Fig. 7 the M_halo axis
def M_halo_from_M_stellar(M_stellar,stellarstuff):
	M_min_Behroozi = 10**10
	M_max_Behroozi = 10**15
	something = optimize.root_scalar(lambda M_halo: root_M_halo_from_M_stellar(M_halo,M_stellar,stellarstuff),bracket=[M_min_Behroozi,M_max_Behroozi],method ='brentq')
	return(something.root)


#Mass of dust as a function of stellar mass, this is the most optimistic case where M_dust = yield * M_stellar
def M_dust_optimistic(M_halo,stellarstuff):
	return(0.015*10**(logM_stellar(M_halo,stellarstuff)))

#Normalization constant that is defined by the total dust mass in a halo of mass M_halo
def normalization_rho(M_halo,stellarstuff):
	r_integral = 0
	delta_r_halo = r_halo_virial(M_halo)/n
	r_halo_mid = np.linspace(0.5*delta_r_halo,(n-1/2)*delta_r_halo,n)
	r_integral = np.sum((r_halo_mid)**(-1.84) * r_halo_mid**2) * delta_r_halo
	return(M_dust_optimistic(M_halo,stellarstuff)/(4*np.pi * r_integral))

#Dust density as a function of M_halo and size of the halo (r_halo)
def rho_dust(r_halo,M_halo,stellarstuff):
	return(normalization_rho(M_halo,stellarstuff)*(r_halo)**(-1.84))


##And now, I am assmebling the matter matter dust bispectrum
##This will be similar to the classic bispectrum set up but now I will denote the third leg of the k-triangle in k-space to the dust component
##this means that any functions of k3 that depend on density will be the dust density function, the other k-legs of the triangle will be y_halo_parameter

#dimensionaless Fourier Transform of dust density, similar to y_halo_parameter
def u_dust_halo_parameter(k,M,stellarstuff):
	dust_parameter = 0
	M_dust_opt = M_dust_optimistic(M,stellarstuff)
	delta_r_dust = r_halo_virial(M)/n
	r_dust_mid = np.linspace(0.5*delta_r_dust,(n-1/2)*delta_r_dust,n)
	dust_parameter = np.sum(r_dust_mid**2 * np.sin(k*r_dust_mid)/(k*r_dust_mid) * rho_dust(r_dust_mid,M,stellarstuff)) * delta_r_dust
	return(1/M_dust_opt * 4*np.pi * dust_parameter)

#background dust density
def rho_bar_dust(z,stellarstuff):
	rhobardust = 0
	delta_M_halo = M_halo_max/n
	M_halo_mid = np.linspace(0.5*delta_M_halo,(n-1/2)*delta_M_halo,n)
	rhobardust = np.sum(M_dust_optimistic(M_halo_mid,stellarstuff) * halo_distribution_function(z,M_halo_mid)) * delta_M_halo
	return(rhobardust)


#Building the I-integrals that will be used to build the single, double, and triple halo bispectrum contributors but with dust density profiles for k3
def I_03_dust(myTriangle,halo_stuff,stellarstuff):
	I03dust = 0
	#dn_dm[i] = halo_info(z,M_halo_min,M_halo_max,n_halo_integral_step).dn_dm_array
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		I03dust += (M_halo_mid/rho_background_matter)**2 * (M_dust_optimistic(M_halo_mid,stellarstuff)) * halo_stuff.dn_dm_array[i] * y_halo_parameter2(myTriangle.k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k2,M_halo_mid,halo_stuff,i) * u_dust_halo_parameter(myTriangle.k3,M_halo_mid,stellarstuff) * delta_M_halo
	return(I03dust)


def I_12_dust(myTriangle,halo_stuff,stellarstuff,index):
	I12dust = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		k1 = myTriangle.k1; k2 = myTriangle.k2
		profile_func1 = (M_halo_mid/rho_background_matter)*y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
		profile_func2 = (M_halo_mid/rho_background_matter)*y_halo_parameter2(k2,M_halo_mid,halo_stuff,i)

		if index==1:
			k1 = myTriangle.k2; k2 = myTriangle.k3
			profile_func1 = (M_halo_mid/rho_background_matter)*y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
			profile_func2 = (M_dust_optimistic(M_halo_mid,stellarstuff))*u_dust_halo_parameter(k2,M_halo_mid,stellarstuff)

		if index==2:
			k1 = myTriangle.k3; k2 = myTriangle.k1
			profile_func1 = (M_dust_optimistic(M_halo_mid,stellarstuff))*u_dust_halo_parameter(k1,M_halo_mid,stellarstuff)
			profile_func2 = (M_halo_mid/rho_background_matter)*y_halo_parameter2(k2,M_halo_mid,halo_stuff,i)
		I12dust += halo_stuff.dn_dm_array[i] * halo_stuff.bias1_array[i] * profile_func1 * profile_func2 * delta_M_halo
		#print (I12)
	return(I12dust)

def I_11_dust(myTriangle,halo_stuff,stellarstuff,index):
	#bias_1 = 1  + (2*p_halo - 1)/critical_density_parameter
	#transformI11dust = 0
	I11dust = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		k1 = myTriangle.k1
		profile_func = y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
		prefactor = (M_halo_mid/rho_background_matter)
		if index==1:
			k1 = myTriangle.k2
			profile_func = y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
			prefactor = (M_halo_mid/rho_background_matter) 
		if index==2:
			k1 = myTriangle.k3
			profile_func = u_dust_halo_parameter(k1,M_halo_mid,stellarstuff)
			prefactor = M_dust_optimistic(M_halo_mid,stellarstuff)
		I11dust += prefactor * halo_stuff.dn_dm_array[i] * halo_stuff.bias1_array[i] * profile_func * delta_M_halo
		#transformI11dust += prefactor * halo_stuff.dn_dm_array[i] * (bias_1 - halo_stuff.bias1_array[i] * profile_func) * delta_M_halo
		#print (I11)
	#return(bias_1 - transformI11dust)
	return(I11dust)

def I_21_dust(myTriangle,halo_stuff,stellarstuff,index):
	#bias_2 = -8/21 * (1-2*p_halo)/critical_density_parameter + 2*p_halo*(2*p_halo - 1)/critical_density_parameter**2
	#transformI21dust = 0
	I21dust = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		k1 = myTriangle.k1
		profile_func = y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
		prefactor = (M_halo_mid/rho_background_matter) 
		if index==1:
			k1 = myTriangle.k2
			profile_func = y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)
			prefactor = (M_halo_mid/rho_background_matter)
		if index==2:
			k1 = myTriangle.k3
			profile_func = u_dust_halo_parameter(k1,M_halo_mid,stellarstuff)
			prefactor = M_dust_optimistic(M_halo_mid,stellarstuff)
		I21dust += prefactor * halo_stuff.dn_dm_array[i] * halo_stuff.bias2_array[i] * profile_func * delta_M_halo
		#print (I21)
	#return(bias_2 - transformI21dust)
	return(I21dust)


#single halo dust contribution is I_03_dust

#double halo dust contribution
def double_halo_dust_bispectrum(z,myTriangle,halo_stuff,stellarstuff):
	return(I_12_dust(myTriangle,halo_stuff,stellarstuff,0) * I_11_dust(myTriangle,halo_stuff,stellarstuff,2) * PSetLin.P_interp(z,myTriangle.k3) + I_12_dust(myTriangle,halo_stuff,stellarstuff,2) * I_11_dust(myTriangle,halo_stuff,stellarstuff,1) * PSetLin.P_interp(z,myTriangle.k2) + I_12_dust(myTriangle,halo_stuff,stellarstuff,1) * I_11_dust(myTriangle,halo_stuff,stellarstuff,0) * PSetLin.P_interp(z,myTriangle.k1))

#triple halo dust contribution
def triple_halo_dust_bispectrum(z,myTriangle,halo_stuff,stellarstuff):
	return((2*analy_F(myTriangle,0)*I_11_dust(myTriangle,halo_stuff,stellarstuff,2) + I_21_dust(myTriangle,halo_stuff,stellarstuff,2)) * I_11_dust(myTriangle,halo_stuff,stellarstuff,0)*I_11_dust(myTriangle,halo_stuff,stellarstuff,1)*PSetLin.P_interp(z,myTriangle.k1)*PSetLin.P_interp(z,myTriangle.k2) + (2*analy_F(myTriangle,2)*I_11_dust(myTriangle,halo_stuff,stellarstuff,1) + I_21_dust(myTriangle,halo_stuff,stellarstuff,1)) * I_11_dust(myTriangle,halo_stuff,stellarstuff,2)*I_11_dust(myTriangle,halo_stuff,stellarstuff,0)*PSetLin.P_interp(z,myTriangle.k3)*PSetLin.P_interp(z,myTriangle.k1) + (2*analy_F(myTriangle,1)*I_11_dust(myTriangle,halo_stuff,stellarstuff,0) + I_21_dust(myTriangle,halo_stuff,stellarstuff,0)) * I_11_dust(myTriangle,halo_stuff,stellarstuff,1)*I_11_dust(myTriangle,halo_stuff,stellarstuff,2)*PSetLin.P_interp(z,myTriangle.k2)*PSetLin.P_interp(z,myTriangle.k3))

#total halo dust bispectrum
def total_halo_dust_bispectrum(z,myTriangle,halo_stuff,stellarstuff):
	return((I_03_dust(myTriangle,halo_stuff,stellarstuff) + double_halo_dust_bispectrum(z,myTriangle,halo_stuff,stellarstuff) + triple_halo_dust_bispectrum(z,myTriangle,halo_stuff,stellarstuff))/rho_bar_dust(z,stellarstuff))