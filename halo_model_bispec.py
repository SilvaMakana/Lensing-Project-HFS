##Building Halo Model Matter Bispectrum from arXiv:astro-ph/0001493v1 and APJ 548:7-18, 2001 February 10
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from perturb_matter_bispec import *
from CLASS_matter_powerspec import *
from global_variables import * 
#Input paramters
n_M = 124 #number of mass bins
n_z = 31 #number of redshift values
critical_density_parameter = 1.68 #value of a spherical overdensity at which it collapses for Einstein de-Sitter Model
rho_background_matter = Omega_m * rho_critial #background density of matter
alpha = -1 # NFW Halo Profile

#computing lagrangian radius from input parameters
def r_halo_lagrangian(M):
	return((3*M/(4*np.pi*rho_background_matter))**(1/3))

#building rms fluctuation within a top-hat filter at the virial radius corresponding to mass M
#empty array that has rows of mass entries from M_halo_array and columns of z entries from PSetLin.z_array
sigma_halo_array = np.zeros((n_z,n_M))

#halo mass array from 10**-4 M_S to 10**16 M_S
M_halo_min = 10**(10)
M_halo_max = 10**15
M_halo_array = np.logspace(10,15,n_M)

#zarray,Marray = np.meshgrid(PSetLin.z_array,M_halo_array)

#for loop to fill in the rows of sigma_halo_array
for i in range(n_M):
	sigma_halo_array[:,i] = sigma(PSetLin.z_array,kstart,kend,r_halo_lagrangian(M_halo_array[i]),n)

#Interpolated function for sigma of z and M
sigma_halo_interp = RectBivariateSpline(PSetLin.z_array,M_halo_array,sigma_halo_array,kx=3,ky=3)
#sigma_halo_interp = RectBivariateSpline(zarray,Marray,sigma_halo_array,kx=3,ky=3)

#defining mass scale for when nu_halo(M_scale) = 1
def scale_M_critical(z,M):
	return((critical_density_parameter/sigma_halo_interp(z,M)[0])**2 - 1)

def M_halo_critical(z):
	something = optimize.root_scalar(lambda M: scale_M_critical(z,M),bracket=[10**(-4),10**16],method ='brentq')
	return something.root

#defining concentration as a function of M and redshift
def c_concentration(z,M,M_crit):
	return(10*(M/M_crit)**(-0.2))

#virial radius in terms of lagrangian radius
def r_halo_virial(M):
	return(1/(200)**(1/3)*r_halo_lagrangian(M))

#defining characteristic radius
def r_characteristic(z,M,M_crit):
	return(r_halo_virial(M)/c_concentration(z,M,M_crit))

def rho_characteristic(z,M,M_crit):
	return(M/(4*np.pi*r_characteristic(z,M,M_crit)**3*(np.log(1+c_concentration(z,M,M_crit))-c_concentration(z,M,M_crit)/(1+c_concentration(z,M,M_crit)))))

#density profile for general dark matter profiles
def rho_halo(r,z,M,M_crit):
	return(rho_characteristic(z,M,M_crit)/((r/r_characteristic(z,M,M_crit))**(-alpha)*(1 + r/r_characteristic(z,M,M_crit))**(3 + alpha)))

#mass function
a_halo = 0.707
p_halo = 0.3
def f_halo_mass(z,M):
	nu_halo = (critical_density_parameter/sigma_halo_interp(z,M)[0])**2
	nu_a = a_halo*nu_halo
	return(1/7.78012*(1+nu_a**(-p_halo))*nu_a**(1/2)*e**(-nu_a/2)/nu_halo)


#defining the derivative of nu_halo w.r.t to M (mass)
def dnu_dM(z,M):
	sigma_again = RectBivariateSpline(PSetLin.z_array,M_halo_array,sigma_halo_array,kx=3,ky=3)
	#sigma_again = RectBivariateSpline(zarray,Marray,sigma_halo_array,kx=3,ky=3)
	#anti_dnu_dM = (critical_density_parameter / sigma_again)**2
	return(-2*critical_density_parameter**2/sigma_again(z,M)**3 * sigma_again(z,M,dy=1))
#dark matter distribution function
def halo_distribution_function(z,M):
	return(rho_background_matter/M*f_halo_mass(z,M)*dnu_dM(z,M))

#dimesionaless Fourier Transform of density profile
#def y_halo_parameter1(k,z,M):
#	y_halo = 0
#	for i in range(n):
#		delta_r_halo = r_halo_virial(M)/n
#		r_halo_i = delta_r_halo*i
#		r_halo_mid = 1/2*(r_halo_i + (i+1)*delta_r_halo)
#		y_halo += 1/M * 4*np.pi*r_halo_mid**2 * rho_halo(r_halo_mid,z,M) * np.sin(k*r_halo_mid)/(k*r_halo_mid) *delta_r_halo
#	return(y_halo)

#def y_halo_parameter2(k,M,halo_stuff,g):
#	y_halo = 0
#	delta_r_halo = r_halo_virial(M)/n
#	for i in range(n):
#		r_halo_i = delta_r_halo*i
#		r_halo_mid = 1/2*(r_halo_i + (i+1)*delta_r_halo)
#		y_halo += r_halo_mid**2 * halo_stuff.rho_halo_array[g,i] * np.sin(k*r_halo_mid)/(k*r_halo_mid) *delta_r_halo
#	return(1/M * 4 * np.pi * y_halo)

def y_halo_parameter2(k,M,halo_stuff,g):
	y_halo = 0
	delta_r_halo = r_halo_virial(M)/n
	r_halo_mid = np.linspace(0.5*delta_r_halo,(n-1/2)*delta_r_halo,n)
	y_halo = np.sum(r_halo_mid**2 * halo_stuff.rho_halo_array[g,:].T * np.sin(k*r_halo_mid)/(k*r_halo_mid),axis=0) * delta_r_halo
	return(1/M * 4 * np.pi * y_halo)


#halo bias parameters
#bias parameter 1
def bias_parameter_1(z,M):
	nu_halo = (critical_density_parameter/sigma_halo_interp(z,M)[0])**2
	return(1 + (a_halo*nu_halo-1)/critical_density_parameter + 2*p_halo/(critical_density_parameter*(1 + (a_halo*nu_halo)**p_halo)))

#bias parameter 2
#def bias_parameter_2(z,M):
#	nu_halo = (critical_density_parameter/sigma_halo_interp(z,M)[0])**2
#	return(8/21*(bias_parameter_1(z,M)-1) + (nu_halo - 3)/sigma_halo_interp(z,M)[0]**2 + 2*p_halo/((critical_density_parameter**2)*(1 + (a_halo*nu_halo)**p_halo))*(2*p_halo + 2*a_halo*nu_halo -1))

#bias parameter 2 from arxiv 1201.4827
def bias_parameter_2(z,M):
	nu_halo = (critical_density_parameter/sigma_halo_interp(z,M)[0])**2
	b1_L = -2*nu_halo/critical_density_parameter*((1 - a_halo*nu_halo)/(2*nu_halo) - p_halo/(nu_halo*(1+(a_halo*nu_halo)**p_halo)))
	b2_L = 4*nu_halo**2/critical_density_parameter**2*((p_halo**2 + nu_halo*a_halo*p_halo)/(nu_halo**2*(1+(a_halo*nu_halo)**p_halo)) + ((a_halo*nu_halo)**2 - 2*a_halo*nu_halo - 1)/(4*nu_halo**2)) + 2*nu_halo/critical_density_parameter**2*((1 - a_halo*nu_halo)/(2*nu_halo) - p_halo/(nu_halo*(1+(a_halo*nu_halo)**p_halo)))
	return(8/21* b1_L + b2_L)

#Defining analytic F_2 function
def analy_F(myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return(5/7+1/2*cos12*(k1/k2+k2/k1)+2/7*cos12**2)

#defining integrals in Eq(5) of https://iopscience.iop.org/article/10.1086/318660/fulltext/
n_halo_integral_step = 1000

#defining a class to store halo_distribution_function(z,M) and bias_parameter_1(z,M) and bias_parameter_2(z,M)
class halo_info(object):
	def __init__(self,in1,in2,in3,in4):
		self.z = in1
		self.M_halo_min = in2
		self.M_halo_max = in3
		self.n_halo_integral_step = in4
		#self.M = in5

		self.dn_dm_array = np.zeros((self.n_halo_integral_step))
		self.bias1_array = np.zeros((self.n_halo_integral_step))
		self.bias2_array = np.zeros((self.n_halo_integral_step))
		self.transform_bias1_array = np.zeros((self.n_halo_integral_step))
		self.transform_bias2_array = np.zeros((self.n_halo_integral_step))
		#self.r_halo_virial_array = np.zeros((self.n_halo_integral_step))
		self.c_concentration_array = np.zeros((self.n_halo_integral_step))
		self.M_halo_critical_array = np.zeros((self.n_halo_integral_step))
		self.r_characteristic_array = np.zeros((self.n_halo_integral_step))
		self.rho_characteristic_array = np.zeros((self.n_halo_integral_step))
		self.rho_halo_array = np.zeros((self.n_halo_integral_step,n))

		self.M_critical = M_halo_critical(self.z)

		self.transform_bias1_array = bias_parameter_1(self.z,0)
		self.transform_bias2_array = bias_parameter_2(self.z,0)


		for i in range(self.n_halo_integral_step):
			epsilon = (self.M_halo_max/self.M_halo_min)**(1/self.n_halo_integral_step) - 1
			delta_M_halo = self.M_halo_min* (self.M_halo_max/self.M_halo_min)**(i/self.n_halo_integral_step)*epsilon
			M_halo_mid = self.M_halo_min * (self.M_halo_max/self.M_halo_min)**(i/self.n_halo_integral_step) * (1 + epsilon/2)
			self.dn_dm_array[i] = halo_distribution_function(self.z,M_halo_mid)
			self.bias1_array[i] = bias_parameter_1(self.z,M_halo_mid)
			self.bias2_array[i] = bias_parameter_2(self.z,M_halo_mid)
			#self.r_halo_virial_array[i] = r_halo_virial(M_halo_mid)
			self.c_concentration_array[i] = c_concentration(self.z,M_halo_mid,self.M_critical)
			#self.scale_M_critical_array[i] = scale_M_critical(self.z,M_halo_mid)
			self.r_characteristic_array[i] = r_characteristic(self.z,M_halo_mid,self.M_critical)
			self.rho_characteristic_array[i] = rho_characteristic(self.z,M_halo_mid,self.M_critical)
			for j in range(n):
				delta_r_halo = r_halo_virial(M_halo_mid)/n
				r_halo_j = delta_r_halo*j
				r_halo_mid = 1/2*(r_halo_j + (j+1)*delta_r_halo)
				self.rho_halo_array[i,j] = rho_halo(r_halo_mid,self.z,M_halo_mid,self.M_critical)



#Logarithmic integrated version of halo distribution function
def integral_halo_dist_M_func(z):
	halo_dist_M = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		halo_dist_M += M_halo_mid * halo_distribution_function(z,M_halo_mid) * delta_M_halo
	return(halo_dist_M)



def I_03(myTriangle,halo_stuff):
	I03 = 0
	#dn_dm[i] = halo_info(z,M_halo_min,M_halo_max,n_halo_integral_step).dn_dm_array
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		I03 += (M_halo_mid/rho_background_matter)**3 * halo_stuff.dn_dm_array[i] * y_halo_parameter2(myTriangle.k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k2,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k3,M_halo_mid,halo_stuff,i) *delta_M_halo
	return(I03)

def integrand_I_03(myTriangle,halo_stuff):
	integrand_I03_array_w_M = np.zeros(n_halo_integral_step)
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		integrand_I03_array_w_M[i] = (M_halo_mid/rho_background_matter)**3 * halo_stuff.dn_dm_array[i] * y_halo_parameter2(myTriangle.k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k2,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k3,M_halo_mid,halo_stuff,i) * M_halo_mid
	return(integrand_I03_array_w_M)


def I_12(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1
	I12 = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		I12 += (M_halo_mid/rho_background_matter)**2 * halo_stuff.dn_dm_array[i] * halo_stuff.bias1_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(k2,M_halo_mid,halo_stuff,i) * delta_M_halo
		#print (I12)
	return(I12)

def integrand_I_12(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1
	integrand_I12_array_w_M = np.zeros(n_halo_integral_step)
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		integrand_I12_array_w_M[i] = (M_halo_mid/rho_background_matter)**2 * halo_stuff.dn_dm_array[i] * halo_stuff.bias1_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(k2,M_halo_mid,halo_stuff,i) * M_halo_mid
	return(integrand_I12_array_w_M)

def I_11(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1
	if i==1:
		k1 = myTriangle.k2
	if i==2:
		k1 = myTriangle.k3
	bias_1 = 1  + (2*p_halo - 1)/critical_density_parameter
	transformI11 = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		transformI11 += (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * (bias_1 - halo_stuff.bias1_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)) * delta_M_halo
		#print (I11)
	return(bias_1 - transformI11)

def integrand_I_11(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1
	if i==1:
		k1 = myTriangle.k2
	if i==2:
		k1 = myTriangle.k3
	bias_1 = 1  + (2*p_halo - 1)/critical_density_parameter
	integrand_I11_array_w_M = np.zeros(n_halo_integral_step)
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		integrand_I11_array_w_M[i] = (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * (bias_1 - halo_stuff.bias1_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)) * M_halo_mid
	return(integrand_I11_array_w_M)



#def I_11(myTriangle,halo_stuff,i):
#	k1 = myTriangle.k1
#	if i==1:
#		k1 = myTriangle.k2
#	if i==2:
#		k1 = myTriangle.k3
#	I11 = 0
#	for i in range(n_halo_integral_step):
#		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
#		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
#		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
#		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
#		#M_halo_i = delta_M_halo*i
#		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
#		I11 += (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * halo_stuff.bias1_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i) * delta_M_halo
#		#print (I11)
#	return(I11)

def I_01(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1
	if i==1:
		k1 = myTriangle.k2
	if i==2:
		k1 = myTriangle.k3
	I01 = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		I01 += (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i) * delta_M_halo
		#print (I11)
	return(I01)


def I_21(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1
	if i==1:
		k1 = myTriangle.k2
	if i==2:
		k1 = myTriangle.k3
	bias_2 = -8/21 * (1-2*p_halo)/critical_density_parameter + 2*p_halo*(2*p_halo - 1)/critical_density_parameter**2
	transformI21 = 0
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		transformI21 += (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * (bias_2 - halo_stuff.bias2_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)) * delta_M_halo
		#print (I21)
	return(bias_2 - transformI21)

def integrand_I_21(myTriangle,halo_stuff,i):
	k1 = myTriangle.k1
	if i==1:
		k1 = myTriangle.k2
	if i==2:
		k1 = myTriangle.k3
	bias_2 = -8/21 * (1-2*p_halo)/critical_density_parameter + 2*p_halo*(2*p_halo - 1)/critical_density_parameter**2
	integrand_I21_array_w_M = np.zeros(n_halo_integral_step)
	for i in range(n_halo_integral_step):
		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
		#M_halo_i = delta_M_halo*i
		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
		integrand_I21_array_w_M[i] = (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * (bias_2 - halo_stuff.bias2_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i)) * M_halo_mid
	return(integrand_I21_array_w_M)


#def I_21(myTriangle,halo_stuff,i):
#	k1 = myTriangle.k1
#	if i==1:
#		k1 = myTriangle.k2
#	if i==2:
#		k1 = myTriangle.k3
#	I21 = 0
#	for i in range(n_halo_integral_step):
#		epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
#		delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
#		M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
#		#delta_M_halo = (10**16 - 10**8)/n_halo_integral_step
#		#M_halo_i = delta_M_halo*i
#		#M_halo_mid = 1/2*(M_halo_i + (i+1)*delta_M_halo)
#		I21 += (M_halo_mid/rho_background_matter) * halo_stuff.dn_dm_array[i] * halo_stuff.bias2_array[i] * y_halo_parameter2(k1,M_halo_mid,halo_stuff,i) * delta_M_halo
#		#print (I21)
#	return(I21)

#defining single, double, and triple halo contribution to halo model bispectrum as formulated in https://iopscience.iop.org/article/10.1086/318660/fulltext/

#single halo contribution is simply I_03 function

#double halo contribution
def double_halo_bispectrum(z,myTriangle,halo_stuff):
	return(I_12(myTriangle,halo_stuff,0) * I_11(myTriangle,halo_stuff,2) * PSetLin.P_interp(z,myTriangle.k3) + I_12(myTriangle,halo_stuff,2) * I_11(myTriangle,halo_stuff,1) * PSetLin.P_interp(z,myTriangle.k2) + I_12(myTriangle,halo_stuff,1) * I_11(myTriangle,halo_stuff,0) * PSetLin.P_interp(z,myTriangle.k1))

#triple halo contribution
def triple_halo_bispectrum(z,myTriangle,halo_stuff):
	return((2*analy_F(myTriangle,0)*I_11(myTriangle,halo_stuff,2) + I_21(myTriangle,halo_stuff,2)) * I_11(myTriangle,halo_stuff,0)*I_11(myTriangle,halo_stuff,1)*PSetLin.P_interp(z,myTriangle.k1)*PSetLin.P_interp(z,myTriangle.k2) + (2*analy_F(myTriangle,2)*I_11(myTriangle,halo_stuff,1) + I_21(myTriangle,halo_stuff,1)) * I_11(myTriangle,halo_stuff,2)*I_11(myTriangle,halo_stuff,0)*PSetLin.P_interp(z,myTriangle.k3)*PSetLin.P_interp(z,myTriangle.k1) + (2*analy_F(myTriangle,1)*I_11(myTriangle,halo_stuff,0) + I_21(myTriangle,halo_stuff,0)) * I_11(myTriangle,halo_stuff,1)*I_11(myTriangle,halo_stuff,2)*PSetLin.P_interp(z,myTriangle.k2)*PSetLin.P_interp(z,myTriangle.k3))

#permutations for triple_halo_bispectrum
# first - (2*perturb_F(z,myTriangle,0)*I_11(z,myTriangle,2) + I_21(z,myTriangle,2)) * I_11(z,myTriangle,0)*I_11(z,myTriangle,1)*PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k2)

# second - (2*perturb_F(z,myTriangle,2)*I_11(z,myTriangle,1) + I_21(z,myTriangle,1)) * I_11(z,myTriangle,2)*I_11(z,myTriangle,0)*PSetNL.P_interp(z,myTriangle.k3)*PSetNL.P_interp(z,myTriangle.k1)

# third - (2*perturb_F(z,myTriangle,1)*I_11(z,myTriangle,0) + I_21(z,myTriangle,0)) * I_11(z,myTriangle,1)*I_11(z,myTriangle,2)*PSetNL.P_interp(z,myTriangle.k2)*PSetNL.P_interp(z,myTriangle.k3)

#total halo bispectrum is the sum of all the individual contribution
def total_halo_bispectrum(z,myTriangle,halo_stuff):
	return(I_03(myTriangle,halo_stuff) + double_halo_bispectrum(z,myTriangle,halo_stuff) + triple_halo_bispectrum(z,myTriangle,halo_stuff))