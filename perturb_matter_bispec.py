##Perturbative Matter Bispectrum. Defining the functions from arXiv:astro-ph/9709112 Eq. 26-31
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 
from CLASS_matter_powerspec import *

def s_transfer(Omega_b):
	return(44.5*np.log(9.83/(Omega_0*h_cosmo**2))/(1 + 10*(Omega_b*h_cosmo**2)**0.75)**0.5)

def alpha_Gamma(Omega_b):
	return(1 - 0.328*np.log(431*Omega_0*h_cosmo**2)*Omega_b/Omega_0 + 0.38*np.log(22.3*Omega_0*h_cosmo**2)*(Omega_b/Omega_0)**2)

def Gamma_transfer(k):
	return(Omega_0*h_cosmo*(alpha_Gamma(Omega_b) + (1-alpha_Gamma(Omega_b))/(1+(0.43*k*s_transfer(Omega_b))**4)))

def q_transfer(k):
	return(k*Tempfactor_CMB**2/(Gamma_transfer(k)))

def L_transfer(k):
	return(np.log(2*np.e + 1.8*q_transfer(k)))

def C_transfer(k):
	return(14.2 + 731/(1+62.5*q_transfer(k)))

def Transfer(k):
	L_transfer1 = L_transfer(k)
	return(L_transfer1/(L_transfer1 + C_transfer(k)*q_transfer(k)**2))


##Defining spectral_n(z,k) such that n does NOT produce wiggles from the equations in arXiv:astro-ph/9709112, so we are smoothing it out##
def spectral_n_nowiggle(k):
	return(n_s_primordial + 1/(0.01)*np.log(Transfer(np.exp(0.01)*k)/(Transfer(np.exp(-0.01)*k)))) 


#defining non-linear k scale (h Mpc^-1) from arXiv:1111.4477v2
def scale_nonlin(z,k):
	return k**3 * PSetLin.P_interp(z,k)/(2*np.pi**2) - 1
def k_nonlin(z):
	return optimize.root_scalar(lambda k: scale_nonlin(z,k),bracket=[kstart,kend],method ='brentq')
def q_nonlin(z,k):
	return k/k_nonlin(z).root

#defining constants for parameter functions from arXiv:1111.4477v2
a1 = 0.25
a2 = 3.5
a3 = 2
a4 = 1
a5 = 2
a6 = -0.2

#defining Window function for sigma_8 as a function of wavenumber (k [Mpc^-1]) and size of the sphere (R [Mpc])
R_8 = 8 #by definition of sigma_8 function, radius of sphere is 8 Mpc
def window(k,R):
	return 3/(R**3)*(np.sin(k*R) - k*R*np.cos(k*R))/k**3

#defining sigma_8 function
def sigma(z,kstart,kend,R,n): #Sigma cosmological function of redshift, z
	Ai = 0
	for i in range (n):
		delta_k = (kend-kstart)/n
		#k_i = (kstart + i*delta_k)
		#thetamid = 1/2*(theta_i + starttheta + (i+1)*delta_theta)
		kmid = kstart + (2*i + 1)/2 * delta_k
		Ai += kmid**2*PSetLin.P_interp(z,kmid)[:,0]*window(kmid,R)**2
		#if i<1000:
			#print('{:4d} {:11.5e} {:11.5e} {:11.5e} {:11.5e}'.format(i,kmid,PSetLin.P_interp(z,kmid)[0,0],window(kmid),Ai))
	return np.sqrt(Ai/(2*np.pi**2)*delta_k)

sigma_8_interp = interp1d(PSetLin.z_array,sigma(PSetLin.z_array,kstart,kend,R_8,n),kind = "cubic")

#defining parameter functions from arXiv:1111.4477v2
def Q3(z,k):
	return ((4 - 2**spectral_n_nowiggle(k))/(1 + 2**(spectral_n_nowiggle(k) + 1)))
def perturb_a(z,k):
	spectral_n_nowiggle1 = spectral_n_nowiggle(k)
	q_nonlin1 = q_nonlin(z,k)
	Q31 = (4 - 2**spectral_n_nowiggle1)/(1 + 2**(spectral_n_nowiggle1 + 1))
	return (1 + sigma_8_interp(z)**a6 * (0.7*Q31)**0.5 * (q_nonlin1*a1)**(spectral_n_nowiggle1 + a2))/ (1 + (q_nonlin1*a1)**(spectral_n_nowiggle1 + a2))
def perturb_b(z,k):
	spectral_n_nowiggle1 = spectral_n_nowiggle(k)
	q_nonlin1 = q_nonlin(z,k)
	return (1 + 0.2*a3*(spectral_n_nowiggle1+3)*q_nonlin1**(spectral_n_nowiggle1+3))/(1 + q_nonlin1**(spectral_n_nowiggle1+3.5))
def perturb_c(z,k):
	spectral_n_nowiggle1 = spectral_n_nowiggle(k)
	q_nonlin1 = q_nonlin(z,k)
	return ((1 + 4.5*a4/((1.5 + (spectral_n_nowiggle1 +3)**4))*(q_nonlin1*a5)**(spectral_n_nowiggle1 + 3)))/(1 + (q_nonlin1*a5)**(spectral_n_nowiggle1 + 3.5))

#Defining perturbative F_2 function
def perturb_F(z,myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return (5/7*perturb_a(z,k1)*perturb_a(z,k2)+1/2*cos12*(k1/k2+k2/k1)*perturb_b(z,k1)*perturb_b(z,k2)+2/7*cos12**2*perturb_c(z,k1)*perturb_c(z,k2))

#Defining analytic F_2 function
def analy_F(myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return(5/7+1/2*cos12*(k1/k2+k2/k1)+2/7*cos12**2)

#matter-matter-matter bispectrum from perturbation theory
def B_matterspec(z,myTriangle):
	return(2*perturb_F(z,myTriangle,0)*PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k2) + 2*perturb_F(z,myTriangle,1)*PSetNL.P_interp(z,myTriangle.k2)*PSetNL.P_interp(z,myTriangle.k3) + 2*perturb_F(z,myTriangle,2)*PSetNL.P_interp(z,myTriangle.k3)*PSetNL.P_interp(z,myTriangle.k1))

#Reduced bispectrum, useful since it scales away cosmological dependencies
def Q123(z,myTriangle):
	return(B_matterspec(z,myTriangle)/(PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k2) + PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k3) + PSetNL.P_interp(z,myTriangle.k2)*PSetNL.P_interp(z,myTriangle.k3)))