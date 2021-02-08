import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
## kstart is the first k value from the CLASS output files in units of h^-1Mpc ##
kstart = 1.045e-5
kend = 210.674
##Integration step size##
n=10000

#start = 0
#end = 10
#n = 5
#Ai=0
#
#Trying to define numerical integration as a function 
#def area (start,end,n,Ai):
#	for i in range (n):
#		delta_x = (end-start)/n
#		xi = (start + i*delta_x)
#		xmid = 1/2*(xi + start + (i+1)*delta_x)
#		Ai += (xmid)**2 * delta_x 
#	return Ai
#print(area(0,10,5,0))

###Define numerical integration as a for loop
#for i in range (n):
#	delta_x = (end-start)/n
#	xi = (start + i*delta_x)
#	xmid = 1/2*(xi + start + (i+1)*delta_x)
#	Ai += (xmid)**2 * delta_x 
#print(xmid,Ai)

##Defining radial distance from redshift integrator
H0 = 67.37
h_cosmo = H0/100
Omega_k = 0
Omega_r = 0
Omega_b = 0.02233/h_cosmo**2
Omega_CDM = 0.1198/h_cosmo**2
Omega_0 = (Omega_b + Omega_CDM)
Omega_L = 0.680
Omega_m = 0.320
n_s_primordial = 0.963
d_h = 3000
Tempfactor_CMB = 1.00
rho_critial = 2.775*10**(11) #in units of h^-1*M_sun/(h^-1MpC)^3
#start = 0
#end = 3
#Chi = 0
#n = 10000
def distance(z_ini,z_f):
	Chi = 0
	for i in range (n):
		delta_z = (z_f-z_ini)/n
		zi = (z_ini + i*delta_z)
		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
		Chi += d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
	return Chi
#print(distance (0.3089,0.6911,0,0,3000,0,0.1,10000))

##Defining Window Function
def window_distance(distance2, distance1):
	if (distance1 >= distance2):
		return (1/distance2 - 1/distance1)
	else:
		return(0)
#Chi1 = 10
#Chi2 = 2
#if (Chi1 >= Chi2):
#	W = 1/Chi2 - 1/Chi1
#else:
#	W = 0
#print(W)


##Building Power Spectra from CLASS data output
class PowerSpectrumSingleZ(object):
	"""class to store a power spectrum at a single redshift, loaded from an input file, with input k range"""
	def __init__(self,filename,log_k_array):
		"""create a power spectrum object:
				filename: name of file containing power spectrum as table
				log_k_array: grid of log(k) values in h Mpc^-1"""
		self.filename = filename
		self.log_k_array = log_k_array

		#load the table from a text file, two columns k (h Mpc^-1) and P(k)
		table = np.loadtxt(filename)

		#construct grid of k values to interpolate to 


		#construct a linear interpolating fucnction from the input table for the power spectrum in log space
		self.log_P_interp = InterpolatedUnivariateSpline(np.log(table[:,0]),np.log(table[:,1]),k=3,ext=0)
		self.P_array = 	np.exp(self.log_P_interp(self.log_k_array))	

class PowerSpectrumMultiZ(object):
	"""class to store a power spectrum on a grid of k and z values"""
	def __init__(self,name_base,name_end,n_z,k_min,k_max,n_k,z_space=0.1,z_min=0.):
		"""create a power spectrum object:
				filenames of form name_base+(z index)+name_end
				n_z: number of files to search for z values
				k_min: minimum k in h Mpc^-1
				k_max: maximum k in h Mpc^-1
				n_k: number of k slices to build grid
				z_space: spacing of z grid 
				z_min minimum z value"""		
		self.name_base = name_base
		self.name_end = name_end
		self.k_min = k_min
		self.k_max = k_max
		self.n_k = n_k	
		self.n_z = n_z	
		self.z_space = z_space
		self.z_min = z_min

		self.k_min = k_min
		self.k_max = k_max
		self.n_k = n_k

		#build a k grid
		self.log_k_array = np.linspace(np.log(self.k_min), np.log(k_max), num=self.n_k) 
		self.k_array = np.exp(self.log_k_array) #wavenumber array from data file in h/Mpc

		#build a z grid
		self.z_max = self.z_min+self.z_space*self.n_z
		self.z_array = np.arange(self.z_min,self.z_max,self.z_space)

		#construct an array of PowerSpectrumSingleZ objects for each redshift
		self.Ps = np.zeros(self.n_z,dtype=PowerSpectrumSingleZ)
		for i in range(0,self.n_z):
			fname = self.name_base+str(i+1)+self.name_end
			self.Ps[i] = PowerSpectrumSingleZ(fname,self.log_k_array)

		#construct a grid of P values
		self.P_grid = np.zeros((self.n_z,self.n_k))
		for i in range(0,self.n_z):
			self.P_grid[i,:] = self.Ps[i].P_array

		#construct a linear interpolating function in linear space for z and log space for k
		self.log_P_interp = RectBivariateSpline(self.z_array,self.log_k_array,np.log(self.P_grid),kx=3,ky=3)


	def P_interp(self,zs,ks):
		"""interpolate to a specific grid of ks (in h/Mpc) and zs 
		ks or zs can be either scalar or arrays"""
		return np.exp(self.log_P_interp(zs,np.log(ks)))

	def spectral_n(self,z,k):
		"""get n=d ln(P)/d ln(k) as a function of z and k"""
		return self.log_P_interp(z,np.log(k),dy=1)

##Building a momentum (k) triangle that only allows for closed k-space configurations in bispectrum
class kTriangle(object):
	def __init__(self,in1,in2,in3,input="SAS"):
		"""if using SAS, then, in1=length of k1, in2=length of k2, in3 = angle in radians between k1 and k2"""
		"""if using SSS, then, in1=length of k1, in2=length of k2, in3 = length of k3"""
		if input == "SAS":
			self.k1 = in1
			self.k2 = in2
			self.cos12 = np.cos(in3)
			self.k3 = np.sqrt(self.k1**2 + self.k2**2 + 2*self.k1*self.k2*self.cos12)

		elif input == "SSS":
			self.k1 = in1
			self.k2 = in2
			self.k3 = in3
			self.cos12 = (-self.k1**2 - self.k2**2 + self.k3**2)/(2*self.k1*self.k2)
		
		
		self.cos13 = (-self.k1**2 - self.k3**2 + self.k2**2)/(2*self.k1*self.k3)
		self.cos23 = (-self.k2**2 - self.k3**2 + self.k1**2)/(2*self.k2*self.k3)

		assert self.k1 + self.k2 > self.k3
	"""output for SAS method, length of k1, length of k2, and cosine of angle between k1 and k2"""
	def output_SAS(self):
		return (self.k1,self.k2,self.cos12)
	"""output for SSS method, length of k1, length of k2, length of k3"""
	def output_SSS(self):
		return (self.k1,self.k2,self.k3)

		#assert mod_k3 = np.sqrt(mod_k1**2 + mod_k2**2 - 2*mod_k1*mod_k2*np.cos(theta_12))
		#assert np.cos(theta_12) = 1/2 * (mod_k3**2 - mod_k2**2 - mod_k1**2)/(mod_k1*mod_k2)

#if __name__=='__main__':
#k range from data files generated by CLASS, number of k steps in each file NOT UNIFORMELY SPACED
#number of redshift slices
k_min = 1.045e-5
k_max = 210.674
n_k = 625
n_z = 31

dir_base = "/Users/makanas/class_public-2.8.2/output/LENSING_PROJECT/"
name_base = dir_base+"LENSING_PROJECT06_z"
name_endNL = "_pk_nl.dat"
name_endLin = "_pk.dat"
#Ps = np.zeros(n_z,dtype=PowerSpectrum)
#for i in range(0,n_z):
#	fname = name_base+str(i+1)+name_end
#	Ps[i] = PowerSpectrumSingleZ(fname,k_min,k_max,n_k)
PSetNL = PowerSpectrumMultiZ(name_base,name_endNL,n_z,k_min,k_max,n_k)
PSetLin = PowerSpectrumMultiZ(name_base,name_endLin,n_z,k_min,k_max,n_k)
#print(testP.filename,filename)
#table = np.loadtxt("LENSING_PROJECT00_z1_pk_nl.dat")

#log_k_array = np.linspace(-11.4691, 3.5082, num=617) #wavenumber array from data file in h/Mpc
#k_array = np.exp(log_k_array)
#log_P_interp = InterpolatedUnivariateSpline(np.log(table[:,0]),np.log(table[:,1]),k=1,ext=0)
#P_array = np.exp(log_P_interp(log_k_array))
#
#print(k_array)i
#for i in range(0,n_z):
#	plt.loglog(PSet.k_array,PSet.Ps[i].P_array)
#plt.loglog(PSetNL.k_array,PSetNL.P_grid.T)
#plt.loglog(PSetLin.k_array,PSetLin.P_grid.T)
#plt.show()
#sys.exit()
#for i in range(0,3):
#	for j in range(0,10):
#		print(i,j,PSetNL.P_interp(i,j))

##Matter Bispecrum. Defining the functions from arXiv:astro-ph/9709112 Eq. 26-31

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

#print(k_nonlin(1.0))
#sys.exit()
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

#print(Q3(1.0,0.08),perturb_a(1.0,0.08),perturb_b(1.0,0.08),perturb_c(1.0,0.08))
#sys.exit()
def perturb_F(z,myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return (5/7*perturb_a(z,k1)*perturb_a(z,k2)+1/2*cos12*(k1/k2+k2/k1)*perturb_b(z,k1)*perturb_b(z,k2)+2/7*cos12**2*perturb_c(z,k1)*perturb_c(z,k2))

def analy_F(myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return(5/7+1/2*cos12*(k1/k2+k2/k1)+2/7*cos12**2)

#print(perturb_F(1.0,kTriangle(0.08,0.08,0.2*np.pi),0),perturb_F(1.0,kTriangle(0.08,0.08,0.2*np.pi),1),perturb_F(1.0,kTriangle(0.08,0.08,0.2*np.pi),2))
#sys.exit()
#x = perturb_F(z,tri,0)

def B_matterspec(z,myTriangle):
	return(2*perturb_F(z,myTriangle,0)*PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k2) + 2*perturb_F(z,myTriangle,1)*PSetNL.P_interp(z,myTriangle.k2)*PSetNL.P_interp(z,myTriangle.k3) + 2*perturb_F(z,myTriangle,2)*PSetNL.P_interp(z,myTriangle.k3)*PSetNL.P_interp(z,myTriangle.k1))
#print(B_matterspec(z,tri))

def Q123(z,myTriangle):
	return(B_matterspec(z,myTriangle)/(PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k2) + PSetNL.P_interp(z,myTriangle.k1)*PSetNL.P_interp(z,myTriangle.k3) + PSetNL.P_interp(z,myTriangle.k2)*PSetNL.P_interp(z,myTriangle.k3)))

#tri[i] = kTriangle(k_input[i],k_input[i],0.2*np.pi)

#for i in range (10):
#z=1.0
#k_input = np.logspace(-1.52287874528,-0.69897000433,100)
#for i in range (100):
#	tri = kTriangle(k_input[i],2*k_input[i],0.6*np.pi)
#        #print(tri.k1)
#	plt.scatter(k_input[i],Q123(z,tri))
#plt.show()

#for i in range (100):
       # tri = kTriangle(k_input[i],2*k_input[i],0.6*np.pi)
        #print(tri.k1)
#        plt.scatter(k_input[i],spectral_n_nowiggle(k_input[i]))
#plt.show()

##Building Dust Model from  arXiv:0902.4240v1
#wavelength_V = 5.50*10**(-7) #in units of meters
#r_virial = 0.110 #in units of h^-1 Mpc
#numberdensity_galaxy = 0.037 #comoving number density of galaxies in units of h^3 Mpc^-3

#calling data in from kext_albedo_WD_MW_3.1_60_D03.all found in the website https://www.astro.princeton.edu/~draine/dust/dustmix.html
#cols = ['lambda', 'albedo', '<cos>', 'C_ext/H' , 'K_abs', '<cos^2>', 'comment']

#table = pd.read_csv('kext_albedo_WD_MW_3.1_60_D03.all', names=cols, skiprows=78, delim_whitespace=True)


#wavelength = table['lambda']
#extinction_per_H = table['C_ext/H']

#dust_interp = interp1d(wavelength, extinction_per_H, kind='linear')

#tau_g_Vband = 0.005871

#def tau_g(z):
#	return(tau_g_Vband*dust_interp(wavelength_V*10**(6)/(1+z))/dust_interp(wavelength_V*10**(6)/(1.36)))

#defining dust model integral

#def tau_meandust(wavelength,n,z_ini,z_f):
#	sigma_galaxy = np.pi * r_virial**2
#	tau_dust = 0
#	for i in range (n):
#		delta_z = (z_f-z_ini)/n
#		zi = (z_ini + i*delta_z)
#		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
#		tau_dust += sigma_galaxy*numberdensity_galaxy*tau_g(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
#	return (tau_dust)
#z_f = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
#ax = plt.gca()
#for j in range (20):
#	plt.scatter(z_f[j],1.086*tau_meandust(wavelength_V,n,0,z_f[j]))
#	print(z_f[j],1.086*tau_meandust(wavelength_V,n,0,z_f[j]))
#plt.yscale('log')
#plt.show()

##Building reduced shear model with Matter Bispectrum model from perturbation theory
#z_step = 80
#l_tripleprime_step = 160
#phi_step = 160
#def reduced_shear(z_ini,l_tripleprime_max,z_alpha,z_beta,l_mag,l_phi):
#	sigma_galaxy = np.pi * r_virial**2
#	shear = 0
#	D_alpha = distance(z_ini,z_alpha)
#	D_beta = distance(z_ini,z_beta)
#	#redshift integral from 0 to Chi(z_alpha)
#	for i in range (z_step):
#		delta_z = (z_alpha-z_ini)/z_step
#		zi = (z_ini + i*delta_z)
#		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
#		D_mid = distance(z_ini,zmid)
		#lʻʻʻ magnitude integral from 0 to some max
#		for j in range (l_tripleprime_step):
#			delta_l_tripleprime = (l_tripleprime_max)/l_tripleprime_step
#			l_tripleprime_j = j*delta_l_tripleprime
#			l_tripleprime_mid = 1/2*(l_tripleprime_j + (j+1)*delta_l_tripleprime)
			#angular integral for lʻʻʻ from 0 to pi FOR THE SPECIAL CASE l_phi = 0 rad!!
#			for k in range (phi_step):
#				delta_phi = np.pi/phi_step
#				phi_k = k*delta_phi
#				phi_mid = 1/2*(phi_k + (k+1)*delta_phi)
#				shear += 2*window_distance(D_mid,D_alpha) * window_distance(D_mid,D_beta) * sigma_galaxy*numberdensity_galaxy*tau_g(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * np.cos(2*l_phi - 2*phi_mid) * (9*Omega_m**2*d_h**(-4))/(4 *1/(1+zmid)**2) * B_matterspec(zmid,kTriangle(l_mag/D_mid,l_tripleprime_mid/D_mid,l_phi - phi_mid)) * 1/(2*np.pi)**2 * delta_z * delta_phi * l_tripleprime_mid * delta_l_tripleprime
				#print(i,j,k,shear)
#	return (shear)

#print(reduced_shear(0,10000,1,1,1000,0.6*np.pi))
#if __name__ == '__main__':
#	log_l_array = np.logspace(1,4,100)
#	for j in range (100):
#		reduced_shear_value = reduced_shear(0,10000,1,1,log_l_array[j],0)[0,0]
#		plt.scatter(log_l_array[j],reduced_shear_value)
#		print(log_l_array[j],reduced_shear_value)
	#plt.loglog(log_l_array[j],reduced_shear(0,10000,1,1,log_l_array[j],0))
#	plt.xscale('log')
#	plt.yscale('log')
#	plt.show()


##Building Halo Model Matter Bispectrum

#Input paramters
n_M = 124
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
M_halo_min = 10**(-4)
M_halo_max = 10**16
M_halo_array = np.logspace(-4,16,n_M)

#for loop to fill in the rows of sigma_halo_array
for i in range(n_M):
	sigma_halo_array[:,i] = sigma(PSetLin.z_array,kstart,kend,r_halo_lagrangian(M_halo_array[i]),n)

#Interpolated function for sigma of z and M
sigma_halo_interp = RectBivariateSpline(PSetLin.z_array,M_halo_array,sigma_halo_array,kx=3,ky=3)


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
	#anti_dnu_dM = (critical_density_parameter / sigma_again)**2
	return(-2*critical_density_parameter**2/sigma_again(z,M)**3 * sigma_again(z,M,dy=1))
#dark matter distribution function
def halo_distribution_function(z,M):
	return(rho_background_matter/M*f_halo_mass(z,M)*dnu_dM(z,M))


#print(dnu_dM(0,10**6),(critical_density_parameter/sigma_halo_interp(0,10**6)[0])**2,((critical_density_parameter/sigma_halo_interp(0,10**6 + 10**4)[0])**2 - (critical_density_parameter/sigma_halo_interp(0,10**6)[0])**2)/10**4)


#plt.plot(M_halo_array,dnu_dM(0,M_halo_array)[0], "r")
#plt.plot(M_halo_array,(critical_density_parameter/sigma_halo_interp(0,M_halo_array)[0,:])**2, "g")
#plt.yscale("log")
#plt.xscale("log")
#plt.show()

#print(halo_distribution_function(1,10**9))
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
	y_halo = np.sum(r_halo_mid**2 * halo_stuff.rho_halo_array[g,:] * np.sin(k*r_halo_mid)/(k*r_halo_mid)) * delta_r_halo
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


#defining integrals in Eq(5) of https://iopscience.iop.org/article/10.1086/318660/fulltext/
n_halo_integral_step = 10000

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


#print(y_halo_parameter1(0.0001,0,10**19))
#print(y_halo_parameter2(0.0001,10**19,halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step),99))
#print(new_y_halo_parameter2(0.5,1,halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step),0))
#print(y_halo_parameter2(0.5,1,halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step),0))
#sys.exit()


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

#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#test_tri = kTriangle(0.0001,0.0001,2/3*np.pi)
#print(I_03(test_tri,halo_data),I_12(test_tri,halo_data,0),I_11(test_tri,halo_data,0),transform_I_11(test_tri,halo_data,0),I_01(test_tri,halo_data,0),I_21(test_tri,halo_data,0),transform_I_21(test_tri,halo_data,0))
#sys.exit()


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

#test_triangle = kTriangle(33,33,2/3*np.pi) 
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#M_halo_mid_array = np.zeros(n_halo_integral_step)
#for i in range(n_halo_integral_step):
#	epsilon = (M_halo_max/M_halo_min)**(1/n_halo_integral_step) - 1
#	delta_M_halo = M_halo_min* (M_halo_max/M_halo_min)**(i/n_halo_integral_step)*epsilon
#	M_halo_mid = M_halo_min * (M_halo_max/M_halo_min)**(i/n_halo_integral_step) * (1 + epsilon/2)
#	M_halo_mid_array[i] = M_halo_mid
#	#I03_array[i] = ((M_halo_array[i]/rho_background_matter)**3 * halo_stuff.dn_dm_array[i] * y_halo_parameter2(myTriangle.k1,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k2,M_halo_mid,halo_stuff,i) * y_halo_parameter2(myTriangle.k3,M_halo_mid,halo_stuff,i) *delta_M_halo)*

#k_I_array = np.logspace(-2,2,100)
#I11_array = np.zeros(100)
#I03_array = np.zeros(100)
#I12_array = np.zeros(100)
#I21_array = np.zeros(100)
#double_halo_firstterm = np.zeros(100)
#double_halo_secondterm = np.zeros(100)
#double_halo_thirdterm = np.zeros(100)
#triple_halo_firstterm = np.zeros(100)
#triple_halo_secondterm = np.zeros(100)
#triple_halo_thirdterm = np.zeros(100)
#for i in range(100):
	#I11_array[i] = I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)
	#I03_array[i] = I_03(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data)
	#I12_array[i] = I_12(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)
	#I21_array[i] = I_21(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)
	#double_halo_firstterm[i] = I_12(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2) * PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k3)
	#double_halo_secondterm[i] = I_12(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1) * PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k2)
	#double_halo_thirdterm[i] = I_12(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0) * PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k1)
	#triple_halo_firstterm[i] = (2*analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),0)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2) + I_21(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2)) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k1)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k2)
	#triple_halo_secondterm[i] = (2*analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),2)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1) + I_21(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1)) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k3)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k1)
	#triple_halo_thirdterm[i] = (2*analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),1)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0) + I_21(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,0)) * I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,1)*I_11(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),halo_data,2)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k2)*PSetLin.P_interp(0,kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi).k3)
	#print(k_I_array[i],double_halo_firstterm[i],double_halo_secondterm[i],double_halo_thirdterm[i],triple_halo_firstterm[i],triple_halo_secondterm[i],triple_halo_thirdterm[i])

#analy_F_array_0 = np.zeros(100)
#analy_F_array_1 = np.zeros(100)
#analy_F_array_2 = np.zeros(100)
#for i in range (100):
#	analy_F_array_0[i] = analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),0)
#	analy_F_array_1[i] = analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),1)
#	analy_F_array_2[i] = analy_F(kTriangle(k_I_array[i],k_I_array[i],2/3*np.pi),2)
#	print(k_I_array[i],analy_F_array_0[i],analy_F_array_1[i],analy_F_array_2[i])

#plt.title("Analytic F2_0 vs k")
#plt.plot(k_I_array,analy_F_array_0)
#plt.show()
#plt.title("Analytic F2_1 vs k")
#plt.plot(k_I_array,analy_F_array_1)
#plt.show()
#plt.title("Analytic F2_2 vs k")
#plt.plot(k_I_array,analy_F_array_2)
#plt.show()
#plt.xscale("log")
##plt.yscale("log")
#plt.title("I11 vs k")
#plt.plot(k_I_array,I11_array)
#plt.show()
#plt.xscale("log")
##plt.yscale("log")
#plt.title("I03 vs k")
#plt.plot(k_I_array,I03_array)
#plt.show()
#plt.xscale("log")
##plt.yscale("log")
#plt.title("I12 vs k")
#plt.plot(k_I_array,I12_array)
#plt.show()
#plt.xscale("log")
##plt.yscale("log")
#plt.title("I21 vs k")
#plt.plot(k_I_array,I21_array)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("I03_Integrand*M vs M at k=33")
#plt.plot(M_halo_mid_array,integrand_I_03(test_triangle,halo_data))
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("I11_Integrand*M vs M at k=33")
#plt.plot(M_halo_mid_array,integrand_I_11(test_triangle,halo_data,0))
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("I12_Integrand*M vs M at k=33")
#plt.plot(M_halo_mid_array,integrand_I_12(test_triangle,halo_data,0))
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("I21_Integrand*M vs M at k=33")
#plt.plot(M_halo_mid_array,integrand_I_21(test_triangle,halo_data,0))
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_Phh_Firstterm vs k")
#plt.plot(k_I_array,double_halo_firstterm)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_Phh_secondterm vs k")
#plt.plot(k_I_array,double_halo_secondterm)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_Phh_thirdterm vs k")
#plt.plot(k_I_array,double_halo_thirdterm)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_hhh_firstterm vs k")
#plt.plot(k_I_array,triple_halo_firstterm)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_hhh_secondterm vs k")
#plt.plot(k_I_array,triple_halo_secondterm)
#plt.show()
#plt.xscale("log")
#plt.yscale("log")
#plt.title("B_hhh_thirdterm vs k")
#plt.plot(k_I_array,triple_halo_thirdterm)
#plt.show()

#tri = kTriangle(0.3,0.3,2/3*np.pi)
#halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
#print(B_matterspec(0,tri))
#print(total_halo_bispectrum(0,tri,halo_data))

#log_halo_k_array = np.logspace(-2,2,100)
#for j in range (100):
	#reduced_shear_value = reduced_shear(0,10000,1,1,log_l_array[j],0)[0,0]
	#tri = kTriangle(log_halo_k_array[i],log_halo_k_array[i],2/3*np.pi)
#	B1 = I_03(kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data)
#	B2 = double_halo_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data)[0,0]
#	B3 = triple_halo_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data)[0,0]
#	total_halo_bispectrum_value = total_halo_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data)[0,0]
#	matter_bispectrum_perturb = B_matterspec(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi))[0,0]
	#plt.scatter(log_halo_k_array[j],total_halo_bispectrum_value)
#	print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(B1)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B2)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B3)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))
	#print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))


##Replicate Stellar Mass vs Halo Mass curve from arXiv:1207.6105v2 Eq(3) and Intrinsic Parameters from Results section

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

stellar_info = parameters_stellarMvshaloM(0)

def infer_dust_mass(M_dust_measured,M_halo,r_ini):
	return(M_dust_measured * (r_halo_virial(M_halo)/r_ini)**(-1.84))

print(np.log10(M_halo_from_M_stellar(10**(10.53),stellar_info)),np.log10(infer_dust_mass(117489755.5,M_halo_from_M_stellar(10**(10.53),stellar_info),0.010/h_cosmo)))
sys.exit()
#stellar_parameters = parameters_stellarMvshaloM(0.1)
#M_halostellar_array = np.logspace(10,15,100)
#logM_stellar_array = np.zeros(100)
#for i in range(100):
#	logM_stellar_array[i] = logM_stellar(M_halostellar_array[i],stellar_parameters)
#plt.xscale("log")
#plt.yscale("log")
#plt.title("Dust Mass vs Halo Mass")
#plt.plot(4.1*10**11,4109506.229262641,"ro")
#plt.plot(M_halostellar_array,0.015*10**(logM_stellar_array))
#plt.show()
##sys.exit()

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

#calculating the amount of dust from a halo of virial radius r_virial. This is given assuming the surface density of dust is a power law and thus the volumetric density is also a power law
def Menard_value(r_halo):
	K_ext_V = 3.21689961*10**(-12) / h_cosmo #units of (Mpc*h^-1)^2/(M_solar*h^-1), values quoted in arXiv:0902.4240v1 Eq.(43) is 1.54*10^(4) cm^2/g
	r_halo_eff = 0.02 #lower integration limit since we integrate an annulus with the galaxy at the center
	Gamma_num = 1.0530 #Gamma function of power of volumetric density profile (1.84) divided by 2
	Gamma_denom = 2.1104 #Gamma function of power of volumetric density profile (1.84) minus 1, then divided by 2
	Menardvalue = 0
	delta_r_halo = (r_halo - r_halo_eff)/n
	r_halo_mid = np.linspace(0.5*delta_r_halo,(n-1/2)*delta_r_halo,n)
	Menardvalue = np.sum(4.14*10**(-3) * 0.1**(0.84) * (r_halo_mid)**(-1.84) * r_halo_mid**2) * delta_r_halo
	#return(Menardvalue)	
	return(Gamma_num/(Gamma_denom*np.sqrt(np.pi))*4*np.pi*np.log(10)/(2.5*K_ext_V)*Menardvalue)

#print(Menard_value(0.177))
#sys.exit()

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


#print(u_dust_halo_parameter(0.01,10**8,stellar_info),u_dust_halo_parameter(0.01,10**10,stellar_info),u_dust_halo_parameter(0.01,10**12,stellar_info),M_dust_optimistic(10**8,stellar_info)/10**8,M_dust_optimistic(10**10,stellar_info)/10**10,M_dust_optimistic(10**12,stellar_info)/10**12)
#print(r_halo_virial(4.1*10**11))
#print(4/3 * np.pi * (.177)**3 * 200 * rho_background_matter,M_dust_optimistic(4/3 * np.pi * (.177)**3 * 200 * rho_background_matter,stellar_info))
sys.exit()
def rho_bar_dust(z,stellarstuff):
	rhobardust = 0
	delta_M_halo = M_halo_max/n
	M_halo_mid = np.linspace(0.5*delta_M_halo,(n-1/2)*delta_M_halo,n)
	rhobardust = np.sum(M_dust_optimistic(M_halo_mid,stellarstuff) * halo_distribution_function(z,M_halo_mid)) * delta_M_halo
	return(rhobardust)

#class profile_function(object):
#	def __init__(in1,in2,in3,self):
#		#self.myTriangle = in1
#		self.M_halo = in1
#		#self.halo_stuff = in3
#		#self.stellarstuff = in4
#		self.index_halo = in2
#		self.index_which_k_leg = in3
#		#self.index_which_function = in7
#
#		k1 = myTriangle.k1
#		y_halo_parameter2(k1,self.M_halo,halo_stuff,self.index_halo)
#
#		if self.index_which_k_leg == 1:
#			k1 = myTriangle.k2
#			y_halo_parameter2(k1,self.M_halo,halo_stuff,self.index_halo)
#
#		if self.index_which_k_leg == 2:
#			k1 = myTriangle.k3
#			u_dust_halo_parameter(k1,self.M_halo,stellarstuff)
#
#def profile_function(index):
#	profile_func = 
#
#
test_tri = kTriangle(0.01,0.01,2/3*np.pi)

#print(profile_function(10**10,0,0),profile_function(10**10,0,1),profile_function(10**10,0,2))
#sys.exit()


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
#print(I_12_dust(test_tri,halo_data,stellar_info,0),I_12_dust(test_tri,halo_data,stellar_info,1),I_12_dust(test_tri,halo_data,stellar_info,2))
#sys.exit()

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



stellar_info = parameters_stellarMvshaloM(0)
halo_data = halo_info(0,M_halo_min,M_halo_max,n_halo_integral_step)
print(rho_bar_dust(0,stellar_info))
sys.exit()
log_halo_k_array = np.logspace(-2,2,100)
for j in range (100):
	#reduced_shear_value = reduced_shear(0,10000,1,1,log_l_array[j],0)[0,0]
	#tri = kTriangle(log_halo_k_array[i],log_halo_k_array[i],2/3*np.pi)
	B1_dust = I_03_dust(kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data,stellar_info)
	B2_dust = double_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data,stellar_info)[0,0]
	B3_dust = triple_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data,stellar_info)[0,0]
	total_halo_dust_bispectrum_value = total_halo_dust_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data,stellar_info)[0,0]
	total_halo_bispectrum_value = total_halo_bispectrum(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi),halo_data)[0,0]
	matter_bispectrum_perturb = B_matterspec(0,kTriangle(log_halo_k_array[j],log_halo_k_array[j],2/3*np.pi))[0,0]
	#plt.scatter(log_halo_k_array[j],total_halo_bispectrum_value)
#	print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(B1)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B2)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B3)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))
	print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(B1_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B2_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(B3_dust)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_dust_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))
	#print(log_halo_k_array[j],(log_halo_k_array[j])**3/(2*np.pi**2)*(total_halo_bispectrum_value)**(1/2),(log_halo_k_array[j])**3/(2*np.pi**2)*(matter_bispectrum_perturb)**(1/2))

#print(rho_bar_dust(0,stellar_info),(0.01)**3/(2*np.pi**2)*total_halo_dust_bispectrum(0,test_tri,halo_data,stellar_info)**(1/2),(0.01)**3/(2*np.pi**2)*total_halo_bispectrum(0,test_tri,halo_data)**(1/2))

#testing

#print(I_03(test_tri,halo_data),I_03_dust(test_tri,halo_data,stellar_info),I_11(test_tri,halo_data,0),I_11_dust(test_tri,halo_data,stellar_info,0),I_12(test_tri,halo_data,0),I_12_dust(test_tri,halo_data,stellar_info,0),I_21(test_tri,halo_data,0),I_21_dust(test_tri,halo_data,stellar_info,0))