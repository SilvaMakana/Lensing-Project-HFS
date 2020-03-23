import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline

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

#Define numerical integration as a for loop
#for i in range (n):
#	delta_x = (end-start)/n
#	xi = (start + i*delta_x)
#	xmid = 1/2*(xi + start + (i+1)*delta_x)
#	Ai += (xmid)**2 * delta_x 
#print(xmid,Ai)

#Defining radial distance from redshift integrator
#Omega_k = 0
#Omega_r = 0
#Omega_m = 0.3089
#Omega_L = 0.6911
#d_h = 3000
#start = 0
#end = 3
#Chi = 0
#n = 10000
#def distance (Omega_m,Omega_L,Omega_r,Omega_k,d_h,z_ini,z_f,n):
#	Chi = 0
#	for i in range (n):
#		delta_z = (z_f-z_ini)/n
#		zi = (z_ini + i*delta_z)
#		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
#		Chi += d_h/numpy.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
#	return Chi
#print(distance (0.3089,0.6911,0,0,3000,0,0.1,10000))

#Defining Window Function
#Chi1 = 10
#Chi2 = 2
#if (Chi1 >= Chi2):
#	W = 1/Chi2 - 1/Chi1
#else:
#	W = 0
#print(W)

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


class kTriangle(object):
	def __init__(self,in1,in2,in3,input="SAS"):
		"""if using SAS, then, in1=length of k1, in2=length of k2, in3 = angle in radians between k1 and k2"""
		"""if using SSS, then, in1=length of k1, in2=length of k2, in3 = length of k3"""
		if input == "SAS":
			self.k1 = in1
			self.k2 = in2
			self.cos12 = np.cos(in3)
			self.k3 = np.sqrt(self.k1**2 + self.k2**2 - 2*self.k1*self.k2*self.cos12)

		elif input == "SSS":
			self.k1 = in1
			self.k2 = in2
			self.k3 = in3
			self.cos12 = 1/2(self.k1**2 + self.k2**2 - self.k3**2)/self.k1*self.k2
	
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
k_max = 33.308
n_k = 617
n_z = 31

dir_base = "/Users/makanas/class_public-2.8.2/output/LENSING_PROJECT/"
name_base = dir_base+"LENSING_PROJECT00_z"
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
#
#plt.show()
#for i in range(0,3):
#	for j in range(0,10):
#		print(i,j,PSetNL.P_interp(i,j))

#defining non-linear k scale (h Mpc^-1) from arXiv:1111.4477v2
def scale_nonlin(z,k):
	return k**3 * PSetLin.P_interp(z,k)/(2*np.pi**2) - 1
def k_nonlin(z):
	return optimize.root_scalar(lambda k: scale_nonlin(z,k),bracket=[0,33],method ='brentq')

#print(k_nonlin(1.5))
def q_nonlin(z,k):
	return k/k_nonlin(z)

#defining constants for parameter functions from arXiv:1111.4477v2
a1 = 0.25
a2 = 3.5
a3 = 2
a4 = 1
a5 = 2
a6 = -0.2

#defining Window function for sigma_8
R = 8.0 #in units of h^-1 MpC
def window(k):
	return 3/(R**3)*(np.sin(k*R) - k*R*np.cos(k*R))/k**3


def sigma_8(z,kstart,kend,n): #Sigma_8 cosmological function of redshift, z
	Ai = 0
	for i in range (n):
		delta_k = (kend-kstart)/n
		#k_i = (kstart + i*delta_k)
		#thetamid = 1/2*(theta_i + starttheta + (i+1)*delta_theta)
		kmid = kstart + (2*i + 1)/2 * delta_k
		Ai += kmid**2*PSetLin.P_interp(z,kmid)[0,0]*window(kmid)**2
		#if i<1000:
			#print('{:4d} {:11.5e} {:11.5e} {:11.5e} {:11.5e}'.format(i,kmid,PSetLin.P_interp(z,kmid)[0,0],window(kmid),Ai))
	return np.sqrt(Ai/(2*np.pi**2)*delta_k)

#defining parameter functions from arXiv:1111.4477v2
def Q3(z,k):
	return (4 - 2**spectral_n(z,k))/(1 + 2**(spectral_n(z,k) + 1))
def a(z,k,kstart,kend,n):
	return (1 + sigma_8(z,kstart,kend,n)**a6 * (0.7*Q3(z,k))**0.5 * (q_nonlin(z,k)*a1)**(spectral_n(z,k) + a2))/ (1 + (q_nonlin(z,k)*a1)**(spectral_n(z,k) + a2))
def b(z,k):
	return (1 + 0.2*a3*(spectral_n(z,k)+3)*q_nonlin(z,k)**(spectral_n(z,k)+3))/(1 + q_nonlin(z,k)**(spectral_n(z,k)+3.5))
def c(z,k):
	return (1 + 4.5*a4/((1.5 + (spectral_n(z,k) +3)**4)*(q_nonlin(z,k)*a5)**(spectral_n(z,k) +3)))/(1 + (q_nonlin(z,k)*a5)**(spectral_n(z,k) +3.5))

def F(z,k1,k2):
	return (5/7*a(z,k1)*a(z,k2)+1/2*cos12*(k1/k2+k2/k1)*b(z,k1)*b(z,k2)+2/7*cos12**2*c(z,k1)*c(z,k2))

#print(PSetNL.spectral_n(0.3,10))
#plt.semilogx(PSetNL.k_array,PSetNL.spectral_n(0.3,PSetNL.k_array).T)
#plt.semilogx(PSetLin.k_array,PSetLin.spectral_n(0.3,PSetLin.k_array).T)
#plt.show()

