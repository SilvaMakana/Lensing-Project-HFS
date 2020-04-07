import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
kstart = 1.045e-5
kend = 33.308
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

#Define numerical integration as a for loop
#for i in range (n):
#	delta_x = (end-start)/n
#	xi = (start + i*delta_x)
#	xmid = 1/2*(xi + start + (i+1)*delta_x)
#	Ai += (xmid)**2 * delta_x 
#print(xmid,Ai)

#Defining radial distance from redshift integrator
H0 = 67.37
h_cosmo = H0/100
#Omega_k = 0
#Omega_r = 0
Omega_b = 0.02233/h_cosmo**2
Omega_CDM = 0.1198/h_cosmo**2
Omega_0 = (Omega_b + Omega_CDM)/h_cosmo**2
#Omega_L = 0.6911
n_s_primordial = 0.963
#d_h = 3000
Tempfactor_CMB = 1.00
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
			self.k3 = np.sqrt(self.k1**2 + self.k2**2 + 2*self.k1*self.k2*self.cos12)

		elif input == "SSS":
			self.k1 = in1
			self.k2 = in2
			self.k3 = in3
			self.cos12 = (self.k1**2 + self.k2**2 + self.k3**2)/(2*self.k1*self.k2)
		
		
		self.cos13 = (self.k1**2 - self.k3**2 - self.k2**2)/(2*self.k1*self.k3)
		self.cos23 = (self.k2**2 - self.k3**2 - self.k1**2)/(2*self.k2*self.k3)

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
#plt.show()
#sys.exit()
#for i in range(0,3):
#	for j in range(0,10):
#		print(i,j,PSetNL.P_interp(i,j))

##Defining the functions from arXiv:astro-ph/9709112 Eq. 26-31##

def s_transfer(Omega_b):
	return(44.5*np.log(9.83/(Omega_0*h_cosmo**2))/(1 + 10*(Omega_b*h_cosmo**2)**0.75))

def alpha_Gamma(Omega_b):
	return(1 - 0.328*np.log(431*Omega_0*h_cosmo**2)*Omega_b/Omega_0 + 0.38*np.log(22.3*Omega_0*h_cosmo**2)*(Omega_b/Omega_0)**2)

def Gamma_transfer(k):
	return(Omega_0*h_cosmo*(alpha_Gamma(Omega_b) + (1-alpha_Gamma(Omega_b))/(1+(0.43*k*s_transfer(Omega_b))**4)))

def q_transfer(k):
	return(k*Tempfactor_CMB**2/Gamma_transfer(k))

def L_transfer(k):
	return(np.log(2*np.e + 1.8*q_transfer(k)))

def C_transfer(k):
	return(14.2 + 731/(1+62.5*q_transfer(k)))

def Transfer(k):
	return(L_transfer(k)/(L_transfer(k) + C_transfer(k)*q_transfer(k)**2))


##defining spectral_n(z,k) such that n does NOT produce wiggles from the equations in arXiv:astro-ph/9709112, so we are smoothing it out##
def spectral_n_nowiggle(k):
	return(n_s_primordial + 1/(2*0.01)*np.log(np.exp(0.01)*Transfer(k)/(np.exp(-0.01)*Transfer(k)))) 


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
	return ((4 - 2**spectral_n_nowiggle(k))/(1 + 2**(spectral_n_nowiggle(k) + 1)))
def perturb_a(z,k):
	return (1 + sigma_8(z,kstart,kend,n)**a6 * (0.7*Q3(z,k))**0.5 * (q_nonlin(z,k)*a1)**(spectral_n_nowiggle(k) + a2))/ (1 + (q_nonlin(z,k)*a1)**(spectral_n_nowiggle(k) + a2))
def perturb_b(z,k):
	return (1 + 0.2*a3*(spectral_n_nowiggle(k)+3)*q_nonlin(z,k)**(spectral_n_nowiggle(k)+3))/(1 + q_nonlin(z,k)**(spectral_n_nowiggle(k)+3.5))
def perturb_c(z,k):
	return ((1 + 4.5*a4/((1.5 + (spectral_n_nowiggle(k) +3)**4))*(q_nonlin(z,k)*a5)**(spectral_n_nowiggle(k) + 3)))/(1 + (q_nonlin(z,k)*a5)**(spectral_n_nowiggle(k) + 3.5))

#print(Q3(1.0,0.08),perturb_a(1.0,0.08),perturb_b(1.0,0.08),perturb_c(1.0,0.08))
#sys.exit()
def perturb_F(z,myTriangle,i):
	k1 = myTriangle.k1; k2 = myTriangle.k2; cos12 = myTriangle.cos12
	if i==1:
		k1 = myTriangle.k2; k2 = myTriangle.k3; cos12 = myTriangle.cos23
	if i==2:
		k1 = myTriangle.k3; k2 = myTriangle.k1; cos12 = myTriangle.cos13
	return (5/7*perturb_a(z,k1)*perturb_a(z,k2)+1/2*cos12*(k1/k2+k2/k1)*perturb_b(z,k1)*perturb_b(z,k2)+2/7*cos12**2*perturb_c(z,k1)*perturb_c(z,k2))
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
z=1.0
k_input = np.logspace(-1.52287874528,-0.69897000433,100)
for i in range (100):
	tri = kTriangle(k_input[i],2*k_input[i],0.6*np.pi)
        #print(tri.k1)
	plt.scatter(k_input[i],Q123(z,tri))
plt.show()




#print(PSetNL.spectral_n(0.3,10))
#plt.semilogx(PSetNL.k_array,PSetNL.spectral_n(0.3,PSetNL.k_array).T)
#plt.semilogx(PSetLin.k_array,PSetLin.spectral_n(0.3,PSetLin.k_array).T)
#plt.show()

