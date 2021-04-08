##Building Power Spectra from CLASS data output
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 
import subprocess
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


k_min = 1.045e-5 #starting k value in CLASS data files
k_max = 210.674 #final k value in CLASS data files
n_k = 625 #number of k values in CLASS files
n_z = 31 #number of redshift values

#dir_base = "/Users/makanas/class_public-2.8.2/output/LENSING_PROJECT" #directory on home laptop where LENSING_PROJECT06_z... is located
#dir_base = subprocess.check_output('cat local.config', shell=True).strip()
with open('local.config', 'r') as file:
	dir_base = file.read().replace('\n', '')
name_base = dir_base+"LENSING_PROJECT06_z"
name_endNL = "_pk_nl.dat"
name_endLin = "_pk.dat"

PSetNL = PowerSpectrumMultiZ(name_base,name_endNL,n_z,k_min,k_max,n_k)
PSetLin = PowerSpectrumMultiZ(name_base,name_endLin,n_z,k_min,k_max,n_k)