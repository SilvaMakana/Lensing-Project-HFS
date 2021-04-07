##Defining cosmological distance integrator as a function of an inital redshift to a final redshift
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 

#comoving distance radial distance as a function of redshift in units of Mpc*h^-1
def distance(z_ini,z_f):
	Chi = 0
	delta_z = (z_f-z_ini)/n
	#for i in range (n):
		#zi = (z_ini + i*delta_z)
		#zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
	zmid = np.linspace(0.5*delta_z,(n-1/2)*delta_z,n)
	Chi = np.sum(d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L)) * delta_z
	return Chi

##Defining Window Function
#def window_distance(distance2, distance1):
#	if (distance1 >= distance2):
#		return(1/distance2 - 1/distance1)
#	else:
#		return(0)

def window_distance(distance2, distance1):
	return(np.where(distance1 >= distance2,1/distance2 - 1/distance1,np.zeros_like(distance2)))


#def window_distance(distance2, distance1):
#	window = any(1/distance2 - 1/distance1)
#	return(window)


#WORK