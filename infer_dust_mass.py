#scaling measured dust mass at some smaller radius than the virial radius according to the power law derived for the volumetric density of dust
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from CLASS_matter_powerspec import *
from halo_model_bispec import *
from global_variables import * 

#root equation to find the halo mass given a stellar mass, since the Behroozi et al. paper gives the stellar mass as a function of halo mass
def root_M_halo_from_M_stellar(M_halo,M_stellar,stellarstuff):
	return(10**(logM_stellar(M_halo,stellarstuff)) - M_stellar)

#solving the root equation above, the limits of the halo mass range are from arXiv:1207.6105v2 in Fig. 7 the M_halo axis
def M_halo_from_M_stellar(M_stellar,stellarstuff):
	M_min_Behroozi = 10**10
	M_max_Behroozi = 10**15
	something = optimize.root_scalar(lambda M_halo: root_M_halo_from_M_stellar(M_halo,M_stellar,stellarstuff),bracket=[M_min_Behroozi,M_max_Behroozi],method ='brentq')
	return(something.root)

def infer_dust_mass(M_dust_measured,M_halo,r_ini):
	return(M_dust_measured * (r_halo_virial(M_halo)/r_ini)**(-1.84))