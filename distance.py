##Defining cosmological distance integrator as a function of an inital redshift to a final redshift
def distance(z_ini,z_f):
	Chi = 0
	for i in range (n):
		delta_z = (z_f-z_ini)/n
		zi = (z_ini + i*delta_z)
		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
		Chi += d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * delta_z
	return Chi

##Defining Window Function
def window_distance(distance2, distance1):
	if (distance1 >= distance2):
		return (1/distance2 - 1/distance1)
	else:
		return(0)

#WORK