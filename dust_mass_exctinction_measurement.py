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