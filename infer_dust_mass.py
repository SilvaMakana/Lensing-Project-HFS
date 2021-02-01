#scaling measured dust mass at some smaller radius than the virial radius according to the power law derived for the volumetric density of dust
def infer_dust_mass(M_dust_measured,M_halo,r_ini):
	return(M_dust_measured * (r_halo_virial(M_halo)/r_ini)**(-1.84))