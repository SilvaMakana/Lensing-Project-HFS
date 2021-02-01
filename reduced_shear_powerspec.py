##Building reduced shear model with Matter Bispectrum model from perturbation theory
z_step = 80
l_tripleprime_step = 160
phi_step = 160
def reduced_shear(z_ini,l_tripleprime_max,z_alpha,z_beta,l_mag,l_phi):
	sigma_galaxy = np.pi * r_virial**2
	shear = 0
	D_alpha = distance(z_ini,z_alpha)
	D_beta = distance(z_ini,z_beta)
	#redshift integral from 0 to Chi(z_alpha)
	for i in range (z_step):
		delta_z = (z_alpha-z_ini)/z_step
		zi = (z_ini + i*delta_z)
		zmid = 1/2*(zi + z_ini + (i+1)*delta_z)
		D_mid = distance(z_ini,zmid)
		#l_tripleprime magnitude integral from 0 to some max
		for j in range (l_tripleprime_step):
			delta_l_tripleprime = (l_tripleprime_max)/l_tripleprime_step
			l_tripleprime_j = j*delta_l_tripleprime
			l_tripleprime_mid = 1/2*(l_tripleprime_j + (j+1)*delta_l_tripleprime)
			#angular integral for l_tripleprime from 0 to pi but this is FOR THE SPECIAL CASE l_phi = 0 rad
			for k in range (phi_step):
				delta_phi = np.pi/phi_step
				phi_k = k*delta_phi
				phi_mid = 1/2*(phi_k + (k+1)*delta_phi)
				shear += 2*window_distance(D_mid,D_alpha) * window_distance(D_mid,D_beta) * sigma_galaxy*numberdensity_galaxy*tau_g(zmid)*(1+zmid)**(2)*d_h/np.sqrt(Omega_r*(1+zmid)**4 + Omega_m*(1+zmid)**3 + Omega_k*(1+zmid)**2 + Omega_L) * np.cos(2*l_phi - 2*phi_mid) * (9*Omega_m**2*d_h**(-4))/(4 *1/(1+zmid)**2) * B_matterspec(zmid,kTriangle(l_mag/D_mid,l_tripleprime_mid/D_mid,l_phi - phi_mid)) * 1/(2*np.pi)**2 * delta_z * delta_phi * l_tripleprime_mid * delta_l_tripleprime
	return (shear)