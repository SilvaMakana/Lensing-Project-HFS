global H0
global h_cosmo
global Omega_k
global Omega_r
global Omega_b
global Omega_CDM
global Omega_0
global Omega_L
global Omega_m
global n_s_primordial
global d_h
global Tempfactor_CMB
global rho_critial


## kstart is the first k value from the CLASS output files in units of h^-1Mpc ##
kstart = 1.045e-5
kend = 210.674
##Integration step size##
n=10000

#Cosmological parameters
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



