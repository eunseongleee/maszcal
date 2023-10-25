import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import maszcal.cosmology
import maszcal.density
import maszcal.twohalo

import time

path = '/Users/eunseonglee/advact/'
cat = 'DR5_cluster-catalog_v1.1.fits'

SNRcut = 5.5
zmin = 0.15
zmax = 0.7

cosmo = maszcal.cosmology.CosmoParams()
delta = 200
mdef = 'mean'

radii = np.logspace(-1, 1, 5)
a_2h = 1
con = 3
alpha = 0.88 
beta = 3.8 


def read_cat(cat, SNRcut, zmin, zmax):

    list = fits.open(path+cat)
    data = list[1].data
    
    z = data.field("redshift")
    M = data.field("M200m") # should be consistent with delta and mdef
    SNR = data.field("fixed_SNR")
    DES = data.field("footprint_DESY3")

    ind = np.where((SNR > SNRcut) & (z > zmin) & (z < zmax) & (DES == 1))[0]
    
    print("Number of chosen clusters :", len(ind))

    return z[ind], M[ind] * 1e14


def get_Gnfw_model():
    return maszcal.density.SingleMassGnfw( 
        cosmo_params = cosmo,
        delta = delta,
        mass_definition = mdef,
        comoving_radii = True,
    )


def get_Nfw_model():
    return maszcal.density.NfwModel(
        cosmo_params = cosmo,
        delta = delta,
        mass_definition = mdef,
        comoving = True,
    )  


def get_one_halo_sd(radii, redshift, mass, con, alpha, beta, include_baryon=True):

    mu = np.log(mass)  
    con = np.array([con])
    alpha = np.array([alpha])
    beta = np.array([beta])
    gamma = np.array([0.2]) # fixed   
    
    if include_baryon:        
        rho_model = get_Gnfw_model()
        one_halo = rho_model.surface_density(radii, redshift, mu, con, alpha, beta, gamma).squeeze()      
        
    else:         
        rho_model = get_Nfw_model()
        one_halo = rho_model.surface_density(radii, redshift, mass, con).squeeze()   
        
    return one_halo 


def get_two_halo_sd():
    model = maszcal.twohalo.TwoHaloShearModel(
        cosmo_params = cosmo,
        mass_definition = mdef,
        delta = delta,
    )
    return model.surface_density


def get_two_halo_emulator(two_halo_sd):
    return maszcal.twohalo.TwoHaloEmulator.from_function(
        two_halo_func = two_halo_sd,
        r_grid = np.geomspace(0.01, 100, 120),
        z_lims = np.array([0, 1.2]),
        mu_lims = np.log(np.array([1e13, 1e15])),
        num_emulator_samples = 800,
    )


def get_surface_density(radii, redshift, mass, a_2h, con, alpha, beta, emulator, include_baryon=True):

    redshift_arr = np.array([redshift])
    mass_arr = np.array([mass])
    mu = np.log(mass) 
    a_2h = np.array([a_2h])  
 
    one_halo = get_one_halo_sd(radii, redshift_arr, mass_arr, con, alpha, beta, include_baryon=True).T
    two_halo = a_2h * emulator(radii, redshift, mu).squeeze()    

    return np.where(one_halo > two_halo, one_halo, two_halo)


def get_sd_array():

    print("Reading in catalogue")
    
    zs, Ms = read_cat(cat, SNRcut, zmin, zmax)
 
    start = time.time()
   
    two_halo_sd = get_two_halo_sd()
    emulator = get_two_halo_emulator(two_halo_sd)

    end = time.time() - start
    print("Setting up an emulator for two halo term takes", end)

    
    start = time.time()

    sd_arr = get_surface_density(radii, zs, Ms, a_2h, con, alpha, beta, emulator, include_baryon=True)
    
    end = time.time() - start
    print("Getting surface density profiles for clusters takes", end)

    return sd_arr





sigma = get_sd_array()

print(np.shape(sigma))



