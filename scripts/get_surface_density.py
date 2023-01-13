import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import maszcal.cosmology
import maszcal.density

path = '/Users/eunseonglee/advact/'
cat = 'DR5_cluster-catalog_v1.1.fits'

SNRcut = 5.5
zmin = 0.15
zmax = 0.7

cosmo = maszcal.cosmology.CosmoParams()
delta = 200
mdef = 'mean'

radii = np.logspace(-1, 1, 5)
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


def get_surface_density(radii, redshift, mass, con, alpha, beta, include_baryon=True):
    
    redshift = np.array([redshift])
    mass = np.array([mass])
    mu = np.log(mass) 
    
    con = np.array([con])
    alpha = np.array([alpha])
    beta = np.array([beta])
    gamma = np.array([0.2]) # fixed 
    
    if include_baryon:        
        rho_model = get_Gnfw_model()
        sd = rho_model.surface_density(radii, redshift, mu, con, alpha, beta, gamma)
        
    else:         
        rho_model = get_Nfw_model()
        sd = rho_model.surface_density(radii, redshift, mass, con)    
    
    return sd


def get_sd_array():

    print("Reading in catalogue")
    
    zs, Ms = read_cat(cat, SNRcut, zmin, zmax)
    
    print("Getting a surface density profile for each cluster") 

    sd_arr = []    
    for i in range(len(zs)):
        sd_arr.append(get_surface_density(radii, zs[i], Ms[i], con, alpha, beta, include_baryon=True))

    return np.squeeze(np.array(sd_arr))


sigma = get_sd_array()

print(np.shape(sigma))



