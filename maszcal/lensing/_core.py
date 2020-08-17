from functools import partial
from dataclasses import dataclass
import numpy as np
from astropy import units as u
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.concentration


@dataclass
class Miscentering:
    rho_func: object
    misc_distrib: object
    miscentering_func: object

    def _misc_rho(self, radii, misc_dist_params, rho_params):
        return self.miscentering_func(
            radii,
            lambda r: self.rho_func(r, *rho_params),
            lambda r: self.misc_distrib(r, *misc_dist_params),
        )

    def rho(self, radii, misc_params, rho_params):
        prob_centered = misc_params[-1]
        return (prob_centered*self.rho_func(radii, *rho_params)
                + (1-prob_centered)*self._misc_rho(radii, misc_params[:-1], rho_params))


@dataclass
class Shear:
    rho_func: object
    units: u.Quantity
    esd_func: object

    def delta_sigma_total(self, rs, zs, mus, *rho_params):
        return self.esd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
        ) * (u.Msun/u.Mpc**2).to(self.units)


@dataclass
class Convergence:
    CMB_REDSHIFT = 1100

    rho_func: object
    cosmo_params: maszcal.cosmology.CosmoParams
    comoving: bool
    units: u.Quantity
    sd_func: object

    def __post_init__(self):
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, comoving=self.comoving, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def kappa(self, rs, zs, mus, *rho_params):
        return self.sd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
        ) * (u.Msun/u.Mpc**2).to(self.units) / self.sigma_crit(z_lens=zs)[None, :, None]


@dataclass
class MatchingModel:
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    rho_func: object
    units: u.Quantity = u.Msun/u.pc**2

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]
