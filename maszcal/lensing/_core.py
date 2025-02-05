from functools import partial
from types import MappingProxyType
from dataclasses import dataclass
import numpy as np
from astropy import units as u
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.concentration
import maszcal.stats


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
        rhos_centered = self.rho_func(radii, *rho_params)
        rhos_misc = self._misc_rho(radii, misc_params[:-1], rho_params)
        prob_centered = maszcal.mathutils.atleast_kd(misc_params[-1], rhos_misc.ndim, append_dims=False)
        return prob_centered*rhos_centered + (1-prob_centered)*rhos_misc


@dataclass
class Shear:
    rho_func: object
    units: u.Quantity
    esd_func: object
    sd_func: object
    esd_kwargs: MappingProxyType
    sd_kwargs: MappingProxyType

    def excess_surface_density(self, rs, zs, mus, *rho_params):
        return self.esd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
            **self.esd_kwargs,
        ) * (u.Msun/u.Mpc**2).to(self.units)


    def surface_density(self, rs, zs, mus, *rho_params):
        return self.sd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
            **self.sd_kwargs,
        ) * (u.Msun/u.Mpc**2).to(self.units)


@dataclass
class Convergence:
    CMB_REDSHIFT = 1100

    rho_func: object
    cosmo_params: maszcal.cosmology.CosmoParams
    comoving: bool
    units: u.Quantity
    sd_func: object
    sd_kwargs: MappingProxyType

    def __post_init__(self):
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, comoving=self.comoving, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def convergence(self, rs, zs, mus, *rho_params, **kwargs):
        sds =  self.sd_func(
            rs,
            lambda r: self.rho_func(r, zs, mus, *rho_params),
            **self.sd_kwargs,
        ) * (u.Msun/u.Mpc**2).to(self.units)
        sd_crits =  maszcal.mathutils.atleast_kd(
            self.sigma_crit(z_lens=zs)[:, None],
            sds.ndim,
            append_dims=False,
        )
        return sds/sd_crits



@dataclass
class MatchingModel:
    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    lensing_func: object
    units: u.Quantity = u.Msun/u.pc**2

    def normed_lensing_weights(self, a_szs):
        return np.repeat(
            self.lensing_weights/self.lensing_weights.sum(),
            a_szs.size,
        )

    def mu_from_sz_mu(self, sz_mu, a_sz):
        return sz_mu[:, None] - a_sz[None, :]



@dataclass
class ScatteredMatchingModel:
    B_SZ = np.ones(1)
    SZ_SCATTER = np.array([0.2])

    sz_masses: np.ndarray
    redshifts: np.ndarray
    lensing_weights: np.ndarray
    lensing_func: object
    logmass_prob_dist_func: object
    num_mu_bins: int = 64
    units: u.Quantity = u.Msun/u.pc**2

    def normed_lensing_weights(self, a_szs):
        return self.lensing_weights/self.lensing_weights.sum()

    def prob_musz_given_mu(self, mus, mu_szs, a_szs):
        return maszcal.stats.SzMassDistributions.lognormal_dist(mus, mu_szs, a_szs, self.B_SZ, self.SZ_SCATTER).squeeze(
            axis=(3, 4),
        )
