from types import MappingProxyType
from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.cosmology
import maszcal.lensing
import maszcal.mathutils
from . import _nfw


class BaryonDensity:
    def _init_shear_wrapper(self):
        self._shear_wrapper = maszcal.lensing.Shear(
            rho_func=self.rho_tot,
            units=self.units,
            esd_func=self.esd_func,
            esd_kwargs=self.esd_kwargs,
            sd_func=self.sd_func,
            sd_kwargs=self.sd_kwargs,
        )

    def _init_convergence_wrapper(self):
        self._convergence_wrapper = maszcal.lensing.Convergence(
            rho_func=self.rho_tot,
            cosmo_params=self.cosmo_params,
            comoving=self.comoving_radii,
            units=self.units,
            sd_func=self.sd_func,
            sd_kwargs=self.sd_kwargs,
        )

    def surface_density(self, rs, zs, mus, *rho_params):
        return self._shear_wrapper.surface_density(rs, zs, mus, *rho_params)

    def excess_surface_density(self, rs, zs, mus, *rho_params):
        return self._shear_wrapper.excess_surface_density(rs, zs, mus, *rho_params)

    def convergence(self, rs, zs, mus, *rho_params):
        return self._convergence_wrapper.convergence(rs, zs, mus, *rho_params)


@dataclass
class _Gnfw(BaryonDensity):
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    NUM_INTEGRATION_RADII = 200

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    comoving_radii: bool

    def _init_nfw(self):
        self.nfw_model = self.nfw_class(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def __post_init__(self):
        self.baryon_frac = self.cosmo_params.omega_bary/self.cosmo_params.omega_matter
        self._init_nfw()
        self._init_shear_wrapper()
        self._init_convergence_wrapper()

    def mass_from_mu(self, mu):
        return np.exp(mu)

    def _r_delta(self, zs, mus):
        '''
        SHAPE mu, z
        '''
        masses = self.mass_from_mu(mus)
        return self.nfw_model.radius_delta(zs, masses)

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        ys = (rs[..., None, :]
              / maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+1, append_dims=False)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = maszcal.mathutils.atleast_kd(alphas, rs.ndim+1, append_dims=False)
        betas = maszcal.mathutils.atleast_kd(betas, rs.ndim+1, append_dims=False)
        gammas = maszcal.mathutils.atleast_kd(gammas, rs.ndim+1, append_dims=False)

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rsflat = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.baryon_norm_radius,
            self.NUM_INTEGRATION_RADII,
        )
        rs = rsflat[:, None]

        drs = np.gradient(rsflat)
        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[..., None, :, None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=0)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
        norm = maszcal.mathutils.atleast_kd(norm, rs.ndim+norm.ndim-1, append_dims=False)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def _rho_nfw(self, rs, zs, mus, cons):
        masses = self.mass_from_mu(mus)

        return self.nfw_model.rho(rs, zs, masses, cons)

    def rho_bary(self, rs, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, zs, mus, cons, alphas, betas, gammas)

    def rho_cdm(self, rs, zs, mus, cons):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, zs, mus, cons)

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self.rho_cdm(rs, zs, mus, cons)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm


@dataclass
class Gnfw(_Gnfw):
    nfw_class: object = _nfw.NfwModel
    units: u.Quantity = u.Msun/u.pc**2
    sd_func: object = projector.SurfaceDensity.calculate
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    esd_func: object = projector.ExcessSurfaceDensity.calculate
    sd_func: object = projector.SurfaceDensity.calculate
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    baryon_norm_radius: float = 3.3


class _SingleMassGnfw(_Gnfw):
    def _r_delta(self, zs, mus):
        '''
        SHAPE z, params
        '''
        masses = self.mass_from_mu(mus)
        return self.nfw_model.radius_delta(zs, masses).T

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE rs.shape, z, params
        '''

        # when mass and redshift input is an array, denominator is needed to be transposed, but this also works for non-array case
        ys = (rs[..., None]/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim+1, append_dims=False).T) / self.CORE_RADIUS

        alphas = maszcal.mathutils.atleast_kd(alphas, rs.ndim+1, append_dims=False)
        betas = maszcal.mathutils.atleast_kd(betas, rs.ndim+1, append_dims=False)
        gammas = maszcal.mathutils.atleast_kd(gammas, rs.ndim+1, append_dims=False)

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE z, params
        '''
        rsflat = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.baryon_norm_radius,
            self.NUM_INTEGRATION_RADII,
        )
        rs = rsflat[:, None]
        drs = np.gradient(rsflat)
        top_numerator = self._rho_nfw(rs, zs, mus, cons)

        # when mass and redshift input is an array
        if np.ndim(zs) != 1:
            top_numerator = top_numerator[:, 0, ...]

        top_integrand = top_numerator * rs[..., None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=0)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):

        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)[None, ...]
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)

        # when mass and redshift input is an array
        if np.ndim(zs) != 1:
            norm = norm[None, ...]
            profile_shape = profile_shape[..., None, :]

        return norm * profile_shape

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self.rho_cdm(rs, zs, mus, cons)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm


@dataclass
class SingleMassGnfw(_SingleMassGnfw):
    nfw_class: object = _nfw.SingleMassNfwModel
    units: u.Quantity = u.Msun/u.pc**2
    sd_func: object = projector.SurfaceDensity.calculate
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    esd_func: object = projector.ExcessSurfaceDensity.calculate
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -1})
    baryon_norm_radius: float = 3.3


@dataclass
class _CmGnfw(_Gnfw):
    CORE_RADIUS = 0.5
    MIN_INTEGRATION_RADIUS = 1e-4
    NUM_INTEGRATION_RADII = 200

    cosmo_params: maszcal.cosmology.CosmoParams
    mass_definition: str
    delta: float
    comoving_radii: bool

    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def _con(self, zs, masses):
        mass_def = str(self.delta) + self.mass_definition[0]
        try:
            return self._con_model.c(masses, zs, mass_def)
        except AttributeError:
            self._init_con_model()
            return self._con_model.c(masses, zs, mass_def)

    def _gnfw_norm(self, zs, mus, alphas, betas, gammas):
        '''
        SHAPE mu, z, params
        '''
        rsflat = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.baryon_norm_radius,
            self.NUM_INTEGRATION_RADII,
        )
        rs = rsflat[:, None]
        drs = np.gradient(rsflat)

        top_integrand = self._rho_nfw(rs, zs, mus) * rs[..., None, :]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None, :, None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=0)[..., None]
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, alphas, betas, gammas)
        norm = maszcal.mathutils.atleast_kd(norm, rs.ndim+norm.ndim-1, append_dims=False)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def _rho_nfw(self, rs, zs, mus):
        masses = self.mass_from_mu(mus)
        cons = self._con(zs, masses)
        return self.nfw_model.rho(rs, zs, masses, cons)

    def rho_bary(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE r, mu, z, params
        '''
        return self.baryon_frac * self._rho_gnfw(rs, zs, mus, alphas, betas, gammas)

    def rho_cdm(self, rs, zs, mus):
        '''
        SHAPE mu, z, r, params
        '''
        return (1-self.baryon_frac) * self._rho_nfw(rs, zs, mus)

    def rho_tot(self, rs, zs, mus, alphas, betas, gammas):
        rho_cdm = self.rho_cdm(rs, zs, mus)[..., None]
        return self.rho_bary(rs, zs, mus, alphas, betas, gammas) + rho_cdm


@dataclass
class CmGnfw(_CmGnfw):
    con_class: object = maszcal.concentration.ConModel
    nfw_class: object = _nfw.CmNfwModel
    units: u.Quantity = u.Msun/u.pc**2
    sd_func: object = projector.SurfaceDensity.calculate
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    esd_func: object = projector.ExcessSurfaceDensity.calculate
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    baryon_norm_radius: float = 3.3


@dataclass
class _MatchingGnfw(_Gnfw):
    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, rs.shape, params
        '''
        ys = (rs/maszcal.mathutils.atleast_kd(self._r_delta(zs, mus), rs.ndim, append_dims=False)) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = maszcal.mathutils.atleast_kd(alphas, rs.ndim+1, append_dims=False)
        betas = maszcal.mathutils.atleast_kd(betas, rs.ndim+1, append_dims=False)
        gammas = maszcal.mathutils.atleast_kd(gammas, rs.ndim+1, append_dims=False)

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, cons, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rsflat = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.baryon_norm_radius,
            self.NUM_INTEGRATION_RADII,
        )
        rs = rsflat[:, None]
        drs = np.gradient(rsflat)

        top_integrand = self._rho_nfw(rs, zs, mus, cons) * rs[..., None]**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=0)
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, cons, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, cons, alphas, betas, gammas)
        norm = maszcal.mathutils.atleast_kd(norm, rs.ndim+norm.ndim-1, append_dims=False)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def rho_tot(self, rs, zs, mus, cons, alphas, betas, gammas):
        rho_cdm = self.rho_cdm(rs, zs, mus, cons)
        return self.rho_bary(rs, zs, mus, cons, alphas, betas, gammas) + rho_cdm


@dataclass
class MatchingGnfw(_MatchingGnfw):
    nfw_class: object = _nfw.MatchingNfwModel
    units: u.Quantity = u.Msun/u.pc**2
    sd_func: object = projector.SurfaceDensity.calculate
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    esd_func: object = projector.ExcessSurfaceDensity.calculate
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    baryon_norm_radius: float = 3.3


class _MatchingCmGnfw(_CmGnfw):
    def _init_con_model(self):
        mass_def = str(self.delta) + self.mass_definition[0]
        self._con_model = self.con_class(mass_def, cosmology=self.cosmo_params)

    def gnfw_shape(self, rs, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, rs.shape, params
        '''
        ys = (
            rs/maszcal.mathutils.atleast_kd(
                self._r_delta(zs, mus),
                rs.ndim,
                append_dims=False,
            )
        ) / self.CORE_RADIUS
        ys = ys[..., None]

        alphas = maszcal.mathutils.atleast_kd(alphas, rs.ndim+1, append_dims=False)
        betas = maszcal.mathutils.atleast_kd(betas, rs.ndim+1, append_dims=False)
        gammas = maszcal.mathutils.atleast_kd(gammas, rs.ndim+1, append_dims=False)

        return 1 / (ys**gammas * (1 + ys**(1/alphas))**((betas-gammas) * alphas))

    def _gnfw_norm(self, zs, mus, alphas, betas, gammas):
        '''
        SHAPE cluster, params
        '''
        rsflat = np.linspace(
            self.MIN_INTEGRATION_RADIUS,
            self.baryon_norm_radius,
            self.NUM_INTEGRATION_RADII,
        )
        rs = rsflat[:, None]
        drs = np.gradient(rsflat)

        top_integrand = self._rho_nfw(rs, zs, mus) * rs**2
        bottom_integrand = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas) * rs[..., None]**2

        return (maszcal.mathutils.trapz_(top_integrand, dx=drs, axis=0)[:, None]
                / maszcal.mathutils.trapz_(bottom_integrand, dx=drs, axis=0))

    def _rho_gnfw(self, rs, zs, mus, alphas, betas, gammas):
        norm = self._gnfw_norm(zs, mus, alphas, betas, gammas)
        norm = maszcal.mathutils.atleast_kd(norm, rs.ndim+norm.ndim-1, append_dims=False)
        profile_shape = self.gnfw_shape(rs, zs, mus, alphas, betas, gammas)
        return norm * profile_shape

    def rho_tot(self, rs, zs, mus, alphas, betas, gammas):
        rho_cdm = maszcal.mathutils.atleast_kd(self.rho_cdm(rs, zs, mus), rs.ndim+1)
        return self.rho_bary(rs, zs, mus, alphas, betas, gammas) + rho_cdm


@dataclass
class MatchingCmGnfw(_MatchingCmGnfw):
    con_class: object = maszcal.concentration.MatchingConModel
    nfw_class: object = _nfw.MatchingCmNfwModel
    units: u.Quantity = u.Msun/u.pc**2
    sd_func: object = projector.SurfaceDensity.calculate
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    esd_func: object = projector.ExcessSurfaceDensity.calculate
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': 1, 'density_axis': -2})
    baryon_norm_radius: float = 3.3
