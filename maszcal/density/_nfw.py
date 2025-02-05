from functools import partial
import numpy as np
import astropy.units as u
import maszcal.cosmology
from maszcal.defaults import DefaultCosmology
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_astropy_cosmology
import maszcal.mathutils as mathutils


class NfwModel:
    CMB_REDSHIFT = 1100

    def __init__(
        self,
        cosmo_params=DefaultCosmology(),
        units=u.Msun/u.pc**2,
        delta=200,
        mass_definition='mean',
        comoving=True,
    ):
        self._delta = delta
        self._check_mass_def(mass_definition)
        self.mass_definition = mass_definition
        self.comoving = comoving

        if isinstance(cosmo_params, DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self._astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        self.units = units
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, comoving=self.comoving, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def _check_mass_def(self, mass_def):
        if mass_def not in ['mean', 'crit']:
            raise ValueError('Mass definition must be \'crit\' or \'mean\'')

    @property
    def mass_definition(self):
        return self._mass_definition

    @mass_definition.setter
    def mass_definition(self, new_mass_def):
        self._check_mass_def(new_mass_def)
        self._mass_definition = new_mass_def

    def _reference_density_comoving(self, zs):
        if self.mass_definition == 'mean':
            rho_mass_def = self._astropy_cosmology.critical_density0 * self._astropy_cosmology.Om0 * np.ones(zs.shape)
        elif self.mass_definition == 'crit':
            rho_mass_def = self._astropy_cosmology.critical_density(zs) / (1+zs)**3

        return rho_mass_def

    def _reference_density_nocomoving(self, zs):
        if self.mass_definition == 'mean':
            rho_mass_def = self._astropy_cosmology.critical_density(zs) * self._astropy_cosmology.Om(zs)
        elif self.mass_definition == 'crit':
            rho_mass_def = self._astropy_cosmology.critical_density(zs)

        return rho_mass_def

    def reference_density(self, zs):
        '''
        SHAPE z
        '''

        if self.comoving:
            rho_mass_def = self._reference_density_comoving(zs)
        else:
            rho_mass_def = self._reference_density_nocomoving(zs)

        return rho_mass_def.to(u.Msun/u.Mpc**3).value

    def radius_delta(self, zs, masses):
        '''
        SHAPE mass, z
        '''
        pref = 3 / (4*np.pi)
        return (pref * masses[:, None] / (self._delta*self.reference_density(zs))[None, :])**(1/3)

    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE mass, z, cons
        '''
        return self.radius_delta(zs, masses)[:, :, None]/cons[None, None, :]

    def delta_c(self, cons):
        '''
        SHAPE cons
        '''
        return (self._delta * cons**3)/(3 * (np.log(1+cons) - cons/(1+cons)))

    def _sd_less_than_func(self, x):
        return (
            1 - 2*np.arctanh(
                np.sqrt((1-x)/(1+x))
            )/np.sqrt(1 - x**2)
        )/(x**2 - 1)

    def _sd_equal_func(self, x):
        return 1/3

    def _sd_greater_than_func(self, x):
        return (
            1 - 2*np.arctan(
                np.sqrt((x-1)/(1+x))
            )/np.sqrt(x**2 - 1)
        )/(x**2 - 1)

    def _esd_less_than_func(self, x):
        return (
            8 * np.arctanh(np.sqrt((1-x) / (1+x))) / (x**2 * np.sqrt(1 - x**2))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctanh(np.sqrt((1-x) / (1+x))) / ((x**2 - 1) * np.sqrt(1 - x**2))
        )

    def _esd_equal_func(self, x):
        return 10/3 + 4*np.log(1/2)

    def _esd_greater_than_func(self, x):
        return (
            8 * np.arctan(np.sqrt((x-1) / (x+1))) / (x**2 * np.sqrt(x**2 - 1))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctan(np.sqrt((x-1) / (x+1))) / ((x**2 - 1)**(3/2))
        )

    def _sd_inequality_func(self, xs):
        full_func_vals = np.zeros(xs.shape)

        full_func_vals[xs < 1] = self._sd_less_than_func(xs[xs < 1])
        full_func_vals[xs == 1] = self._sd_equal_func(xs[xs == 1])
        full_func_vals[xs > 1] = self._sd_greater_than_func(xs[xs > 1])

        return full_func_vals

    def _esd_inequality_func(self, xs):
        full_func_vals = np.zeros(xs.shape)

        full_func_vals[xs < 1] = self._esd_less_than_func(xs[xs < 1])
        full_func_vals[xs == 1] = self._esd_equal_func(xs[xs == 1])
        full_func_vals[xs > 1] = self._esd_greater_than_func(xs[xs > 1])

        return full_func_vals

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE r, mass, z, cons
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        numerator = self.delta_c(cons)[None, None, :] * self.reference_density(zs)[None, :, None]
        total_num_dims = rs.ndim + numerator.ndim - 1
        numerator = mathutils.atleast_kd(numerator, total_num_dims, append_dims=False)
        xs = rs[..., None, :, None] / (
            mathutils.atleast_kd(scale_radii, total_num_dims, append_dims=False)
        )

        denominator = xs * (1+xs)**2
        return numerator/denominator

    def surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, mass, z, cons
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = 2 * scale_radii * self.delta_c(cons)[None, None, :] * self.reference_density(zs)[None, :, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[..., None, :, None]/mathutils.atleast_kd(scale_radii, rs.ndim+scale_radii.ndim-1, append_dims=False)

        postfactor = self._sd_inequality_func(xs)

        return prefactor[None, :] * postfactor

    def excess_surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, mass, z, cons
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons)[None, None, :] * self.reference_density(zs)[None, :, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[..., None, :, None]/mathutils.atleast_kd(scale_radii, rs.ndim+scale_radii.ndim-1, append_dims=False)

        postfactor = self._esd_inequality_func(xs)

        return prefactor[None, :] * postfactor

    def convergence(self, rs, zs, mus, cons):
        masses = np.exp(mus)
        sds = self.surface_density(rs, zs, masses, cons)
        sd_crits =  maszcal.mathutils.atleast_kd(
            self.sigma_crit(z_lens=zs)[:, None],
            sds.ndim,
            append_dims=False,
        )
        return sds/sd_crits


class ProjectorSafeNfwModel(NfwModel):
    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE z, params
        '''
        return self.radius_delta(zs, masses).T / cons[None, :]

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE r, z, params
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        scale_radii = mathutils.atleast_kd(scale_radii, rs.ndim+2, append_dims=False)
        numerator = self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        numerator = mathutils.atleast_kd(numerator, rs.ndim+2, append_dims=False)
        xs = rs/scale_radii

        denominator = xs * (1+xs)**2
        return numerator/denominator


class SingleMassNfwModel(NfwModel):
    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE z, params
        '''
        return self.radius_delta(zs, masses).T / cons[None, :]

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE r, z, params
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        numerator = self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        total_num_dims = rs.ndim+numerator.ndim-1
        numerator = mathutils.atleast_kd(numerator, total_num_dims, append_dims=False)
        scale_radii = mathutils.atleast_kd(scale_radii, total_num_dims, append_dims=False)

        # when mass and redshift input is an array
        if np.ndim(zs) != 1:
            scale_radii = np.swapaxes(scale_radii, 1, 3)
            xs = rs[..., None, None]/scale_radii
        else:
            xs = rs[..., None]/scale_radii

        denominator = xs * (1+xs)**2
        return numerator/denominator

    def surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, z, params
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = 2 * scale_radii * self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[..., None]/scale_radii[None, ...]

        postfactor = self._sd_inequality_func(xs)

        return prefactor[None, ...] * postfactor

    def excess_surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, z, params
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[..., None]/scale_radii[None, ...]

        postfactor = self._esd_inequality_func(xs)

        return prefactor[None, ...] * postfactor


class CmNfwModel(NfwModel):
    '''
    Overwrites some methods to make it work for a concentration-mass relation
    '''
    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE mass, z, c
        '''
        return self.radius_delta(zs, masses)/cons

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE mass, r, z, c
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        numerator = self.delta_c(cons) * self.reference_density(zs)[None, :]
        total_num_dims = rs.ndim + numerator.ndim - 1
        numerator = mathutils.atleast_kd(numerator, total_num_dims, append_dims=False)
        xs = rs[..., None, :]/mathutils.atleast_kd(scale_radii, total_num_dims, append_dims=False)
        denominator = xs * (1+xs)**2
        return numerator/denominator

    def surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE mass, r, z, c
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = 2 * scale_radii * self.delta_c(cons) * self.reference_density(zs)[None, :]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[:, None]/scale_radii[None, ...]

        postfactor = self._sd_inequality_func(xs)

        return prefactor[None, :, :] * postfactor

    def excess_surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE mass, r, z, c
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons) * self.reference_density(zs)[None, :]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[:, None]/scale_radii[None, ...]

        postfactor = self._esd_inequality_func(xs)

        return prefactor[None, :, :] * postfactor

    def convergence(self, rs, zs, mus, cons):
        masses = np.exp(mus)
        sds = self.surface_density(rs, zs, masses, cons)
        sd_crits =  maszcal.mathutils.atleast_kd(
            self.sigma_crit(z_lens=zs),
            sds.ndim,
            append_dims=False,
        )
        return sds/sd_crits


class MatchingNfwModel(NfwModel):
    '''
    Overwrites some methods to make it work for a matching stack
    '''
    def radius_delta(self, zs, masses):
        '''
        SHAPE cluster
        '''
        pref = 3 / (4*np.pi)
        return (pref * masses / (self._delta*self.reference_density(zs)))**(1/3)

    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE cluster, cons
        '''
        return self.radius_delta(zs, masses)[:, None]/cons[None, :]

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster, c
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        numerator = self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        total_num_dims = rs.ndim + numerator.ndim - 1
        numerator = mathutils.atleast_kd(numerator, total_num_dims, append_dims=False)
        xs = rs[..., None] / (
            mathutils.atleast_kd(scale_radii, total_num_dims, append_dims=False)
        )
        denominator = xs * (1+xs)**2
        return numerator/denominator

    def surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = 2 * scale_radii * self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)
        xs = rs[..., None]/scale_radii[None, ...]
        postfactor = self._sd_inequality_func(xs)
        return prefactor[None, ...] * postfactor

    def excess_surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons)[None, :] * self.reference_density(zs)[:, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)
        xs = rs[..., None]/scale_radii[None, ...]
        postfactor = self._esd_inequality_func(xs)
        return prefactor[None, ...] * postfactor


class MatchingCmNfwModel(NfwModel):
    '''
    Overwrites some methods to make it work for a matching stack
    '''
    def radius_delta(self, zs, masses):
        '''
        SHAPE cluster
        '''
        pref = 3 / (4*np.pi)
        return (pref * masses / (self._delta*self.reference_density(zs)))**(1/3)

    def scale_radius(self, zs, masses, cons):
        '''
        SHAPE cluster
        '''
        return self.radius_delta(zs, masses)/cons

    def rho(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        numerator = self.delta_c(cons) * self.reference_density(zs)
        total_num_dims = rs.ndim + numerator.ndim - 1
        numerator = mathutils.atleast_kd(numerator, total_num_dims, append_dims=False)
        xs = rs / (
            mathutils.atleast_kd(scale_radii, total_num_dims, append_dims=False)
        )
        denominator = xs * (1+xs)**2
        return numerator/denominator

    def surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = 2 * scale_radii * self.delta_c(cons) * self.reference_density(zs)
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs/scale_radii[None, :]

        postfactor = self._sd_inequality_func(xs)

        return prefactor[None, :] * postfactor

    def excess_surface_density(self, rs, zs, masses, cons):
        '''
        SHAPE r, cluster
        '''
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons) * self.reference_density(zs)
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs/scale_radii[None, :]

        postfactor = self._esd_inequality_func(xs)

        return prefactor[None, :] * postfactor

    def convergence(self, rs, zs, mus, cons):
        masses = np.exp(mus)
        sds = self.surface_density(rs, zs, masses, cons)
        sd_crits =  maszcal.mathutils.atleast_kd(
            self.sigma_crit(z_lens=zs),
            sds.ndim,
            append_dims=False,
        )
        return sds/sd_crits
