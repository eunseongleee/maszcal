### HIGH LEVEL DEPENDENCIES ###
import json
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d, interp2d
### MID LEVEL DEPENDCIES ###
import camb
from astropy import units as u
### IN-MODULE DEPENDCIES ###
from maszcal.offset_nfw.nfw import NFWModel
from maszcal.tinker import dn_dlogM
from maszcal.cosmo_utils import get_camb_params, get_astropy_cosmology
from maszcal.cosmology import CosmoParams, Constants
from maszcal.mathutils import _trapz
from maszcal.nothing import NoParams
import maszcal.defaults as defaults


class StackedModel:
    """
    Canonical variable order:
    mu_sz, mu, z, r, params
    """
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            params=NoParams(),
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            cosmo_params=defaults.DefaultCosmology(),
    ):

        ### FITTING PARAMETERS AND LIKELIHOOD ###
        self.sigma_muszmu = 0.2
        self.b_sz = 1

        if not isinstance(params, NoParams):
            self.params = params

        ### SPATIAL QUANTITIES AND MATTER POWER ###
        self.max_k = 10
        self.min_k = 1e-4
        self.number_ks = 400

        ### COSMOLOGICAL PARAMETERS ###
        if isinstance(cosmo_params, defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        ### CLUSTER MASSES AND REDSHIFTS###
        self.mu_szs = mu_bins
        self.mus = mu_bins
        self.zs = redshift_bins

        ### SELECTION FUNCTION ###
        if isinstance(selection_func_file, defaults.DefaultSelectionFunc):
            self.selection_func = self._default_selection_func
        else:
            self.selection_func = self._get_selection_func_interpolator(selection_func_file)

        ### LENSING WEIGHTS ###
        if isinstance(lensing_weights_file, defaults.DefaultLensingWeights):
            self.lensing_weights = self._default_lensing_weights
        else:
            self.lensing_weights = self._get_lensing_weights_interpolator(lensing_weights_file)

        ### MISC ###
        self.constants = Constants()
        self._comoving_radii = True

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_parameters):
        self._params = new_parameters
        self.concentrations = self.params[:, 0]
        self.a_sz = self.params[:, 1]
        try:
            self.centered_fraction = self.params[:, 2]
            self.miscenter_radius = self.params[:, 3]
        except IndexError:
            pass

    @property
    def comoving_radii(self):
        return self._comoving_radii

    @comoving_radii.setter
    def comoving_radii(self, rs_are_comoving):
        self._comoving_radii = rs_are_comoving
        self.init_onfw()

    def calc_power_spect(self):
        params = get_camb_params(self.cosmo_params, self.max_k, self.zs)

        results = camb.get_results(params)

        self.ks, _, self.power_spect = results.get_matter_power_spectrum(minkh=self.min_k,
                                                                         maxkh=self.max_k,
                                                                         npoints=self.number_ks)

    def init_onfw(self):
        self.onfw_model = NFWModel(self.astropy_cosmology, comoving=self.comoving_radii)

    def prob_musz_given_mu(self, mu_szs, mus):
        """
        SHAPE mu_sz, mu, params
        """
        pref = 1/(np.sqrt(2*np.pi) * self.sigma_muszmu)

        diff = (mu_szs[:, None] - mus[None, :])[..., None] - self.a_sz[None, None, :]

        exps = np.exp(-diff**2 / (2*(self.sigma_muszmu)**2))

        return pref*exps

    def mass_sz(self, mu_szs):
        return np.exp(mu_szs)

    def mass(self, mus):
        return np.exp(mus)

    def _get_selection_func_interpolator(self, selection_func_file):
        with open(selection_func_file, 'r') as json_file:
            selec_func_dict = json.load(json_file)

        mus = np.asarray(selec_func_dict['mus'])
        zs = np.asarray(selec_func_dict['zs'])
        selection_fs = np.asarray(selec_func_dict['selection_fs'])
        interpolator = interp2d(zs, mus, selection_fs, kind='linear')

        return lambda mu, z: interpolator(z, mu)

    def _default_selection_func(self, mu_szs, zs):
        """
        SHAPE mu_sz, z
        """
        sel_func = np.ones((mu_szs.size, zs.size))

        low_mass_indices = np.where(mu_szs < np.log(3e14))
        sel_func[low_mass_indices, :] = 0

        return sel_func

    def sigma_of_mass(self, rs, mus, concentrations, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, c
        """
        masses = self.mass(mus)

        try:
            result = self.onfw_model.sigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.sigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

    def _offset_sigma_of_mass(self, rs, r_offsets, thetas, mus, concentrations, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, c, r_offset, theta
        """
        masses = self.mass(mus)

        try:
            result = self.onfw_model.offset_sigma_theory(rs, r_offsets, thetas, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.offset_sigma_theory(rs, r_offsets, thetas, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

    def misc_sigma(self, rs, mus, concentrations, cen_frac, r_misc, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, params
        """
        r_offsets = np.linspace(r_misc.min()/1e3, 10*r_misc.max(), 30)
        thetas = np.linspace(0, 2*np.pi, 10)

        offset_sigmas = self._offset_sigma_of_mass(rs, r_offsets, thetas, mus, concentrations, units)

        dthetas = np.gradient(thetas)
        theta_integral = _trapz(offset_sigmas, axis=-1, dx=dthetas)/(2*np.pi)

        misc_kernel = ((r_offsets[None, :]/r_misc[:, None]**2)
                       * np.exp(-0.5*(r_offsets[None, :]/r_misc[:, None])**2))

        dr_offsets = np.gradient(r_offsets)

        r_offset_integral = _trapz(
            theta_integral*misc_kernel,
            axis=-1,
            dx=dr_offsets
        )

        cen_frac = cen_frac[None, None, None, :]

        return (cen_frac * self.sigma_of_mass(rs, mus, concentrations, units)
                + (1-cen_frac) * r_offset_integral)

    def delta_sigma_of_mass(self, rs, mus, concentrations, units=u.Msun/u.pc**2, miscentered=False):
        """
        SHAPE mu, z, r, params
        """
        if not isinstance(miscentered, bool):
            raise ValueError("miscentered must be True or False")

        if miscentered:
            sigma_func = lambda rs, mus, cons, units: self.misc_sigma(
                rs,
                mus,
                cons,
                self.centered_fraction,
                self.miscenter_radius,
                units,
            )
        else:
            sigma_func = self.sigma_of_mass

        sigmas = sigma_func(rs, mus, concentrations, units)

        INNER_LEN = 20
        inner_rs = np.logspace(np.log10(rs[0]/1e2), np.log10(rs[0]), INNER_LEN)

        inner_sigmas = sigma_func(inner_rs, mus, concentrations, units)
        extended_sigmas = np.concatenate((inner_sigmas, sigmas), axis=2)

        extended_rs = np.concatenate((inner_rs, rs))

        extended_rs_ndim = extended_rs[None, None, :, None]

        sigmas_inside_r = integrate.cumtrapz(
            extended_sigmas * extended_rs_ndim,
            extended_rs,
            axis=2,
            initial=0
        ) / integrate.cumtrapz(
            extended_rs_ndim,
            extended_rs,
            axis=2,
            initial=0
        )
        return sigmas_inside_r[:, :, INNER_LEN:, ...] - sigmas

    def delta_sigma_of_mass_nfw(self, rs, mus, concentrations=None, units=u.Msun/u.pc**2):
        """
        SHAPE mu, z, r, params
        """
        masses = self.mass(mus)

        if concentrations is None:
            concentrations = self.concentrations

        try:
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result
        except AttributeError:
            self.init_onfw()
            result = self.onfw_model.deltasigma_theory(rs, masses, concentrations, self.zs)
            result = result * (u.Msun/u.Mpc**2).to(units)
            return result

    def dnumber_dlogmass(self):
        """
        SHAPE mu, z
        """
        masses = self.mass(self.mus)
        overdensity = 200
        rho_matter = self.cosmo_params.rho_crit * self.cosmo_params.omega_matter / self.cosmo_params.h**2

        try:
            power_spect = self.power_spect
        except AttributeError:
            self.calc_power_spect()
            power_spect = self.power_spect

        dn_dlogms = dn_dlogM(
            masses,
            self.zs,
            rho_matter,
            overdensity,
            self.ks,
            power_spect,
            comoving=True
        )

        return dn_dlogms

    def _get_lensing_weights_interpolator(self, lensing_weights_file):
        with open(lensing_weights_file, 'r') as json_file:
            weights_dict = json.load(json_file)

        zs = np.asarray(weights_dict['zs'])
        weights = np.asarray(weights_dict['weights'])

        return interp1d(zs, weights, kind='cubic')

    def _default_lensing_weights(self, zs):
        """
        SHAPE mu, z
        """
        return np.ones(zs.shape)

    def comoving_vol(self):
        """
        SHAPE z
        """
        c = self.constants.speed_of_light
        comov_dist = self.astropy_cosmology.comoving_distance(self.zs).value
        hubble_z = self.astropy_cosmology.H(self.zs).value

        return c * comov_dist**2 / hubble_z

    def _sz_measure(self):
        """
        SHAPE mu_sz, mu, z, params
        """
        return (self.mass_sz(self.mu_szs)[:, None, None, None]
                * self.selection_func(self.mu_szs, self.zs)[:, None, :, None]
                * self.prob_musz_given_mu(self.mu_szs, self.mus)[:, :, None, :])

    def number_sz(self):
        """
        SHAPE params
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(self._sz_measure(), axis=0, dx=dmu_szs)

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(self.dnumber_dlogmass()[..., None] * mu_sz_integral, axis=0, dx=dmus)

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral

    def _delta_sigma_miscentered(self, rs, units):
        """
        SHAPE R, params
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure()[:, :, :, None, :]
             * self.delta_sigma_of_mass(
                 rs,
                 self.mus,
                 self.concentrations,
                 units=units,
                 miscentered=True,
             )[None, ...]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz()[None, :]

    def _delta_sigma(self, rs, units):
        """
        SHAPE r, c, a_sz
        """
        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            (self._sz_measure()[:, :, :, None, :]
             * self.delta_sigma_of_mass(
                 rs,
                 self.mus,
                 self.concentrations,
                 units=units,
                 miscentered=False,
             )[None, ...]),
            axis=0,
            dx=dmu_szs,
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None, None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
             )[:, None, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz()[None, :]

    def delta_sigma(self, rs, units=u.Msun/u.Mpc**2, miscentered=False):
        if not miscentered:
            return self._delta_sigma(rs, units)
        else:
            return self._delta_sigma_miscentered(rs, units)

    def weak_lensing_avg_mass(self):
        mass_wl = self.mass(self.mus)

        dmu_szs = np.gradient(self.mu_szs)
        mu_sz_integral = _trapz(
            self._sz_measure() * mass_wl[None, :, None, None],
            axis=0,
            dx=dmu_szs
        )

        dmus = np.gradient(self.mus)
        mu_integral = _trapz(
            self.dnumber_dlogmass()[..., None] * mu_sz_integral,
            axis=0,
            dx=dmus
        )

        dzs = np.gradient(self.zs)
        z_integral = _trapz(
            ((
                self.lensing_weights(self.zs) * self.comoving_vol()
            )[:, None] * mu_integral),
            axis=0,
            dx=dzs
        )

        return z_integral/self.number_sz()
