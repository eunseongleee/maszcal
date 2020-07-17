from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
from maszcal.cosmo_utils import get_astropy_cosmology
from maszcal.cosmology import CosmoParams
import maszcal.nfw
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
import maszcal.lensing._core as _core


@dataclass
class SingleMassBaryonShearModel:
    redshifts: np.ndarray
    units: u.Quantity = u.Msun/u.pc**2
    comoving_radii: bool = True
    delta: float = 200
    mass_definition: str = 'mean'
    cosmo_params: object = maszcal.defaults.DefaultCosmology()
    shear_class: object = _core.SingleMassGnfwBaryonShear

    def __post_init__(self):
        if isinstance(self.cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()

        self._shear = self.shear_class(
            cosmo_params=self.cosmo_params,
            mass_definition=self.mass_definition,
            delta=self.delta,
            units=self.units,
            comoving_radii=self.comoving_radii,
            nfw_class=maszcal.nfw.SingleMassNfwModel,
            gnfw_class=maszcal.gnfw.SingleMassGnfw,
            esd_func=projector.esd,
        )

    def delta_sigma(self, rs, mus, cons, alphas, betas, gammas):
        return self._shear.delta_sigma_total(rs, self.redshifts, mus, cons, alphas, betas, gammas)


class SingleMassNfwShearModel:
    def __init__(
            self,
            redshifts,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=maszcal.defaults.DefaultCosmology(),
    ):

        self.redshifts = redshifts
        self.units = units
        self.comoving_radii = comoving_radii
        self.delta = delta
        self.mass_definition = mass_definition

        if isinstance(cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def _init_nfw(self):
        self.nfw_model = maszcal.nfw.NfwModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def mass(self, mu):
        return np.exp(mu)

    def delta_sigma(self, rs, mus, concentrations):
        masses = self.mass(mus)

        try:
            return self.nfw_model.delta_sigma(rs, self.redshifts, masses, concentrations)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.delta_sigma(rs, self.redshifts, masses, concentrations)
