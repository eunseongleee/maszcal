
import numpy as np
import pytest
from maszcal.lensing import SingleBaryonLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeSingleMassBaryonShearModel:
    def __init__(
            self,
            redshift,
            units=1,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
    ):
        pass

    def delta_sigma(self, rs, mus, cons, alphas, betas, gammas):
        return np.ones(13)


def describe_single_mass_lensing_signal():

    def describe_esd():

        def it_fails_if_there_are_too_many_redshifts():
            redshifts = np.ones(10)
            with pytest.raises(TypeError):
                lensing_signal = SingleBaryonLensingSignal(redshift=redshifts)

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.SingleMassBaryonShearModel', new=FakeSingleMassBaryonShearModel)
            redshift = np.array([0])
            return SingleBaryonLensingSignal(redshift=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1, 1, 1, 1]])

            esd = lensing_signal.esd(rs, params)
            assert np.all(esd == np.ones(13))
