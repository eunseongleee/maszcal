import pytest
import numpy as np
from maszcal.model import StackedModel


def describe_stacked_model():

    def describe_init():

        def it_requires_you_to_provide_mass_and_redshift():
            with pytest.raises(TypeError):
                StackedModel()

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = StackedModel(mus, zs)

            return model

        def it_computes_weak_lensing_avg_mass():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = StackedModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            stacked_model.delta_sigma_of_mass = lambda rs,mus,cons: np.ones(
                (stacked_model.mus.size, rs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)

        def it_can_use_different_mass_definitions():
            cons = np.array([2, 3, 4])
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = StackedModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma_of_mass(rs, mus, cons)

            delta = 200
            kind = 'mean'
            model = StackedModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma_of_mass(rs, mus, cons)

            assert np.all(delta_sigs_200m < delta_sigs_500c)
