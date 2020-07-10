from dataclasses import dataclass
import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.interp_utils
import maszcal.twohalo
import maszcal.cosmology


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model(mocker):
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 10)
        mus = np.ones(10) * np.log(1e14)
        rs = np.logspace(-3, 3, 200)

        esds = two_halo_model.esd(rs, mus, zs)

        assert not np.any(np.isnan(esds))
        assert esds.shape == zs.shape + rs.shape

        plt.plot(rs, esds.T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\Delta \Sigma \; (M_\odot/\mathrm{pc}^2)$')
        plt.savefig('figs/test/two_halo_esd.svg')

        plt.gcf().clear()

    def it_calculates_halo_matter_correlations(two_halo_model):
        zs = np.linspace(0, 1, 10)
        mus = np.ones(1) * np.log(1e14)
        rs = np.logspace(-2, 3, 600)

        xis = two_halo_model.halo_matter_correlation(rs, mus, zs)

        assert not np.any(np.isnan(xis))
        assert xis.shape == zs.shape + rs.shape

        plt.plot(rs, rs[:, None]**2 * xis.T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\xi$')
        plt.savefig('figs/test/halo_matter_correlation.svg')

        plt.gcf().clear()


def describe_TwoHaloConvergenceModel():

    @pytest.fixture
    def two_halo_model(mocker):
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo)
        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0.1, 1, 4)
        mus = np.ones(4) * np.log(1e14)
        from_arcmin = 2 * np.pi / 360 / 60
        thetas = np.logspace(-4, np.log10(15*from_arcmin), 200)

        kappas = two_halo_model.kappa(thetas, mus, zs)

        assert not np.any(np.isnan(kappas))
        assert kappas.shape == zs.shape + thetas.shape

        plt.plot(thetas, thetas[:, None] * kappas.T)
        plt.xscale('log')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\theta \kappa(\theta)$')
        plt.savefig('figs/test/two_halo_kappa.svg')

        plt.gcf().clear()
