import pytest
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
from maszcal.lensing import BaryonShearModel


def describe_gnfw_baryonic_model():

    def describe_gnfw_rho():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return BaryonShearModel(mus, zs, mass_definition='crit', delta=500)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            rhos = baryon_model._shear.rho_bary(radii, zs, mus, cons, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, rhos)
            plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$\rho(R)$')

            plt.savefig('figs/test/rho_bary.svg')
            plt.gcf().clear()

    def describe_gnfw_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return BaryonShearModel(mus, zs, units=u.Msun/u.pc**2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            esds = baryon_model._shear.delta_sigma_bary(radii, zs, mus, cons, alphas, betas, gammas)[0, 0, :, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_bary.svg')
            plt.gcf().clear()

    def describe_total_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return BaryonShearModel(mus, zs, units=u.Msun/u.pc**2)

        @pytest.fixture
        def baryon_model_():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            model = BaryonShearModel(mus, zs, units=u.Msun/u.pc**2)
            model.CORE_RADIUS = 1/2 # 1/concentration to be used
            return model

        def it_can_recreate_an_nfw_model(baryon_model_):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1) # It is critical this concentration match the core radius above
            alphas = np.ones(1)
            betas = 3*np.ones(1)
            gammas = np.ones(1)

            assert np.all(baryon_model_.CORE_RADIUS == 1/cons)

            esds_bary = baryon_model_._shear.delta_sigma_bary(radii,
                                                              zs,
                                                              mus,
                                                              cons,
                                                              alphas,
                                                              betas,
                                                              gammas)/baryon_model_._shear.baryon_frac

            esds_total = baryon_model_._shear.delta_sigma_total(radii,
                                                                zs,
                                                                mus,
                                                                cons,
                                                                alphas,
                                                                betas,
                                                                gammas)

            esds_nfw = baryon_model_._shear.delta_sigma_cdm(radii,
                                                            zs,
                                                            mus,
                                                            cons)/(1-baryon_model_._shear.baryon_frac)

            assert np.allclose(esds_bary, esds_nfw, rtol=1e-2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            esds = baryon_model._shear.delta_sigma_total(radii, zs, mus, cons, alphas, betas, gammas)[0, 0, :, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_total.svg')
            plt.gcf().clear()

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return BaryonShearModel(mus, zs, units=u.Msun/u.pc**2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = 0.3*np.ones(1)

            esds = baryon_model.stacked_delta_sigma(radii, cons, alphas, betas, gammas, a_szs)

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()
