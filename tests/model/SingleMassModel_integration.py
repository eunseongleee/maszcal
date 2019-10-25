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
from maszcal.model import SingleMassModel
from maszcal.lensing import SingleMassLensingSignal


def describe_single_mass_bin():

    @pytest.fixture
    def single_mass_model():
        zs = np.ones(1)
        return SingleMassModel(redshift=zs)

    @pytest.fixture
    def lensing_signal():
        zs = np.ones(1)
        return SingleMassLensingSignal(redshift=zs)

    def both_classes_give_the_same_thing(lensing_signal, single_mass_model):
        rs = np.logspace(-2, 1, 50)
        mu = np.array([np.log(1e15)])
        concentration = np.array([3])
        params = np.array([[mu, concentration]])

        ds = single_mass_model.delta_sigma(rs, mu, concentration)
        esd = lensing_signal.esd(rs, params)

        assert np.all(ds == esd)

    def the_plot_looks_correct():
        z = np.array([0.43])
        single_mass_model = SingleMassModel(redshift=z, delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 50)
        mu = np.array([np.log(4.26e14)])
        concentration = np.array([2.08])
        params = np.array([[mu, concentration]])

        ds = single_mass_model.delta_sigma(rs, mu, concentration)

        plt.plot(rs, ds.flatten())
        plt.xlabel(r'$R \; (\mathrm{Mpc}/h)$')
        plt.ylabel(r'$\Delta \Sigma \; (h\, M_\circ / \mathrm{pc}^2)$')

        plt.ylim((1e0, 3.3e2))

        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('figs/test/single_mass_bin.svg')
        plt.gcf().clear()
