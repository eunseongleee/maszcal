import pytest


def describe_nbatta_sim():

    def describe_impact_of_baryons():

        def using_gnfw_reduces_bias():
            nfw_model = select_model('nfw-cm-noemulation-singlemass')

            nfw_params_shape = (1, sim_data.radii.size, sim_data.redshifts.size)
            nfw_fits = np.zeros(nfw_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                def _pool_func(data): return nfw_model.get_best_fit(data, z)
                pool = Pool(NUM_THREADS)
                nfw_fits[:, :, i] = np.array(
                    pool.map(_pool_func, sim_data.wl_signal[:, :, i].T),
                ).T
                pool.close()
                pool.join()

            gnfw_model = select_model('gnfw-nocm-noemulation-singlemass')

            gnfw_params_shape = (4, sim_data.radii.size, sim_data.redshifts.size)
            gnfw_fits = np.zeros(gnfw_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                def _pool_func(data): return gnfw_model.get_best_fit(data, z)
                pool = Pool(NUM_THREADS)
                gnfw_fits[:, :, i] = np.array(
                    pool.map(_pool_func, sim_data.wl_signal[:, :, i].T),
                ).T
                pool.close()
                pool.join()

            nfw_masses = nfw_model.get_masses_from_params(nfw_fits)
            nfw_bias = _calc_bias(nfw_masses, sim_data.masses)

            gnfw_masses = gnfw_model.get_masses_from_params(gnfw_fits)
            gnfw_bias = _calc_bias(gnfw_masses, sim_data.masses)

            save_data({'nfw-bias_nbatta-2010_single-mass': nfw_bias, 'gnfw-bias_nbatta-2010_single-mass': gnfw_bias})
            _plot_biases({'NFW only': nfw_bias, 'Baryons': gnfw_bias})

            assert gnfw_bias.mean() < nfw_bias.mean()
