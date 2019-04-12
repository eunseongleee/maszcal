from maszcal.model import StackedModel
import numpy as np

def test_mu_sz():
    stacked_model = StackedModel()

    mus = np.linspace(1, 10)
    bs = np.linspace(-5,5)
    as_ = np.ones(50)
    mu_szs = stacked_model.mu_sz(mus, as_, bs)

    precomp_mu_szs = np.array(
        [-4.        , -4.67680133, -5.2786339 , -5.80549771, -6.25739275,
       -6.63431903, -6.93627655, -7.16326531, -7.3152853 , -7.39233653,
       -7.39441899, -7.32153269, -7.17367763, -6.95085381, -6.65306122,
       -6.28029988, -5.83256976, -5.30987089, -4.71220325, -4.03956685,
       -3.29196168, -2.46938776, -1.57184506, -0.59933361,  0.44814661,
        1.57059559,  2.76801333,  4.04039983,  5.3877551 ,  6.81007913,
        8.30737193,  9.87963349, 11.52686381, 13.24906289, 15.04623074,
       16.91836735, 18.86547272, 20.88754686, 22.98458975, 25.15660142,
       27.40358184, 29.72553103, 32.12244898, 34.59433569, 37.14119117,
       39.76301541, 42.45980841, 45.23157018, 48.07830071, 51.        ]
    )

    np.testing.assert_allclose(mu_szs, precomp_mu_szs)

def test_prob_musz_given_mu():
    stacked_model = StackedModel()

    mus = np.linspace(1, 10)
    bs = np.linspace(-5,5)
    as_ = np.ones(50)

    probs = stacked_model.prob_musz_given_mu(mus, as_, bs)

    precomp_probs = np.array(
        [7.36823067e-196, 6.20808079e-256, 8.58288864e-318, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 4.93886467e-315, 3.27872392e-253,
         2.90647215e-193, 2.67479729e-137, 1.30344701e-087, 1.12544087e-046,
         3.77974403e-017, 7.11100045e-002, 7.08076925e-004, 2.31312449e-026,
         1.00811114e-072, 1.56374262e-146, 1.51098727e-251, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
         0.00000000e+000, 0.00000000e+000]
    )

    np.testing.assert_allclose(probs, precomp_probs)
