"""
GaussInterpolator

Interpolates maszcal model grids
"""
import numpy as np
from maszcal.interpolate import GaussInterpolator




length = 3
xs = np.linspace(0, 3, length)
ys = np.linspace(0, 3, length)
zs = np.linspace(0, 3, length)
coords = (xs, ys, zs)
grid = np.ones((length, length, length))

gauss_interpolator = GaussInterpolator(coords, grid)


"""
    - it interpolates a constant to be a constant
"""
def test_interp_over_constant_input():
    gauss_interpolator.process()

    test_xs = np.linspace(0, 3, 10)
    test_ys = test_xs
    test_zs = test_xs

    test_coords = (test_xs, test_ys, test_zs)

    interpolated_grid,_ = gauss_interpolator.interp(test_coords)

    assert np.allclose(interpolated_grid, np.ones((10, 10, 10)))
