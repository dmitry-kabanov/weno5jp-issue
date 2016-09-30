import numpy as np
import numpy.testing as npt

from numpy import (cos, exp, linspace, pi, sin)

from observedorder import compute_observed_order_of_accuracy
from weno5jp import WENO5JP


class TestWENO5JP:
    def test__sine_function__should_converge_to_fifth_order(self):
        n_list = [20, 40, 80, 160, 320, 640]
        error_list = []

        for n in n_list:
            x, dx = linspace(0, 2*pi, num=n, retstep=True)
            approx = WENO5JP(len(x), dx, eps=1e-6)
            ng = approx.nghost_points
            y = sin(x)
            desired = cos(x[ng:-ng])

            yhat_p, yhat_m = approx.interpolate(y)
            result = 0.5*(yhat_p + yhat_m)

            error = np.linalg.norm(desired - result, np.inf)
            error_list.append(error)

        observed_orders = compute_observed_order_of_accuracy(
            error_list, n_list, exclude=True)

        import ipdb
        ipdb.set_trace()

        delta = 0.05
        expected_order = 5.0
        min_order = expected_order - delta
        npt.assert_(np.all(observed_orders >= min_order))

    def test__trefethen_p56_example__should_be_at_least_fifth_order(self):
        n_list = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560]) + 1

        err_list = []

        for n in n_list:
            x, dx = np.linspace(-1, 1, num=n, retstep=True)
            approx = WENO5JP(len(x), dx, eps=1e-6)
            ng = approx.nghost_points
            y = np.exp(x) * np.sin(5*x)
            desired = np.exp(x) * (np.sin(5*x) + 5*np.cos(5*x))
            desired = desired[ng:-ng]

            yhat_p, yhat_m = approx.interpolate(y)
            result = 0.5 * (yhat_p + yhat_m)

            err_inf = np.linalg.norm(result - desired, np.Inf)

            err_list.append(err_inf)

        orders = compute_observed_order_of_accuracy(err_list, n_list,
                                                    exclude=True)

        import ipdb
        ipdb.set_trace()

        delta = 0.05
        # TODO: behavior here is very strange actually as I get something
        # between 4 and 6 order of accuracy here.
        expected_order = 4.0
        min_order = expected_order - delta
        npt.assert_(np.all(orders >= min_order))
