import numpy as np
import matplotlib.pyplot as plt

from observedorder import compute_observed_order_of_accuracy
from weno5jp import WENO5JP


def test_sine_function():
    n_list = np.array([10, 40, 80, 160, 320, 640, 1280, 2560]) + 1
    error_list = []

    for n in n_list:
        x, dx = np.linspace(0, 2*np.pi, num=n, retstep=True)
        approx = WENO5JP(len(x), dx, eps=1e-6)
        ng = approx.nghost_points
        y = np.sin(x)
        desired = np.cos(x[ng:-ng])

        yhat_p, yhat_m = approx.interpolate(y)
        result = 0.5*(yhat_p + yhat_m)

        error = np.linalg.norm(desired - result, np.inf)
        error_list.append(error)

    orders = compute_observed_order_of_accuracy(error_list, n_list)

    print('Sine function')
    _print_convergence_table(n_list, orders)
    _save_convergence_figure(n_list, error_list, 'sine.png')


def test_trefethen_p56_example():
    n_list = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]) + 1

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

    orders = compute_observed_order_of_accuracy(err_list, n_list)

    print('Trefethen\'s example, p. 56')
    _print_convergence_table(n_list, orders)
    _save_convergence_figure(n_list, err_list, 'trefethen_p56.png')


def _print_convergence_table(n_list, orders):
    assert len(n_list) == len(orders)

    print('|------------|-------|')
    print('| Resolution | Order |')
    for i, __ in enumerate(n_list):
        print('| {0:10d} | {1:5.2f} |'.format(n_list[i], orders[i]))

    print('|------------|-------|')


def _save_convergence_figure(n_list, errors, filename):
    plt.figure(figsize=(6, 4))
    plt.loglog(n_list, errors, 'o-')
    plt.xlabel('Resolution')
    plt.ylabel(r'$L_{\infty}$-error')
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':
    test_sine_function()
    test_trefethen_p56_example()
