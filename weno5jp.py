import numpy as np

# from . import weno5jp_impl_c
import weno5jp_impl_py


class WENO5JP:
    """Computation of derivative approximations using WENO5JP method.

    The method is fully described in [1]_.
    In short, it computes right- and left- biased approximations of the spatial
    derivatives on the uniform grid.

    Parameters
    ----------
    npoints : int
        Number of grid points
    dx : float
        Grid spatial step
    eps : float
        Small parameter that is used in WENO schemes to compute weights.
    impl : str, optional
        Implementation of the WENO scheme (in C or Python)

    References
    ----------
    [1] Jiang, Guang-Shan and Peng, Danping.
        Weighted {ENO} Schemes for {Hamilton--Jacobi} Equations
        Journal of Scientific Computing, vol. 21, pp. 2126--2143, 2000.

    """
    def __init__(self, npoints, dx, eps=1e-6, impl='weno5jp_impl_py'):
        self.npoints = npoints
        self.dx = dx
        self.eps = eps

        self.nghost_points = 3
        self.u_x_plus = np.empty(npoints - 2*self.nghost_points)
        self.u_x_minus = np.empty(npoints - 2*self.nghost_points)

        size = self.npoints - 2*self.nghost_points

        if impl == 'weno5jp_impl_py':
            self._impl = weno5jp_impl_py
        # elif impl == 'weno5jp_impl_c':
        #     self._impl = weno5jp_impl_c
        else:
            raise Exception('weno5jp: implementation is not given')

        self._impl.init(self.eps, size, self.dx)

    def interpolate(self, u):
        l1 = self.nghost_points
        l2 = self.npoints - self.nghost_points

        um3 = u[l1-3:l2-3]
        um2 = u[l1-2:l2-2]
        um1 = u[l1-1:l2-1]
        u0 = u[l1:l2]
        up1 = u[l1+1:l2+1]
        up2 = u[l1+2:l2+2]
        up3 = u[l1+3:l2+3]

        self._impl.interpolate(um3, um2, um1, u0, up1, up2, up3,
                               self.u_x_plus, self.u_x_minus)

        return self.u_x_plus, self.u_x_minus
