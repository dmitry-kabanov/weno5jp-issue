_eps = 1e-6
_size = 0
_dx = 0.0
_dx_inv = 0.0
_dx_inv_12 = 0.0


def init(eps, size, dx):
    global _eps
    global _size
    global _dx
    global _dx_inv
    global _dx_inv_12

    _eps = eps
    _size = size
    _dx = dx
    _dx_inv = 1.0 / _dx
    _dx_inv_12 = 1.0 / (12.0 * _dx)


def interpolate(um3, um2, um1, u0, up1, up2, up3, u_x_plus, u_x_minus):
    """Compute right- and left-biased approximations of derivative."""
    global _dx

    der1 = um1 - um2
    der2 = u0 - um1
    der3 = up1 - u0
    der4 = up2 - up1
    numer = -der1 + 7 * der2 + 7 * der3 - der4
    common = numer / (12.0 * _dx)
    # numer = um2 + 8*(up1 - um1) - up2
    # common = numer * _dx_inv_12

    # Compute second derivatives
    der1 = (up3 - 2*up2 + up1) / _dx
    der2 = (up2 - 2*up1 + u0) / _dx
    der3 = (up1 - 2*u0 + um1) / _dx
    der4 = (u0 - 2*um1 + um2) / _dx
    der5 = (um1 - 2*um2 + um3) / _dx

    weno_plus_flux = _weno_flux(der1, der2, der3, der4)
    u_x_plus[:] = common + weno_plus_flux
    weno_minus_flux = _weno_flux(der5, der4, der3, der2)
    u_x_minus[:] = common - weno_minus_flux

    return u_x_plus, u_x_minus


def _weno_flux(a, b, c, d):
    """Calculate WENO approximation of the flux."""
    global _eps

    is0 = 13.0*(a - b)**2 + 3.0*(a - 3*b)**2
    is1 = 13.0*(b - c)**2 + 3.0*(b + c)**2
    is2 = 13.0*(c - d)**2 + 3.0*(3*c - d)**2

    alpha0 = 1.0 / (_eps + is0)**2
    alpha1 = 6.0 / (_eps + is1)**2
    alpha2 = 3.0 / (_eps + is2)**2
    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w2 = alpha2 / sum_alpha

    return (1.0/3.0)*w0*(a - 2*b + c) + (1.0/6.0)*(w2 - 0.5)*(b - 2*c + d)
