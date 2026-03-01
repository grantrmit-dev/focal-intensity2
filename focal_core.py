"""
Focal region intensity distribution calculator.
Vectorial Debye diffraction integral for high-NA objectives.
Converted from MATLAB (vectorial Debye theory, Richards & Wolf).
"""
import numpy as np
from scipy.special import jv

N_TH = 120  # integration points (increase for accuracy, decrease for speed)


def _integrate(f3d, th):
    """Trapezoid integration over axis 0: f3d shape (N_th, ny, nx)."""
    return np.trapz(f3d, th, axis=0)


def compute_intensity(wl, NA, n, m, e, z0, x_extent, y_extent, z_extent, re, polarization, plane):
    """
    Compute focal intensity distribution.

    Parameters
    ----------
    wl          : float  - wavelength (m)
    NA          : float  - numerical aperture
    n           : float  - refractive index of immersion medium
    m           : int    - topological charge (vortex phase); 0 = plane wave
    e           : float  - normalised inner radius of annular aperture (0=full)
    z0          : float  - axial position of image plane (m)
    x_extent    : float  - X half-width of calculation region (m)
    y_extent    : float  - Y half-width of calculation region (m)
    z_extent    : float  - half-width of Z calculation region (m)
    re          : int    - grid resolution (pixels per side)
    polarization: str    - 'Linear'|'Circular'|'Radial'|'Azimuthal'
    plane       : str    - 'X-Y'|'R-Z'

    Returns
    -------
    dict with keys I, Ix, Iy, Iz (2-D lists, normalised 0-1),
    x_axis, y_axis (1-D lists in units of lambda),
    x_label, y_label (strings)
    """
    a   = np.arcsin(NA / n)
    k   = 2.0 * np.pi / wl
    Tho = NA / n
    th1 = float(np.arcsin(e * Tho)) if e > 0 else 0.0
    th  = np.linspace(th1, a, N_TH)

    if plane == 'X-Y':
        X = np.linspace(-x_extent, x_extent, re)
        Y = np.linspace(-y_extent, y_extent, re)
        xg, yg = np.meshgrid(X, Y)
        ang = np.arctan2(yg, xg)
        r   = np.sqrt(xg**2 + yg**2)
        res = _xy(polarization, th, k, n, m, r, ang, float(z0))
        res['x_axis']  = (X / wl).tolist()
        res['y_axis']  = (Y / wl).tolist()
        res['x_label'] = 'X (λ)'
        res['y_label'] = 'Y (λ)'
    else:  # R-Z
        R = np.linspace(-x_extent, x_extent, re)
        Z = np.linspace(-z_extent, z_extent, re)
        rg, zg = np.meshgrid(R, Z)
        res = _rz(polarization, th, k, n, m, rg, zg, ang=0.0)
        res['x_axis']  = (R / wl).tolist()
        res['y_axis']  = (Z / wl).tolist()
        res['x_label'] = 'R (λ)'
        res['y_label'] = 'Z (λ)'

    return res


# ── XY plane ─────────────────────────────────────────────────────────────────

def _xy(pol, th, k, n, m, r, ang, z):
    t  = th[:, None, None]
    r3 = r[None, :, :]
    ct = np.cos(t);  st = np.sin(t)
    sc = ct ** 0.5
    ez = np.exp(-1j * k * z * n * ct)   # scalar z

    if pol == 'Linear':
        I0, I2, I3, I4, I5 = _lin_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**m*np.exp(1j*m*ang)*I0 - 0.5*(1j**(m+2)*np.exp(1j*(m+2)*ang)*I2
                                               + 1j**(m-2)*np.exp(1j*(m-2)*ang)*I3)
        Fy = 0.5*(-(1j**(m+2))*np.exp(1j*(m+2)*ang)*I2
                   + 1j**(m-2)*np.exp(1j*(m-2)*ang)*I3)
        Fz = (1j**(m+1)*np.exp(1j*(m+1)*ang)*I4
              + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I5)

    elif pol == 'Circular':
        I0, I1, I2 = _circ_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**(m+2)*np.exp(1j*(m+2)*ang)*I0 + 1j**m*np.exp(1j*m*ang)*I1
        Fy = -1j*(1j**(m+2)*np.exp(1j*(m+2)*ang)*I0 - 1j**m*np.exp(1j*m*ang)*I1)
        Fz = 2*1j**(m+1)*np.exp(1j*(m+1)*ang)*I2

    elif pol == 'Radial':
        I0, I1, I2 = _radial_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1
        Fy = -1j*(1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 - 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1)
        Fz = 4*1j**m*np.exp(1j*m*ang)*I2

    elif pol == 'Azimuthal':
        I0, I1 = _azim_integrals(t, sc, st, ez, k, n, m, r3, th)
        Fx = 1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1
        Fy = -1j*(1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 - 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1)
        Fz = np.zeros_like(Fx)

    return _pack(Fx, Fy, Fz)


# ── RZ plane ─────────────────────────────────────────────────────────────────

def _rz(pol, th, k, n, m, r, z, ang):
    t  = th[:, None, None]
    r3 = r[None, :, :]
    z3 = z[None, :, :]
    ct = np.cos(t);  st = np.sin(t)
    sc = ct ** 0.5
    ez = np.exp(-1j * k * z3 * n * ct)   # 3-D z

    if pol == 'Linear':
        I0, I2, I3, I4, I5 = _lin_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**m*np.exp(1j*m*ang)*I0 - 0.5*(1j**(m+2)*np.exp(1j*(m+2)*ang)*I2
                                               + 1j**(m-2)*np.exp(1j*(m-2)*ang)*I3)
        Fy = 0.5*(-(1j**(m+2))*np.exp(1j*(m+2)*ang)*I2
                   + 1j**(m-2)*np.exp(1j*(m-2)*ang)*I3)
        Fz = (1j**(m+1)*np.exp(1j*(m+1)*ang)*I4
              + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I5)

    elif pol == 'Circular':
        I0, I1, I2 = _circ_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**(m+2)*np.exp(1j*(m+2)*ang)*I0 + 1j**m*np.exp(1j*m*ang)*I1
        Fy = -1j*(1j**(m+2)*np.exp(1j*(m+2)*ang)*I0 - 1j**m*np.exp(1j*m*ang)*I1)
        Fz = 2*1j**(m+1)*np.exp(1j*(m+1)*ang)*I2

    elif pol == 'Radial':
        I0, I1, I2 = _radial_integrals(t, sc, st, ct, ez, k, n, m, r3, th)
        Fx = 1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1
        Fy = -1j*(1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 - 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1)
        Fz = 4*1j**m*np.exp(1j*m*ang)*I2

    elif pol == 'Azimuthal':
        I0, I1 = _azim_integrals(t, sc, st, ez, k, n, m, r3, th)
        Fx = 1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 + 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1
        Fy = -1j*(1j**(m+1)*np.exp(1j*(m+1)*ang)*I0 - 1j**(m-1)*np.exp(1j*(m-1)*ang)*I1)
        Fz = np.zeros_like(Fx)

    return _pack(Fx, Fy, Fz)


# ── Shared integrand helpers ──────────────────────────────────────────────────

def _lin_integrals(t, sc, st, ct, ez, k, n, m, r3, th):
    B = lambda order: jv(order, k * r3 * n * st)
    I0 = _integrate(sc * st * (1+ct) * B(m)   * ez, th)
    I2 = _integrate(sc * st * (1-ct) * B(m+2) * ez, th)
    I3 = _integrate(sc * st * (1-ct) * B(m-2) * ez, th)
    I4 = _integrate(sc * st**2       * B(m+1) * ez, th)
    I5 = _integrate(sc * st**2       * B(m-1) * ez, th)
    return I0, I2, I3, I4, I5

def _circ_integrals(t, sc, st, ct, ez, k, n, m, r3, th):
    B = lambda order: jv(order, k * r3 * n * st)
    I0 = _integrate(sc * st * (ct-1) * B(m+2) * ez, th)
    I1 = _integrate(sc * st * (ct+1) * B(m)   * ez, th)
    I2 = _integrate(sc * st**2       * B(m+1) * ez, th)
    return I0, I1, I2

def _radial_integrals(t, sc, st, ct, ez, k, n, m, r3, th):
    B = lambda order: jv(order, k * r3 * n * st)
    sin2t = np.sin(2*t)
    I0 = _integrate(sc * sin2t * B(m+1) * ez, th)
    I1 = _integrate(sc * sin2t * B(m-1) * ez, th)
    I2 = _integrate(sc * st**2 * B(m)   * ez, th)
    return I0, I1, I2

def _azim_integrals(t, sc, st, ez, k, n, m, r3, th):
    B = lambda order: jv(order, k * r3 * n * st)
    I0 = _integrate(sc * st * B(m+1) * ez, th)
    I1 = _integrate(sc * st * B(m-1) * ez, th)
    return I0, I1


# ── Pack results ─────────────────────────────────────────────────────────────

def _pack(Fx, Fy, Fz):
    Ix = np.abs(Fx)**2
    Iy = np.abs(Fy)**2
    Iz = np.abs(Fz)**2
    I  = Ix + Iy + Iz
    s  = I.max() if I.max() > 0 else 1.0
    return {
        'I':  (I  / s).tolist(),
        'Ix': (Ix / s).tolist(),
        'Iy': (Iy / s).tolist(),
        'Iz': (Iz / s).tolist(),
    }
