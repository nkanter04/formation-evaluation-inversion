"""Deterministic petrophysics: shale volume, porosity, saturation, pressures.

Conventions: porosities and saturations are fractions (v/v), densities in g/cc,
resistivities in ohm.m, depths in ft. Laminated sand-shale model:

    1/Rh = Csh/Rsh_h + (1 - Csh)/Rss      (layers in parallel)
    Rv   = Csh*Rsh_v + (1 - Csh)*Rss      (layers in series)
"""
import numpy as np

PSI_PER_FT_PER_GCC = 0.4335  # pressure gradient of a 1 g/cc fluid


# ---------------------------------------------------------------- shale volume
def gr_index(gr, gr_sand, gr_shale):
    return np.clip((np.asarray(gr) - gr_sand) / (gr_shale - gr_sand), 0.0, 1.0)


def csh_larionov_tertiary(igr):
    """Larionov (1969) correction for unconsolidated Tertiary rocks."""
    return 0.083 * (2.0 ** (3.7 * np.asarray(igr)) - 1.0)


def neutron_density_ternary(rhob, nphi, sand, shale, fluid):
    """Solve sand-grain / shale / pore-fluid volumes from density and neutron.

    `sand`, `shale`, `fluid` are (density, neutron) points; any coordinate may be
    an array over samples (depth-varying end-points). Returns (v_sand, v_shale,
    phi) for every sample; the three always sum to 1 and are not clipped.
    """
    rhob, nphi = np.asarray(rhob, float), np.asarray(nphi, float)
    n = rhob.size
    col = lambda v: np.broadcast_to(np.asarray(v, float), (n,))
    A = np.stack([np.stack([col(sand[0]), col(shale[0]), col(fluid[0])], -1),
                  np.stack([col(sand[1]), col(shale[1]), col(fluid[1])], -1),
                  np.ones((n, 3))], axis=1)
    b = np.stack([rhob, nphi, np.ones(n)], -1)
    v = np.linalg.solve(A, b[..., None])[..., 0]
    return v[:, 0], v[:, 1], v[:, 2]


def laminated_resistivity(rh, rv, rsh_h, rsh_v):
    """Closed-form laminated Csh and sand resistivity Rss from Rh and Rv.

    Substituting Rss = (Rv - C*Rsh_v)/(1 - C) into the parallel equation gives
    C^2 (Rsh_v/Rsh_h - 1) + C (2 - Rsh_v/Rh - Rv/Rsh_h) + (Rv/Rh - 1) = 0.
    The root inside [0, 1] that yields a positive Rss is returned (NaN if none).
    Shale resistivities may be scalars or arrays over samples.
    """
    rh, rv, rsh_h, rsh_v = np.broadcast_arrays(*(np.asarray(v, float) for v in (rh, rv, rsh_h, rsh_v)))
    qa = rsh_v / rsh_h - 1.0
    qb = 2.0 - rsh_v / rh - rv / rsh_h
    qc = rv / rh - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        linear = np.abs(qa) < 1e-12  # isotropic shale: the quadratic degenerates
        disc = np.sqrt(qb * qb - 4 * qa * qc)
        roots = [np.where(linear, -qc / qb, (-qb + disc) / (2 * qa)),
                 np.where(linear, np.nan, (-qb - disc) / (2 * qa))]
        csh = np.full(rh.shape, np.nan)
        rss = np.full(rh.shape, np.nan)
        for r in roots:
            r_rss = (rv - r * rsh_v) / (1.0 - r)
            ok = np.isnan(csh) & (r >= 0) & (r < 1) & (r_rss > 0)
            csh[ok], rss[ok] = r[ok], r_rss[ok]
        # Rv/Rh below the shale anisotropy line: no laminated solution, treat as shale.
    return csh, rss


# ------------------------------------------------------------------- porosity
def sand_density(rhob, csh, rho_sh):
    """Density of the sand laminae after removing the shale laminae."""
    return (np.asarray(rhob) - csh * rho_sh) / (1.0 - csh)


def density_porosity(rho, rho_ma, rho_f):
    return (rho_ma - np.asarray(rho)) / (rho_ma - rho_f)


def fluid_density_from_core(rho_sand, phi_core, rho_grain):
    """Apparent pore-fluid density from log sand density and core porosity/grain density."""
    return rho_grain - (rho_grain - np.asarray(rho_sand)) / np.asarray(phi_core)


def thomas_stieber_lines(phi_clean, phi_sh, n=50):
    """Total porosity vs. Csh for laminar, dispersed and structural shale."""
    c = np.linspace(0, 1, n)
    laminar = (c, (1 - c) * phi_clean + c * phi_sh)
    cd = np.linspace(0, phi_clean, n)
    dispersed = (cd, phi_clean - cd * (1 - phi_sh))
    cs = np.linspace(0, 1 - phi_clean, n)
    structural = (cs, phi_clean + cs * phi_sh)
    return {"laminar": laminar, "dispersed": dispersed, "structural": structural}


# ----------------------------------------------------------------- saturation
def archie_rt(rw, phi, sw, a, m, n):
    return a * rw / (np.power(phi, m) * np.power(sw, n))


def archie_sw(rt, rw, phi, a, m, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.power(a * rw / (np.power(phi, m) * np.asarray(rt)), 1.0 / n)


def archie_rw(rt, phi, a, m):
    """Water resistivity from a 100 % water-saturated rock (Sw = 1)."""
    return np.asarray(rt) * np.power(phi, m) / a


def sand_rxo_parallel(rxo, csh, rsh_h):
    """Flushed-zone sand resistivity, treating the shallow reading as parallel to bedding."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = 1.0 / np.asarray(rxo) - csh / rsh_h
        return np.where(cond > 0, (1.0 - csh) / cond, np.nan)


# ----------------------------------------------------------- well trajectory
def tvd_minimum_curvature(md, inc_deg, azi_deg):
    """True vertical depth relative to the first sample, by minimum curvature."""
    md = np.asarray(md, float)
    inc, azi = np.radians(np.asarray(inc_deg, float)), np.radians(np.asarray(azi_deg, float))
    i1, i2, a1, a2 = inc[:-1], inc[1:], azi[:-1], azi[1:]
    cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
    dl = np.arccos(np.clip(cos_dl, -1, 1))
    rf = np.where(dl > 1e-9, 2 / np.where(dl > 1e-9, dl, 1) * np.tan(dl / 2), 1.0)
    dtvd = np.diff(md) / 2 * (np.cos(i1) + np.cos(i2)) * rf
    return np.r_[0.0, np.cumsum(dtvd)]


# -------------------------------------------------------------- pressures
def fit_gradient(tvd, pressure):
    """Least-squares line P = P0 + g*TVD. Returns slope (psi/ft), intercept and residual std."""
    tvd, pressure = np.asarray(tvd), np.asarray(pressure)
    A = np.vstack([tvd, np.ones_like(tvd)]).T
    (g, p0), *_ = np.linalg.lstsq(A, pressure, rcond=None)
    resid = pressure - (g * tvd + p0)
    dof = max(len(tvd) - 2, 1)
    s = np.sqrt(resid @ resid / dof)
    # Standard error of the slope.
    se = s / np.sqrt(np.sum((tvd - tvd.mean()) ** 2)) if len(tvd) > 2 else np.nan
    return dict(gradient=g, intercept=p0, resid_std=s, gradient_se=se,
                density=g / PSI_PER_FT_PER_GCC, density_se=se / PSI_PER_FT_PER_GCC, n=len(tvd))
