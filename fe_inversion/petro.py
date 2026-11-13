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


CSH_RELATIONS = ("linear", "larionov_tertiary", "larionov_older", "clavier", "stieber")


def csh_from_ish(ish, method):
    """Csh from the shale index (book §8.5). Linear holds only for laminated sandstone-shale systems."""
    i = np.clip(np.asarray(ish, float), 0.0, 1.0)
    if method == "linear":
        return i
    if method == "larionov_tertiary":
        return 0.083 * (2.0 ** (3.7 * i) - 1.0)
    if method == "larionov_older":
        return 0.33 * (2.0 ** (2.0 * i) - 1.0)
    if method == "clavier":
        return 1.7 - np.sqrt(3.38 - (i + 0.7) ** 2)
    if method == "stieber":
        return i / (3.0 - 2.0 * i)
    raise ValueError(f"unknown Csh relation {method!r}")


def csh_neutron_density(phi_n, phi_d, phi_n_sh, phi_d_sh):
    """Csh from neutron-density separation, valid only in water-saturated sandstone (book §8.5)."""
    return (np.asarray(phi_n) - np.asarray(phi_d)) / (phi_n_sh - phi_d_sh)


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


# ------------------------------------------------- grain-coating clay (dispersed)
def rms_porosity(phi_d, phi_n):
    """Neutron-density porosity, sqrt[(phiD^2 + phiN^2) / 2] (book eq. 9.6.1)."""
    return np.sqrt((np.asarray(phi_d) ** 2 + np.asarray(phi_n) ** 2) / 2.0)


def grain_coating_porosity(phi_d, phi_n, csh, phi_d_sh, phi_n_sh, phi_sh):
    """Non-shale and total porosity with grain-coating clay (book A.11.4-6, 10.15.12).

    phiDc = phiD - Csh*phiD_sh, phiNc = phiN - Csh*phiN_sh, phi_s = RMS(phiDc, phiNc),
    phi_t = phi_s + Csh*phi_sh. Returns (phi_s, phi_t).
    """
    phi_dc = np.maximum(np.asarray(phi_d) - csh * phi_d_sh, 0.0)
    phi_nc = np.maximum(np.asarray(phi_n) - csh * phi_n_sh, 0.0)
    phi_s = rms_porosity(phi_dc, phi_nc)
    return phi_s, phi_s + csh * phi_sh


def bound_water_fraction(csh, phi_sh, phi_t):
    """Sb = Csh*phi_sh/phi_t, also Juhasz's normalized Qv (book 10.15.13, A.14.5)."""
    return csh * phi_sh / phi_t


def waxman_smits_b(temp_f, rw):
    """Equivalent counterion conductance B for T in degF and Rw in ohm.m (book A.14.4)."""
    return (-5.41 + 0.133 * temp_f - 0.0001253 * temp_f ** 2) / (1.0 + rw ** 1.23 * (0.025 * temp_f - 1.07))


def juhasz_rt(rw, phi_t, sw, sb, b, a, m, n):
    """Juhasz: 1/Rt = (phi_t^m Sw^n / a) (1/Rw + B Sb / Sw) (book A.14.6)."""
    return a / (np.power(phi_t, m) * np.power(sw, n) * (1.0 / rw + b * sb / sw))


def juhasz_sw(rt, rw, phi_t, sb, b, a, m, n, iterations=200, tol=1e-7):
    """Total water saturation by nested iteration from the Archie value (book A.14.a), limited to Sb..1."""
    rt, phi_t, sb = np.broadcast_arrays(*(np.asarray(x, float) for x in (rt, phi_t, sb)))
    lo = np.maximum(sb, 1e-3)
    sw = np.clip(archie_sw(rt, rw, phi_t, a, m, n), lo, 1.0)
    for _ in range(iterations):
        with np.errstate(invalid="ignore", divide="ignore"):
            new = np.power(a / (np.power(phi_t, m) * rt * (1.0 / rw + b * sb / sw)), 1.0 / n)
        new = np.clip(0.5 * (sw + new), lo, 1.0)
        done = np.nanmax(np.abs(new - sw)) < tol
        sw = new
        if done:
            break
    return sw


def juhasz_rw(rt, phi_t, sb, temp_f, a, m, iterations=100):
    """Rw from a water-saturated rock (Sw = 1): 1/Rw = phi-normalized conductivity minus B*Sb, B depending on Rw."""
    rt, phi_t, sb = (np.asarray(x, float) for x in (rt, phi_t, sb))
    cond = a / (rt * np.power(phi_t, m))      # equals 1/Rw + B*Sb at Sw = 1
    rw = 1.0 / cond
    for _ in range(iterations):
        c_w = np.maximum(cond - waxman_smits_b(temp_f, rw) * sb, 1e-3)
        rw = 0.5 * (rw + 1.0 / c_w)
    return rw


def sandstone_sw(sw_t, sb):
    """Non-shale water saturation, Sws = (Sw - Sb)/(1 - Sb) (book 10.15.21)."""
    return (np.asarray(sw_t) - sb) / (1.0 - sb)


def rw_at_temperature(r1, t1, t2):
    """Arps: resistivity of an electrolyte moved from t1 to t2, both in degF."""
    return r1 * (t1 + 6.77) / (t2 + 6.77)


def wyllie_porosity(dt, dt_ma, dt_f, cp=1.0):
    """Sonic porosity with compaction coefficient Cp (book A.15.2-3)."""
    return (np.asarray(dt) - dt_ma) / (dt_f - dt_ma) / cp


def dispersed_forward(phi_s, sws, sxos, csh, p):
    """Predicted (RHOB, NPHI, Rt, Rxo) of a sandstone with grain-coating clay (book A.9.1, A.14.6).

    phi_s is non-shale porosity; sws and sxos are non-shale water saturations in the uninvaded and
    flushed zones. The clay fraction Csh carries the reference shale's density, neutron and porosity.
    Density and neutron read the flushed zone (mud filtrate + residual hydrocarbon); Rt reads Sw, Rxo reads Sxo.
    """
    rho_f = sxos * p["rho_mf"] + (1 - sxos) * p["rho_h"]
    rhob = phi_s * rho_f + (1 - phi_s - csh) * p["rho_ma"] + csh * p["rho_sh"]
    hi_f = sxos + (1 - sxos) * p["hi_h"]
    nphi = phi_s * hi_f + (1 - phi_s - csh) * p["nphi_ma"] + csh * p["nphi_sh"]
    phi_t = phi_s + csh * p["phi_sh"]
    sb = csh * p["phi_sh"] / phi_t
    sw_t = (sws * phi_s + csh * p["phi_sh"]) / phi_t
    sxo_t = (sxos * phi_s + csh * p["phi_sh"]) / phi_t
    rt = juhasz_rt(p["rw"], phi_t, sw_t, sb, p["b_w"], p["a"], p["m"], p["n"])
    rxo = juhasz_rt(p["rmf"], phi_t, sxo_t, sb, p["b_mf"], p["a"], p["m"], p["n"])
    return rhob, nphi, rt, rxo


# ------------------------------------------------------------------ permeability
MD_TO_UM2 = 9.869e-4  # 1 md in square micrometers


def throat_radius_um(k_md, phi):
    """Equivalent throat radius of a bundle of equal capillary tubes, R = sqrt(8 k / phi) (book eq. 3.7.3)."""
    return np.sqrt(8.0 * np.asarray(k_md) * MD_TO_UM2 / np.asarray(phi))


def timur_k(phi, swi, alpha, beta, gamma):
    """Timur / Tixier form, k = (alpha phi^beta / Swi)^gamma, phi and Swi as fractions, k in md (book 12.2.2)."""
    return (alpha * np.power(phi, beta) / np.asarray(swi)) ** gamma


def coates_k(phi, swi, alpha, beta, gamma):
    """Coates form, k = (alpha phi^beta (1 - Swi) / Swi)^gamma (book 12.2.3)."""
    return (alpha * np.power(phi, beta) * (1.0 - np.asarray(swi)) / np.asarray(swi)) ** gamma


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
