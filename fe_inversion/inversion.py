"""Laminated sand-shale inversion with Monte Carlo uncertainty.

Two methods share the forward model below.

"sequential" (default, the course workflow):
  1. Depth by depth, solve Rh and Rv for Csh and the sand resistivity Rss,
     with 0 <= Csh <= 1 and the spectral-GR Csh as a soft bias.
  2. With Csh fixed, solve density, neutron and Rss (through Archie) for
     sand porosity phi_ss and water saturation Sw, with 0 < Sw <= 1.

"joint": solve Csh, phi_ss and Sw together from Rh, Rv, RHOB and NPHI.

Forward model:

    rho_f = Sw*rho_w + (1 - Sw)*rho_hc
    RHOB  = Csh*rho_sh + (1 - Csh)*[phi*rho_f + (1 - phi)*rho_ma]
    NPHI  = Csh*nphi_sh + (1 - Csh)*[phi*(Sw + (1 - Sw)*HI_hc) + (1 - phi)*nphi_ma]
    Rss   = a*Rw / (phi^m * Sw^n)                                   (Archie)
    Rh    = 1 / [Csh/Rsh_h + (1 - Csh)/Rss],  Rv = Csh*Rsh_v + (1 - Csh)*Rss

The GR-derived Csh is the starting model and a Gaussian prior. Every Monte
Carlo realization draws new end-points / fluid properties, perturbs the
measurements with their noise and the prior with its spread, and solves the
regularized least-squares problem (randomized maximum likelihood). The spread
of the solutions approximates the posterior uncertainty.
"""
import numpy as np

from . import petro

BOUNDS = np.array([[0.0, 1.0],     # Csh
                   [0.02, 0.42],   # phi_ss
                   [0.02, 1.0]])   # Sw_ss
RES_BOUNDS = np.array([[0.0, 1.0],                        # Csh
                       [np.log(0.01), np.log(2000.0)]])   # ln Rss
SAT_BOUNDS = BOUNDS[1:]                                   # phi_ss, Sw_ss
U_DAMPING = 0.02                   # keeps the bounded transform away from saturation
LOG_PARAMS = ("rsh_h", "rsh_v", "rw", "rmf")


def to_x(u, bounds=BOUNDS):
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) / (1.0 + np.exp(-np.clip(u, -50, 50)))


def to_u(x, bounds=BOUNDS):
    f = (np.asarray(x) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    f = np.clip(f, 1e-4, 1 - 1e-4)
    return np.log(f / (1 - f))


def _dx_du(u, bounds):
    f = 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))
    return (bounds[:, 1] - bounds[:, 0]) * f * (1 - f)


def _noise(s):
    """Noise of Rh and Rv in natural-log units (Rv is noisier: fewer coils)."""
    return s.get("sigma_log_rh", s.get("sigma_log_r")), s.get("sigma_log_rv", s.get("sigma_log_r"))


def lm(resid, u0, iterations=40, tol=1e-8):
    """Batched Levenberg-Marquardt. `resid(u)` returns residuals (..., m) and Jacobian (..., m, k)."""
    u = np.array(u0, float)
    k = u.shape[-1]
    r, J = resid(u)
    cost = np.sum(r * r, axis=-1)
    mu = np.full(cost.shape, 1e-3)
    eye = np.eye(k)
    for _ in range(iterations):
        Jt = np.swapaxes(J, -1, -2)
        JTJ = Jt @ J
        g = (Jt @ r[..., None])[..., 0]
        A = JTJ + mu[..., None, None] * (JTJ * eye + 1e-9 * eye)
        u_new = u - np.linalg.solve(A, g[..., None])[..., 0]
        r_new, J_new = resid(u_new)
        cost_new = np.sum(r_new * r_new, axis=-1)
        better = cost_new < cost
        gain = np.max(np.where(better, cost - cost_new, 0.0))
        u = np.where(better[..., None], u_new, u)
        r = np.where(better[..., None], r_new, r)
        J = np.where(better[..., None, None], J_new, J)
        cost = np.where(better, cost_new, cost)
        mu = np.clip(np.where(better, mu * 0.2, mu * 5.0), 1e-9, 1e9)
        if gain < tol:
            break
    return u, r, cost


def forward(csh, phi, sw, p):
    """Predicted (Rh, Rv, RHOB, NPHI) for the laminated model."""
    rho_f = sw * p["rho_w"] + (1 - sw) * p["rho_hc"]
    rho_ss = phi * rho_f + (1 - phi) * p["rho_ma"]
    rhob = csh * p["rho_sh"] + (1 - csh) * rho_ss
    nphi_ss = phi * (sw + (1 - sw) * p["hi_hc"]) + (1 - phi) * p["nphi_ma"]
    nphi = csh * p["nphi_sh"] + (1 - csh) * nphi_ss
    rss = petro.archie_rt(p["rw"], phi, sw, p["a"], p["m"], p["n"])
    rh = 1.0 / (csh / p["rsh_h"] + (1 - csh) / rss)
    rv = csh * p["rsh_v"] + (1 - csh) * rss
    return rh, rv, rhob, nphi


def _residuals(u, obs, p, prior, s, jacobian=False):
    """Weighted residuals (..., 10) and, optionally, their analytic Jacobian (..., 10, 3) in u."""
    x = to_x(u)
    c, phi, sw = x[..., 0], x[..., 1], x[..., 2]
    rho_f = sw * p["rho_w"] + (1 - sw) * p["rho_hc"]
    rho_ss = phi * rho_f + (1 - phi) * p["rho_ma"]
    rhob = c * p["rho_sh"] + (1 - c) * rho_ss
    hi_f = sw + (1 - sw) * p["hi_hc"]
    nphi_ss = phi * hi_f + (1 - phi) * p["nphi_ma"]
    nphi = c * p["nphi_sh"] + (1 - c) * nphi_ss
    rss = petro.archie_rt(p["rw"], phi, sw, p["a"], p["m"], p["n"])
    sig_h = c / p["rsh_h"] + (1 - c) / rss
    rv = c * p["rsh_v"] + (1 - c) * rss

    (sr, sv), sd, sn, sc = _noise(s), s["sigma_rhob"], s["sigma_nphi"], s["csh_prior_std"]
    (phi0, sphi), (sw0, ssw) = s["phi_prior"], s["sw_prior"]
    valid = [np.isfinite(obs[k]) for k in ("log_rh", "log_rv", "rhob", "nphi")]
    r = np.empty(c.shape + (10,))
    r[..., 0] = (-np.log(sig_h) - obs["log_rh"]) / sr
    r[..., 1] = (np.log(rv) - obs["log_rv"]) / sv
    r[..., 2] = (rhob - obs["rhob"]) / sd
    r[..., 3] = (nphi - obs["nphi"]) / sn
    r[..., 4] = (c - prior["csh"]) / sc
    r[..., 5] = (phi - phi0) / sphi
    r[..., 6] = (sw - sw0) / ssw
    r[..., 7:] = U_DAMPING * u
    for i, ok in enumerate(valid):  # missing or flagged measurements contribute nothing
        r[..., i] = np.where(ok, r[..., i], 0.0)
    if not jacobian:
        return r

    # d(ln Rss)/d(phi) = -m/phi and d(ln Rss)/d(Sw) = -n/Sw (Archie).
    dl_phi, dl_sw = -p["m"] / phi, -p["n"] / sw
    k_h = (1 - c) / (rss * sig_h)
    k_v = (1 - c) * rss / rv
    Jx = np.zeros(c.shape + (10, 3))
    Jx[..., 0, 0] = -(1 / p["rsh_h"] - 1 / rss) / sig_h / sr
    Jx[..., 0, 1] = k_h * dl_phi / sr
    Jx[..., 0, 2] = k_h * dl_sw / sr
    Jx[..., 1, 0] = (p["rsh_v"] - rss) / rv / sv
    Jx[..., 1, 1] = k_v * dl_phi / sv
    Jx[..., 1, 2] = k_v * dl_sw / sv
    Jx[..., 2, 0] = (p["rho_sh"] - rho_ss) / sd
    Jx[..., 2, 1] = (1 - c) * (rho_f - p["rho_ma"]) / sd
    Jx[..., 2, 2] = (1 - c) * phi * (p["rho_w"] - p["rho_hc"]) / sd
    Jx[..., 3, 0] = (p["nphi_sh"] - nphi_ss) / sn
    Jx[..., 3, 1] = (1 - c) * (hi_f - p["nphi_ma"]) / sn
    Jx[..., 3, 2] = (1 - c) * phi * (1 - p["hi_hc"]) / sn
    Jx[..., 4, 0] = 1 / sc
    Jx[..., 5, 1] = 1 / sphi
    Jx[..., 6, 2] = 1 / ssw
    for i, ok in enumerate(valid):
        Jx[..., i, :] *= ok[..., None]
    J = Jx * _dx_du(u, BOUNDS)[..., None, :]
    J[..., 7, 0] = J[..., 8, 1] = J[..., 9, 2] = U_DAMPING
    return r, J


def solve(obs, p, prior, s, x0, iterations=40, tol=1e-8):
    """Joint method: solve (Csh, phi_ss, Sw) over arrays of shape (realizations, depths)."""
    u, r, cost = lm(lambda u: _residuals(u, obs, p, prior, s, jacobian=True),
                    to_u(x0), iterations, tol)
    data_misfit = np.sqrt(np.mean(r[..., :4] ** 2, axis=-1))
    return to_x(u), cost, data_misfit


# ------------------------------------------------------------ sequential, step 1
def _res_stage1(u, obs, p, prior, s):
    """Rh, Rv residuals for unknowns (Csh, ln Rss); optional bias toward the spectral-GR Csh.

    Residuals: Rh, Rv, Csh bias, weak ln Rss prior, two damping terms. With
    prior=None (shale bracketing) the Csh bias is switched off.
    """
    x = to_x(u, RES_BOUNDS)
    c, t = x[..., 0], x[..., 1]
    rss = np.exp(t)
    sig_h = c / p["rsh_h"] + (1 - c) / rss
    rv = c * p["rsh_v"] + (1 - c) * rss
    sr, sv = _noise(s)
    t0, st = np.log(s.get("rss_prior", 3.0)), s.get("log_rss_prior_std", 4.0)
    ok_h, ok_v = np.isfinite(obs["log_rh"]), np.isfinite(obs["log_rv"])
    w = 0.0 if prior is None else 1.0 / s["csh_prior_std"]
    target = 0.0 if prior is None else prior["csh"]

    r = np.empty(c.shape + (6,))
    r[..., 0] = np.where(ok_h, (-np.log(sig_h) - obs["log_rh"]) / sr, 0.0)
    r[..., 1] = np.where(ok_v, (np.log(rv) - obs["log_rv"]) / sv, 0.0)
    r[..., 2] = (c - target) * w
    r[..., 3] = (t - t0) / st
    r[..., 4:] = U_DAMPING * u
    Jx = np.zeros(c.shape + (6, 2))
    Jx[..., 0, 0] = -(1 / p["rsh_h"] - 1 / rss) / sig_h / sr * ok_h
    Jx[..., 0, 1] = (1 - c) / (rss * sig_h) / sr * ok_h
    Jx[..., 1, 0] = (p["rsh_v"] - rss) / rv / sv * ok_v
    Jx[..., 1, 1] = (1 - c) * rss / rv / sv * ok_v
    Jx[..., 2, 0] = w
    Jx[..., 3, 1] = 1 / st
    J = Jx * _dx_du(u, RES_BOUNDS)[..., None, :]
    J[..., 4, 0] = J[..., 5, 1] = U_DAMPING
    return r, J


def solve_stage1(obs, p, prior, s, x0, iterations=40):
    """Csh and Rss from Rh and Rv. Returns csh, rss, cost and RMS resistivity misfit."""
    u, r, cost = lm(lambda u: _res_stage1(u, obs, p, prior, s), to_u(x0, RES_BOUNDS), iterations)
    x = to_x(u, RES_BOUNDS)
    return x[..., 0], np.exp(x[..., 1]), cost, np.sqrt(np.mean(r[..., :2] ** 2, axis=-1))


def stage1_start(log_rh, log_rv, p, csh_guess):
    """Starting model: closed-form laminated solution where it exists, else the Csh guess."""
    c, rss = petro.laminated_resistivity(np.exp(log_rh), np.exp(log_rv), p["rsh_h"], p["rsh_v"])
    c = np.where(np.isfinite(c), c, csh_guess)
    rss = np.where(np.isfinite(rss), rss, np.exp(log_rv))
    rss = np.nan_to_num(rss, nan=3.0)
    return np.stack([np.clip(c, 0.02, 0.98), np.log(np.clip(rss, 0.02, 1000.0))], -1)


# ------------------------------------------------------------ sequential, step 2
def _res_stage2(u, obs, p, csh, s):
    """Density, neutron and Rss residuals for unknowns (phi_ss, Sw) with Csh fixed."""
    x = to_x(u, SAT_BOUNDS)
    phi, sw = x[..., 0], x[..., 1]
    c = csh
    rho_f = sw * p["rho_w"] + (1 - sw) * p["rho_hc"]
    rho_ss = phi * rho_f + (1 - phi) * p["rho_ma"]
    rhob = c * p["rho_sh"] + (1 - c) * rho_ss
    hi_f = sw + (1 - sw) * p["hi_hc"]
    nphi = c * p["nphi_sh"] + (1 - c) * (phi * hi_f + (1 - phi) * p["nphi_ma"])
    lrss = np.log(petro.archie_rt(p["rw"], phi, sw, p["a"], p["m"], p["n"]))
    sd, sn, ss = s["sigma_rhob"], s["sigma_nphi"], s.get("sigma_log_rss", 0.10)
    (phi0, sphi), (sw0, ssw) = s["phi_prior"], s["sw_prior"]
    valid = [np.isfinite(obs[k]) for k in ("rhob", "nphi", "log_rss")]

    r = np.empty(c.shape + (7,))
    r[..., 0] = (rhob - obs["rhob"]) / sd
    r[..., 1] = (nphi - obs["nphi"]) / sn
    r[..., 2] = (lrss - obs["log_rss"]) / ss
    r[..., 3] = (phi - phi0) / sphi
    r[..., 4] = (sw - sw0) / ssw
    r[..., 5:] = U_DAMPING * u
    for i, ok in enumerate(valid):
        r[..., i] = np.where(ok, r[..., i], 0.0)
    Jx = np.zeros(c.shape + (7, 2))
    Jx[..., 0, 0] = (1 - c) * (rho_f - p["rho_ma"]) / sd
    Jx[..., 0, 1] = (1 - c) * phi * (p["rho_w"] - p["rho_hc"]) / sd
    Jx[..., 1, 0] = (1 - c) * (hi_f - p["nphi_ma"]) / sn
    Jx[..., 1, 1] = (1 - c) * phi * (1 - p["hi_hc"]) / sn
    Jx[..., 2, 0] = -p["m"] / phi / ss
    Jx[..., 2, 1] = -p["n"] / sw / ss
    Jx[..., 3, 0] = 1 / sphi
    Jx[..., 4, 1] = 1 / ssw
    for i, ok in enumerate(valid):
        Jx[..., i, :] *= ok[..., None]
    J = Jx * _dx_du(u, SAT_BOUNDS)[..., None, :]
    J[..., 5, 0] = J[..., 6, 1] = U_DAMPING
    return r, J


def solve_stage2(obs, p, csh, s, x0, iterations=40):
    """phi_ss and Sw with Csh fixed. Returns phi, sw, cost and RMS density/neutron/Rss misfit."""
    u, r, cost = lm(lambda u: _res_stage2(u, obs, p, csh, s), to_u(x0, SAT_BOUNDS), iterations)
    x = to_x(u, SAT_BOUNDS)
    return x[..., 0], x[..., 1], cost, np.sqrt(np.mean(r[..., :3] ** 2, axis=-1))


def sample_parameters(means, stds, k, rng, fixed):
    """Draw k parameter sets. Log-normal for resistivities, normal otherwise.

    A mean may be a scalar or an array over depths (zoned shale end-points). Each
    realization draws one standard-normal shift per parameter and applies it at
    every depth, so the result has shape (k, 1) or (k, depths).
    """
    p = {}
    for name, mean in means.items():
        sd = stds.get(name, 0.0)
        z = rng.normal(0.0, 1.0, (k, 1))
        mean = np.asarray(mean, float)[None, ...] if np.ndim(mean) else mean
        p[name] = mean * np.exp(sd * z) if name in LOG_PARAMS else mean + sd * z
    p["rho_hc"] = np.clip(p["rho_hc"], 0.05, 1.0)
    p["hi_hc"] = np.clip(p["hi_hc"], 0.05, 1.1)
    for sand, shale in (("gr_sand", "gr_shale"), ("spec_sand", "spec_shale")):
        if sand in p and shale in p:
            span = np.min(np.abs(np.asarray(means[shale]) - np.asarray(means[sand])))
            p[shale] = np.maximum(p[shale], p[sand] + 0.2 * span)
    p.update({key: val for key, val in fixed.items()})
    return p


OUTPUT_KEYS = ("csh", "phi", "sw", "rss", "hpv", "phit", "sxo", "misfit", "csh_prior",
               "pred_rh", "pred_rv", "pred_rhob", "pred_nphi")


def _solve_chunk(logs, means, stds, archie, s, k, seed):
    """Draw k realizations with their own random stream and invert them."""
    rng = np.random.default_rng(seed)
    n_depth = len(logs["GR"])
    p = sample_parameters(means, stds, k, rng, archie)
    noise = lambda sd: rng.normal(0.0, sd, (k, n_depth))
    sr, sv = _noise(s)
    obs = {
        "log_rh": np.log(logs["RH"]) + noise(sr),
        "log_rv": np.log(logs["RV"]) + noise(sv),
        "rhob": logs["RHOB"] + noise(s["sigma_rhob"]),
        "nphi": logs["NPHI"] + noise(s["sigma_nphi"]),
    }
    if s.get("method", "sequential") == "sequential":
        return _sequential_chunk(logs, obs, p, s, noise, means)
    csh_gr = petro.gr_index(logs["GR"][None, :], p["gr_sand"], p["gr_shale"])
    prior = {"csh": np.clip(csh_gr + noise(s["csh_prior_std"]), 0.0, 1.0)}

    # Starting model: GR Csh and density porosity of the sand laminae. Short solves
    # from a hydrocarbon-like and a water-like Sw, then refine the better of the two.
    rho_ss = petro.sand_density(np.nan_to_num(logs["RHOB"], nan=2.3)[None, :],
                                np.minimum(csh_gr, 0.9), p["rho_sh"])
    phi0 = np.clip(petro.density_porosity(rho_ss, p["rho_ma"], 0.9), 0.05, 0.38)
    best = None
    for sw0 in (0.3, 0.95):
        x0 = np.stack([np.clip(csh_gr, 0.02, 0.98), phi0, np.full_like(phi0, sw0)], -1)
        x, cost, _ = solve(obs, p, prior, s, x0, s.get("screen_iterations", 12))
        best = x if best is None else np.where((cost < best_cost)[..., None], x, best)
        best_cost = cost if sw0 == 0.3 else np.minimum(cost, best_cost)
    x, _, misfit = solve(obs, p, prior, s, best, s["iterations"])

    csh, phi, sw = x[..., 0], x[..., 1], x[..., 2]
    rss = petro.archie_rt(p["rw"], phi, sw, p["a"], p["m"], p["n"])
    return _outputs(logs, p, s, means, csh, phi, sw, rss, misfit, prior["csh"])


def _sequential_chunk(logs, obs, p, s, noise, means):
    """Step 1: Csh and Rss from Rh, Rv (biased toward the spectral-GR Csh). Step 2: phi, Sw."""
    csh_spec = petro.gr_index(logs["SPEC"][None, :], p["spec_sand"], p["spec_shale"])
    prior = {"csh": np.clip(csh_spec + noise(s["csh_prior_std"]), 0.0, 1.0)}
    x0 = stage1_start(obs["log_rh"], obs["log_rv"], p, csh_spec)
    csh, rss, _, mis_r = solve_stage1(obs, p, prior, s, x0, s["iterations"])

    obs2 = {"rhob": obs["rhob"], "nphi": obs["nphi"], "log_rss": np.log(rss)}
    rho_ss = petro.sand_density(np.nan_to_num(logs["RHOB"], nan=2.3)[None, :],
                                np.minimum(csh, 0.9), p["rho_sh"])
    phi0 = np.clip(petro.density_porosity(rho_ss, p["rho_ma"], 0.9), 0.05, 0.38)
    best = None
    for sw0 in (0.3, 0.95):
        x0 = np.stack([phi0, np.full_like(phi0, sw0)], -1)
        phi, sw, cost, _ = solve_stage2(obs2, p, csh, s, x0, s.get("screen_iterations", 12))
        x = np.stack([phi, sw], -1)
        best = x if best is None else np.where((cost < best_cost)[..., None], x, best)
        best_cost = cost if sw0 == 0.3 else np.minimum(cost, best_cost)
    phi, sw, _, mis_n = solve_stage2(obs2, p, csh, s, best, s["iterations"])
    # RMS over Rh, Rv (step 1) and RHOB, NPHI, Rss (step 2).
    misfit = np.sqrt((2 * mis_r ** 2 + 3 * mis_n ** 2) / 5)
    return _outputs(logs, p, s, means, csh, phi, sw, rss, misfit, prior["csh"])


def _outputs(logs, p, s, means, csh, phi, sw, rss, misfit, csh_prior):
    """Derived results for one chunk; sand properties are masked in near-pure shale."""
    sand = csh < s["shale_mask"]
    rxo_ss = petro.sand_rxo_parallel(logs["RXO"][None, :], csh, p["rsh_h"])
    sxo = np.clip(petro.archie_sw(rxo_ss, p["rmf"], phi, p["a"], p["m"], p["n"]), 0, 1)
    out = {
        "csh": csh,
        "phi": np.where(sand, phi, np.nan),
        "sw": np.where(sand, sw, np.nan),
        "rss": np.where(sand, rss, np.nan),
        "sxo": np.where(sand, sxo, np.nan),
        "hpv": np.where(sand, (1 - csh) * phi * (1 - sw), 0.0),
        "phit": (1 - csh) * phi + csh * p["phi_sh"],
        "misfit": misfit,
        "csh_prior": csh_prior,
    }
    _, _, rhob, nphi = forward(csh, phi, sw, p)
    rh = 1.0 / (csh / p["rsh_h"] + (1 - csh) / rss)
    rv = csh * p["rsh_v"] + (1 - csh) * rss
    out.update(pred_rh=rh, pred_rv=rv, pred_rhob=rhob, pred_nphi=nphi)
    out = {name: val.astype(np.float32) for name, val in out.items()}
    # One value per realization (depth average for zoned end-points), for sensitivity analysis.
    return out, {name: p[name].mean(axis=1) for name in means}


def run_monte_carlo(logs, means, stds, archie, s, progress=None):
    """Solve the joint inversion for many realizations, in parallel chunks.

    `logs` holds 1-D arrays GR, RH, RV, RHOB, NPHI, RXO (NaN where unusable).
    Returns per-realization arrays of shape (realizations, depths) and the
    sampled parameters. Each chunk has its own seed spawned from s["seed"],
    so results do not depend on the number of worker processes.
    """
    from concurrent.futures import ProcessPoolExecutor
    import os

    n_real, chunk = s["realizations"], s["chunk"]
    n_depth = len(logs["GR"])
    sizes = [min(chunk, n_real - i) for i in range(0, n_real, chunk)]
    seeds = np.random.SeedSequence(s["seed"]).spawn(len(sizes))
    out = {k: np.empty((n_real, n_depth), np.float32) for k in OUTPUT_KEYS}
    params = {k: np.empty(n_real) for k in means}
    starts = np.r_[0, np.cumsum(sizes)[:-1]]

    def store(i, result):
        sl = slice(starts[i], starts[i] + sizes[i])
        chunk_out, chunk_params = result
        for name in OUTPUT_KEYS:
            out[name][sl] = chunk_out[name]
        for name in means:
            params[name][sl] = chunk_params[name]

    workers = s.get("workers", 0) or max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(sizes))
    done = 0
    if workers == 1:
        for i, (k, seed) in enumerate(zip(sizes, seeds)):
            store(i, _solve_chunk(logs, means, stds, archie, s, k, seed))
            done += k
            if progress:
                progress(done, n_real)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_chunk, logs, means, stds, archie, s, k, seed): i
                       for i, (k, seed) in enumerate(zip(sizes, seeds))}
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                i = futures[fut]
                store(i, fut.result())
                done += sizes[i]
                if progress:
                    progress(done, n_real)
    return out, params


def summarize(samples, percentiles=(10, 50, 90)):
    """P10/P50/P90, mean and std over realizations (axis 0), ignoring NaN."""
    res = {}
    for name, arr in samples.items():
        with np.errstate(all="ignore"), _quiet():
            q = np.nanpercentile(arr, percentiles, axis=0)
            for p, row in zip(percentiles, q):
                res[f"{name}_p{p}"] = row
            res[f"{name}_mean"] = np.nanmean(arr, axis=0)
            res[f"{name}_std"] = np.nanstd(arr, axis=0)
    return res


class _quiet:
    """Silence 'All-NaN slice' warnings for shale samples where sand is undefined."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)
