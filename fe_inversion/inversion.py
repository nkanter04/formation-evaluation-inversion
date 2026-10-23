"""Joint laminated sand-shale inversion with Monte Carlo uncertainty.

Unknowns at every depth: shale concentration Csh, porosity of the sand laminae
phi_ss and water saturation of the sand laminae Sw_ss. Forward model:

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
U_DAMPING = 0.02                   # keeps the bounded transform away from saturation
LOG_PARAMS = ("rsh_h", "rsh_v", "rw", "rmf")


def to_x(u):
    return BOUNDS[:, 0] + (BOUNDS[:, 1] - BOUNDS[:, 0]) / (1.0 + np.exp(-np.clip(u, -50, 50)))


def to_u(x):
    f = (np.asarray(x) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    f = np.clip(f, 1e-4, 1 - 1e-4)
    return np.log(f / (1 - f))


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

    sr, sd, sn, sc = s["sigma_log_r"], s["sigma_rhob"], s["sigma_nphi"], s["csh_prior_std"]
    (phi0, sphi), (sw0, ssw) = s["phi_prior"], s["sw_prior"]
    valid = [np.isfinite(obs[k]) for k in ("log_rh", "log_rv", "rhob", "nphi")]
    r = np.empty(c.shape + (10,))
    r[..., 0] = (-np.log(sig_h) - obs["log_rh"]) / sr
    r[..., 1] = (np.log(rv) - obs["log_rv"]) / sr
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
    Jx[..., 1, 0] = (p["rsh_v"] - rss) / rv / sr
    Jx[..., 1, 1] = k_v * dl_phi / sr
    Jx[..., 1, 2] = k_v * dl_sw / sr
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
    f = 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))
    J = Jx * ((BOUNDS[:, 1] - BOUNDS[:, 0]) * f * (1 - f))[..., None, :]
    J[..., 7, 0] = J[..., 8, 1] = J[..., 9, 2] = U_DAMPING
    return r, J


def solve(obs, p, prior, s, x0, iterations=40, tol=1e-8):
    """Batched Levenberg-Marquardt over arrays of shape (realizations, depths)."""
    u = to_u(x0)
    r, J = _residuals(u, obs, p, prior, s, jacobian=True)
    cost = np.sum(r * r, axis=-1)
    mu = np.full(cost.shape, 1e-3)
    eye = np.eye(3)
    for _ in range(iterations):
        Jt = np.swapaxes(J, -1, -2)
        JTJ = Jt @ J
        g = (Jt @ r[..., None])[..., 0]
        A = JTJ + mu[..., None, None] * (JTJ * eye + 1e-9 * eye)
        step = -np.linalg.solve(A, g[..., None])[..., 0]
        u_new = u + step
        r_new, J_new = _residuals(u_new, obs, p, prior, s, jacobian=True)
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
    data_misfit = np.sqrt(np.mean(r[..., :4] ** 2, axis=-1))
    return to_x(u), cost, data_misfit


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
    p["gr_shale"] = np.maximum(p["gr_shale"], p["gr_sand"] + 5.0)
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
    obs = {
        "log_rh": np.log(logs["RH"]) + noise(s["sigma_log_r"]),
        "log_rv": np.log(logs["RV"]) + noise(s["sigma_log_r"]),
        "rhob": logs["RHOB"] + noise(s["sigma_rhob"]),
        "nphi": logs["NPHI"] + noise(s["sigma_nphi"]),
    }
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
    sand = csh < s["shale_mask"]
    rss = petro.archie_rt(p["rw"], phi, sw, p["a"], p["m"], p["n"])
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
        "csh_prior": prior["csh"],
    }
    pred = forward(csh, phi, sw, p)
    out.update(zip(("pred_rh", "pred_rv", "pred_rhob", "pred_nphi"), pred))
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
