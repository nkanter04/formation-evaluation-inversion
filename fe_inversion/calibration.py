"""Per-section shale bracketing (course workflow).

For each depth section:
  1. Take the thickest shale bed as the pure-shale reference (or a given interval).
  2. For a grid of shale resistivities (Rsh_h, Rsh_v) around that bed's readings,
     solve Rh and Rv for Csh and Rss at thick-bed center points (no GR bias).
  3. Read the spectral GR where the inversion gives Csh ~ 0 (sand) and Csh ~ 1
     (shale), and compute Csh = (G - G_sand) / (G_shale - G_sand).
  4. Keep the (Rsh_h, Rsh_v) pair whose inversion Csh best agrees with that
     spectral-GR Csh while the reference bed still reads Csh ~ 1.
"""
import numpy as np

from . import beds, inversion, petro

SPECTRAL = ("THOR", "POTA", "CGR", "GR")


def section_beds(depth, idx, top, bottom):
    """Beds that overlap [top, bottom), clipped to the section."""
    out = []
    for t, b, _ in beds.bed_table(depth, idx):
        t, b = max(t, top), min(b, bottom)
        if b - t > 0:
            out.append((t, b))
    return out


def reference_shale(df, top, bottom, spec, shale_fraction=0.75, smooth_ft=1.0, merge_gap_ft=1.0):
    """Thickest continuous shale in [top, bottom): longest run of high spectral GR with Rh/Rv present.

    "High" means within the top quarter of the section's spectral-GR range
    (P3 to P97). Thin bed boundaries inside a thick shale are ignored on purpose.
    """
    from scipy.ndimage import uniform_filter1d
    d = df.index.values
    step = float(np.median(np.diff(d)))
    sec = (d >= top) & (d < bottom)
    g = uniform_filter1d(np.nan_to_num(np.asarray(spec, float), nan=np.nanmedian(spec)),
                         max(1, int(round(smooth_ft / step))))
    lo, hi = np.nanpercentile(g[sec], 3), np.nanpercentile(g[sec], 97)
    high = sec & (g >= lo + shale_fraction * (hi - lo)) & (df.RH > 0).values & (df.RV > 0).values
    runs, start = [], None
    for i, h in enumerate(high):
        if h and start is None:
            start = i
        elif not h and start is not None:
            runs.append([d[start], d[i - 1] + step])
            start = None
    if start is not None:
        runs.append([d[start], d[-1] + step])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap_ft:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if not merged:
        return None
    return tuple(max(merged, key=lambda r: r[1] - r[0]))


def _endpoints(spec, csh, sand_max, shale_min, min_points):
    """Spectral-GR sand and shale values where the inversion says Csh ~ 0 and ~ 1."""
    sand, shale = csh <= sand_max, csh >= shale_min
    g_sand = np.median(spec[sand]) if sand.sum() >= min_points else np.percentile(spec, 3)
    g_shale = np.median(spec[shale]) if shale.sum() >= min_points else np.percentile(spec, 97)
    return float(g_sand), float(g_shale), int(sand.sum()), int(shale.sum())


def agreement(csh_inv, spec, sand_max=0.1, shale_min=0.9, min_points=5):
    """Spectral-GR Csh with end-points read from the inversion, and its agreement with it."""
    g_sand, g_shale, n_sand, n_shale = _endpoints(spec, csh_inv, sand_max, shale_min, min_points)
    csh_spec = petro.gr_index(spec, g_sand, g_shale)
    diff = csh_inv - csh_spec
    corr = np.corrcoef(csh_inv, csh_spec)[0, 1] if np.std(csh_spec) > 0 else np.nan
    return dict(spec_sand=g_sand, spec_shale=g_shale, n_sand=n_sand, n_shale=n_shale,
                rms=float(np.sqrt(np.mean(diff ** 2))), bias=float(diff.mean()), corr=float(corr))


def bracket_section(df, sec, ref, centers, spec_name, s, bc):
    """Scan (Rsh_h, Rsh_v) for one section. Returns the chosen values and the scan grid."""
    d = df.index.values
    sel = centers & (d >= sec["top"]) & (d < sec["bottom"]) & (df.RH > 0).values & (df.RV > 0).values
    ref_m = (d >= ref[0]) & (d < ref[1]) & (df.RH > 0).values
    ref_c = ref_m & centers
    if ref_c.sum() < 3:
        ref_c = ref_m
    rsh_h0, rsh_v0 = float(df.RH[ref_c].median()), float(df.RV[ref_c].median())

    f = np.logspace(np.log10(bc["factor_min"]), np.log10(bc["factor_max"]), bc["steps"])
    fh, fv = np.meshgrid(f, f, indexing="ij")
    p = {"rsh_h": (rsh_h0 * fh).reshape(-1, 1), "rsh_v": (rsh_v0 * fv).reshape(-1, 1)}
    # Solve at the section's thick-bed centers plus the reference bed.
    use = sel | ref_c
    obs = {"log_rh": np.log(df.RH.values[use])[None, :], "log_rv": np.log(df.RV.values[use])[None, :]}
    obs = {k: np.broadcast_to(v, (p["rsh_h"].shape[0], v.shape[1])) for k, v in obs.items()}
    x0 = inversion.stage1_start(obs["log_rh"], obs["log_rv"], p, np.full(obs["log_rh"].shape, 0.5))
    csh, _, _, misfit = inversion.solve_stage1(obs, p, None, s, x0, s["iterations"])

    spec = df[spec_name].values[use]
    is_center, is_ref = sel[use], ref_c[use]
    span = np.nanpercentile(spec[is_center], 97) - np.nanpercentile(spec[is_center], 3)
    rms = np.empty(len(csh))
    ref_csh = np.empty(len(csh))
    ref_fit = np.empty(len(csh))
    valid = np.ones(len(csh), bool)
    # How far the reference shale's own readings are from (Rsh_h, Rsh_v), in noise units. Rh and
    # Rv can always be fit with two unknowns (e.g. Csh 0.99 plus a sliver of very resistive
    # "sand"), so the pure-shale point itself must match the shale resistivities.
    sr, sv = inversion._noise(s)
    ln_h, ln_v = np.log(df.RH.values[ref_c]).mean(), np.log(df.RV.values[ref_c]).mean()
    pure_fit = np.sqrt((((ln_h - np.log(p["rsh_h"][:, 0])) / sr) ** 2
                        + ((ln_v - np.log(p["rsh_v"][:, 0])) / sv) ** 2) / 2)
    for g in range(len(csh)):
        a = agreement(csh[g][is_center], spec[is_center], bc["sand_max"], bc["shale_min"], bc["min_points"])
        ref_csh[g] = np.median(csh[g][is_ref])
        ref_fit[g] = pure_fit[g]
        rms[g] = a["rms"]
        # Sand must read clearly lower spectral GR than shale, or the end-points are meaningless.
        valid[g] = a["spec_shale"] - a["spec_sand"] > bc["min_span_fraction"] * span
    # The reference shale must read Csh ~ 1 AND sit at (Rsh_h, Rsh_v) within the noise.
    objective = (rms + bc["shale_penalty"] * np.maximum(0.0, bc["shale_min"] - ref_csh)
                 + bc["fit_weight"] * np.maximum(0.0, ref_fit - 1.0))
    objective = np.where(valid, objective, np.inf)
    best = int(np.argmin(objective))
    chosen = agreement(csh[best][is_center], spec[is_center], bc["sand_max"], bc["shale_min"], bc["min_points"])
    return dict(rsh_h0=rsh_h0, rsh_v0=rsh_v0, rsh_h=float(p["rsh_h"][best, 0]),
                rsh_v=float(p["rsh_v"][best, 0]), ref_csh=float(ref_csh[best]),
                ref_fit=float(ref_fit[best]),
                res_misfit=float(np.median(misfit[best])), n_centers=int(is_center.sum()),
                grid_h=rsh_h0 * f, grid_v=rsh_v0 * f, objective=objective.reshape(fh.shape),
                rms_grid=rms.reshape(fh.shape), csh=csh[best], use=use, is_center=is_center,
                **chosen)


def compare_spectral(df, use, is_center, csh, bc):
    """Agreement of the inversion Csh with Csh from each spectral curve (end-points from the inversion)."""
    out = {}
    for name in SPECTRAL:
        if name in df and df[name].notna().any():
            spec = df[name].values[use][is_center]
            out[name] = agreement(csh[is_center], spec, bc["sand_max"], bc["shale_min"], bc["min_points"])
    return out
