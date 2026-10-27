"""End-to-end interpretation workflow driven by a TOML configuration."""
import time

import numpy as np
import pandas as pd

from . import inversion, io, petro


def _zone(df, interval):
    return df.loc[interval[0]:interval[1]]


def _value(entry, auto_value):
    """Config entries are [mean, std]; 'auto' means use the estimated value."""
    mean, std = entry
    return (auto_value if mean == "auto" else float(mean)), float(std)


ZONED = ("gr_shale", "rho_sh", "nphi_sh", "rsh_h", "rsh_v", "phi_sh")


def shale_sets(df, cfg, props):
    """Shale end-points of every shale zone, estimated from its reference interval."""
    sets = []
    for z in cfg["shale_zones"]:
        sh = _zone(df, z["reference"])
        auto = {"gr_shale": float(sh.GR.median()), "rho_sh": float(sh.RHOB.median()),
                "nphi_sh": float(sh.NPHI.median()), "rsh_h": float(sh.RH.median()),
                "rsh_v": float(sh.RV.median())}
        # Shale porosity seen by the density log (Thomas-Stieber shale point).
        auto["phi_sh"] = float(petro.density_porosity(auto["rho_sh"], props["rho_ma"], props["rho_w"]))
        sets.append(dict(name=z["name"], top=float(z["top"]), bottom=float(z["bottom"]),
                         reference=tuple(z["reference"]), auto=auto, used={}))
    return sets


def zone_index(depth, sets):
    """Index of the shale zone that applies at each depth (nearest zone outside all ranges)."""
    depth = np.asarray(depth, float)
    idx = np.full(depth.shape, -1)
    for i, s in enumerate(sets):
        idx[(depth >= s["top"]) & (depth < s["bottom"]) & (idx < 0)] = i
    outside = idx < 0
    idx[outside] = np.where(depth[outside] < sets[0]["top"], 0, len(sets) - 1)
    return idx


def _subset(ep, mask):
    """End-points restricted to the samples in `mask` (arrays are sliced, scalars kept)."""
    return {k: (v[mask] if np.ndim(v) else v) for k, v in ep.items()}


def shale_consistency(df, sets):
    """Compare the reference shales of the shale zones (task 4)."""
    rows = {}
    for s in sets:
        z = _zone(df, s["reference"])
        rows[f"{s['name']} {s['reference'][0]:.0f}-{s['reference'][1]:.0f} ft"] = {
            c: z[c].median() for c in ("GR", "RHOB", "NPHI", "PE", "POTA", "THOR", "RH", "RV")}
    return pd.DataFrame(rows).T


def pressure_analysis(press, traj, cfg, rho_w):
    """Gradients in TVD from formation-tester points, fluid densities and free-water level."""
    md, tvd = traj
    inside = (press.DEPTH >= md[0]) & (press.DEPTH <= md[-1])
    press = press[inside].copy()
    press["TVD"] = np.interp(press.DEPTH, md, tvd)
    res = {"points": press}
    hc = press[press.DEPTH.between(*cfg["pressure"]["hc_interval"])]
    wt = press[press.DEPTH.between(*cfg["pressure"]["water_interval"])]
    if len(hc) >= 2:
        res["hc"] = petro.fit_gradient(hc.TVD, hc.FM_PRESS)
    if len(wt) >= 2:
        res["water"] = petro.fit_gradient(wt.TVD, wt.FM_PRESS)
    elif len(wt) == 1:
        # Single water point: water line through it with the assumed brine density.
        g = rho_w * petro.PSI_PER_FT_PER_GCC
        res["water"] = dict(gradient=g, intercept=wt.FM_PRESS.iloc[0] - g * wt.TVD.iloc[0],
                            density=rho_w, density_se=np.nan, n=1, assumed=True)
    if "hc" in res and "water" in res:
        h, w = res["hc"], res["water"]
        fwl_tvd = (w["intercept"] - h["intercept"]) / (h["gradient"] - w["gradient"])
        res["fwl_tvd"] = fwl_tvd
        res["fwl_md"] = float(np.interp(fwl_tvd, tvd, md))
    return res


def calibrate_water(df, ep, archie, rho_ma, rho_w, interval):
    """Rw and Rmf from the water sand, using the laminated model (Sw = Sxo = 1)."""
    mask = (df.index >= interval[0]) & (df.index <= interval[1])
    z, ep = df[mask], _subset(ep, mask)
    csh, rss = petro.laminated_resistivity(z.RH, z.RV, ep["rsh_h"], ep["rsh_v"])
    phi = petro.density_porosity(petro.sand_density(z.RHOB, csh, ep["rho_sh"]), rho_ma, rho_w)
    phi = np.asarray(phi)
    ok = (csh < 0.5) & (phi > 0.05) & np.isfinite(rss)
    rw = petro.archie_rw(rss[ok], phi[ok], archie["a"], archie["m"])
    rsh_h = np.broadcast_to(ep["rsh_h"], csh.shape)[ok]
    rxo_ss = petro.sand_rxo_parallel(z.RXO.values[ok], csh[ok], rsh_h)
    rmf = petro.archie_rw(rxo_ss, phi[ok], archie["a"], archie["m"])
    spread = lambda v: float(np.nanstd(np.log(v)))
    return dict(rw=float(np.nanmedian(rw)), rw_log_spread=spread(rw),
                rmf=float(np.nanmedian(rmf)), rmf_log_spread=spread(rmf), n=int(ok.sum()))


def conventional(df, ep, archie, props, phi_sh):
    """Sequential (non-inversion) interpretation used as a reference."""
    out = pd.DataFrame(index=df.index)
    igr = petro.gr_index(df.GR, ep["gr_sand"], ep["gr_shale"])
    out["CSH_GR"] = igr
    out["CSH_LARIONOV"] = petro.csh_larionov_tertiary(igr)
    vs, vsh, phi = petro.neutron_density_ternary(
        df.RHOB, df.NPHI, (props["rho_ma"], props["nphi_ma"]),
        (ep["rho_sh"], ep["nphi_sh"]), (props["rho_w"], 1.0))
    out["CSH_ND"] = np.clip(vsh, 0, 1)
    csh_lam, rss = petro.laminated_resistivity(df.RH, df.RV, ep["rsh_h"], ep["rsh_v"])
    out["CSH_RES"] = np.where(np.isnan(csh_lam) & (df.RH > 0), 1.0, csh_lam)
    out["RSS_RES"] = rss
    # Sand-lamina porosity from the density log with the resistivity Csh.
    rho_f = props["rho_w"]
    rho_ss = petro.sand_density(df.RHOB, np.minimum(out.CSH_RES, 0.95), ep["rho_sh"])
    out["PHI_SS_D"] = np.clip(petro.density_porosity(rho_ss, props["rho_ma"], rho_f), 0, 0.45)
    out["SW_SS"] = np.clip(petro.archie_sw(rss, props["rw"], out.PHI_SS_D,
                                           archie["a"], archie["m"], archie["n"]), 0, 1)
    out["PHIT_D"] = np.clip(petro.density_porosity(df.RHOB, props["rho_ma"], rho_f), 0, 0.5)
    # Thomas-Stieber: total porosity of the laminated mix.
    out["PHIT_TS"] = (1 - out.CSH_RES) * out.PHI_SS_D + out.CSH_RES * phi_sh
    return out


def flagged_intervals(depth, mask, min_len, merge_gap):
    """Contiguous (top, bottom) depth ranges where `mask` is true."""
    depth, mask = np.asarray(depth, float), np.asarray(mask, bool)
    step = np.median(np.diff(depth))
    runs, start = [], None
    for d, m in zip(depth, mask):
        if m and start is None:
            start = d
        elif not m and start is not None:
            runs.append([start, prev + step])
            start = None
        prev = d
    if start is not None:
        runs.append([start, depth[-1] + step])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


def uranium_analysis(df, ep, sets, zidx, ucfg):
    """Flag depths where uranium makes the GR an unreliable shale indicator.

    Two flags: abnormal uranium concentration (robust outlier above background),
    and uranium inflating the GR Csh beyond the uranium-free (Th + K) CGR Csh.
    """
    out = pd.DataFrame(index=df.index)
    u = df.URAN
    med = float(u.median())
    mad = 1.4826 * float((u - med).abs().median())
    u_max = med + ucfg["u_mad_k"] * mad if ucfg["u_max_ppm"] == "auto" else float(ucfg["u_max_ppm"])
    out["U_ANOMALY"] = u > u_max

    cgr_sand = float(np.nanpercentile(df.CGR, 3))
    cgr_shale = np.array([float(_zone(df, s["reference"]).CGR.median()) for s in sets])[zidx]
    out["CSH_CGR"] = petro.gr_index(df.CGR, cgr_sand, cgr_shale)
    out["CSH_U_INFLATION"] = petro.gr_index(df.GR, ep["gr_sand"], ep["gr_shale"]) - out.CSH_CGR
    out["U_INFLATES_GR"] = out.CSH_U_INFLATION > ucfg["max_csh_inflation"]

    args = (ucfg["min_interval_ft"], ucfg["merge_gap_ft"])
    intervals = {"anomaly": flagged_intervals(df.index, out.U_ANOMALY, *args),
                 "inflation": flagged_intervals(df.index, out.U_INFLATES_GR, *args)}
    info = dict(u_background=med, u_mad=mad, u_max=u_max, cgr_sand=cgr_sand,
                max_csh_inflation=ucfg["max_csh_inflation"], intervals=intervals)
    return out, info


def core_fluid_density(df, csh, rho_sh):
    """Apparent pore-fluid density at core depths (task 5)."""
    has_core = df.CORE_POR.notna().values
    c = df[has_core].copy()
    cs = np.minimum(csh.loc[c.index], 0.9)
    c["RHO_SS"] = petro.sand_density(c.RHOB, cs, np.asarray(rho_sh)[has_core])
    c["RHO_F_CORE"] = petro.fluid_density_from_core(c.RHO_SS, c.CORE_POR, c.CORE_RHOG)
    return c


def run(cfg, progress=None, realizations=None):
    t0 = time.time()
    d = cfg["data"]
    full, las = io.load_well(d["las"], cfg["curves"])
    df = full.loc[d["top"]:d["bottom"]].copy()
    flags = io.quality_flags(df, cfg["zones"]["max_drho"])
    press = io.load_pressures(d["pressures"]) if d.get("pressures") else None

    archie = {k: float(v) for k, v in cfg["archie"].items()}
    props, props_std = {}, {}
    for name in ("rho_ma", "nphi_ma", "rho_w", "hi_hc"):
        props[name], props_std[name] = _value(cfg["properties"][name], None)
    sets = shale_sets(df, cfg, props)
    zidx = zone_index(df.index.values, sets)
    gr_sand_auto = float(np.nanpercentile(df.GR, cfg["zones"]["clean_sand_gr_percentile"]))
    ep, ep_std = {}, {}
    for name, (mean, std) in cfg["endpoints"].items():
        ep_std[name] = float(std)
        if name in ZONED:
            for s in sets:
                s["used"][name] = s["auto"][name] if mean == "auto" else float(mean)
            ep[name] = np.array([s["used"][name] for s in sets])[zidx]
        else:
            ep[name] = gr_sand_auto if mean == "auto" else float(mean)

    # Trajectory over the whole LAS so pressure points outside the window can be placed.
    traj_ok = full.SDEV.notna() & full.HAZI.notna()
    md = full.index.values[traj_ok]
    tvd = petro.tvd_minimum_curvature(md, full.SDEV[traj_ok], full.HAZI[traj_ok])
    df["TVD_REL"] = np.interp(df.index, md, tvd)

    pres = None
    rho_hc_auto = 0.7
    if press is not None and len(press):
        pres = pressure_analysis(press, (md, tvd), cfg, props["rho_w"])
        if "hc" in pres:
            rho_hc_auto = float(np.clip(pres["hc"]["density"], 0.1, 1.0))
    props["rho_hc"], props_std["rho_hc"] = _value(cfg["properties"]["rho_hc"], rho_hc_auto)

    water = calibrate_water(df, ep, archie, props["rho_ma"], props["rho_w"], cfg["zones"]["water"])
    props["rw"], props_std["rw"] = _value(cfg["properties"]["rw"], water["rw"])
    props["rmf"], props_std["rmf"] = _value(cfg["properties"]["rmf"], water["rmf"])

    conv = conventional(df, ep, archie, props, ep["phi_sh"])
    uran, uran_info = uranium_analysis(df, ep, sets, zidx, cfg["uranium"])
    conv = conv.join(uran)
    core = core_fluid_density(df, conv.CSH_RES, ep["rho_sh"])

    # ---- Monte Carlo joint inversion
    s = dict(cfg["inversion"])
    if realizations:
        s["realizations"] = realizations
    s["phi_prior"] = tuple(s["phi_prior"])
    s["sw_prior"] = tuple(s["sw_prior"])
    usable = ~flags.NO_RHRV.values
    logs = {
        "GR": df.GR.values,
        "RH": np.where(usable, df.RH.values, np.nan),
        "RV": np.where(usable, df.RV.values, np.nan),
        "RHOB": np.where(flags.BAD_RHOB.values, np.nan, df.RHOB.values),
        "NPHI": df.NPHI.values,
        "RXO": df.RXO.values,
    }
    means = {**ep, **props}
    stds = {**ep_std, **props_std}
    samples, params = inversion.run_monte_carlo(logs, means, stds, archie, s, progress)
    # Without Rh/Rv the resistivity part is missing; do not report those depths.
    for k in samples:
        samples[k][:, ~usable] = np.nan
    summary = pd.DataFrame(inversion.summarize(samples), index=df.index)

    # Hydrocarbon column (ft, TVD) per realization: sum of HPV * dTVD.
    dtvd = np.gradient(df.TVD_REL.values)
    hpv = np.nan_to_num(samples["hpv"])
    zones = zone_intervals(cfg, df.index)
    hc_column = {"Whole interval": hpv @ dtvd}
    for name, (a, b) in zones.items():
        m = (df.index >= a) & (df.index < b)
        hc_column[name] = hpv[:, m] @ dtvd[m]

    return dict(df=df, flags=flags, conv=conv, summary=summary, samples=samples,
                params=pd.DataFrame(params), endpoints=ep, endpoint_std=ep_std,
                shale_sets=sets, zone_idx=zidx, props=props, props_std=props_std,
                archie=archie, water=water, pressure=pres, core=core, hc_column=hc_column,
                shale_check=shale_consistency(df, sets), uranium=uran_info, settings=s, las=las,
                elapsed=time.time() - t0, cfg=cfg)


def zone_intervals(cfg, depth):
    tops = sorted(cfg.get("tops", {}).items(), key=lambda kv: kv[1])
    bounds = [t for _, t in tops] + [float(depth[-1]) + 1e-6]
    return {name: (bounds[i], bounds[i + 1]) for i, (name, _) in enumerate(tops)}
