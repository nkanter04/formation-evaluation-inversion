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


def estimate_endpoints(df, cfg):
    """Shale and clean-sand end-points from the configured zones."""
    sh = _zone(df, cfg["zones"]["shale"])
    return {
        "gr_sand": float(np.nanpercentile(df.GR, cfg["zones"]["clean_sand_gr_percentile"])),
        "gr_shale": float(sh.GR.median()),
        "rho_sh": float(sh.RHOB.median()),
        "nphi_sh": float(sh.NPHI.median()),
        "rsh_h": float(sh.RH.median()),
        "rsh_v": float(sh.RV.median()),
    }


def shale_consistency(df, cfg, other=(9650.0, 9690.0)):
    """Compare the reference shale with the shale at the top of the interval (task 4)."""
    rows = {}
    for label, iv in (("Reference shale", cfg["zones"]["shale"]), ("Upper shale", other)):
        z = _zone(df, iv)
        rows[f"{label} {iv[0]:.0f}-{iv[1]:.0f} ft"] = {
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
    z = _zone(df, interval)
    csh, rss = petro.laminated_resistivity(z.RH, z.RV, ep["rsh_h"], ep["rsh_v"])
    phi = petro.density_porosity(petro.sand_density(z.RHOB, csh, ep["rho_sh"]), rho_ma, rho_w)
    ok = (csh < 0.5) & (phi > 0.05) & np.isfinite(rss)
    rw = petro.archie_rw(rss[ok], phi[ok], archie["a"], archie["m"])
    rxo_ss = petro.sand_rxo_parallel(z.RXO.values[ok], csh[ok], ep["rsh_h"])
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


def core_fluid_density(df, csh, rho_sh):
    """Apparent pore-fluid density at core depths (task 5)."""
    c = df[df.CORE_POR.notna()].copy()
    cs = np.minimum(csh.loc[c.index], 0.9)
    c["RHO_SS"] = petro.sand_density(c.RHOB, cs, rho_sh)
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
    ep_auto = estimate_endpoints(df, cfg)
    # Shale porosity seen by the density log (Thomas-Stieber shale point).
    ep_auto["phi_sh"] = float(petro.density_porosity(ep_auto["rho_sh"], props["rho_ma"], props["rho_w"]))
    ep, ep_std = {}, {}
    for name, entry in cfg["endpoints"].items():
        ep[name], ep_std[name] = _value(entry, ep_auto.get(name))

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
                endpoints_auto=ep_auto, props=props, props_std=props_std, archie=archie,
                water=water, pressure=pres, core=core, hc_column=hc_column,
                shale_check=shale_consistency(df, cfg), settings=s, las=las,
                elapsed=time.time() - t0, cfg=cfg)


def zone_intervals(cfg, depth):
    tops = sorted(cfg.get("tops", {}).items(), key=lambda kv: kv[1])
    bounds = [t for _, t in tops] + [float(depth[-1]) + 1e-6]
    return {name: (bounds[i], bounds[i + 1]) for i, (name, _) in enumerate(tops)}
