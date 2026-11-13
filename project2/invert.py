"""PGE 385K Homework Project 2: Monte Carlo inversion for grain-coating clay sandstones.

Run:  python project2/invert.py
Writes figures and inversion_summary.md to the Project 2 "Interpretation" folder.

At each depth, Csh comes from the gamma ray through the calibrated relation (capped at the clean-sandstone porosity), and
(phi_s, Sws, Sxos) are found by minimizing the cost function

    Phi = sum over d in {RHOB, NPHI, ln Rt, ln Rxo} of [(d_calc - d_meas) / sigma_d]^2 + weak priors,

with d_calc from petro.dispersed_forward (book A.9.1 and Juhasz A.14.6). Each realization redraws
the uncertain inputs, recalibrates Rw in the water sandstones with them, adds noise to the logs
and solves again; the spread of the solutions gives P10-P90 bands.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explore as ex
import interpret as it
from fe_inversion import inversion, petro

C = it.CFG["inversion"]
A = it.CFG["archie"]["a"]


def setup():
    df, core = ex.load_logs(), ex.load_core()
    w = df.loc[ex.TOP:ex.BASE]
    sands, vsh, gr_ends = ex.pick_sands(w)
    sands, _ = ex.classify(sands)
    groups = it.water_sand_samples(w, sands, vsh)
    shale = it.shale_properties(w)
    det, cal = it.calculate(w, vsh, groups, shale)
    s = it.CFG["shale"]
    upper = w.index.values < s["boundary"]
    sh = {}
    for name in ("upper", "middle"):
        z = w.loc[s[name][0]:s[name][1]]
        z = z[~z.WASHOUT]
        sh[name] = dict(rho=z.ZDEN.median(), nphi=z.NPHI.median(), phid=z.PHID.median())
    pick = lambda k: np.where(upper, sh["upper"][k], sh["middle"][k])
    return w, core, sands, groups, det, cal, gr_ends, dict(rho=pick("rho"), nphi=pick("nphi"), phid=pick("phid"))


def draw(k, rng, w, shale, gr_ends, phi_clean):
    """Draw k realizations of the uncertain inputs; arrays have shape (k, depths) or (k, 1)."""
    g = lambda mean_sd: rng.normal(mean_sd[0], mean_sd[1], (k, 1))
    gr_cl = gr_ends[0] + rng.normal(0, C["gr_endpoint_sd"], (k, 1))
    gr_sh = gr_ends[1] + rng.normal(0, C["gr_endpoint_sd"], (k, 1))
    i_gr = np.clip((w.GR.values[None, :] - gr_cl) / (gr_sh - gr_cl), 0, 1)
    alpha = rng.uniform(C["alpha"][0], C["alpha"][1], (k, 1))
    rho_sh = shale["rho"][None, :] + rng.normal(0, C["shale_shift"][0], (k, 1))
    nphi_sh = shale["nphi"][None, :] + rng.normal(0, C["shale_shift"][1], (k, 1))
    phid_sh = (C["rho_ma"][0] - rho_sh) / (C["rho_ma"][0] - ex.RHO_F)
    depth = w.index.values[None, :]
    temp = w.TEMP.values[None, :] + rng.normal(0, C["surface_temp_sd"], (k, 1)) * (1 - depth / ex.BHT_DEPTH)
    return dict(csh=np.minimum(petro.csh_from_ish(i_gr, it._CSH["method"]), phi_clean), i_gr=i_gr, rho_sh=rho_sh, nphi_sh=nphi_sh, phid_sh=phid_sh,
                phi_sh=alpha * phid_sh + (1 - alpha) * nphi_sh, temp=temp,
                rho_h=np.clip(g(C["rho_h"]), 0.3, 1.0), hi_h=np.clip(g(C["hi_h"]), 0.5, 1.1), rho_mf=g(C["rho_mf"]),
                rho_ma=g(C["rho_ma"]), m=g(C["m"]), n=g(C["n"]),
                rmf75=ex.RMF_75 * np.exp(rng.normal(0, C["rmf_log_sd"], (k, 1))),
                bfac=np.exp(rng.normal(0, C["b_log_sd"], (k, 1))))


def calibrate_rw(w, groups, d):
    """Rw at 75 degF per realization: Juhasz Rw (Sw = 1) at the water-sandstone centers, median over sandstones."""
    pos = {dep: i for i, dep in enumerate(w.index)}
    idx = [np.array([pos[x] for x in g]) for g in groups.values()]
    rw75 = np.empty(d["m"].shape[0])
    for j in range(len(rw75)):
        per = []
        for i in idx:
            csh = d["csh"][j, i]
            _, phi_t = petro.grain_coating_porosity(w.PHID.values[i], w.NPHI.values[i], csh, d["phid_sh"][j, i],
                                                    d["nphi_sh"][j, i], d["phi_sh"][j, i])
            phi_t = np.maximum(phi_t, 0.01)
            sb = petro.bound_water_fraction(csh, d["phi_sh"][j, i], phi_t)
            t = d["temp"][j, i]
            rw = petro.juhasz_rw(w.RIPD.values[i], phi_t, sb, t, A, d["m"][j, 0])
            per.append(np.median(petro.rw_at_temperature(rw, t, 75.0)))
        rw75[j] = np.median(per)
    return rw75[:, None]


def run(w, groups, det, cal, shale, gr_ends):
    rng = np.random.default_rng(C["seed"])
    n_real, n_dep = C["realizations"], len(w)
    keys = ("phi_s", "sws", "sxos", "phi_t", "sw_t", "hpv", "misfit", "res_rhob", "res_nphi", "res_rt", "res_rxo", "at_sb", "sb")
    out = {k: np.empty((n_real, n_dep), np.float32) for k in keys}
    rw_all = np.empty(n_real)
    x_det = np.stack([np.clip(det.phi_s.values, 0.02, 0.4), np.clip(det.sws.values, 0.05, 0.98),
                      np.full(n_dep, 0.8)], -1)
    for start in range(0, n_real, C["chunk"]):
        k = min(C["chunk"], n_real - start)
        d = draw(k, rng, w, shale, gr_ends, cal["phi_clean"])
        rw75 = calibrate_rw(w, groups, d)
        rw = petro.rw_at_temperature(rw75, 75.0, d["temp"])
        rmf = petro.rw_at_temperature(d["rmf75"], 75.0, d["temp"])
        p = dict(rho_mf=d["rho_mf"], rho_h=d["rho_h"], rho_ma=d["rho_ma"], rho_sh=d["rho_sh"], hi_h=d["hi_h"], nphi_ma=0.0,
                 nphi_sh=d["nphi_sh"], phi_sh=d["phi_sh"], rw=rw, rmf=rmf, a=A, m=d["m"], n=d["n"],
                 b_w=petro.waxman_smits_b(d["temp"], rw) * d["bfac"], b_mf=petro.waxman_smits_b(d["temp"], rmf) * d["bfac"])
        noise = lambda sd: rng.normal(0, sd, (k, n_dep))
        obs = dict(rhob=w.ZDEN.values + noise(C["sigma_rhob"]), nphi=w.NPHI.values + noise(C["sigma_nphi"]),
                   log_rt=np.log(w.RIPD.values) + noise(C["sigma_log_rt"]), log_rxo=np.log(w.RSFL.values) + noise(C["sigma_log_rxo"]))
        x, r = inversion.solve_dispersed(obs, d["csh"], p, C, np.broadcast_to(x_det, (k, n_dep, 3)).copy(), C["iterations"])
        sl = slice(start, start + k)
        phi_t = x[..., 0] + d["csh"] * d["phi_sh"]
        out["phi_s"][sl], out["sws"][sl] = x[..., 0], x[..., 1]
        out["sxos"][sl] = x[..., 2] if C["use_rxo"] else np.nan
        out["at_sb"][sl] = x[..., 1] < 0.02
        out["sb"][sl] = d["csh"] * d["phi_sh"] / phi_t
        out["phi_t"][sl] = phi_t
        out["sw_t"][sl] = (x[..., 1] * x[..., 0] + d["csh"] * d["phi_sh"]) / phi_t
        out["hpv"][sl] = x[..., 0] * (1 - x[..., 1])
        out["misfit"][sl] = np.sqrt(np.mean(r[..., :4 if C["use_rxo"] else 3] ** 2, axis=-1))
        for i, name in enumerate(("res_rhob", "res_nphi", "res_rt", "res_rxo")):
            out[name][sl] = r[..., i]
        rw_all[sl] = rw75[:, 0]
        print(f"  realizations {start + k}/{n_real}")
    return out, rw_all


def pct(a, q):
    return np.nanpercentile(a, q, axis=0)


def fig_tracks(w, out, det, core, sands, path):
    fig, ax = plt.subplots(1, 6, figsize=(16, 15), sharey=True)
    z = w.index.values
    ax[0].plot(w.GR, z, "g", lw=0.7, label="GR (API)"); ax[0].set(xlim=(0, 150), title="Gamma ray")
    third = ("sxos", "Sxos", (1, 0)) if C["use_rxo"] else ("sw_t", "Sw total", (1, 0))
    for a, key, lab, xl in [(ax[1], "phi_t", "φt", (0.45, 0)), (ax[2], "sws", "Sws", (1, 0)), (ax[3], *third),
                            (ax[4], "hpv", "HPV", (0, 0.35))]:
        a.fill_betweenx(z, pct(out[key], 10), pct(out[key], 90), color="#9ecae1", lw=0, label="P10–P90")
        a.plot(pct(out[key], 50), z, "#08519c", lw=0.7, label="P50")
        a.set(xlim=xl, title=f"{lab} (v/v)")
    ax[1].plot(det.phi_t, z, "k", lw=0.5, alpha=0.7, label="Deterministic Juhasz")
    ax[1].plot(core.phi_3500, core.depth_log, "ro", ms=3.5, mfc="none", label="Core φ (3,500 psi)")
    ax[2].plot(det.sws, z, "k", lw=0.5, alpha=0.7, label="Deterministic Juhasz")
    ax[4].plot(det.hpv_j, z, "k", lw=0.5, alpha=0.7, label="Deterministic Juhasz")
    if not C["use_rxo"]:
        ax[3].plot(det.sw_j, z, "k", lw=0.5, alpha=0.7, label="Deterministic Juhasz")
    ax[5].plot(pct(out["misfit"], 50), z, "m", lw=0.6, label="RMS weighted misfit, P50")
    ax[5].axvline(1, color="k", ls=":", lw=1); ax[5].set(xlim=(0, 5), title="Data misfit")
    for a in ax:
        ex.shade_sands(a, sands); a.grid(alpha=0.3); a.legend(fontsize=6.5, loc="lower right")
    ax[0].set_ylim(ex.BASE, ex.TOP); ax[0].set_ylabel("Measured depth (ft)")
    fig.suptitle("ORM-75: Monte Carlo inversion with grain-coating clay (bands = P10–P90 of all realizations)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_fit_and_column(out, cols, det_cols, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    names = [("res_rhob", "RHOB"), ("res_nphi", "NPHI"), ("res_rt", "ln Rt")] + ([("res_rxo", "ln Rxo")] if C["use_rxo"] else [])
    for key, lab in names:
        ax[0].hist(np.clip(pct(out[key], 50)[cols["net_any"]], -6, 6), bins=60, histtype="step", lw=1.3, label=lab)
    ax[0].set(xlabel="Weighted residual (calculated − measured)/σ, P50", ylabel="Samples", title="(a) Data residuals, net sandstones")
    for key, lab in names:
        ax[1].hist(np.clip(pct(out[key], 50)[cols["water_any"]], -6, 6), bins=60, histtype="step", lw=1.3, label=lab)
    ax[1].set(xlabel="Weighted residual (calculated − measured)/σ, P50", ylabel="Samples", title="(b) Data residuals, water sandstones")
    ax[2].hist(cols["column"], bins=30, color="#9ecae1", edgecolor="#08519c", label="Inversion realizations")
    for v, lab, c in [(det_cols["col_a"], "Deterministic Archie", "b"), (det_cols["col_j"], "Deterministic Juhasz", "k")]:
        ax[2].axvline(v, color=c, ls="--", lw=1.2, label=f"{lab} {v:.1f} ft")
    ax[2].set(xlabel="Hydrocarbon column in net hydrocarbon sandstones (ft)", ylabel="Realizations", title="(c) Hydrocarbon column")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    w, core, sands, groups, det, cal, gr_ends, shale = setup()
    hc = sands[sands.fluid == "possible hydrocarbon"]
    out, rw_all = run(w, groups, det, cal, shale, gr_ends)

    z = w.index.values
    net = det.i_gr.values < it.CFG["net"]["max_vsh"]
    in_hc = np.zeros(len(z), bool)
    for _, s in hc.iterrows():
        in_hc |= (z >= s.top) & (z <= s.base)
    in_w = np.zeros(len(z), bool)
    for key in groups:
        in_w |= (z >= key[0]) & (z <= key[1])
    column = np.nansum(out["hpv"][:, in_hc & net], axis=1) * it.STEP
    det_cols = it.hc_summary(det, hc)
    cols = dict(column=column, net_any=in_hc & net, water_any=in_w & net)

    per = []
    for _, s in hc.iterrows():
        m = (z >= s.top) & (z <= s.base) & net
        colk = np.nansum(out["hpv"][:, m], axis=1) * it.STEP
        per.append(dict(sandstone=f"{s.top:.0f}–{s.base:.0f}", phi_s=np.nanmedian(out["phi_s"][:, m]), sws_p50=np.nanmedian(out["sws"][:, m]),
                        sws_det=det.sws.values[m].mean(), pct_sws_at_bound=100 * float(np.mean(out["at_sb"][:, m])),
                        sb=float(np.nanmedian(out["sb"][:, m])),
                        col_p10=np.percentile(colk, 10), col_p50=np.percentile(colk, 50), col_p90=np.percentile(colk, 90),
                        col_det=det.hpv_j.values[m].sum() * it.STEP))
    per = pd.DataFrame(per)

    fit_rows = []
    logs = [("res_rhob", "RHOB", f"{C['sigma_rhob']} g/cc"), ("res_nphi", "NPHI", f"{C['sigma_nphi']} v/v"), ("res_rt", "ln Rt", f"{C['sigma_log_rt']}")]
    if C["use_rxo"]:
        logs.append(("res_rxo", "ln Rxo", f"{C['sigma_log_rxo']}"))
    for key, lab, sd in logs:
        med = pct(out[key], 50)
        fit_rows.append(dict(log=lab, sigma=sd, rms_hc=float(np.sqrt(np.nanmean(med[in_hc & net] ** 2))),
                             mean_hc=float(np.nanmean(med[in_hc & net])), rms_water=float(np.sqrt(np.nanmean(med[in_w & net] ** 2))),
                             mean_water=float(np.nanmean(med[in_w & net]))))
    fit = pd.DataFrame(fit_rows)

    fig_tracks(w, out, det, core, sands, it.OUT / "08_inversion_tracks.png")
    fig_fit_and_column(out, cols, det_cols, it.OUT / "09_inversion_fit_column.png")
    core_phi = np.interp(core.depth_log, z, pct(out["phi_t"], 50))
    e = core_phi - core.phi_3500
    band = pct(out["hpv"], 90) - pct(out["hpv"], 10)

    L = ["# Project 2: Monte Carlo inversion with grain-coating clay", "",
         "## Method", "",
         "At every depth (0.5 ft), Csh is taken from the gamma ray (calibrated relation, capped at the clean-sandstone porosity) and the "
         "unknowns are the non-shale porosity φs and the non-shale water saturations of the uninvaded (Sws) and flushed (Sxos) zones. "
         "They minimize the cost function", "",
         "Φ = [(ρb,calc − ρb)/σρ]² + [(φN,calc − φN)/σN]² + [(ln Rt,calc − ln Rt)/σRt]² + [(ln Rxo,calc − ln Rxo)/σRxo]² + weak priors on φs, Sws and Sxos,", "",
         "where the calculated logs come from the grain-coating clay model (book A.9.1) and Juhasz's equation (A.14.6): "
         "ρb = φs·ρf + (1 − φs − Csh)·ρma + Csh·ρsh, φN likewise with hydrogen index, and 1/Rt = (φt^m·Sw^n/a)(1/Rw + B·Sb/Sw) with "
         "φt = φs + Csh·φsh, Sb = Csh·φsh/φt and Sw = (Sws·φs + Csh·φsh)/φt. Density and neutron read the flushed zone (filtrate and "
         "residual hydrocarbon), Rt reads Sw and RSFL reads Sxo with Rmf. Bounds: 0.005 ≤ φs ≤ 0.45, 0.01 ≤ Sws, Sxos ≤ 1, "
         "enforced by the change of variable x = x_min + (x_max − x_min)/(1 + e^(−u)), which maps each bounded unknown onto an "
         "unbounded u (after Habashy & Abubakar, 2004). Solver: Levenberg–Marquardt on u.", "",
         f"Csh relation: {it._CSH['method']} (see summary.md). RSFL is not fitted (`use_rxo = false`), following the class setup, "
         "because it does not read a fully flushed zone in the water sandstones.", "",
         f"Monte Carlo: {C['realizations']} realizations. Each redraws the GR end-points (±{C['gr_endpoint_sd']} API), shale density and "
         f"neutron, the shale-porosity weight α ({C['alpha'][0]}–{C['alpha'][1]}), ρh {C['rho_h'][0]} ± {C['rho_h'][1]} g/cc, hydrocarbon "
         f"hydrogen index, ρma, m ({C['m'][0]} ± {C['m'][1]}), n ({C['n'][0]} ± {C['n'][1]}), Rmf (±{C['rmf_log_sd']*100:.0f} %), B "
         f"(±{C['b_log_sd']*100:.0f} %) and the surface temperature (±{C['surface_temp_sd']:.0f} °F), recalibrates Rw in the water "
         "sandstones with those inputs, adds noise of the stated σ to the logs, and solves again.", "",
         f"Rw at 75 °F across realizations: P10 {np.percentile(rw_all, 10):.3f}, P50 {np.percentile(rw_all, 50):.3f}, "
         f"P90 {np.percentile(rw_all, 90):.3f} Ω·m (deterministic Juhasz {cal['rw_j75']:.3f}).", "",
         "## Hydrocarbon column", "",
         f"Net hydrocarbon sandstones: P10 {np.percentile(column, 10):.1f}, P50 {np.percentile(column, 50):.1f}, "
         f"P90 {np.percentile(column, 90):.1f} ft (deterministic: Archie {det_cols['col_a']:.1f}, Juhasz {det_cols['col_j']:.1f} ft). "
         "Well assumed vertical.", "",
         per.to_markdown(index=False, floatfmt=("", ".3f", ".3f", ".3f", ".0f", ".3f", ".2f", ".2f", ".2f", ".2f")), "",
         f"Median width of the HPV P10–P90 band in net hydrocarbon sandstones: {np.nanmedian(band[in_hc & net]):.3f} v/v.", "",
         "## Data fit (P50 weighted residuals, (calculated − measured)/σ)", "",
         fit.to_markdown(index=False, floatfmt=("", "", "+.2f", "+.2f", ".2f", "+.2f")), "",
         "An RMS near 1 means the log is fitted to within its assumed noise.", "",
         "## Porosity against core", "",
         f"P50 total porosity at the plugs: bias {e.mean():+.3f}, RMS {np.sqrt((e ** 2).mean()):.3f}, r {np.corrcoef(core_phi, core.phi_3500)[0, 1]:.2f}."]
    (it.OUT / "inversion_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
