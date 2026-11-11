"""PGE 385K Homework Project 2: Archie and Juhasz interpretation (Tasks 5-7).

Run:  python project2/interpret.py
Writes figures, results.csv and summary.md to the Project 2 "Interpretation" folder.

Workflow (from class and the book): first assume no shale and apply Archie over the whole
interval; then compute Csh from gamma ray, correct porosity for grain-coating clay, recalibrate
Rw and apply Juhasz. Equations are in fe_inversion/petro.py.
"""
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import explore as ex  # noqa: E402
from fe_inversion import petro  # noqa: E402

CFG = tomllib.loads((ROOT / "config" / "project2.toml").read_text(encoding="utf-8"))
OUT = ex.P2 / "Interpretation"
A0, M0, N0 = (CFG["archie"][k] for k in "amn")
STEP = 0.5
BASE_SCALES = dict(rw=1.0, phi=1.0, rt=1.0, rxo=1.0, m=1.0, n=1.0, csh=1.0, phi_sh=1.0, b=1.0)


# ------------------------------------------------------------------------------- set-up
def shale_properties(w):
    """Per-depth shale density porosity, neutron and total porosity from the reference shales."""
    s = CFG["shale"]
    vals = {}
    for name in ("upper", "middle"):
        z = w.loc[s[name][0]:s[name][1]]
        z = z[~z.WASHOUT]
        vals[name] = dict(phid=z.PHID.median(), phin=z.NPHI.median())
    upper = w.index < s["boundary"]
    pick = lambda k: np.where(upper, vals["upper"][k], vals["middle"][k])
    phid, phin = pick("phid"), pick("phin")
    return phid, phin, s["alpha"] * phid + (1 - s["alpha"]) * phin, vals


def water_sand_samples(w, sands, vsh):
    """Bed-center clean samples of each water sandstone: {(top, base): index}."""
    c = CFG["water_sands"]
    out = {}
    for _, s in sands[sands.fluid == "water"].iterrows():
        z = w.loc[s.top + c["edge_ft"]:s.base - c["edge_ft"]]
        z = z[(vsh.reindex(z.index) < c["max_vsh"]) & ~z.WASHOUT]
        if len(z) >= c["min_samples"]:
            out[(s.top, s.base)] = z.index
    return out


# --------------------------------------------------------------------------- calculation
def calculate(w, vsh, groups, shale, sc=BASE_SCALES, recalibrate=True, fixed=None):
    """Archie and Juhasz results for one set of input scale factors.

    With recalibrate=True, Rw is re-derived from the water sandstones using the perturbed inputs, as a
    real interpretation would be; otherwise `fixed` is used. Sxo uses the header Rmf.
    """
    a, m, n = A0, M0 * sc["m"], N0 * sc["n"]
    temp = w.TEMP.values
    rt, rxo = w.RIPD.values * sc["rt"], w.RSFL.values * sc["rxo"]
    phid, phin = w.PHID.values * sc["phi"], w.NPHI.values * sc["phi"]
    phid_sh, phin_sh, phi_sh, _ = shale
    phi_sh = phi_sh * sc["phi_sh"]
    pos = {d: i for i, d in enumerate(w.index)}

    # Pass 1: no shale, Archie
    phi_a = petro.rms_porosity(phid, phin)

    def calibrate(values):
        """Median over the water sandstones of a per-sample water resistivity, referred to 75 degF."""
        rows = {}
        for key, idx in groups.items():
            i = [pos[d] for d in idx]
            rows[key] = petro.rw_at_temperature(np.nanmedian(values[i]), temp[i].mean(), 75.0)
        return rows

    # Pass 2: grain-coating clay
    clean = np.concatenate([[pos[d] for d in idx] for idx in groups.values()])
    phi_clean = np.nanpercentile(phi_a[clean], 90)
    i_gr = vsh.values
    csh = np.minimum(i_gr * sc["csh"], phi_clean)
    phi_s, phi_t = petro.grain_coating_porosity(phid, phin, csh, phid_sh, phin_sh, phi_sh)
    phi_t = np.maximum(phi_t, 0.01)
    sb = np.minimum(petro.bound_water_fraction(csh, phi_sh, phi_t), 0.99)

    if recalibrate:
        rw_a_sands = calibrate(petro.archie_rw(rt, phi_a, a, m))
        rz_a_sands = calibrate(petro.archie_rw(rxo, phi_a, a, m))
        rw_j_sands = calibrate(petro.juhasz_rw(rt, phi_t, sb, temp, a, m))
        rz_j_sands = calibrate(petro.juhasz_rw(rxo, phi_t, sb, temp, a, m))
        rw_a75, rz_a75 = (np.median(list(d.values())) for d in (rw_a_sands, rz_a_sands))
        rw_j75, rz_j75 = (np.median(list(d.values())) for d in (rw_j_sands, rz_j_sands))
    else:
        rw_a_sands = rw_j_sands = None
        rw_a75, rz_a75, rw_j75, rz_j75 = (fixed[k] * (sc["rw"] if k.startswith("rw") else 1.0)
                                          for k in ("rw_a75", "rz_a75", "rw_j75", "rz_j75"))
    at_t = lambda r75: petro.rw_at_temperature(r75, 75.0, temp)
    rw_a, rw_j, rmf = at_t(rw_a75), at_t(rw_j75), at_t(ex.RMF_75)

    sw_a_raw = petro.archie_sw(rt, rw_a, phi_a, a, m, n)
    sxo_a_raw = petro.archie_sw(rxo, rmf, phi_a, a, m, n)
    b = petro.waxman_smits_b(temp, rw_j) * sc["b"]
    b_z = petro.waxman_smits_b(temp, rmf) * sc["b"]
    sw_j = petro.juhasz_sw(rt, rw_j, phi_t, sb, b, a, m, n)
    sxo_j = petro.juhasz_sw(rxo, rmf, phi_t, sb, b_z, a, m, n)
    sw_a, sxo_a = np.minimum(sw_a_raw, 1.0), np.minimum(sxo_a_raw, 1.0)

    res = pd.DataFrame(dict(
        temp=temp, i_gr=i_gr, csh=csh, capped=i_gr * sc["csh"] > phi_clean, phi_a=phi_a, phi_s=phi_s, phi_t=phi_t,
        sb=sb, rw_a=rw_a, rw_j=rw_j, b=b, sw_a_raw=sw_a_raw, sw_a=sw_a, sxo_a=sxo_a, sw_j=sw_j, sxo_j=sxo_j,
        sws=np.clip(petro.sandstone_sw(sw_j, sb), 0, 1), hpv_a=phi_a * (1 - sw_a), wpv_a=phi_a * sw_a,
        hpv_j=phi_t * (1 - sw_j), wpv_j=phi_t * sw_j), index=w.index)
    return res, dict(rw_a75=rw_a75, rz_a75=rz_a75, rw_j75=rw_j75, rz_j75=rz_j75, phi_clean=phi_clean,
                     rw_a_sands=rw_a_sands, rw_j_sands=rw_j_sands)


def hc_summary(res, zones):
    """Depth-summed hydrocarbon column over the given sandstones (STEP-ft samples, vertical well assumed)."""
    idx = np.zeros(len(res), bool)
    for _, s in zones.iterrows():
        idx |= (res.index >= s.top) & (res.index <= s.base)
    net = idx & (res.i_gr.values < CFG["net"]["max_vsh"])
    return dict(col_a=res.hpv_a[net].sum() * STEP, col_j=res.hpv_j[net].sum() * STEP, sw_j=res.sw_j[net].mean(),
                sw_a=res.sw_a[net].mean(), n=int(net.sum()))


def sensitivity(w, vsh, groups, shale, zones, base):
    """One-at-a-time +/- perturbations of each input, with Rw held and with Rw recalibrated."""
    f = CFG["sensitivity"]["fraction"]
    ref, _ = calculate(w, vsh, groups, shale, recalibrate=False, fixed=base)
    b0 = hc_summary(ref, zones)
    rows = []
    labels = dict(rw="Rw", phi="Porosity logs", rt="Rt", m="m", n="n", csh="Csh", phi_sh="Shale porosity", b="B")
    for k, lab in labels.items():
        row = dict(input=lab)
        for sign, tag in ((-1, "lo"), (1, "hi")):
            sc = dict(BASE_SCALES); sc[k] = 1 + sign * f
            held, _ = calculate(w, vsh, groups, shale, sc, recalibrate=False, fixed=base)
            h = hc_summary(held, zones)
            row[f"sw_{tag}"] = 100 * (h["sw_j"] / b0["sw_j"] - 1)
            row[f"col_{tag}"] = 100 * (h["col_j"] / b0["col_j"] - 1)
            if k != "rw":
                rec, _ = calculate(w, vsh, groups, shale, sc, recalibrate=True)
                r = hc_summary(rec, zones)
                row[f"sw_rec_{tag}"] = 100 * (r["sw_j"] / b0["sw_j"] - 1)
        rows.append(row)
    return pd.DataFrame(rows), b0


def alpha_sweep(w, vsh, groups, zones, core):
    """Porosity bias against core and saturation results for a range of shale-porosity weights."""
    base, rows = CFG["shale"]["alpha"], []
    for al in (1.0, 0.8, 0.6, 0.5, 0.4):
        CFG["shale"]["alpha"] = al
        sh = shale_properties(w)
        r, _ = calculate(w, vsh, groups, sh)
        e = np.interp(core.depth_log, r.index, r.phi_t) - core.phi_3500
        h = hc_summary(r, zones)
        m = np.zeros(len(r), bool)
        for _, s in zones.iterrows():
            m |= (r.index >= s.top) & (r.index <= s.base)
        m &= r.i_gr.values < CFG["net"]["max_vsh"]
        rows.append(dict(alpha=al, phi_sh_mean=float(np.mean(sh[2])), bias_vs_core=float(e.mean()), col_j=h["col_j"], sw_j=h["sw_j"],
                         pct_at_sb=100 * float((r.sw_j[m] <= r.sb[m] + 1e-3).mean())))
    CFG["shale"]["alpha"] = base
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- figures
def fig_results(w, res, sands, core, path):
    fig, ax = plt.subplots(1, 7, figsize=(17, 15), sharey=True)
    ax[0].plot(w.GR, w.index, "g", lw=0.7, label="GR (API)"); ax[0].set(xlim=(0, 150), title="Gamma ray")
    for k, col in [("RIPD", "r"), ("RSFL", "orange")]:
        ax[1].semilogx(w[k].clip(0.2, 2000), w.index, color=col, lw=0.7, label=k)
    ax[1].set(xlim=(0.3, 300), title="Resistivity (Ω·m)")
    ax[2].plot(res.i_gr, res.index, "0.6", lw=0.6, label="GR index")
    ax[2].plot(res.csh, res.index, "k", lw=0.8, label="Csh (capped at φclean)"); ax[2].set(xlim=(0, 1), title="Csh (v/v)")
    ax[3].plot(res.phi_a, res.index, "b", lw=0.7, label="Archie φ (no shale)")
    ax[3].plot(res.phi_t, res.index, "k", lw=0.7, label="φt (Juhasz)")
    ax[3].plot(res.phi_s, res.index, "m", lw=0.6, label="φs (non-shale)")
    ax[3].plot(core.phi_3500, core.depth_log, "ro", ms=3.5, mfc="none", label="Core φ (3,500 psi)")
    ax[3].set(xlim=(0.5, 0), title="Porosity (v/v)")
    ax[4].plot(res.sw_a, res.index, "b", lw=0.7, label="Archie"); ax[4].plot(res.sw_j, res.index, "k", lw=0.7, label="Juhasz")
    ax[4].set(xlim=(1, 0), title="Sw, total (v/v)")
    ax[5].plot(res.sxo_a, res.index, "b", lw=0.7, label="Archie"); ax[5].plot(res.sxo_j, res.index, "k", lw=0.7, label="Juhasz")
    ax[5].set(xlim=(1, 0), title="Sxo (v/v)")
    net = res.i_gr < CFG["net"]["max_vsh"]
    ax[6].fill_betweenx(res.index, 0, res.hpv_j.where(net, 0), color="#2e9e44", step="mid", lw=0, label="HPV Juhasz (net)")
    ax[6].plot(res.hpv_a, res.index, "b", lw=0.5, label="HPV Archie (no shale)")
    ax[6].set(xlim=(0, 0.35), title="HPV (v/v)")
    for a in ax:
        ex.shade_sands(a, sands); a.grid(alpha=0.3); a.legend(fontsize=6.5, loc="lower right")
    ax[0].set_ylim(ex.BASE, ex.TOP); ax[0].set_ylabel("Measured depth (ft)")
    fig.suptitle("ORM-75: Archie (no shale) vs. Juhasz (grain-coating clay), 5,400–6,300 ft", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_rw(table, rw_a75, rw_j75, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(table)); wd = 0.27
    ax.bar(x - wd, table.rw_archie, wd, label="Archie", color="#4c78a8")
    ax.bar(x, table.rw_juhasz, wd, label="Juhasz", color="#333333")
    ax.bar(x + wd, table.rw_sp, wd, label="SP", color="#e45756")
    ax.axhline(rw_a75, color="#4c78a8", ls="--", lw=1, label=f"Archie median {rw_a75:.3f}")
    ax.axhline(rw_j75, color="#333333", ls=":", lw=1, label=f"Juhasz median {rw_j75:.3f}")
    ax.set_xticks(x, table.sandstone, rotation=30, ha="right")
    ax.set(ylabel="Rw at 75 °F (Ω·m)", xlabel="Water sandstone (ft MD)", title="Rw from each clean water sandstone")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_compare(res, core, net, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    r = res[net]
    s = ax[0].scatter(r.sw_a, r.sw_j, c=r.csh, s=5, cmap="viridis")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="1:1")
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Sw, Archie (v/v)", ylabel="Sw, Juhasz (v/v)", title="(a) Sw: Juhasz vs. Archie, net sandstones")
    plt.colorbar(s, ax=ax[0], label="Csh (v/v)")
    for col, lab, c in [("phi_a", "Archie φ", "b"), ("phi_t", "φt", "k"), ("phi_sonic", "Sonic φ", "r")]:
        ax[1].scatter(core.phi_3500, core[col], s=16, color=c, alpha=0.7, label=lab)
    ax[1].plot([0, 0.45], [0, 0.45], "k--", lw=1, label="1:1")
    ax[1].set(xlim=(0.05, 0.45), ylim=(0.05, 0.45), xlabel="Core porosity at 3,500 psi (v/v)", ylabel="Log porosity (v/v)", title="(b) Log vs. core porosity")
    ax[2].scatter(r.sw_j, r.sxo_j, c=r.phi_t, s=5, cmap="plasma")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1, label="Sxo = Sw")
    ax[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Sw, Juhasz (v/v)", ylabel="Sxo, Juhasz (v/v)", title="(c) Sxo vs. Sw, net sandstones")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_juhasz_plot(x, y, fit, b_book, max_sb, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x, y, s=8, color="#4c78a8", label="Water-sandstone samples")
    ax.axvline(max_sb, color="0.5", ls=":", lw=1, label=f"Fit limit, Sb = {max_sb}")
    xx = np.linspace(0, np.nanmax(x), 20)
    ax.plot(xx, fit[1] + fit[0] * xx, "k", lw=1.5, label=f"Fit (Sb < {max_sb}): 1/Rw = {fit[1]:.1f} S/m, B = {fit[0]:.1f}")
    ax.plot(xx, fit[1] + b_book * xx, "r--", lw=1.2, label=f"Book B = {b_book:.1f}, same intercept")
    ax.set(xlabel="Sb = Csh·φsh/φt (v/v)", ylabel="a / (Rt·φt^m) (S/m)", title="Juhasz normalization in the water sandstones (Sw = 1)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_tornado(sens, path):
    s = sens.assign(span=(sens.sw_hi - sens.sw_lo).abs()).sort_values("span")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(s))
    ax.barh(y, s.sw_lo, color="#4c78a8", label="Input −10 %")
    ax.barh(y, s.sw_hi, color="#e45756", label="Input +10 %")
    ax.set_yticks(y, s.input); ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Change in mean Sw of the hydrocarbon sandstones (%)", ylabel="Input", title="Sensitivity of Juhasz Sw to ±10 % inputs (Rw held)")
    ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    df, core = ex.load_logs(), ex.load_core()
    w = df.loc[ex.TOP:ex.BASE]
    sands, vsh, (gr_cl, gr_sh) = ex.pick_sands(w)
    sands, _ = ex.classify(sands)
    ex.RW_EST = None
    groups = water_sand_samples(w, sands, vsh)
    shale = shale_properties(w)
    res, cal = calculate(w, vsh, groups, shale)
    hc = sands[sands.fluid == "possible hydrocarbon"]
    net = res.i_gr < CFG["net"]["max_vsh"]

    # Rw per sand, with the SP check
    shale_base = np.mean([w.loc[slice(*CFG["shale"][k])].SP.median() for k in ("upper", "middle")])
    rows = []
    for key, idx in groups.items():
        z = w.loc[idx]; t = z.TEMP.mean()
        sp = ex.rw_from_sp(z.SP.median() - shale_base, t)
        rows.append(dict(sandstone=f"{key[0]:.0f}–{key[1]:.0f}", n=len(idx), temp=t, ssp=z.SP.median() - shale_base,
                         phi_nd_diff=(z.NPHI - z.PHID).median(), rw_archie=cal["rw_a_sands"][key],
                         rw_juhasz=cal["rw_j_sands"][key], rw_sp=petro.rw_at_temperature(sp["rw"], t, 75.0)))
    rw_table = pd.DataFrame(rows)

    # Juhasz normalization plot in all water sandstones, at mean temperature
    ws = sands[sands.fluid == "water"]
    in_ws = np.zeros(len(w), bool)
    for _, s in ws.iterrows():
        in_ws |= (w.index >= s.top + 1.5) & (w.index <= s.base - 1.5)
    in_ws &= ~w.WASHOUT.values & (res.csh.values > 0)
    t_mean = res.temp[in_ws].mean()
    scale = (res.temp[in_ws] + 6.77) / (t_mean + 6.77)           # refer conductivities to the mean temperature
    x = res.sb[in_ws].values
    y = (A0 / (w.RIPD[in_ws] * res.phi_t[in_ws] ** M0) / scale).values
    low = x < CFG["juhasz"]["fit_max_sb"]
    fit, cov = np.polyfit(x[low], y[low], 1, cov=True)
    b_se = float(np.sqrt(cov[0, 0]))
    fit_all = np.polyfit(x, y, 1)
    rw_t = petro.rw_at_temperature(cal["rw_j75"], 75.0, t_mean)
    b_book = float(petro.waxman_smits_b(t_mean, rw_t))

    # Core and sonic
    for col in ("phi_a", "phi_t", "phi_s", "sw_a", "sw_j", "sxo_j"):
        core[col] = np.interp(core.depth_log, res.index, res[col])
    core["zden"] = np.interp(core.depth_log, w.index, w.ZDEN)
    m = CFG["matrix"]
    wy = petro.wyllie_porosity(w.DT24.values, m["dt_ma"], m["dt_f"])
    clean = np.concatenate([np.array(idx) for idx in groups.values()])
    cp = float(np.median(pd.Series(wy, index=w.index).loc[clean] / res.phi_a.loc[clean]))
    res["phi_sonic"] = wy / cp
    core["phi_sonic"] = np.interp(core.depth_log, res.index, res.phi_sonic)
    stats = lambda c: (float((core[c] - core.phi_3500).mean()), float(np.sqrt(((core[c] - core.phi_3500) ** 2).mean())),
                       float(np.corrcoef(core[c], core.phi_3500)[0, 1]))
    # Pore-fluid density at the plugs from core porosity and grain density, then the hydrocarbon density
    core["rho_f"] = (core.zden - (1 - core.phi_3500) * core.rho_grain) / core.phi_3500
    good = core.phi_3500 > 0.25

    # Per-sand table
    per = []
    for _, s in hc.iterrows():
        z = res.loc[s.top:s.base]; z = z[z.i_gr < CFG["net"]["max_vsh"]]
        per.append(dict(sandstone=f"{s.top:.0f}–{s.base:.0f}", net_ft=len(z) * STEP, csh=z.csh.mean(), phi_a=z.phi_a.mean(), phi_t=z.phi_t.mean(),
                        phi_s=z.phi_s.mean(), sw_a=z.sw_a.mean(), sw_j=z.sw_j.mean(), sws=z.sws.mean(), sxo_j=z.sxo_j.mean(),
                        col_a=z.hpv_a.sum() * STEP, col_j=z.hpv_j.sum() * STEP, wat_j=z.wpv_j.sum() * STEP))
    per = pd.DataFrame(per)
    tot = hc_summary(res, hc)
    sens, b0 = sensitivity(w, vsh, groups, shale, hc, cal)
    alpha_table = alpha_sweep(w, vsh, groups, hc, core)

    fig_results(w, res, sands, core, OUT / "01_results.png")
    fig_rw(rw_table, cal["rw_a75"], cal["rw_j75"], OUT / "02_rw_by_sand.png")
    fig_compare(res, core, net, OUT / "03_comparisons.png")
    fig_juhasz_plot(x, y, fit, b_book, CFG["juhasz"]["fit_max_sb"], OUT / "04_juhasz_normalization.png")
    fig_tornado(sens, OUT / "05_sensitivity.png")
    res.round(5).to_csv(OUT / "results.csv", index_label="depth_ft")

    sv = shale[3]
    wsamples = np.concatenate([np.array(i) for i in groups.values()])
    in_hc = np.zeros(len(res), bool)
    for _, s in hc.iterrows():
        in_hc |= (res.index >= s.top) & (res.index <= s.base)
    in_hc &= net.values
    sxo_flag = (res.sxo_j[in_hc] < res.sw_j[in_hc] - 0.02).mean()
    hc_sxo = res.sxo_j[in_hc].mean()
    peg = (res.sw_j[in_hc] <= res.sb[in_hc] + 1e-3).mean()
    bvw = res.phi_t[in_hc] * res.sw_j[in_hc]
    rmf_t = petro.rw_at_temperature(ex.RMF_75, 75.0, res.temp.loc[wsamples].mean())
    sxo_header = petro.archie_sw(w.RSFL.loc[wsamples], rmf_t, res.phi_a.loc[wsamples], A0, M0, N0)
    whole_a = res.hpv_a.sum() * STEP
    L = ["# Project 2: Archie and Juhasz results", "",
         f"Interval {ex.TOP:.0f}–{ex.BASE:.0f} ft MD. a = {A0}, m = {M0}, n = {N0}. Well assumed vertical (no survey in the data). "
         "Generated by `project2/interpret.py`.", "",
         "## Water resistivity", "",
         rw_table.to_markdown(index=False, floatfmt=("", ".0f", ".0f", ".1f", "+.3f", ".3f", ".3f", ".3f")), "",
         f"Rw at 75 °F: Archie {cal['rw_a75']:.3f}, Juhasz {cal['rw_j75']:.3f}, SP median {rw_table.rw_sp.median():.3f} Ω·m. "
         "The SP reads about 20 % higher; it depends on the assumed temperature, the Rmf-to-Rmfe conversion and the shale baseline, "
         "and thin sandstones do not develop the full static SP.", "",
         f"Juhasz normalization in the water sandstones, fitted for Sb < {CFG['juhasz']['fit_max_sb']} ({int(low.sum())} samples): "
         f"intercept 1/Rw = {fit[1]:.1f} S/m (Rw = {1/fit[1]:.3f} Ω·m at {t_mean:.0f} °F, or "
         f"{petro.rw_at_temperature(1/fit[1], t_mean, 75.0):.3f} Ω·m at 75 °F), fitted B = {fit[0]:.1f} ± {b_se:.1f}, book eq. A.14.4 B = {b_book:.1f}. "
         "Clay conduction is not detectable above the scatter in these saline water sandstones, as expected: at Sb = 0.1 the book's B "
         "adds only about 7 % to the water conductivity. "
         f"Including the shaliest samples the slope rises to {fit_all[0]:.1f}, which likely reflects laminated shale that the "
         "dispersed-clay model does not represent.", "",
         "## Shale and clay", "",
         f"GR clean {gr_cl:.0f}, GR shale {gr_sh:.0f} API. φclean (Csh cap) = {cal['phi_clean']:.3f}. "
         f"Upper shale φD {sv['upper']['phid']:.3f}, φN {sv['upper']['phin']:.3f}; middle shale φD {sv['middle']['phid']:.3f}, "
         f"φN {sv['middle']['phin']:.3f}. Of the net samples, {res.capped[net].mean()*100:.0f} % have a GR index above the cap.", "",
         "## Hydrocarbon sandstones", "",
         per.to_markdown(index=False, floatfmt=("", ".1f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f", ".2f", ".2f", ".2f")), "",
         f"Total hydrocarbon column in net hydrocarbon sandstones: Archie {tot['col_a']:.2f} ft, Juhasz {tot['col_j']:.2f} ft "
         f"({100*(tot['col_j']/tot['col_a']-1):+.1f} %). Mean Sw: Archie {tot['sw_a']:.3f}, Juhasz {tot['sw_j']:.3f}.", "",
         f"Archie with no shale over the whole interval gives {whole_a:.1f} ft of apparent hydrocarbon column, most of it "
         "false: in shale the high neutron reading inflates porosity, so the predicted water-bearing resistivity is too low and "
         "the measured resistivity reads as hydrocarbon.", "",
         "## Porosity against core and sonic", "",
         "| Log porosity | Bias | RMS | r |", "|---|---|---|---|"]
    for c, lab in [("phi_a", "Archie φ (no shale)"), ("phi_t", "φt (Juhasz)"), ("phi_s", "φs (non-shale)"), ("phi_sonic", f"Sonic (Wyllie, Cp = {cp:.2f})")]:
        b_, r_, c_ = stats(c); L.append(f"| {lab} | {b_:+.3f} | {r_:.3f} | {c_:.2f} |")
    L += ["", "## Fluid density at the plugs", "",
          f"Pore-fluid density from core porosity (3,500 psi), core grain density and ZDEN, at plugs with φ > 0.25: median "
          f"{core.rho_f[good].median():.2f} g/cc, interquartile range {core.rho_f[good].quantile(.25):.2f}–"
          f"{core.rho_f[good].quantile(.75):.2f} ({int(good.sum())} plugs). This is water-like: the density tool reads mainly the "
          "invaded zone, and a 0.02 error in porosity moves the result by about 0.1 g/cc. There are no pressure data in this "
          "project, so hydrocarbon density cannot be resolved. The absence of neutron-density crossover in the hydrocarbon sandstones "
          "argues against gas in the zone the tools read.", "",
          "## Sxo and consistency", "",
          f"- Sxo uses the header Rmf at formation temperature. In the water sandstones the unclipped value has a median of "
          f"{np.nanmedian(sxo_header):.2f}, i.e. above 1: RSFL there reads about {cal['rz_a75']/ex.RMF_75*100:.0f} % of what a fully "
          "flushed zone would, so filtrate has not fully replaced the saline water near the well. Sxo in water sandstones is limited to 1.",
          f"- In the hydrocarbon sandstones, Sxo averages {hc_sxo:.2f} against Sw {tot['sw_j']:.2f}, and Sxo is below Sw by more than 0.02 at "
          f"only {sxo_flag*100:.0f} % of samples, so Sw and Sxo are consistent with total porosity: invasion displaced part of the hydrocarbon.",
          f"- Bulk volume water φt·Sw in the hydrocarbon sandstones: median {bvw.median():.3f}, coefficient of variation "
          f"{bvw.std()/bvw.mean():.2f}; r(φt, Sw) = {np.corrcoef(res.phi_t[in_hc], res.sw_j[in_hc])[0, 1]:.2f}. Sw falls as porosity rises, "
          "as expected, but BVW is not constant, which indicates several rock classes, sandstones not at irreducible saturation, or both; "
          "the logs alone cannot separate these.",
          f"- Juhasz Sw reaches its lower limit (Sw = Sb, no free water) at {peg*100:.0f} % of net hydrocarbon samples.",
          f"- Core Sw vs. log Sw at the plugs: mean core {core.sw.mean():.2f}, Archie {core.sw_a.mean():.2f}, Juhasz {core.sw_j.mean():.2f}. "
          f"Core So + Sw has a median of {(core.so + core.sw).median():.2f}, so the plugs lost fluid and core Sw is not a reliable check.", "",
          "## Sensitivity (±10 % on each input, hydrocarbon sandstones)", "",
          "Change in mean Juhasz Sw (%), with Rw held, and with Rw recalibrated in the water sandstones using the same perturbed input.", "",
          sens[["input", "sw_lo", "sw_hi", "sw_rec_lo", "sw_rec_hi", "col_lo", "col_hi"]].map(
              lambda v: v if isinstance(v, str) else ("n/a" if pd.isna(v) else f"{v:+.1f}")).to_markdown(index=False), "",
          "## Effect of the shale-porosity choice (α in book eq. 10.15.11)", "",
          alpha_table.to_markdown(index=False, floatfmt=(".1f", ".3f", "+.3f", ".2f", ".3f", ".0f")), "",
          "α near 0.5 removes the porosity bias against core but forces about a third of the hydrocarbon samples to Sw = Sb; "
          "α = 1 keeps the saturations physical but reads low against core. The results use α = 1.", "",
          "## Choices made (for the appendix)", "",
          "- Rt = RIPD with no invasion or shoulder-bed correction (charts not digitized); Rw from bed-center samples at least 1.5 ft inside sandstones.",
          "- Rw = median over clean water sandstones, referred to 75 °F with Arps, not one sandstone, because the sandstones disagree by about ±10 %.",
          f"- Temperature linear from {ex.T_SURF:.0f} °F (assumed) at surface to {ex.BHT:.0f} °F at {ex.BHT_DEPTH:.0f} ft.",
          "- Csh = linear GR index capped at φclean, per class and book A.11.b; the GR-to-volume conversion is not applied.",
          f"- φsh = α·φD,sh + (1 − α)·φN,sh with α = {CFG['shale']['alpha']} (book 10.15.11).",
          "- Reference shale by depth: upper shale above 5,885 ft, middle shale below.",
          "- m and n are the assignment's values for both Archie and Juhasz, although book §10.16.a says clay changes them."]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
