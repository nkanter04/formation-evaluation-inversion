"""Remaining Project 1 checks: core depth shift, hydrocarbon density, Sw-Sxo, permeability.

Run after the main program (it reads its results.csv):
    python -m fe_inversion
    python -m fe_inversion.project_checks
"""
import argparse
import copy
import sys

import lasio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io, manual_hpv, workflow

PAY = (9825.0, 10005.0)   # hydrocarbon zone inside Section 2 (Sw jumps to ~1 below ~10,005 ft)


def _style(ax):
    ax.grid(True, color="#DDE3EA", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FCFCFD")


def _fit(x, y):
    """log10(y) = a + b x least squares; returns a, b, R^2."""
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    b, a = np.polyfit(x[ok], np.log10(y[ok]), 1)
    r2 = np.corrcoef(x[ok], np.log10(y[ok]))[0, 1] ** 2
    return a, b, r2, int(ok.sum())


def load(cfg):
    """Well window, raw LAS, inversion results and the HW 6 calculation."""
    d = cfg["data"]
    full, las = io.load_well(d["las"], cfg["curves"])
    raw = las.df()
    win = full.loc[d["top"]:d["bottom"]].copy()
    for c in ("GR_CORE", "DEPTH_ORIG", "PHID_SS"):
        win[c] = raw.loc[win.index, c].values if c in raw else np.nan
    inv = pd.read_csv(io.ensure_dir(d["output_dir"]) / "results.csv", index_col=0)
    if len(inv) != len(win):
        raise SystemExit("results.csv does not match the interval; rerun `python -m fe_inversion` first")
    inv.index = win.index   # depths in the CSV are rounded; rows match the window
    hw6 = manual_hpv.run(cfg)
    # Same calculation with HW 6's own porosity rule, which does not assume an oil density
    # (needed to estimate the hydrocarbon density from the logs without circularity).
    cfg_ind = copy.deepcopy(cfg)
    cfg_ind["manual_hpv"]["porosity_method"] = "hw6"
    hw6_ind = manual_hpv.run(cfg_ind)
    return win, raw, inv, hw6, hw6_ind


# --------------------------------------------------------------------- 1. depth shift
def depth_shift(win, path):
    c = win[win.GR_CORE.notna() & win.DEPTH_ORIG.notna()]
    shift = (c.index - c.DEPTH_ORIG).values
    log_d, log_gr = win.index.values, win.GR.interpolate().values
    lags = np.arange(-15, 15.01, 0.25)

    def corr(depths):
        return np.array([np.corrcoef(np.interp(depths + s, log_d, log_gr), c.GR_CORE)[0, 1] for s in lags])

    r_shifted, r_orig = corr(c.index.values), corr(c.DEPTH_ORIG.values)
    best_shifted, best_orig = lags[np.argmax(r_shifted)], lags[np.argmax(r_orig)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    for a in ax:
        _style(a)
    ax[0].plot(shift, c.index, color="#1565C0", lw=1.2)
    ax[0].set(xlabel="Applied shift = log depth - original core depth (ft)", ylabel="Log depth (ft MD)",
              title=f"(a) Shift in the LAS: {shift.min():.1f} to {shift.max():.1f} ft (median {np.median(shift):.1f})")
    ax[0].invert_yaxis()
    ax[1].plot(lags, r_orig, color="#C62828", lw=1.4,
               label=f"Core at original depths: best extra shift {best_orig:+.2f} ft (r = {r_orig.max():.2f})")
    ax[1].plot(lags, r_shifted, color="#2E7D32", lw=1.4,
               label=f"Core at shifted depths: best extra shift {best_shifted:+.2f} ft (r = {r_shifted.max():.2f})")
    ax[1].axvline(0, color="k", lw=0.8, ls=":")
    ax[1].set(xlabel="Extra shift applied to the core (ft, + = deeper)", ylabel="Correlation, core GR vs. log GR",
              title="(b) Core-to-log gamma-ray correlation vs. depth shift")
    ax[1].legend(loc="lower center", fontsize=8)
    m = (win.index >= c.index.min() - 15) & (win.index <= c.index.max() + 5)
    ax[2].plot(win.GR[m], win.index[m], color="k", lw=0.9, label="Log GR")
    ax[2].plot(c.GR_CORE + 0, c.DEPTH_ORIG, color="#C62828", lw=0.9, label="Core GR, original depth")
    ax[2].plot(c.GR_CORE, c.index, color="#2E7D32", lw=0.9, label="Core GR, shifted depth")
    ax[2].set(xlabel="Gamma ray (gAPI; core GR on its own scale)", ylabel="Depth (ft MD)",
              title="(c) Core vs. log gamma ray")
    ax[2].invert_yaxis()
    ax[2].legend(loc="lower right", fontsize=8)
    fig.suptitle("Core-to-log depth matching", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(n=len(c), shift_min=shift.min(), shift_max=shift.max(), shift_med=float(np.median(shift)),
                best_shifted=best_shifted, r_shifted=r_shifted.max(), r_zero=r_shifted[np.argmin(np.abs(lags))],
                best_orig=best_orig, r_orig=r_orig.max(), r_orig_zero=r_orig[np.argmin(np.abs(lags))])


# --------------------------------------------------------------------- 2. hydrocarbon density
def hc_density(win, hw6_ind, cfg, pres_density, path):
    m = cfg["manual_hpv"]
    res, sh, ar = hw6_ind["res"], hw6_ind["shale"], hw6_ind["a_res"]
    rho_w, archie = float(m["rho_w"]), hw6_ind["archie"]
    pay = res.loc[PAY[0]:PAY[1]]
    # (a) logs only: HW 6 porosity rule (no oil density assumed)
    rf = manual_hpv.fluid_density(pay.PHID, pay.CSH, sh["phi_d"], pay.PHI_S, float(m["rho_ma"]))
    rh = manual_hpv.hydrocarbon_density(rf, pay.SW, rho_w)
    keep = (pay.FLAG == "") & (pay.SW <= 0.6) & (pay.PHI_S >= 0.10) & (pay.CSH <= 0.6)
    logs_only = pd.DataFrame({"RHO_F": rf, "RHO_HC": rh})[keep]
    # (b) core-anchored: core porosity as phi_s at the plugs, Sw re-solved with it
    plugs = win.loc[pay.index][win.loc[pay.index].CORE_POR.notna()].index
    p = pay.loc[plugs]
    phi_c = win.loc[plugs, "CORE_POR"].values
    sw_c = manual_hpv.archie_sw(phi_c, p.RS.values, ar["rw"], archie)
    rf_c = manual_hpv.fluid_density(p.PHID.values, p.CSH.values, sh["phi_d"], phi_c, float(m["rho_ma"]))
    rh_c = manual_hpv.hydrocarbon_density(rf_c, sw_c, rho_w)
    kc = (p.FLAG == "").values & (sw_c <= 0.6) & (p.CSH.values <= 0.6)
    core_anch = pd.DataFrame({"RHO_F": rf_c, "RHO_HC": rh_c, "SW": sw_c}, index=plugs)[kc]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.3]})
    for a in ax:
        _style(a)
    ax[0].scatter(logs_only.RHO_HC, logs_only.index, s=6, color="#90A4AE", label="Logs only (HW 6 porosity)")
    ax[0].scatter(core_anch.RHO_HC, core_anch.index, s=22, color="#C62828", label="Core-anchored (core porosity)")
    ax[0].axvline(pres_density, color="#2E7D32", lw=2, label=f"Pressure gradient {pres_density:.3f} g/cc")
    ax[0].axvline(rho_w, color="#1565C0", lw=1, ls="--", label="Formation water")
    ax[0].set(xlim=(-0.5, 1.5), xlabel="Hydrocarbon density (g/cc)", ylabel="Measured depth (ft)",
              title="(a) rho_H = (rho_f - Sw rho_w) / (1 - Sw), point by point")
    ax[0].invert_yaxis()
    ax[0].legend(loc="lower left", fontsize=8)
    bins = np.linspace(-0.5, 1.5, 41)
    ax[1].hist(logs_only.RHO_HC.clip(-0.5, 1.5), bins=bins, color="#90A4AE", alpha=0.7,
               label=f"Logs only: median {logs_only.RHO_HC.median():.2f} (n = {len(logs_only)})")
    ax[1].hist(core_anch.RHO_HC.clip(-0.5, 1.5), bins=bins, color="#C62828", alpha=0.6,
               label=f"Core-anchored: median {core_anch.RHO_HC.median():.2f} (n = {len(core_anch)})")
    ax[1].axvline(pres_density, color="#2E7D32", lw=2, label=f"Pressure gradient {pres_density:.3f}")
    ax[1].set(xlabel="Hydrocarbon density (g/cc)", ylabel="Samples",
              title="(b) Distribution in the pay (Sw <= 0.6, Csh <= 0.6, phi_s >= 0.10)")
    ax[1].legend(fontsize=8)
    fig.suptitle("Hydrocarbon density from the logs (HW 6 parts h-i) vs. the formation-pressure gradient",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    q = lambda s: (s.median(), s.quantile(0.25), s.quantile(0.75), len(s))
    return dict(logs=q(logs_only.RHO_HC), core=q(core_anch.RHO_HC), rho_f_logs=logs_only.RHO_F.median(),
                rho_f_core=core_anch.RHO_F.median(), pressure=pres_density)


# --------------------------------------------------------------------- 3. Sw vs. Sxo
def sw_sxo(win, inv, n_exp, water, path):
    """Sw-Sxo consistency in the pay.

    Two Sxo estimates: the inversion's (AT10 treated as a parallel-to-bedding flushed-zone
    reading), and the ratio method Rxo/Rt = (Rmf/Rw) (Sw/Sxo)^n with Rmf/Rw = AT10/AT90 in
    the water sand, which cancels most of the deviation/anisotropy effect on both curves.
    """
    wz = win.loc[water[0]:water[1]]
    rmf_rw = float((wz.RXO / wz.RT).median())
    pa = inv.loc[PAY[0]:PAY[1]]
    wp = win.loc[pa.index]
    sxo_ratio = (pa.sw_p50 * (rmf_rw / (wp.RXO / wp.RT)) ** (1 / n_exp)).clip(0, 1)
    p = pa[pa.sw_p50.notna() & pa.phi_p50.notna()]
    sw, phi = p.sw_p50, p.phi_p50
    sxo = sxo_ratio.loc[p.index]
    inv_sxo = p.sxo_p50
    d = p.index.values
    below = sxo < sw - 0.05
    inv_ok = inv_sxo.notna()
    fig, ax = plt.subplots(1, 3, figsize=(17, 7), gridspec_kw={"width_ratios": [1.1, 0.8, 1.1]})
    for a in ax:
        _style(a)
    ax[0].scatter(sw[inv_ok], inv_sxo[inv_ok], s=6, color="#BDBDBD",
                  label=f"Inversion Sxo (AT10 as parallel), defined at {inv_ok.mean():.0%}")
    sc = ax[0].scatter(sw, sxo, c=d, cmap="viridis_r", s=6, label="Ratio-method Sxo")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Sw, deep (P50)", ylabel="Sxo, flushed zone",
              title=f"(a) Ratio-method Sxo < Sw - 0.05 at {below.mean():.0%} of pay samples")
    ax[0].legend(loc="upper left", fontsize=7.5)
    fig.colorbar(sc, ax=ax[0], label="Depth (ft)")
    ax[1].plot(phi * sw, d, color="#1565C0", lw=0.8, label="BVW = phi Sw")
    ax[1].plot(phi * sxo, d, color="#AD1457", lw=0.8, label="BVWxo = phi Sxo")
    ax[1].fill_betweenx(d, phi * sw, phi * sxo, where=(sxo > sw).values, color="#A5D6A7", alpha=0.6, lw=0,
                        label="Moved hydrocarbon")
    ax[1].fill_betweenx(d, phi * sw, phi * sxo, where=(sxo < sw).values, color="#FFCC80", alpha=0.6, lw=0,
                        label="Sxo < Sw")
    ax[1].set(xlim=(0, 0.35), xlabel="Bulk volume water (v/v)", ylabel="Measured depth (ft)",
              title="(b) Bulk volume water, deep vs. flushed")
    ax[1].invert_yaxis()
    ax[1].legend(loc="lower right", fontsize=7)
    sc = ax[2].scatter(sw, phi, c=d, cmap="viridis_r", s=6)
    s_grid = np.linspace(0.05, 1, 100)
    for bvw in (0.04, 0.06, 0.08, 0.10):
        ax[2].plot(s_grid, bvw / s_grid, color="#455A64", lw=0.8, ls="--")
        ax[2].annotate(f"BVW {bvw:.2f}", (0.97, bvw / 0.97), textcoords="offset points", xytext=(-38, 3),
                       fontsize=7, color="#455A64")
    ax[2].set(xlim=(0, 1), ylim=(0, 0.45), xlabel="Sw (P50)", ylabel="Sand porosity (P50)",
              title=f"(c) Buckles plot: median BVW {np.median(phi * sw):.3f}, CV {np.std(phi * sw) / np.mean(phi * sw):.2f}")
    fig.colorbar(sc, ax=ax[2], label="Depth (ft)")
    fig.suptitle("Consistency of Sw, Sxo and porosity in the pay (two-step inversion, P50)",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    bvw = phi * sw
    return dict(n=len(p), rmf_rw=rmf_rw, inv_defined=float(inv_ok.mean()),
                inv_below=float((inv_sxo[inv_ok] < sw[inv_ok] - 0.05).mean()),
                frac_below=below.mean(), med_diff=float(np.median(sxo - sw)),
                moved=float(np.median(phi * (sxo - sw))),
                bvw_med=float(np.median(bvw)), bvw_cv=float(np.std(bvw) / np.mean(bvw)),
                bvw_r_phi=float(np.corrcoef(phi, sw)[0, 1]))


# --------------------------------------------------------------------- 4. permeability
def permeability(win, inv, hw6, height_ft, path):
    plugs = win[win.CORE_POR.notna() & (win.CORE_PERM > 0)].index
    k = win.loc[plugs, "CORE_PERM"].values
    res = hw6["res"]
    in_hw6 = plugs.isin(res.index)
    phi_hw6 = np.full(len(plugs), np.nan)
    sw_hw6 = np.full(len(plugs), np.nan)
    ok = res.loc[plugs[in_hw6]]
    phi_hw6[in_hw6] = np.where(ok.FLAG == "", ok.PHI_S, np.nan)
    sw_hw6[in_hw6] = np.where(ok.FLAG == "", ok.SW, np.nan)
    sets = {"Core porosity": win.loc[plugs, "CORE_POR"].values,
            "Inversion porosity (P50)": inv.loc[plugs, "phi_p50"].values,
            "HW 6 porosity": phi_hw6}
    sws = {"Core Sw": win.loc[plugs, "CORE_SW"].values,
           "Inversion Sw (P50)": inv.loc[plugs, "sw_p50"].values,
           "HW 6 Sw": sw_hw6}
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for a in ax:
        _style(a)
    colors = ("#111111", "#087E8B", "#C62828")
    fits = {}
    for (lab, x), col in zip(sets.items(), colors):
        a0, b, r2, n = _fit(x, k)
        fits[lab] = (a0, b, r2, n)
        ax[0].scatter(x, k, s=16, color=col, alpha=0.7, label=f"{lab}: log k = {a0:.2f} + {b:.1f} phi, R² = {r2:.2f}")
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 20)
        ax[0].plot(xx, 10 ** (a0 + b * xx), color=col, lw=1)
    ax[0].set(yscale="log", xlabel="Porosity (v/v)", ylabel="Core permeability (md)",
              title="(a) Permeability vs. porosity at the core plugs")
    ax[0].legend(fontsize=7.5, loc="lower right")
    swfits = {}
    for (lab, x), col in zip(sws.items(), colors):
        ok = np.isfinite(x)
        r = np.corrcoef(x[ok], np.log10(k[ok]))[0, 1]
        swfits[lab] = (r, int(ok.sum()), float(np.nanmedian(x)))
        ax[1].scatter(x, k, s=16, color=col, alpha=0.7, label=f"{lab}: r(Sw, log k) = {r:.2f}")
    ax[1].set(yscale="log", xlim=(0, 1), xlabel="Water saturation (v/v)", ylabel="Core permeability (md)",
              title=f"(b) Permeability vs. Sw (plugs {height_ft[0]:.0f}-{height_ft[1]:.0f} ft TVD above the FWL)")
    ax[1].legend(fontsize=7.5)
    fig.suptitle("Core permeability vs. log-calculated porosity and Sw", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(fits=fits, swfits=swfits, n=len(plugs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/project1.toml")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = io.load_config(args.config)
    out = io.ensure_dir(cfg["data"]["output_dir"])
    win, raw, inv, hw6, hw6_ind = load(cfg)
    full, _ = io.load_well(cfg["data"]["las"], cfg["curves"])
    from . import petro
    ok = full.SDEV.notna() & full.HAZI.notna()
    md = full.index.values[ok]
    tvd = petro.tvd_minimum_curvature(md, full.SDEV[ok], full.HAZI[ok])
    pres = workflow.pressure_analysis(io.load_pressures(cfg["data"]["pressures"]), (md, tvd), cfg, 1.05)

    ds = depth_shift(win, out / "11_core_depth_shift.png")
    hc = hc_density(win, hw6_ind, cfg, pres["hc"]["density"], out / "12_hydrocarbon_density.png")
    sx = sw_sxo(win, inv, float(cfg["archie"]["n"]), cfg["zones"]["water"], out / "13_sw_sxo_consistency.png")
    plug_md = win.index[win.CORE_POR.notna()].values
    plug_tvd = np.interp(plug_md, md, tvd)
    height = (pres["fwl_tvd"] - plug_tvd.max(), pres["fwl_tvd"] - plug_tvd.min())
    pk = permeability(win, inv, hw6, height, out / "14_permeability.png")
    pk["height"] = height

    L = ["# Project 1: remaining checks", "",
         "## 1. Core-to-log depth shift (`11_core_depth_shift.png`)", "",
         f"The core data in the LAS were already shifted to log depth: log depth − original core depth = "
         f"{ds['shift_min']:.1f} to {ds['shift_max']:.1f} ft (median {ds['shift_med']:.1f} ft) over {ds['n']} samples, "
         "so the shift varies along the core.", "",
         "| Core gamma ray placed at | Correlation with log GR at 0 ft | Best extra shift | Best correlation |",
         "|---|---|---|---|",
         f"| Original core depths | {ds['r_orig_zero']:.2f} | {ds['best_orig']:+.2f} ft | {ds['r_orig']:.2f} |",
         f"| Shifted (LAS) depths | {ds['r_zero']:.2f} | {ds['best_shifted']:+.2f} ft | {ds['r_shifted']:.2f} |", "",
         "No further shift is needed if the best extra shift for the shifted core is within about ±1 ft "
         "(the gamma-ray tool's vertical resolution).", "",
         "## 2. Hydrocarbon density (`12_hydrocarbon_density.png`)", "",
         "HW 6 parts (h)–(i) at every pay sample: ρf from the shale-corrected density and the sandstone "
         "porosity, then ρH = (ρf − Sw ρw)/(1 − Sw). The oil-corrected porosity assumes ρH, so this check uses "
         "HW 6's own porosity rule (logs only) and, at the plugs, core porosity (core-anchored). Samples with "
         "Sw > 0.6, Csh > 0.6 or φs < 0.10 are left out (ρH blows up as Sw → 1).", "",
         "| Estimate | Median ρH (g/cc) | IQR | Samples |", "|---|---|---|---|",
         f"| Logs only | {hc['logs'][0]:.2f} | {hc['logs'][1]:.2f}–{hc['logs'][2]:.2f} | {hc['logs'][3]} |",
         f"| Core-anchored | {hc['core'][0]:.2f} | {hc['core'][1]:.2f}–{hc['core'][2]:.2f} | {hc['core'][3]} |",
         f"| Formation-pressure gradient | {hc['pressure']:.3f} | | |", "",
         f"Median pore-fluid density: {hc['rho_f_logs']:.2f} g/cc (logs only), {hc['rho_f_core']:.2f} g/cc (core-anchored).", "",
         "## 3. Sw–Sxo consistency (`13_sw_sxo_consistency.png`)", "",
         f"Two-step inversion P50 in the pay, {sx['n']} samples.", "",
         "- **Inversion Sxo is not usable here.** It treats AT10 as a resistivity parallel to bedding, but in a "
         "50° well through laminated, anisotropic rock the induction reads between Rh and Rv. AT10 then exceeds "
         f"the parallel-model limit, so Sxo is defined at only {sx['inv_defined']:.0%} of the pay, and where it is "
         f"defined it reads below Sw − 0.05 at {sx['inv_below']:.0%} of samples.",
         f"- **Ratio method instead:** Rxo/Rt = (Rmf/Rw)(Sw/Sxo)^n, with Rmf/Rw = AT10/AT90 in the water sand = "
         f"{sx['rmf_rw']:.2f}. Using AT10/AT90 cancels most of the deviation effect on both curves. A modest ratio "
         "in the water sand points to water-base mud with Rmf a little above Rw; oil-base mud would make AT10 "
         "much higher than AT90 there.",
         f"- Ratio-method Sxo is below Sw − 0.05 at **{sx['frac_below']:.0%}** of samples; median Sxo − Sw = "
         f"{sx['med_diff']:+.3f}; median moved-hydrocarbon volume φ(Sxo − Sw) = {sx['moved']:.3f}. With water-base mud "
         "Sxo ≥ Sw is expected, so this is consistent: the flushed zone holds about the same saturation as the "
         "deep zone, i.e. little hydrocarbon was moved by invasion.",
         f"- Bulk volume water φ·Sw: median {sx['bvw_med']:.3f}, coefficient of variation {sx['bvw_cv']:.2f}; "
         f"correlation of φ with Sw {sx['bvw_r_phi']:+.2f}. At irreducible saturation BVW is roughly constant "
         "(CV well below ~0.2) and φ and Sw are strongly inversely related; a CV above that and a moderate "
         "correlation mean the pay is only partly at irreducible water, consistent with its shaly, "
         "variable-permeability sands.", "",
         "## 4. Permeability vs. log porosity and Swirr (`14_permeability.png`)", "",
         f"{pk['n']} core plugs with permeability, {pk['height'][0]:.0f}–{pk['height'][1]:.0f} ft TVD above the "
         f"free-water level (~{pres['fwl_md']:.0f} ft MD).", "",
         "| Porosity used | log k = a + b φ | R² | Plugs |", "|---|---|---|---|"]
    for lab, (a0, b, r2, n) in pk["fits"].items():
        L.append(f"| {lab} | {a0:.2f} + {b:.1f} φ | {r2:.2f} | {n} |")
    L += ["", "| Sw used | r(Sw, log k) | Median Sw | Plugs |", "|---|---|---|---|"]
    for lab, (r, n, med) in pk["swfits"].items():
        L.append(f"| {lab} | {r:+.2f} | {med:.2f} | {n} |")
    (out / "project_checks.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
