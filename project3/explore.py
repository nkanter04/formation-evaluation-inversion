"""PGE 385K Homework Project 3: load, quality-check and visualize the logs, core and pressure data (well 16/28-A05).

Run:  python project3/explore.py
Writes figures and summary.md to the Project 3 "4 Python Figures and Results/Exploration" folder.
"""
import sys
import tomllib
import warnings
from pathlib import Path

import lasio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fe_inversion.charts import ls_to_ss, nd_lines  # noqa: E402

warnings.filterwarnings("ignore", module="lasio")

P3 = Path(r"C:\Users\nkant\OneDrive\The University of Texas at Austin\Fall 2026\PGE 385K - Adv. Formation Eval"
          r"\PGE 385K - Homework Problems\Homework Projects\Homework Project No. 3")
RAW = P3 / "2 Raw Data (Canvas)"
RESULTS = P3 / "4 Python Figures and Results"
OUT = RESULTS / "Exploration"
CFG = tomllib.loads((Path(__file__).resolve().parents[1] / "config" / "project3.toml").read_text(encoding="utf-8"))
TOP, BASE = CFG["interval"]["top"], CFG["interval"]["base"]
A, M, N = (CFG["archie"][k] for k in "amn")
RHO_MA, RHO_F = CFG["matrix"]["rho_ma"], CFG["matrix"]["rho_f"]
M2F = 3.28084


# ----------------------------------------------------------------------------------- loading
def load_logs():
    """Logs over the study interval, ascending depth, with quality flags and porosities.

    TNPH is the Schlumberger thermal neutron in limestone units; NPHI_SS is it converted with chart Por-5.
    """
    df = lasio.read(str(RAW / "Logs.las")).df().sort_index()
    w = df.loc[TOP - 50:BASE + 50].copy()
    q = CFG["quality"]
    w["WASHOUT"] = w.HCAL > q["bit"] + q["washout_in"]
    w["BAD_RHO"] = (w.HDRA.abs() > q["hdra_max"]) | (w.RHOZ < q["rho_min"]) | (w.RHOZ > q["rho_max"])
    w["PHID"] = (RHO_MA - w.RHOZ) / (RHO_MA - RHO_F)
    w["NPHI_SS"] = ls_to_ss(w.TNPH.values)
    w["TVD"] = md_to_tvd(w.index.values)
    return w


def load_rft():
    """Pressure-test points with depths in ft (MD and TVD as given in the file)."""
    r = lasio.read(str(RAW / "16_28-a05_rft.las")).df()
    r = r.assign(md=r.index * M2F, tvd=r.TVD * M2F).reset_index(drop=True)
    return r


def md_to_tvd(md_ft):
    """TVD from MD with a quadratic fitted to the pressure-test MD/TVD pairs (no deviation survey in the logs)."""
    r = load_rft() if "_RFT" not in globals() else _RFT
    c = np.polyfit(r.md, r.tvd, 2)
    return np.polyval(c, md_ft)


_RFT = load_rft()


def load_core():
    """Core plugs with depth in ft, porosity as a fraction, permeability in md."""
    c = lasio.read(str(RAW / "16_28-a05_core.las")).df()
    c = c.assign(depth=c.index * M2F, phi=c.CPOR / 100.0, kh=c.CKHL, kv=c.CKVL, rho_g=c.CDEN).reset_index(drop=True)
    return c


def load_core_gamma():
    g = lasio.read(str(RAW / "16_28-a05_core_gamma.las")).df()
    return g.assign(depth=g.index * M2F, core_depth=g.CDEP * M2F).reset_index(drop=True)


# ------------------------------------------------------------------------------- analysis
def depth_shift(w, gamma, core):
    """Correlation of core gamma with log GR, and of core porosity with density porosity, against an extra shift."""
    d = CFG["depth_shift"]
    lags = np.arange(-d["max_shift_ft"], d["max_shift_ft"] + 1e-9, d["step_ft"])
    ok = ~w.BAD_RHO
    r_gr, r_phi = [], []
    for lag in lags:
        gr = np.interp(gamma.depth + lag, w.index, w.GR)
        r_gr.append(np.corrcoef(gr, gamma.SGR)[0, 1])
        phid = np.interp(core.depth + lag, w.index[ok], w.PHID[ok])
        r_phi.append(np.corrcoef(phid, core.phi)[0, 1])
    return lags, np.array(r_gr), np.array(r_phi)


def fit_segments(tvd, p, n_seg=3, min_pts=4):
    """Piecewise-linear fit of pressure against TVD with n_seg segments, breakpoints between sorted points."""
    order = np.argsort(tvd)
    z, y = np.asarray(tvd)[order], np.asarray(p)[order]
    best = None
    idx = range(min_pts, len(z) - min_pts + 1)
    for i in idx:
        for j in idx:
            if j - i < min_pts or len(z) - j < min_pts:
                continue
            parts = [(0, i), (i, j), (j, len(z))]
            sse, fits = 0.0, []
            for a, b in parts:
                c = np.polyfit(z[a:b], y[a:b], 1)
                sse += float(((np.polyval(c, z[a:b]) - y[a:b]) ** 2).sum())
                fits.append((c, z[a], z[b - 1], b - a))
            if best is None or sse < best[0]:
                best = (sse, fits)
    return best[1]


def contacts(fits):
    """TVD where consecutive gradient lines cross."""
    out = []
    for (c1, *_), (c2, *_) in zip(fits[:-1], fits[1:]):
        out.append((c2[1] - c1[1]) / (c1[0] - c2[0]))
    return out


def tvd_to_md(tvd_ft):
    md = np.linspace(TOP - 200, BASE + 200, 4001)
    return np.interp(tvd_ft, md_to_tvd(md), md)


def sand_bodies(w, i_gr, rw_ref):
    """Sandstone bodies (GR index below the limit, at least min_ft thick) with a quick fluid call from Rwa."""
    z = CFG["zones"]
    sand = (i_gr < z["max_gr_index"]).values
    edges = np.flatnonzero(np.diff(np.r_[0, sand.astype(int), 0]))
    rows = []
    for a, b in zip(edges[::2], edges[1::2]):
        s = w.iloc[a:b]
        if s.index[-1] - s.index[0] < z["min_ft"]:
            continue
        phi = np.sqrt((s.PHID.clip(0) ** 2 + s.NPHI_SS.clip(0) ** 2) / 2)
        rwa = (s.AT90 * phi ** M / A).median()
        rows.append(dict(top=s.index[0], base=s.index[-1], thick=s.index[-1] - s.index[0] + 0.5,
                         tvd_top=s.TVD.iloc[0], gr=s.GR.median(), phid=s.PHID.median(), nphi_ss=s.NPHI_SS.median(),
                         rt=s.AT90.median(), rxo=s.HMNO.median(), rwa=rwa, rt_over_r0=rwa / rw_ref))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- figures
def fig_composite(w, core, gamma, zones, cont_md, path):
    fig, ax = plt.subplots(1, 7, figsize=(19, 15), sharey=True)
    z = w.index
    a = ax[0]
    a.plot(w.GR, z, "g", lw=0.7, label="GR (API)")
    a.plot(gamma.SGR, gamma.depth, color="0.4", lw=0.5, label="Core gamma, SGR (API)")
    a.plot((w.HCAL - CFG["quality"]["bit"]) * 20 + 75, z, "k", lw=0.6, label="Caliper − bit (×20 in, offset 75)")
    a.set(xlim=(0, 160), title="Gamma ray / caliper")
    a = ax[1]
    for k, col in [("AT90", "r"), ("AT30", "orange"), ("AT10", "m")]:
        a.semilogx(w[k].clip(0.1, 2000), z, color=col, lw=0.6, label=k)
    # The micro-resistivity is spiky; a running median shows its level without hiding the other curves.
    a.semilogx(w.HMNO.rolling(9, center=True, min_periods=1).median().clip(0.1, 2000), z, color="c", lw=0.8,
               label="HMNO, 4 ft running median")
    a.set(xlim=(0.2, 300), title="Resistivity (Ω·m)")
    a = ax[2]
    a.plot(w.RHOZ, z, "r", lw=0.7, label="RHOZ (g/cc)")
    a.set(xlim=(1.95, 2.95), title="Density (g/cc)")
    a2 = a.twiny(); a2.plot(w.TNPH, z, "b", lw=0.7, label="TNPH (LS units)"); a2.set(xlim=(0.45, -0.15))
    a2.legend(fontsize=6.5, loc="lower left")
    a = ax[3]
    a.plot(w.PHID, z, "r", lw=0.6, label="Density porosity φD (2.65)")
    a.plot(w.NPHI_SS, z, "b", lw=0.6, label="Neutron, sandstone units (Por-5)")
    a.plot(core.phi, core.depth, "ko", ms=2, mfc="none", label="Core porosity (He)")
    a.set(xlim=(0.4, 0), title="Porosity (v/v)")
    a = ax[4]
    a.plot(w.PEFZ, z, "m", lw=0.6, label="PEFZ (b/e)"); a.set(xlim=(0, 10), title="PE (b/e)")
    a = ax[5]
    a.semilogx(core.kh, core.depth, "o", ms=2.5, color="#1f77b4", label="Core kh, air (md)")
    a.semilogx(core.kv, core.depth, "^", ms=2.5, color="#d62728", label="Core kv, air (md)")
    a.set(xlim=(0.01, 5000), title="Core permeability (md)")
    a = ax[6]
    a.plot(w.TVD, z, "k", lw=1, label="TVD (ft)")
    a.set(title="TVD (ft)")
    for lab, (zz, col) in cont_md.items():
        for x in ax:
            x.axhline(zz, color=col, ls="--", lw=1)
        ax[6].text(a.get_xlim()[0], zz, f" {lab}", color=col, fontsize=8, va="bottom")
    for _, s in zones.iterrows():
        col = "#c7e9c0" if s.fluid != "water" else "#c6dbef"
        for x in ax[:5]:
            x.axhspan(s.top, s.base, color=col, alpha=0.6, lw=0)
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=6.5, loc="lower right")
    ax[0].set(ylim=(BASE, TOP), ylabel="Measured depth (ft)")
    fig.suptitle("16/28-A05: logs, core and contacts from the pressure tests, 8,150–9,000 ft MD "
                 "(green: possible hydrocarbon sandstones; blue: water sandstones)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_depth_shift(lags, r_gr, r_phi, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lags, r_gr, lw=2, label="Core gamma (SGR) vs. log GR")
    ax.plot(lags, r_phi, lw=1.5, label="Core porosity vs. density porosity")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Additional shift added to the core depth (ft; positive = deeper)", ylabel="Correlation coefficient r",
           title="Core-to-log depth check (core depths converted from metres)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_pressure(r, fits, cont, path):
    p = CFG["pressure"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    used = r.MOBILITY >= p["min_mobility"]
    a = ax[0]
    a.plot(r.ORIGINAL[used], r.tvd[used], "ko", label="Original pressure (depletion-corrected)")
    a.plot(r.ORIGINAL[~used], r.tvd[~used], "o", mfc="none", color="0.5", label=f"Mobility < {p['min_mobility']:g} md/cp (not fitted)")
    a.plot(r.PRESSURE, r.tvd, "x", color="#d62728", ms=5, label="Measured (depleted) pressure")
    names = ["Gas", "Oil", "Water"]
    cols = ["#d62728", "#2ca02c", "#1f77b4"]
    zz = np.linspace(r.tvd.min() - 50, r.tvd.max() + 50, 50)
    for (c, z0, z1, n), nm, col in zip(fits, names, cols):
        a.plot(np.polyval(c, zz), zz, color=col, lw=1, ls=":")
        a.plot(np.polyval(c, [z0, z1]), [z0, z1], color=col, lw=2.5,
               label=f"{nm}: {c[0]:.3f} psi/ft = {c[0] / p['psi_per_ft_per_gcc']:.2f} g/cc ({n} points)")
    for lab, zc in zip(("Gas–oil contact", "Free-water level (oil–water)"), cont):
        a.axhline(zc, color="k", ls="--", lw=0.8)
        a.text(a.get_xlim()[0] if False else r.ORIGINAL.min(), zc, f" {lab}: {zc:,.0f} ft TVD", fontsize=8, va="bottom")
    a.set(xlabel="Pressure (psia)", ylabel="TVD (ft)", title="(a) Pressure tests and fluid gradients")
    a.invert_yaxis(); a.grid(alpha=0.3); a.legend(fontsize=7, loc="lower left")
    a = ax[1]
    a.semilogx(r.MOBILITY, r.tvd, "ko")
    a.axvline(p["min_mobility"], color="0.5", ls=":")
    a.set(xlabel="Mobility (md/cp)", title="(b) Test mobility")
    a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_crossplots(w, zones_mask, rw_ref, path):
    good = w[~w.WASHOUT & ~w.BAD_RHO].loc[TOP:BASE]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    a = ax[0, 0]
    s = a.scatter(good.TNPH, good.RHOZ, c=good.GR, s=4, cmap="viridis")
    nd_lines(a, units="limestone")
    a.set(xlim=(-0.05, 0.5), ylim=(2.9, 1.9), xlabel="TNPH (v/v, limestone units)", ylabel="RHOZ (g/cc)",
          title="(a) Neutron–density (lines as chart Por-11, fresh water)")
    a.legend(fontsize=7); plt.colorbar(s, ax=a, label="GR (API)")
    a = ax[0, 1]
    s = a.scatter(good.GR, good.AT90, c=good.TVD, s=4, cmap="plasma")
    a.set(yscale="log", xlabel="GR (API)", ylabel="AT90 (Ω·m)", title="(b) Deep resistivity vs. GR")
    plt.colorbar(s, ax=a, label="TVD (ft)")
    a = ax[0, 2]
    s = a.scatter(good.GR, good.RHOZ, c=good.TNPH, s=4, cmap="viridis")
    a.set(ylim=(2.9, 1.9), xlabel="GR (API)", ylabel="RHOZ (g/cc)", title="(c) GR vs. bulk density")
    plt.colorbar(s, ax=a, label="TNPH (v/v)")
    a = ax[1, 0]
    s = a.scatter(good.RHOZ, good.PEFZ, c=good.GR, s=4, cmap="viridis")
    a.set(xlim=(2.0, 2.9), ylim=(1, 8), xlabel="RHOZ (g/cc)", ylabel="PEFZ (b/e)", title="(d) PE vs. bulk density")
    plt.colorbar(s, ax=a, label="GR (API)")
    a = ax[1, 1]
    phi = np.sqrt((good.PHID.clip(0.01) ** 2 + good.NPHI_SS.clip(0.01) ** 2) / 2)
    s = a.scatter(phi, good.AT90, c=good.GR, s=4, cmap="coolwarm")
    pp = np.logspace(-1.3, -0.4, 20)
    for sw in (1.0, 0.5, 0.25, 0.1):
        a.plot(pp, A * rw_ref / (pp ** M * sw ** N), "k:", lw=0.8)
        a.text(pp[-1], A * rw_ref / (pp[-1] ** M * sw ** N), f" Sw = {sw}", fontsize=7)
    a.set(xscale="log", yscale="log", xlabel="Neutron–density porosity (v/v)", ylabel="AT90 (Ω·m)",
          title=f"(e) Pickett plot, m = {M}, n = {N}, Rw = {rw_ref:.3f} Ω·m (formation temperature)")
    plt.colorbar(s, ax=a, label="GR (API)")
    a = ax[1, 2]
    s = a.scatter(good.HMNO.clip(0.05), good.AT90, c=good.TVD, s=4, cmap="plasma")
    a.plot([0.1, 500], [0.1, 500], "k--", lw=0.8)
    a.set(xscale="log", yscale="log", xlabel="HMNO, micro-normal (Ω·m)", ylabel="AT90 (Ω·m)",
          title="(f) Flushed-zone vs. deep resistivity")
    plt.colorbar(s, ax=a, label="TVD (ft)")
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = load_logs()
    wi = w.loc[TOP:BASE]
    core, gamma, r = load_core(), load_core_gamma(), _RFT
    gr_cl, gr_sh = np.nanpercentile(wi.GR, [3, 97])
    i_gr = ((wi.GR - gr_cl) / (gr_sh - gr_cl)).clip(0, 1)

    # MD-TVD from the pressure-test pairs
    tvd_res = r.tvd - md_to_tvd(r.md)
    inc = np.degrees(np.arccos(np.diff(r.tvd) / np.diff(r.md)))

    # Pressure gradients and contacts (depletion-corrected pressures, mobile points)
    p = CFG["pressure"]
    used = r.MOBILITY >= p["min_mobility"]
    fits = fit_segments(r.tvd[used].values, r[p["column"]][used].values, 3, p["min_points"])
    cont = contacts(fits)
    cont_md = {"GOC": (float(tvd_to_md(cont[0])), "#d62728"), "FWL": (float(tvd_to_md(cont[1])), "#1f77b4")}
    fits_dep = fit_segments(r.tvd[used].values, r.PRESSURE[used].values, 3, p["min_points"])

    # Rw from the cleanest water-bearing sandstone below the free-water level
    below = wi[(wi.index > cont_md["FWL"][0] + 10) & ~wi.WASHOUT & ~wi.BAD_RHO & (i_gr.reindex(wi.index) < 0.1)]
    phi_b = np.sqrt((below.PHID ** 2 + below.NPHI_SS ** 2) / 2)
    rwa_b = below.AT90 * phi_b ** M / A
    rw_ref = float(np.nanpercentile(rwa_b, 10)) if len(below) else 0.05

    zones = sand_bodies(wi, i_gr, rw_ref)
    zones["fluid"] = np.where(zones.rt_over_r0 > 3, "possible hydrocarbon", np.where(zones.rt_over_r0 < 1.6, "water", "transition/shaly"))
    zones["above"] = np.where(zones.top < cont_md["GOC"][0], "gas cap", np.where(zones.top < cont_md["FWL"][0], "oil leg", "below FWL"))

    lags, r_gr, r_phi = depth_shift(wi, gamma, core)
    best_gr, best_phi = lags[np.argmax(r_gr)], lags[np.argmax(r_phi)]

    fig_composite(wi, core, gamma, zones, cont_md, OUT / "01_composite.png")
    fig_depth_shift(lags, r_gr, r_phi, OUT / "02_depth_shift.png")
    fig_pressure(r, fits, cont, OUT / "03_pressure_gradients.png")
    fig_crossplots(w, None, rw_ref, OUT / "04_crossplots.png")

    q = CFG["quality"]
    clean = wi[i_gr < 0.1]
    L = ["# Project 3: first look at the data (well 16/28-A05)", "",
         f"Interval {TOP:.0f}–{BASE:.0f} ft MD, 0.5 ft sampling, {len(wi)} samples. Generated by `project3/explore.py`.", "",
         "## Data", "",
         "- Logs (Schlumberger, ft MD): GR, ECGR, caliper, AIT resistivities (A10–A90, 1 ft and 2 ft), MCFL micro-resistivity, "
         "density (RHOZ, HDRA), neutron (TNPH, NPHI; limestone units), PE. No SP, sonic or spectral GR.",
         f"- Core (BP, depths in m): {len(core)} plugs, cores {int(core.CNUM.min())}–{int(core.CNUM.max())}, "
         f"{core.depth.min():,.0f}–{core.depth.max():,.0f} ft; helium porosity, air kh ({core.kh.notna().sum()}) and kv "
         f"({core.kv.notna().sum()}), grain density (median {core.rho_g.median():.3f} g/cc).",
         f"- Core gamma: {len(gamma)} points (SGR, CGR, K, Th, U), log-depth and core-depth columns.",
         f"- Pressure tests: {len(r)} points, {r.md.min():,.0f}–{r.md.max():,.0f} ft MD, "
         f"{r.tvd.min():,.0f}–{r.tvd.max():,.0f} ft TVD; measured (depleted) and original (depletion-corrected) pressure, mobility.",
         "- The core LAS header stores BP's interpretation parameters (for comparison only, not used): field Andrew; "
         "GOC 2,503 m and OWC 2,554 m; Rw 0.12 Ω·m; measured temperature 240 °F; m 1.89, n 1.92; GR clean/shale 30/105 API.", "",
         "## Well geometry", "",
         f"No deviation survey in the logs. TVD is a quadratic in MD fitted to the {len(r)} pressure-test MD/TVD pairs "
         f"(residual RMS {np.sqrt((tvd_res ** 2).mean()):.2f} ft). Inclination between test points {np.median(inc):.1f}° "
         f"(range {inc.min():.1f}–{inc.max():.1f}°). Interval in TVD: {md_to_tvd(TOP):,.0f}–{md_to_tvd(BASE):,.0f} ft.", "",
         "## Data quality", "",
         f"- Caliper more than {q['washout_in']:g} in over the {q['bit']:g} in bit: {int(wi.WASHOUT.sum())} samples "
         f"({wi.WASHOUT.mean() * 100:.0f} %).",
         f"- Density flagged (|HDRA| > {q['hdra_max']:g} g/cc or RHOZ outside {q['rho_min']}–{q['rho_max']} g/cc): "
         f"{int(wi.BAD_RHO.sum())} samples.",
         f"- GR end-points 3rd/97th percentile: {gr_cl:.0f} / {gr_sh:.0f} API.", "",
         "## Core-to-log depth", "",
         f"Core and core-gamma depths converted from metres. Core gamma vs. log GR peaks at {best_gr:+.1f} ft "
         f"(r = {r_gr.max():.2f}; {r_gr[lags == 0][0]:.2f} at 0 ft). Core porosity vs. density porosity peaks at "
         f"{best_phi:+.1f} ft (r = {r_phi.max():.2f}; {r_phi[lags == 0][0]:.2f} at 0 ft). See 02_depth_shift.png.", "",
         "## Pressure gradients and contacts (03_pressure_gradients.png)", "",
         f"Fitted to the original (depletion-corrected) pressures at {int(used.sum())} points with mobility ≥ "
         f"{p['min_mobility']:g} md/cp, three segments:", "",
         "| Fluid | Gradient (psi/ft) | Density (g/cc) | Points | TVD range (ft) |", "|---|---|---|---|---|",
         *[f"| {nm} | {c[0]:.3f} | {c[0] / p['psi_per_ft_per_gcc']:.2f} | {n} | {z0:,.0f}–{z1:,.0f} |"
           for (c, z0, z1, n), nm in zip(fits, ("Gas", "Oil", "Water"))], "",
         f"- Gas–oil contact: {cont[0]:,.0f} ft TVD ({cont[0] / M2F:,.0f} m), {cont_md['GOC'][0]:,.0f} ft MD.",
         f"- Free-water level (oil–water): {cont[1]:,.0f} ft TVD ({cont[1] / M2F:,.0f} m), {cont_md['FWL'][0]:,.0f} ft MD.",
         f"- Same fit to the measured (depleted) pressures: gradients "
         + ", ".join(f"{c[0]:.3f}" for c, *_ in fits_dep) + " psi/ft; contacts "
         + ", ".join(f"{z:,.0f}" for z in contacts(fits_dep)) + " ft TVD. Depletion shifts the pressures but the "
         "contacts should come from the original pressures.",
         "- Compare BP's GOC 2,503 m and OWC 2,554 m (datum not stated in the file).", "",
         "## Lithology", "",
         f"Clean samples (GR index < 0.1, {len(clean)}): RHOZ {clean.RHOZ.median():.3f} g/cc, TNPH {clean.TNPH.median():.3f} (LS), "
         f"PE {clean.PEFZ.median():.2f} b/e. On the neutron–density cross-plot (04_crossplots.png a) the clean samples sit "
         "near the sandstone line; PE about 2–3 b/e is quartz with some heavier minerals.", "",
         "## Rw (first estimate)", "",
         f"Clean samples (GR index < 0.1) more than 10 ft below the free-water level: {len(below)}; Rwa = Rt·φ^m, 10th "
         f"percentile {rw_ref:.3f} Ω·m, median {np.nanmedian(rwa_b):.3f} Ω·m (formation temperature). BP's stored Rw: 0.12 Ω·m.", "",
         "## Sandstone bodies (GR index < 0.4, at least 4 ft)", "",
         zones.to_markdown(index=False, floatfmt=".2f"), "",
         "## Open questions", "",
         "- Formation temperature: the logs carry only cartridge temperature (CTEM); BP's file stores 240 °F (source unknown).",
         "- Mud type and Rmf are not in the data.",
         "- MICP sheet (PC.xlsx): the brine and oil gradient labels look reversed (0.921 vs. 1.477 psi/m)."]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
