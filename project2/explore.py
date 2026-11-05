"""PGE 385K Homework Project 2: load, quality-check and visualize the logs and core data.

Run:  python project2/explore.py
Writes figures and summary.md to the Project 2 "Exploration" folder.
"""
from pathlib import Path

import lasio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P2 = Path(r"C:\Users\nkant\OneDrive\The University of Texas at Austin\Fall 2026\PGE 385K - Adv. Formation Eval"
          r"\PGE 385K - Homework Problems\Homework Projects\Homework Project No. 2")
LAS = P2 / "Data" / "HA_2_reader_compatible.las"
CORE = P2 / "Data" / "HA2_core_data.xls"
OUT = P2 / "Exploration"

TOP, BASE = 5400.0, 6300.0
BIT = 8.375                      # in, log header
RHO_MA, RHO_F = 2.65, 1.0        # quartz; mud-filtrate density for density porosity
A, M, N = 1.0, 2.1, 1.95         # Archie parameters given in the assignment
RMF_75, RM_75 = 1.32, 2.52       # ohm-m at 75 degF, log header
BHT, BHT_DEPTH = 163.0, 6900.0   # degF at logger depth
T_SURF = 80.0                    # degF, ASSUMED mean surface temperature (not on the header)
LARGE_ZCOR = 0.10                # g/cc; ZDEN is already corrected, this only marks large corrections
NPHI_SS_TO_LS = -0.04            # CNCF sandstone -> limestone units at zero porosity (quartz reads about
                                 # -0.04), tapering with porosity as on the Schlumberger conversion chart


# ----------------------------------------------------------------------------------- loading
def ss_to_ls(nphi_ss):
    """Sandstone- to limestone-unit neutron porosity; the same offset as the sandstone line in nd_lines."""
    return nphi_ss + NPHI_SS_TO_LS * (1 - nphi_ss / 0.45 * 0.6)


def load_logs():
    df = lasio.read(LAS).df()
    df = df[~df.index.duplicated()]
    df["NPHI"] = df.CNCF / 100.0
    df["PHID"] = (RHO_MA - df.ZDEN) / (RHO_MA - RHO_F)
    df["TEMP"] = T_SURF + (BHT - T_SURF) * df.index / BHT_DEPTH
    df["NPHI_LS"] = ss_to_ls(df.NPHI)
    df["LARGE_ZCOR"] = df.ZCOR.abs() > LARGE_ZCOR
    df["WASHOUT"] = df.CAL > BIT + 1.0
    df["BAD_RT"] = df.RTBR < 0
    return df


def load_core():
    raw = pd.read_excel(CORE, sheet_name="conventional_core_analysis", header=None).iloc[6:]
    raw = raw[raw[0].notna()]
    cols = {0: "sample", 1: "core", 2: "depth_core", 3: "depth_log", 4: "sand", 5: "kh_800", 7: "kv_800",
            9: "phi_800", 10: "kh_3500", 12: "kv_3500", 14: "phi_3500", 15: "so", 16: "sw", 17: "rho_grain"}
    raw = raw.rename(columns=cols)[list(cols.values())]
    vert = raw["sample"].astype(str).str.endswith("V")
    core = raw[~vert].copy()
    kv = raw[vert].set_index(raw[vert]["sample"].astype(str).str[:-1])
    core.index = core["sample"].astype(str)
    core["kv_800"], core["kv_3500"] = kv["kv_800"], kv["kv_3500"]
    for c in ["kh_800", "kv_800", "kh_3500", "kv_3500"]:
        txt = core[c].astype(str)
        core[c + "_censored"] = txt.str.startswith(">")
        core[c] = pd.to_numeric(txt.str.lstrip(">"), errors="coerce")
    for c in ["depth_core", "depth_log", "phi_800", "phi_3500", "so", "sw", "rho_grain"]:
        core[c] = pd.to_numeric(core[c], errors="coerce")
    for c in ["phi_800", "phi_3500", "so", "sw"]:
        core[c] /= 100.0
    core["shift"] = core.depth_log - core.depth_core
    return core.reset_index(drop=True)


# ------------------------------------------------------------------------------- analysis
def gr_end_points(w):
    return np.percentile(w.GR.dropna(), 3), np.percentile(w.GR.dropna(), 97)


def pick_sands(w, vsh_cut=0.4, min_ft=4.0):
    """Contiguous intervals with GR-index Vsh below the cutoff and at least min_ft thick."""
    gr_cl, gr_sh = gr_end_points(w)
    vsh = ((w.GR - gr_cl) / (gr_sh - gr_cl)).clip(0, 1)
    sand = (vsh < vsh_cut).values
    edges = np.flatnonzero(np.diff(np.r_[0, sand.astype(int), 0]))
    rows = []
    for a, b in zip(edges[::2], edges[1::2]):
        z = w.iloc[a:b]
        if z.index[-1] - z.index[0] + 0.5 < min_ft:
            continue
        good = z
        phi = good.PHID.median()
        rt = z.RIPD.median()
        rwa = rt * phi ** M / A
        rows.append(dict(top=z.index[0], base=z.index[-1], thick=z.index[-1] - z.index[0] + 0.5,
                         gr=z.GR.median(), vsh=vsh.iloc[a:b].median(), sp=z.SP.median(), rt=rt,
                         rxo=z.RSFL.median(), phid=phi, nphi=z.NPHI.median(), rwa=rwa))
    return pd.DataFrame(rows), vsh, (gr_cl, gr_sh)


def classify(sands):
    """Water vs. possible hydrocarbon from the apparent-Rw ratio (Rt / R0 with Rw = lowest Rwa)."""
    rw = sands.loc[sands.thick >= 20, "rwa"].min()
    sands["rt_over_r0"] = sands.rwa / rw
    sands["sw_est"] = sands.rt_over_r0 ** (-1 / N)
    sands["fluid"] = np.where(sands.rt_over_r0 > 3.0, "possible hydrocarbon",
                              np.where(sands.rt_over_r0 < 1.6, "water", "transition / shaly"))
    return sands, rw


def rw_from_sp(ssp, temp_f):
    """Rw from the static SP (Bateman & Konen form of Schlumberger chart SP-1/SP-2)."""
    rmf = RMF_75 * (75 + 6.77) / (temp_f + 6.77)
    rmfe = 0.85 * rmf if rmf > 0.1 else (146 * rmf - 5) / (337 * rmf + 77)
    k = 61 + 0.133 * temp_f
    rwe = rmfe * 10 ** (ssp / k)
    rw = (rwe + 0.131 * 10 ** (1 / np.log10(temp_f / 19.9) - 2)) / (-0.5 * rwe + 10 ** (0.0426 / np.log10(temp_f / 50.8)))
    return dict(rmf=rmf, rmfe=rmfe, k=k, rwe=rwe, rw=rw)


def depth_shift_scan(df, core, lags=np.arange(-8, 8.5, 0.5)):
    """Correlation of core porosity with density porosity for extra shifts added to the given log depth."""
    out = {}
    groups = [(f"core {k}", core.index[core.core == k]) for k in sorted(core.core.unique()) if (core.core == k).sum() >= 4]
    for name, sel in [("all cores", core.index)] + groups:
        c = core.loc[sel]
        r = []
        for lag in lags:
            phid = np.interp(c.depth_log + lag, df.index, df.PHID)
            ok = np.isfinite(phid) & c.phi_3500.notna()
            r.append(np.corrcoef(phid[ok], c.phi_3500[ok])[0, 1] if ok.sum() > 3 else np.nan)
        out[name] = np.array(r)
    return lags, out


# ------------------------------------------------------------------------------- figures
def shade_sands(ax, sands):
    for _, s in sands.iterrows():
        col = {"water": "#9ecae1", "possible hydrocarbon": "#a1d99b"}.get(s.fluid, "#fdd0a2")
        ax.axhspan(s.top, s.base, color=col, alpha=0.35, lw=0)


def fig_composite(df, core, sands, path):
    w = df.loc[TOP - 20:BASE + 20]
    fig, ax = plt.subplots(1, 7, figsize=(17, 15), sharey=True)
    ax[0].plot(w.GR, w.index, "g", lw=0.7, label="GR (API)")
    ax[0].plot((w.CAL - BIT) * 20 + 60, w.index, "k", lw=0.5, label="CAL − bit (×20 in, offset 60)")
    ax[0].set(xlim=(0, 150), title="Gamma ray")
    ax[1].plot(w.SP, w.index, "b", lw=0.7); ax[1].set(title="SP (mV)")
    for k, col in [("RIPD", "r"), ("RIPM", "m"), ("RSFL", "orange")]:
        ax[2].semilogx(w[k].clip(0.2, 2000), w.index, color=col, lw=0.7, label=k)
    ax[2].set(xlim=(0.3, 300), title="Resistivity (Ω·m)")
    ax[3].plot(w.PHID, w.index, "r", lw=0.7, label="φD (2.65 / 1.0)")
    ax[3].plot(w.NPHI, w.index, "b", lw=0.7, label="CNCF (v/v, sandstone units)")
    ax[3].fill_betweenx(w.index, w.PHID, w.NPHI, where=w.PHID > w.NPHI, color="gold", alpha=0.6, label="φD > φN")
    ax[3].plot(core.phi_3500, core.depth_log, "ko", ms=4, mfc="none", label="Core φ (3,500 psi)")
    ax[3].set(xlim=(0.6, -0.05), title="Porosity (v/v)")
    ax[4].plot(w.DT24, w.index, "k", lw=0.7); ax[4].set(title="DT24 (µs/ft)")
    ax[5].plot(w.PE, w.index, "m", lw=0.7, label="PE (b/e)")
    ax[5].plot(w.ZCOR * 10 + 4, w.index, "c", lw=0.6, label="ZCOR ×10 + 4 (g/cc)")
    big = w[w.LARGE_ZCOR]; ax[5].plot(np.full(len(big), 5.8), big.index, "r|", ms=4, label="|ZCOR| > 0.1 (applied)")
    ax[5].set(xlim=(0, 6), title="PE and density correction")
    k = core.kh_3500.where(~core.kh_3500_censored)
    ax[6].semilogx(k, core.depth_log, "ko", ms=4, label="kh (3,500 psi)")
    cens = core[core.kh_3500_censored]
    ax[6].semilogx(np.full(len(cens), 9999), cens.depth_log, "r>", ms=6, label="> 9,999 md")
    ax[6].set(xlim=(0.01, 3e4), title="Core permeability (md)")
    for a in ax:
        shade_sands(a, sands)
        a.axhline(TOP, color="k", ls="--", lw=0.8); a.axhline(BASE, color="k", ls="--", lw=0.8)
        a.grid(alpha=0.3); a.legend(fontsize=6.5, loc="lower right")
    ax[0].set_ylim(BASE + 20, TOP - 20); ax[0].set_ylabel("Measured depth (ft)")
    fig.suptitle("ORM-75: logs and core, 5,400–6,300 ft (blue: water sands, green: possible hydrocarbon, "
                 "orange: transition/shaly)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_shift(lags, scan, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in scan.items():
        ax.plot(lags, r, lw=2 if name == "all cores" else 1, label=name)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Additional shift added to the given log depth (ft)", ylabel="r (core φ vs. density φD)",
           title="Core-to-log depth check")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def nd_lines(ax, units="sandstone"):
    """Approximate neutron response of sandstone, limestone and dolomite (Schlumberger CP-1) in the given units."""
    phi = np.linspace(0, 0.45, 20)
    offsets = {"sandstone": (0.0, 0.04, 0.09), "limestone": (-0.04, 0.0, 0.05)}[units]
    for (name, rho, col), off in zip([("Sandstone", 2.65, "goldenrod"), ("Limestone", 2.71, "steelblue"),
                                      ("Dolomite", 2.87, "purple")], offsets):
        ax.plot(phi + off * (1 - phi / 0.45 * 0.6), rho - phi * (rho - 1.0), color=col, lw=1.2, label=name)


def fig_crossplots(df, sands, vsh, path):
    w = df.loc[TOP:BASE]
    good = w[~w.WASHOUT]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    a = ax[0, 0]
    s = a.scatter(good.NPHI, good.ZDEN, c=good.GR, s=4, cmap="viridis")
    nd_lines(a); a.text(0.02, 2.75, "Lithology lines approximate (sandstone-unit neutron)", fontsize=7); a.set(xlim=(-0.05, 0.6), ylim=(2.8, 1.8), xlabel="CNCF (v/v, sandstone units)",
                       ylabel="ZDEN (g/cc)", title="(a) Neutron–density, colored by GR")
    a.legend(fontsize=7); plt.colorbar(s, ax=a, label="GR (API)")
    a = ax[0, 1]
    s = a.scatter(w.GR, w.RIPD, c=w.index, s=4, cmap="viridis_r")
    a.set(yscale="log", xlabel="GR (API)", ylabel="RIPD (Ω·m)", title="(b) Deep resistivity vs. GR")
    plt.colorbar(s, ax=a, label="Depth (ft)")
    a = ax[0, 2]
    s = a.scatter(good.GR, good.ZDEN, c=good.RIPD, s=4, cmap="plasma", norm=matplotlib.colors.LogNorm())
    a.set(xlabel="GR (API)", ylabel="ZDEN (g/cc)", ylim=(2.7, 1.8), title="(c) GR vs. bulk density")
    plt.colorbar(s, ax=a, label="RIPD (Ω·m)")
    a = ax[1, 0]
    sp = w[(w.K > 0) & (w.TH > 0)]
    s = a.scatter(sp.K, sp.TH, c=sp.GR, s=4, cmap="viridis")
    for ratio in (2, 3.5, 12, 25):
        kk = np.linspace(0, 3, 10); a.plot(kk, ratio * kk, "k:", lw=0.7)
        xk = min(2.75, 23 / ratio); a.text(xk, ratio * xk, f" Th/K = {ratio}", fontsize=7, va="bottom")
    a.set(xlim=(0, 3), ylim=(0, 25), xlabel="K (%)", ylabel="Th (ppm)", title="(d) Thorium vs. potassium")
    plt.colorbar(s, ax=a, label="GR (API)")
    a = ax[1, 1]
    wet = good[(vsh.reindex(good.index) < 0.3) & (good.RIPD < 3)]
    s = a.scatter(good.PHID.clip(0.01), good.RIPD, c=vsh.reindex(good.index), s=4, cmap="coolwarm", vmin=0, vmax=1)
    phi = np.logspace(np.log10(0.05), np.log10(0.45), 20)
    for rw, sw in [(RW_EST, 1.0), (RW_EST, 0.5), (RW_EST, 0.25)]:
        a.plot(phi, A * rw / (phi ** M * sw ** N), "k--" if sw == 1 else "k:", lw=1)
        a.text(0.42, A * rw / (0.42 ** M * sw ** N), f"Sw = {sw}", fontsize=7, ha="right", va="bottom")
    a.set(xscale="log", yscale="log", xlim=(0.05, 0.5), ylim=(0.3, 200), xlabel="φD (v/v)", ylabel="RIPD (Ω·m)",
          title=f"(e) Pickett plot, m = {M}, n = {N}, Rw = {RW_EST:.3f} Ω·m")
    plt.colorbar(s, ax=a, label="Vsh (GR)")
    a = ax[1, 2]
    a.scatter(good.SP, good.GR, c=good.index, s=4, cmap="viridis_r")
    a.set(xlabel="SP (mV)", ylabel="GR (API)", title="(f) SP vs. GR")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_thomas_stieber(df, vsh, aquifer, phi_clean, phi_sh, path):
    w = df.loc[TOP:BASE]
    good = w[~w.WASHOUT]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, sub, title in [(ax[0], good, "(a) Whole interval"), (ax[1], good.loc[aquifer[0]:aquifer[1]],
                          f"(b) Water sand {aquifer[0]:.0f}–{aquifer[1]:.0f} ft")]:
        s = a.scatter(vsh.reindex(sub.index), sub.PHID, c=sub.index, s=6, cmap="viridis_r")
        # laminar: straight line from clean sand to shale
        a.plot([0, 1], [phi_clean, phi_sh], "g-", lw=1.5, label="Laminar")
        # dispersed: shale fills sand pores, phiT = phi_clean - Vsh (1 - phi_sh), up to Vsh = phi_clean
        a.plot([0, phi_clean], [phi_clean, phi_clean * phi_sh], "r-", lw=1.5, label="Dispersed (ends at Vsh = φclean)")
        # structural: shale replaces sand grains, phiT = phi_clean + Vsh phi_sh, up to Vsh = 1 - phi_clean
        a.plot([0, 1 - phi_clean], [phi_clean, phi_clean + (1 - phi_clean) * phi_sh], "b-", lw=1.0, label="Structural")
        a.plot([phi_clean, 1], [phi_clean * phi_sh, phi_sh], "k:", lw=1)
        a.plot([1 - phi_clean, 1], [phi_clean + (1 - phi_clean) * phi_sh, phi_sh], "k:", lw=1)
        a.set(xlim=(0, 1), ylim=(0, 0.5), xlabel="Vsh from GR (v/v)", ylabel="Total porosity φD (v/v)", title=title)
        a.grid(alpha=0.3); a.legend(fontsize=8)
        plt.colorbar(s, ax=a, label="Depth (ft)")
    fig.suptitle(f"Thomas–Stieber diagram (φclean = {phi_clean:.3f}, φsh = {phi_sh:.3f}); Vsh is the linear GR index",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def sandstone_nphi_ls(rhob):
    """Limestone-unit neutron of water-filled quartz sandstone at bulk density rhob (the nd_lines sandstone line)."""
    return ss_to_ls((RHO_MA - rhob) / (RHO_MA - RHO_F))


def fig_nd_aquifer(clean, aquifer, path):
    """Neutron-density cross-plot of the clean water sand with CNCF converted to limestone units."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    s = ax.scatter(clean.NPHI_LS, clean.ZDEN, c=clean.GR, s=12, cmap="viridis")
    nd_lines(ax, units="limestone")
    ax.plot(clean.NPHI_LS.median(), clean.ZDEN.median(), "r*", ms=16, label="Median of clean sand")
    ax.text(0.01, 2.77, f"CNCF converted to limestone units ({NPHI_SS_TO_LS:+.2f} at zero porosity, tapering; chart approximation);\n"
            "lithology lines approximate (Schlumberger CP-1)", fontsize=7)
    ax.set(xlim=(-0.05, 0.45), ylim=(2.8, 1.9), xlabel="Neutron porosity, limestone units (v/v)",
           ylabel="Bulk density ZDEN (g/cc)",
           title=f"Neutron–density in the clean water sand, {aquifer[0]:.0f}–{aquifer[1]:.0f} ft (Vsh < 0.1)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(s, ax=ax, label="GR (API)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------- main
RW_EST = None


def main():
    global RW_EST
    OUT.mkdir(exist_ok=True)
    df, core = load_logs(), load_core()
    w = df.loc[TOP:BASE]
    sands, vsh, (gr_cl, gr_sh) = pick_sands(w)
    sands, rw_rwa = classify(sands)
    RW_EST = rw_rwa

    # neutron units: cleanest water sand
    aq = sands[sands.fluid == "water"].sort_values("thick").iloc[-1]
    aquifer = (aq.top, aq.base)
    cl = w.loc[aq.top:aq.base]; cl = cl[(vsh.loc[aq.top:aq.base] < 0.1) & ~cl.WASHOUT]
    phid_cl, nphi_cl = cl.PHID.median(), cl.NPHI.median()
    d_ss = cl.NPHI_LS - sandstone_nphi_ls(cl.ZDEN)          # distance from the sandstone line, limestone units
    near_ss = (d_ss.abs() <= 0.03).mean()
    d_ls = cl.NPHI_LS - (2.71 - cl.ZDEN) / (2.71 - RHO_F)   # distance from the limestone line
    closer_ss = (d_ss.abs() < d_ls.abs()).mean()

    # SP Rw in the aquifer
    shale_base = np.mean([w.loc[5400:5500].SP.median(), w.loc[5885:5985].SP.median()])
    ssp = w.loc[aq.top:aq.base].SP.median() - shale_base
    t_aq = w.loc[aq.top:aq.base].TEMP.mean()
    sp = rw_from_sp(ssp, t_aq)

    # shales
    shales = []
    for name, (a, b) in {"upper shale": (5400, 5500), "middle shale": (5885, 5985)}.items():
        z = w.loc[a:b]; z = z[~z.WASHOUT]
        shales.append(dict(shale=name, interval=f"{a}–{b}", GR=z.GR.median(), ZDEN=z.ZDEN.median(),
                           CNCF=z.NPHI.median(), PHID=z.PHID.median(), DT=z.DT24.median(), RIPD=z.RIPD.median(),
                           PE=z.PE.median(), K=z.K.median(), TH=z.TH.median(), U=z.U.median()))
    shales = pd.DataFrame(shales)
    phi_sh = shales.PHID.mean()
    phi_clean = cl.PHID.quantile(0.9)

    lags, scan = depth_shift_scan(df, core)
    best = lags[np.nanargmax(scan["all cores"])]

    fig_composite(df, core, sands, OUT / "01_composite.png")
    fig_shift(lags, scan, OUT / "02_depth_shift.png")
    fig_crossplots(df, sands, vsh, OUT / "03_crossplots.png")
    fig_thomas_stieber(df, vsh, aquifer, phi_clean, phi_sh, OUT / "04_thomas_stieber.png")
    fig_nd_aquifer(cl, aquifer, OUT / "05_nd_aquifer_limestone.png")

    # core vs. log at the plugs
    core["phid_log"] = np.interp(core.depth_log, df.index, df.PHID)
    e = core.phid_log - core.phi_3500

    fmt = lambda x, d=2: "" if pd.isna(x) else f"{x:.{d}f}"
    L = ["# Project 2: first look at the data", "",
         f"Interval {TOP:.0f}–{BASE:.0f} ft MD, 0.5 ft sampling. Figures in this folder; generated by "
         "`project2/explore.py` in the formation-evaluation-inversion repo.", "",
         "## Data quality", "",
         f"- ZDEN is already corrected (Baker reports the applied correction as ZCOR). A correction above "
         f"{LARGE_ZCOR} g/cc was applied at {int(w.LARGE_ZCOR.sum())} samples ({w.LARGE_ZCOR.mean()*100:.0f} %); "
         "these are kept but marked in the composite as less certain.",
         f"- Caliper more than 1 in over bit size ({BIT} in): {int(w.WASHOUT.sum())} samples ({w.WASHOUT.mean()*100:.0f} %).",
         f"- RTBR has {int(w.BAD_RT.sum())} negative values; RTBR is not used.",
         "- Spectral K/Th/U are zero above 5,238 ft (not recorded there); inside the interval they are present.",
         "- No Rh/Rv (triaxial) and no deviation survey: shale topology cannot be diagnosed from anisotropy here.", "",
         "## Core", "",
         f"- {len(core)} plugs, cores 1–5, sands {', '.join(core.sand.unique())}; log depths "
         f"{core.depth_log.min():.1f}–{core.depth_log.max():.1f} ft.",
         "- Core-to-log shifts already applied in the spreadsheet: " + ", ".join(
             f"core {k}: {core.loc[core.core == k, 'shift'].iloc[0]:+.0f} ft" for k in sorted(core.core.unique())) + ".",
         f"- Check: correlation of core φ with density φD is highest at {best:+.1f} ft of additional shift "
         f"(r = {np.nanmax(scan['all cores']):.2f}; r = {scan['all cores'][lags == 0][0]:.2f} at 0 ft). "
         "See 02_depth_shift.png.",
         f"- Density φD vs. core φ (3,500 psi) at the plugs: bias {e.mean():+.3f}, RMS {np.sqrt((e**2).mean()):.3f}.",
         f"- {int(core.kh_3500_censored.sum())} plugs report kh > 9,999 md (censored).", "",
         "## Neutron units and lithology in the aquifer", "",
         f"CNCF is in sandstone units (from class). For the lithology cross-plot it was converted to limestone units "
         f"with an offset of {NPHI_SS_TO_LS:+.2f} v/v at zero porosity that tapers with porosity (an approximation "
         "of the chart conversion), and plotted on the "
         f"limestone-unit chart lines (05_nd_aquifer_limestone.png). In the clean part of the water sand "
         f"({aq.top:.0f}–{aq.base:.0f} ft, Vsh < 0.1, {len(cl)} samples): density φD = {phid_cl:.3f}, "
         f"CNCF = {nphi_cl:.3f} (sandstone units).", "",
         f"- {closer_ss*100:.0f} % of the samples lie closer to the sandstone line than to the limestone line, "
         f"and {near_ss*100:.0f} % within ±0.03 v/v of it: the dominant lithology is quartz sandstone. The median "
         f"sample plots {d_ss.median():+.3f} v/v from the sandstone line, toward lower neutron.",
         f"- That offset matches the {phid_cl - nphi_cl:.3f} gap between the medians in sandstone units: converting "
         "units moves "
         "the data and the chart lines together, so it does not change it. A heavier, saline pore fluid would raise "
         "φD and widen the gap, so it is not the cause; a matrix density slightly below 2.65 g/cc (core grain "
         "density 2.63–2.68 g/cc) or the neutron's environmental corrections could account for it. Unresolved.", "",
         "## Gamma-ray end-points and shales", "",
         f"GR clean {gr_cl:.0f} API, GR shale {gr_sh:.0f} API (3rd/97th percentiles over the interval).", "",
         shales.to_markdown(index=False, floatfmt=".3g"), "",
         "## Sand bodies (GR-index Vsh < 0.4, at least 4 ft thick)", "",
         f"Fluid call from the apparent Rw (Rwa = Rt·φD^{M}); the reference Rw = lowest Rwa of the thick "
         f"(≥ 20 ft) sands = {rw_rwa:.3f} Ω·m. Rt/R0 > 3 → possible hydrocarbon; < 1.6 → water. Sw estimate is a "
         "quick Archie number for orientation only.", "",
          sands[["top", "base", "thick", "gr", "vsh", "sp", "rt", "rxo", "phid", "nphi", "rwa", "rt_over_r0", "sw_est", "fluid"]]
              .to_markdown(index=False, floatfmt=(".1f", ".1f", ".1f", ".0f", ".2f", ".0f", ".2f", ".2f", ".3f", ".3f",
                                                  ".3f", ".1f", ".2f", "")), "",
         "## Rw from the SP (first estimate)", "",
         f"Thick water sand {aq.top:.0f}–{aq.base:.0f} ft, SSP = {ssp:.1f} mV (median sand SP minus the shale "
         f"baseline {shale_base:.1f} mV, the mean of the two shales' medians). "
         f"Temperature {t_aq:.0f} °F assuming {T_SURF:.0f} °F at surface and {BHT:.0f} °F at {BHT_DEPTH:.0f} ft.", "",
         f"- Rmf at formation temperature = {sp['rmf']:.3f} Ω·m; Rmfe = {sp['rmfe']:.3f}; K = {sp['k']:.1f}",
         f"- Rwe = {sp['rwe']:.4f} Ω·m → Rw = {sp['rw']:.4f} Ω·m (Bateman–Konen form of chart SP-2)",
         f"- Compare: Rwa in the same sand = {aq.rwa:.4f} Ω·m (Archie, m = {M}, φD).", "",
         "## Open questions for Thursday", "",
         f"- Surface temperature for the gradient (assumed {T_SURF:.0f} °F).",
         "- Which core porosity / permeability to use: 800 psi or 3,500 psi confining pressure.",
         "- Shale porosity and B for Juhasz: from the shale density/neutron or from core?",
         "- The 'Dufasche' formula mentioned in class."]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
