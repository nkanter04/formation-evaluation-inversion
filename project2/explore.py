"""PGE 385K Homework Project 2: load, quality-check and visualize the logs and core data.

Run:  python project2/explore.py
Writes figures and summary.md to the Project 2 "4 Python Figures and Results/Exploration" folder.
"""
from pathlib import Path

import tomllib

import lasio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P2 = Path(r"C:\Users\nkant\OneDrive\The University of Texas at Austin\Fall 2026\PGE 385K - Adv. Formation Eval"
          r"\PGE 385K - Homework Problems\Homework Projects\Homework Project No. 2")
RAW = P2 / "2 Raw Data (Canvas)"
RESULTS = P2 / "4 Python Figures and Results"
LAS = RAW / "HA_2_reader_compatible.las"
CORE = RAW / "HA2_core_data.xls"
OUT = RESULTS / "Exploration"

TOP, BASE = 5400.0, 6300.0
BIT = 8.375                      # in, log header
RHO_MA, RHO_F = 2.65, 1.0        # quartz; mud-filtrate density for density porosity
A, M, N = 1.0, 2.1, 1.95         # Archie parameters given in the assignment
RMF_75, RM_75 = 1.32, 2.52       # ohm-m at 75 degF, log header
BHT, BHT_DEPTH = 163.0, 6900.0   # degF at logger depth
T_SURF = 80.0                    # degF, ASSUMED mean surface temperature (not on the header)
LARGE_ZCOR = 0.10                # g/cc; ZDEN is already corrected, this only marks large corrections
# Schlumberger chart Por-5 (former Por-13b), thermal neutron NPHI, porosity equivalence: true porosity of
# quartz sandstone and dolomite against the apparent limestone reading (v/v). Digitized from the curve colours
# of the 2009 chart book (about ±0.005); the sandstone value at zero porosity was read by eye.
POR5_PHI = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
POR5_SS_LS = np.array([-0.024, 0.0148, 0.0578, 0.1042, 0.1531, 0.2015, 0.2525, 0.3031, 0.3524])
POR5_DOL_LS = np.array([0.0323, 0.1192, 0.1798, 0.2312, 0.2780, 0.3202, 0.3611, 0.3971, 0.4330])


CFG = tomllib.loads((Path(__file__).resolve().parents[1] / "config" / "project2.toml").read_text(encoding="utf-8"))
QUARTZ, WATER_U = (4.78, 2.65), (0.398, 1.11)   # (U in b/cc, electron density) for the PE-density lines


# ----------------------------------------------------------------------------------- loading
def bed_centers(w):
    """Mask of bed-center samples: inside beds >= min thickness, away from GR/density boundaries."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fe_inversion import beds
    b = CFG["beds"]
    z = w.index.values
    idx = beds.boundaries(z, [w.GR.values, w.ZDEN.values], [b["gr_gradient"], b["rho_gradient"]])
    return pd.Series(beds.thick_bed_centers(z, idx, b["min_thickness_ft"], b["edge_ft"]), index=w.index), len(idx) - 1


def _interp_ext(x, xp, fp):
    """Linear interpolation, extended linearly beyond the last chart point (shales read about 0.5)."""
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)
    hi = x > xp[-1]
    return np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)


def ss_to_ls(nphi_ss):
    """Sandstone- to limestone-unit neutron porosity from chart Por-5 (about -0.047 above 0.15)."""
    return _interp_ext(nphi_ss, POR5_PHI, POR5_SS_LS)


def ls_to_ss(nphi_ls):
    return _interp_ext(nphi_ls, POR5_SS_LS, POR5_PHI)


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
    fig.suptitle("ORM-75: logs and core, 5,400–6,300 ft (blue: water sandstones, green: possible hydrocarbon, "
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
    """Water-filled sandstone, limestone and dolomite lines (fresh water, as chart Por-11) in the given neutron units,
    with the neutron responses of chart Por-5."""
    phi = np.linspace(0, 0.40, 21)
    ls = {"Sandstone": ss_to_ls(phi), "Limestone": phi, "Dolomite": np.interp(phi, POR5_PHI, POR5_DOL_LS)}
    for name, rho, col in [("Sandstone", 2.65, "goldenrod"), ("Limestone", 2.71, "steelblue"), ("Dolomite", 2.87, "purple")]:
        x = ls[name] if units == "limestone" else ls_to_ss(ls[name])
        ax.plot(x, rho - phi * (rho - 1.0), color=col, lw=1.2, label=name)


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
    fluid = pd.Series("other", index=w.index)
    fluid[vsh.reindex(w.index) >= 0.6] = "shale"
    for _, s in sands.iterrows():
        fluid.loc[s.top:s.base] = {"water": "water", "possible hydrocarbon": "light hydrocarbon"}.get(s.fluid, "other")
    for lab, col in [("other", "0.75"), ("shale", "#1f77b4"), ("water", "#2ca02c"), ("light hydrocarbon", "#17becf")]:
        k = fluid == lab
        a.scatter(w.GR[k], w.RIPD[k], s=4, color=col, label=lab)
    a.set(yscale="log", xlabel="GR (API)", ylabel="RIPD (Ω·m)", title="(b) Deep resistivity vs. GR, by zone")
    a.legend(fontsize=7, markerscale=3)
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
    centers, _ = bed_centers(w)
    good = w[~w.WASHOUT & centers]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, sub, title in [(ax[0], good, "(a) Whole interval"), (ax[1], good.loc[aquifer[0]:aquifer[1]],
                          f"(b) Water sandstone {aquifer[0]:.0f}–{aquifer[1]:.0f} ft")]:
        s = a.scatter(vsh.reindex(sub.index), sub.PHID, c=sub.index, s=6, cmap="viridis_r")
        # laminar: straight line from clean sandstone to shale
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
    fig.suptitle(f"Thomas–Stieber diagram, bed-center samples only (φclean = {phi_clean:.3f}, φsh = {phi_sh:.3f}); Vsh is the linear GR index",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_pe_density(clean, aquifer, path):
    """PE vs. bulk density in the clean water sandstone against water-filled quartz, calcite and dolomite lines."""
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    s = ax.scatter(clean.ZDEN, clean.PE, c=clean.GR, s=14, cmap="viridis")
    phi = np.linspace(0, 0.45, 30)
    for name, (u_ma, rho_e), rho_ma, col in [("Quartz", QUARTZ, 2.65, "goldenrod"), ("Calcite", (13.77, 2.708), 2.71, "steelblue"),
                                             ("Dolomite", (9.0, 2.863), 2.87, "purple")]:
        pe = (phi * WATER_U[0] + (1 - phi) * u_ma) / (phi * WATER_U[1] + (1 - phi) * rho_e)
        ax.plot(phi * RHO_F + (1 - phi) * rho_ma, pe, color=col, lw=1.5, label=f"Water-filled {name.lower()}")
    ax.set(xlim=(1.9, 2.9), ylim=(0, 6), xlabel="Bulk density ZDEN (g/cc)", ylabel="Photoelectric factor PE (b/e)",
           title=f"PE vs. density in the clean water sandstone, {aquifer[0]:.0f}–{aquifer[1]:.0f} ft")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.colorbar(s, ax=ax, label="GR (API)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# Schlumberger Lith-1 / Lith-2 (course book appendix pp. 239-240). Th/K lines are exact; the mineral boxes were
# read off the chart grid (about +/-0.1 in K and PE).
LITH2_RATIOS = (25, 12, 3.5, 2.0, 0.6, 0.3)
LITH1_PE_K = {"Kaolinite": (0.03, 0.64, 1.58, 1.80), "Montmorillonite": (0.03, 4.93, 1.96, 2.23),
              "Illite": (3.58, 8.39, 3.27, 3.62), "Muscovite": (7.93, 9.88, 2.23, 2.49), "Chlorite": (0.0, 0.27, 6.17, 6.38),
              "Glauconite": (3.2, 5.8, 5.33, 7.29), "Biotite": (6.2, 10.0, 6.17, 6.38)}
LITH1_PE_THK = {"Kaolinite": (10, 30, 1.66, 1.84), "Chlorite": (10, 30, 6.12, 6.36), "Mixed layer": (4.34, 9.71, 1.95, 3.85),
                "Montmorillonite": (4.34, 9.71, 1.95, 2.22), "Illite": (2.37, 4.77, 3.29, 3.58), "Muscovite": (1.75, 3.24, 2.19, 2.46),
                "Biotite": (1.75, 3.24, 6.17, 6.36), "Glauconite": (0.76, 1.39, 5.35, 7.30)}


def fig_clay_charts(w, vsh, path):
    """Shale samples on Lith-2 (Th vs. K) and Lith-1 (PE vs. K and PE vs. Th/K), by section."""
    centers, _ = bed_centers(w)
    sh = w[(vsh.reindex(w.index) > 0.75) & centers & ~w.WASHOUT & (w.K > 0) & (w.TH > 0)]
    upper = sh.index < CFG["shale"]["boundary"]
    groups = [("Upper shale section (above 5,885 ft)", sh[upper], "#d62728"), ("Lower section (below 5,885 ft)", sh[~upper], "#1f77b4")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.8))
    a = ax[0]
    kk = np.linspace(0, 5, 10)
    for r in LITH2_RATIOS:
        a.plot(kk, r * kk, "k-", lw=0.8)
        xk = min(4.6, 23.5 / r); a.text(xk, r * xk, f" Th/K = {r}", fontsize=7)
    a.plot([0, 5], [16.7, 20.1], "r--", lw=1, label="Possible 100 % kaolinite, montmorillonite, illite 'clay line'")
    for x, y, lab in [(0.45, 16, "Kaolinite"), (1.6, 13, "Mixed-layer clay"), (2.6, 7.5, "Illite"), (3.4, 4.0, "Micas")]:
        a.text(x, y, lab, fontsize=8, style="italic")
    a.set(xlim=(0, 5), ylim=(0, 25), xlabel="Potassium K (%)", ylabel="Thorium Th (ppm)", title="(a) Lith-2: Th vs. K")
    a = ax[1]
    for name, (k0, k1, p0, p1) in LITH1_PE_K.items():
        a.add_patch(plt.Rectangle((k0, p0), k1 - k0, p1 - p0, color="#9fa8da", alpha=0.6))
        a.text(k0, p1 + 0.08, name, fontsize=7)
    a.set(xlim=(0, 10), ylim=(0, 10), xlabel="Potassium K (%)", ylabel="Photoelectric factor PE (b/e)", title="(b) Lith-1: PE vs. K")
    a = ax[2]
    for name, (r0, r1, p0, p1) in LITH1_PE_THK.items():
        a.add_patch(plt.Rectangle((r0, p0), r1 - r0, p1 - p0, color="#9fa8da", alpha=0.45 if name == "Mixed layer" else 0.6))
        a.text(r0, p1 + 0.08, name, fontsize=7)
    a.set(xscale="log", xlim=(0.1, 100), ylim=(0, 10), xlabel="Th/K ratio", ylabel="Photoelectric factor PE (b/e)",
          title="(c) Lith-1: PE vs. Th/K")
    for lab, g, col in groups:
        med = g[["K", "TH", "PE"]].median()
        ax[0].scatter(g.K, g.TH, s=6, color=col, alpha=0.4)
        ax[1].scatter(g.K, g.PE, s=6, color=col, alpha=0.4)
        ax[2].scatter(g.TH / g.K, g.PE, s=6, color=col, alpha=0.4)
        for a, x, y in [(ax[0], med.K, med.TH), (ax[1], med.K, med.PE), (ax[2], med.TH / med.K, med.PE)]:
            a.plot(x, y, "*", ms=16, color=col, mec="k", label=f"{lab}: median")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7, loc="upper right")
    fig.suptitle("Shale samples (GR index > 0.75, bed centers) on the Schlumberger clay-identification charts; "
                 "PE of water-filled shale reads below its clay minerals' PE", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {lab: g[["K", "TH", "PE"]].median().to_dict() | {"n": len(g)} for lab, g, _ in groups}


def sandstone_nphi_ls(rhob):
    """Limestone-unit neutron of water-filled quartz sandstone at bulk density rhob (the nd_lines sandstone line)."""
    return ss_to_ls((RHO_MA - rhob) / (RHO_MA - RHO_F))


def fig_nd_aquifer(clean, aquifer, path):
    """Neutron-density cross-plot of the clean water sandstone with CNCF converted to limestone units."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    s = ax.scatter(clean.NPHI_LS, clean.ZDEN, c=clean.GR, s=12, cmap="viridis")
    nd_lines(ax, units="limestone")
    ax.plot(clean.NPHI_LS.median(), clean.ZDEN.median(), "r*", ms=16, label="Median of clean sandstone")
    ax.text(0.01, 2.77, "CNCF converted to limestone units with chart Por-5 (NPHI); lithology lines for fresh water\n"
            "as chart Por-11, built from the Por-5 neutron responses", fontsize=7)
    ax.set(xlim=(-0.05, 0.45), ylim=(2.8, 1.9), xlabel="Neutron porosity, limestone units (v/v)",
           ylabel="Bulk density ZDEN (g/cc)",
           title=f"Neutron–density in the clean water sandstone, {aquifer[0]:.0f}–{aquifer[1]:.0f} ft (Vsh < 0.1)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(s, ax=ax, label="GR (API)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------- main
RW_EST = None


SEQUENCES = [  # (top, base, label, colour): stacking read from the GR and the sandstone fraction
    (5400, 5512, "Flooding shale\n(transgressive)", "#c7d3c0"),
    (5512, 5866, "Upper package: fluvial channels\nand splays; sandstone fraction\nfalls upward into the shale\n(retrogradational top)", "#f6dcc0"),
    (5866, 5985, "Flooding shale\n(transgressive)", "#c7d3c0"),
    (5985, 6300, "Lower package: deltaic\ndistributary channels (fining\nupward or blocky) and coarsening-\nupward mouth bars (progradational),\nthin coal at 6,259 ft", "#cfe0f1"),
]


def fig_sequence(w, sands, vsh, path):
    """GR, Csh, sandstone motifs from the GR trend inside each sandstone, coal, and the sequence interpretation."""
    fig, ax = plt.subplots(1, 4, figsize=(13, 14), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 1, 0.8, 1.6]))
    cgr = 4 * w.TH + 16 * w.K
    ax[0].plot(w.GR, w.index, "g", lw=0.7, label="GR (API)")
    ax[0].plot(cgr, w.index, "0.4", lw=0.6, label="K + Th GR, 4 Th + 16 K (API)")
    ax[0].set(xlim=(0, 150), title="Gamma ray", xlabel="API")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fe_inversion import petro
    csh = petro.csh_from_ish(vsh.values, CFG["csh"]["method"])
    ax[1].fill_betweenx(w.index, 0, csh, color="0.6", label="Shale")
    ax[1].fill_betweenx(w.index, csh, 1, color="#e8c66a", label="Sandstone")
    ax[1].set(xlim=(0, 1), title="Csh (Larionov, Tertiary)", xlabel="v/v")
    motif_col = {"Blocky (channel)": "#d9822b", "Coarsening upward (mouth bar / splay)": "#7fb3e0",
                 "Fining upward (channel fill)": "#2c5f9e"}
    a = ax[2]
    a.axvspan(0, 1, color="#d6d8db")
    used = set()
    for _, s in sands.iterrows():
        z = w.loc[s.top:s.base].GR
        slope = np.polyfit(z.index, z.values, 1)[0]          # API/ft, positive = GR rises downward
        motif = ("Coarsening upward (mouth bar / splay)" if slope > 0.3 else
                 "Fining upward (channel fill)" if slope < -0.3 else "Blocky (channel)")
        a.fill_between([0, 1], s.top, s.base, color=motif_col[motif], label=None if motif in used else motif)
        used.add(motif)
    # Coal: runs of at least 3 ft with low density, high neutron and low GR (single low-density spikes are skipped)
    flag = ((w.ZDEN < 2.0) & (w.NPHI > 0.4) & (w.GR < 70)).values
    edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        z0, z1 = w.index[i0], w.index[i1 - 1]
        if z1 - z0 >= 2.5:
            a.fill_between([0, 1], z0, z1, color="k", label=f"Coal ({z0:,.1f}–{z1:,.1f} ft)")
    a.set(xlim=(0, 1), xticks=[], title="Sandstone motif")
    a = ax[3]
    for top, base, lab, col in SEQUENCES:
        a.fill_between([0, 1], top, base, color=col)
        a.text(0.5, (top + base) / 2, lab, ha="center", va="center", fontsize=9)
        a.axhline(top, color="k", lw=0.8)
    a.set(xlim=(0, 1), xticks=[], title="Sequences")
    ax[0].set(ylim=(6300, 5400), ylabel="Measured depth (ft)")
    for x in ax[:2]:
        x.legend(fontsize=7, loc="lower right"); x.grid(alpha=0.3)
    ax[2].legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.suptitle("ORM-75 sedimentary sequences, 5,400–6,300 ft: motifs from the GR trend inside each sandstone", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    global RW_EST
    OUT.mkdir(exist_ok=True)
    df, core = load_logs(), load_core()
    w = df.loc[TOP:BASE]
    sands, vsh, (gr_cl, gr_sh) = pick_sands(w)
    sands, rw_rwa = classify(sands)
    RW_EST = rw_rwa

    # neutron units: cleanest water sandstone
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
    fig_pe_density(cl, aquifer, OUT / "06_pe_density.png")
    clay = fig_clay_charts(w, vsh, OUT / "07_clay_charts.png")
    fig_sequence(w, sands, vsh, OUT / "08_sequence.png")
    rho_e_b = 1.0704 * cl.ZDEN - 0.1883                    # electron density from bulk density (Schlumberger)
    pe_quartz = (cl.PHID * WATER_U[0] + (1 - cl.PHID) * QUARTZ[0]) / (cl.PHID * WATER_U[1] + (1 - cl.PHID) * QUARTZ[1])
    pe_gap = (cl.PE - pe_quartz).median()
    centers, n_beds = bed_centers(w)

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
         "- No Rh/Rv (triaxial) and no deviation survey: shale topology cannot be diagnosed from anisotropy here.",
         f"- Bed boundaries from GR and density gradients: {n_beds} beds; {centers.mean()*100:.0f} % of samples are bed centers "
         "(beds ≥ 3 ft, ≥ 1 ft from a boundary). The Thomas–Stieber plot uses bed centers only.", "",
         "## Core", "",
         f"- {len(core)} plugs, cores 1–5, sandstones {', '.join(core.sand.unique())}; log depths "
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
         "with Schlumberger chart Por-5 (NPHI curve, digitized; about -0.047 v/v above 0.15), and plotted on the "
         f"limestone-unit chart lines (05_nd_aquifer_limestone.png). In the clean part of the water sandstone "
         f"({aq.top:.0f}–{aq.base:.0f} ft, Vsh < 0.1, {len(cl)} samples): density φD = {phid_cl:.3f}, "
         f"CNCF = {nphi_cl:.3f} (sandstone units).", "",
         f"- {closer_ss*100:.0f} % of the samples lie closer to the sandstone line than to the limestone line, "
         f"and {near_ss*100:.0f} % within ±0.03 v/v of it: the dominant lithology is quartz sandstone. The median "
         f"sample plots {d_ss.median():+.3f} v/v from the sandstone line, toward lower neutron.",
         f"- That offset matches the {phid_cl - nphi_cl:.3f} gap between the medians in sandstone units: the unit "
         "conversion moves the data and the sandstone line together. A heavier, saline pore fluid would raise "
         "φD and widen the gap, so it is not the cause; a matrix density slightly below 2.65 g/cc (core grain "
         "density 2.63–2.68 g/cc) or the neutron's environmental corrections could account for it. Unresolved.", "",
         f"- PE vs. density (06_pe_density.png): in the clean water sandstone PE has a median of {cl.PE.median():.2f} b/e, "
         f"{pe_gap:+.2f} b/e from water-filled quartz at the same porosity, consistent with a dominantly quartz matrix.", "",
         "## Clay type from the spectral gamma ray and PE (07_clay_charts.png)", "",
         "Shale samples (GR index > 0.75, bed centers) on Schlumberger Lith-2 and Lith-1 (book appendix):", "",
         "| Section | Samples | K (%) | Th (ppm) | Th/K | PE (b/e) |", "|---|---|---|---|---|---|",
         *[f"| {lab} | {v['n']} | {v['K']:.2f} | {v['TH']:.1f} | {v['TH'] / v['K']:.1f} | {v['PE']:.2f} |" for lab, v in clay.items()], "",
         "## Gamma-ray end-points and shales", "",
         f"GR clean {gr_cl:.0f} API, GR shale {gr_sh:.0f} API (3rd/97th percentiles over the interval).", "",
         shales.to_markdown(index=False, floatfmt=".3g"), "",
         "## Sandstone bodies (GR-index Vsh < 0.4, at least 4 ft thick)", "",
         f"Fluid call from the apparent Rw (Rwa = Rt·φD^{M}); the reference Rw = lowest Rwa of the thick "
         f"(≥ 20 ft) sandstones = {rw_rwa:.3f} Ω·m. Rt/R0 > 3 → possible hydrocarbon; < 1.6 → water. Sw estimate is a "
         "quick Archie number for orientation only.", "",
          sands[["top", "base", "thick", "gr", "vsh", "sp", "rt", "rxo", "phid", "nphi", "rwa", "rt_over_r0", "sw_est", "fluid"]]
              .to_markdown(index=False, floatfmt=(".1f", ".1f", ".1f", ".0f", ".2f", ".0f", ".2f", ".2f", ".3f", ".3f",
                                                  ".3f", ".1f", ".2f", "")), "",
         "## Rw from the SP (first estimate)", "",
         f"Thick water sandstone {aq.top:.0f}–{aq.base:.0f} ft, SSP = {ssp:.1f} mV (median sandstone SP minus the shale "
         f"baseline {shale_base:.1f} mV, the mean of the two shales' medians). "
         f"Temperature {t_aq:.0f} °F assuming {T_SURF:.0f} °F at surface and {BHT:.0f} °F at {BHT_DEPTH:.0f} ft.", "",
         f"- Rmf at formation temperature = {sp['rmf']:.3f} Ω·m; Rmfe = {sp['rmfe']:.3f}; K = {sp['k']:.1f}",
         f"- Rwe = {sp['rwe']:.4f} Ω·m → Rw = {sp['rw']:.4f} Ω·m (Bateman–Konen form of chart SP-2)",
         f"- Compare: Rwa in the same sandstone = {aq.rwa:.4f} Ω·m (Archie, m = {M}, φD).", "",
         "## Open questions for Thursday", "",
         f"- Surface temperature for the gradient (assumed {T_SURF:.0f} °F).",
         "- Which core porosity / permeability to use: 800 psi or 3,500 psi confining pressure.",
         "- Shale porosity and B for Juhasz: from the shale density/neutron or from core?",
         "- The 'Dufasche' formula mentioned in class."]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
