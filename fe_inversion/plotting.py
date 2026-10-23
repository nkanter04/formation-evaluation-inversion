"""Figures: log composite, cross-plots, inversion results with uncertainty."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import petro

C = dict(gr="#2E7D32", cali="#6D4C41", rxo="#F9A825", rt="#C62828", rh="#1565C0",
         rv="#8E24AA", rhob="#C62828", nphi="#1565C0", pe="#6A1B9A", k="#00838F",
         th="#EF6C00", core="#111111", inv="#087E8B", band="#65B8BE", prior="#D18A28",
         conv="#94A3B8", hc="#2E7D32", water="#1565C0", sxo="#AD1457")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 110})


def _style(ax):
    ax.grid(True, color="#DDE3EA", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FCFCFD")


def _tracks(n, depth, title, width=2.4, height=11):
    fig, axes = plt.subplots(1, n, figsize=(width * n + 0.8, height), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax in axes:
        _style(ax)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
    axes[0].set_ylim(depth.max(), depth.min())
    axes[0].set_ylabel("Measured depth (ft)")
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=0.995)
    return fig, axes


def _tops(axes, tops, label_ax=-1):
    for name, d in tops.items():
        for ax in axes:
            ax.axhline(d, color="#455A64", lw=0.8, ls="--", alpha=0.7)
        axes[label_ax].text(1.02, d, name, transform=axes[label_ax].get_yaxis_transform(),
                            fontsize=7, va="center", color="#455A64")


def _band(ax, depth, s, name, color=C["band"], label="P10-P90", scale=1.0):
    ax.fill_betweenx(depth, s[f"{name}_p10"] * scale, s[f"{name}_p90"] * scale,
                     color=color, alpha=0.35, lw=0, label=label)
    ax.plot(s[f"{name}_p50"] * scale, depth, color=C["inv"], lw=0.9, label="P50 inversion")


def _twin(ax, lim, label, color):
    t = ax.twiny()
    t.set_xlim(*lim)
    t.set_xlabel(label, color=color)
    t.tick_params(axis="x", colors=color)
    t.xaxis.set_label_position("top")
    t.spines["top"].set_position(("outward", 32))
    return t


def log_composite(r, path):
    df, cfg = r["df"], r["cfg"]
    d = df.index.values
    fig, ax = _tracks(6, d, "Well logs and core data (Chevron2009)")
    ax[0].plot(df.GR, d, color=C["gr"], lw=0.8, label="GR")
    ax[0].set_xlim(0, 150)
    ax[0].set_xlabel("GR (gAPI)", color=C["gr"])
    ax[0].fill_betweenx(d, r["endpoints"]["gr_sand"], df.GR, color="#FFF59D", alpha=0.5, lw=0)
    t = _twin(ax[0], (11, 15), "Caliper (in)", C["cali"])
    t.plot(df.CALI, d, color=C["cali"], lw=0.7)

    for col, c, lab in (("RXO", C["rxo"], "AT10"), ("RT", C["rt"], "AT90"),
                        ("RH", C["rh"], "Rh (parallel)"), ("RV", C["rv"], "Rv (perpendicular)")):
        ax[1].plot(df[col], d, color=c, lw=0.8, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.2, 100)
    ax[1].set_xlabel("Resistivity (ohm.m)")
    ax[1].legend(loc="lower right")

    ax[2].plot(df.RHOB, d, color=C["rhob"], lw=0.8, label="RHOB")
    ax[2].set_xlim(1.95, 2.95)
    ax[2].set_xlabel("Bulk density (g/cc)", color=C["rhob"])
    t = _twin(ax[2], (0.45, -0.15), "NPHI (v/v, LS)  |  core porosity", C["nphi"])
    t.plot(df.NPHI, d, color=C["nphi"], lw=0.8, ls="--")
    t.scatter(df.CORE_POR, d, s=9, color=C["core"], zorder=5, label="Core porosity")
    # Neutron-density crossover (density porosity > neutron): light fluid.
    nphi_on_rho = 1.95 + (0.45 - df.NPHI) / 0.60 * 1.0
    ax[2].fill_betweenx(d, df.RHOB, nphi_on_rho, where=nphi_on_rho > df.RHOB,
                        color="#FFD54F", alpha=0.6, lw=0, label="N-D crossover")
    ax[2].legend(loc="lower right")
    bad = r["flags"].BAD_RHOB
    if bad.any():
        ax[2].scatter(np.full(bad.sum(), 2.9), d[bad.values], marker="x", s=8, color="k",
                      label="|DRHO| flag")

    ax[3].plot(df.PE, d, color=C["pe"], lw=0.8)
    ax[3].set_xlim(0, 8)
    ax[3].set_xlabel("PEF (b/e)", color=C["pe"])

    ax[4].plot(df.POTA, d, color=C["k"], lw=0.8)
    ax[4].set_xlim(0, 3)
    ax[4].set_xlabel("Potassium (%)", color=C["k"])
    t = _twin(ax[4], (0, 25), "Thorium (ppm)", C["th"])
    t.plot(df.THOR, d, color=C["th"], lw=0.8)

    ax[5].scatter(df.CORE_PERM, d, s=10, color=C["core"])
    ax[5].set_xscale("log")
    ax[5].set_xlim(0.01, 3000)
    ax[5].set_xlabel("Core permeability (md)")
    t = _twin(ax[5], (0, 1), "Core Sw | So (v/v)", C["water"])
    t.scatter(df.CORE_SW, d, s=8, color=C["water"], marker="s")
    t.scatter(df.CORE_SO, d, s=8, color=C["hc"], marker="^")
    _tops(ax, cfg.get("tops", {}))
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def crossplots(r, path):
    df, ep, props, conv = r["df"], r["endpoints"], r["props"], r["conv"]
    d = df.index.values
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ax = ax.ravel()
    for a in ax:
        _style(a)

    sc = ax[0].scatter(df.GR, df.RT, c=d, s=5, cmap="viridis_r")
    ax[0].set(yscale="log", xlabel="GR (gAPI)", ylabel="Deep resistivity AT90 (ohm.m)",
              title="(a) Deep resistivity vs. GR")
    fig.colorbar(sc, ax=ax[0], label="Depth (ft)")

    sc = ax[1].scatter(df.NPHI, df.RHOB, c=df.GR, s=5, cmap="copper_r", vmin=20, vmax=110)
    pts = {"Quartz": (props["nphi_ma"], props["rho_ma"]), "Shale": (ep["nphi_sh"], ep["rho_sh"]),
           "Water": (1.0, props["rho_w"])}
    tri = np.array(list(pts.values()) + [list(pts.values())[0]])
    ax[1].plot(tri[:, 0], tri[:, 1], "k-", lw=0.8)
    for name, (x, y) in pts.items():
        ax[1].plot(x, y, "ko", ms=5)
        ax[1].annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    for phi in (0.1, 0.2, 0.3, 0.4):  # clean-sand water line porosity ticks
        x = phi + (1 - phi) * props["nphi_ma"]
        y = phi * props["rho_w"] + (1 - phi) * props["rho_ma"]
        ax[1].plot(x, y, "k|", ms=8)
        ax[1].annotate(f"{phi:.1f}", (x, y), textcoords="offset points", xytext=(-6, -12), fontsize=7)
    ax[1].set(xlim=(-0.05, 0.6), ylim=(2.7, 1.9), xlabel="NPHI (v/v, LS units)",
              ylabel="RHOB (g/cc)", title="(b) Neutron-density (colored by GR)")
    fig.colorbar(sc, ax=ax[1], label="GR (gAPI)")

    sc = ax[2].scatter(df.THOR, df.POTA, c=df.GR, s=5, cmap="copper_r", vmin=20, vmax=110)
    for ratio in (3.5, 12, 25):
        k = np.linspace(0, 3, 10)
        ax[2].plot(k * ratio, k, "k--", lw=0.6)
        x = min(2.6 * ratio, 21.0)
        ax[2].annotate(f"Th/K = {ratio:g}", (x, x / ratio), textcoords="offset points",
                       xytext=(-50, 4), fontsize=7)
    ax[2].set(xlim=(0, 25), ylim=(0, 3), xlabel="Thorium (ppm)", ylabel="Potassium (%)",
              title="(c) K vs. Th (clay type)")
    fig.colorbar(sc, ax=ax[2], label="GR (gAPI)")

    sc = ax[3].scatter(df.GR, df.RHOB, c=d, s=5, cmap="viridis_r")
    ax[3].set(xlabel="GR (gAPI)", ylabel="RHOB (g/cc)", title="(d) GR vs. bulk density")
    fig.colorbar(sc, ax=ax[3], label="Depth (ft)")

    ok = (df.RH > 0) & (df.RV > 0)
    sc = ax[4].scatter(df.RH[ok], df.RV[ok], c=df.GR[ok], s=5, cmap="copper_r", vmin=20, vmax=110)
    rss_grid = np.logspace(-1, 2.5, 200)
    for c in (0.0, 0.2, 0.4, 0.6, 0.8):
        rh = 1 / (c / ep["rsh_h"] + (1 - c) / rss_grid)
        rv = c * ep["rsh_v"] + (1 - c) * rss_grid
        ax[4].plot(rh, rv, color="#455A64", lw=0.6)
        i = np.searchsorted(rv, 60.0)  # label where the curve leaves the top of the axes
        if i < len(rv) and rh[i] < 20:
            ax[4].annotate(f"Csh={c:.1f}", (rh[i], rv[i]), fontsize=7, color="#455A64",
                           rotation=80, ha="right")
    csh_grid = np.linspace(0, 1, 100)
    for rss in (0.3, 1, 3, 10, 30):
        rh = 1 / (csh_grid / ep["rsh_h"] + (1 - csh_grid) / rss)
        rv = csh_grid * ep["rsh_v"] + (1 - csh_grid) * rss
        ax[4].plot(rh, rv, color="#D18A28", lw=0.6, ls="--")
        ax[4].annotate(f"Rss={rss:g}", (rh[0], rv[0]), fontsize=7, color="#D18A28")
    ax[4].plot([0.1, 100], [0.1, 100], "k:", lw=0.8)
    ax[4].plot(ep["rsh_h"], ep["rsh_v"], "ks", ms=6)
    ax[4].set(xscale="log", yscale="log", xlim=(0.2, 20), ylim=(0.2, 100),
              xlabel="Rh, parallel (ohm.m)", ylabel="Rv, perpendicular (ohm.m)",
              title="(e) Rv vs. Rh with laminated-model grid")
    fig.colorbar(sc, ax=ax[4], label="GR (gAPI)")

    phi_clean = float(np.nanpercentile(conv.PHI_SS_D[conv.CSH_RES < 0.2], 90)) \
        if (conv.CSH_RES < 0.2).any() else 0.33
    sc = ax[5].scatter(conv.CSH_RES, conv.PHIT_D, c=d, s=5, cmap="viridis_r")
    for name, (x, y) in petro.thomas_stieber_lines(phi_clean, ep["phi_sh"]).items():
        ax[5].plot(x, y, lw=1.2, label=name)
    ax[5].set(xlim=(0, 1), ylim=(0, 0.5), xlabel="Csh (resistivity, laminated)",
              ylabel="Total density porosity (v/v)", title="(f) Thomas-Stieber")
    ax[5].legend(loc="upper right")
    fig.colorbar(sc, ax=ax[5], label="Depth (ft)")
    fig.suptitle("Lithology, fluid and shale-distribution cross-plots", fontsize=13,
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def csh_methods(r, path):
    df, conv, s = r["df"], r["conv"], r["summary"]
    d = df.index.values
    fig, ax = _tracks(3, d, "Shale concentration: conventional methods vs. inversion", width=3.2)
    ax[0].plot(conv.CSH_GR, d, color=C["gr"], lw=0.8, label="GR linear")
    ax[0].plot(conv.CSH_LARIONOV, d, color="#81C784", lw=0.8, label="GR Larionov (Tertiary)")
    ax[0].plot(conv.CSH_ND, d, color=C["rhob"], lw=0.8, label="Neutron-density")
    ax[0].set(xlim=(0, 1), xlabel="Csh (v/v)")
    ax[0].legend(loc="lower left")
    ax[1].plot(conv.CSH_RES, d, color=C["rv"], lw=0.8, label="Rh-Rv laminated (closed form)")
    ax[1].plot(conv.CSH_GR, d, color=C["gr"], lw=0.6, alpha=0.6, label="GR linear")
    ax[1].set(xlim=(0, 1), xlabel="Csh (v/v)")
    ax[1].legend(loc="lower left")
    _band(ax[2], d, s, "csh")
    ax[2].plot(s.csh_prior_p50, d, color=C["prior"], lw=0.7, ls="--", label="GR prior (P50)")
    ax[2].set(xlim=(0, 1), xlabel="Csh (v/v), joint inversion")
    ax[2].legend(loc="lower left")
    _tops(ax, r["cfg"].get("tops", {}))
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def inversion_tracks(r, path):
    df, conv, s = r["df"], r["conv"], r["summary"]
    d = df.index.values
    n_real = r["settings"]["realizations"]
    fig, ax = _tracks(7, d, f"Joint laminated inversion, {n_real} Monte Carlo realizations "
                            "(bands = P10-P90)", width=2.7)
    _band(ax[0], d, s, "csh")
    ax[0].plot(s.csh_prior_p50, d, color=C["prior"], lw=0.7, ls="--", label="GR prior")
    ax[0].set(xlim=(0, 1), xlabel="Csh (v/v)")
    ax[0].legend(loc="lower left")

    _band(ax[1], d, s, "phi")
    ax[1].plot(conv.PHI_SS_D, d, color=C["conv"], lw=0.6, label="Density, sequential")
    ax[1].scatter(df.CORE_POR, d, s=9, color=C["core"], zorder=5, label="Core")
    ax[1].set(xlim=(0, 0.45), xlabel="Sand porosity (v/v)")
    ax[1].legend(loc="lower left")

    _band(ax[2], d, s, "sw")
    ax[2].plot(conv.SW_SS, d, color=C["conv"], lw=0.6, label="Archie, sequential")
    ax[2].scatter(df.CORE_SW, d, s=9, color=C["core"], zorder=5, label="Core Sw")
    ax[2].set(xlim=(0, 1), xlabel="Sand Sw (v/v)")
    ax[2].legend(loc="lower left")

    _band(ax[3], d, s, "sxo", color="#F48FB1", label="Sxo P10-P90")
    ax[3].plot(s.sw_p50, d, color=C["water"], lw=0.7, label="Sw P50")
    ax[3].set(xlim=(0, 1), xlabel="Sand Sxo (v/v)")
    ax[3].legend(loc="lower left")

    _band(ax[4], d, s, "rss")
    ax[4].plot(conv.RSS_RES, d, color=C["conv"], lw=0.6, label="Closed form")
    ax[4].set(xscale="log", xlim=(0.1, 300), xlabel="Sand Rss (ohm.m)")
    ax[4].legend(loc="lower left")

    _band(ax[5], d, s, "hpv", color="#A5D6A7")
    ax[5].set(xlim=(0, 0.3), xlabel="HPV (v/v)")
    ax[5].legend(loc="lower left")

    ax[6].plot(s.misfit_p50, d, color="#B65B80", lw=0.8)
    ax[6].axvline(1, color="k", ls=":", lw=0.8)
    ax[6].set(xlim=(0, 4), xlabel="Misfit (RMS / noise)")
    _tops(ax, r["cfg"].get("tops", {}))
    fig.text(0.5, 0.005, "Sand properties are blank where Csh exceeds "
             f"{r['settings']['shale_mask']:.2f} (not resolvable in pure shale). Bands include "
             "end-point, fluid, Rw/Rmf and measurement uncertainty.",
             ha="center", fontsize=8, color="#596576")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def data_fit(r, path):
    df, s = r["df"], r["summary"]
    d = df.index.values
    fig, ax = _tracks(4, d, "Measured logs vs. logs predicted by the inversion (P10-P90)",
                      width=2.8)
    spec = (("RH", "pred_rh", "Rh (ohm.m)", True, (0.2, 20)),
            ("RV", "pred_rv", "Rv (ohm.m)", True, (0.2, 100)),
            ("RHOB", "pred_rhob", "RHOB (g/cc)", False, (1.95, 2.55)),
            ("NPHI", "pred_nphi", "NPHI (v/v)", False, (0.6, 0.0)))
    for a, (obs, pred, lab, log, lim) in zip(ax, spec):
        a.plot(df[obs], d, color="k", lw=0.7, label="Measured")
        a.fill_betweenx(d, s[f"{pred}_p10"], s[f"{pred}_p90"], color=C["band"], alpha=0.5, lw=0)
        a.plot(s[f"{pred}_p50"], d, color=C["inv"], lw=0.7, label="Predicted P50")
        a.set(xlabel=lab, xlim=lim)
        if log:
            a.set_xscale("log")
        a.legend(loc="lower left")
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def pressure_plot(r, path):
    pres, core = r["pressure"], r["core"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    for a in ax:
        _style(a)
    pts = pres["points"]
    ax[0].scatter(pts.FM_PRESS, pts.TVD, color="k", s=18, label="Formation pressure", zorder=5)
    t = np.linspace(pts.TVD.min() - 20, pts.TVD.max() + 20, 10)
    for key, color, name in (("hc", C["hc"], "Hydrocarbon"), ("water", C["water"], "Water")):
        if key in pres:
            g = pres[key]
            lab = f"{name}: {g['gradient']:.3f} psi/ft = {g['density']:.2f} g/cc"
            if g.get("assumed"):
                lab += " (assumed, 1 point)"
            ax[0].plot(g["gradient"] * t + g["intercept"], t, color=color, lw=1.2, label=lab)
    if "fwl_tvd" in pres:
        ax[0].axhline(pres["fwl_tvd"], color="#455A64", ls="--", lw=0.9,
                      label=f"Free-water level ~ {pres['fwl_md']:.0f} ft MD")
    ax[0].invert_yaxis()
    ax[0].set(xlabel="Pressure (psi)", ylabel="TVD below 9,400 ft MD (ft)",
              title="Formation pressure vs. true vertical depth")
    ax[0].legend(loc="upper right")

    rho = core.RHO_F_CORE[(core.RHO_F_CORE > -0.5) & (core.RHO_F_CORE < 1.5)]
    ax[1].hist(rho, bins=25, color="#B0BEC5", edgecolor="white", label="Core-calibrated, per plug")
    ax[1].axvline(np.median(rho), color="k", lw=1.2, label=f"Median = {np.median(rho):.2f} g/cc")
    if "hc" in pres:
        ax[1].axvline(pres["hc"]["density"], color=C["hc"], lw=1.5,
                      label=f"Pressure gradient = {pres['hc']['density']:.2f} g/cc")
    ax[1].axvline(r["props"]["rho_w"], color=C["water"], lw=1.2, ls="--", label="Formation water")
    ax[1].set(xlabel="Pore-fluid density (g/cc)", ylabel="Core samples",
              title="Fluid density: core porosity + density log vs. pressures")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def core_plots(r, path):
    df, s = r["df"], r["summary"]
    c = df[df.CORE_POR.notna()]
    sc_ = s.loc[c.index]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    for a in ax:
        _style(a)
    sc = ax[0].scatter(c.CORE_POR, c.CORE_PERM, c=c.CORE_SW, cmap="coolwarm_r", s=22, vmin=0, vmax=0.8)
    good = (c.CORE_PERM > 0) & c.CORE_POR.notna()
    if good.sum() > 3:
        b, a0 = np.polyfit(c.CORE_POR[good], np.log10(c.CORE_PERM[good]), 1)
        x = np.linspace(c.CORE_POR.min(), c.CORE_POR.max(), 20)
        r2 = np.corrcoef(c.CORE_POR[good], np.log10(c.CORE_PERM[good]))[0, 1] ** 2
        ax[0].plot(x, 10 ** (a0 + b * x), "k--", lw=1, label=f"log k = {a0:.2f} + {b:.1f} phi, R² = {r2:.2f}")
        ax[0].legend()
    ax[0].set(yscale="log", xlabel="Core porosity (v/v)", ylabel="Core permeability (md)",
              title="Core k vs. phi (color = core Sw)")
    fig.colorbar(sc, ax=ax[0], label="Core Sw")

    ax[1].scatter(sc_.sw_p50, c.CORE_PERM, s=22, color=C["inv"], label="Inverted Sw (P50)")
    ax[1].scatter(c.CORE_SW, c.CORE_PERM, s=14, color="#BDBDBD", label="Core Sw")
    ax[1].set(yscale="log", xlim=(0, 1), xlabel="Water saturation (v/v)", ylabel="Core permeability (md)",
              title="Permeability vs. (irreducible) Sw")
    ax[1].legend()

    for a, (inv, core, lab) in zip(ax[2:], (("phi", "CORE_POR", "porosity"), ("sw", "CORE_SW", "Sw"))):
        err = np.vstack([sc_[f"{inv}_p50"] - sc_[f"{inv}_p10"], sc_[f"{inv}_p90"] - sc_[f"{inv}_p50"]])
        a.errorbar(c[core], sc_[f"{inv}_p50"], yerr=np.clip(err, 0, None), fmt="o", ms=3.5,
                   color=C["inv"], ecolor="#9FD3D7", elinewidth=0.8)
        lim = (0, 0.45) if inv == "phi" else (0, 1)
        a.plot(lim, lim, "k:", lw=1)
        a.set(xlim=lim, ylim=lim, xlabel=f"Core {lab}", ylabel=f"Inverted sand {lab} (P50, P10-P90)",
              title=f"Inversion vs. core {lab}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def monte_carlo_plots(r, path):
    hc, params = r["hc_column"], r["params"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a in ax:
        _style(a)
    for name, v in hc.items():
        if np.nanmax(v) <= 0.05:
            continue
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        ax[0].hist(v, bins=40, alpha=0.55, label=f"{name}: P10 {p10:.1f} | P50 {p50:.1f} | P90 {p90:.1f} ft")
    ax[0].set(xlabel="Hydrocarbon column, sum of HPV x dTVD (ft)", ylabel="Realizations",
              title="Distribution of hydrocarbon pore volume")
    ax[0].legend(fontsize=7)

    total = hc["Whole interval"]
    names, rho = [], []
    for col in params.columns:
        x = params[col].values
        if np.std(x) == 0:
            continue
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(total))
        names.append(col)
        rho.append(np.corrcoef(rx, ry)[0, 1])
    order = np.argsort(np.abs(rho))
    ax[1].barh(np.array(names)[order], np.array(rho)[order],
               color=["#C62828" if v < 0 else "#2E7D32" for v in np.array(rho)[order]])
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set(xlabel="Rank correlation with total hydrocarbon column",
              title="Which uncertain inputs drive the HPV uncertainty?")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def all_figures(r, out_dir):
    figs = {
        "01_log_composite.png": log_composite,
        "02_crossplots.png": crossplots,
        "03_csh_methods.png": csh_methods,
        "04_inversion.png": inversion_tracks,
        "05_data_fit.png": data_fit,
        "06_pressure_fluid_density.png": pressure_plot,
        "07_core_comparison.png": core_plots,
        "08_monte_carlo.png": monte_carlo_plots,
    }
    for name, fn in figs.items():
        if fn is pressure_plot and not r.get("pressure"):
            continue
        fn(r, out_dir / name)
    return list(figs)
