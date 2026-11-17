"""Project 2: UTAPWeLS compositional joint inversion against deterministic Juhasz and the Python Monte Carlo inversion.

Reads the CJI LAS exports listed in config [cji], Interpretation/results.csv (interpret.py) and
Interpretation/inversion_p50.csv (invert.py). HPT is recomputed from phi_t and Sw, because the CJI's own HPT
curve is 3.281 times too large. All three methods are compared on the same samples: the CJI intervals, net
sandstone (GR index below the [net] limit).
"""
import sys
from pathlib import Path

import lasio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import explore as ex  # noqa: E402
import interpret as it  # noqa: E402
from fe_inversion import petro  # noqa: E402

CFG = it.CFG
STEP = it.STEP


def load_cji():
    """CJI curves of all runs on the 0.5 ft log depths, with the derived volumes."""
    parts = []
    for name in CFG["cji"]["files"]:
        c = lasio.read(str(ex.P2 / name)).df()
        c.index = c.index.round(1)
        parts.append(c)
    c = pd.concat(parts).sort_index()
    c = c[~c.index.duplicated()]
    shale_water = c.CSH * c.PHISH
    c["phi_s"] = (c.PHIT - shale_water).clip(lower=0)
    c["sws"] = ((c.PHIT * c.SWT - shale_water) / c.phi_s.where(c.phi_s > 1e-4)).clip(0, 1)
    # Bulk volumes from the inverted phi_t, Sw, Csh and phi_sh with the dispersed-clay volumetrics
    # (quartz + shale grains + shale water + sandstone pores = 1). UTAPWeLS's VOL_SOLID_* curves are not bulk
    # fractions (the shale ones sum to about Csh), so they are not used.
    c["v_shale_grains"] = c.CSH * (1 - c.PHISH)
    c["v_shale_water"] = shale_water
    c["v_quartz"] = (1 - c.CSH - c.phi_s).clip(lower=0)
    c["v_hc"] = c.PHIT * (1 - c.SWT)
    c["v_sand_water"] = (c.PHIT * c.SWT - shale_water).clip(lower=0)
    return c


def main():
    df = ex.load_logs()
    w = df.loc[ex.TOP:ex.BASE].copy()
    w.index = w.index.round(1)
    core = ex.load_core()
    sands, _, _ = ex.pick_sands(df.loc[ex.TOP:ex.BASE])
    cji = load_cji()
    det = pd.read_csv(it.OUT / "results.csv", index_col="depth_ft")
    det.index = det.index.round(1)
    mc = pd.read_csv(it.OUT / "inversion_p50.csv", index_col="depth_ft")
    mc.index = mc.index.round(1)
    gr_csh = pd.Series(petro.csh_from_ish(det.i_gr.values, CFG["csh"]["method"]), index=det.index)

    z = cji.index
    d, m = det.reindex(z), mc.reindex(z)
    sands, _ = ex.classify(sands)
    in_hc = np.zeros(len(z), bool)
    for _, s in sands[sands.fluid == "possible hydrocarbon"].iterrows():
        in_hc |= (z >= s.top) & (z <= s.base)
    net = (d.i_gr < CFG["net"]["max_vsh"]).values & in_hc
    hpv = {"CJI (UTAPWeLS)": (cji.PHIT * (1 - cji.SWT)).values, "Juhasz (deterministic)": d.hpv_j.values,
           "Monte Carlo P50 (Python)": m.hpv.values}
    cum = {k: np.cumsum(np.where(net, v, 0.0)) * STEP for k, v in hpv.items()}

    runs = [(lasio.read(str(ex.P2 / f)).df().index.min(), lasio.read(str(ex.P2 / f)).df().index.max())
            for f in CFG["cji"]["files"]]
    # Plot every method on the same samples: the CJI intervals, with gaps (NaN) where nothing was inverted so
    # lines break, and Sw and HPV blanked in shale (GR index > 0.75), where sandstone saturation has no meaning.
    full = det.loc[runs[0][0]:runs[-1][1]].index
    inrun = np.zeros(len(full), bool)
    for top, base in runs:
        inrun |= (full >= top) & (full <= base)
    shale = (det.i_gr.reindex(full) > 0.75).values
    cji, det, mc, gr_csh = (x.reindex(full) for x in (cji, det, mc, gr_csh))
    det = det.select_dtypes("number").astype(float)
    for frame in (det, mc):
        frame.loc[~inrun, :] = np.nan
    gr_csh[~inrun] = np.nan
    cji.loc[shale, ["sws"]] = np.nan
    det.loc[shale, ["sws", "hpv_j"]] = np.nan
    mc.loc[shale, ["sws", "hpv"]] = np.nan
    cum = {k: pd.Series(v, index=z).reindex(full).values for k, v in cum.items()}
    z = full
    fig, ax = plt.subplots(1, 6, figsize=(17, 15), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1, 1.1]))

    # (a) CJI bulk volumes, stacked in the order of the example: quartz, sandstone water, hydrocarbon,
    # shale water, shale grains
    a = ax[0]
    layers = [("v_quartz", "Sandstone grains (quartz)", "#f0d27a"), ("v_sand_water", "Sandstone water", "#4a90d9"),
              ("v_hc", "Hydrocarbon", "#2ca02c"), ("v_shale_water", "Shale water", "#9ecae1"),
              ("v_shale_grains", "Shale grains", "#7f7f7f")]
    for top, base in runs:
        s = cji.loc[top:base]
        left = np.zeros(len(s))
        for key, lab, col in layers:
            a.fill_betweenx(s.index, left, left + s[key].values, color=col, lw=0, step="mid",
                            label=lab if top == runs[0][0] else None)
            left = left + s[key].values
    a.set(xlim=(0, 1), title="CJI bulk volume\n(fraction)")

    a = ax[1]
    a.plot(cji.CSH, z, "k", lw=0.7, label="CJI")
    a.plot(gr_csh, det.index, color="#d62728", lw=0.6, alpha=0.8, label="Gamma ray (Larionov, Juhasz input)")
    a.set(xlim=(0, 1), title="Shale concentration\nCsh (v/v)")

    a = ax[2]
    a.plot(cji.PHIT, z, "k", lw=0.7, label="CJI")
    a.plot(det.phi_t, det.index, color="#d62728", lw=0.6, alpha=0.8, label="Juhasz")
    a.plot(mc.phi_t, mc.index, color="#1f77b4", lw=0.6, alpha=0.8, label="Monte Carlo P50")
    a.plot(core.phi_3500, core.depth_log, "o", ms=4, mfc="none", color="#08306b", label="Core (3,500 psi)")
    a.set(xlim=(0.45, 0), title="Total porosity\nφt (v/v)")

    a = ax[3]
    a.plot(cji.sws, z, "k", lw=0.7, label="CJI")
    a.plot(det.sws, det.index, color="#d62728", lw=0.6, alpha=0.8, label="Juhasz")
    a.plot(mc.sws, mc.index, color="#1f77b4", lw=0.6, alpha=0.8, label="Monte Carlo P50")
    a.set(xlim=(1, 0), title="Sandstone water\nsaturation Sws (v/v)")

    a = ax[4]
    for top, base in runs:
        s = cji.loc[top:base]
        a.fill_betweenx(s.index, 0, s.v_hc, color="#2ca02c", lw=0, step="mid", label="CJI" if top == runs[0][0] else None)
    a.plot(det.hpv_j, det.index, color="#d62728", lw=0.6, alpha=0.9, label="Juhasz")
    a.plot(mc.hpv, mc.index, color="#1f77b4", lw=0.6, alpha=0.9, label="Monte Carlo P50")
    a.set(xlim=(0, 0.3), title="Hydrocarbon pore\nvolume HPV (v/v)")

    a = ax[5]
    for (lab, c), col in zip(cum.items(), ("k", "#d62728", "#1f77b4")):
        a.plot(c, z, color=col, lw=1.2, label=f"{lab} ({c[-1]:.1f} ft)")
    a.set(xlim=(0, None), title="Cumulative HPT,\nnet HC sandstones (ft)")

    gap = (runs[0][1], runs[1][0])
    for a in ax:
        a.axhspan(*gap, color="0.9", zorder=0)
        for top, base in runs:
            for zz in (top, base):
                a.axhline(zz, color="m", ls="--", lw=0.8)
        a.grid(alpha=0.3)
        a.legend(fontsize=6.5, loc="lower right" if a is not ax[0] else "lower left")
    ax[0].text(0.5, sum(gap) / 2, "Middle shale:\nnot inverted", ha="center", va="center", fontsize=9)
    ax[0].set(ylim=(ex.BASE, runs[0][0] - 10), ylabel="Measured depth (ft)")
    fig.suptitle("ORM-75: UTAPWeLS compositional joint inversion (Runs 1 and 2) compared with deterministic Juhasz and "
                 "the Python Monte Carlo inversion", fontsize=12)
    fig.tight_layout()
    path = it.OUT / "13_cji_comparison.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    print("saved", path)
    for k, c in cum.items():
        print(f"{k}: {c[-1]:.2f} ft")
    s = cji.v_quartz + cji.v_shale_grains + cji.PHIT
    print(f"volume closure: median {s.median():.4f}, max |1 - sum| {np.abs(1 - s).max():.4f}")


if __name__ == "__main__":
    main()
