"""Point-by-point HPV by the PGE 358 HW 6 laminated-shale method.

Run: python -m fe_inversion.manual_hpv --config config/project1.toml

Part A (once, in the aquifer, Sw = 1): Csh from the neutron-density difference,
shale-corrected porosity, sandstone resistivity from the parallel model, then Rw.
Part B (every depth): Csh from gamma ray anchored at the aquifer, shale-corrected
porosities, sandstone porosity, sandstone resistivity, Sw, Sh and HPV.
See docs/manual_hpv_method.md for the equations.
"""
import argparse
import sys

import lasio
import numpy as np
import pandas as pd

from . import io, petro


# ---------------------------------------------------------------- equations
def nd_csh(phi_n, phi_d, sh):
    """Csh in the aquifer from the neutron-density difference (A1)."""
    return (phi_n - phi_d) / (sh["phi_n"] - sh["phi_d"])


def corrected_porosities(phi_d, phi_n, csh, sh):
    """Shale-corrected density and neutron porosities of the sandstone (A2, B2)."""
    return ((phi_d - csh * sh["phi_d"]) / (1 - csh),
            (phi_n - csh * sh["phi_n"]) / (1 - csh))


def sand_porosity(phi_dc, phi_nc):
    """RMS average where neutron < density (light fluid), else the plain average (B3)."""
    phi_dc, phi_nc = np.asarray(phi_dc, float), np.asarray(phi_nc, float)
    light = phi_nc < phi_dc
    return np.where(light, np.sqrt((phi_dc ** 2 + phi_nc ** 2) / 2), (phi_dc + phi_nc) / 2), light


def sand_resistivity(rt, csh, rsh):
    """Parallel laminated model solved for Rs (A3, B4). NaN where Rsh <= Csh*Rt (no valid Rs)."""
    rt, csh = np.asarray(rt, float), np.asarray(csh, float)
    den = rsh - csh * rt
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (1 - csh) * rt * rsh / den, np.nan)


def gr_csh(gr, gr_w, gr_sh, csh_w, method="linear"):
    """Csh from gamma ray anchored at the aquifer (B1).

    linear:   Csh = Csh_w + (1 - Csh_w) * I,   I = (GR - GR_w) / (GR_sh - GR_w)
    larionov: I is replaced by the Larionov (Tertiary) relation 0.083 * (2^(3.7 I) - 1).
    I may be negative (rock cleaner than the aquifer, as in HW 6 Problem 1); only the final
    Csh is clipped to 0-1.
    """
    idx = np.minimum((np.asarray(gr, float) - gr_w) / (gr_sh - gr_w), 1.0)
    if method == "larionov":
        idx = petro.csh_larionov_tertiary(idx)
    elif method != "linear":
        raise ValueError(f"unknown csh_method {method!r}")
    return np.clip(csh_w + (1 - csh_w) * idx, 0, 1)


def aquifer(aq, sh, rsh, archie, clip_csh=True):
    """Part A: Csh, sandstone porosity, sandstone resistivity and Rw in the aquifer."""
    csh_w_raw = float(nd_csh(aq["phi_n"], aq["phi_d"], sh))
    csh_w = max(csh_w_raw, 0.0) if clip_csh else csh_w_raw
    phi_dc, phi_nc = corrected_porosities(aq["phi_d"], aq["phi_n"], csh_w, sh)
    phi_w = float((phi_dc + phi_nc) / 2)   # equal in a true water sand; average any small difference
    rs_w = float(sand_resistivity(aq["rt"], csh_w, rsh))
    rw = rs_w * phi_w ** archie["m"] / archie["a"]
    return dict(csh_w_raw=csh_w_raw, csh_w=csh_w, phi_dc=float(phi_dc), phi_nc=float(phi_nc),
                phi_w=phi_w, rs_w=rs_w, rw=rw)


def fluid_density(phi_d, csh, phi_d_sh, phi_s, rho_ma=2.65, rho_f_tool=1.0):
    """Pore-fluid density of the sandstone (HW 6 part h).

    rho_R  = phi_D + (1 - phi_D) rho_ma             bulk density behind PHID (tool fluid 1.0 g/cc)
    rho_sh = phi_D,sh + (1 - phi_D,sh) rho_ma       shale density, same way
    rho_s  = (rho_R - Csh rho_sh) / (1 - Csh)       sandstone density (laminated)
    rho_f  = (rho_s - (1 - phi_s) rho_ma) / phi_s   fluid filling the sandstone pores
    """
    rho_r = np.asarray(phi_d, float) * rho_f_tool + (1 - np.asarray(phi_d, float)) * rho_ma
    rho_sh = phi_d_sh * rho_f_tool + (1 - phi_d_sh) * rho_ma
    rho_s = (rho_r - csh * rho_sh) / (1 - csh)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (rho_s - (1 - phi_s) * rho_ma) / phi_s


def hydrocarbon_density(rho_f, sw, rho_w):
    """rho_f = Sw rho_w + (1 - Sw) rho_H solved for rho_H (HW 6 part i)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(rho_f, float) - rho_w * np.asarray(sw, float)) / (1 - np.asarray(sw, float))


def archie_sw(phi, rs, rw, archie):
    """Sw from Archie in the sandstone, capped at 1 (B5). NaN where Rs is undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sw = (archie["a"] * rw / (np.asarray(phi, float) ** archie["m"] * np.asarray(rs, float))) ** (1 / archie["n"])
    return np.clip(sw, 0, 1)


def oil_corrected_porosity(phi_dc, rs, rw, archie, fluid, iterations=8):
    """Sandstone porosity from the shale-corrected density, with the actual pore fluid.

    PHID assumes a 1.0 g/cc fluid. The sandstone density it implies is
    rho_s = rho_ma - phi_dc * (rho_ma - 1.0); re-solving with
    rho_f = Sw * rho_w + (1 - Sw) * rho_hc (HW 6 part h, in reverse) gives
    phi_s = (rho_ma - rho_s) / (rho_ma - rho_f). Sw comes from Archie with that
    phi_s, so the two are iterated. Where Rs is undefined, Sw = 1 is used.
    """
    rho_ma, rho_w, rho_hc = fluid["rho_ma"], fluid["rho_w"], fluid["rho_hc"]
    rho_s = rho_ma - np.asarray(phi_dc, float) * (rho_ma - 1.0)
    phi = np.asarray(phi_dc, float).copy()
    sw = np.ones_like(phi)
    for _ in range(iterations):
        sw = np.where(np.isfinite(rs), np.nan_to_num(archie_sw(phi, rs, rw, archie), nan=1.0), 1.0)
        rho_f = sw * rho_w + (1 - sw) * rho_hc
        phi = (rho_ma - rho_s) / (rho_ma - rho_f)
    return phi, rho_f


def point_by_point(logs, aq, sh, rsh, a_res, archie, csh_method="linear", shale_cutoff=0.95,
                   porosity_method="hw6", fluid=None):
    """Part B at every sample. `logs` has GR, PHID, NPHI, RT; returns all intermediate values.

    porosity_method "hw6": RMS of the corrected density and neutron where neutron < density,
    else their average (B3). "oil_density": corrected density re-solved with the pore-fluid
    density, iterated with Sw (see oil_corrected_porosity); needs `fluid`.
    """
    out = pd.DataFrame(index=logs.index)
    out["GR"], out["PHID"], out["NPHI"], out["RT"] = logs.GR, logs.PHID, logs.NPHI, logs.RT
    out["CSH"] = gr_csh(logs.GR, aq["gr"], sh["gr"], a_res["csh_w"], csh_method)
    out["PHID_C"], out["PHIN_C"] = corrected_porosities(logs.PHID, logs.NPHI, out.CSH, sh)
    out["RS"] = sand_resistivity(logs.RT, out.CSH, rsh)
    phi_hw6, out["LIGHT_FLUID"] = sand_porosity(out.PHID_C, out.PHIN_C)
    if porosity_method == "hw6":
        out["PHI_S"] = phi_hw6
    elif porosity_method == "oil_density":
        out["PHI_S"], out["RHO_F"] = oil_corrected_porosity(out.PHID_C.values, out.RS.values,
                                                            a_res["rw"], archie, fluid)
    else:
        raise ValueError(f"unknown porosity_method {porosity_method!r}")
    out["SW"] = archie_sw(out.PHI_S, out.RS, a_res["rw"], archie)
    out["SH"] = 1 - out.SW
    # Flags: why a sample has no HPV. Pure shale first, then the two equation failures.
    flag = np.full(len(out), "", dtype=object)
    flag[(out.PHI_S <= 0).values] = "phi_s <= 0"
    flag[out.RS.isna().values] = "Rs undefined (Rsh <= Csh*Rt)"
    flag[(out.CSH >= shale_cutoff).values] = "shale"
    flag[logs[["GR", "PHID", "NPHI", "RT"]].isna().any(axis=1).values] = "missing log"
    out["FLAG"] = flag
    valid = out.FLAG == ""
    out["HPV"] = np.where(valid, (1 - out.CSH) * out.PHI_S * out.SH, 0.0)
    return out


# ---------------------------------------------------------------- workflow
def _median(df, interval, pad=0.0):
    z = df.loc[interval[0] - pad:interval[1] + pad]
    return dict(gr=float(z.GR.median()), phi_d=float(z.PHID.median()),
                phi_n=float(z.NPHI.median()), rt=float(z.RT.median()))


def run(cfg):
    m = cfg["manual_hpv"]
    d = cfg["data"]
    full, _ = io.load_well(d["las"], cfg["curves"])
    raw = lasio.read(d["las"]).df()
    full["PHID"] = raw.loc[full.index, m["density_porosity_curve"]].values
    full["NPHI"] = full["NPHI"] + m["neutron_offset"]
    full["RT"] = full[m["rt_curve"]]
    archie = {k: float(v) for k, v in cfg["archie"].items()}

    sh = _median(full, m["shale_point"])
    rsh = float(m["rsh"]) if m["rsh"] != "shale_point" else sh["rt"]
    aq = _median(full, m["aquifer"])
    a_res = aquifer(aq, sh, rsh, archie, m.get("clip_aquifer_csh", True))

    top, bottom = m["interval"]
    logs = full.loc[top:bottom, ["GR", "PHID", "NPHI", "RT", "SDEV"]]
    fluid = {k: float(m[k]) for k in ("rho_ma", "rho_w", "rho_hc") if k in m}
    res = point_by_point(logs, aq, sh, rsh, a_res, archie, m["csh_method"], m["shale_cutoff"],
                         m.get("porosity_method", "hw6"), fluid)
    res["DTVD"] = np.gradient(logs.index.values) * np.cos(np.radians(logs.SDEV.values))
    res["HC_COLUMN_CUM"] = np.cumsum(res.HPV * res.DTVD)
    return dict(res=res, shale=sh, rsh=rsh, aquifer=aq, a_res=a_res, archie=archie, cfg=cfg)


def write_outputs(r, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, m = r["res"], r["cfg"]["manual_hpv"]
    out = io.ensure_dir(out_dir)
    res.to_csv(out / "manual_hpv.csv", float_format="%.5g")

    d = res.index.values
    fig, ax = plt.subplots(1, 6, figsize=(16, 11), sharey=True, gridspec_kw={"wspace": 0.08})
    for a in ax:
        a.grid(True, color="#DDE3EA", lw=0.6)
        a.xaxis.set_label_position("top")
        a.xaxis.tick_top()
    ok = res.FLAG == ""
    bad_rs = res.FLAG.str.startswith("Rs")
    ax[0].plot(res.GR, d, color="#2E7D32", lw=0.8)
    ax[0].set(xlim=(0, 150), xlabel="GR (gAPI)")
    ax[1].plot(res.CSH, d, color="#455A64", lw=0.9)
    ax[1].set(xlim=(0, 1), xlabel=f"Csh ({m['csh_method']} GR)")
    ax[2].plot(res.PHID_C, d, color="#C62828", lw=0.7, label="Density, corrected")
    ax[2].plot(res.PHIN_C, d, color="#1565C0", lw=0.7, label="Neutron, corrected")
    ax[2].plot(res.PHI_S.where(ok), d, color="k", lw=1.0, label=f"Sandstone porosity ({m.get('porosity_method', 'hw6')})")
    ax[2].set(xlim=(0, 0.5), xlabel="Porosity (v/v)")
    ax[2].legend(loc="lower left", fontsize=7)
    ax[3].plot(res.RT, d, color="#8E24AA", lw=0.7, label="Rt (parallel)")
    ax[3].plot(res.RS, d, color="k", lw=0.9, label="Rs (sandstone)")
    ax[3].set(xscale="log", xlim=(0.1, 200), xlabel="Resistivity (ohm.m)")
    ax[3].legend(loc="lower left", fontsize=7)
    ax[4].plot(res.SW.where(ok), d, color="#1565C0", lw=0.9)
    for b in (0, 1):
        ax[4].axvline(b, color="#455A64", ls="--", lw=0.8)
    ax[4].set(xlim=(-0.03, 1.05), xlabel="Sw (v/v)")
    ax[5].fill_betweenx(d, 0, res.HPV, color="#A5D6A7", lw=0)
    ax[5].plot(res.HPV, d, color="#2E7D32", lw=0.8)
    ax[5].set(xlim=(0, 0.3), xlabel="HPV = (1-Csh) phi_s Sh")
    for a in ax:
        a.fill_betweenx(d, *a.get_xlim(), where=bad_rs.values, color="#F8BBD0", alpha=0.5, lw=0)
    ax[0].set_ylim(d.max(), d.min())
    ax[0].set_ylabel("Measured depth (ft)")
    fig.suptitle(f"HW 6 point-by-point HPV, {d.min():.0f}-{d.max():.0f} ft "
                 f"(pink: Rs undefined). Hydrocarbon column {res.HC_COLUMN_CUM.iloc[-1]:.1f} ft TVD",
                 fontsize=12, fontweight="semibold")
    fig.savefig(out / "manual_hpv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    a, sh, aq, ar = r["archie"], r["shale"], r["aquifer"], r["a_res"]
    top, bottom = m["interval"]
    flags = res.FLAG.replace("", "valid").value_counts()
    valid = res.FLAG == ""
    L = ["# HW 6 point-by-point HPV", "",
         f"Interval {top:.0f}-{bottom:.0f} ft MD, {len(res)} samples. Archie a = {a['a']}, m = {a['m']}, "
         f"n = {a['n']}. Csh method: {m['csh_method']} GR. Porosity method: {m.get('porosity_method', 'hw6')}. Neutron offset added to "
         f"TNPH: {m['neutron_offset']:+.3f}. Rt = {m['rt_curve']}.", "",
         "## Inputs", "",
         "| Point | Depth (ft) | GR | φD | φN | Rt (Ω·m) |", "|---|---|---|---|---|---|",
         f"| Pure shale | {m['shale_point'][0]:.0f}-{m['shale_point'][1]:.0f} | {sh['gr']:.1f} | "
         f"{sh['phi_d']:.3f} | {sh['phi_n']:.3f} | {sh['rt']:.3f} |",
         f"| Aquifer | {m['aquifer'][0]:.0f}-{m['aquifer'][1]:.0f} | {aq['gr']:.1f} | "
         f"{aq['phi_d']:.3f} | {aq['phi_n']:.3f} | {aq['rt']:.3f} |", "",
         f"Shale resistivity used in the parallel model: Rsh = {r['rsh']:.3f} Ω·m.", "",
         "## Part A: aquifer", "",
         f"- Csh,w = (φN − φD)/(φN,sh − φD,sh) = {ar['csh_w_raw']:.3f}"
         + (f" → clipped to {ar['csh_w']:.3f} (Sw = 1 assumed)" if ar["csh_w"] != ar["csh_w_raw"] else ""),
         f"- Corrected porosities: φD(c) = {ar['phi_dc']:.4f}, φN(c) = {ar['phi_nc']:.4f}, "
         f"φs,w = {ar['phi_w']:.4f}",
         f"- Rs,w = {ar['rs_w']:.4f} Ω·m",
         f"- **Rw = Rs,w · φs,w^m / a = {ar['rw']:.5f} Ω·m**", "",
         "## Part B: results", "",
         f"- Hydrocarbon column, sum of HPV × ΔTVD: **{res.HC_COLUMN_CUM.iloc[-1]:.2f} ft TVD**",
         f"- Valid samples: {valid.sum()} of {len(res)} ({valid.mean():.0%})",
         "- Sample status: " + ", ".join(f"{k}: {v}" for k, v in flags.items()),
         f"- Means over valid samples: Csh {res.CSH[valid].mean():.3f}, φs {res.PHI_S[valid].mean():.3f}, "
         f"Sw {res.SW[valid].mean():.3f}, HPV {res.HPV[valid].mean():.4f}", "",
         "## Every 5 ft", "",
         "| Depth (ft) | GR | Csh | φD(c) | φN(c) | φs | Rt | Rs | Sw | Sh | HPV | Note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dep in np.arange(top, bottom + 0.01, 5.0):
        z = res.loc[dep]
        f = lambda v, p=3: "–" if pd.isna(v) else f"{v:.{p}f}"
        L.append(f"| {dep:.0f} | {z.GR:.1f} | {f(z.CSH)} | {f(z.PHID_C)} | {f(z.PHIN_C)} | {f(z.PHI_S)} | "
                 f"{f(z.RT, 2)} | {f(z.RS, 2)} | {f(z.SW)} | {f(z.SH)} | {f(z.HPV, 4)} | {z.FLAG} |")
    L += ["", "All samples (every 0.25 ft) with every intermediate value are in `manual_hpv.csv`."]
    (out / "manual_hpv.md").write_text("\n".join(L), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/project1.toml")
    ap.add_argument("--output", help="output directory (default: the config's output_dir)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = io.load_config(args.config)
    r = run(cfg)
    out = write_outputs(r, args.output or cfg["data"]["output_dir"])
    print((out / "manual_hpv.md").read_text(encoding="utf-8").split("## Every 5 ft")[0])
    print(f"Wrote {out / 'manual_hpv.csv'}, manual_hpv.png and manual_hpv.md")


if __name__ == "__main__":
    sys.exit(main())
