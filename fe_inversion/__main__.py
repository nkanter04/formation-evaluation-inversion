"""Run the interpretation: python -m fe_inversion --config config/project1.toml"""
import argparse
import sys

import numpy as np
import pandas as pd

from . import io, plotting, workflow


def write_outputs(r, out):
    df, conv, s = r["df"], r["conv"], r["summary"]
    table = pd.concat([df, conv, s, r["flags"].astype(int)], axis=1)
    table.to_csv(out / "results.csv", float_format="%.5g")

    las = r["las"]
    keep = ["csh_p10", "csh_p50", "csh_p90", "phi_p10", "phi_p50", "phi_p90",
            "sw_p10", "sw_p50", "sw_p90", "sxo_p50", "rss_p50", "hpv_p10", "hpv_p50",
            "hpv_p90", "misfit_p50"]
    depth = las.index.astype(float)
    for col in list(conv.columns) + keep:
        src = conv[col] if col in conv else s[col]
        las.append_curve(col.upper(), src.reindex(depth).values,
                         descr="fe_inversion output (see results.csv)")
    las.write(str(out / "results.las"), version=2.0)


def _calibration_section(r):
    """Per-section shale bracketing, spectral-GR cross-check and Rv-Rh trends."""
    sets, df, centers = r["shale_sets"], r["df"], r["centers"]
    s = r["settings"]
    spec = s.get("csh_spectral_curve", "GR")
    d = df.index.values
    L = ["", f"## Section calibration (inversion method: {s.get('method', 'sequential')})", "",
         f"Bed boundaries: {len(r['bed_idx']) - 1} beds; {centers.sum()} thick-bed center samples "
         f"({centers.mean():.0%} of the interval) used for calibration and cross-checks. "
         f"In each section Rsh_h and Rsh_v are bracketed around the thickest shale bed's readings; "
         f"the chosen pair best matches the Rh-Rv inversion Csh to the {spec} Csh while the shale "
         f"bed itself still reads Csh ~ 1.", "",
         "| Section | Shale bed (ft) | Rsh_h start -> chosen | Rsh_v start -> chosen | "
         f"Shale bed Csh | {spec} sand / shale | Sand / shale points | RMS Csh difference | Correlation |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sec in sets:
        b = sec["bracket"]
        L.append(f"| {sec['name']} {sec['top']:.0f}-{sec['bottom']:.0f} | {sec['reference'][0]:.1f}-"
                 f"{sec['reference'][1]:.1f} ({sec['reference'][1] - sec['reference'][0]:.1f} ft) | "
                 f"{b['rsh_h0']:.3f} -> {b['rsh_h']:.3f} | {b['rsh_v0']:.3f} -> {b['rsh_v']:.3f} | "
                 f"{b['ref_csh']:.2f} | {b['spec_sand']:.2f} / {b['spec_shale']:.2f} | "
                 f"{b['n_sand']} / {b['n_shale']} | {b['rms']:.3f} | {b['corr']:.2f} |")
    names = list(sets[0]["bracket"]["spectral"])
    L += ["", "Agreement of the inversion Csh with Csh from each gamma-ray curve (thick-bed centers; "
          "end-points read where the inversion gives Csh ~ 0 and ~ 1). GR includes uranium; "
          "THOR, POTA and CGR do not.", "",
          "| Section | " + " | ".join(f"{n} RMS / r" for n in names) + " |",
          "|---" * (len(names) + 1) + "|"]
    for sec in sets:
        sp = sec["bracket"]["spectral"]
        L.append(f"| {sec['name']} | " + " | ".join(f"{sp[n]['rms']:.3f} / {sp[n]['corr']:.2f}" for n in names) + " |")
    L += ["", "Rv-Rh trend (slope of log10 Rv against log10 Rh): all samples vs. thick-bed centers only.", "",
          "| Section | Slope, all samples | Slope, thick-bed centers | Samples / centers |", "|---|---|---|---|"]
    for sec in sets:
        m = (d >= sec["top"]) & (d < sec["bottom"]) & (df.RH > 0).values & (df.RV > 0).values
        mc = m & centers
        slope = lambda k: np.polyfit(np.log10(df.RH[k]), np.log10(df.RV[k]), 1)[0] if k.sum() > 10 else np.nan
        L.append(f"| {sec['name']} | {slope(m):.2f} | {slope(mc):.2f} | {m.sum()} / {mc.sum()} |")
    return L


def write_report(r, out, figures):
    ep, eps, sets = r["endpoints"], r["endpoint_std"], r["shale_sets"]
    pr, prs = r["props"], r["props_std"]
    s, df = r["summary"], r["df"]
    L = ["# Homework Project 1: interpretation summary", "",
         f"Interval {df.index[0]:.0f}-{df.index[-1]:.0f} ft MD, "
         f"{r['settings']['realizations']} Monte Carlo realizations, "
         f"run time {r['elapsed']:.0f} s.", "",
         "## Shale end-points by zone (mean; std applies to every zone)", "",
         "| Parameter | " + " | ".join(
             f"{s['name']}<br>{s['top']:.0f}-{s['bottom']:.0f} ft, ref. {s['reference'][0]:.0f}-"
             f"{s['reference'][1]:.0f} ft" for s in sets) + " | Std |",
         "|---" * (len(sets) + 2) + "|"]
    rel = {"rsh_h", "rsh_v", "rw", "rmf"}
    std_txt = lambda k, sd: f"{100 * sd:.0f} %" if k in rel else f"{sd:.3g}"
    for k in workflow.ZONED:
        L.append(f"| {k} | " + " | ".join(f"{s['used'][k]:.4g}" for s in sets)
                 + f" | {std_txt(k, eps[k])} |")
    L += _calibration_section(r)
    L += ["", "## Other end-points and properties", "",
          "| Parameter | Value | Std | Source |", "|---|---|---|---|"]
    for k, v in {**{k: v for k, v in ep.items() if k not in workflow.ZONED}, **pr}.items():
        sd = {**eps, **prs}[k]
        src = "estimated" if k in ("gr_sand", "rw", "rmf", "rho_hc") else "config"
        L.append(f"| {k} | {v:.4g} | {std_txt(k, sd)} | {src} |")
    w = r["water"]
    L += ["", f"Rw and Rmf come from the water sand ({w['n']} samples, Sw = Sxo = 1). "
          f"Rw = {w['rw']:.4f} ohm.m (log spread {w['rw_log_spread']:.2f}), "
          f"Rmf = {w['rmf']:.4f} ohm.m (log spread {w['rmf_log_spread']:.2f}).", ""]

    p = r.get("pressure")
    if p:
        L += ["## Pressures (gradients in TVD)", ""]
        for key, name in (("hc", "Hydrocarbon"), ("water", "Water")):
            if key in p:
                g = p[key]
                extra = " (assumed brine density through 1 point)" if g.get("assumed") else \
                    f" ± {g['density_se']:.3f}, {g['n']} points"
                L.append(f"- {name}: {g['gradient']:.4f} psi/ft → {g['density']:.3f} g/cc{extra}")
        if "fwl_md" in p:
            L.append(f"- Free-water level ≈ {p['fwl_md']:.0f} ft MD")
        L.append("")
    rho = r["core"].RHO_F_CORE
    rho = rho[(rho > -0.5) & (rho < 1.5)]
    L += [f"Core-calibrated pore-fluid density: median {rho.median():.2f} g/cc, "
          f"IQR {rho.quantile(0.25):.2f}-{rho.quantile(0.75):.2f} ({len(rho)} plugs).", "",
          "## Shale consistency", "", r["shale_check"].round(3).to_markdown(), ""]

    u = r.get("uranium")
    if u:
        conv = r["conv"]
        L += ["## Uranium: where the GR is unreliable for Csh", "",
              f"Background uranium {u['u_background']:.2f} ppm (MAD {u['u_mad']:.2f}). "
              f"Abnormal above {u['u_max']:.2f} ppm. The uranium effect is Csh from GR minus Csh "
              f"from the uranium-free CGR (Th + K), flagged above {u['max_csh_inflation']:.2f}.", "",
              "| Flag | Interval (ft MD) | Thickness (ft) | Max U (ppm) | Max uranium effect on Csh |",
              "|---|---|---|---|---|"]
        for kind, label in (("anomaly", "Abnormal uranium"), ("inflation", "Uranium inflates GR Csh")):
            for a, b in u["intervals"][kind]:
                m = (df.index >= a) & (df.index < b)
                L.append(f"| {label} | {a:.1f}-{b:.1f} | {b - a:.1f} | {df.URAN[m].max():.1f} | "
                         f"{conv.CSH_U_INFLATION[m].max():+.2f} |")
        n = {k: sum(b - a for a, b in v) for k, v in u["intervals"].items()}
        L += ["", f"Total flagged: {n['anomaly']:.1f} ft abnormal uranium, "
              f"{n['inflation']:.1f} ft where uranium inflates the GR Csh.", ""]

    core = df[df.CORE_POR.notna()]
    if len(core):
        max_rhog = r["cfg"]["zones"].get("representative_plug_max_rhog", 9.9)
        L += ["## Inversion vs. core (P50 at core depths)", "",
              f"Representative plugs have grain density ≤ {max_rhog:.2f} g/cc. Heavier plugs "
              "likely contain clay or mica, so they are not clean sand laminae and are not "
              "a like-for-like test of the sand-lamina porosity.", "",
              "| Quantity | Plugs | Bias (inv - core) | RMS difference | Correlation | n |",
              "|---|---|---|---|---|---|"]
        groups = (("all", core.CORE_RHOG.notna() | True),
                  ("representative", core.CORE_RHOG <= max_rhog),
                  ("clay-bearing", core.CORE_RHOG > max_rhog))
        for inv, col in (("phi", "CORE_POR"), ("sw", "CORE_SW")):
            for gname, sel in groups:
                a, b = s.loc[core.index, f"{inv}_p50"][sel], core[col][sel]
                ok = a.notna() & b.notna()
                if ok.sum() < 3:
                    continue
                dlt = a[ok] - b[ok]
                L.append(f"| {inv} | {gname} | {dlt.mean():+.3f} | {np.sqrt((dlt ** 2).mean()):.3f} | "
                         f"{np.corrcoef(a[ok], b[ok])[0, 1]:.2f} | {ok.sum()} |")
        L.append("")

    L += ["## Zone averages (P50 of depth averages over sand, Csh < shale mask)", "",
          "| Zone | Csh | phi_ss | Sw_ss | Sxo | HC column P10 / P50 / P90 (ft TVD) |",
          "|---|---|---|---|---|---|"]
    zones = workflow.zone_intervals(r["cfg"], df.index)
    for name, (a, b) in zones.items():
        m = (df.index >= a) & (df.index < b)
        hc = np.percentile(r["hc_column"][name], [10, 50, 90])
        L.append(f"| {name} | {np.nanmean(s.csh_p50[m]):.2f} | {np.nanmean(s.phi_p50[m]):.2f} | "
                 f"{np.nanmean(s.sw_p50[m]):.2f} | {np.nanmean(s.sxo_p50[m]):.2f} | "
                 f"{hc[0]:.1f} / {hc[1]:.1f} / {hc[2]:.1f} |")
    tot = np.percentile(r["hc_column"]["Whole interval"], [10, 50, 90])
    L += [f"| **Whole interval** | | | | | **{tot[0]:.1f} / {tot[1]:.1f} / {tot[2]:.1f}** |", "",
          "## Figures", ""] + [f"- `{f}`" for f in figures]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/project1.toml")
    ap.add_argument("--realizations", type=int, help="override the number of Monte Carlo realizations")
    ap.add_argument("--output", help="override the output directory")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    cfg = io.load_config(args.config)
    out = io.ensure_dir(args.output or cfg["data"]["output_dir"])

    def progress(done, total):
        print(f"\r  Monte Carlo: {done}/{total} realizations", end="", flush=True)

    print(f"Running {args.config} ...")
    r = workflow.run(cfg, progress=progress, realizations=args.realizations)
    print(f"\n  done in {r['elapsed']:.0f} s. Writing outputs to {out}")
    figures = plotting.all_figures(r, out)
    write_outputs(r, out)
    write_report(r, out, figures)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
