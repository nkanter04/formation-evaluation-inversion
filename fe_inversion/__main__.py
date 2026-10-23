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


def write_report(r, out, figures):
    ep, eps, auto = r["endpoints"], r["endpoint_std"], r["endpoints_auto"]
    pr, prs = r["props"], r["props_std"]
    s, df = r["summary"], r["df"]
    L = ["# Homework Project 1: interpretation summary", "",
         f"Interval {df.index[0]:.0f}-{df.index[-1]:.0f} ft MD, "
         f"{r['settings']['realizations']} Monte Carlo realizations, "
         f"run time {r['elapsed']:.0f} s.", "",
         "## End-points (mean ± std used in the Monte Carlo)", "",
         "| Parameter | Value | Std | Source |", "|---|---|---|---|"]
    rel = {"rsh_h", "rsh_v", "rw", "rmf"}
    for k, v in {**ep, **pr}.items():
        std = {**eps, **prs}[k]
        std_txt = f"{100 * std:.0f} %" if k in rel else f"{std:.3g}"
        src = "estimated" if (k in auto or k in ("rw", "rmf", "rho_hc")) else "config"
        L.append(f"| {k} | {v:.4g} | {std_txt} | {src} |")
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

    core = df[df.CORE_POR.notna()]
    if len(core):
        L += ["## Inversion vs. core (P50 at core depths)", "",
              "| Quantity | Bias (inv - core) | RMS difference | Correlation | Plugs |",
              "|---|---|---|---|---|"]
        for inv, col in (("phi", "CORE_POR"), ("sw", "CORE_SW")):
            a, b = s.loc[core.index, f"{inv}_p50"], core[col]
            ok = a.notna() & b.notna()
            dlt = a[ok] - b[ok]
            L.append(f"| {inv} | {dlt.mean():+.3f} | {np.sqrt((dlt ** 2).mean()):.3f} | "
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
