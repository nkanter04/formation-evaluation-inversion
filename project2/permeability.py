"""PGE 385K Homework Project 2, Task 8: core permeability vs. calculated porosity and Swi.

Run:  python project2/permeability.py
Writes figures and permeability_summary.md to the Project 2 "Interpretation" folder.

Rocks are classified first (Leverett, sqrt(k/phi)), because one regression over mixed throat
sizes is not meaningful. Models are compared by leave-one-out prediction, since there are few plugs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explore as ex
import interpret as it
from fe_inversion import petro

P = it.CFG["permeability"]
COLORS = ["#c62828", "#ef8a17", "#2e9e44", "#1f5fa8", "#6a3d9a", "#555555"]


def plug_table():
    """Core plugs with the calculated log properties at each plug depth."""
    df, core = ex.load_logs(), ex.load_core()
    w = df.loc[ex.TOP:ex.BASE]
    sands, vsh, _ = ex.pick_sands(w)
    sands, _ = ex.classify(sands)
    groups = it.water_sand_samples(w, sands, vsh)
    res, _ = it.calculate(w, vsh, groups, it.shale_properties(w))
    for col in ("phi_a", "phi_t", "phi_s", "csh", "sw_a", "sw_j", "sws", "sb"):
        core[col] = np.interp(core.depth_log, res.index, res[col])
    s = P["stress"]
    core["k"], core["kv"], core["phi"] = core[f"kh_{s}"], core[f"kv_{s}"], core[f"phi_{s}"]
    core["censored"] = core[f"kh_{s}_censored"]
    core["group"] = core.sand.str[0]
    core["swi"] = core.sw_j.clip(upper=P["max_sw"])
    return core, res, sands


def assign_classes(core):
    """Equal-count classes of sqrt(k/phi); censored plugs rank at the top, class 1 is the largest throats."""
    core["radius"] = petro.throat_radius_um(core.k, core.phi)
    rank = core.radius.rank(method="first", ascending=False)
    core["rock_class"] = np.ceil(rank / (len(core) / P["n_classes"])).astype(int)
    return core


def loo(x_cols, data):
    """Leave-one-out prediction of log10 k: returns R2, RMSE, predictions, in-sample R2 and coefficients."""
    X = np.column_stack([np.ones(len(data))] + x_cols)
    y = np.log10(data.k.values)
    pred = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        pred[i] = X[i] @ b
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ b
    tss = ((y - y.mean()) ** 2).sum()
    return dict(r2_loo=1 - ((y - pred) ** 2).sum() / tss, rmse=float(np.sqrt(((y - pred) ** 2).mean())),
                r2=1 - ((y - fit) ** 2).sum() / tss, pred=pred, coef=b)


def models(data):
    lphi_c, lphi_t = np.log10(data.phi.values), np.log10(data.phi_t.values)
    lsw = np.log10(data.swi.values)
    lmob = np.log10((1 - data.swi.values) / data.swi.values)
    group = (data.group == "E").astype(float).values
    return {"Core φ only": [lphi_c], "Log φt only": [lphi_t], "Log Sw only": [lsw],
            "Log φt + Sw (Timur form)": [lphi_t, lsw], "Log φt + (1−Sw)/Sw (Coates form)": [lphi_t, lmob],
            "Log φt + Sw + Csh": [lphi_t, lsw, data.csh.values], "Log φt + Sw + sandstone group": [lphi_t, lsw, group]}


def irreducible_check(res, sands):
    """Within each hydrocarbon sandstone, does Sw rise toward the base, as it would in a transition zone?"""
    rows = []
    for _, s in sands[sands.fluid == "possible hydrocarbon"].iterrows():
        z = res.loc[s.top:s.base]
        z = z[z.i_gr < it.CFG["net"]["max_vsh"]]
        third = max(len(z) // 3, 1)
        bvw = z.phi_t * z.sw_j
        rows.append(dict(sandstone=f"{s.top:.0f}–{s.base:.0f}", sw_top=z.sw_j.iloc[:third].mean(), sw_base=z.sw_j.iloc[-third:].mean(),
                         r_sw_depth=np.corrcoef(z.index, z.sw_j)[0, 1], r_sw_phi=np.corrcoef(z.phi_t, z.sw_j)[0, 1], bvw=bvw.median()))
    return pd.DataFrame(rows)


def class_prediction(data):
    """Leave-one-out nearest-centroid class prediction from log properties alone."""
    feats = data[["phi_t", "swi", "csh"]]
    Z = ((feats - feats.mean()) / feats.std()).values
    y = data.rock_class.values
    hit = 0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        cents = {c: Z[keep & (y == c)].mean(axis=0) for c in np.unique(y[keep])}
        hit += min(cents, key=lambda c: ((Z[i] - cents[c]) ** 2).sum()) == y[i]
    return hit / len(y)


def fig_classes(core, data, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    a = ax[0]
    phi = np.linspace(0.08, 0.40, 30)
    bounds = [core.loc[core.rock_class == c, "radius"].min() for c in range(1, P["n_classes"])]
    for r in bounds:
        a.plot(phi, r ** 2 * phi / (8 * petro.MD_TO_UM2), "k:", lw=1)
    for c in range(1, P["n_classes"] + 1):
        z = data[data.rock_class == c]
        a.scatter(z.phi, z.k, s=38, color=COLORS[c - 1], label=f"Class {c} (median R = {z.radius.median():.1f} µm)")
    cens = core[core.censored]
    a.scatter(cens.phi, np.full(len(cens), 9999), marker="^", s=70, color=COLORS[0], edgecolors="k", label="k > 9,999 md (not fitted)")
    a.set(yscale="log", xlim=(0.08, 0.40), ylim=(0.02, 3e4), xlabel="Core porosity (v/v)", ylabel="Core kh (md)",
          title="(a) Leverett classes; dotted lines are constant √(k/φ)")
    for col, a, xl, ttl in [("phi_t", ax[1], "Calculated total porosity φt (v/v)", "(b) Core k vs. calculated porosity"),
                            ("sw_j", ax[2], "Calculated Sw, Juhasz (v/v)", "(c) Core k vs. calculated Sw")]:
        for c in range(1, P["n_classes"] + 1):
            z = data[data.rock_class == c]
            a.scatter(z[col], z.k, s=38, color=COLORS[c - 1], label=f"Class {c}")
        a.set(yscale="log", ylim=(0.02, 3e4), xlabel=xl, ylabel="Core kh (md)", title=ttl)
    for a in ax:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_models(core, data, fits, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    a = ax[0]
    y = data.k.values
    for lab, col, mk in [("Log φt only", "0.55", "o"), ("Log φt + Sw (Timur form)", "#c62828", "s")]:
        a.scatter(10 ** fits[lab]["pred"], y, s=30, marker=mk, color=col, alpha=0.85, label=f"{lab}: LOO R² = {fits[lab]['r2_loo']:.2f}")
    a.plot([0.02, 3e4], [0.02, 3e4], "k--", lw=1, label="1:1")
    a.set(xscale="log", yscale="log", xlim=(0.02, 3e4), ylim=(0.02, 3e4), xlabel="Predicted kh, leave-one-out (md)", ylabel="Core kh (md)",
          title="(a) Prediction from calculated properties")
    a = ax[1]
    z = core[core.kh_800.notna() & core.kh_3500.notna() & ~core.kh_3500_censored]
    a.scatter(z.kh_800, z.kh_3500, s=30, color="#1f5fa8", label="Plugs")
    a.plot([0.02, 2e4], [0.02, 2e4], "k--", lw=1, label="1:1")
    a.set(xscale="log", yscale="log", xlabel="kh at 800 psi (md)", ylabel="kh at 3,500 psi (md)", title="(b) Effect of confining pressure")
    a = ax[2]
    z = core[core.kv.notna() & core.k.notna() & ~core.censored]
    a.scatter(z.k, z.kv, s=30, color="#2e9e44", label="Plugs")
    a.plot([0.02, 2e4], [0.02, 2e4], "k--", lw=1, label="kv = kh")
    a.set(xscale="log", yscale="log", xlabel="kh (md)", ylabel="kv (md)", title="(c) Permeability anisotropy, 3,500 psi")
    for a in ax:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    core, res, sands = plug_table()
    core = assign_classes(core)
    data = core[~core.censored & core.k.notna() & (core.k > 0)].copy()
    fits = {k: loo(v, data) for k, v in models(data).items()}
    irr = irreducible_check(res, sands)
    acc = class_prediction(data)

    # Class table
    rows = []
    for c in range(1, P["n_classes"] + 1):
        z, zc = data[data.rock_class == c], core[core.rock_class == c]
        rows.append(dict(rock_class=c, plugs=len(zc), censored=int(zc.censored.sum()), radius_um=zc.radius.median(), k_min=z.k.min(), k_max=z.k.max(),
                         core_phi=z.phi.mean(), phi_t=z.phi_t.mean(), sw_j=z.sw_j.mean(), csh=z.csh.mean(),
                         sandstones=", ".join(f"{s}:{n}" for s, n in zc.sand.value_counts().sort_index().items())))
    classes = pd.DataFrame(rows)

    # Within-class fit of log k on the independent (calculated) properties, with the caveat in the text
    within = []
    for c in range(1, P["n_classes"] + 1):
        z = data[data.rock_class == c]
        within.append(dict(rock_class=c, n=len(z), r_phi_t=np.corrcoef(np.log10(z.phi_t), np.log10(z.k))[0, 1],
                           r_sw=np.corrcoef(np.log10(z.swi), np.log10(z.k))[0, 1]))
    within = pd.DataFrame(within)

    # Published equations
    pub = []
    y = np.log10(data.k.values)
    for name, fn, args in [("Tixier", petro.timur_k, (250, 3, 2)), ("Timur", petro.timur_k, (100, 2.25, 2)), ("Coates", petro.coates_k, (100, 2, 2))]:
        e = np.log10(fn(data.phi_t.values, data.swi.values, *args)) - y
        pub.append(dict(equation=name, alpha=args[0], beta=args[1], gamma=args[2], bias_log10=e.mean(), rmse_log10=float(np.sqrt((e ** 2).mean()))))
    pub = pd.DataFrame(pub)

    both = core[core.kh_800.notna() & core.kh_3500.notna() & ~core.kh_3500_censored]
    stress_ratio = (both.kh_3500 / both.kh_800)
    aniso = data[data.kv.notna()]
    kvkh = aniso.kv / aniso.k
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    lk = np.log10(data.k)
    good = data[data.phi >= P["good_rock_phi"]]        # selected on porosity, not on k, so the test is not circular

    fig_classes(core, data, it.OUT / "06_permeability_classes.png")
    fig_models(core, data, fits, it.OUT / "07_permeability_models.png")

    tbl = pd.DataFrame([dict(model=k, r2_fit=v["r2"], r2_loo=v["r2_loo"], rmse_log10=v["rmse"],
                             coefficients=", ".join(f"{c:.2f}" for c in v["coef"])) for k, v in fits.items()])
    L = ["# Project 2, Task 8: permeability", "",
         f"{len(core)} plugs, {int(core.censored.sum())} reported as kh > 9,999 md and left out of fits ({len(data)} fitted). "
         f"Core data at {P['stress']} psi confining pressure. Calculated properties are read at each plug's log depth.", "",
         "## Core observations", "",
         f"- Stress: kh at 3,500 psi is a median {stress_ratio.median():.2f} of kh at 800 psi (range {stress_ratio.min():.2f}–{stress_ratio.max():.2f}); r between log kh(800) and this ratio is {corr(np.log10(both.kh_800), stress_ratio):.2f}, "
         f"and the {int((both.kh_800 < 20).sum())} plugs below 20 md keep a median of only {stress_ratio[both.kh_800 < 20].median():.2f}.",
         f"- Direction: kv/kh has a median of {kvkh.median():.2f} (range {kvkh.min():.2f}–{kvkh.max():.2f}, {len(kvkh)} plugs). All values below are horizontal permeability.", "",
         "## Rock classes (Leverett, equal-count fairways of √(k/φ))", "",
         classes.to_markdown(index=False, floatfmt=("", "", "", ".1f", ".3g", ".4g", ".3f", ".3f", ".3f", ".3f", "")), "",
         "Classes are defined from core k and φ, so a k–φ fit inside a class is partly circular: a narrow range of √(k/φ) forces k to follow φ. "
         "Within-class correlations of log k are therefore shown only for the calculated log properties, which are independent of the classification:", "",
         within.to_markdown(index=False, floatfmt=(".0f", ".0f", ".2f", ".2f")), "",
         "With 6–9 plugs per class these correlations are only indicative, and their signs are not consistent between classes.", "",
         f"Predicting a plug's class from log φt, Sw and Csh alone (leave-one-out, nearest centroid) is right {acc*100:.0f} % of the time, against {100/P['n_classes']:.0f} % by chance.", "",
         "## Does permeability correlate with calculated porosity and Swi?", "",
         tbl.to_markdown(index=False, floatfmt=("", ".2f", ".2f", ".2f", "")), "",
         "Coefficients are the intercept, then each input in order; all are base-10 logs and k is in md. Models are ranked by leave-one-out R², since in-sample R² always rises with more inputs.", "",
         f"Single correlations of log10 k: core φ r = {corr(np.log10(data.phi), lk):.2f}; log φt r = {corr(np.log10(data.phi_t), lk):.2f}; "
         f"log Sw r = {corr(np.log10(data.swi), lk):.2f}; core Sw r = {corr(np.log10(data.sw.clip(lower=.01)), lk):.2f}; Csh r = {corr(data.csh, lk):.2f}.", "",
         f"Among the {len(good)} plugs with core porosity of at least {P['good_rock_phi']} (selected on porosity, not k, to avoid a circular test), the correlations are: core φ r = "
         f"{corr(np.log10(good.phi), np.log10(good.k)):.2f}; log φt r = {corr(np.log10(good.phi_t), np.log10(good.k)):.2f}; "
         f"log Sw r = {corr(np.log10(good.swi), np.log10(good.k)):.2f}. Log k spans {np.log10(good.k).max() - np.log10(good.k).min():.1f} decades in this subset "
         f"against {lk.max() - lk.min():.1f} in all plugs.", "",
         "By sandstone group, log Sw vs. log k: " + "; ".join(
             f"{g} sandstones r = {corr(np.log10(z.swi), np.log10(z.k)):.2f} ({len(z)} plugs)" for g, z in data.groupby("group")) + ".", "",
         "## Is the calculated Sw irreducible?", "",
         irr.to_markdown(index=False, floatfmt=("", ".2f", ".2f", ".2f", ".2f", ".3f")), "",
         "Sw rising toward the base of a sandstone is consistent with a transition zone, but this test is not conclusive either way: each rock class has its own capillary-pressure curve, so a flat Sw profile does not prove irreducible saturation, and Sw also changes with rock quality. The calculated Sw is therefore an estimate of Swi, not a measurement of it. There are no pressure or capillary-pressure data here.", "",
         "## Published equations with calculated φt and Sw", "",
         pub.to_markdown(index=False, floatfmt=("", "", "", "", "+.2f", ".2f")), ""]
    (it.OUT / "permeability_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
