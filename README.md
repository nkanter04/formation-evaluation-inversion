# Formation Evaluation Inversion

Python program for interpreting well logs in laminated shaly sandstones (PGE 385K
Homework Project 1). It loads LAS logs, core and formation-pressure data, runs the
conventional interpretation, and then a **Monte Carlo inversion** that estimates shale
concentration, sand porosity and sand water saturation with uncertainty bands.

## Run

```bash
pip install -r requirements.txt
python -m fe_inversion --config config/project1.toml
```

The PGE 358 HW 6 point-by-point HPV calculation runs separately (see
[docs/manual_hpv_method.md](docs/manual_hpv_method.md)):

```bash
python -m fe_inversion.manual_hpv
```

The remaining Project 1 checks (core depth shift, hydrocarbon density from the logs, Sw-Sxo
consistency, permeability vs. log porosity) read the main program's `results.csv`, so run
them after it:

```bash
python -m fe_inversion.project_checks
```

Options: `--realizations N` changes the number of Monte Carlo realizations, and
`--output DIR` changes where results go (default `outputs/project1`). 1,000 realizations
take about a minute on 16 cores when plugged in; on battery Windows throttles the CPU and
it can take 15+ minutes. Set `workers = 1` in the config to run serially.

Outputs:

| File | Contents |
|---|---|
| `01_log_composite.png` | GR/caliper, resistivities incl. Rh/Rv, density-neutron with crossover and core porosity, PEF, K/Th, core k/Sw/So |
| `02_crossplots.png` | Rt-GR, neutron-density, K-Th by section, GR-RHOB, Rv-Rh with laminated-model grid, Thomas-Stieber |
| `03_csh_methods.png` | Uranium track, Csh from GR / Larionov / CGR / neutron-density / Rh-Rv closed form, and the inversion |
| `04_inversion.png` | Csh, sand porosity, Sw, Sxo, Rss and HPV with P10-P90 bands, plus data misfit |
| `05_data_fit.png` | Measured vs. predicted Rh, Rv, RHOB, NPHI |
| `06_pressure_fluid_density.png` | Pressure gradients in TVD, free-water level, core-calibrated fluid density |
| `07_core_comparison.png` | Core k-phi, k vs. Sw, inversion vs. core porosity and Sw |
| `08_monte_carlo.png` | Hydrocarbon-column distributions and which inputs drive the uncertainty |
| `09_section_calibration.png` | Per section: shale-resistivity bracketing, inversion vs. spectral-GR Csh, Rv-Rh trends |
| `10_uncertainty_vs_csh.png` | Width of the P10-P90 bands against Csh |
| `11_core_depth_shift.png` | Applied core shift and core-vs-log gamma-ray correlation (project_checks) |
| `12_hydrocarbon_density.png` | Hydrocarbon density from the logs (HW 6 h-i) vs. the pressure gradient (project_checks) |
| `13_sw_sxo_consistency.png` | Sxo vs. Sw (ratio method), bulk volume water, Buckles plot (project_checks) |
| `14_permeability.png` | Core permeability vs. log porosity and Sw (project_checks) |
| `results.csv`, `results.las` | All input curves, conventional results, and P10/P50/P90 of every output |
| `summary.md` | Section calibration, end-points, Rw/Rmf, gradients, zone averages, HC column P10/P50/P90 |

## Method

**Laminated model.** Sand and shale layers are in parallel for the resistivity measured
along bedding and in series for the resistivity measured across bedding, and the sand
follows Archie:

    Rv = Csh*Rsh_v + (1 - Csh)*Rss        1/Rh = Csh/Rsh_h + (1 - Csh)/Rss
    Rss = Rw * a / (phi^m * Sw^n)          (a = 1, m = 2.3, n = 1.97)

The sand laminae are taken as isotropic (one Rss for both directions).

**Sections.** The interval is bracketed into three sections by gamma-ray trend:
9,600-9,825 and 9,825-10,080 ft are fining-upward cycles, and 10,080-10,200 ft has the
opposite trend. Each section has its own shale properties.

**Beds and thick-bed centers.** Bed boundaries are picked where log10(Rv) or thorium
changes sharply. Samples at least 1 ft inside beds 3 ft or thicker are "thick-bed
centers"; readings nearer boundaries average both beds, so calibration and cross-checks
use only the centers. The first and last 4 ft of the Rh/Rv log are blanked (tool start-up).

**Shale bracketing (per section).**
1. The thickest continuous shale (high thorium, Rh/Rv present) is the pure-shale reference.
2. Rsh_h and Rsh_v are scanned from 0.6x to 1.6x that bed's readings. For each pair, Rh and
   Rv are inverted for Csh and Rss at thick-bed centers, with no GR input.
3. The thorium values where the inversion gives Csh ~ 0 and Csh ~ 1 become the sand and
   shale end-points, and Csh = (Th - Th_sand) / (Th_shale - Th_sand).
4. The pair kept is the one whose inversion Csh best matches the thorium Csh, while the
   reference shale still reads Csh ~ 1 and its own Rh, Rv stay within noise of (Rsh_h, Rsh_v).

Thorium is used instead of total GR because uranium is not tied to clay. POTA, CGR or GR
can be chosen with `csh_spectral_curve`; the report compares all four.

**Two-step inversion** (`method = "sequential"`, default), depth by depth:
1. Csh and Rss from Rh and Rv by bounded least squares (0 <= Csh <= 1, which avoids the
   blow-up of the closed form as Csh -> 1), biased toward the thorium Csh. Rv is weighted
   as noisier than Rh (10 % vs. 5 %).
2. With Csh fixed, sand porosity and Sw (0 < Sw <= 1) from density, neutron and the step-1
   Rss through Archie.

`method = "joint"` solves Csh, porosity and Sw together from Rh, Rv, RHOB and NPHI instead.
The solver is a batched Levenberg-Marquardt with analytic Jacobians and bounded parameters.

**Monte Carlo.** Each realization draws new shale and sand end-points, matrix and fluid
densities, hydrocarbon hydrogen index, Rw and Rmf from their uncertainties, adds
measurement noise to the logs, perturbs the thorium Csh bias, and re-solves (randomized
maximum likelihood). The spread of the solutions gives the P10/P50/P90 bands and the
distribution of the hydrocarbon column. Uncertainty grows toward shale-rich depths.

**Other calibration from the data** (all can be overridden in the config):
- Rw and Rmf: from the water sand, with Sw = Sxo = 1 in the laminated model.
- Hydrocarbon density: oil gradient from formation-tester points, converted to TVD with
  the minimum-curvature trajectory (the well is about 50° deviated).

## Caveats

- Non-uniqueness: several (Csh, Rss) and (Rsh_h, Rsh_v) combinations can fit the data.
  The bracketing picks the most consistent one; perfect agreement with the GR is not expected.
- Sand properties are blank where Csh > 0.95: they are not resolvable in pure shale.
- The bands are conditional on the model (laminated sand-shale, Archie, linear density and
  neutron mixing) and on the stated input uncertainties.
- Sxo uses AT10 as if it measured resistivity parallel to bedding. In a 50° well through
  anisotropic laminations this is an approximation.
- Plugs with grain density above 2.70 g/cc likely contain clay or mica and are not clean
  sand laminae. The report compares the inversion with all plugs and with representative
  plugs separately. Core Sw is much higher than log Sw (core plugs look flushed: core So is
  low). See `summary.md`.

## Layout

```
config/project1.toml        data paths, curve names, sections, uncertainties, inversion settings
fe_inversion/io.py          LAS / pressure loading, QC flags
fe_inversion/petro.py       deterministic petrophysics (Csh, porosity, Archie, TVD, gradients)
fe_inversion/beds.py        bed boundaries and thick-bed center points
fe_inversion/calibration.py per-section shale bracketing and spectral-GR cross-check
fe_inversion/manual_hpv.py  HW 6 point-by-point HPV (aquifer Rw, then Csh, porosity, Sw, HPV)
fe_inversion/project_checks.py  depth shift, hydrocarbon density, Sw-Sxo, permeability checks
fe_inversion/inversion.py   forward model, LM solver, two-step and joint methods, Monte Carlo
fe_inversion/workflow.py    calibration and the end-to-end workflow
fe_inversion/plotting.py    figures
tests/                      synthetic recovery and unit tests (python -m pytest)
```

The LAS data is read from the course folder named in the config and is not stored in
this repository.
