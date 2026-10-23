# Formation Evaluation Inversion

Python program for interpreting well logs in laminated shaly sandstones (PGE 385K
Homework Project 1). It loads LAS logs, core and formation-pressure data, runs the
conventional interpretation, and then a **joint Monte Carlo inversion** that estimates
shale concentration, sand porosity and sand water saturation with uncertainty bands.

## Run

```bash
pip install -r requirements.txt
python -m fe_inversion --config config/project1.toml
```

Options: `--realizations N` changes the number of Monte Carlo realizations, and
`--output DIR` changes where results go (default `outputs/project1`). 1,000 realizations
take about a minute on 16 cores. Set `workers = 1` in the config to run serially.

Outputs:

| File | Contents |
|---|---|
| `01_log_composite.png` | GR/caliper, resistivities incl. Rh/Rv, density-neutron with crossover and core porosity, PEF, K/Th, core k/Sw/So |
| `02_crossplots.png` | Rt-GR, neutron-density, K-Th, GR-RHOB, Rv-Rh with laminated-model grid, Thomas-Stieber |
| `03_csh_methods.png` | Csh from GR (linear and Larionov), neutron-density, Rh-Rv closed form, and the inversion |
| `04_inversion.png` | Csh, sand porosity, Sw, Sxo, Rss and HPV with P10-P90 bands, plus data misfit |
| `05_data_fit.png` | Measured vs. predicted Rh, Rv, RHOB, NPHI |
| `06_pressure_fluid_density.png` | Pressure gradients in TVD, free-water level, core-calibrated fluid density |
| `07_core_comparison.png` | Core k-phi, k vs. Sw, inversion vs. core porosity and Sw |
| `08_monte_carlo.png` | Hydrocarbon-column distributions and which inputs drive the uncertainty |
| `results.csv`, `results.las` | All input curves, conventional results, and P10/P50/P90 of every output |
| `summary.md` | End-points, Rw/Rmf, gradients, zone averages, HC column P10/P50/P90 |

## Method

**Laminated model.** Sand and shale layers are in parallel for the resistivity measured
along bedding and in series for the resistivity measured across bedding:

    1/Rh = Csh/Rsh_h + (1 - Csh)/Rss        Rv = Csh*Rsh_v + (1 - Csh)*Rss

These two equations have a closed-form solution for Csh and Rss (`petro.laminated_resistivity`).

**Joint inversion.** At every depth the unknowns are Csh, sand-lamina porosity phi_ss and
sand-lamina Sw. The forward model predicts Rh and Rv (with Archie, a=1, m=2.3, n=1.97 in
the sand), bulk density and neutron porosity from those three unknowns. The GR-derived Csh
is the starting model and a Gaussian prior, and weak priors on phi_ss and Sw keep the problem
stable where the data can't resolve them (pure shale). The solver is a batched
Levenberg-Marquardt with an analytic Jacobian and bounded parameters.

**Monte Carlo.** Each realization draws new shale and sand end-points, matrix and fluid
densities, hydrocarbon hydrogen index, Rw and Rmf from their uncertainties, adds
measurement noise to the logs, perturbs the GR prior, and re-solves (randomized maximum
likelihood). The spread of the 1,000 solutions gives the P10/P50/P90 bands and the
distribution of the hydrocarbon column.

**Calibration from the data** (all can be overridden in the config):
- Shale end-points: zoned by depth. Sequences 1-2 use the upper shale (9,650-9,690 ft) and
  the base uses the base shale (10,160-10,200 ft). The base shale is denser (2.39 vs. about
  2.35 g/cc), and using it everywhere biased sand porosity about +0.04 high against core.
  Each realization shifts all zones by one draw, and rho_sh has a wide 0.03 g/cc std.
- Clean-sand GR: 3rd percentile of GR.
- Rw and Rmf: from the water sand, with Sw = Sxo = 1 in the laminated model.
- Hydrocarbon density: oil gradient from formation-tester points, converted to TVD with
  the minimum-curvature trajectory (the well is about 50° deviated).

## Caveats

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
config/project1.toml     data paths, curve names, zones, uncertainties, inversion settings
fe_inversion/io.py       LAS / pressure loading, QC flags
fe_inversion/petro.py    deterministic petrophysics (Csh, porosity, Archie, TVD, gradients)
fe_inversion/inversion.py forward model, LM solver, parallel Monte Carlo
fe_inversion/workflow.py calibration and the end-to-end workflow
fe_inversion/plotting.py figures
tests/                   synthetic recovery and unit tests (python -m pytest)
```

The LAS data is read from the course folder named in the config and is not stored in
this repository.
