# CLAUDE.md

Formation-evaluation code for PGE 385K (Advanced Multi-Well Formation Evaluation, UT Austin,
Prof. Torres-Verdín). Each homework project interprets a shaly-sandstone well from LAS logs
and core data.

- **Project 1** (`fe_inversion/`, `config/project1.toml`): laminated shale, triaxial Rh/Rv,
  two-step Csh/Rss then porosity/Sw inversion with Monte Carlo uncertainty. Complete.
- **Project 2** (`project2/`): grain-coating (dispersed) clay, single deep resistivity, Archie
  vs. Juhasz saturations. `project2/explore.py` is the data exploration step;
  `project2/interpret.py` and `config/project2.toml` do Tasks 5-7, `project2/permeability.py`
  does Task 8,
  `project2/invert.py` is the Monte Carlo inversion (data RHOB, NPHI, Rt; unknowns phi_s, Sws).

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 30 tests, about 20 s
python -m fe_inversion --config config/project1.toml  # Project 1 inversion -> outputs/project1
python -m fe_inversion.manual_hpv                     # PGE 358 HW 6 hand calculation
python -m fe_inversion.project_checks                 # Project 1 checks; needs results.csv first
python project2/explore.py                            # Project 2 figures + summary.md
python project2/interpret.py                          # Project 2 Archie and Juhasz results
python project2/permeability.py                       # Project 2 permeability and rock classes
python project2/invert.py                             # Project 2 Monte Carlo inversion, about 3 min
```

The Monte Carlo runs in parallel processes. 1,000 realizations take about a minute on AC
power; on battery Windows throttles the CPU. Use `--realizations 100` for quick checks.

## Data and outputs

- Input data is **not** in the repo. It lives in OneDrive under
  `.../PGE 385K - Homework Problems/Homework Projects/Homework Project No. N/Data/`, and the
  paths are absolute in `config/project1.toml` and at the top of `project2/explore.py`. If
  the user reorganizes those folders, update the paths.
- `outputs/` is untracked. Project 2 writes into its OneDrive project folder (`Exploration/`,
  `Interpretation/`).
- LAS files are read with `lasio`. Some exported CSVs round depth to 0.1 ft or to five
  significant digits; match depths by sample index or use the LAS output, not rounded CSV
  depths.

## Domain conventions

- Depths are measured depth (MD) in ft. Project 1's well is about 51° deviated: pressure
  gradients and hydrocarbon columns must be in TVD (minimum curvature from SDEV/HAZI).
- Neutron units differ by project. Project 1's TNPH is in **limestone units** (quartz reads
  about −0.035). Project 2's CNCF is in **sandstone units** (confirmed by the user from class);
  convert it to limestone units before the lithology cross-plot in the clean water sand.
- Project 2's density (ZDEN, Baker) is already corrected; ZCOR only reports the correction
  applied. Mark large corrections, do not discard them.
- Project 2 workflow from class: first assume no shale and run Archie over the whole interval
  (porosity, Rw, Rxo, Sw), then add the grain-coating clay (Juhasz). For dispersed clay, Csh
  from GR cannot exceed the clean-sand porosity.
- Archie parameters are set by each assignment: Project 1 a = 1, m = 2.3, n = 1.97;
  Project 2 a = 1, m = 2.1, n = 1.95.
- Project 1 uses three depth sections, each with its own reference shale and bracketed
  Rsh_h/Rsh_v (see README). The inversion's shale indicator is thorium, not total GR.
- HPV = (1 − Csh)·φss·(1 − Sw), identical to φT(1 − Sw,T) when the shale holds only water.
- Juhasz adds the clay term to the water **conductivity**: 1/Rw + B·Qvn/Sw, with
  Qvn = Csh·φsh/φt.
- Core Sw is unreliable in Project 1 (plugs flushed, Sw + So median 0.67).
- Mud type: Project 1 was **oil-based** (from the instructor; not in the LAS), so an Rmf-based Sxo does not
  apply there. Project 2 is water-based Polyplus (Rmf 1.32 Ω·m at 75 °F). Never assume the mud type.
- Flat Sw with height does not prove irreducible saturation; each rock class has its own capillary curve.
- Csh relation depends on the clay type (book §8.5): linear Csh = Ish only for laminated shale. Larionov
  was marked not applicable in Project 1 (laminated) but applies to grain-coating clay (Project 2), where the
  relation is chosen against neutron-density Csh, which is valid only in water-saturated sandstone.
- Csh must be 1 in a pure shale; anchor inverted Csh to the shale end-points. Use bed-center samples for
  cross-plots and calibrations (beds from GR/density gradients, settings in config).
- Permeability: classify rocks first (Leverett, √(k/φ) as the throat-radius proxy, 4–5
  fairways with equal sample counts, tied to the stratigraphy) and fit each class. Never a
  single k–φ regression over all plugs. Use core values at reservoir stress.

## Code style

- Match the existing modules: plain functions, NumPy-vectorized over depth and realizations,
  short docstrings stating the equation, comments only where the reason is not obvious.
- Physics lives in `petro.py`, the solvers in `inversion.py`, plots in `plotting.py`, and
  settings in the TOML config. Keep new end-points and thresholds in the config, not hard-coded.
- Figures follow the course rules: labeled axes with units, a title, and a legend.

## Working with the user

- Reports and explanations must be accurate and concise: state only what the data and the
  workflow support, and give caveats where results are uncertain.
- The instructor grades critical thinking over execution. Reports need an executive summary
  with quantitative results, a sensitivity analysis showing which inputs control Sw and by how
  much, and a conclusion that evaluates the work (did it work, what is inaccurate, which
  assumptions limit it, what to do better), not a recap.
- In reports write **sandstone**, not sand; state gamma-ray trends explicitly alongside terms like fining
  upward; show an inversion's cost function where it is first described; include best recommended
  practices and lessons learned; leave out detail that was not asked for.
- The course requires an appendix describing how Claude was used. When adding a method or
  making an interpretive decision (an end-point, a cutoff, a parameter choice), say what was
  chosen and why, so the appendix can be written from it.
- Class notes the user pastes may contain transcription errors. Check them against the
  assignment and the literature before implementing them.
- Commit only when asked. Commit messages: a short imperative subject line, then a body
  explaining why.
