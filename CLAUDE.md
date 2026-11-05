# CLAUDE.md

Formation-evaluation code for PGE 385K (Advanced Multi-Well Formation Evaluation, UT Austin,
Prof. Torres-Verdín). Each homework project interprets a shaly-sandstone well from LAS logs
and core data.

- **Project 1** (`fe_inversion/`, `config/project1.toml`): laminated shale, triaxial Rh/Rv,
  two-step Csh/Rss then porosity/Sw inversion with Monte Carlo uncertainty. Complete.
- **Project 2** (`project2/`): grain-coating (dispersed) clay, single deep resistivity, Archie
  vs. Juhasz saturations. `project2/explore.py` is the data exploration step; the
  interpretation is not written yet.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 16 tests, about 20 s
python -m fe_inversion --config config/project1.toml  # Project 1 inversion -> outputs/project1
python -m fe_inversion.manual_hpv                     # PGE 358 HW 6 hand calculation
python -m fe_inversion.project_checks                 # Project 1 checks; needs results.csv first
python project2/explore.py                            # Project 2 figures + summary.md
```

The Monte Carlo runs in parallel processes. 1,000 realizations take about a minute on AC
power; on battery Windows throttles the CPU. Use `--realizations 100` for quick checks.

## Data and outputs

- Input data is **not** in the repo. It lives in OneDrive under
  `.../PGE 385K - Homework Problems/Homework Projects/Homework Project No. N/Data/`, and the
  paths are absolute in `config/project1.toml` and at the top of `project2/explore.py`. If
  the user reorganizes those folders, update the paths.
- `outputs/` is untracked. Project 2 writes into its OneDrive project folder (`Exploration/`).
- LAS files are read with `lasio`. Some exported CSVs round depth to 0.1 ft or to five
  significant digits; match depths by sample index or use the LAS output, not rounded CSV
  depths.

## Domain conventions

- Depths are measured depth (MD) in ft. Project 1's well is about 51° deviated: pressure
  gradients and hydrocarbon columns must be in TVD (minimum curvature from SDEV/HAZI).
- Neutron units differ by project. Project 1's TNPH is in **limestone units** (quartz reads
  about −0.035). Project 2's CNCF is in **sandstone units** (confirmed by the user from class).
- Archie parameters are set by each assignment: Project 1 a = 1, m = 2.3, n = 1.97;
  Project 2 a = 1, m = 2.1, n = 1.95.
- Project 1 uses three depth sections, each with its own reference shale and bracketed
  Rsh_h/Rsh_v (see README). The inversion's shale indicator is thorium, not total GR.
- HPV = (1 − Csh)·φss·(1 − Sw), identical to φT(1 − Sw,T) when the shale holds only water.
- Juhasz adds the clay term to the water **conductivity**: 1/Rw + B·Qvn/Sw, with
  Qvn = Csh·φsh/φt.
- Core Sw is unreliable in Project 1 (plugs flushed, Sw + So median 0.67).

## Code style

- Match the existing modules: plain functions, NumPy-vectorized over depth and realizations,
  short docstrings stating the equation, comments only where the reason is not obvious.
- Physics lives in `petro.py`, the solvers in `inversion.py`, plots in `plotting.py`, and
  settings in the TOML config. Keep new end-points and thresholds in the config, not hard-coded.
- Figures follow the course rules: labeled axes with units, a title, and a legend.

## Working with the user

- Reports and explanations must be accurate and concise: state only what the data and the
  workflow support, and give caveats where results are uncertain.
- The course requires an appendix describing how Claude was used. When adding a method or
  making an interpretive decision (an end-point, a cutoff, a parameter choice), say what was
  chosen and why, so the appendix can be written from it.
- Class notes the user pastes may contain transcription errors. Check them against the
  assignment and the literature before implementing them.
- Commit only when asked. Commit messages: a short imperative subject line, then a body
  explaining why.
