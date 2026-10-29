# Point-by-point HPV calculation (PGE 358 HW 6 method)

This is the "by hand" laminated-shale workflow from HW 6, Problems 1 and 2, written so it can
be applied at every depth sample. It needs only two reference points read off the logs:

- a **pure shale** point, and
- an **aquifer** point (water-saturated sandstone, $S_w = 1$).

From those it finds $R_w$ once, and then $C_{sh}$, $\phi_s$, $S_w$, $S_h$ and HPV at every depth,
**assuming the shale and the sandstone have the same properties throughout the section.**

> To see the equations rendered, open this file on GitHub or press `Ctrl+Shift+V` in VS Code.

---

## Symbols

| Symbol | Meaning | Log / source |
|---|---|---|
| $\gamma$ | Gamma ray | GR (gAPI) |
| $\phi_D$ | Apparent density porosity, sandstone units | PHID_SS (matrix 2.65 g/cc, fluid 1.0 g/cc) |
| $\phi_N$ | Apparent neutron porosity, sandstone units | TNPH converted to sandstone units (see note 1) |
| $R_t$ | Resistivity parallel to bedding (horizontal) | RH72H_1D_FLT (see note 2) |
| $C_{sh}$ | Volumetric concentration of shale | calculated |
| $\phi_s$ | Porosity of the sandstone (non-shale) part | calculated |
| $R_s$ | Resistivity of the sandstone part | calculated |
| $R_w$ | Formation-water resistivity | calculated in the aquifer |
| $S_w$, $S_h$ | Water and hydrocarbon saturation of the sandstone | calculated |
| $a$, $m$, $n$ | Archie parameters | $a = 1$, $m = 2.3$, $n = 1.97$ (Project 1 handout) |

Subscripts: $sh$ = pure-shale reading, $w$ = aquifer reading.

---

## Part A: the aquifer (done once)

### A1. Shale concentration in the aquifer: neutron-density difference

In a water sand the neutron and density logs see the same fluid, so their difference comes only
from shale. This is why this method **only works in the aquifer**:

$$
C_{sh,w} = \frac{\phi_{N,w} - \phi_{D,w}}{\phi_{N,sh} - \phi_{D,sh}}
$$

### A2. Shale-corrected (sandstone) porosities in the aquifer

Remove the shale's contribution from each log:

$$
\phi_D^{(c)} = \frac{\phi_D - C_{sh}\,\phi_{D,sh}}{1 - C_{sh}}
\qquad
\phi_N^{(c)} = \frac{\phi_N - C_{sh}\,\phi_{N,sh}}{1 - C_{sh}}
$$

In a water sand the two corrected values come out equal, which checks the aquifer pick:

$$
\phi_{s,w} = \phi_D^{(c)} = \phi_N^{(c)}
$$

### A3. Sandstone resistivity in the aquifer

Laminated shale: sand and shale layers conduct in parallel along bedding, so

$$
\frac{1}{R_t} = \frac{C_{sh}}{R_{sh}} + \frac{1 - C_{sh}}{R_s}
\quad\Longrightarrow\quad
R_s = (1 - C_{sh})\,\frac{R_t\,R_{sh}}{R_{sh} - C_{sh}\,R_t}
$$

Applied with the aquifer values ($R_{t,w}$, $C_{sh,w}$) this gives $R_{s,w}$.

### A4. Formation-water resistivity

Archie with $S_w = 1$ in the aquifer sand:

$$
R_{s,w} = R_w \,\frac{a}{\phi_{s,w}^{\,m}}
\quad\Longrightarrow\quad
R_w = \frac{R_{s,w}\;\phi_{s,w}^{\,m}}{a}
$$

---

## Part B: every other depth in the section

### B1. Shale concentration from gamma ray, anchored to the aquifer

There is no clean sandstone to read $\gamma_{sand}$ from, so the gamma-ray scale is anchored at
the aquifer, whose shale concentration is already known from A1:

$$
C_{sh} = C_{sh,w} + \frac{1 - C_{sh,w}}{\gamma_{sh} - \gamma_w}\,\bigl(\gamma - \gamma_w\bigr)
$$

At the aquifer ($\gamma = \gamma_w$) this returns $C_{sh,w}$, and at the shale
($\gamma = \gamma_{sh}$) it returns 1. Values outside 0–1 are clipped.

### B2. Shale-corrected porosities

Same correction as A2, now with this depth's $C_{sh}$:

$$
\phi_D^{(c)} = \frac{\phi_D - C_{sh}\,\phi_{D,sh}}{1 - C_{sh}}
\qquad
\phi_N^{(c)} = \frac{\phi_N - C_{sh}\,\phi_{N,sh}}{1 - C_{sh}}
$$

### B3. Sandstone porosity

- If $\phi_N^{(c)} < \phi_D^{(c)}$ (crossover), the pore fluid is lighter than water. Use the
  root-mean-square average, as both HW 6 solutions do:

$$
\phi_s \approx \sqrt{\frac{\left(\phi_N^{(c)}\right)^2 + \left(\phi_D^{(c)}\right)^2}{2}}
$$

- Otherwise (no crossover, liquid-filled), use the plain average
  $\phi_s = \tfrac{1}{2}\bigl(\phi_D^{(c)} + \phi_N^{(c)}\bigr)$.

### B4. Sandstone resistivity

Same equation as A3, with this depth's $R_t$ and $C_{sh}$:

$$
R_s = (1 - C_{sh})\,\frac{R_t\,R_{sh}}{R_{sh} - C_{sh}\,R_t}
$$

It is only defined when $R_{sh} > C_{sh} R_t$. In a hydrocarbon sand $R_s > R_t$: removing the
conductive shale laminae exposes the more resistive sand.

### B5. Water and hydrocarbon saturation

Archie in the sandstone, solved for $S_w$ (capped at 1):

$$
R_s = R_w\,\frac{a}{\phi_s^{\,m}}\,\frac{1}{S_w^{\,n}}
\quad\Longrightarrow\quad
S_w = \left(\frac{a\,R_w}{\phi_s^{\,m}\,R_s}\right)^{1/n}
\qquad
S_h = 1 - S_w
$$

### B6. Hydrocarbon pore volume

Net-to-gross is $1 - C_{sh}$, so per unit of rock volume:

$$
\mathrm{HPV} = (1 - C_{sh})\;\phi_s\;S_h = (1 - C_{sh})\;\phi_s\;(1 - S_w)
$$

Summing $\mathrm{HPV} \times \Delta z$ over the samples gives the hydrocarbon column in feet.

---

## Check against the HW 6 solutions

The equations above, coded as written, reproduce both worked solutions (HW 6 Archie
parameters: $m = 2.1$, $n = 1.9$ in Problem 1; $m = 2.2$, $n = 1.9$ in Problem 2):

| | $R_w$ (Ω·m) | $C_{sh}$ | $\phi_s$ | $S_w$ | HPV |
|---|---|---|---|---|---|
| Problem 1, solution | 0.02559 | 0.0893 | 0.1491 | 0.3927 | 0.08244 |
| Problem 1, these equations | 0.0256 | 0.0893 | 0.1491 | 0.3925 | 0.0825 |
| Problem 2, solution | 0.03706 | 0.2638 | 0.3068 | 0.3786 | 0.14035 |
| Problem 2, these equations | 0.0371 | 0.2639 | 0.3068 | 0.3786 | 0.1403 |

The small differences are rounding in the solutions.

---

## Applying it to Project 1 (Section 2, 9,825–10,080 ft)

Run it with:

```bash
python -m fe_inversion.manual_hpv
```

Settings are in `[manual_hpv]` in `config/project1.toml`. Outputs go to `outputs/project1/`:
`manual_hpv.csv` (every 0.25 ft sample with every intermediate value), `manual_hpv.md`
(inputs, Part A, totals and a 5-ft table) and `manual_hpv.png` (tracks).

### Inputs used

| Point | Depth (ft MD) | $\gamma$ (gAPI) | $\phi_D$ | $\phi_N$ (TNPH) | $R_t$ = RH (Ω·m) |
|---|---|---|---|---|---|
| Pure shale | 10,189–10,191 | 95.7 | 0.155 | 0.399 | 0.597 |
| Aquifer | 10,065–10,080 | 27.2 | 0.267 | 0.231 | 0.379 |

- TNPH is used as sandstone units (no offset). The neutron reads 0.035 below density in the
  aquifer, attributed to residual (trapped) gas; the aquifer is taken as $S_w = 1$, so the
  negative neutron-density $C_{sh,w}$ (−0.150) is set to 0.
- Archie $a = 1$, $m = 2.3$, $n = 1.97$.

### Two changes from HW 6 as given, and why

Run exactly as in HW 6 (linear GR, $R_{sh}$ = the shale point's 0.60 Ω·m), $R_s$ is undefined
at **98 %** of the pay. $R_s$ needs $C_{sh} < R_{sh}/R_t$: in the pay $R_{sh}/R_t$ has a median of
0.44, but the linear GR gives 0.69 (diagnosis figure:
`outputs/diagnostics/rs_undefined_diagnosis.png`). Uranium is not the cause: thorium and
potassium fail the same way. So:

1. **Larionov (Tertiary) GR.** The GR index $I$ in B1 is replaced by
   $0.083\,(2^{3.7 I} - 1)$, which suits young (Tertiary) deltaic sands:

$$
C_{sh} = C_{sh,w} + (1 - C_{sh,w})\cdot 0.083\left(2^{\,3.7\,I} - 1\right),
\qquad I = \frac{\gamma - \gamma_w}{\gamma_{sh} - \gamma_w}
$$

2. **$R_{sh}$ = 0.89 Ω·m**, the parallel resistivity of the high-GR bed centers inside the pay,
   instead of the base shale's 0.60 Ω·m.

With both changes $R_s$ is undefined at only 25 of 1,021 samples (2 %).

### Results

| | Value |
|---|---|
| $R_w$ | 0.01547 Ω·m |
| Valid samples | 976 of 1,021 (96 %); 25 with undefined $R_s$, 20 pure shale |
| Mean $C_{sh}$, $\phi_s$, $S_w$ (valid samples) | 0.321, 0.317, 0.448 |
| Hydrocarbon column, Section 2 | **17.5 ft TVD** (pay 9,825–10,005: 14.4 ft; 10,005–10,045: 2.5 ft; water sand: 0.7 ft) |
| Two-step inversion, same interval | 14.4 ft TVD |

### Known limitations

- **Porosity reads high.** $\phi_s$ is 0.094 above core porosity on average, and does not
  correlate with core (62 plugs). The inversion's bias is +0.012. Likely causes: the base
  shale's density and neutron values used for the shale correction (the shales in the pay are
  lighter), and the root-mean-square average, which is meant for gas, applied to oil.
- **Water sand is not fully water.** Mean $S_w$ in the aquifer interval is 0.88, not 1, so it
  contributes a spurious 0.7 ft. The 10,005–10,045 ft transition adds 2.5 ft that the
  inversion puts at $S_w \approx 1$.
- **Constant properties.** One shale point and one aquifer point are applied to the whole
  section.
