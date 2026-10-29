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
| Pure shale (GR, $\phi_D$, $\phi_N$) | 9,654–9,690 | 94.6 | 0.187 | 0.436 | 0.703 (not used) |
| Aquifer | 10,065–10,080 | 27.2 | 0.267 | 0.231 | 0.379 |

- $R_{sh}$ = 0.89 Ω·m (change 2 below), not the shale point's own 0.703 Ω·m.
- TNPH is used as sandstone units (no offset). The neutron reads 0.035 below density in the
  aquifer, attributed to residual (trapped) gas; the aquifer is taken as $S_w = 1$, so the
  negative neutron-density $C_{sh,w}$ (−0.148) is set to 0.
- Archie $a = 1$, $m = 2.3$, $n = 1.97$. Fluids: $\rho_{ma}$ = 2.65, $\rho_w$ = 1.05 and
  $\rho_{hc}$ = 0.644 g/cc (oil, from the formation-pressure gradient).

### Four changes from HW 6 as given, and why

**Changes 1–2: sandstone resistivity.** Run exactly as in HW 6 (linear GR, $R_{sh}$ = the base
shale's 0.60 Ω·m), $R_s$ is undefined at **98 %** of the pay. $R_s$ needs $C_{sh} < R_{sh}/R_t$: in the pay $R_{sh}/R_t$ has a median of
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

With both changes $R_s$ is undefined at only about 2 % of the samples.

**Changes 3–4: sandstone porosity.** With changes 1–2 only, $\phi_s$ read **0.094 above core**
and did not correlate with it. Two causes:

3. **Shale point for the density and neutron correction: the upper shale (9,654–9,690 ft)**
   instead of the base shale. The correction
   $\phi^{(c)} = (\phi - C_{sh}\,\phi_{sh})/(1 - C_{sh})$ *raises* porosity wherever the rock reads
   more porous than the shale. The base shale's density porosity (0.155) is far below the
   pay's, so every unit of $C_{sh}$ added porosity. The upper shale (0.187) is also the shale
   the inversion uses for Section 2.

4. **Oil-corrected density porosity in place of B3.** PHID assumes a 1.0 g/cc pore fluid, but
   the pay holds oil, so density porosity reads high. The RMS average in B3 is a gas
   correction and does not fix this: RMS vs. plain average changed the bias by only 0.002, and
   the neutron (0.11 high even after correction) only adds error. Instead, the corrected
   density porosity is re-solved with the actual pore fluid, HW 6 part (h) used in reverse:

$$
\rho_s = \rho_{ma} - \phi_D^{(c)}\,(\rho_{ma} - 1.0),
\qquad
\rho_f = S_w\,\rho_w + (1 - S_w)\,\rho_{hc},
\qquad
\phi_s = \frac{\rho_{ma} - \rho_s}{\rho_{ma} - \rho_f}
$$

   $S_w$ comes from B5 with this $\phi_s$, so the two are iterated (8 passes; it converges in
   a few). Where $R_s$ is undefined, $S_w = 1$ is used. Set `porosity_method = "hw6"` in the
   config to use B3 as given.

| Porosity vs. core (60 plugs) | Bias | Bias, representative plugs | r |
|---|---|---|---|
| Changes 1–2 only | +0.094 | +0.075 | 0.00 |
| + upper shale (3) | +0.038 | +0.031 | 0.52 |
| **+ oil-corrected density (3 + 4)** | **+0.009** | **+0.004** | **0.56** |
| Two-step inversion | +0.012 | +0.007 | 0.72 |

"Representative" plugs have grain density ≤ 2.70 g/cc; heavier plugs likely contain clay or mica.

### Results (all four changes)

| | Value |
|---|---|
| $R_w$ | 0.01547 Ω·m |
| Valid samples | 964 of 1,021 (94 %); 27 pure shale, 26 with undefined $R_s$, 4 with $\phi_s \le 0$ |
| Mean $C_{sh}$, $\phi_s$, $S_w$ in the pay (valid samples) | 0.414, 0.269, 0.403 |
| Porosity vs. core | bias +0.009 (RMS 0.042, r = 0.56) |
| Hydrocarbon column, Section 2 | **14.0 ft TVD** (pay 9,825–10,005: 11.2 ft; 10,005–10,045: 2.0 ft; water sand: 0.8 ft) |
| Two-step inversion, same interval | 14.4 ft TVD |

### Known limitations

- **Water sand is not fully water.** Mean $S_w$ in the aquifer interval is 0.87, not 1, so it
  contributes about 0.8 ft that should be zero. The 10,005–10,045 ft transition adds 2.0 ft
  that the inversion puts at $S_w \approx 1$.
- **Mixed shale points.** The density and neutron shale values come from the upper shale and
  $R_{sh}$ from the shale beds inside the pay. HW 6 uses one shale point for everything; with
  the upper shale's own $R_t$ (0.703 Ω·m), $R_s$ is undefined at 23 % of the pay.
- **Constant properties.** One shale point and one aquifer point are applied to the whole
  section.
