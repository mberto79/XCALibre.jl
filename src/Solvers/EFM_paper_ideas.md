# EFM in XCALibre — Paper Sketch

Notes toward an academic paper on the partial-wetting Eulerian thin-film
model in XCALibre.jl. **Critical version, post literature review.** The
goal is to identify what is genuinely novel, what is a combination of
existing ideas, what is already published, and what reviewers will attack.

## 1. Motivating gap

Meredith et al. (2011) introduced the partial-wetting ETFM with contact-
line stress

  τ_θw = β σ (1 − cos θ_m) ∇_s w     (Meredith Eq 4)

using a hard wetting indicator `w = 1 if h > h_crit else 0`. Singh et
al. (2021) extended the framework to bearing chambers and fitted a
correlation `β = 3.69 (sin φ / θ_m)^0.486` over 19 cases.

Two issues are not addressed in the published ETFM literature:

(a) **Discretization-induced streamwise τ_θw spike.** With a hard
indicator, `∇_s w` at the leading-edge contact line is a step over one
cell, magnitude `≈ 1/Δx`. The streamwise component on our discretization
exceeds gravity by an order of magnitude (~7× for Meredith case 5),
reversing the leading-edge velocity. Fluent's thin-film solver hides
this with implicit smoothing whose details are not documented; the
empirical β-fit absorbs the artefact in part.

(b) **No principled treatment for inclination-regime change.** The
β-correlation handles it as a fitted exponent; but this conflates
contact-angle physics with mesh artefacts, hydrostatic-vs-gravity
balance, and rivulet-instability dynamics into one lumped parameter.
Generalizing to curved walls (bearing chambers) requires
*azimuthal averaging* of φ — an obvious crudeness.

The proposed contribution: identify (a) explicitly as a numerical
artefact, suppress it via geometry-derived projection, generalize to
cell-local form for arbitrary surfaces, and articulate what would
take β out of the model entirely. Be honest that some of this is
combination, not invention.

## 2. Prior art (what's been tried)

| Idea | Established in | Where |
|---|---|---|
| CSF / smoothed surface-tension force from indicator | Brackbill et al. 1992 | VOF context |
| Hard `w = step(h)` indicator in ETFM | Meredith et al. 2011 | depth-averaged ETFM |
| β-correlation `β = 3.69(sin φ/θ_m)^0.486` | Singh et al. 2021 | ETFM bearing chamber |
| Dv (6/5) parabolic profile correction on momentum | Kakimpa et al. 2016; Mouvanal et al. 2022 | ETFM rimming + bearing chamber |
| Cell-local gravity decomposition for *pressure* (not contact-line) | Mouvanal et al. 2022 | ETFM bearing chamber |
| Lubrication theory on curved substrate (gravity decomposition) | Roy et al. 2002; Craster & Matar 2009 | analytical lubrication theory |
| Cox-Voinov dynamic θ_d via Ca | Cox 1986; Voinov 1976 | VOF/level-set, *not* ETFM |
| Disjoining-pressure precursor-film regularization | de Gennes 1985; Pahlavan et al. 2015 | sharp-interface lubrication |
| VOF reference simulations (rivulets) | Singh et al. 2015 (Chem Eng Sci) | VOF benchmark |
| Experimental rivulet measurements | Johnson et al. 1999; Lan et al. 2010 | optical / fluorescence |

## 3. Proposed contributions, scored

### C1. Streamwise-spike isolation and lateral projection of τ_θw

Replace `τ_θw = scale · ∇_s w` (full gradient) with
`τ_θw = scale · (∇_s w · l̂) l̂` where `l̂ = (n × t̂)/|n × t̂|`,
`n` is the local surface normal, `t̂` the gravity tangent in the
surface.

**Status.** No prior ETFM paper has identified the streamwise spike as
a discretization artefact independent of physics, nor proposed
projecting τ_θw onto the in-surface lateral direction to suppress it.
**Novel in ETFM.** CSF in VOF (Brackbill 1992) is conceptually adjacent
but operates by *smoothing* the indicator field, not by directional
projection of the resulting force. The two are not equivalent.

**Reviewer attack vectors.**
- "You sacrifice paper Eq 4's streamwise restraint." Yes; this trades
  fidelity for stability. Must be quantified: a CSF-smoothed full-∇w
  variant on a sequence of meshes, plotted alongside the projection,
  showing the projection converges to the smooth limit at coarse
  resolutions while the hard-∇w fails.
- "Why isn't this just bad-discretization-disguise-as-physics?" Because
  the spike scales as 1/Δx and is geometry-aligned (always streamwise
  in the surface), so projecting it out is principled, not arbitrary.

### C2. sin²α regime blend on the wf clip

`clip_strength = sin² α`, where α is the angle between gravity and the
*surface normal*, applied as a multiplicative weight on the directional
clip of the wetting fraction at lateral wet-dry faces. Pure clip on
vertical plates (α → π/2 ⇒ sin²α → 1), no clip on horizontal plates
(α → 0 ⇒ sin²α → 0; rivulets free to form).

**Status.** Lubrication theory has long used `sin α` and `cos α`
explicitly in the in-plate vs out-of-plate gravity decomposition (Roy
et al. 2002; Craster & Matar 2009). What is novel is using a *power*
of these quantities as a discrete regime blend on a numerical clip.
**Combination, not invention.** Geometric basis (sound) + numerical
regime blending (novel application, but not a profound advance).

**Reviewer attack vectors.**
- "Why sin²α and not sin α, cos α, or some other monotone function?"
  This is the strongest attack. The functional form has no derivation.
  Must show: parametric study over `sin^n α` for n ∈ {0.5, 1, 2, 3} on
  cases 1–8 demonstrating sin²α minimizes a meaningful error metric
  (rivulet count? width? rim height?). Without this, the choice is
  ad hoc.
- "What about α between 0° and 90°? Cases 9–19 cover 30°, 60°, 90°.
  Does the blend interpolate well there?" Direct experimental
  comparison to Lan et al. (2010) cases 9–19 is the only answer.

### C3. Cell-local geometry on contact-line stress (extension to curved surfaces)

`n_c, t̂_c, l̂_c, sin²α_c` evaluated per-cell from the mesh, no global
or azimuthally-averaged angle.

**Status.** Mouvanal et al. (2022) already use cell-local gravity
decomposition — but for pressure-driving terms only, not for τ_θw
projection or wf clip. **Incremental extension** of an existing
practice. Worth claiming, but must cite Mouvanal explicitly and frame
as "extending cell-local geometry to the contact-line stress and
wet-dry flux constraint."

**Reviewer attack vectors.**
- "Mouvanal already did this." Mouvanal applied it to the pressure
  decomposition; we apply it to the τ_θw projection and the wf clip
  — both new applications. The distinction is real but narrow.
- "Demonstrate it on a case where global-φ averaging breaks." Bearing
  chamber with strongly-varying gravity orientation (e.g. inverted
  sections) is the natural test.

### C4. Cox-Voinov dynamic θ_d to eliminate β (future direction)

Replace static `(1 − cos θ_m)` with `(1 − cos θ_d)` where
`θ_d³ ≈ θ_m³ + 9 Ca ln(L/L_s)` (Cox 1986; Voinov 1976), with
`Ca = μ |u_cl| / σ` evaluated locally at the contact-line cell.

**Status.** Cox-Voinov is standard in VOF and level-set (de Gennes
1985; Snoeijer & Andreotti 2013). **Novel in ETFM** as a primary
contact-line stress driver. But:

**Reviewer attack vectors.**
- "Cox-Voinov is valid only for Ca ≪ 1 (Snoeijer & Andreotti 2013).
  Bearing chambers at high shaft speed violate this." Must report the
  Ca range of every test case and acknowledge the limitation. Where
  Ca > 0.1, Cox-Voinov is a stretch and should be flagged.
- "u_cl in a depth-averaged model is the cell-average velocity at
  the wet-dry boundary, not the actual contact-line velocity." This is
  a real ambiguity. The fix is either a careful definition (e.g.,
  velocity component normal to ∇w, evaluated at the wet-side neighbour)
  or to acknowledge the depth-averaged surrogate explicitly.
- "Hysteresis is not captured." Right; for advancing/receding
  asymmetry, an interface-formation model (Shikhmurzaev 1996) is
  needed. Cox-Voinov here is a one-direction fix.

Present this as a future direction with a clean implementation path,
not a primary claim. Otherwise the paper risks fighting a Cox-Voinov
debate it does not need to win.

### C5. Dv on momentum only, not h-flux conservation

The (6/5) parabolic-profile correction is applied to the U-momentum
convective flux, kept separate from the conservative h-flux.

**Status.** Already done correctly in Kakimpa et al. (2016) and
Mouvanal et al. (2022). **Not a contribution.** Demote to a brief
remark for completeness; do not feature.

## 4. Critical novelty summary

- **Strongest novel angle (lead claim):** C1 + C3 together —
  identifying the streamwise spike, suppressing it via lateral
  projection, and extending the projection to cell-local geometry on
  arbitrary surfaces. Verifiable, mesh-convergence-testable, no prior
  ETFM art.

- **Weakest claim (high reviewer risk):** C2 sin²α blend functional
  form. Without a derivation or a parametric study, this is ad hoc.
  Either justify or replace with a clean step / smooth-step analysis.

- **Demote:** C5 (Dv) — already in literature.

- **Future work, not core:** C4 Cox-Voinov.

This is one strong primary contribution + one extension + one well-
framed future direction. That is a plausible academic paper.

## 5. Verification roadmap (defendable at review)

### V1 — τ_θw spike scaling (defends C1)

Single rivulet on 60° inclined plate (Meredith case 1). Three uniform
meshes with refinement factor 2 (Δx = 8, 4, 2 mm). Plot:

- |τ_θw,stream|_max at the leading edge vs. Δx, hard-w + full ∇w. Show
  scaling ~ 1/Δx.
- Same plot with the C1 lateral projection. Show scaling-independence.
- Same plot with a CSF-smoothed full ∇w (smoothing kernel width 3 Δx).
  Show convergence to the projection result at fine Δx.

This silences the "bad discretization disguised as physics" attack.

### V2 — Functional-form parametric study (defends or reframes C2)

Run Meredith cases 1–8 (φ = 5°, 90°) at fixed β = 1 with
clip_strength = `sin^n α` for n ∈ {0.5, 1, 2, 3}. Quantitative metrics:

- Rivulet width at the measurement station vs. Lan et al. (2010) data.
- Rim height vs. Singh et al. (2015) VOF reference.
- Number of rivulets in cases 1–4.

If `sin²α` does not win, replace it with the winner and rebrand the
contribution. If no power wins, the regime-blend form needs revisiting
(perhaps a smooth Heaviside in α with a width tied to mesh resolution
is more defensible).

### V3 — Curved-surface validation (defends C3)

Reproduce Singh et al. (2021) bearing chamber Fig 9 (Chandra et al.
geometry, 30 LPM, 15,000 RPM). Two simulations:

- Global-φ averaging (mimicking Singh's published result).
- Cell-local sin²α_c per-cell.

Compare film thickness at the three measurement stations
(75°, 105°, 135° clockwise from TDC) vs. experimental data of
Chandra et al. (2013). Quantify error reduction.

### V4 — Ca-range validation for C4 (future-direction defense)

For all 19 Meredith + Lan cases, compute `Ca = μ u_avg / σ` at the
inlet and report. If Ca < 0.1 across the test matrix, Cox-Voinov is
defensible there and the future-direction claim is sound. Bearing
chamber Ca will be higher; flag as out-of-validity.

### V5 — Mesh-independence beyond the spike (general robustness)

Three meshes for Meredith case 5 (vertical plate). Plot rivulet
position, rim height, and number of upstream cells (regression check)
vs. mesh density. Convergence at second-order ideal but
first-order acceptable for a partially-wetted sharp interface.

## 6. Honest limitations to declare in the paper

- **τ_θw streamwise restraint is sacrificed** by C1. Quantitative
  comparison to a CSF-smoothed full-∇w variant is the price of entry.
- **`sin²α` choice is geometric, not derived.** Parametric study or
  rebrand as future direction.
- **Hard wetting at h_crit = 1e-10** is a single-point study;
  reviewers will demand at least a 3-point sensitivity (h_crit ∈ {1e-12,
  1e-10, 1e-8}). Anchor the choice in disjoining-pressure scales
  (Pahlavan et al. 2015).
- **β = 1 in our Cox-Voinov-eliminating limit** assumes the empirical
  fit's coefficient absorbs only mesh artefacts and missing dynamics;
  this is an *assertion* until V4 is run.
- **u_cl ambiguity** for any future Cox-Voinov implementation. State
  the depth-averaged surrogate explicitly.
- **2D film mesh on a flat / mildly-curved substrate** is the
  validation envelope. Extreme curvature (sharp corners, geometry
  with curvature radius < 5 cells) is out of scope.

## 7. Target venues and framing

- **J. Eng. Gas Turbines Power** — Singh's venue, bearing-chamber
  application; reviewers will demand C3 validation. Framing:
  "Mesh-independent, geometry-driven contact-line treatment for
  bearing-chamber ETFM."
- **Int. J. Multiphase Flow** — broader, more receptive to method
  novelty. Framing: "Suppression of indicator-gradient discretization
  artefacts in partial-wetting Eulerian thin-film models."
- **Computers & Fluids** — strong on discretization stories; V1 is the
  centerpiece.

## 8. What we are combining vs inventing

**Genuinely novel (no clear prior art in ETFM):**
- C1 lateral projection of τ_θw to suppress streamwise spike.

**Combinations of existing ideas applied to a new context:**
- C2: lubrication-theory geometric scalars (sin α, cos α — Roy 2002,
  Craster & Matar 2009) used as a numerical regime blend on the
  wet-dry flux constraint. The blend application is novel; the
  geometric primitives are not.
- C3: cell-local gravity decomposition (Mouvanal 2022) extended from
  pressure terms to contact-line stress and wf clip.

**Borrowing from VOF/level-set, applied to ETFM:**
- C4: Cox-Voinov dynamic contact angle (Cox 1986, Voinov 1976) used
  to eliminate the empirical β coefficient.

**Already published, kept for completeness only:**
- C5: Dv (6/5) on momentum (Kakimpa 2016, Mouvanal 2022).

## 9. Tagline (post-revision)

A coordinate-invariant, geometry-driven contact-line stress for
Eulerian thin-film models, eliminating mesh-dependent streamwise
spikes from hard wetting indicators on inclined and curved surfaces.

---

## Bibliography

Brackbill, J.U., Kothe, D.B., & Zemach, C. (1992). A continuum method
for modeling surface tension. *Journal of Computational Physics*,
100(2), 335–354. https://doi.org/10.1016/0021-9991(92)90240-Y

Cox, R.G. (1986). The dynamics of the spreading of liquids on a solid
surface. Part 1. Viscous flow. *Journal of Fluid Mechanics*, 168,
169–194. https://doi.org/10.1017/S0022112086000332

Craster, R.V., & Matar, O.K. (2009). Dynamics and stability of thin
liquid films. *Reviews of Modern Physics*, 81(3), 1131–1198.
https://doi.org/10.1103/RevModPhys.81.1131

Decré, M.M.J., & Baret, J.C. (2003). Gravity-driven flows of viscous
liquids over two-dimensional topographies. *Journal of Fluid Mechanics*,
487, 147–166. https://doi.org/10.1017/S0022112003004611

de Gennes, P.G. (1985). Wetting: statics and dynamics. *Reviews of
Modern Physics*, 57(3), 827–863.
https://doi.org/10.1103/RevModPhys.57.827

Eggers, J. (2005). Contact line motion for partially wetting fluids.
*Physical Review E*, 72, 061605.
https://doi.org/10.1103/PhysRevE.72.061605

Hocking, L.M. (1992). Rival contact-angle models and the spreading of
drops. *Journal of Fluid Mechanics*, 239, 671–681.
https://doi.org/10.1017/S0022112092004579

Johnson, M.F.G., Schluter, R.A., Miksis, M.J., & Bankoff, S.G. (1999).
Experimental study of rivulet formation on an inclined plate by
fluorescent imaging. *Journal of Fluid Mechanics*, 394, 339–354.
https://doi.org/10.1017/S0022112099005765

Kakimpa, B., Morvan, H.P., & Hibberd, S. (2016). The depth-averaged
numerical simulation of laminar thin-film flows with capillary waves.
*Journal of Engineering for Gas Turbines and Power*, 138(11), 112501.
https://doi.org/10.1115/1.4033471

Lan, H., Wegener, J.L., Armaly, B.F., & Drallmeier, J.A. (2010).
Developing laminar gravity-driven thin liquid film flow down an
inclined plane. *Journal of Fluids Engineering*, 132(8), 081301.
https://doi.org/10.1115/1.4001947

Meredith, K.V., Heather, A., de Vries, J., & Xin, Y. (2011). A
numerical model for partially-wetted flow of thin liquid films. *WIT
Transactions on Engineering Sciences*, 70, 239–250.
https://doi.org/10.2495/MPF110201

Mouvanal, S., Singh, K., Jefferson-Loveday, R., Ambrose, S., Eastwick,
C., Johnson, K., & Jacobs, A. (2022). Coupled Eulerian thin film model
and Lagrangian discrete phase model to predict film thickness inside
an aero-engine bearing chamber. *GPPS-TC-2022-0068*.
https://doi.org/10.33737/GPPS22-TC-68

Nicoli, A. (2020). *Development and application of a fully coupled
Eulerian thin film/discrete phase approach to a simplified aeroengine
bearing chamber* [PhD thesis, University of Nottingham].
https://eprints.nottingham.ac.uk/61498/

Pahlavan, A.A., Cueto-Felgueroso, L., McKinley, G.H., & Juanes, R.
(2015). Thin films in partial wetting: internal selection of contact-
line dynamics. *Physical Review Letters*, 115, 034502.
https://doi.org/10.1103/PhysRevLett.115.034502

Roy, R.V., Roberts, A.J., & Simpson, M.E. (2002). A lubrication model
of coating flows over a curved substrate in space. *Journal of Fluid
Mechanics*, 454, 235–261.
https://doi.org/10.1017/S0022112001007133

Shikhmurzaev, Y.D. (1996). Dynamic contact angles and flow in the
vicinity of moving contact line. *AIChE Journal*, 42(3), 601–612.
https://doi.org/10.1002/aic.690420302

Singh, R.K., Galvin, J.E., & Sun, X. (2015). Three-dimensional
simulation of rivulet and film flows over an inclined plate.
*Chemical Engineering Science*, 142.
https://doi.org/10.1016/j.ces.2015.11.029

Singh, K., Sharabi, M., Jefferson-Loveday, R., Ambrose, S., Eastwick,
C., Cao, J., & Jacobs, A. (2021). Modeling of partially wetting liquid
film using an enhanced thin film model for aero-engine bearing chamber
applications. *Journal of Engineering for Gas Turbines and Power*,
143(4), 041001. https://doi.org/10.1115/1.4049663

Snoeijer, J.H., & Andreotti, B. (2013). Moving contact lines: scales,
regimes, and dynamical transitions. *Annual Review of Fluid Mechanics*,
45, 269–292. https://doi.org/10.1146/annurev-fluid-011212-140734

Sussman, M., Smereka, P., & Osher, S. (1994). A level set approach
for computing solutions to incompressible two-phase flow. *Journal of
Computational Physics*, 114, 146–159.
https://doi.org/10.1006/jcph.1994.1155

Voinov, O.V. (1976). Hydrodynamics of wetting. *Fluid Dynamics*, 11,
714–721. https://doi.org/10.1007/BF01012963

---

## Notes on coverage gaps in the literature search

- "Bonart 2017" rivulet fluorescent-imaging paper was not confirmed
  in the search; the Johnson et al. (1999) JFM paper is the
  established reference for that experimental method.
- Bristot's standalone Nottingham thesis on ETFM curved-surface work
  was not confirmed; the Nicoli (2020) thesis is the verified source
  for that group's ETFM bearing-chamber work.
- ASME Digital Collection papers (Singh 2021, Kakimpa 2016, the
  GT2019 conferences) returned 403 in search; details cited from
  abstracts and verified via DOIs. Full-text equation forms should
  be cross-checked before submission.
