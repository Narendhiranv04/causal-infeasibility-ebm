# Proof-of-Concept Plan: Intervention-Grounded Infeasibility + Structured Repair

## 0. Purpose

This document is the execution contract for the proof-of-concept (PoC).

The PoC must answer, in the smallest possible sequence of experiments:

1. Can next-action geometric infeasibility be defined and certified from an action envelope?
2. Can controlled interventions recover the true minimal corrective set?
3. What mathematical structure does the intervention-response function actually have?
4. Is a unary model sufficient, or are pairwise/higher-order interactions genuinely needed?
5. If pairwise structure is sufficient, does a QUBO represent the repair problem correctly?
6. If a QUBO is valid, does a Hopfield network solve it reliably enough to justify using Hopfield?
7. Do the same conclusions survive in 3D MuJoCo with realistic manipulation scenes?
8. Is the result strong enough to justify building the full dataset and later adding learning from perception?

The PoC is deliberately **oracle-first**.  
Do not add perception learning until the geometry, intervention logic, and optimization assumptions have survived the tests below.

---

# 1. Strict Scope

## In scope

- Next-action infeasibility only.
- Geometric infeasibility relative to a complete action envelope.
- Entity-level geometric conflict.
- Controlled interventions.
- Minimal blocking / corrective sets.
- Unary, pairwise, and sampled higher-order intervention analysis.
- Exact subset search as ground truth.
- QUBO construction only when mathematically justified.
- Hopfield only as a candidate QUBO solver.
- 2D mathematical sanity checks.
- 3D MuJoCo proof-of-concept.
- Initial 3D scene families:
  - cupboard insertion,
  - cupboard extraction,
  - hinged box opening.
- Multi-blocker scenes, including more than two causal objects.

## Explicitly out of scope for this PoC

- Long-horizon plan suffix reasoning.
- Action "phases" or manually defined phase bins.
- Slot Attention.
- Flow matching.
- Equilibrium Propagation.
- VLM planning.
- Learned world models.
- Full RGB-D learning.
- Full dataset scale.
- Drawer scenarios.
- Multi-candidate grasp benchmark design.
- Fluid simulation.
- End-to-end policy learning.
- Learned repair ranking before the oracle formulation is validated.

---

# 2. Non-Negotiable Rules for Claude Code

These rules are strict.

## 2.1 Stage-by-stage execution

Claude Code must work on **exactly one stage at a time**.

It must:

1. Read this `plan.md`.
2. Identify `CURRENT_STAGE`.
3. Implement only that stage.
4. Run that stage's tests and experiments.
5. Produce the mandatory stage summary.
6. STOP.

It must **not** begin the next stage in the same turn.

Proceeding requires an explicit user message such as:

> Continue to Stage 2.

No inferred approval.  
No automatic continuation.

---

## 2.2 No premature files

Each stage has an explicit `Allowed files` list.

Claude Code may:

- create files listed for the current stage,
- modify files explicitly listed as modifiable,
- update only the status/checklist section of this `plan.md`.

Claude Code must **not create any unlisted source/config/document file**.

If an additional file is genuinely required:

1. STOP.
2. Explain why it is required.
3. Ask for approval.
4. Do not create it until approved.

No:
- scratch scripts,
- duplicate implementations,
- temporary Python modules,
- extra markdown notes,
- notebooks,
- backup files,
- `_old.py`,
- `_new.py`,
- `final_v2.py`,
- miscellaneous helper files.

---

## 2.3 File-size rule

Every code file must remain:

- target: **150–350 lines**,
- soft warning: **> 350 lines**,
- hard maximum: **500 lines**.

A file approaching 400 lines must be reviewed before adding more code.

Splitting is allowed only when there is a clear responsibility boundary.

Never split a file merely to satisfy the line count if the resulting files become meaningless fragments.

---

## 2.4 Keep the repository simple

Do not introduce:

- Hydra,
- complex configuration frameworks,
- plugin systems,
- registries,
- abstract factories,
- dependency injection,
- database infrastructure,
- Docker,
- notebooks,
- distributed training,
- logging frameworks,
- experiment-management frameworks.

Plain Python first.

Use dataclasses and small pure functions where practical.

---

## 2.5 No silent refactors

A later stage must not silently rewrite earlier validated logic.

If a prior module must change:

1. explain why,
2. keep the change minimal,
3. rerun all affected earlier tests,
4. report the regression status in the stage summary.

---

## 2.6 Reproducibility

All randomized experiments must use explicit seeds.

Default seed:

```text
7
```

If multiple seeds are used, record them in the stage metrics.

Every experiment must be runnable from one command documented in the stage summary.

---

## 2.7 Generated output

Runtime outputs go only under:

```text
out/
```

Use:

```text
out/s1/
out/s2/
...
```

Generated outputs are not source code.

Do not create documentation inside `out/`.

At most produce:

- `metrics.json`
- one or two useful plots per stage when needed

Do not dump per-scene files unless the stage explicitly requires them.

---

# 3. Core Mathematical Contract

This notation is fixed for the PoC.

## Scene and action

- \(s\): full simulator state.
- \(a\): next manipulation action.
- \(\mathcal{E}(a,s)\): complete action envelope.

The action envelope is:

\[
\mathcal{E}(a,s)
=
\bigcup_{\tau \in [0,1]} \mathcal{G}_a(\tau)
\]

where \(\mathcal{G}_a(\tau)\) is the relevant geometry occupied during execution.

The envelope may include:

- manipulated object,
- gripper,
- relevant end-effector geometry.

Numerical trajectory samples may approximate the continuous envelope.

They are **not semantic action phases** and must never be exposed as hand-coded phase bins.

---

## Entities

Let visible / known scene entities be:

\[
O = \{o_1,\ldots,o_N\}.
\]

Each entity has geometry:

\[
\mathcal{G}(o_i).
\]

---

## Entity conflict

For the PoC, define one scalar conflict value per entity:

\[
c_i(s,a)
=
\phi\left(
\mathcal{E}(a,s),
\mathcal{G}(o_i)
\right).
\]

The exact choice of \(\phi\) must be stated per stage.

Desired properties:

- \(c_i = 0\): no geometric interference,
- \(c_i > 0\): geometric interference,
- larger value: stronger conflict under the chosen metric.

Conflict vector:

\[
c(s,a)
=
[c_1,\ldots,c_N]^T.
\]

Do not invent a neural conflict predictor in the PoC.

The oracle provides the truth.

---

## Candidate interventions

Let:

\[
\mathcal{I}
=
\{I_1,\ldots,I_P\}.
\]

Each \(I_p\) is a candidate corrective intervention.

Examples:

- relocate blocker to an allowed staging pose,
- shift target before retry,
- reposition box,
- move a movable neighboring object.

Keep diagnostic interventions separate from executable repairs.

Example:

- `disable/remove wall` may be used to establish causality,
- it is not an executable robot repair.

---

## Intervention subset

Binary variable:

\[
x_p \in \{0,1\}
\]

means whether intervention \(I_p\) is selected.

Vector:

\[
x=[x_1,\ldots,x_P]^T.
\]

Equivalent selected set:

\[
S(x)
=
\{I_p : x_p=1\}.
\]

---

## Graded conflict after interventions

After applying set \(S\):

\[
c^S
=
c(do(S)(s),a).
\]

Define a graded remaining-conflict function:

\[
G(S)
=
g(c^S).
\]

Do **not** assume the final form of \(g\) before Stage 1.

Candidate choices to compare include:

\[
\|c^S\|_1,
\qquad
\|c^S\|_2^2,
\qquad
\max_i c_i^S.
\]

The chosen form must have a physical interpretation.

---

## Binary feasibility oracle

\[
F(S)
=
\begin{cases}
0, & \text{target action is feasible after } S\\
1, & \text{target action remains infeasible after } S.
\end{cases}
\]

This is the final feasibility certification.

---

## Repair length

\[
K(S)
=
\text{number of primitive repair actions in } S.
\]

For the early PoC, one candidate intervention may count as one macro-action.

If macro lengths differ, store explicit integer lengths.

---

## Oracle repair objective

Define:

\[
J^*(S)
=
B\,F(S)+K(S)
\]

with:

\[
B > K_{\max}.
\]

This guarantees:

1. every feasible repair scores below every infeasible repair,
2. among feasible repairs, the shortest repair is preferred.

Ground-truth minimal repair:

\[
S^*
=
\arg\min_S J^*(S).
\]

For small PoC problems, compute \(S^*\) by exhaustive enumeration.

---

# 4. Critical Scientific Distinction

The PoC must never confuse:

## Blocker cardinality

How many interventions are needed:

\[
|S^*|.
\]

This may be:

\[
1,2,3,4,5,6,\ldots
\]

## Interaction order

How complex the set function \(G(S)\) is.

A general pseudo-Boolean expansion is:

\[
G(x)
=
G_0
+
\sum_p \alpha_p x_p
+
\sum_{p<q}\beta_{pq}x_px_q
+
\sum_{p<q<r}\gamma_{pqr}x_px_qx_r
+\cdots
\]

A five-object repair can still be exactly unary or pairwise.

Do not limit dataset blocker count merely because a QUBO is pairwise.

---

# 5. Decision Logic for the Entire PoC

The project must be allowed to simplify or reject the current architecture.

## Outcome A — Unary structure is sufficient

If realistic intervention responses are essentially:

\[
G(x)
\approx
G_0+\sum_p\alpha_px_p,
\]

then pairwise QUBO/Hopfield is probably unnecessary.

Prefer a simpler exact discrete solver.

This is a valid scientific result.

---

## Outcome B — Pairwise structure matters, higher order is negligible

If:

\[
G(x)
\approx
G_0
+
\sum_p\alpha_px_p
+
\sum_{p<q}\beta_{pq}x_px_q
\]

and adding triples gives little improvement, QUBO is justified.

Then test Hopfield against exact QUBO.

---

## Outcome C — Higher-order interactions are important

If third/higher-order terms materially reduce reconstruction error, a vanilla pairwise Hopfield model is not sufficient.

Possible later directions:

- higher-order Hopfield,
- quadratization with auxiliary variables,
- factor-graph / exact discrete solver,
- other structured optimization.

Do not force pairwise QUBO.

---

## Outcome D — QUBO is correct but Hopfield is unreliable

Keep the QUBO formulation.

Drop Hopfield.

Use exact / approximate QUBO, MIQP, CP-SAT, annealing, or another solver.

---

# 6. Repository Structure

This is the maximum intended structure for the full PoC.

Do not create all files on Day 1.

Create them only when their stage is active.

```text
.
├── plan.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── poc/
│       ├── __init__.py
│       ├── types.py
│       ├── toy2d.py
│       ├── energy.py
│       ├── hopfield.py
│       ├── mj_scene.py
│       ├── envelope.py
│       ├── oracle.py
│       └── cases.py
├── scripts/
│   ├── s1_toy.py
│   ├── s2_energy.py
│   ├── s3_hopfield.py
│   ├── s4_mujoco.py
│   └── s5_validate.py
└── tests/
    ├── test_core.py
    ├── test_toy2d.py
    ├── test_energy.py
    ├── test_hopfield.py
    └── test_mujoco.py
```

Generated at runtime only:

```text
out/
```

No `data/` directory is needed for this PoC unless Stage 5 proves that a dataset should be generated.

---

# 7. Current Stage Status

```text
CURRENT_STAGE: 4
```

- [x] Stage 0 — repository contract + mathematical types (gate: PASS)
- [x] Stage 1 — 2D oracle and exhaustive intervention truth (gate: PASS)
- [x] Stage 2 — interaction-order analysis + exact QUBO (gate: PASS; class B, narrow)
- [x] Stage 3 — Hopfield solver validation (gate: PASS on accuracy targets; exact solver stays reference)
- [x] Stage 4 — minimal 3D MuJoCo envelope oracle (gate: PASS WITH CHANGE — certified signed-distance wrapper; awaiting approval for Stage 5)
- [ ] Stage 5 — 3D multi-blocker validation + final PoC verdict

Only the current stage may be implemented.

---

# 8. Stage 0 — Repository Contract + Mathematical Types

## Goal

Create the minimal project scaffold and lock the interfaces before implementing geometry.

No optimization yet.

No MuJoCo yet.

No Hopfield yet.

---

## Questions answered

- Are the core concepts represented consistently?
- Are diagnostic interventions and executable repairs distinct?
- Is the repository structure clean enough to grow stage by stage?

---

## Allowed files

Create:

```text
pyproject.toml
.gitignore
src/poc/__init__.py
src/poc/types.py
tests/test_core.py
```

Modify:

```text
plan.md
```

No other files.

---

## `types.py` responsibilities

Only data structures / enums needed across stages.

Expected concepts:

- `ActionSpec`
- `Entity2D` or generic entity metadata
- `Intervention`
- `InterventionKind`
- `RepairResult`
- `StageMetrics` only if truly needed

Do not place geometry or solver logic in `types.py`.

Keep it small.

---

## Dependencies

Use the smallest reasonable set.

Initial dependencies may include:

- Python >= 3.11
- numpy
- pytest

Do not add MuJoCo or Shapely until the stage that requires them.

---

## Acceptance criteria

- package imports cleanly,
- tests pass,
- no file exceeds 250 lines in this stage,
- notation in code matches this document,
- no stage-1 logic is implemented.

---

## Mandatory Stage 0 summary

Report:

```text
STAGE 0 COMPLETE

Files created:
- ...

Files modified:
- ...

Tests:
- command
- passed / failed

Interface decisions:
- ...

Open issues:
- ...

Gate:
PASS / FAIL

STOPPED.
Awaiting explicit approval for Stage 1.
```

Then stop.

---

# 9. Stage 1 — 2D Oracle + Exhaustive Intervention Ground Truth

## Purpose

This is the mathematical unit test.

It is **not** intended as publishable evidence.

Use 2D so every intervention subset can be inspected and exhaustively verified.

---

## Core question

Can we define:

- an action envelope,
- geometric conflict,
- corrective interventions,
- exact feasibility,
- exact minimal repair,

without any learning or Hopfield assumptions?

---

## 2D scene types

Implement exactly two initially:

### A. Translational envelope

A carried rectangle moves along a fixed 2D path.

Envelope:

\[
\mathcal{E}
=
\bigcup_{\tau\in[0,1]}
\mathcal{G}_{object+gripper}(\tau)
\]

Blockers intersect the envelope.

---

### B. Hinged envelope

A rectangular panel rotates around a hinge.

The envelope is the union of the panel geometry over the rotation.

No hand-defined action phases.

Dense angle sampling is allowed only as numerical approximation.

---

## 2D interventions

Start with executable-style interventions:

- relocate blocker to one of a few predetermined safe staging positions,
- optionally shift the target object when the blocker is fixed.

Keep diagnostic `remove/disable entity` separate.

---

## Required case families

Generate deterministic examples for:

1. feasible hard negative,
2. one blocker,
3. two independent blockers,
4. five independent blockers,
5. redundant alternative repairs,
6. one deliberately coupled repair case.

The coupled case must be physically interpretable.

Do not create an artificial coupling only to make Hopfield look useful.

---

## Ground truth

For each case:

1. enumerate all intervention subsets,
2. apply each subset,
3. recompute geometry,
4. calculate \(G(S)\),
5. calculate \(F(S)\),
6. calculate \(J^*(S)\),
7. obtain all minimal optimal repairs.

For Stage 1, keep:

\[
P \le 10
\]

so exhaustive enumeration is trivial.

---

## Graded conflict study

Compare at least:

\[
G_1(S)=\|c^S\|_1
\]

and

\[
G_2(S)=\|c^S\|_2^2.
\]

Choose one for later stages based on:

- interpretability,
- monotonic behavior,
- sensitivity to independent blockers,
- numerical stability.

Do not choose squared \(L_2\) merely because it creates a quadratic expression.

---

## Allowed files

Create:

```text
src/poc/toy2d.py
scripts/s1_toy.py
tests/test_toy2d.py
```

Modify if necessary:

```text
src/poc/types.py
pyproject.toml
plan.md
```

No energy/QUBO/Hopfield module yet.

---

## Stage 1 outputs

Runtime:

```text
out/s1/metrics.json
```

Optional:

```text
out/s1/cases.png
```

Only create the plot if it materially helps inspect geometry.

---

## Acceptance criteria

Must demonstrate:

- feasible scenes have zero conflict,
- blocker insertion produces positive conflict,
- near-boundary feasible/infeasible cases behave correctly,
- exhaustive subset search returns expected minimal repair,
- five-blocker cases are handled correctly,
- repeated runs are deterministic.

---

## Stop conditions

STOP and report failure if:

- envelope geometry is inconsistent,
- minimal repair does not match manual expectation,
- intervention application mutates state irreversibly across subset tests,
- numerical approximation causes unstable labels.

Do not proceed to Stage 2 until fixed.

---

# 10. Stage 2 — Interaction-Order Analysis + Exact QUBO

## Purpose

Determine whether QUBO is mathematically justified.

This stage must **not assume** pairwise structure.

---

## Part A — Exact pseudo-Boolean analysis

For each Stage 1 scene, use the exhaustive table:

\[
S \mapsto G(S)
\]

to analyze interaction order.

Fit / reconstruct:

### Unary

\[
\hat G_1(x)
=
G_0+\sum_p\alpha_px_p
\]

### Pairwise

\[
\hat G_2(x)
=
G_0
+
\sum_p\alpha_px_p
+
\sum_{p<q}\beta_{pq}x_px_q
\]

### Third-order

\[
\hat G_3(x)
=
\hat G_2(x)
+
\sum_{p<q<r}\gamma_{pqr}x_px_qx_r
\]

Use exact finite-difference / Möbius coefficients where possible.

For pairwise interaction:

\[
\beta_{pq}
=
G(p,q)-G(p)-G(q)+G(\emptyset).
\]

---

## Metrics

For each approximation report:

- maximum absolute error,
- mean absolute error,
- RMSE,
- argmin agreement with exact \(J^*\),
- repair feasibility agreement.

Do not hide cases where higher order matters.

---

## Part B — Exact QUBO construction

Only for cases where pairwise structure is exact or sufficiently accurate, construct:

\[
H(x)
=
\kappa \hat G_2(x)
+
\lambda K(x).
\]

Rewrite as:

\[
H(x)
=
x^TQx+q^Tx+c.
\]

Implement exact solution by exhaustive enumeration.

No third-party QUBO solver is needed yet.

---

## Required decision at end of Stage 2

Classify the PoC into one of:

```text
A: unary sufficient
B: pairwise/QUBO justified
C: significant higher-order structure
```

This classification must be based on measurements.

---

## Important anti-bias rule

If the realistic 2D cases are unary, say so.

Do not invent pairwise couplings solely to justify Hopfield.

If only mutually exclusive repair choices create pairwise terms, distinguish:

- physical interaction,
- repair-choice constraint.

These are not the same phenomenon.

---

## Allowed files

Create:

```text
src/poc/energy.py
scripts/s2_energy.py
tests/test_energy.py
```

Modify if necessary:

```text
src/poc/types.py
src/poc/toy2d.py
plan.md
```

No Hopfield file yet.

---

## Stage 2 outputs

```text
out/s2/metrics.json
```

Optional:

```text
out/s2/order_error.png
```

---

## Acceptance criteria

For known synthetic test functions:

- unary coefficients reconstruct unary truth exactly,
- pairwise finite-difference coefficients reconstruct pairwise truth exactly,
- third-order test detects a known triple term,
- exact QUBO argmin matches exhaustive oracle on all pairwise-valid cases.

---

## Kill criterion

If exact QUBO does **not** reproduce the intended repair on pairwise-valid cases:

STOP.

The QUBO formulation is wrong.

Do not implement Hopfield.

---

# 11. Stage 3 — Hopfield Solver Validation

## Purpose

Test Hopfield only after QUBO has been validated independently.

Hopfield is a candidate solver, not a premise.

---

## Required implementation

Implement classical binary/spin Hopfield optimization with:

- symmetric weights,
- explicit QUBO-to-Hopfield conversion,
- asynchronous updates,
- deterministic update order option,
- randomized restarts,
- final energy reporting.

Keep the implementation small and transparent.

No learned weights.

The Hopfield network receives the oracle/exact QUBO.

---

## Baseline

Exact exhaustive QUBO solution from Stage 2 is ground truth.

For each problem instance compare:

\[
x_{Hopfield}
\]

to:

\[
x_{exact}.
\]

---

## Experiments

Test across:

- 1–8 selected blockers,
- increasing number of candidate interventions,
- independent cases,
- redundant choices,
- pairwise synergy cases if physically justified,
- random pairwise toy energies with known exact minima.

---

## Metrics

Report:

- exact-optimum recovery rate,
- feasible-repair rate,
- minimal-repair rate,
- energy gap to exact optimum,
- number of updates,
- number of restarts,
- runtime.

---

## Allowed files

Create:

```text
src/poc/hopfield.py
scripts/s3_hopfield.py
tests/test_hopfield.py
```

Modify:

```text
src/poc/energy.py
plan.md
```

No MuJoCo files yet.

---

## Acceptance / decision

Hopfield is retained as the preferred solver only if it is competitive enough to justify the added approximation.

Suggested PoC target:

- >= 95% exact-optimum recovery on the intended small/medium toy distribution with a reasonable restart budget,
- >= 99% feasible-repair recovery.

These are evaluation targets, not numbers to fake.

If Hopfield underperforms:

```text
QUBO VALID
HOPFIELD NOT JUSTIFIED
```

Proceed to MuJoCo using the exact solver as the reference.

Do not force Hopfield into later stages.

---

# 12. Stage 4 — Minimal 3D MuJoCo Envelope Oracle

## Purpose

Determine whether the mathematical formulation survives realistic 3D manipulation geometry.

No learned perception.

No dataset scaling.

Use privileged MuJoCo geometry.

---

## 3D scene families

Implement only these initially:

### A. Cupboard insertion

Target object and gripper/object composite must enter a deep cupboard.

Infeasibility examples:

- side wall,
- top shelf / overhang,
- neighboring movable object,
- structural lip or frame.

---

### B. Cupboard extraction

Target object is visible and initially graspable.

Extraction envelope intersects:

- side wall,
- top shelf,
- neighboring objects,
- cupboard frame.

---

### C. Hinged box opening

The lid starts collision-free.

The continuous swept envelope intersects:

- neighboring object,
- fixed overhang,
- another structural entity.

---

## No action phases

The envelope is continuous conceptually.

Numerically:

- sample trajectory densely enough for stable geometry,
- refine sampling until labels stop changing,
- never expose samples as semantic phases.

---

## Robot detail

Use the simplest model that still preserves the real geometric claim.

Preferred order:

1. prescribed SE(3) manipulated-object + gripper envelope,
2. then xArm7-compatible trajectory if needed for credibility.

Do not introduce IK complexity until the envelope oracle itself is stable.

This PoC is about geometric action-envelope infeasibility, not IK failure.

---

## MuJoCo conflict oracle

Use exact MuJoCo geometry queries where applicable.

For entity-level conflict:

\[
c_i
=
\phi(\mathcal{E},o_i).
\]

The implementation must distinguish:

- no intersection,
- intersection,
- near-boundary clearance.

Sampling density must be validated.

---

## Diagnostic vs executable interventions

Diagnostic examples:

- disable wall collision,
- temporarily remove shelf.

Executable repair examples:

- relocate movable blocker,
- shift target before retry,
- move box to a different pose.

Never report a diagnostic structural removal as an executable robot repair.

---

## Required tests

For each scene family:

- one feasible case,
- one single-blocker infeasible case,
- one near-boundary pair,
- one immovable-cause case where repair target differs from cause.

---

## Allowed files

Create:

```text
src/poc/mj_scene.py
src/poc/envelope.py
src/poc/oracle.py
src/poc/cases.py
scripts/s4_mujoco.py
tests/test_mujoco.py
```

Modify:

```text
pyproject.toml
src/poc/types.py
plan.md
```

Avoid adding XML files unless absolutely necessary.

Prefer programmatic MJCF generation or one compact model asset.

If a separate MJCF/XML file is required, STOP and ask for approval first.

---

## Stage 4 outputs

```text
out/s4/metrics.json
```

Optional:

```text
out/s4/scene.png
```

No dataset dump.

---

## Acceptance criteria

- all three scene families run deterministically,
- envelope conflict is stable under increased sampling density,
- near-boundary pairs flip labels for correct geometric reasons,
- diagnostic intervention identifies the expected cause,
- executable repair restores feasibility,
- cause and repair target can differ,
- no semantic action phases are used.

---

# 13. Stage 5 — 3D Multi-Blocker Validation + Final PoC Verdict

## Purpose

This is the final proof-of-concept stage.

The goal is to decide whether the methodology is strong enough to justify building the real dataset.

---

## Scene scaling

Use the Stage 4 generators.

Create controlled scenes with:

\[
|S^*| = 1,2,3,4,5,6+
\]

where physically reasonable.

Prioritize:

- independent movable blockers,
- structural + movable combinations,
- redundant repair options,
- multiple valid minimal repairs,
- physically justified pair interactions.

Do not create artificial complexity solely to produce nonzero \(\beta\).

---

## Critical experiment 1 — High-cardinality does not imply high-order

For scenes with 4–6+ required interventions, measure whether:

- unary model is sufficient,
- pairwise model is sufficient,
- third-order terms materially improve reconstruction.

Report interaction order separately from blocker count.

---

## Critical experiment 2 — Exact repair recovery

For every manageable scene:

1. exhaustively enumerate candidate intervention subsets,
2. compute exact \(J^*\),
3. obtain exact minimal repair set(s),
4. compare structured approximation.

If pairwise-valid:

- exact QUBO must match oracle minimal repair.

---

## Critical experiment 3 — Hopfield only if retained

If Stage 3 retained Hopfield:

- run the same 3D QUBOs with Hopfield,
- compare to exact QUBO.

If Stage 3 rejected Hopfield:

- do not revive it here without a new reason.

---

## Critical experiment 4 — Scaling trend

Report performance against:

\[
|S^*|
\]

and:

\[
P
\]

number of candidate interventions.

At minimum report:

- feasibility recovery,
- minimality,
- exact-set match,
- energy approximation error,
- solver runtime.

---

## Critical experiment 5 — Near-boundary robustness

Perturb geometry around the feasibility boundary.

Confirm:

- oracle labels are stable,
- interaction coefficients change smoothly where expected,
- the minimal repair does not flip due only to coarse envelope sampling.

---

## Allowed files

Create:

```text
scripts/s5_validate.py
```

Modify as needed:

```text
src/poc/cases.py
src/poc/oracle.py
src/poc/energy.py
src/poc/hopfield.py    # only if Hopfield survived Stage 3
tests/test_mujoco.py
plan.md
```

No new source module unless approved.

---

## Stage 5 outputs

```text
out/s5/metrics.json
out/s5/scaling.png
out/s5/order_error.png
```

Maximum three generated files.

---

# 14. Final PoC Decision Table

At the end of Stage 5, classify the project honestly.

## GREEN — proceed to full dataset

Requirements:

- 3D envelope oracle is stable,
- minimal causal/corrective sets are certifiable,
- multi-blocker scenes scale correctly,
- low-order structure is empirically supported,
- exact optimization recovers true minimal repairs,
- the chosen solver is reliable enough.

Then the next project phase can begin:

```text
full dataset generation
→ learned conflict/intervention-effect prediction
→ RGB-D deployment model
```

---

## YELLOW-A — geometry works, unary structure dominates

Conclusion:

- dataset idea is valid,
- causal intervention supervision is valid,
- QUBO/Hopfield is unnecessary or over-engineered.

Simplify the method before scaling.

---

## YELLOW-B — geometry + QUBO work, Hopfield fails

Conclusion:

- structured optimization is justified,
- Hopfield is not.

Use a better QUBO/discrete solver.

---

## YELLOW-C — higher-order interactions matter

Conclusion:

- pairwise QUBO is insufficient as-is.

Investigate:

- quadratization,
- higher-order energy,
- factor graph,
- other structured solver.

Do not scale the dataset under the false assumption that pairwise Hopfield is enough.

---

## RED — intervention formulation does not recover meaningful repairs

Do not proceed to learning.

Revisit:

- action definition,
- envelope definition,
- candidate intervention semantics,
- conflict metric,
- oracle correctness.

---

# 15. What Must Be Proven Sequentially

This is the core scientific logic.

## Proof 1 — Oracle correctness

Show:

\[
\text{geometry}
\rightarrow
\text{correct feasible/infeasible label}.
\]

If this fails, everything later is meaningless.

---

## Proof 2 — Intervention correctness

Show:

\[
do(S)
\rightarrow
\text{recomputed geometry}
\]

without hidden state corruption.

---

## Proof 3 — Minimal recourse correctness

Show exhaustive search returns:

\[
S^*
=
\arg\min_S J^*(S).
\]

---

## Proof 4 — Interaction-order evidence

Measure whether:

\[
G(S)
\]

is:

- unary,
- pairwise,
- higher order.

Do not assume.

---

## Proof 5 — QUBO representation

If pairwise:

\[
\arg\min H_{QUBO}
=
S^*.
\]

This proves the formulation.

---

## Proof 6 — Hopfield suitability

If using Hopfield:

\[
x_{Hopfield}
\approx
x_{QUBO}^{exact}
\]

often enough to justify the approximation.

---

## Proof 7 — 3D transfer

Show Proofs 1–6 are not artifacts of the 2D toy world.

---

## Proof 8 — Multi-blocker scaling

Show that increasing:

\[
|S^*|
\]

does not itself break the formulation.

Separate this from interaction order.

---

# 16. Stage Summary Format — Mandatory

At the end of every stage Claude Code must respond with exactly this structure:

```text
STAGE <N> COMPLETE

Goal:
- ...

Files created:
- ...

Files modified:
- ...

Command(s) run:
- ...

Tests:
- <passed>/<total>

Key measured results:
- ...

Assumptions validated:
- ...

Assumptions rejected / weakened:
- ...

Problems found:
- ...

Decision gate:
PASS / FAIL / PASS WITH CHANGE

Recommended next stage:
- Stage <N+1>, only after explicit approval.

STOPPED.
Awaiting user approval.
```

Do not continue after this summary.

---

# 17. Coding Style

- Prefer pure functions.
- Prefer dataclasses for structured records.
- Type-hint public functions.
- Keep functions short and single-purpose.
- Avoid inheritance unless genuinely required.
- No global mutable simulator state.
- Every intervention test must restore/copy state deterministically.
- Assertions should encode invariants, not replace error handling.
- Avoid magic thresholds; name them.
- Put constants close to the logic that owns them.
- Do not duplicate formulas across files.

---

# 18. Testing Rules

Every stage must contain tests for the mathematical invariants introduced in that stage.

Important invariant examples:

- feasible scene => zero / below-threshold conflict,
- intervention subset evaluation is order-independent when interventions commute,
- exhaustive optimum is deterministic,
- pairwise coefficient formula matches known synthetic function,
- QUBO and explicit polynomial energies agree numerically,
- Hopfield energy never increases under the chosen valid update rule,
- MuJoCo scene reset reproduces identical conflict,
- denser envelope sampling does not change stable labels.

Do not test implementation trivia.

Test the scientific contract.

---

# 19. Performance Rules

This PoC prioritizes correctness over speed.

However:

- exhaustive subset enumeration should remain limited to manageable \(P\),
- do not add multiprocessing before correctness is established,
- do not optimize MuJoCo queries prematurely,
- profile before optimizing.

No GPU requirement for the PoC.

---

# 20. What Not to Claim from the PoC

Even if Stage 5 succeeds, the PoC does **not** yet prove:

- RGB-D perception can predict the required quantities,
- sim-to-real transfer,
- long-horizon planning improvement,
- arbitrary higher-order generalization,
- Hopfield is superior to all discrete solvers,
- the full dataset is sufficient.

It proves only that the **underlying intervention + structured optimization formulation is viable enough to justify the full dataset and learning phase**.

---

# 21. Final Deliverable of This Plan

At the end of Stage 5, there must be enough evidence to answer:

### A. Is the action-envelope oracle meaningful in 3D?

### B. Can controlled interventions certify minimal repair sets?

### C. How does blocker cardinality scale?

### D. What interaction order is actually needed?

### E. Does QUBO faithfully represent the measured repair objective?

### F. Is Hopfield a useful solver, or should it be replaced?

### G. Is the architecture sufficiently justified to begin the real dataset?

Only after those questions are answered should the project move into:

```text
dataset scale
→ supervised prediction
→ perception
→ full robotics evaluation
```
