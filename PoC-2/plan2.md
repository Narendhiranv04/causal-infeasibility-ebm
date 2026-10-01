# plan2.md — Extended Proof of Concept for Causal Infeasibility and Structured Recourse

## 0. Purpose of PoC-2

PoC-1 established that the basic formulation is internally coherent:

\[
(s,a)
\rightarrow
\mathcal E(a,s)
\rightarrow
c,G,F
\rightarrow
do(S)
\rightarrow
V(S),F(S),K(S)
\rightarrow
S^*.
\]

It also showed that unary and pairwise intervention effects can be measured, that a pairwise
response can be written as a QUBO, and that a Hopfield network can optimize the resulting
quadratic energy on the tested small problems.

PoC-2 does **not** replace that formulation.

PoC-2 asks the missing scientific question:

\[
\boxed{
\text{Do realistic, benchmark-inspired manipulation repair problems naturally produce}
\\
\text{non-trivial structured recourse that actually benefits from pairwise optimization?}
}
\]

The central research problem remains:

\[
\boxed{
\text{Why is the proposed next robot action infeasible, which entities caused it,}
\\
\text{and what minimal executable intervention repairs it?}
}
\]

QUBO and Hopfield remain downstream hypotheses. They are not the project definition.

---

# 1. PoC-2 North Star

For a current scene \(s\) and a proposed **next manipulation action** \(a\):

1. determine whether the prescribed action envelope is feasible;
2. localize the original infeasibility to scene entities;
3. certify causal responsibility using controlled diagnostic interventions;
4. enumerate physically meaningful executable corrective interventions;
5. identify all minimal valid corrective sets;
6. measure how the corrective interventions interact;
7. determine whether unary reasoning is sufficient, pairwise reasoning is necessary, or higher-order structure appears;
8. only if pairwise structure is decision-relevant, evaluate the existing QUBO \(\rightarrow\) Hopfield pipeline.

The desired causal/recourse separation is:

\[
\boxed{
C^*=\text{minimal causal explanation of the original failure}
}
\]

versus

\[
\boxed{
S^*=\text{minimal valid executable corrective intervention set}
}
\]

and in general:

\[
\boxed{
C^* \neq S^*.
}
\]

A structural object can cause the failure while the executable repair acts on a different object.

---

# 2. Scope

## In scope

PoC-2 studies **next-action geometric/manipulation infeasibility**.

Task templates should be benchmark-inspired but remain simple, controlled MuJoCo geometry:

- make-space / shelf insertion,
- storage insertion,
- storage extraction,
- articulated opening,
- mixed clutter + repositioning cases.

Repair mechanisms initially allowed:

1. relocate a movable object to one of several candidate placement regions;
2. reposition the target/container before retrying the same skill;
3. clear an occupied region through relocation;
4. choose among alternative/substitutable valid repairs.

The three non-additivity mechanisms PoC-2 explicitly investigates are:

### Mechanism A — Envelope-changing repair

A target/container repositioning changes the future action envelope and therefore changes which
objects matter.

### Mechanism B — Repair compatibility / placement competition

Two individually valid repairs cannot coexist because their destination poses overlap or otherwise
make the post-repair scene invalid.

### Mechanism C — Alternative / substitutable repair

Two different interventions solve the same conflict, so selecting both is redundant.

These three mechanisms are enough for PoC-2.

Do **not** add more failure mechanisms merely to create complicated optimization.

## Out of scope

PoC-2 does not include:

- RGB-D learning,
- neural networks,
- EBM training,
- Slot Attention,
- VLM planning,
- world models,
- sim-to-real,
- long-horizon suffix reasoning,
- robot IK failure,
- grasp-search failure,
- dynamics or force control,
- full benchmark asset integration,
- photorealistic rendering,
- large-scale dataset generation,
- learned intervention ranking,
- new QUBO solvers,
- new Hopfield variants.

Do not introduce any of these without a new explicit plan.

---

# 3. Relationship to PoC-1

PoC-1 is a **frozen scientific baseline**.

After Stage 0 of this plan:

```text
PoC-1/
```

must contain the complete current PoC implementation at the frozen validated commit.

PoC-2 may **import** stable PoC-1 functionality, but it must never silently modify PoC-1.

PoC-1 provides trusted implementations for:

- geometric entities and MuJoCo distance queries,
- action-envelope sampling,
- \(d,p,c,G,F\) oracle semantics,
- the validated signed-distance wrapper,
- exact Stage-1/2 energy utilities,
- QUBO construction,
- Hopfield conversion and descent.

PoC-2 owns only the new repair-space and benchmark-inspired structural-recourse logic.

This prevents duplicated implementations while preserving PoC-1 reproducibility.

---

# 4. Repository Layout

After Stage 0, the intended repository is:

```text
.
├── .gitignore
├── PoC-1/
│   ├── plan.md
│   ├── pyproject.toml
│   ├── src/
│   │   └── poc/
│   │       └── ...                 # frozen PoC-1 package
│   ├── scripts/
│   │   └── ...                     # frozen PoC-1 scripts
│   └── tests/
│       └── ...                     # frozen PoC-1 tests
│
└── PoC-2/
    ├── plan2.md
    ├── pyproject.toml
    ├── src/
    │   └── poc2/
    │       ├── __init__.py
    │       ├── types.py
    │       ├── oracle.py
    │       ├── scenes.py
    │       ├── dataset.py
    │       ├── structure.py
    │       └── optimize.py
    ├── scripts/
    │   ├── s1_makespace.py
    │   ├── s2_sample.py
    │   ├── s3_structure.py
    │   ├── s4_tasks.py
    │   ├── s5_analyze.py
    │   └── s6_validate.py
    └── tests/
        ├── test_core.py
        ├── test_makespace.py
        ├── test_dataset.py
        ├── test_structure.py
        ├── test_tasks.py
        └── test_optimize.py
```

No additional source module may be created unless the current stage explicitly allows it.

If `scenes.py` approaches 350 lines during Stage 4, Claude must STOP and request approval before
splitting it. A single additional `tasks.py` split may then be approved.

---

# 5. Repository Hygiene Rules

These rules are strict.

## PoC-1

- PoC-1 becomes read-only after Stage 0.
- Never refactor PoC-1 during PoC-2.
- Never modify PoC-1 only to make a PoC-2 test easier.
- If a genuine PoC-1 bug is discovered, STOP and report it separately before changing anything.

## PoC-2

- Work exactly one stage at a time.
- Create only files authorized by that stage.
- Do not create scratch scripts.
- Do not create notebooks.
- Do not create `_old`, `_new`, `_final`, `_v2`, backup, or temporary source files.
- Do not introduce a configuration framework.
- No Hydra.
- No database server.
- No SQLite.
- No parquet dependency initially.
- No Docker.
- No experiment-management framework.
- No multiprocessing until correctness is established.
- No hidden caches committed to git.
- No generated `out/` content committed to git.

## Line limits

Target:

```text
150–300 lines per source file
```

Review threshold:

```text
350 lines
```

Hard maximum:

```text
500 lines
```

If a source file would exceed 500 lines, STOP before doing so.

---

# 6. Runtime and Environment

PoC-1 and PoC-2 are independent packages inside one repository.

Preferred development command from repository root:

```bash
PYTHONPATH=PoC-1/src:PoC-2/src python -m pytest -q --import-mode=importlib PoC-1/tests PoC-2/tests
```

(`--import-mode=importlib` is required because both packages contain `tests/test_core.py`.)

Stage scripts are run similarly, for example:

```bash
PYTHONPATH=PoC-1/src:PoC-2/src python PoC-2/scripts/s1_makespace.py
```

PoC-2 may import:

```python
from poc import ...
```

for validated PoC-1 components.

Do not copy PoC-1 geometry/QUBO/Hopfield implementations into PoC-2.

---

# 7. Dataset Storage Rules

PoC-2 is a structural oracle study, not yet the final learning dataset.

Use simple files only.

Runtime data goes under:

```text
PoC-2/out/sN/
```

Use:

- `*.json` for summaries,
- `*.jsonl` for scene records,
- at most one or two plots when a stage explicitly allows them.

Do not use a database.

Do not save rendered RGB images unless a stage explicitly requires a diagnostic figure.

Do not save thousands of per-subset files.

A scene record should contain enough deterministic parameters to reconstruct the scene and recompute
its exhaustive oracle table.

Large derived intervention tables should normally be recomputed rather than stored as separate files.

---

# 8. Core Mathematical Contract

## 8.1 Scene and action

Let:

\[
s=\text{current scene}
\]

and:

\[
a=\text{proposed next manipulation action}.
\]

The continuous action envelope is:

\[
\boxed{
\mathcal E(a,s)=\bigcup_{\tau\in[0,1]}\mathcal G_a(\tau).
}
\]

The oracle evaluates the prescribed skill/action.

It does **not** prove that no alternative motion path exists.

## 8.2 Entity-level conflict

For entity \(o_i\):

\[
d_i=\min_{\tau}d(\mathcal G_a(\tau),\mathcal G(o_i)).
\]

Using PoC-1 semantics:

\[
p_i=\max(0,-d_i)
\]

and:

\[
c_i=\max(0,p_i-\epsilon).
\]

The conflict vector is:

\[
c=[c_1,\ldots,c_N].
\]

Binary infeasibility:

\[
F=1[\exists i:c_i>0].
\]

Graded conflict:

\[
\boxed{
G=\sum_i c_i.
}
\]

## 8.3 Original causal explanation

Define the directly conflicting entities:

\[
B^0=\{o_i:c_i^0>0\}.
\]

A diagnostic intervention temporarily ignores an entity during target-action evaluation.

A minimal causal explanation is any inclusion-minimal set:

\[
C\subseteq B^0
\]

such that:

\[
F^{do(\text{ignore }C)}=0.
\]

Store all tied minimal causal sets:

\[
\boxed{
\mathcal C^*.
}
\]

Do not equate this with the corrective repair set.

Diagnostic removals are never executable robot repairs.

## 8.4 Executable intervention catalogue

Candidate executable repairs:

\[
\mathcal I=\{I_1,\ldots,I_P\}.
\]

Examples:

\[
I_{i,r}=\operatorname{relocate}(o_i,r)
\]

for object \(i\) and placement region \(r\), and:

\[
I_{\text{shift}}=\operatorname{reposition}(target).
\]

Each intervention has:

- intervention id,
- target entity,
- mechanism type,
- destination / shift parameters,
- integer repair cost.

## 8.5 Choice consistency

PoC-2 allows multiple candidate destinations for one object.

Therefore a subset can be syntactically inconsistent.

Define:

\[
\boxed{
M(S)=1
}
\]

iff the selected intervention set is choice-consistent.

At minimum:

- at most one relocation destination may be selected for one object;
- mutually exclusive target-repositioning alternatives may not be selected simultaneously.

Example:

\[
x_{A,r_1}+x_{A,r_2}\le1.
\]

Choice inconsistency is **not** a physical causal interaction.

It must be reported separately.

## 8.6 Static post-repair validity

For a choice-consistent intervention set:

\[
do(S)(s)
\]

produces a repaired scene.

Define:

\[
\boxed{
V(S)=1
}
\]

iff the resulting static scene is geometrically valid.

Examples of \(V(S)=0\):

- two relocated objects overlap,
- a relocated object intersects structural geometry,
- the repositioned target intersects a wall,
- two selected placement regions are simultaneously occupied by overlapping objects.

Again:

\[
V(S)
\]

is not the same as target-action feasibility.

## 8.7 Corrective feasibility

After applying \(S\), rerun the original proposed action.

Obtain:

\[
F(S),\quad G(S).
\]

A valid corrective set satisfies:

\[
M(S)=1,\qquad V(S)=1,\qquad F(S)=0.
\]

## 8.8 Minimal repair

Let:

\[
K(S)=\sum_{I_p\in S}\ell_p.
\]

The exact oracle correction is:

\[
\boxed{
\mathcal S^*
=
\arg\min_{\substack{S\\M(S)=1\\V(S)=1\\F(S)=0}}
K(S).
}
\]

Store all tied minimal repairs.

If no valid corrective set exists inside the candidate catalogue, mark the scene:

```text
NO_RECOURSE_IN_CATALOGUE
```

Do not invent a repair.

---

# 9. Intervention-Effect Structure

PoC-2 must distinguish the source of interaction.

## 9.1 Unary geometric effect

For compatible intervention \(p\):

\[
\alpha_p
=
G(\{p\})-G(\emptyset).
\]

## 9.2 Pairwise geometric effect

For a choice-compatible pair \(p,q\):

\[
\boxed{
\beta^G_{pq}
=
G(\{p,q\})
-G(\{p\})
-G(\{q\})
+G(\emptyset).
}
\]

Only compute this when all required subsets are choice-consistent.

Possible interpretation:

- negative: synergy / one intervention enables the benefit of another;
- positive: overlapping/substitutable benefit;
- zero: additive.

## 9.3 Higher-order geometric effect

For compatible triple \(p,q,r\):

\[
\gamma^G_{pqr}
=
G(p,q,r)
-G(p,q)-G(p,r)-G(q,r)
+G(p)+G(q)+G(r)-G(\emptyset).
\]

Do not assume this is zero.

Measure it.

## 9.4 Compatible-domain Möbius analysis

Because same-object alternative destinations create invalid binary combinations, PoC-2 must **not**
invent a geometric \(G(S)\) for contradictory action selections.

For every choice-consistent set \(T\), all subsets of \(T\) are also choice-consistent.

Therefore define compatible-domain Möbius coefficients:

\[
a(T)
=
\sum_{U\subseteq T}
(-1)^{|T|-|U|}G(U).
\]

Reconstruction on a choice-consistent set \(S\):

\[
\hat G_k(S)
=
\sum_{\substack{T\subseteq S\\|T|\le k}}a(T).
\]

Evaluate representation error **only on the choice-consistent domain**:

\[
\epsilon_k
=
\max_{S:M(S)=1}
|G(S)-\hat G_k(S)|.
\]

Do not fill contradictory states with fake geometry values.

---

# 10. Three Pairwise Sources Must Remain Separate

Whenever pairwise structure is reported, classify its source.

## A. Physical / conditional geometric interaction

Example:

\[
\operatorname{shiftTarget}
\times
\operatorname{moveBottle}.
\]

The target shift changes whether the bottle intersects the future action envelope.

This is measured through:

\[
\beta^G_{pq}.
\]

## B. Static repair compatibility

Example:

\[
A\rightarrow r_2
\]

and:

\[
B\rightarrow r_2
\]

are each valid alone but jointly cause overlap.

This belongs to:

\[
V(S),
\]

not to the original geometric conflict response \(G\).

## C. Choice constraint

Example:

\[
A\rightarrow r_1
\]

and:

\[
A\rightarrow r_2
\]

cannot both be selected.

This belongs to:

\[
M(S),
\]

not to physical causal coupling.

Never combine A, B, and C into one statistic and call all of them “causal pairwise interaction.”

---

# 11. What PoC-2 Must Measure

The most important statistic is not:

\[
P(\beta\neq0).
\]

It is:

\[
\boxed{
R_{\text{pair}}
=
P(
\mathcal S^*_{\text{unary}}\neq\mathcal S^*
\;\land\;
\mathcal S^*_{\text{pairwise}}=\mathcal S^*
).
}
\]

In words:

> How often does pairwise reasoning actually change an incorrect unary repair decision into the correct one?

Also measure:

- frequency of significant geometric pairwise terms;
- frequency of placement-compatibility edges;
- frequency of choice-constraint edges;
- higher-order geometric residuals;
- fraction of scenes where pairwise reasoning still fails;
- repair graph density;
- \(|\mathcal C^*|\);
- \(|\mathcal S^*|\);
- candidate count \(P\);
- number of tied minimal repairs;
- validity-binding rate;
- fraction of distractor repair options;
- family-wise breakdown.

---

# 12. QUBO Contract

QUBO is used only if the measured pairwise approximation is decision-sufficient.

The total pairwise repair energy may contain several clearly separated sources:

\[
\boxed{
H(x)
=
H_{\text{geom}}(x)
+
H_{\text{choice}}(x)
+
H_{\text{valid}}(x)
+
H_{\text{cost}}(x).
}
\]

Conceptually:

\[
H_{\text{geom}}
=
\kappa
\left(
G_0+\sum_i\alpha_i x_i+\sum_{i<j}\beta^G_{ij}x_ix_j
\right),
\]

\[
H_{\text{choice}}
=
\sum_{(p,q)\in E_M}
M_{pq}x_px_q,
\]

\[
H_{\text{valid}}
=
\sum_{(p,q)\in E_V}
V_{pq}x_px_q,
\]

and:

\[
H_{\text{cost}}
=
\lambda\sum_i\ell_i x_i.
\]

If this pairwise energy cannot reproduce the exact corrective decision, report it.

Do not add hidden higher-order approximations solely to preserve QUBO.

---

# 13. Hopfield Contract

PoC-2 does not redesign Hopfield.

Use the frozen PoC-1 implementation.

Hopfield is evaluated only after:

1. exact oracle repair is known;
2. pairwise QUBO is shown to be decision-sufficient;
3. exact QUBO optimum is known.

Comparison:

\[
x_{\text{Hopfield}}
\]

versus:

\[
x_{\text{QUBO}}^{\text{exact}}
\]

versus:

\[
\mathcal S^*.
\]

Hopfield success does not prove QUBO correctness.

QUBO correctness does not prove Hopfield usefulness.

---

# 14. Stage Execution Contract

At every stage Claude must:

1. read `PoC-2/plan2.md`;
2. identify `CURRENT_STAGE`;
3. implement exactly that stage;
4. create/modify only authorized files;
5. run all new tests;
6. run affected earlier PoC-2 tests;
7. run PoC-1 regression tests if PoC-1 APIs are imported;
8. produce the mandatory summary;
9. STOP.

Never begin the next stage automatically.

---

# 15. Stage Status

```text
CURRENT_STAGE: 3

[x] Stage 0 — Repository split + PoC-2 scaffold (gate: PASS)
[x] Stage 1 — Deterministic make-space recourse oracle (gate: PASS)
[x] Stage 2 — Small randomized make-space structural dataset (gate: PASS; C* = {B0} by construction under the
    separable per-entity oracle, kept as the localization label; unit costs)
[x] Stage 3 — Interaction structure and pairwise-necessity analysis (gate: PASS; rules pre-registered in
    poc2/structure.py; awaiting approval for Stage 4)
[ ] Stage 4 — Multi-task benchmark-inspired extension
[ ] Stage 5 — Cross-family structural recourse analysis
[ ] Stage 6 — Optimization evaluation + final PoC-2 verdict
```

---

# 16. Stage 0 — Repository Split + PoC-2 Scaffold

## Purpose

Preserve PoC-1 exactly and create a clean independent workspace for PoC-2.

No new scientific experiment in this stage.

## Required repository migration

Use `git mv`, not copy/delete, for tracked PoC-1 files.

Move the current validated PoC into:

```text
PoC-1/
```

Move:

```text
plan.md
pyproject.toml
src/
scripts/
tests/
```

into `PoC-1/`.

Keep the root:

```text
.gitignore
```

and update it only as needed to ignore:

```text
PoC-1/out/
PoC-2/out/
**/__pycache__/
.pytest_cache/
```

Do not move git metadata.

Do not move generated untracked `out/` data into git.

## Create PoC-2 scaffold

Create:

```text
PoC-2/plan2.md
PoC-2/pyproject.toml
PoC-2/src/poc2/__init__.py
PoC-2/src/poc2/types.py
PoC-2/tests/test_core.py
```

Nothing else.

## `types.py` responsibilities

Records only.

Suggested minimal records:

- `PlacementRegion`
- `RepairOption`
- `RepairGroup`
- `CauseResult`
- `RepairOracleResult`
- `SceneRecord`

Do not implement geometry, exhaustive search, Möbius analysis, QUBO, or dataset generation here.

Reuse PoC-1 entity/intervention records where sensible rather than duplicating them.

## Tests

Must verify:

- PoC-1 imports cleanly from new location;
- PoC-1 full test suite still passes unchanged;
- PoC-2 imports cleanly;
- PoC-2 records enforce basic invariants;
- no PoC-1 source file content changes during migration except path changes tracked by git.

## Allowed files

Create/modify only:

```text
.gitignore
PoC-1/...          # git mv only
PoC-2/plan2.md
PoC-2/pyproject.toml
PoC-2/src/poc2/__init__.py
PoC-2/src/poc2/types.py
PoC-2/tests/test_core.py
```

## Acceptance

PASS if:

- PoC-1 regression suite passes;
- PoC-2 imports;
- repository tree matches the intended layout;
- no duplicated PoC-1 implementation appears in PoC-2.

Then STOP.

---

# 17. Stage 1 — Deterministic Make-Space Recourse Oracle

## Purpose

Create one richer repair problem before randomization.

This is the unit test for PoC-2's new intervention space.

## Single task family

Use:

\[
\boxed{\text{make-space shelf insertion}}
\]

A target object must be inserted into a storage shelf / cabinet-like receptacle.

Use primitive MuJoCo boxes.

No external benchmark assets.

## Required scene ingredients

Support:

- 3–6 movable objects;
- a target insertion action;
- 2–4 candidate placement regions;
- movable blockers;
- structural geometry;
- irrelevant distractor objects;
- optional target-repositioning macro.

## Multiple relocation choices

An object may have multiple repair options:

\[
I_{A,r_1},\quad I_{A,r_2}.
\]

These are separate candidate interventions.

Selecting both must yield:

\[
M(S)=0.
\]

Do not resolve contradictory choices by arbitrary intervention order.

## Deterministic test cases

Create exactly these mechanism cases first:

1. independent blockers with several candidate destination regions;
2. placement competition: two objects can individually use the same region but not jointly;
3. substitutable repair: relocate blocker OR reposition target;
4. envelope-changing coupling: target repositioning makes another object become a blocker;
5. distractor candidates that do not belong in \(S^*\);
6. cause/repair-target mismatch with a structural cause.

Do not add more deterministic cases in Stage 1.

## Causal outputs

For every case compute:

- original \(c^0\);
- original \(F^0\);
- direct blocker set \(B^0\);
- all minimal diagnostic causal sets \(\mathcal C^*\).

Diagnostic cause computation must not use executable repair candidates.

## Repair outputs

For every candidate subset compute:

- \(M(S)\);
- if \(M(S)=1\), the post-repair scene;
- \(V(S)\);
- \(F(S)\);
- \(G(S)\);
- \(K(S)\).

Then obtain all:

\[
\mathcal S^*.
\]

Use exact enumeration.

Keep:

\[
P\le10.
\]

## Allowed files

Create:

```text
PoC-2/src/poc2/oracle.py
PoC-2/src/poc2/scenes.py
PoC-2/scripts/s1_makespace.py
PoC-2/tests/test_makespace.py
```

Modify if needed:

```text
PoC-2/src/poc2/types.py
PoC-2/pyproject.toml
PoC-2/plan2.md
```

Do not create dataset or structure-analysis modules yet.

## Runtime outputs

Maximum:

```text
PoC-2/out/s1/metrics.json
PoC-2/out/s1/cases.png
```

Plot is optional.

## Acceptance

Must prove:

- \(\mathcal C^*\) matches manual expectation in every deterministic case;
- \(\mathcal S^*\) matches manual expectation;
- multiple destination options are handled without ambiguity;
- \(M,V,F\) are distinct;
- placement competition produces a validity interaction rather than a fake geometric \(\beta^G\);
- envelope-changing coupling produces a true geometric pair effect;
- distractors are excluded from the minimal repair;
- cause and repair target can differ;
- all outputs are deterministic.

No QUBO/Hopfield analysis yet.

---

# 18. Stage 2 — Small Randomized Make-Space Structural Dataset

## Purpose

Determine whether the deterministic Stage-1 behavior survives random scene variation.

This is still oracle-only.

No learning.

## Dataset size

Generate exactly:

```text
40 scenes
```

using fixed seed 7.

Target composition:

```text
30 repairable infeasible scenes
10 feasible hard negatives
```

Do not keep generating until desirable interaction statistics appear.

If a generated scene fails validity or catalogue requirements, bounded rejection sampling is allowed,
but record:

- number of rejected attempts,
- rejection reason,
- maximum attempts.

If the generator cannot meet the target composition within a fixed attempt budget, STOP and report.

## Randomization

Randomize within declared ranges:

- 3–6 movable objects;
- object sizes;
- blocker positions;
- distractor positions;
- 2–4 placement regions;
- which placement regions are currently occupied;
- legal relocation destinations;
- target size;
- target insertion lane;
- optional target-repositioning candidate;
- small near-boundary perturbations.

Write all ranges explicitly in code constants.

Do not tune individual seeds manually.

## Candidate count

Target:

\[
5\le P\le10.
\]

Do not exceed 10 in Stage 2.

## Data record

Each JSONL scene record must include enough information to rebuild it deterministically:

```text
scene_id
seed
family
scene parameters
object parameters
placement regions
candidate interventions
candidate groups
original F/G/c
B0
C*
S*
P
|C*|
|S*|
number of tied repairs
number of M-invalid subsets
number of V-invalid subsets
```

Do not store rendered images.

Do not store one file per subset.

## Correctness checks

For a fixed subset of at least 10 scenes:

- compare optimized table evaluation against direct `do(S)` evaluation;
- verify base vs fine envelope sampling does not change \(F\), \(\mathcal C^*\), or \(\mathcal S^*\).

Do not perform pairwise-hypothesis analysis yet.

## Allowed files

Create:

```text
PoC-2/src/poc2/dataset.py
PoC-2/scripts/s2_sample.py
PoC-2/tests/test_dataset.py
```

Modify:

```text
PoC-2/src/poc2/scenes.py
PoC-2/src/poc2/oracle.py
PoC-2/src/poc2/types.py
PoC-2/plan2.md
```

## Runtime outputs

Exactly:

```text
PoC-2/out/s2/scenes.jsonl
PoC-2/out/s2/summary.json
```

Optional:

```text
PoC-2/out/s2/examples.png
```

Maximum three outputs.

## Acceptance

PASS if:

- 40-scene fixed dataset is reproducible;
- original scenes are statically valid;
- required hard negatives are actually feasible;
- required repairable scenes have at least one valid repair;
- exact \(\mathcal C^*\) and \(\mathcal S^*\) are stable under sampling refinement;
- no scene was manually edited after looking at its interaction coefficients.

---

# 19. Stage 3 — Interaction Structure and Pairwise-Necessity Analysis

## Purpose

Answer the central PoC-2 question on the make-space dataset:

\[
\boxed{
\text{Does pairwise structure actually matter for repair decisions?}
}
\]

Do not generate new scenes in this stage.

## Compatible-domain structure analysis

For every repairable Stage-2 scene:

1. compute compatible-domain unary coefficients;
2. compute compatible pairwise coefficients;
3. compute compatible third-order coefficients where feasible;
4. reconstruct \(\hat G_1,\hat G_2,\hat G_3\) on choice-consistent subsets;
5. measure errors only on the choice-consistent domain.

No regression fitting.

Use exact inclusion-exclusion coefficients.

## Resolution-aware significance

Reuse the PoC-1 principle:

a coefficient is significant only if:

- sign is stable at base and fine envelope resolution;
- magnitude is clearly larger than base-vs-fine numerical variation.

Define the exact quantitative rule before viewing aggregate interaction statistics.

## Separate interaction sources

Report separately:

### Physical geometry

\[
\beta^G
\]

from envelope-changing / conditional repair effects.

### Static compatibility

pairwise invalidity edges from \(V\).

### Choice constraints

mutual-exclusion edges from \(M\).

### Substitutability

positive overlap in geometric repair benefit.

Do not collapse these into one “pairwise causality” statistic.

## Required decision metrics

For every scene compute:

- exact \(\mathcal S^*\);
- unary-selected \(\mathcal S^*_1\);
- pairwise-selected \(\mathcal S^*_2\);
- third-order-selected \(\mathcal S^*_3\).

Primary statistic:

\[
\boxed{
R_{\text{pair}}
=
P(
\mathcal S^*_1\neq\mathcal S^*
\land
\mathcal S^*_2=\mathcal S^*
).
}
\]

Also report:

- unary exact-repair rate;
- pairwise exact-repair rate;
- higher-order decision failure rate;
- significant physical-pair incidence;
- compatibility-edge incidence;
- choice-edge incidence;
- average interaction graph density;
- tied-repair rate.

## Important anti-bias rule

Do not call PoC-2 successful merely because nonzero \(\beta\) terms exist.

The important question is whether pairwise information changes the repair decision.

If unary recovers almost every repair, state that clearly.

## Allowed files

Create:

```text
PoC-2/src/poc2/structure.py
PoC-2/scripts/s3_structure.py
PoC-2/tests/test_structure.py
```

Modify:

```text
PoC-2/src/poc2/dataset.py
PoC-2/src/poc2/oracle.py
PoC-2/plan2.md
```

Do not create QUBO optimization code yet.

## Runtime outputs

```text
PoC-2/out/s3/analysis.json
PoC-2/out/s3/pairwise_need.png
PoC-2/out/s3/interaction_sources.png
```

Maximum three outputs.

## Acceptance

PASS is about analysis correctness, not getting a desired pairwise rate.

PASS if:

- compatible-domain coefficients reconstruct known synthetic functions correctly;
- \(M,V,G\) interaction sources remain separate;
- base/fine significance is deterministic;
- unary/pairwise/third-order decisions are computed against exact \(\mathcal S^*\);
- no scenes are removed because they are “too unary.”

Then proceed regardless of whether pairwise incidence is high or low.

---

# 20. Stage 4 — Multi-Task Benchmark-Inspired Extension

## Purpose

Test whether the repair structure is specific to one make-space template or appears across multiple
common manipulation task structures.

Still use primitive MuJoCo geometry.

Do not import RoboCasa/ManiSkill yet.

## Required task families

Keep exactly these four PoC-2 families:

1. make-space shelf insertion;
2. storage insertion;
3. storage extraction;
4. articulated opening.

No fifth family without explicit approval.

## Benchmark-inspired meaning

These should resemble common manipulation scenarios such as:

- cabinet/fridge/microwave loading,
- retrieving an object from storage,
- clearing room before insertion,
- opening a cabinet/fridge/hinged container while nearby objects obstruct the sweep.

The goal is task-structure realism, not asset realism.

## Repair mechanisms

Reuse the same three non-additivity mechanisms:

- envelope-changing target/container repositioning;
- placement/staging compatibility;
- alternative/substitutable repairs.

Do not invent a new interaction mechanism merely because one family is too unary.

## Dataset extension

Add exactly:

```text
20 storage-insertion scenes
20 storage-extraction scenes
20 articulated-opening scenes
```

The Stage-2 40-scene make-space set remains unchanged.

Total PoC-2 structural dataset:

```text
100 scenes
```

No larger dataset in PoC-2 without explicit approval.

## Per-family requirements

Each added family must contain a mixture of:

- infeasible repairable cases;
- feasible hard negatives;
- distractor interventions;
- multiple placement choices where physically sensible;
- at least one deterministic mechanism regression case.

Do not force a fixed number of pairwise scenes.

## Allowed files

Create:

```text
PoC-2/scripts/s4_tasks.py
PoC-2/tests/test_tasks.py
```

Modify:

```text
PoC-2/src/poc2/scenes.py
PoC-2/src/poc2/dataset.py
PoC-2/src/poc2/oracle.py
PoC-2/plan2.md
```

If `scenes.py` exceeds 350 lines, STOP and request approval before creating:

```text
PoC-2/src/poc2/tasks.py
```

No other new module.

## Runtime outputs

```text
PoC-2/out/s4/scenes.jsonl
PoC-2/out/s4/summary.json
PoC-2/out/s4/examples.png
```

Maximum three outputs.

## Acceptance

PASS if:

- all 100 scenes reconstruct deterministically;
- causal labels are stable;
- minimal repairs are stable;
- original scene validity holds;
- no family uses a different definition of \(F,G,V,M,\mathcal C^*,\mathcal S^*\);
- no manual seed deletion occurs based on interaction outcome.

---

# 21. Stage 5 — Cross-Family Structural Recourse Analysis

## Purpose

Perform the full scientific analysis across the fixed 100-scene dataset.

Do not generate new scenes.

## Required distributions

Report distributions of:

\[
P,
\]

\[
|\mathcal C^*|,
\]

\[
|\mathcal S^*|,
\]

tied repair count, validity-binding rate, distractor fraction, and family.

## Required pairwise-necessity analysis

Compute overall and per-family:

\[
R_{\text{pair}}.
\]

Also report:

\[
R_{\text{unary}}
=
P(\mathcal S^*_1=\mathcal S^*)
\]

and:

\[
R_{\text{pair-exact}}
=
P(\mathcal S^*_2=\mathcal S^*).
\]

Report the residual set where pairwise reasoning still fails.

Inspect those cases for stable third-order structure.

## Required mechanism breakdown

Among decision-changing pairwise cases, report counts for:

- physical envelope-changing interaction;
- placement/static compatibility;
- substitutable alternatives;
- choice constraints.

Choice constraints must be reported but must not be used as evidence that the **physical causal**
interaction is pairwise.

## Graph structure

For each scene construct a repair interaction graph.

Report:

- node count \(P\);
- edge count;
- graph density;
- physical-edge density;
- compatibility-edge density;
- maximum degree;
- number of connected components.

Determine whether the measured graphs are:

- mostly independent;
- sparse stars around global macros;
- sparse general graphs;
- dense graphs.

This informs whether QUBO/Hopfield is justified or whether a simpler branch-on-macro strategy is enough.

## Higher-order analysis

Report:

- representation error of unary;
- representation error of pairwise;
- third-order improvement;
- fraction of scenes where higher-order terms alter \(\mathcal S^*\).

Do not declare pairwise sufficient if higher-order structure changes the repair decision.

## Allowed files

Create:

```text
PoC-2/scripts/s5_analyze.py
```

Modify:

```text
PoC-2/src/poc2/structure.py
PoC-2/src/poc2/dataset.py
PoC-2/plan2.md
```

No optimizer module yet.

## Runtime outputs

```text
PoC-2/out/s5/analysis.json
PoC-2/out/s5/decision_recovery.png
PoC-2/out/s5/graph_structure.png
```

Maximum three outputs.

---

# 22. Stage 6 — Optimization Evaluation + Final PoC-2 Verdict

## Purpose

Only now decide whether the measured repair structure warrants QUBO/Hopfield.

No new scene generation.

## Build pairwise optimization only where justified

Create the pairwise energy from:

- measured geometric unary/pairwise effects;
- explicit choice constraints;
- explicit static compatibility penalties;
- repair cost.

Compare against the exact oracle repair set.

## Baselines

Use only three solvers/decision rules:

1. unary-only structured selection;
2. exact pairwise QUBO enumeration;
3. existing PoC-1 Hopfield solver.

Do not add generic optimization libraries.

A very small branch-on-global-macro baseline may be added only if Stage-5 graphs are predominantly
star-shaped around one or two macro variables. If needed, implement it inside `optimize.py`, not as
another module.

## Candidate count

Use measured PoC-2 scenes first.

Do not fabricate large random QUBOs merely to make Hopfield look useful.

If the real dataset has insufficient \(P\) to study solver scaling beyond 10 candidates, report that
honestly.

## Required measurements

For each scene where pairwise optimization is applicable:

- exact oracle repair match;
- QUBO repair match;
- Hopfield repair match for restart budgets \(R=\{1,4,16\}\);
- valid-feasible repair rate;
- minimality rate;
- runtime;
- energy gap;
- result versus \(P\);
- result versus graph density;
- result versus number of physical pair edges.

## Allowed files

Create:

```text
PoC-2/src/poc2/optimize.py
PoC-2/scripts/s6_validate.py
PoC-2/tests/test_optimize.py
```

Modify:

```text
PoC-2/src/poc2/structure.py
PoC-2/plan2.md
```

Do not modify PoC-1 Hopfield/QUBO code.

## Runtime outputs

```text
PoC-2/out/s6/verdict.json
PoC-2/out/s6/solver_accuracy.png
PoC-2/out/s6/overview.png
```

Maximum three outputs.

---

# 23. Final PoC-2 Decision

Do not compress the result into a single GREEN/YELLOW label.

Report four independent conclusions.

## A. Causal diagnosis

### STRONG

if entity-level causal sets are stable and meaningful across task families.

### WEAK

if the diagnostic cause set is unstable, trivial, or poorly aligned with scene semantics.

## B. Corrective recourse

### STRONG

if candidate repair catalogues produce stable, physically valid, nontrivial \(\mathcal S^*\).

### WEAK

if repairs are mostly arbitrary, invalid, or obvious one-to-one blocker removals.

## C. Pairwise optimization hypothesis

### SUPPORTED

if pairwise information changes incorrect unary decisions into the correct exact repair across
multiple task families and more than one physical mechanism, while higher-order decision failures
remain limited.

### NOT NEEDED

if unary reasoning recovers nearly all exact repairs and pairwise structure is mainly explicit
choice constraints.

### INSUFFICIENT

if stable higher-order structure frequently changes the optimal repair.

## D. Hopfield hypothesis

### RETAIN

only if it remains accurate and offers a plausible reason to keep it beyond exact small-problem
enumeration.

### OPTIONAL

if it is accurate but has no practical advantage.

### DROP

if it is unreliable or consistently dominated by simpler exact/structured selection.

---

# 24. Decision to Begin Learning

The project may proceed to the real learning dataset only if:

1. causal labels \(\mathcal C^*\) are stable;
2. corrective labels \(\mathcal S^*\) are stable and physically meaningful;
3. the candidate repair catalogue is rich enough to make correction nontrivial;
4. the structural analysis clearly establishes whether unary, pairwise, or higher-order modeling is
   actually warranted.

The learning architecture must then follow the evidence.

If PoC-2 concludes that unary structure dominates, simplify the learning problem.

If pairwise structure is repeatedly decision-relevant, retain structured optimization.

If higher-order structure is common, do not force the data into a QUBO.

---

# 25. What PoC-2 Must Not Claim

Even if PoC-2 succeeds, it does not prove:

- RGB-D prediction works;
- sim-to-real works;
- candidate corrective macros are low-level motion-plannable;
- no alternative motion path exists;
- QUBO is universally sufficient;
- Hopfield is the best solver;
- benchmark-native asset geometry behaves identically;
- the 100-scene structural dataset is a final training dataset.

It proves only whether the **causal diagnosis + structured corrective-recourse formulation** is
realistic and structured enough to justify the next learning phase.

---

# 26. Mandatory Stage Summary

Every stage must end with:

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

Dataset / scenes:
- ...

Key measured results:
- ...

Causal diagnosis findings:
- ...

Corrective recourse findings:
- ...

Interaction / optimization findings:
- ...

Assumptions validated:
- ...

Assumptions rejected / weakened:
- ...

Problems found:
- ...

Decision gate:
PASS / FAIL / PASS WITH CHANGE

Git:
- commit SHA
- push result

Recommended next stage:
- Stage <N+1>, only after explicit approval.

STOPPED.
Awaiting user approval.
```

Then STOP.

---

# 27. Git Rules

Repository:

```text
https://github.com/Narendhiranv04/causal-infeasibility-ebm
```

Branch:

```text
main
```

Rules:

- verify working tree is clean before every stage;
- verify local `main == origin/main`;
- never force-push;
- never rewrite published history;
- do not commit `out/`;
- commit one validated stage at a time;
- do not push broken stages;
- if remote history is unexpected, STOP.

Suggested commit messages:

```text
poc2 stage0: split frozen baseline and scaffold
poc2 stage1: validate make-space recourse oracle
poc2 stage2: generate randomized make-space structure set
poc2 stage3: analyze repair interaction necessity
poc2 stage4: extend benchmark-inspired task families
poc2 stage5: analyze cross-family recourse structure
poc2 stage6: evaluate structured optimization
```

---

# 28. Coding Style

- Prefer pure functions.
- Use immutable dataclasses for records.
- Type-hint public functions.
- Keep source modules single-purpose.
- Keep scene-generation parameters explicit.
- No hidden global simulator state.
- Every randomized generator must accept an explicit seed.
- Default project seed: 7.
- Every generated scene must be deterministically reconstructable.
- Separate original causal diagnosis from corrective optimization.
- Separate choice validity \(M\), static validity \(V\), and target-action feasibility \(F\).
- Never silently discard a scene based on an undesirable result.
- Never hand-edit one random seed to manufacture a desired \(\beta\).

---

# 29. Testing Contract

Tests should encode scientific invariants.

Required examples:

- PoC-1 regression suite still passes after repository split;
- contradictory same-object repair choices give \(M=0\);
- two placements that overlap give \(V=0\);
- target action can have \(F=0\) while repaired scene has \(V=0\);
- causal diagnostic removal cannot appear in executable \(S^*\);
- exact \(\mathcal C^*\) is deterministic;
- exact \(\mathcal S^*\) is deterministic;
- direct intervention evaluation matches optimized table construction;
- base/fine sampling does not change stable labels;
- compatible-domain pair coefficient matches a planted analytic function;
- choice-constraint edge is not labeled physical;
- compatibility edge is not labeled physical;
- unary and pairwise decision recovery are checked against exact oracle repair;
- QUBO polynomial and matrix energies agree;
- Hopfield is checked only against exact QUBO truth.

Do not test superficial implementation details.

---

# 30. Final Scientific Question

At the end of PoC-2, the project must be able to answer:

\[
\boxed{
\text{When a proposed manipulation action is infeasible, can we identify its causal entities}
\\
\text{and select minimal physically valid corrective actions compositionally?}
}
\]

and, separately:

\[
\boxed{
\text{Does realistic repair choice structure actually require pairwise optimization often enough}
\\
\text{to justify QUBO/Hopfield, or should the final method remain simpler?}
}
\]

The answer is allowed to be:

- causal recourse is strong, optimization unnecessary;
- causal recourse is strong, pairwise QUBO useful;
- causal recourse is strong, higher-order model needed;
- causal recourse itself is not yet well-defined.

The PoC exists to discover which one is true.
