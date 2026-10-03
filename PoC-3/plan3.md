# plan3.md — Learned Intervention Energy for Minimal Corrective Recourse

**Project:** `lfd_infeasibility`  
**Repository:** `https://github.com/Narendhiranv04/causal-infeasibility-ebm`  
**Dependency baseline:** PoC-1 frozen; PoC-2 frozen at `bb6f335`  
**Status:** `CURRENT_STAGE: 4.6C.3 (full-training fit diagnostic; train-only)`  
**Primary purpose:** learn a scene-conditioned energy over executable intervention subsets, then compare exact, Hopfield, and Boltzmann-style inference on the **same learned energy**.

---

# 1. Why PoC-3 exists

PoC-2 established a reliable intervention oracle and a stable corrective-recourse label \(S^*\), but it also showed that the original causal-set label is not the difficult part of the problem.

Under the current oracle,

\[
C^* = B^0
\]

by construction, where \(B^0\) is the set of entities that directly conflict with the prescribed next-action envelope.

That makes \(B^0\) a useful **failure-localization label**, but not a nontrivial causal-discovery target.

The nontrivial structure is in the executable intervention space:

\[
\mathcal I = \{I_1,\ldots,I_P\},
\]

with

\[
S^*
=
\arg\min_{S}
K(S)
\quad
\text{s.t.}
\quad
M(S)=1,\;
V(S)=1,\;
F(S)=0.
\]

PoC-2 also established that, for the current intervention semantics:

- local relocation effects are usually unary;
- changing the shared action envelope can create conditional pairwise effects;
- exact static-validity and choice constraints matter;
- pairwise energy is sufficient for the current oracle;
- Hopfield can solve the quadratic energy but is not needed at \(P\le 10\);
- branch-on-global-macro is exact because the current geometric factorization is macro-centred.

PoC-3 therefore asks a different question:

\[
\boxed{
\text{Can the intervention energy itself be learned directly from the scene and proposed action,}
\\
\text{so that minimal corrective interventions are inferred without oracle intervention enumeration at test time?}
}
\]

This is the point where the direct Hopfield / Boltzmann formulation becomes scientifically meaningful.

---

# 2. Core scientific question

The main PoC-3 question is:

\[
\boxed{
(s,a,\mathcal I)
\longrightarrow
E_\theta(x)
\longrightarrow
\hat S
}
\]

where:

- \(s\) is the current scene state;
- \(a\) is the prescribed next manipulation action;
- \(\mathcal I\) is the given catalogue of executable candidate interventions;
- \(x\in\{0,1\}^P\) is the intervention-selection vector;
- \(E_\theta(x)\) is a learned scene-conditioned quadratic energy;
- \(\hat S\) is the selected corrective intervention set.

The main model is:

\[
\boxed{
E_\theta^{\mathrm{geom}}(x\mid s,a,\mathcal I)
=
\sum_{p=1}^{P} q_{\theta,p} x_p
+
\sum_{p<r} Q_{\theta,pr}x_px_r
}
\]

with:

\[
q_\theta,Q_\theta
=
f_\theta(s,a,\mathcal I).
\]

The binary variables represent **candidate interventions**, not objects.

Therefore the inferred state directly means:

\[
x_p=1
\iff
I_p\text{ should be executed}.
\]

This is the formulation to test.

---

# 3. What is learned and what is not learned

PoC-3 must keep this separation explicit.

## 3.1 Learned

The model learns the scene-conditioned preference over intervention combinations:

\[
E_\theta^{\mathrm{geom}}(x).
\]

This is the part that answers:

> Given the current scene and proposed action, which interventions make the corrective set good or bad, including conditional interactions between interventions?

The parameters learned are only:

- entity encoder parameters;
- action encoder parameters;
- intervention encoder parameters;
- unary-energy head;
- pairwise-energy head;
- optional independent marginal baseline head.

## 3.2 Kept exact

The first PoC-3 experiment must keep these exact:

### Choice consistency

\[
M(x).
\]

Example:

- `object A -> r1`
- `object A -> r2`

cannot both be selected.

### Static repaired-scene validity

\[
V(x).
\]

Example:

two relocated objects cannot occupy overlapping poses.

### Repair cost

\[
K(x)=\sum_p k_px_p.
\]

These are known constraints/costs and are not the main unknown physical intervention-response relation.

The first learning experiment should therefore not force the neural model to relearn facts that can already be computed exactly.

## 3.3 Not learned in PoC-3

Do not learn:

- RGB-D perception;
- segmentation;
- Slot Attention;
- candidate repair proposal;
- motion planning;
- grasp planning;
- object discovery;
- \(M\);
- \(V\);
- repair cost;
- alternative continuous trajectories;
- long-horizon plan repair.

---

# 4. Scope boundary

PoC-3 remains a **next-action corrective-recourse** experiment.

It does not claim:

- global plan repair;
- suffix feasibility;
- whole-task causal discovery;
- no alternative motion path exists;
- real-world visual perception;
- sim-to-real;
- arbitrary action generation.

The question is still:

\[
\boxed{
\text{The next prescribed manipulation is infeasible. Which executable interventions should be applied?}
}
\]

---

# 5. Fixed semantics inherited from PoC-2

PoC-3 imports PoC-2 without modifying it.

The following semantics are frozen.

## 5.1 Action infeasibility

For entity \(i\),

\[
d_i=\min_{\tau\in[0,1]}d_i(\tau),
\]

\[
p_i=\max(0,-d_i),
\]

\[
c_i=\max(0,p_i-\epsilon),
\]

\[
G=\sum_i c_i,
\]

\[
F=1[\exists i:c_i>0].
\]

PoC-3 must not change these labels.

## 5.2 Executable repair labels

For intervention subset \(S\),

\[
M(S)
\]

is choice consistency,

\[
V(S)
\]

is static repaired-scene validity,

\[
F(S)
\]

is post-repair target-action infeasibility,

and:

\[
S^*
=
\arg\min_{\substack{S\\M(S)=1\\V(S)=1\\F(S)=0}}
K(S).
\]

All tied minima are retained.

## 5.3 Diagnostic localization

\[
B^0
\]

is retained in the dataset for analysis only.

Do not treat:

\[
C^*=B^0
\]

as a nontrivial learned causal-set discovery result.

---

# 6. Main PoC-3 hypothesis

The hypothesis is not:

> Hopfield is the best solver.

The hypothesis is:

\[
\boxed{
\text{A compact scene-conditioned quadratic energy can learn the mapping from}
\\
\text{scene/action/candidate interventions to minimal corrective intervention sets.}
}
\]

Then, **holding the learned energy fixed**, compare inference methods:

1. exact enumeration;
2. deterministic Hopfield;
3. fixed-temperature Boltzmann / Gibbs sampling;
4. annealed stochastic inference.

This isolates:

\[
\text{representation/learning error}
\]

from:

\[
\text{inference/solver error}.
\]

---

# 7. Important interpretation of “Boltzmann”

PoC-3 must not conflate two different ideas.

## 7.1 Conditional Boltzmann distribution

For the learned energy:

\[
P_\theta(x\mid s,a,\mathcal I)
=
\frac{\exp[-E_\theta(x)/T]}
{\sum_{x'}\exp[-E_\theta(x')/T]}.
\]

This gives a probability distribution over intervention sets.

It is useful for:

- tied repairs;
- uncertainty;
- marginal intervention probabilities;
- calibration analysis.

## 7.2 Stochastic optimization

Gibbs / Metropolis updates can also be annealed to search for low-energy states.

This is useful for:

- escaping local minima;
- finding the minimum without exhaustive enumeration.

These are related but different tasks.

A fixed-temperature chain is for distributional sampling.

An annealed chain is for optimization.

PoC-3 must evaluate both separately.

## 7.3 What PoC-3 is not doing

PoC-3 is **not** training a separate classical Boltzmann machine using contrastive divergence.

All solvers use the **same learned quadratic energy**.

That is required for a fair Hopfield-vs-Boltzmann comparison.

---

# 8. Main energy decomposition

During training, exact \(M,V\) can be implemented as a hard admissibility mask.

Define:

\[
\mathcal A(s)
=
\{x:M(x)=1,\;V(x)=1\}.
\]

The learned structured energy on admissible states is:

\[
\boxed{
E_\theta(x)
=
E_\theta^{\mathrm{geom}}(x)
+
K(x)
}
\]

for:

\[
x\in\mathcal A(s).
\]

The geometric learned part is:

\[
E_\theta^{\mathrm{geom}}(x)
=
q_\theta^\top x
+
\sum_{p<r}Q_{\theta,pr}x_px_r.
\]

At solver-comparison time, convert the hard constraints to exact quadratic penalties using the PoC-2 encoding.

Then:

\[
\boxed{
H_\theta(x)
=
E_\theta^{\mathrm{geom}}(x)
+
K(x)
+
\lambda_VV_{\mathrm{pen}}(x)
+
\lambda_MM_{\mathrm{pen}}(x).
}
\]

All exact, Hopfield, and Gibbs/annealed solvers in the solver experiment must use this same \(H_\theta\).

---

# 9. Dynamic exact penalty bounds for the learned energy

PoC-2 derived fixed bounds for oracle coefficients.

PoC-3 has scene-dependent learned coefficients, so the penalty must be derived from the actual predicted energy.

For one scene define:

\[
L_\theta
=
\sum_p\min(0,q_p)
+
\sum_{p<r}\min(0,Q_{pr}).
\]

This is a lower bound on the learned geometric energy over all binary states.

Because:

\[
K(x)\ge0,
\]

a lower bound on:

\[
E_\theta^{\mathrm{geom}}(x)+K(x)
\]

is also \(L_\theta\).

The empty set has zero learned intervention energy and zero cost:

\[
E_\theta(0)=0.
\]

The original scenes in the PoC-3 dataset must be statically valid, so \(x=0\) is always \(M,V\)-admissible.

Therefore the best admissible state has energy at most 0.

Use:

\[
\lambda_V
=
-L_\theta+1.
\]

For a choice-consistent but statically invalid state:

\[
V_{\mathrm{pen}}\ge1,
\]

so:

\[
H_\theta
\ge
L_\theta+\lambda_V
=
1
>
0.
\]

For choice-inconsistent states, use the same \(W\) bound from PoC-2 on the magnitude of the quadratic validity polynomial outside the one-pose-per-body domain.

Then:

\[
\lambda_M
=
-L_\theta+\lambda_VW+1.
\]

This guarantees that an \(M\)-violating state cannot beat the best admissible state.

These penalties are:

- scene-dependent;
- derived from the current predicted coefficients;
- not tuned;
- exact for the current constraint formulation.

Stage 5 must test this exhaustively.

---

# 10. Representation philosophy

PoC-3 must start from privileged simulator state.

This is deliberate.

The experiment should answer:

> Can we learn the intervention energy?

before asking:

> Can we infer the necessary state from pixels?

Adding Slot Attention now would combine:

1. object discovery;
2. representation learning;
3. energy learning;
4. structured inference.

That would make failure attribution impossible.

Therefore:

\[
\boxed{
\text{PoC-3 uses explicit entities and action geometry only.}
}
\]

Slot Attention is considered only after PoC-3 succeeds.

---

# 11. Exact feature contract

No feature should be included without a reason.

No oracle label may be used as an input.

All primary models must be permutation-invariant to entity ordering and candidate ordering.

---

# 12. Entity features

Represent every static scene entity as:

\[
e_i.
\]

Include movable objects, structural objects, and the target fixture when present.

Use:

\[
e_i=
[
\tilde p_i,
\tilde h_i,
r_i
].
\]

Where:

### Position

\[
\tilde p_i = p_i/L_0
\]

with:

\[
L_0=0.5\text{ m}.
\]

Three values:

\[
(x,y,z).
\]

Why:

- current relative placement determines whether an intervention is useful;
- structural causes must remain visible to the scene encoder;
- no collision oracle is encoded directly.

### Box half-extents

\[
\tilde h_i=h_i/L_0.
\]

Three values.

Why:

centroids alone are insufficient.

Two objects with the same center but different size can have different collision effects.

### Role one-hot

Use:

- movable;
- structural;
- target/fixture.

Why:

the same geometry has different intervention semantics depending on whether the entity can move.

Do **not** include:

- entity name;
- scene id;
- family id;
- seed;
- \(c_i\);
- blocker label;
- \(B^0\);
- \(C^*\);
- \(F\);
- \(G\);
- \(S^*\).

Those would leak oracle information or dataset identity.

---

# 13. Action representation

The model must be conditioned on the proposed action.

A scene entity is not globally a blocker.

It blocks a specific action.

Therefore the model must approximate:

\[
P(S^*\mid s,a,\mathcal I),
\]

not merely:

\[
P(S^*\mid s,\mathcal I).
\]

Use a common action representation for linear, polyline, and hinge motions.

## 13.1 Fixed normalized action samples

Use:

\[
K_A=8
\]

uniform samples of normalized action progress:

\[
\tau_k
=
\frac{k}{K_A-1}.
\]

For each sample store:

- moving-frame position: 3 values;
- moving-frame rotation in 6D continuous representation: 6 values.

Thus each action sample has:

\[
9
\]

values.

Action path tensor:

\[
A\in\mathbb R^{8\times9}.
\]

Do not use adaptive collision-sampling points as model input.

The feature samples are purely kinematic.

Why:

- one fixed-length descriptor works for linear, polyline, and hinge motion;
- the network sees the actual path geometry;
- it does not receive signed distances or collision labels.

## 13.2 Moving-composite size and offset

Also include the local axis-aligned bounding box (AABB) of the manipulated object + gripper/wrist
composite, from the composite corners expressed in the moving (action) frame:

\[
c_{\mathrm{moving}}^{local}
=
\frac{\min(\text{corners})+\max(\text{corners})}{2},
\qquad
h_{\mathrm{moving}}^{local}
=
\frac{\max(\text{corners})-\min(\text{corners})}{2},
\]

both in \(\mathbb R^3\) and scaled by \(L_0\).

Why:

the path alone does not determine the swept occupied volume.

A large carried object and a small carried object following the same trajectory have different conflicts.

Half-extents alone are insufficient: hinge objects and asymmetric gripper/wrist composites can be
offset from the action frame, so the same half-extents at a different local offset sweep a different
volume. *(Approved Stage-1 amendment.)*

---

# 14. Candidate intervention features

For each candidate intervention \(I_p\), create feature vector \(u_p\).

Do not include the intervention's oracle effect.

Use:

### Intervention kind

One-hot:

- RELOCATE;
- SHIFT_TARGET.

Why:

the physical meaning differs.

### Affected-entity embedding

Use the learned embedding of the affected entity.

For SHIFT_TARGET, use no affected entity: `affected = NO_ENTITY = -1`, hence the zero
affected-entity embedding, in every family and whether or not the scene has a target fixture.
*(Approved Stage-1 correction.)*

Why:

a relocation is entity-specific.

SHIFT_TARGET is a global action-envelope intervention, not a local intervention on the static
fixture. The learned energy models the geometric action-feasibility response; fixture / static
compatibility is handled separately by the exact \(V\). The shift candidate already receives the
scene/action context, the moving-composite context, its kind, the shift vector and the shifted action
reference point.

### Translation delta

\[
\Delta p_p/L_0.
\]

Three values.

Why:

the geometric consequence depends on where the intervention moves something.

### Proposed absolute center

\[
p'_p/L_0.
\]

Three values.

Why:

two identical deltas from different starting poses do not imply identical final geometry.

### Destination-region half-extents

Three values for RELOCATE.

Zeros for SHIFT_TARGET.

Why:

region geometry is part of the allowed corrective action.

### Region center

Three values for RELOCATE.

For SHIFT_TARGET use the shifted action/fixture reference center or zeros, but keep the convention fixed.

Why:

candidate destinations must be geometrically distinguishable.

### No learned cost feature

Do not feed intervention cost into the learned geometric energy.

Cost already enters exactly through:

\[
K(x).
\]

This prevents the neural geometric score from duplicating or hiding the repair-cost term.

---

# 15. Scene encoder

PoC-3 does not need a transformer or GNN initially.

Use a small DeepSets-style encoder.

For each entity:

\[
h_i=\phi_e(e_i).
\]

Pool:

\[
h_{\mathrm{mean}}
=
\frac1N\sum_i h_i,
\]

\[
h_{\mathrm{max}}
=
\max_i h_i.
\]

Encode the action:

\[
h_a=\phi_a(A,h_{\mathrm{moving}}).
\]

Global scene/action embedding:

\[
h_s
=
\phi_s(
[h_{\mathrm{mean}},h_{\mathrm{max}},h_a]
).
\]

Why:

- entity ordering should not matter;
- the scene contains a variable number of entities;
- mean + max captures global context without a heavy architecture.

No attention mechanism in the primary experiment.

---

# 16. Intervention encoder

For each intervention:

\[
h_p
=
\phi_I(
[h_s,
h_{\mathrm{affected}(p)},
u_p]
).
\]

This produces one candidate embedding per intervention.

Candidate order must not matter.

---

# 17. Unary energy head

Predict:

\[
q_p
=
f_u(h_p).
\]

Then the unary energy is:

\[
E_\theta^{(1)}(x)
=
\sum_pq_px_p.
\]

This model is the primary structured **unary ablation**.

It must be trained with the same structured objective as the pairwise model.

---

# 18. Pairwise energy head

For pair \(p,r\), form a symmetric pair representation:

\[
z_{pr}
=
[
h_p+h_r,
|h_p-h_r|,
h_p\odot h_r,
h_s
].
\]

Then:

\[
Q_{pr}
=
f_2(z_{pr}),
\]

with:

\[
Q_{pr}=Q_{rp},
\qquad
Q_{pp}=0.
\]

Why symmetric construction is required:

the pairwise interaction between interventions should not depend on arbitrary candidate ordering.

Do not explicitly provide:

- `same choice group`;
- `V-invalid pair`;
- oracle \(\beta\);
- pair causal label.

Those belong to exact constraints or supervision analysis, not input shortcuts.

---

# 19. Model-size discipline

The first PoC-3 model should stay small.

Target:

- 2-layer MLPs;
- hidden dimension 128;
- ReLU or GELU;
- no transformer;
- no graph neural network;
- no Slot Attention;
- no foundation model.

Total trainable parameter count should remain approximately:

\[
<500\,000.
\]

If the first implementation exceeds this, STOP and justify why.

The point is not architecture scale.

The point is whether intervention energy is learnable at all.

---

# 20. Direct non-energy baseline

A simple baseline must exist.

Use an independent candidate classifier:

\[
\pi_p
=
\sigma(f_{\mathrm{bit}}(h_p)).
\]

Target for intervention \(p\):

\[
y_p
=
\frac{1}{|S^*|}
\sum_{x\in S^*}x_p.
\]

Thus ties produce soft marginals.

Train with binary cross entropy using soft labels.

Primary baseline inference:

\[
\hat x_p=1[\pi_p\ge0.5].
\]

Do **not** repair the output with exact optimization in the primary baseline.

Measure:

- exact repair hit;
- valid-feasible rate;
- minimality;
- M violations;
- V violations.

A secondary projected version may be reported only as a diagnostic.

This baseline answers:

> Is structured energy actually needed, or does independent multi-label prediction already work?

---

# 21. Training target distribution

For each scene define the oracle optimal set:

\[
\mathcal S^*.
\]

All ties are meaningful.

Define the target distribution:

\[
P^*(x)
=
\begin{cases}
1/|\mathcal S^*| & x\in\mathcal S^*\\
0 & \text{otherwise}.
\end{cases}
\]

This avoids arbitrarily choosing one repair from a tied set.

---

# 22. Primary structured training objective

Training is exact over the admissible state space because:

\[
P\le10
\]

in the main dataset.

Define:

\[
\mathcal A
=
\{x:M(x)=1,V(x)=1\}.
\]

For the unary/pairwise learned energy:

\[
E_\theta(x)
=
E_\theta^{\mathrm{geom}}(x)+K(x).
\]

Use training temperature:

\[
T_{\mathrm{train}}=1.
\]

Define:

\[
P_\theta(x)
=
\frac{\exp[-E_\theta(x)]}
{\sum_{y\in\mathcal A}\exp[-E_\theta(y)]},
\qquad x\in\mathcal A.
\]

Use uniform-tie cross entropy:

\[
\boxed{
\mathcal L_{\mathrm{set}}
=
-\frac{1}{|\mathcal S^*|}
\sum_{x\in\mathcal S^*}
\log P_\theta(x).
}
\]

Why this objective:

- directly learns the repair-set distribution;
- all tied minima receive supervision;
- no hand-selected negative subsets are required;
- the partition function is exact for \(P\le10\);
- the same energy later supports exact, Hopfield, and Boltzmann inference.

Do not use contrastive divergence in the primary experiment.

---

# 23. Regularization

Use only simple regularization.

Primary:

\[
\mathcal L
=
\mathcal L_{\mathrm{set}}
+
\lambda_w\|\theta\|_2^2.
\]

Use AdamW weight decay rather than a separate manually tuned penalty when practical.

Do not add:

- auxiliary collision losses;
- coefficient regression;
- blocker prediction;
- entropy penalties;
- contrastive scene losses;

until the primary model has been evaluated.

Those may become Stage-6 ablations only.

---

# 24. Why not regress PoC-2 Möbius coefficients directly as the primary method

Coefficient regression is a valid baseline, but it should not be the main PoC-3 result.

If the primary method predicts oracle \(\alpha,\beta\) directly, the learning problem becomes:

\[
(s,a,\mathcal I)\rightarrow(\alpha,\beta),
\]

then:

\[
(\alpha,\beta)\rightarrow S^*.
\]

That is useful, but it is not the strongest test of the direct learned-energy formulation.

The primary method should instead train the energy from the actual intervention-derived repair distribution:

\[
S^*.
\]

A coefficient-regression model may be included later as an ablation:

> Does explicit oracle-effect supervision make the learned energy easier to train?

---

# 25. Main dataset size

PoC-2's 100 scenes are too small for the primary neural experiment.

Generate a new **privileged-state learning dataset** using the frozen PoC-2 generators and oracle.

Do not modify PoC-2.

Generate:

\[
800\text{ scenes per family}.
\]

Four families:

\[
3200\text{ total scenes}.
\]

Per family:

\[
600\text{ repairable},
\qquad
200\text{ feasible hard negatives}.
\]

Total:

\[
2400\text{ repairable},
\qquad
800\text{ feasible}.
\]

This preserves the 75/25 composition used in PoC-2.

Do not filter scenes by:

- unary success;
- pairwise need;
- graph density;
- number of geometric pair edges;
- Hopfield difficulty;
- future model performance.

---

# 26. Main train / validation / test split

Per family:

### Train

\[
420\text{ repairable}+140\text{ feasible}
=
560.
\]

### Validation

\[
90\text{ repairable}+30\text{ feasible}
=
120.
\]

### Test

\[
90\text{ repairable}+30\text{ feasible}
=
120.
\]

Total:

\[
2240\text{ train},
\]

\[
480\text{ validation},
\]

\[
480\text{ test}.
\]

Use disjoint seed namespaces.

Example fixed seed prefixes:

```text
train: [31, family_code, attempt]
val:   [41, family_code, attempt]
test:  [51, family_code, attempt]
```

Do not derive validation/test seeds from training attempts.

---

# 27. Dataset acceptance criteria

A generated scene is accepted only if:

- original static scene is valid;
- candidate count is within the declared main range;
- requested feasible/repairable intent is satisfied;
- repairable scenes have at least one valid repair;
- base/fine \(S^*\) is stable;
- base/fine \(B^0\) is stable.

Record all rejection reasons.

Do not reject for undesirable interaction structure.

If the requested dataset cannot be generated within the fixed attempt budget, STOP.

---

# 28. Main candidate range

Primary training/evaluation dataset:

\[
5\le P\le10.
\]

This keeps:

- exact training partition functions;
- exact inference truth;
- exact probability distributions;
- exact solver-error attribution.

Do not increase \(P\) before the primary learned model is validated.

---

# 29. Compact dataset storage

Do not store one file per scene.

Do not store rendered images.

Do not store a giant JSON table of all subsets.

Use exactly:

```text
PoC-3/out/s2/scenes.jsonl
PoC-3/out/s2/labels.npz
PoC-3/out/s2/manifest.json
```

`scenes.jsonl` stores reconstructable scene/action/intervention specifications.

`labels.npz` stores compact padded numerical labels.

For each scene store:

- \(P\);
- admissible \(M\land V\) bitset;
- exact \(S^*\) bitset;
- repair costs;
- optional compact oracle \(G_0,\alpha,\beta\) teacher labels for later ablation only;
- base/fine stability flags.

A 1024-state admissibility mask should be stored as a bitset / packed `uint8`, not as JSON booleans.

No database.

No parquet.

No one-subset-per-row dataset.

---

# 30. Training-data leakage rules

The following may appear only in labels/evaluation, never in model input:

- \(F\);
- \(G\);
- \(c_i\);
- \(B^0\);
- \(C^*\);
- \(S^*\);
- oracle \(\alpha\);
- oracle \(\beta\);
- `validity_edge`;
- repair status;
- dataset family name;
- scene id;
- random seed.

Exact \(M,V,K\) are used by the structured objective and solver, but not fed as learned geometric features.

---

# 31. Normalization

All normalization statistics must be computed from the training split only.

Use fixed physical scale:

\[
L_0=0.5\text{ m}
\]

for positions, sizes, and deltas where possible.

Do not normalize using test-set statistics.

For any learned standardization beyond fixed physical scaling, save:

```text
normalization.json
```

inside the Stage-3 training output.

---

# 32. Training protocol

Use PyTorch only.

No PyTorch Geometric.

No Lightning.

No Hydra.

No experiment-management framework.

Recommended initial settings:

```text
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-5
max epochs: 200
early stopping patience: 20 epochs
training seeds: 7, 17, 27
```

Early stopping metric:

\[
\text{validation }\mathcal L_{\mathrm{set}}.
\]

Do not select checkpoints by test accuracy.

Save only the best validation checkpoint per model/seed.

---

# 33. Models that must be trained

Exactly these primary models:

## A. Independent bit classifier

No learned pairwise structure.

## B. Unary structured energy

\[
Q_\theta=0.
\]

Train with exact structured set likelihood.

## C. Pairwise structured energy

\[
q_\theta,Q_\theta.
\]

Train with the same exact structured set likelihood.

Do not add a transformer model unless all three are complete and there is a specific diagnosed representation failure.

---

# 34. Exact inference metrics

For every test scene, exact enumeration of the **learned** energy is the first diagnostic.

This isolates learning quality from solver quality.

Report:

### Optimal-state hit

If one state is selected:

\[
1[\hat x\in S^*].
\]

### Full optimum-set recovery

If all learned-energy minima are enumerated:

\[
\hat{\mathcal S}^*=\mathcal S^*.
\]

### Optimal-set precision / recall

Useful when the learned energy creates extra ties.

### Valid-feasible rate

\[
M(\hat x)=1,\quad V(\hat x)=1,\quad F(\hat x)=0.
\]

### Minimal-cost rate

\[
K(\hat x)=K^*.
\]

### Excess repair cost

\[
K(\hat x)-K^*.
\]

### Test set NLL

Under the exact learned Boltzmann distribution.

### Probability mass on oracle optima

\[
P_\theta(S^*)
=
\sum_{x\in S^*}P_\theta(x).
\]

---

# 35. Tie-aware probability metrics

For the exact learned distribution report:

### Target KL / cross entropy

against the uniform oracle tie distribution.

### Per-intervention marginal target

\[
y_p
=
\frac1{|S^*|}
\sum_{x\in S^*}x_p.
\]

### Predicted marginal

\[
\hat y_p
=
P_\theta(x_p=1).
\]

Report:

- marginal Brier score;
- marginal MAE;
- reliability bins / ECE-style summary;
- average probability mass on \(S^*\).

Do not call these probabilities “causal probabilities.”

They are:

\[
\boxed{\text{model probabilities over corrective intervention sets}.}
\]

---

# 36. Temperature calibration

The learned energy scale is arbitrary.

After training, calibrate a single scalar temperature:

\[
T_{\mathrm{cal}}>0
\]

on the validation split only.

Choose:

\[
T_{\mathrm{cal}}
=
\arg\min_T
\mathcal L_{\mathrm{set,val}}(T).
\]

Do not optimize temperature on test data.

Report both:

- uncalibrated \(T=1\);
- calibrated \(T=T_{\mathrm{cal}}\).

Argmin repair predictions are independent of positive temperature.

Probability metrics are not.

---

# 37. Exact distribution vs Gibbs sampler

For \(P\le10\), exact enumeration gives the true learned Boltzmann distribution:

\[
P_\theta^{\mathrm{exact}}(x).
\]

A fixed-temperature Gibbs sampler should approximate it.

Compare sampled distribution against exact:

- total variation distance when feasible;
- marginal probability MAE;
- probability mass on \(S^*\);
- state-frequency KL/JS on the enumerated support;
- effective sample size / autocorrelation diagnostic if simple to compute.

This evaluates whether the stochastic sampler is actually sampling the claimed Boltzmann distribution.

---

# 38. Annealed stochastic optimization

Separately test annealed Gibbs / Metropolis as an optimizer.

It should return the best state visited.

Do not use its sample frequencies as calibrated probabilities.

Report:

- optimal-state hit;
- learned-energy gap to exact learned-energy minimum;
- oracle \(S^*\) hit;
- valid-feasible rate;
- minimality;
- number of single-site energy evaluations;
- wall-clock runtime.

---

# 39. Hopfield comparison

Use the existing PoC-1 Hopfield implementation unchanged.

Convert the learned total quadratic energy to its Hopfield form.

Report restart budgets:

\[
R\in\{1,4,16\}.
\]

For Hopfield report:

- learned-QUBO optimum hit;
- oracle \(S^*\) hit;
- valid-feasible rate;
- minimality;
- learned-energy gap;
- update count;
- runtime.

Do not compare “full tied optimum set recovery” to a single Hopfield state.

For Hopfield, the primary metric is:

\[
\boxed{\hat x\in\arg\min H_\theta.}
\]

Then separately report:

\[
\hat x\in S^*.
\]

---

# 40. Fair Hopfield-vs-Boltzmann compute comparison

A stochastic method can look better simply by doing more work.

Instrument both methods.

Count:

- single-neuron / single-bit update proposals;
- total energy-difference evaluations;
- restarts/chains.

For each Hopfield budget \(R\), record the actual number of bit updates used.

Run annealed Gibbs with a matched or nearest-lower update budget.

Report:

\[
\text{accuracy vs update budget}
\]

and:

\[
\text{accuracy vs wall-clock}.
\]

Do not claim one method is better from unmatched budgets.

---

# 41. Cross-family generalization

The primary IID split is not enough.

Run leave-one-family-out experiments.

For each held-out family \(f\):

- train on the other 3 families;
- validate only on those 3 families;
- test on the held-out family's fixed test partition.

Four experiments:

1. hold out make-space;
2. hold out storage insertion;
3. hold out storage extraction;
4. hold out articulated opening.

Use the same architecture and training protocol.

Do not tune a special architecture per held-out family.

Report:

- exact learned-energy optimal hit;
- valid-feasible rate;
- minimality;
- NLL;
- pairwise-vs-unary gap.

This measures whether the learned energy captures transferable geometric/intervention structure or memorizes family-specific templates.

---

# 42. Candidate-count scaling stress test

Exact enumeration is trivial at \(P\le10\), so solver comparisons cannot establish scalability there.

Only after the primary pairwise model passes the main learning gate, create a stress test.

Use the same task families and physical scene generators.

Increase only the intervention catalogue richness.

Target:

\[
12\le P\le16.
\]

Generate:

\[
50\text{ scenes per family}
\]

for:

\[
200\text{ total stress scenes}.
\]

Keep them test-only.

Do not train on them in the primary experiment.

Do not fabricate random QUBOs.

Exact enumeration remains feasible up to \(P=16\) for evaluation truth.

Report solver performance versus:

- \(P\);
- learned graph density;
- number of learned pair terms above a fixed magnitude threshold;
- number of candidate relocations;
- number of repositioning macros.

---

# 43. Learned-energy ablations

Run only after the primary models are complete.

Required:

## A. No action input

Remove action-path representation.

Purpose:

test whether the model really needs the proposed manipulation trajectory.

Expected interpretation:

if performance barely changes, the model may be exploiting scene/candidate priors rather than action-conditioned geometry.

## B. Centroid-only entities

Remove object sizes.

Purpose:

directly test the friend's 2D centroid-style simplification.

## C. Unary-only energy

Already part of the primary models.

## D. Pairwise energy

Main model.

## E. Optional coefficient-supervision ablation

Train a copy to regress oracle \(\alpha,\beta\) from PoC-2 teacher labels.

This is not the primary formulation.

Purpose:

determine whether explicit intervention-effect supervision is easier than direct set-level energy learning.

Do not add more ablations unless one of these reveals a specific failure mode.

---

# 44. Slot Attention policy

Do not implement Slot Attention in PoC-3.

The exact future insertion point is:

```text
RGB-D
  ↓
object-centric visual representation
  ↓
entity embeddings
  ↓
the same intervention-energy model
```

Slot Attention would replace the explicit entity-feature source.

It should not change the energy/inference formulation.

Proceed to a visual object representation only if PoC-3 establishes that the privileged-state energy model works.

If privileged-state learning fails, adding Slot Attention is prohibited until the energy-learning failure is understood.

---

# 45. Repository layout

Keep PoC-1 and PoC-2 frozen.

Create:

```text
.
├── PoC-1/                         # frozen
├── PoC-2/                         # frozen at bb6f335
└── PoC-3/
    ├── plan3.md
    ├── pyproject.toml
    ├── src/poc3/
    │   ├── __init__.py
    │   ├── types.py
    │   ├── features.py
    │   ├── dataset.py
    │   ├── model.py
    │   ├── train.py
    │   ├── inference.py
    │   └── metrics.py
    ├── scripts/
    │   ├── s1_features.py
    │   ├── s2_dataset.py
    │   ├── s3_baselines.py
    │   ├── s4_energy.py
    │   ├── s5_solvers.py
    │   ├── s6_ood.py
    │   └── s7_verdict.py
    └── tests/
        ├── test_core.py
        ├── test_features.py
        ├── test_dataset.py
        ├── test_model.py
        ├── test_training.py
        ├── test_inference.py
        └── test_metrics.py
```

No additional source module without approval.

---

# 46. Code-size discipline

Target:

```text
150–300 lines per source file
review threshold: 350 lines
hard maximum: 500 lines
```

If a source file exceeds 350 lines, STOP before creating a split module unless the split is already explicitly approved in this plan.

No:

- notebooks;
- `_old`;
- `_new`;
- `_final`;
- `_v2`;
- scratch scripts;
- backup copies.

---

# 47. Stage 0 — Freeze dependencies and scaffold PoC-3

## Purpose

Create a clean PoC-3 workspace without changing PoC-1 or PoC-2.

## Create

```text
PoC-3/plan3.md
PoC-3/pyproject.toml
PoC-3/src/poc3/__init__.py
PoC-3/src/poc3/types.py
PoC-3/tests/test_core.py
```

## Requirements

Record frozen dependency SHA:

```text
PoC-2: bb6f335
```

Tests must verify:

- PoC-1 tracked files are unchanged from the approved state;
- PoC-2 tracked files are unchanged from `bb6f335`;
- `poc3` imports correctly;
- torch/numpy versions are recorded.

Do not create learning/model files yet.

## Acceptance

PASS if:

- dependency integrity holds;
- combined tests pass;
- workspace clean;
- no model/dataset code exists yet.

STOP.

---

# 48. Stage 1 — Feature contract and one-scene energy plumbing

## Purpose

Prove that the intended privileged representation is well-defined and contains no oracle leakage.

## Create

```text
PoC-3/src/poc3/features.py
PoC-3/src/poc3/model.py
PoC-3/scripts/s1_features.py
PoC-3/tests/test_features.py
PoC-3/tests/test_model.py
```

## Implement only

- entity feature extraction;
- fixed 8-sample action descriptor;
- intervention feature extraction;
- DeepSets scene encoder;
- candidate encoder;
- unary head;
- symmetric pairwise head;
- exact evaluation of predicted energy on all bitmasks for one scene.

No training.

No optimizer comparison.

## Required invariants

### Permutation invariance

Reordering scene entities must not change predicted energies after corresponding index remapping.

Reordering candidate interventions must only permute \(q,Q\).

### Pair symmetry

\[
Q_{pr}=Q_{rp}.
\]

### Zero diagonal

\[
Q_{pp}=0.
\]

### No oracle leakage

Unit test that input tensors do not contain:

- \(F\);
- \(G\);
- \(c\);
- \(B^0\);
- \(S^*\);
- oracle \(\alpha,\beta\).

### Action sensitivity

Two otherwise identical scenes with different prescribed action paths must yield different action tensors.

### Size sensitivity

Two same-centroid objects with different dimensions must yield different entity tensors.

## Output

Maximum:

```text
PoC-3/out/s1/feature_contract.json
PoC-3/out/s1/example_tensors.npz
```

STOP.

---

# 49. Stage 2 — Fixed privileged-state learning dataset

## Purpose

Generate the 3200-scene fixed learning dataset.

## Create

```text
PoC-3/src/poc3/dataset.py
PoC-3/scripts/s2_dataset.py
PoC-3/tests/test_dataset.py
```

## Generate

Exactly:

```text
3200 scenes
800 per family
```

with the split/composition in Sections 25–27.

## Required checks

- deterministic regeneration;
- fixed digest;
- train/val/test seed disjointness;
- family balance;
- requested feasible/repairable composition;
- \(P\in[5,10]\);
- base/fine \(S^*\) stable;
- original scene validity;
- compact bitsets decode correctly;
- exact \(S^*\) reconstruction from stored labels;
- no outcome-based filtering.

## Outputs

Exactly:

```text
PoC-3/out/s2/scenes.jsonl
PoC-3/out/s2/labels.npz
PoC-3/out/s2/manifest.json
```

Pin the dataset digest in code only after generation is accepted.

STOP.

---

# 50. Stage 3 — Independent non-energy baseline

## Purpose

Establish how far a simple candidate-wise predictor can go.

## Create

```text
PoC-3/src/poc3/train.py
PoC-3/src/poc3/metrics.py
PoC-3/scripts/s3_baselines.py
PoC-3/tests/test_training.py
PoC-3/tests/test_metrics.py
```

## Train

Independent sigmoid candidate classifier.

Three seeds:

```text
7, 17, 27
```

## Report

On test split:

- intervention marginal Brier;
- exact repair hit after thresholding;
- valid-feasible rate;
- minimality;
- M violation rate;
- V violation rate;
- average predicted intervention count.

This is a deliberately simple baseline.

Do not project it through an optimizer for the primary result.

## Outputs

Maximum:

```text
PoC-3/out/s3/metrics.json
PoC-3/out/s3/best_seed7.pt
PoC-3/out/s3/best_seed17.pt
PoC-3/out/s3/best_seed27.pt
```

If more artifacts are needed, use subdirectories but keep exactly one best checkpoint per seed.

STOP.

---

# 51. Stage 4 — Unary vs pairwise learned energy

## Purpose

Answer the primary learning question using exact inference only.

Do not involve Hopfield or Gibbs yet.

## Train

### Unary energy

\[
E_\theta^{(1)}(x)=q_\theta^\top x+K(x).
\]

### Pairwise energy

\[
E_\theta^{(2)}(x)
=
q_\theta^\top x
+
\sum_{p<r}Q_{\theta,pr}x_px_r
+
K(x).
\]

Both use:

\[
\mathcal L_{\mathrm{set}}.
\]

Three seeds each.

## Exact test inference

Enumerate the learned energy over the exact admissible domain.

## Required metrics

Overall and per family:

- oracle-optimal-state hit;
- full optimal-set recovery;
- optimal-set precision;
- optimal-set recall;
- valid-feasible;
- minimality;
- excess cost;
- test NLL;
- probability mass on \(S^*\);
- marginal Brier;
- calibrated temperature;
- parameter count.

## Pairwise necessity in learning

Report paired difference:

\[
\Delta_{\mathrm{pair}}
=
\mathrm{Acc}_{\mathrm{pair}}
-
\mathrm{Acc}_{\mathrm{unary}}.
\]

Use paired bootstrap over test scenes.

Do not call pairwise learning useful merely because predicted \(Q\neq0\).

It must change the repair decision.

## Stage-4 learning gate

The learned-energy formulation is considered established enough for solver comparison only if the pairwise model, under **exact learned-energy inference**:

1. achieves at least 90% oracle optimal-state hit on the IID test set averaged across training seeds;
2. achieves at least 99% valid-feasible rate;
3. improves over the independent classifier;
4. does not collapse to selecting the same repair size/pattern across scenes.

If this gate fails:

- solver experiments may be run only diagnostically;
- do not interpret Hopfield/Boltzmann results as evidence for or against those inference methods;
- first diagnose energy learning.

STOP.

---

# 52. Stage 5 — Same learned energy: exact vs Hopfield vs Boltzmann

## Purpose

Test the friend's central solver claim fairly.

Use the **same fixed pairwise learned energy**.

No retraining per solver.

## Create

```text
PoC-3/src/poc3/inference.py
PoC-3/scripts/s5_solvers.py
PoC-3/tests/test_inference.py
```

## Exact reference

For each test scene:

\[
x_{\mathrm{learned}}^*
=
\arg\min_xH_\theta(x)
\]

by enumeration.

This separates:

- learned-energy error;
- solver error.

## Hopfield

Use PoC-1 unchanged.

Budgets:

\[
R=1,4,16.
\]

## Fixed-temperature Gibbs

Use calibrated:

\[
T_{\mathrm{cal}}.
\]

Compare its sampled distribution against the exact learned Boltzmann distribution.

## Annealed Gibbs / Metropolis

Use a fixed predeclared schedule.

The schedule must be fixed before aggregate solver results are viewed.

Suggested initial schedule:

```text
T_start = 2.0 * T_cal
T_end   = 0.05 * T_cal
geometric cooling
```

The number of bit updates is matched to the measured Hopfield compute budgets.

## Required solver metrics

For every solver:

- learned-energy optimum hit;
- oracle \(S^*\) hit;
- valid-feasible;
- minimality;
- energy gap;
- update count;
- runtime.

For fixed-temperature Gibbs:

- marginal MAE vs exact learned Boltzmann;
- TV / JS when practical;
- \(P(S^*)\) error.

## Important interpretation

If:

\[
\text{exact learned energy is wrong}
\]

but Gibbs is also wrong, that does not mean Gibbs failed.

If:

\[
\text{exact learned energy is correct}
\]

and Hopfield misses but annealed Gibbs succeeds, that is evidence for a solver difference.

STOP.

---

# 53. Stage 6 — Generalization, scaling, and required ablations

## Purpose

Test whether the model learned transferable intervention structure or memorized the fixed templates.

## Part A — Leave-one-family-out

Run all four held-out-family experiments.

Primary model:

pairwise energy.

Secondary:

unary energy.

Do not retune architecture per held-out family.

## Part B — Candidate-count stress

Only if Stage-4 learning gate passed.

Generate the fixed 200-scene stress set:

```text
50 per family
P in [12,16]
test only
```

No interaction-outcome filtering.

Exact enumeration is still the evaluation truth.

Evaluate:

- exact learned energy;
- Hopfield;
- annealed stochastic inference.

## Part C — Required representation ablations

Run:

1. no action input;
2. centroid-only entities;
3. unary energy;
4. pairwise energy.

Optional fifth:

5. oracle coefficient-regression teacher.

## Required conclusions

Answer:

- Does action conditioning materially matter?
- Are centroids sufficient?
- Does pairwise learned energy materially matter?
- Does pairwise gain persist outside the IID family mixture?
- Does stochastic inference help at larger \(P\)?
- Does Hopfield remain competitive when enumeration becomes more expensive?

STOP.

---

# 54. Stage 7 — Final PoC-3 verdict

## Purpose

Freeze the privileged-state learning conclusion before any visual representation is added.

## Create

```text
PoC-3/scripts/s7_verdict.py
```

No new models.

No new data.

No new hyperparameter tuning.

---

# 55. Final PoC-3 verdicts

Report conclusions separately.

Do not compress into one GREEN label.

## A. Direct intervention-energy learning

### ESTABLISHED

if exact inference on the learned pairwise energy achieves the Stage-4 gate and clearly outperforms the independent classifier.

### PARTIAL

if the model is useful but below the gate or unstable across seeds/families.

### NOT ESTABLISHED

if exact learned-energy inference itself is poor.

---

# 56. B. Learned pairwise structure

### NEEDED

if pairwise energy improves oracle repair decisions over unary by at least 5 percentage points overall **and** the paired 95% bootstrap confidence interval for the improvement excludes zero, with improvement in at least two families.

### OPTIONAL

if pairwise improves coefficient/NLL fit but not repair decisions materially.

### NOT NEEDED

if unary matches pairwise on repair decisions.

This verdict is about the **learned model**, not the oracle factorization.

---

# 57. C. Hopfield inference

### RETAIN

only if, on the same learned energy:

- it reaches within 1 percentage point of exact learned-energy optimum-hit by \(R\le16\);
- and on the \(P\in[12,16]\) stress set it offers a practical update/runtime advantage over exact enumeration or annealed inference.

### OPTIONAL

if accurate but without practical advantage.

### DROP

if it remains substantially below exact learned-energy optimum recovery at \(R=16\) or is consistently dominated.

---

# 58. D. Boltzmann / stochastic inference

### RETAIN

if at matched update budget it improves learned-energy optimum-hit over Hopfield by at least 3 percentage points with a paired 95% bootstrap confidence interval excluding zero, **or** if fixed-temperature Gibbs uniquely provides a well-matched useful distribution over tied repairs.

### OPTIONAL

if it matches Hopfield but mainly adds uncertainty estimates.

### DROP

if it mixes poorly, is less accurate, and its probabilities do not match the exact learned distribution.

This verdict must distinguish:

- fixed-temperature probability sampling;
- annealed optimization.

---

# 59. E. Readiness for visual object representations

### READY

only if direct intervention-energy learning is ESTABLISHED from privileged state.

### NOT READY

if privileged-state energy learning is not established.

If READY, the next project may replace explicit entity features with:

- detector/segmenter features;
- object tokens;
- Slot Attention;
- other object-centric visual representations.

PoC-3 itself does not choose among them.

---

# 60. Statistical reporting

For every main accuracy metric report:

- mean across 3 training seeds;
- standard deviation across seeds;
- exact test-scene count;
- per-family values.

For pairwise-vs-unary and Hopfield-vs-stochastic comparisons use paired bootstrap over scenes.

Use:

```text
10,000 bootstrap resamples
95% confidence interval
fixed bootstrap seed 107
```

Do not use p-values as the sole basis for a methodological conclusion.

---

# 61. Anti-bias rules

Do not:

- remove a test scene because the learned model fails;
- alter dataset seeds after looking at model performance;
- change the pairwise threshold after viewing pairwise gains;
- change Boltzmann schedule after viewing test results;
- change solver budgets after viewing which one wins;
- remove tied repairs because they hurt full-set accuracy;
- add Slot Attention to rescue a weak privileged model;
- claim stochastic sampling “eliminates local minima”;
- call Boltzmann probabilities causal probabilities;
- claim pairwise structure is universal physics.

---

# 62. Required sanity checks before any result is trusted

The following must pass.

## Feature sanity

- entity permutation invariance;
- candidate permutation equivariance;
- pair symmetry;
- no oracle label leakage;
- action path changes action tensor;
- object size changes entity tensor.

## Energy sanity

- exact state energies equal direct polynomial evaluation;
- candidate reordering preserves state energies after bit remapping;
- pairwise model reduces exactly to unary if \(Q=0\).

## Constraint sanity

- PoC-2 quadratic \(M,V\) penalties reproduced exactly;
- dynamic learned-energy penalties exclude all invalid states from the global minimum;
- exact hard-mask and penalty argmins agree on every validation scene.

## Training sanity

- train loss decreases on a 20-scene overfit test;
- the pairwise model can fit a planted synthetic quadratic energy;
- tied-label NLL gives equal optimum states equal target status.

## Sampler sanity

- fixed-temperature Gibbs recovers an exact known 3–5 bit Boltzmann distribution;
- annealed sampler can escape a planted local minimum;
- Hopfield reproduces PoC-1 reference behavior.

---

# 63. Overfit test before full training

Before training the full dataset, use exactly 20 training scenes.

Requirement:

the pairwise model should reach near-perfect oracle optimal-state hit on those 20 scenes under exact inference.

If it cannot overfit 20 scenes:

STOP.

Do not start hyperparameter sweeps.

The implementation or representation is wrong.

---

# 64. Hyperparameter policy

Do not perform broad hyperparameter search.

Allowed initial changes if validation fails:

- learning rate from:
  \[
  \{3\times10^{-4},10^{-3},3\times10^{-3}\};
  \]
- hidden width from:
  \[
  \{64,128,256\};
  \]
- weight decay from:
  \[
  \{0,10^{-5},10^{-4}\}.
  \]

Maximum:

\[
3
\]

small controlled adjustments after the default run.

All changes must be selected on validation only.

Do not tune on test.

---

# 65. Cross-family fairness

Do not include explicit one-hot family ID in the primary model.

The model may see:

- action geometry;
- entity geometry;
- intervention geometry.

This forces cross-family transfer to come from physical structure rather than a family label.

A family-ID ablation may be added only after the primary LOFO results are recorded.

---

# 66. Interpretation of a successful pairwise learned energy

A successful result would support:

\[
\boxed{
\text{A learned quadratic intervention energy is sufficient for direct corrective-set inference}
\\
\text{within the current privileged-state task and intervention formulation.}
}
\]

It would not prove:

- real scenes are universally pairwise;
- higher-order interaction never matters;
- the same energy works under unknown objects;
- visual object discovery is solved.

---

# 67. Interpretation of a successful Boltzmann result

If fixed-temperature Gibbs approximates the exact learned Boltzmann distribution, the valid claim is:

> stochastic sampling can represent and recover uncertainty over multiple corrective intervention sets under the learned energy.

If annealed stochastic inference beats Hopfield at matched update budget, the valid claim is:

> stochastic exploration reduced local-attractor failures on this learned energy family.

Do not claim:

> Boltzmann has no local minima problem.

---

# 68. Interpretation of a failed Boltzmann result

If Gibbs mixes poorly:

that does not invalidate the learned energy.

It means:

\[
\boxed{\text{the chosen stochastic inference scheme is inefficient on this energy landscape}.}
\]

The exact learned-energy result remains the representation benchmark.

---

# 69. Interpretation of a failed Hopfield result

If exact learned-energy inference is correct but Hopfield fails:

\[
\boxed{\text{the solver is the bottleneck}.}
\]

Do not retrain the energy specifically to make Hopfield easier unless that becomes a separate research question.

---

# 70. Interpretation of a failed learned energy

If exact learned-energy inference fails:

\[
\boxed{\text{do not blame Hopfield or Gibbs}.}
\]

The issue is:

- representation;
- model capacity;
- supervision;
- or dataset distribution.

Fix that first.

---

# 71. Causal language policy for PoC-3

The strongest justified language is:

\[
\boxed{\text{intervention-grounded corrective recourse}.}
\]

Training labels come from controlled simulator interventions.

Therefore the model is learning from interventional outcomes.

But:

\[
B^0
\]

remains direct geometric localization.

Do not claim latent causal discovery unless a future model introduces a genuinely nontrivial causal variable structure.

---

# 72. Relation to the friend's 2D formulation

PoC-3 intentionally tests the strongest defensible version of that idea.

The 2D intuition is:

```text
object/action features
        ↓
       MLP
        ↓
    energy weights
        ↓
Hopfield / Boltzmann
        ↓
minimal intervention set
```

PoC-3 generalizes it to:

```text
3D privileged scene entities
+
prescribed 3D action trajectory
+
candidate executable interventions
        ↓
permutation-invariant scene/intervention encoder
        ↓
scene-conditioned q and Q
        ↓
same learned energy
        ├── exact enumeration
        ├── Hopfield
        └── Boltzmann / annealed inference
        ↓
corrective intervention set
```

This is the direct comparison to run.

---

# 73. Why Slot Attention is not part of that comparison yet

Centroid/box features and Slot Attention answer different questions.

The current experiment asks:

> Given object-level state, can the energy be learned?

Slot Attention asks:

> Can object-like state be discovered from visual input?

Mixing them would make a negative result ambiguous.

Therefore the correct progression is:

\[
\boxed{
\text{privileged objects}
\rightarrow
\text{learned intervention energy}
\rightarrow
\text{solver comparison}
\rightarrow
\text{visual object representation}
}
\]

not:

\[
\text{pixels}
\rightarrow
\text{Slot Attention}
\rightarrow
\text{unknown energy model}
\rightarrow
\text{unknown stochastic solver}.
\]

---

# 74. Runtime / compute policy

Primary PoC-3 should run on one normal GPU or CPU where practical.

No distributed training.

No multi-GPU.

No large pretrained model.

Record:

- training wall-clock;
- peak CPU RAM;
- peak GPU memory if GPU used;
- inference runtime separately from feature extraction.

Do not optimize runtime before correctness.

---

# 75. Reproducibility

Global random seeds:

```text
dataset: 31/41/51 namespaces
training: 7, 17, 27
bootstrap: 107
```

Record:

- Python version;
- numpy;
- torch;
- mujoco;
- git SHA;
- device;
- deterministic flags.

Every result JSON must include these.

---

# 76. Output hygiene

Do not commit:

```text
PoC-3/out/
```

Do not commit model checkpoints.

Do not create:

- TensorBoard logs unless explicitly approved;
- WandB projects;
- CSV spam;
- per-epoch checkpoint directories.

For each training run retain:

- best checkpoint;
- final metrics JSON;
- minimal training curve arrays if needed.

---

# 77. Git rules

Branch:

```text
main
```

Before every stage:

- working tree clean;
- local main == origin/main;
- PoC-1 unchanged;
- PoC-2 unchanged from `bb6f335`.

Never:

- force push;
- rewrite published history;
- amend a published PoC-1 or PoC-2 commit.

One validated stage per commit.

Suggested messages:

```text
poc3 stage0: scaffold learned intervention energy
poc3 stage1: define privileged feature contract
poc3 stage2: generate fixed learning dataset
poc3 stage3: train independent repair baseline
poc3 stage4: learn unary and pairwise intervention energies
poc3 stage5: compare exact hopfield and boltzmann inference
poc3 stage6: evaluate generalization scaling and ablations
poc3 stage7: finalize learned-energy verdict
```

---

# 78. Mandatory stage summary

Every stage must end with:

```text
STAGE <N> COMPLETE

Goal:
- ...

Files created:
- ...

Files modified:
- ...

Commands run:
- ...

Tests:
- <passed>/<total>

Dataset:
- ...

Model / training:
- ...

Exact-inference results:
- ...

Hopfield results:
- ...

Boltzmann / stochastic results:
- ...

Generalization results:
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

For stages before solver evaluation, unused sections may say `N/A`.

Then STOP.

---

# 79. Stage-level file permissions

The agent must not modify arbitrary files.

## Stage 0

Create only:

```text
PoC-3/plan3.md
PoC-3/pyproject.toml
PoC-3/src/poc3/__init__.py
PoC-3/src/poc3/types.py
PoC-3/tests/test_core.py
```

## Stage 1

Create/modify only:

```text
PoC-3/src/poc3/features.py
PoC-3/src/poc3/model.py
PoC-3/scripts/s1_features.py
PoC-3/tests/test_features.py
PoC-3/tests/test_model.py
PoC-3/plan3.md
```

## Stage 2

Create/modify only:

```text
PoC-3/src/poc3/dataset.py
PoC-3/scripts/s2_dataset.py
PoC-3/tests/test_dataset.py
PoC-3/plan3.md
```

## Stage 3

Create/modify only:

```text
PoC-3/src/poc3/train.py
PoC-3/src/poc3/metrics.py
PoC-3/scripts/s3_baselines.py
PoC-3/tests/test_training.py
PoC-3/tests/test_metrics.py
PoC-3/plan3.md
```

## Stage 4

Modify only existing PoC-3 learning files plus:

```text
PoC-3/scripts/s4_energy.py
PoC-3/plan3.md
```

No new source module.

## Stage 5

Create/modify only:

```text
PoC-3/src/poc3/inference.py
PoC-3/scripts/s5_solvers.py
PoC-3/tests/test_inference.py
PoC-3/plan3.md
```

## Stage 6

Create/modify only:

```text
PoC-3/scripts/s6_ood.py
PoC-3/src/poc3/dataset.py
PoC-3/src/poc3/metrics.py
PoC-3/plan3.md
```

No new model type.

## Stage 7

Create/modify only:

```text
PoC-3/scripts/s7_verdict.py
PoC-3/plan3.md
```

---

# 80. Stop conditions

STOP immediately if any of the following occurs.

### Dependency drift

PoC-1 or PoC-2 differs from the frozen approved state.

### Dataset leakage

Any oracle label is discovered in model input.

### Unstable labels

Base/fine \(S^*\) changes frequently.

### Overfit failure

The model cannot overfit the 20-scene sanity set.

### Penalty failure

The learned-energy penalty encoding permits an \(M/V\)-invalid state to beat the admissible optimum.

### Solver comparison invalid

Exact learned-energy inference is weak enough that Hopfield/Boltzmann results cannot isolate solver quality.

### Test-set tuning

A hyperparameter, temperature, schedule, or solver budget is changed because of test-set results.

Report and wait for approval.

---

# 81. What counts as success

PoC-3 is successful if it can make a clean statement of the form:

\[
\boxed{
\text{From privileged scene/action state, a learned intervention energy predicts minimal corrective sets}
\\
\text{without running the intervention oracle at inference time.}
}
\]

The strongest version is:

- pairwise learned energy materially improves repair decisions over independent/unary prediction;
- exact learned-energy inference is strong;
- cross-family transfer is nontrivial;
- stochastic inference can be fairly compared against Hopfield;
- probability outputs over tied repairs are meaningful and calibrated.

---

# 82. What PoC-3 must not claim

Even if successful, PoC-3 does not prove:

- visual perception works;
- Slot Attention works;
- real-world transfer works;
- candidate interventions can be generated automatically;
- low-level motion plans always exist for selected repairs;
- quadratic energy is universally sufficient;
- Boltzmann sampling globally solves all local minima;
- Hopfield is obsolete;
- model probabilities are causal probabilities;
- long-horizon TAMP repair is solved.

---

# 83. Final scientific question

At the end of PoC-3, answer:

\[
\boxed{
\text{Can the intervention structure discovered by the oracle be replaced at inference time}
\\
\text{by a learned scene-conditioned energy over executable repairs?}
}
\]

Then separately answer:

\[
\boxed{
\text{Given the same learned energy, does deterministic Hopfield inference or stochastic}
\\
\text{Boltzmann-style inference provide the better repair-selection mechanism?}
}
\]

And finally:

\[
\boxed{
\text{Is the privileged-state formulation strong enough to justify adding visual object representations next?}
}
\]

Those are three separate conclusions.

---

# 84. One-sentence north star

\[
\boxed{
\textbf{Learn an intervention-grounded energy that maps a scene and proposed manipulation directly to}
\\
\textbf{the minimal executable corrective intervention set, then test how that energy should be inferred.}
}
\]
