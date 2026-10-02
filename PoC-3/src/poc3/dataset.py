"""PoC-3 Stage 2: fixed privileged-state learning dataset (plan3.md sections 25-30, 49).

PRE-DECLARED BEFORE GENERATION
  Families     fixed order and codes: make_space 0, storage_insertion 1, storage_extraction 2,
               articulated_opening 3 (frozen PoC-2 samplers, builders, oracle, rejection_reason, spec JSON).
  Splits       fixed order train / val / test, split seeds 31 / 41 / 51; per family 420 + 140, 90 + 30,
               90 + 30 repairable + feasible hard negatives (3200 scenes, 800 per family).
  Streams      default_rng([split_seed, family_code, attempt]) with an independent attempt counter per
               (split, family); never PoC-2's generate() / seed-7 stream(); no seed is altered by hand.
  Slots        PoC-2 pattern: slot j is a feasible hard negative iff j % 4 == 3, otherwise repairable.
  Budget       MAX_ATTEMPTS = 10 x requested slots, independently per (split, family); exceeding it STOPS.
  Acceptance   in order, the first failure is the recorded rejection reason:
               catalogue_invalid (sampler / builder / catalogue rejects the draw), then the frozen PoC-2
               rejection_reason (original_static_invalid, P_out_of_range [5, 10], negative_not_feasible,
               repairable_not_infeasible, no_recourse_in_catalogue), then refinement_unstable_B0 and
               refinement_unstable_S_star (base 2 mm vs fine 1 mm). Nothing structural is ever inspected.
  Rows         fixed order split -> family -> slot; row j of labels.npz is line j of scenes.jsonl.
  Storage      scenes.jsonl: reconstructable spec, candidate catalogue and split metadata only (no labels,
               no features). labels.npz: numeric arrays only. State masks over x = 0 .. 2^P - 1, padded
               to 2^10 = 1024 states, bits above 2^P - 1 zero, np.packbits(bits, bitorder="little"):
               state x is bit (x % 8) of byte x // 8. No oracle G / alpha / beta teacher coefficients.
  Digest       sha256 over canonical scenes.jsonl (sort_keys, compact separators) then every NPZ array in
               sorted name order as name | dtype.str | shape followed by its raw C-order bytes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from poc.envelope import ENVELOPE_STEP
from poc2 import dataset as pd
from poc2 import scenes, tasks
from poc2 import structure as st

FAMILIES = ("make_space", "storage_insertion", "storage_extraction", "articulated_opening")
FAMILY_CODE = {f: i for i, f in enumerate(FAMILIES)}
PREFIX = {"make_space": "ms", "storage_insertion": "si", "storage_extraction": "se", "articulated_opening": "ao"}
SPLITS = ("train", "val", "test")
SPLIT_SEED = {"train": 31, "val": 41, "test": 51}
COMPOSITION = {"train": (420, 140), "val": (90, 30), "test": (90, 30)}   # (repairable, negative) per family
ATTEMPT_FACTOR = 10
P_MAX = 10
N_STATES = 2 ** P_MAX
FINE_STEP = ENVELOPE_STEP / 2
INTENT_CODE = {"repairable": 0, "negative": 1}
REASONS = ("catalogue_invalid", "original_static_invalid", "P_out_of_range", "negative_not_feasible",
           "repairable_not_infeasible", "no_recourse_in_catalogue", "refinement_unstable_B0",
           "refinement_unstable_S_star")
ARRAYS = ("P", "cost", "admissible", "optimal", "stable_B0", "stable_S_star", "split", "family", "intent")
FILES = ("scenes.jsonl", "labels.npz", "manifest.json")
STAGE2_DIGEST = "33ebabcd7313822a74456f7eae7a633a59c73d638a9b59cea96dd0db975742f6"  # approved fixed dataset content


def slots(split: str) -> list[str]:
    n_rep, n_neg = COMPOSITION[split]
    out = pd.intents(n_rep, n_neg)
    assert out.count("negative") == n_neg and out.count("repairable") == n_rep
    return out


def max_attempts(split: str) -> int:
    return ATTEMPT_FACTOR * sum(COMPOSITION[split])


def stream(split: str, family: str, attempt: int) -> list[int]:
    return [SPLIT_SEED[split], FAMILY_CODE[family], attempt]


def scene_id(split: str, family: str, slot: int) -> str:
    return f"{split}-{PREFIX[family]}-{slot:03d}"


def build(spec):
    return tasks.build(spec) if isinstance(spec, tasks.TaskSpec) else scenes.build(spec)


def sample(split: str, family: str, slot: int, intent: str, attempt: int):
    sampler = pd.sample_spec if family == "make_space" else tasks.SAMPLERS[family]
    return sampler(np.random.default_rng(stream(split, family, attempt)), scene_id(split, family, slot), intent)


# ------------------------------------------------------------ state-mask bitsets

def pack_states(states, P: int) -> np.ndarray:
    """Set of state indices x < 2^P -> 128 packed bytes (little bit order, padded to 1024 states)."""
    bits = np.zeros(N_STATES, dtype=np.uint8)
    for x in states:
        if not 0 <= int(x) < 2 ** P:
            raise ValueError(f"state {x} outside 0 .. 2^{P} - 1")
        bits[int(x)] = 1
    return np.packbits(bits, bitorder="little")


def unpack_states(packed: np.ndarray, P: int) -> frozenset[int]:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    if len(bits) != N_STATES or bits[2 ** P:].any():
        raise ValueError("packed mask has bits above 2^P - 1")
    return frozenset(int(x) for x in np.flatnonzero(bits))


# ------------------------------------------------------------ labels and acceptance

def oracle(spec, step: float = ENVELOPE_STEP) -> dict:
    """Frozen PoC-2 labels plus the exact tied optimum as state indices.

    pd.label raises ValueError only for an invalid catalogue. structure.exact_decision has no guard for a
    scene without any admissible state (PoC-2 only called it on accepted scenes), so it is skipped there and
    the frozen rejection_reason classifies the draw (original_static_invalid)."""
    lab = pd.label(spec, step)
    t = lab["table"]
    admissible = (t.M == 1) & (t.V == 1)
    lab["S_star_x"] = st.exact_decision(t) if admissible.any() else frozenset()
    lab["admissible_x"] = frozenset(int(x) for x in np.flatnonzero(admissible))
    return lab


def judge(spec, intent: str) -> tuple[str | None, dict | None]:
    """(rejection reason or None, base labels). Fine labels are computed only for base-accepted draws."""
    base = oracle(spec)
    reason = pd.rejection_reason(intent, base)
    if reason is not None:
        return reason, base
    fine = oracle(spec, FINE_STEP)
    if fine["cause"].blockers != base["cause"].blockers:
        return "refinement_unstable_B0", base
    if fine["S_star_x"] != base["S_star_x"]:
        return "refinement_unstable_S_star", base
    if {base["table"].subset(x) for x in base["S_star_x"]} != set(base["repair"].minimal_repairs):
        raise AssertionError("structure.exact_decision disagrees with the oracle repair set")
    return None, base


def generate_stream(split: str, family: str, n_slots: int | None = None) -> tuple[list[dict], dict]:
    """Accepted rows of one (split, family) stream, plus its attempt / rejection statistics."""
    plan = slots(split)[:n_slots]
    budget = max_attempts(split)
    rows, reasons, attempt = [], dict.fromkeys(REASONS, 0), 0
    for j, intent in enumerate(plan):
        while True:
            if attempt >= budget:
                raise RuntimeError(f"{split}/{family}: composition not met within {budget} attempts: {reasons}")
            try:
                spec = sample(split, family, j, intent, attempt)
                reason, lab = judge(spec, intent)
            except ValueError:
                reason, lab = "catalogue_invalid", None
            attempt += 1
            if reason is None:
                rows.append(make_row(split, family, j, intent, attempt - 1, spec, lab))
                break
            reasons[reason] += 1
    return rows, {"split": split, "family": family, "slots": len(plan), "attempts": attempt,
                  "max_attempts": budget, "accepted": len(rows), "rejections": reasons}


def catalogue(options) -> list[dict]:
    return [{"id": o.option_id, "kind": o.intervention.kind.value, "entity": o.intervention.entity_id,
             "region": o.region_id, "params": [float(v) for v in o.intervention.params], "cost": int(o.cost)}
            for o in options]


def make_row(split, family, slot, intent, attempt, spec, lab) -> dict:
    options, P = lab["options"], len(lab["options"])
    record = {"scene_id": scene_id(split, family, slot), "split": split, "family": family, "intent": intent,
              "slot": slot, "stream": stream(split, family, attempt), "spec": pd.spec_to_json(spec),
              "candidates": catalogue(options)}
    cost = np.zeros(P_MAX, dtype=np.int16)
    cost[:P] = [o.cost for o in options]
    arrays = {"P": P, "cost": cost, "admissible": pack_states(lab["admissible_x"], P),
              "optimal": pack_states(lab["S_star_x"], P), "stable_B0": 1, "stable_S_star": 1,
              "split": SPLITS.index(split), "family": FAMILY_CODE[family], "intent": INTENT_CODE[intent]}
    return {"record": record, "arrays": arrays}


ARRAY_DTYPE = {"P": np.uint8, "cost": np.int16, "admissible": np.uint8, "optimal": np.uint8,
               "stable_B0": np.uint8, "stable_S_star": np.uint8, "split": np.uint8, "family": np.uint8,
               "intent": np.uint8}


def assemble(rows: list[dict]) -> tuple[list[dict], dict]:
    records = [r["record"] for r in rows]
    arrays = {k: np.ascontiguousarray(np.stack([np.asarray(r["arrays"][k]) for r in rows]).astype(ARRAY_DTYPE[k]))
              for k in ARRAYS}
    return records, arrays


# ------------------------------------------------------------ storage and digest

def canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def content_digest(records: list[dict], arrays: dict) -> str:
    h = hashlib.sha256()
    h.update("\n".join(canonical(r) for r in records).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"\n{name}|{a.dtype.str}|{a.shape}\n".encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def save(out: Path, records: list[dict], arrays: dict, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenes.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    np.savez_compressed(out / "labels.npz", **arrays)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load(out: Path, expected: str | None = STAGE2_DIGEST) -> tuple[list[dict], dict]:
    records = [json.loads(s) for s in (out / "scenes.jsonl").read_text().splitlines()]
    with np.load(out / "labels.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if expected is not None and content_digest(records, arrays) != expected:
        raise ValueError(f"{out} is not the approved Stage-2 dataset")
    return records, arrays


# ------------------------------------------------------------ verification

def spec_key(record: dict) -> str:
    """Serialized scene content without its id (exact-duplicate detection)."""
    return canonical({k: v for k, v in record["spec"].items() if k != "scene_id"})


def check_row(record: dict, arrays: dict, j: int, recompute: bool = True) -> list[str]:
    """Rebuild row j from its stored spec; with recompute, re-run the base and fine oracle."""
    errors = []
    spec = pd.spec_from_json(record["spec"])
    _, _, options = build(spec)
    P = int(arrays["P"][j])
    if catalogue(options) != record["candidates"] or P != len(options):
        errors.append("catalogue_mismatch")
    if list(arrays["cost"][j][:P]) != [o.cost for o in options] or arrays["cost"][j][P:].any():
        errors.append("cost_mismatch")
    if record["stream"] != stream(record["split"], record["family"], record["stream"][2]):
        errors.append("stream_mismatch")
    meta = (SPLITS.index(record["split"]), FAMILY_CODE[record["family"]], INTENT_CODE[record["intent"]])
    if meta != (int(arrays["split"][j]), int(arrays["family"][j]), int(arrays["intent"][j])):
        errors.append("metadata_mismatch")
    if recompute:
        base, fine = oracle(spec), oracle(spec, FINE_STEP)
        if unpack_states(arrays["admissible"][j], P) != base["admissible_x"]:
            errors.append("admissible_mismatch")
        if unpack_states(arrays["optimal"][j], P) != base["S_star_x"]:
            errors.append("S_star_mismatch")
        if pd.rejection_reason(record["intent"], base) is not None:
            errors.append("acceptance_mismatch")
        stable = (fine["cause"].blockers == base["cause"].blockers, fine["S_star_x"] == base["S_star_x"])
        if stable != (bool(arrays["stable_B0"][j]), bool(arrays["stable_S_star"][j])) or not all(stable):
            errors.append("stability_mismatch")
    return errors


LABEL_KEYS = ("F", "G", "c", "original", "B0", "C_star", "S_star", "status", "alpha", "beta", "repair_size",
              "n_tied_repairs", "seed")


def global_checks(records: list[dict], arrays: dict) -> dict:
    """Dataset-level checks that need no oracle recomputation."""
    n = len(records)
    ids = [r["scene_id"] for r in records]
    streams = [tuple(r["stream"]) for r in records]
    keys = [spec_key(r) for r in records]
    counts = {f"{s}/{f}/{i}": sum(r["split"] == s and r["family"] == f and r["intent"] == i for r in records)
              for s in SPLITS for f in FAMILIES for i in INTENT_CODE}
    expected = {f"{s}/{f}/{i}": COMPOSITION[s][INTENT_CODE[i]] for s in SPLITS for f in FAMILIES for i in INTENT_CODE}
    order = [(SPLITS.index(r["split"]), FAMILY_CODE[r["family"]], r["slot"]) for r in records]
    P = arrays["P"].astype(int)
    opt = [unpack_states(arrays["optimal"][j], P[j]) for j in range(n)]
    adm = [unpack_states(arrays["admissible"][j], P[j]) for j in range(n)]
    neg = arrays["intent"] == INTENT_CODE["negative"]
    return {
        "n_rows_match": all(len(a) == n for a in arrays.values()),
        "ids_unique": len(set(ids)) == n,
        "streams_unique": len(set(streams)) == n,
        "split_namespaces_disjoint": all(r["stream"][0] == SPLIT_SEED[r["split"]] for r in records)
        and len(set(SPLIT_SEED.values())) == len(SPLITS),
        "no_duplicate_specs": len(set(keys)) == n,
        "duplicate_spec_ids": sorted({i for i, k in zip(ids, keys) if keys.count(k) > 1}) if len(set(keys)) < n else [],
        "counts_exact": counts == expected,
        "row_order_fixed": order == sorted(order),
        "P_in_range": bool(((P >= 5) & (P <= P_MAX)).all()),
        "all_stable": bool(arrays["stable_B0"].all() and arrays["stable_S_star"].all()),
        "S_star_admissible": all(o and o <= a for o, a in zip(opt, adm)),
        "negatives_optimum_is_empty_set": all(opt[j] == {0} for j in np.flatnonzero(neg)),
        "repairables_need_repair": all(0 not in opt[j] for j in np.flatnonzero(~neg)),
        "empty_set_admissible": all(0 in a for a in adm),
        "no_labels_in_scenes_jsonl": not any(k in r for r in records for k in LABEL_KEYS),
        "only_numeric_arrays": all(a.dtype.kind in "iub" for a in arrays.values()),
    }
