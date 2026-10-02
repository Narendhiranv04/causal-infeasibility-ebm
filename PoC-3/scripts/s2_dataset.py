"""PoC-3 Stage 2: generate and verify the fixed 3200-scene privileged-state learning dataset.

Run from the repository root:

    PYTHONPATH=PoC-1/src:PoC-2/src:PoC-3/src python PoC-3/scripts/s2_dataset.py

Writes exactly PoC-3/out/s2/{scenes.jsonl, labels.npz, manifest.json}. The 12 (split, family) streams are
independent and are generated in parallel processes; rows are assembled in the fixed split -> family ->
slot order, so parallelism cannot change the content. Every row is then reloaded from disk, rebuilt from
its stored spec, and its labels recomputed with the frozen base and fine oracle. Any failed check, an
exact duplicate spec or an exceeded attempt budget stops the run without a pinned digest.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from poc2 import dataset as pd
from poc3 import RunEnvironment
from poc3 import dataset as ds
from poc3 import features as ft

OUT = Path(__file__).resolve().parents[1] / "out" / "s2"
WORKERS = min(12, os.cpu_count() or 1)


def _stream(job):
    t0 = time.perf_counter()
    rows, stats = ds.generate_stream(*job)
    return rows, stats | {"runtime_s": time.perf_counter() - t0}


def _check(args):
    j, record, row = args
    errors = ds.check_row(record, row, 0)
    try:
        f = ft.extract(*ds.build(pd.spec_from_json(record["spec"])))
        if len(f.candidates) != int(row["P"][0]):
            errors.append("feature_candidate_count_mismatch")
    except Exception as exc:  # report, never hide
        errors.append(f"feature_extraction_failed:{type(exc).__name__}")
    return j, errors


def main() -> None:
    t0 = time.perf_counter()
    jobs = [(s, f) for s in ds.SPLITS for f in ds.FAMILIES]
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(_stream, jobs))
    t_gen = time.perf_counter() - t0
    records, arrays = ds.assemble([row for rows, _ in results for row in rows])
    stats = [s for _, s in results]
    checks = ds.global_checks(records, arrays)
    if not checks["no_duplicate_specs"]:
        sys.exit(f"STOP: exact duplicate scene specs {checks['duplicate_spec_ids']}")
    digest = ds.content_digest(records, arrays)
    manifest = {"stage": 2, "content_digest": digest, "n_scenes": len(records), "conventions": ds.__doc__.strip(),
                "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape)} for k, v in arrays.items()},
                "generation": stats, "generation_runtime_s": t_gen, "workers": WORKERS,
                "environment": RunEnvironment.capture().to_json()}
    ds.save(OUT, records, arrays, manifest)

    t1 = time.perf_counter()
    records2, arrays2 = ds.load(OUT, expected=None)  # independent pass from disk
    reloaded = ds.content_digest(records2, arrays2) == digest
    rows = [(j, records2[j], {k: v[j:j + 1] for k, v in arrays2.items()}) for j in range(len(records2))]
    with ProcessPoolExecutor(WORKERS) as pool:
        row_errors = {j: e for j, e in pool.map(_check, rows, chunksize=8) if e}
    t_ver = time.perf_counter() - t1
    checks = ds.global_checks(records2, arrays2) | {
        "disk_roundtrip_digest_equal": reloaded, "rows_rebuilt_and_relabelled": len(rows),
        "row_errors": {records2[j]["scene_id"]: e for j, e in sorted(row_errors.items())},
        "pinned_digest": ds.STAGE2_DIGEST, "matches_pinned_digest": ds.STAGE2_DIGEST in (None, digest)}
    ok = all(v for k, v in checks.items() if isinstance(v, bool)) and not row_errors
    manifest |= {"checks": checks, "all_checks_pass": ok, "verification_runtime_s": t_ver}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    summary = {"digest": digest, "all_checks_pass": ok, "generation_runtime_s": round(t_gen, 1),
               "verification_runtime_s": round(t_ver, 1),
               "streams": {f"{s['split']}/{s['family']}": [s["accepted"], s["attempts"], s["max_attempts"],
                                                            {k: v for k, v in s["rejections"].items() if v}]
                           for s in stats},
               "failed_checks": [k for k, v in checks.items() if v is False], "n_row_errors": len(row_errors)}
    print(json.dumps(summary, indent=1))
    if not ok:
        sys.exit("STOP: Stage-2 dataset failed verification")


if __name__ == "__main__":
    main()
