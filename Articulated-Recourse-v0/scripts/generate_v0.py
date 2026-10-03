"""Quick-v0 randomized generation (2,800 scenes) + report.

NOT run during the canonical stage: execute only after the canonical scenes are approved.

    python scripts/generate_v0.py                 # full quick-v0 quota from configs/generation.yaml
    python scripts/generate_v0.py --scale 0.02    # tiny smoke run into out/v0_smoke

Writes out/v0/<split>/<scene_id>/{scene.json,inputs.npz,labels.npz},
out/v0/report.json and out/v0/report.md. Stops (non-zero exit) if a variant cannot reach
its quota within the attempt budget, reporting why.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from artrecourse import CONFIGS, OUT  # noqa: E402
from artrecourse.generator import generate  # noqa: E402


def write_report(out: Path, rep: dict):
    recs = rep["records"]
    cat_freq, asset_freq = Counter(), Counter()
    for r in recs:
        for a in r["assets"]:
            asset_freq[a] += 1
            cat_freq[a.split("/")[0]] += 1
    agg = {
        "accepted": {f"{s}/{v}": n for (s, v), n in rep["accepted"].items()},
        "attempted": {f"{s}/{v}": n for (s, v), n in rep["attempted"].items()},
        "rejection_reasons": dict(rep["rejections"]),
        "object_category_frequencies": dict(cat_freq),
        "asset_frequencies": dict(asset_freq),
        "sequence_length_distribution": dict(Counter(r["L"] for r in recs)),
        "meaningful_dependency_edges_per_scene": {v: dict(Counter(r["n_edges"] for r in recs if r["variant"] == v))
                                                  for v in sorted({r["variant"] for r in recs})},
        "generation_runtime_s": rep["runtime_s"],
    }
    (out / "report.json").write_text(json.dumps(agg, indent=1))
    md = ["# Articulated-Recourse-v0 quick-v0 report", ""]
    for k, v in agg.items():
        md += [f"## {k}", "```", json.dumps(v, indent=1), "```", ""]
    (out / "report.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    cfg = yaml.safe_load((CONFIGS / "generation.yaml").read_text())
    core, ctl = cfg["quota"]["core"], cfg["quota"]["controls"]
    quota = {}
    for split in ("train", "val", "test"):
        n = max(1, round(core[split] * a.scale))
        quota[split] = {v: n for v in core["variants"]}
    quota["train"].update({k: max(1, round(v * a.scale)) for k, v in ctl.items()})
    out = OUT / ("v0" if a.scale == 1.0 else "v0_smoke")
    rep = generate(out, cfg["seeds"], quota, cfg["max_attempts_per_accepted"])
    write_report(out, rep)


if __name__ == "__main__":
    main()
