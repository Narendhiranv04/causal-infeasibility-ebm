"""Validate every screened asset, render previews and write assets/asset_manifest.json.

Per object: load in MuJoCo, finite positive mass/inertia, category dimension ranges,
visual-vs-collision extent agreement, drop/settle test on a plane, preview render.
Fixture: joints exist, door genuinely rotates, upper rack genuinely translates, the
fully pulled upper rack does not intersect the fully opened door.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from artrecourse import CACHE, OUT, ROOT  # noqa: E402
from artrecourse.assets import (MANIFEST, analyze_object, dims_reasonable, drop_test,  # noqa: E402
                                load_asset_config, load_object_spec, screening_pool)
from artrecourse.fixture import fixture_report  # noqa: E402
from artrecourse.render import save_jpg  # noqa: E402

MAX_PER_CATEGORY = 4


def preview(asset, size=220) -> np.ndarray:
    child = load_object_spec(asset)
    w = mujoco.MjSpec()
    w.visual.headlight.ambient = [0.45, 0.45, 0.45]
    w.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    w.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], rgba=[0.92, 0.92, 0.9, 1])
    b = w.worldbody.add_body(pos=[0, 0, 0])
    b.add_frame().attach_body(child.body(asset.root_body), "o_", "")
    m = w.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo = np.min([d.geom_xpos[g] - m.geom_rbound[g] for g in range(1, m.ngeom)], 0)
    hi = np.max([d.geom_xpos[g] + m.geom_rbound[g] for g in range(1, m.ngeom)], 0)
    d.qpos[:] = 0
    m.body_pos[1] = [0, 0, -lo[2] + 0.0]
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (hi[2] - lo[2]) / 2]
    cam.distance = 2.4 * float(np.max(hi - lo))
    cam.azimuth, cam.elevation = 135, -30
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    with mujoco.Renderer(m, size, size) as r:
        r.update_scene(d, cam, opt)
        return r.render()


def label(img, text):
    import cv2

    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def git_head(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    cfg = load_asset_config()
    src = cfg["sources"]
    outdir = OUT / "asset_previews"
    outdir.mkdir(parents=True, exist_ok=True)
    records, sheets = [], {}
    per_cat: dict[str, int] = {}
    for a in screening_pool():
        rec = {
            "asset_id": a.asset_id, "category": a.category, "robocasa_category": a.robocasa_category,
            "source_key": a.source, "archive": a.archive, "upstream_id": a.upstream_id, "scale": a.scale,
        }
        if a.source == "robocasa":
            rec.update(source_repository=src["robocasa"]["repo"], upstream_version=src["robocasa"]["commit"],
                       license=src["robocasa"]["license"],
                       upstream_asset=f"{a.archive}:{'objaverse' if a.archive == 'objaverse' else 'lightwheel'}/"
                                      f"{a.robocasa_category}/{a.upstream_id}")
        else:
            rec.update(source_repository=src["scanned_objects"]["repo"], upstream_version=src["scanned_objects"]["commit"],
                       license=src["scanned_objects"]["license"], upstream_asset=f"models/{a.upstream_id}")
        rec["local_cache_path"] = str(a.model_xml.parent.relative_to(ROOT))
        reasons = []
        try:
            info = analyze_object(a)
            drop = drop_test(a)
            spec = load_object_spec(a)
            rec["visual_meshes"] = sorted({Path(me.file).name for me in spec.meshes if "_coll" not in me.name
                                           and "collision" not in me.name})
            rec["collision_meshes"] = sorted({Path(me.file).name for me in spec.meshes if "_coll" in me.name
                                              or "collision" in me.name})
            rec.update(info)
            rec["drop_test"] = drop
            if not (np.isfinite(info["mass"]) and info["mass"] > 0):
                reasons.append("non-positive mass")
            if min(info["inertia_diag"]) <= 0:
                reasons.append("non-positive inertia")
            if not dims_reasonable(a.category, info):
                reasons.append("dimensions outside category range")
            if any(abs(r - 1) > 0.08 for r in info["visual_collision_extent_ratio"]):
                reasons.append("visual/collision extent mismatch")
            if not drop["passed"]:
                reasons.append("drop/settle test failed")
            img = preview(a)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"load error: {e}")
            img = np.full((220, 220, 3), 200, np.uint8)
        n = per_cat.get(a.category, 0)
        accepted = not reasons and n < MAX_PER_CATEGORY
        if not reasons and not accepted:
            reasons.append("category quota reached (kept in pool, not selected)")
        per_cat[a.category] = n + int(accepted)
        rec["accepted"], rec["rejection_reasons"] = accepted, reasons
        records.append(rec)
        tag = "OK" if accepted else "REJ"
        sheets.setdefault(a.category, []).append(label(img, f"{tag} {a.upstream_id[:26]}"))
        print(f"{tag:3s} {a.asset_id:40s} {'; '.join(reasons)}")

    for cat, imgs in sheets.items():
        save_jpg(outdir / f"{cat}.jpg", np.concatenate(imgs, 1))
    allrows = [np.concatenate(v + [np.full_like(v[0], 255)] * (8 - len(v)), 1) for v in sheets.values()]
    save_jpg(outdir / "all_objects.jpg", np.concatenate(allrows, 0), quality=70)

    fx = fixture_report(render_dir=outdir)
    men = CACHE / "mujoco_menagerie" / src["menagerie"]["subdir"]
    manifest = {
        "schema": "artrecourse.asset_manifest.v0",
        "fixture": fx,
        "robot": {
            "asset_id": "franka_emika_panda", "category": "robot", "source_repository": src["menagerie"]["repo"],
            "upstream_asset": src["menagerie"]["subdir"], "upstream_version": git_head(men.parent) or src["menagerie"]["commit"],
            "license": src["menagerie"]["license"], "local_cache_path": str(men.relative_to(ROOT)),
            "visual_mesh": "assets/*.obj", "collision_meshes": ["assets/link*.stl", "assets/hand.stl", "assets/finger_0.obj"],
            "use": "visualisation + IK/arm-collision admissibility filter (never a recourse label)",
        },
        "textures": {k: {"source_repository": src["robocasa"]["repo"], "upstream_asset": f"textures:{v}",
                         "license": src["robocasa"]["license"],
                         "local_cache_path": str((CACHE / "robocasa" / "textures" / v).relative_to(ROOT))}
                     for k, v in cfg["textures"].items()},
        "objects": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    acc = [r for r in records if r["accepted"]]
    print(f"\naccepted {len(acc)}/{len(records)} objects; manifest -> {MANIFEST.relative_to(ROOT)}; previews -> {outdir}")


if __name__ == "__main__":
    main()
