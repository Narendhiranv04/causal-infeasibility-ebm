"""Reproducibly fetch the minimal asset subset used by Articulated-Recourse-v0.

RoboCasa ships its assets as a handful of multi-GB zip archives on Box. Box supports
HTTP range requests, so this script reads each archive's zip central directory remotely
and extracts only the members named in configs/assets.yaml (one dishwasher fixture, the
object screening pool and four textures; roughly 60 MB instead of ~10 GB). The Franka
model comes from a sparse checkout of mujoco_menagerie at a pinned commit.

Usage:
    python assets/download_assets.py            # fetch everything missing
    python assets/download_assets.py --verify   # only check files recorded in the log

Everything lands in .asset_cache/ (gitignored). A download log with sizes and sha256
hashes is written to .asset_cache/download_log.json. Validation and the manifest
(assets/asset_manifest.json) are produced afterwards by scripts/preview_assets.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".asset_cache"
CONFIG = ROOT / "configs" / "assets.yaml"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, timeout: float = 120.0):
        self.timeout = timeout
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.url = r.geturl()  # resolved (redirected) URL
        self.source_url = url
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except Exception as e:  # signed redirect URLs expire; re-resolve and retry
                last = e
                time.sleep(1.0 + attempt)
                try:
                    req = urllib.request.Request(self.source_url, headers={"Range": "bytes=0-0"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        self.url = r.geturl()
                except Exception:
                    pass
        raise RuntimeError(f"range request failed for {self.source_url}: {last}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = self._get(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def box_direct_url(shared_url: str) -> str:
    """Same conversion as robocasa/scripts/download_kitchen_assets.py."""
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.zip"


def open_archive(spec: dict) -> zipfile.ZipFile:
    url = box_direct_url(spec["box_url"])
    for attempt in range(6):
        try:
            raw = HttpRangeFile(url)
            break
        except Exception:  # transient DNS / network failures
            if attempt == 5:
                raise
            time.sleep(5.0 * (attempt + 1))
    if spec.get("bytes") and raw.size != spec["bytes"]:
        print(f"  WARNING: archive size changed upstream ({raw.size} != {spec['bytes']})")
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_prefixes(zf: zipfile.ZipFile, prefixes: list[str], dest: Path, log: dict, archive: str):
    infos = [i for i in zf.infolist() if not i.is_dir() and any(i.filename.startswith(p) for p in prefixes)]
    found = {p for p in prefixes if any(i.filename.startswith(p) for i in infos)}
    missing = [p for p in prefixes if p not in found]
    for info in infos:
        out = dest / info.filename
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            dst.write(src.read())
    for info in infos:
        out = dest / info.filename
        log[str(out.relative_to(CACHE))] = {
            "archive": archive,
            "member": info.filename,
            "bytes": info.file_size,
            "sha256": sha256(out),
        }
    return missing


def fetch_git_subdir(repo: str, commit: str, subdir: str, dest: Path):
    subdirs = subdir.split()
    if all((dest / s).exists() for s in subdirs):
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
    if not dest.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *subdirs], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only verify hashes in the download log")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(CONFIG.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    log_path = CACHE / "download_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    if args.verify:
        bad = [k for k, v in log.items() if not (CACHE / k).exists() or sha256(CACHE / k) != v["sha256"]]
        print(f"{len(log) - len(bad)}/{len(log)} files verified")
        for k in bad:
            print("  MISMATCH", k)
        return 1 if bad else 0

    archives = cfg["sources"]["robocasa"]["archives"]
    wanted: dict[str, list[str]] = {}
    fx = cfg["fixture"]
    wanted.setdefault(fx["archive"], []).append(fx["member_prefix"])
    wanted.setdefault("textures", []).extend(cfg["textures"].values())
    default_archive = cfg["objects"]["archive"]
    gso_models = []
    for cat, spec in cfg["objects"]["categories"].items():
        if spec.get("source") == "scanned_objects":
            gso_models += [f"models/{inst}" for inst in spec["pool"]]
            continue
        arch = spec.get("archive", default_archive)
        top = "objaverse" if arch == "objaverse" else "lightwheel"
        for inst in spec["pool"]:
            wanted.setdefault(arch, []).append(f"{top}/{spec['robocasa_category']}/{inst}/")

    problems = []
    for arch, prefixes in wanted.items():
        print(f"[robocasa:{arch}] {len(prefixes)} member groups")
        try:
            zf = open_archive(archives[arch])
        except Exception as e:  # noqa: BLE001
            problems.append(f"cannot open RoboCasa archive '{arch}' ({archives[arch]['box_url']}): {e}")
            continue
        missing = extract_prefixes(zf, prefixes, CACHE / "robocasa" / arch, log, arch)
        problems += [f"'{p}' not found in RoboCasa archive '{arch}'" for p in missing]
        log_path.write_text(json.dumps(log, indent=1, sort_keys=True))

    men = cfg["sources"]["menagerie"]
    print(f"[menagerie] {men['subdir']} @ {men['commit'][:10]}")
    try:
        fetch_git_subdir(men["repo"], men["commit"], men["subdir"], CACHE / "mujoco_menagerie")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot fetch {men['repo']} {men['subdir']}: {e}")

    if gso_models:
        gso = cfg["sources"]["scanned_objects"]
        print(f"[scanned_objects] {len(gso_models)} models @ {gso['commit'][:10]}")
        try:
            fetch_git_subdir(gso["repo"], gso["commit"], " ".join(gso_models), CACHE / "mujoco_scanned_objects")
        except Exception as e:  # noqa: BLE001
            problems.append(f"cannot fetch {gso['repo']} {gso_models}: {e}")

    log_path.write_text(json.dumps(log, indent=1, sort_keys=True))
    total = sum(v["bytes"] for v in log.values())
    print(f"done: {len(log)} RoboCasa files, {total / 1e6:.1f} MB in {CACHE}")
    if problems:
        print("\nSTOP - the following assets could not be fetched automatically:")
        for p in problems:
            print("  -", p)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
