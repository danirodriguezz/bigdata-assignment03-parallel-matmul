#!/usr/bin/env python3
"""Generate the shared input matrices A and B for every matrix size of the experiment.

Format: raw little-endian float64, row-major, n*n values, files
  data/inputs/n{n}_s{seed}_A.bin and data/inputs/n{n}_s{seed}_B.bin
plus a JSON sidecar n{n}_s{seed}.json (generator, seed, range, dtype, SHA-256) and the
checksum list data/inputs/checksums.sha256 (committed; the .bin files are git-ignored).

Generator: numpy.random.default_rng(seed) (PCG64), A then B drawn with rng.uniform(-1, 1, (n, n)).
Usage: python scripts/gen_inputs.py --config configs/experiment.json [--sizes 64 128]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "inputs"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(n, seed):
    return OUT / f"n{n}_s{seed}_A.bin", OUT / f"n{n}_s{seed}_B.bin", OUT / f"n{n}_s{seed}.json"


def generate(n, seed, lo=-1.0, hi=1.0):
    pa, pb, meta = paths(n, seed)
    if meta.exists() and pa.exists() and pb.exists():
        info = json.loads(meta.read_text())
        if info.get("sha256_A") == sha256(pa) and info.get("sha256_B") == sha256(pb):
            return info  # already generated and intact
    rng = np.random.default_rng(seed)
    A = rng.uniform(lo, hi, size=(n, n)).astype("<f8")
    B = rng.uniform(lo, hi, size=(n, n)).astype("<f8")
    OUT.mkdir(parents=True, exist_ok=True)
    A.tofile(pa)
    B.tofile(pb)
    info = {"n": n, "seed": seed, "generator": f"numpy {np.__version__} default_rng (PCG64)",
            "distribution": f"uniform[{lo}, {hi})", "dtype": "float64 little-endian, row-major",
            "order": "A drawn first, then B, from the same generator",
            "file_A": pa.name, "file_B": pb.name, "sha256_A": sha256(pa), "sha256_B": sha256(pb)}
    meta.write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--sizes", type=int, nargs="*", help="override the size list")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    seed = cfg["seed"]
    sizes = args.sizes or sorted(set(cfg["all_input_sizes"]))
    lines = []
    for n in sizes:
        info = generate(n, seed)
        lines += [f"{info['sha256_A']}  {info['file_A']}", f"{info['sha256_B']}  {info['file_B']}"]
        print(f"n={n}: {info['file_A']} {info['file_B']}")
    if not args.sizes:
        (OUT / "checksums.sha256").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
