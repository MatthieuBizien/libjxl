#!/usr/bin/env python3
"""
repro_spotcolor_diff.py – Minimal reproducible example for the spot-colour
mismatch between the reference C++ decoder (djxl) and the Rust decoder.

Prerequisites
-------------
1. Build the reference decoder once:
     cd libjxl && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
     cmake --build . --target djxl -j$(sysctl -n hw.ncpu)

2. Ensure the Rust workspace builds (this script will invoke `cargo run`).

Usage
-----
    python repro_spotcolor_diff.py [path/to/spot.jxl]

The script:
  • decodes the chosen JXL file twice (reference + Rust) into temporary .npy
    arrays using NumPy's ndarray container;
  • compares the two arrays and prints the maximum absolute per-sample
    difference plus a count of pixels that differ by more than 1e-5.

Exit status is 0 when the arrays are identical (within a 1e-5 tolerance),
otherwise 1.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

def run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

def decode_with_djxl(jxl_path: Path, out_npy: Path):
    djxl = Path("libjxl/build/tools/djxl")
    if not djxl.exists():
        sys.exit("Error: reference decoder not built – please run the cmake/build commands in the script header.")
    run([str(djxl), str(jxl_path), str(out_npy), "--output_format", "npy", "--quiet"])


def decode_with_rust_cli(jxl_path: Path, out_npy: Path):
    # Build + run in one go; stderr/stdout suppressed for brevity.
    run([
        "cargo",
        "run",
        "--release",
        "--package",
        "jxl_cli",
        "--bin",
        "jxl_cli",
        str(jxl_path),
        str(out_npy),
    ])


def compare_arrays(ref_npy: Path, rust_npy: Path) -> bool:
    ref = np.load(ref_npy)
    rust = np.load(rust_npy)

    if ref.shape != rust.shape:
        print("Shape mismatch:", ref.shape, "vs", rust.shape)
        return False

    diff = np.abs(ref - rust)
    max_diff = diff.max()
    mean_diff = float(diff.mean())
    different_pixels = int((diff > 1e-5).sum())

    print("Shapes        :", ref.shape)
    print("dtype         :", ref.dtype)
    print("max |Δ|       :", max_diff)
    print("mean |Δ|      :", mean_diff)
    print("pixels |Δ|>1e-5:", different_pixels)

    return max_diff <= 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-colour golden comparison")
    parser.add_argument(
        "jxl",
        nargs="?",
        default="jxl/resources/test/conformance_test_images/spot.jxl",
        help="Path to a JXL file containing spot colours",
    )
    args = parser.parse_args()

    jxl_path = Path(args.jxl)
    if not jxl_path.exists():
        sys.exit(f"Input JXL file not found: {jxl_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ref_npy = tmpdir / "ref.npy"
        rust_npy = tmpdir / "rust.npy"

        decode_with_djxl(jxl_path, ref_npy)
        decode_with_rust_cli(jxl_path, rust_npy)

        ok = compare_arrays(ref_npy, rust_npy)

    if ok:
        print("Correct: Image is identical")
    else:
        print("Error: Image is different")
        sys.exit(1)


if __name__ == "__main__":
    main() 