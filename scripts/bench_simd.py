#!/usr/bin/env python3
"""SIMD ablation: three builds x three weight formats.

  neon     hand-written NEON kernels (the default build)
  autovec  scalar fallbacks, which LLVM vectorizes on its own
  novec    the same scalar code with LLVM's vectorizers switched off

The middle build is the point: turning the feature off does not turn SIMD off,
so only the third one isolates what vectorization is worth.

Usage: scripts/bench_simd.py [runs] [tokens] [threads]
"""

import shutil
import sys
import tempfile
from pathlib import Path

import bench

NOVEC = "-C llvm-args=-vectorize-loops=false -C llvm-args=-vectorize-slp=false"
BUILDS = {
    "neon": dict(default_features=True),
    "autovec": dict(default_features=False),
    "novec": dict(default_features=False, rustflags=NOVEC),
}


def main(runs=10, tokens=100, threads=6):
    binaries = {}
    staging = Path(tempfile.mkdtemp(prefix="squirrel-simd-"))
    for name, options in BUILDS.items():
        print(f"building {name}...", file=sys.stderr)
        built = bench.build(**options)
        binaries[name] = shutil.copy(built, staging / f"sq_{name}")

    csv = bench.Csv(
        bench.RESULTS / "simd.csv",
        ["build", "format", "run", "ms_per_token"],
        bench.header(runs=runs, tokens=tokens, threads=threads),
    )

    for build, binary in binaries.items():
        for name, model in bench.MODELS.items():
            run_one = lambda: bench.run_engine(
                model, tokens=tokens, threads=threads, binary=binary
            )
            run_one()  # warmup

            times = []
            for run in range(1, runs + 1):
                ms = bench.decode_mean(run_one())
                times.append(ms)
                csv.row(build, name, run, f"{ms:.2f}")

            m, sd = bench.summarize(times)
            print(f"{build:<8} {name:<5}: {m:7.2f} +/- {sd:.2f} ms/token",
                  file=sys.stderr)

    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
