"""Comparison against llama.cpp on the same machine and the same model files.
Usage: scripts/bench_llamacpp.py [threads] [reps]
"""

import os
import subprocess
import sys

import bench

PROMPTS = "6,39,389,1532,4416"
GEN = 100


def bench_one(ngl, out_path, threads, reps):
    with open(out_path, "w") as f:
        print(bench.header(threads=threads, reps=reps, ngl=ngl), file=f)
        for name, model in bench.MODELS.items():
            print(f">>> {name} (ngl={ngl})", file=sys.stderr)
            rows = subprocess.run(
                [
                    "llama-bench",
                    "-m",
                    f"assets/{model}",
                    "-ngl",
                    str(ngl),
                    "-t",
                    str(threads),
                    "-p",
                    PROMPTS,
                    "-n",
                    str(GEN),
                    "-r",
                    str(reps),
                    "-o",
                    "csv",
                ],
                cwd=bench.ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
            for line in rows.splitlines():
                print(f"{name},{line}", file=f)
    print(f"wrote {out_path.relative_to(bench.ROOT)}")


def main(threads=6, reps=10):
    mode = os.environ.get("MODE", "both")
    if mode in ("cpu", "both"):
        bench_one(0, bench.RESULTS / "llamacpp.csv", threads, reps)
    if mode in ("gpu", "both"):
        bench_one(99, bench.RESULTS / "llamacpp_gpu.csv", threads, reps)


if __name__ == "__main__":
    main(*bench.args(__doc__))
