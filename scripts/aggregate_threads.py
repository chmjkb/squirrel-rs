#!/usr/bin/env python3
"""Collapse results/threads.csv into one row per thread count, for pgfplots.

Usage: scripts/aggregate_threads.py
"""

import csv
from collections import defaultdict

import bench

FORMATS = ["BF16", "Q8_0", "Q4_0"]
OUT = bench.ROOT / "thesis/figures/thread_scaling.csv"


def main():
    times = defaultdict(list)
    with open(bench.RESULTS / "threads.csv") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            times[row["format"], int(row["threads"])].append(float(row["ms_per_token"]))

    header = ["threads"] + [f"{fmt}_{stat}" for fmt in FORMATS for stat in ("mean", "sd")]
    with open(OUT, "w") as f:
        print(",".join(header), file=f)
        for threads in sorted({t for _, t in times}):
            cells = []
            for fmt in FORMATS:
                values = times.get((fmt, threads))
                if values:
                    m, sd = bench.summarize(values)
                    cells += [f"{m:.2f}", f"{sd:.2f}"]
                else:
                    cells += ["", ""]
            print(",".join([str(threads)] + cells), file=f)

    print(f"wrote {OUT.relative_to(bench.ROOT)}")
    print(open(OUT).read(), end="")


if __name__ == "__main__":
    main()
