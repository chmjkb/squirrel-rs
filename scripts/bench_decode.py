"""Position-matched decode comparison between two weight formats.
Usage: scripts/bench_decode.py [rounds]
"""

import sys
from statistics import mean

import bench

FORMATS = ["Q8_0", "Q4_0"]
STEPS = 100


def main(rounds=3):
    bench.build()
    for name in FORMATS:
        bench.run_engine(bench.MODELS[name])  # warmup: pages the weights in

    results = {name: [] for name in FORMATS}
    for r in range(1, rounds + 1):
        for name in FORMATS:
            steps = bench.step_times(bench.run_engine(bench.MODELS[name]))
            avg = mean(steps[1 : STEPS + 1])
            results[name].append(avg)
            print(f"round {r}  {name}: {avg:.3f} ms/token")

    print(f"--- mean over {rounds} rounds ---")
    for name in FORMATS:
        print(f"{name}: {mean(results[name]):.2f} ms/token")


if __name__ == "__main__":
    main(*bench.args(__doc__))
