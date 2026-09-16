"""What a cold start costs, measured on the first forward pass.
sudo scripts/bench_coldstart.py [tokens] [reps]
"""

import subprocess
import sys

import bench


def main(tokens=10, reps=10, threads=6):
    bench.build()
    csv = bench.Csv(
        bench.RESULTS / "coldstart.csv",
        ["format", "state", "rep", "prefill_ms", "decode_ms_per_token"],
        bench.header(threads=threads),
    )

    for name in ("Q4_0", "Q8_0", "BF16"):
        model = bench.MODELS[name]
        cold_prefills, warm_prefills = [], []
        for rep in range(1, reps + 1):
            subprocess.run(["purge"], check=True)
            for state, collect in (("cold", cold_prefills), ("warm", warm_prefills)):
                steps = bench.step_times(
                    bench.run_engine(model, tokens=tokens, threads=threads)
                )
                prefill, decode = steps[0], sum(steps[1:]) / len(steps[1:])
                collect.append(prefill)
                csv.row(name, state, rep, f"{prefill:.2f}", f"{decode:.2f}")

        cold = sum(cold_prefills) / reps
        warm = sum(warm_prefills) / reps
        print(
            f"{name:<5} first pass: cold {cold:7.1f} ms   warm {warm:7.1f} ms"
            f"   ({cold / warm:.1f}x)",
            file=sys.stderr,
        )

    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
