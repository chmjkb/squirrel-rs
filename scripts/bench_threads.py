"""Decode speed against thread count, for each weight format.
Usage: scripts/bench_threads.py [runs] [tokens]
"""

import sys

import bench

THREAD_COUNTS = range(1, 13)


def main(runs=10, tokens=100):
    bench.build()
    csv = bench.Csv(
        bench.RESULTS / "threads.csv",
        ["format", "threads", "run", "ms_per_token"],
        bench.header(runs=runs, tokens=tokens),
    )

    for name, model in bench.MODELS.items():
        for threads in THREAD_COUNTS:
            bench.run_engine(model, tokens=tokens, threads=threads)  # warmup

            times = []
            for run in range(1, runs + 1):
                log = bench.run_engine(model, tokens=tokens, threads=threads)
                ms = bench.decode_mean(log)
                times.append(ms)
                csv.row(name, threads, run, f"{ms:.2f}")

            m, sd = bench.summarize(times)
            print(
                f"{name:<5} {threads:2d} threads: {m:6.2f} +/- {sd:.2f} "
                f"ms/token  ({1000 / m:.1f} tok/s)",
                file=sys.stderr,
            )

    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
