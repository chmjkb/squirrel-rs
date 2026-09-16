"""Are the spikes in the no-cache curve reproducible?
Usage: scripts/bench_kv_cache_repeat.py [max_tokens] [runs]
"""

import os
import sys

import bench


def main(max_tokens=300, runs=10):
    bench.build()
    model = os.environ.get("MODEL", bench.MODELS["Q8_0"])
    threads = int(os.environ.get("THREADS", 6))

    csv = bench.Csv(
        bench.RESULTS / "kv_cache_spikes.csv",
        ["run", "step", "ms"],
        bench.header(max=max_tokens, runs=runs, threads=threads, cache="off"),
    )
    for run in range(1, runs + 1):
        print(f"run {run}/{runs} (cache off, {max_tokens} tokens)", file=sys.stderr)
        log = bench.run_engine(
            model, tokens=max_tokens, threads=threads, env={"SQUIRREL_NO_KV_CACHE": "1"}
        )
        for step, ms in enumerate(bench.step_times(log)[1:], start=1):
            csv.row(run, step, f"{ms:.2f}")
    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
