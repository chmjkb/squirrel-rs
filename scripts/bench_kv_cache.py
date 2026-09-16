"""KV-cache ablation: per-step decode time with the cache on and off.
Usage: scripts/bench_kv_cache.py [max_tokens] [every]
"""

import os
import sys

import bench


def main(max_tokens=1000, every=10):
    bench.build()
    model = os.environ.get("MODEL", bench.MODELS["Q8_0"])

    print(f"run 1/2: cache ON  ({max_tokens} tokens)", file=sys.stderr)
    on = bench.step_times(bench.run_engine(model, tokens=max_tokens))

    print(
        f"run 2/2: cache OFF ({max_tokens} tokens - quadratic, the slow one)",
        file=sys.stderr,
    )
    off = bench.step_times(
        bench.run_engine(model, tokens=max_tokens, env={"SQUIRREL_NO_KV_CACHE": "1"})
    )

    csv = bench.Csv(
        bench.RESULTS / "kv_cache_scaling.csv",
        ["step", "cache_on_ms", "cache_off_ms"],
        bench.header(max_tokens=max_tokens, every=every),
    )
    # Step 1 is the prefill, so decode steps start at index 1.
    for step, (a, b) in enumerate(zip(on[1:], off[1:]), start=1):
        if step % every == 0:
            csv.row(step, f"{a:.2f}", f"{b:.2f}")
    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
