"""Effect of prompt length on prefill and decode.
Usage: scripts/bench_prompt_length.py [runs] [gen_tokens]
"""

import os
import sys

import bench


def main(runs=10, gen_tokens=50):
    bench.build()
    model = os.environ.get("MODEL", bench.MODELS["Q8_0"])
    threads = int(os.environ.get("THREADS", 6))
    prompts = os.environ.get("PROMPTS", "xs s m l xl").split()
    append = os.environ.get("APPEND") == "1"

    csv = bench.Csv(
        bench.RESULTS / "prompt_length.csv",
        ["prompt", "prompt_tokens", "run", "prefill_ms", "decode_ms_per_token"],
        bench.header(
            runs=runs, gen=gen_tokens, threads=threads, prompts=",".join(prompts)
        ),
        append=append,
    )

    for name in prompts:
        path = bench.ROOT / "prompts" / f"{name}.txt"
        run_one = lambda: bench.run_engine(
            model, tokens=gen_tokens, threads=threads, prompt_file=path
        )
        tokens = bench.prompt_tokens(run_one())  # warmup, also counts the prompt

        prefills, decodes = [], []
        for run in range(1, runs + 1):
            steps = bench.step_times(run_one())
            prefill, decode = steps[0], sum(steps[1:]) / len(steps[1:])
            prefills.append(prefill)
            decodes.append(decode)
            csv.row(name, tokens, run, f"{prefill:.2f}", f"{decode:.2f}")

        pf, dc = sum(prefills) / runs, sum(decodes) / runs
        print(
            f"{name:<3} {tokens:5d} tokens: prefill {pf:8.1f} ms   "
            f"decode {dc:6.2f} ms/token   ({pf / tokens:.2f} ms per prompt token)",
            file=sys.stderr,
        )

    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
