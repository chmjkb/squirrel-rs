"""The evaluation matrix: three weight formats x two activation paths.
Usage: scripts/bench_perplexity.py [windows]
"""

import os
import subprocess
import sys
from datetime import datetime

import bench

# (model, f32 activations?) — comment is what the configuration is called.
CONFIGS = [
    ("BF16", True),  # W16A32, the reference
    ("Q8_0", False),  # W8A8, the production path
    ("Q8_0", True),  # W8A32, weight-only ablation
    ("Q4_0", False),  # W4A8, the production path
    ("Q4_0", True),  # W4A32, weight-only ablation
]


def main(windows=64):
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=bench.ROOT,
        check=True,
        stderr=subprocess.DEVNULL,
    )
    log_path = bench.RESULTS / "perplexity.log"
    log_path.parent.mkdir(exist_ok=True)

    with open(log_path, "a") as log:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"# {stamp} windows={windows}", file=log, flush=True)

        for name, f32_acts in CONFIGS:
            print(f">>> {name} f32_acts={int(f32_acts)} windows={windows}")
            # The per-window progress stays on the terminal; only the final
            # summary line goes to the log.
            summary = subprocess.run(
                [
                    "./target/release/perplexity",
                    "--model",
                    bench.MODELS[name],
                    "--windows",
                    str(windows),
                ],
                cwd=bench.ROOT,
                text=True,
                stdout=subprocess.PIPE,
                env={**os.environ, "SQUIRREL_F32_ACTS": str(int(f32_acts))},
            ).stdout
            print(summary, end="")
            print(summary, end="", file=log, flush=True)

    print(f"done - appended to {log_path.relative_to(bench.ROOT)}")


if __name__ == "__main__":
    main(*bench.args(__doc__))
