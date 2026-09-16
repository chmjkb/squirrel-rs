"""Memory use per weight format: peak resident set against physical footprint.
Usage: scripts/bench_memory.py [tokens]
"""

import os
import re
import subprocess
import sys
import time

import bench

MB = 1024 * 1024


def peak_rss_mb(model, tokens):
    """macOS time(1) reports maximum resident set size in bytes."""
    out = subprocess.run(
        [
            "/usr/bin/time",
            "-l",
            "./target/release/squirrel-rs",
            "--model",
            model,
            "--max-tokens",
            str(tokens),
        ],
        cwd=bench.ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ).stderr
    return int(re.search(r"(\d+)\s+maximum resident set size", out).group(1)) // MB


def footprint_mb(model):
    """Sampled mid-decode, so the process is still alive when vmmap attaches."""
    proc = subprocess.Popen(
        ["./target/release/squirrel-rs", "--model", model, "--max-tokens", "5000"],
        cwd=bench.ROOT,
        env={**os.environ, "SQUIRREL_IGNORE_EOS": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(6)
        summary = subprocess.run(
            ["/usr/bin/vmmap", "--summary", str(proc.pid)],
            capture_output=True,
            text=True,
        ).stdout
    finally:
        proc.kill()
        proc.wait()

    footprint = re.search(r"Physical footprint:\s+([\d.]+)([MG])", summary)
    mapped = re.search(r"mapped file\s+(\S+)", summary)
    value, unit = float(footprint.group(1)), footprint.group(2)
    return value * 1024 if unit == "G" else value, mapped.group(1) if mapped else "?"


def main(tokens=100):
    bench.build()
    csv = bench.Csv(
        bench.RESULTS / "memory.csv",
        ["format", "file_mb", "peak_rss_mb", "footprint_mb"],
        bench.header(tokens=tokens),
    )

    for name in ("Q4_0", "Q8_0", "BF16"):
        model = bench.MODELS[name]
        file_mb = (bench.ROOT / "assets" / model).stat().st_size // MB
        rss = peak_rss_mb(model, tokens)
        footprint, mapped = footprint_mb(model)
        csv.row(name, file_mb, rss, f"{footprint:g}")
        print(
            f"{name:<5} file {file_mb:5d} MB   peak RSS {rss:5d} MB   "
            f"footprint {footprint:6g} MB   mapped {mapped}",
            file=sys.stderr,
        )

    csv.close()


if __name__ == "__main__":
    main(*bench.args(__doc__))
