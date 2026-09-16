"""Shared helpers for the benchmark scripts."""

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MODELS = {
    "Q8_0": "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "Q4_0": "Llama-3.2-1B-Instruct-Q4_0-pure.gguf",
    "BF16": "Llama-3.2-1B-Instruct-BF16.gguf",
}

_MS = re.compile(r"(\d+\.\d+)ms")
_PROMPT_TOKENS = re.compile(r"prompt: (\d+) tokens")


def build(*, default_features=True, rustflags=None, quiet=True):
    env = {**os.environ}
    if rustflags:
        env["RUSTFLAGS"] = rustflags
    cmd = ["cargo", "build", "--release"]
    if not default_features:
        cmd.append("--no-default-features")
    subprocess.run(
        cmd, cwd=ROOT, env=env, check=True,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    return ROOT / "target/release/squirrel-rs"


def run_engine(model, *, tokens=None, threads=None, prompt_file=None,
               binary=None, env=None):
    """Run one generation and return what the engine logged to stderr."""
    e = {**os.environ, "SQUIRREL_IGNORE_EOS": "1"}
    if threads is not None:
        e["RAYON_NUM_THREADS"] = str(threads)
    e.update(env or {})

    cmd = [str(binary or ROOT / "target/release/squirrel-rs"), "--model", model]
    if tokens is not None:
        cmd += ["--max-tokens", str(tokens)]
    if prompt_file is not None:
        cmd += ["--prompt-file", str(prompt_file)]

    return subprocess.run(
        cmd, cwd=ROOT, env=e, text=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    ).stderr


def step_times(log):
    """Milliseconds per step. The first entry is the prompt prefill."""
    return [float(x) for x in _MS.findall(log)]


def prompt_tokens(log):
    m = _PROMPT_TOKENS.search(log)
    return int(m.group(1)) if m else None


def decode_mean(log):
    """Mean over the decode steps, excluding the prefill."""
    steps = step_times(log)[1:]
    return sum(steps) / len(steps)


def summarize(values):
    """Mean and population standard deviation, accumulated the way the shell
    scripts did, so old and new runs round identically."""
    n = len(values)
    m = sum(values) / n
    variance = sum(v * v for v in values) / n - m * m
    return m, math.sqrt(max(variance, 0.0))


def header(**fields):
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"# {stamp}  commit {commit}  {parts}".rstrip()


class Csv:
    """Writes the '# metadata' line, the column names, then rows as they come."""

    def __init__(self, path, columns, meta, append=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a" if append else "w")
        print(meta, file=self.file, flush=True)
        if not append:
            print(",".join(columns), file=self.file, flush=True)

    def row(self, *values):
        print(",".join(str(v) for v in values), file=self.file, flush=True)

    def close(self):
        self.file.close()
        print(f"wrote {self.path.relative_to(ROOT)}")


def args(doc):
    """Positional integer arguments, or the script's docstring for --help."""
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        print(doc.strip())
        sys.exit(0)
    return [int(a) for a in argv]
