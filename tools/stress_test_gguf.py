"""Stress test GGUF tokenizer performance and memory usage."""
import argparse
import os
import time
from pathlib import Path
from typing import List

from prism_protocol.src.token_space.tokenizer_adapter import get_tokenizer_adapter


def measure_memory(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except FileNotFoundError:
        return 0.0
    return 0.0


def collect_sample_files(sample_path: Path) -> List[Path]:
    if sample_path.is_file():
        return [sample_path]
    if sample_path.is_dir():
        return sorted(
            [
                p
                for p in sample_path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}
            ]
        )
    raise FileNotFoundError(f"Sample path not found: {sample_path}")


def collect_model_files(models_arg: str, models_dir: Path) -> List[str]:
    models = []
    if models_arg:
        models.extend(models_arg.split(","))
    if models_dir and models_dir.exists():
        models.extend(str(p) for p in models_dir.rglob("*.gguf"))
    return [m for m in models if m]


def tokenize_sample(tokenizer, sample: Path) -> tuple[float, int]:
    text = sample.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return 0.0, 0
    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    print(f"  {sample.name}: {len(tokens)} tokens in {elapsed:.3f}s ({len(tokens)/elapsed:.0f} tokens/sec)")
    return elapsed, len(tokens)


def run_stress(models: List[str], samples: List[Path], iterations: int) -> None:
    pid = os.getpid()
    for model in models:
        print(f"\n=== Stress test model: {model} ===")
        try:
            tokenizer = get_tokenizer_adapter(model)
        except Exception as exc:
            print(f"Skipping model {model}: unable to load tokenizer adapter ({exc})")
            continue

        mem_start = measure_memory(pid)
        print(f"Initial RSS: {mem_start:.2f} MB")

        total_tokens = 0
        total_time = 0.0
        for sample in samples:
            try:
                elapsed, token_count = tokenize_sample(tokenizer, sample)
            except Exception as exc:
                print(f"  Skipping sample {sample.name}: tokenizer error ({exc})")
                continue
            total_time += elapsed
            total_tokens += token_count

        mem_mid = measure_memory(pid)
        print(f"After first pass RSS: {mem_mid:.2f} MB")

        for i in range(1, iterations):
            print(f"Iteration {i+1}/{iterations}")
            for sample in samples:
                elapsed = tokenize_sample(tokenizer, sample)
                total_time += elapsed
            mem_iter = measure_memory(pid)
            print(f"  RSS: {mem_iter:.2f} MB")

        print(f"Model {model} total text passes: {iterations}, wall time:{total_time:.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test tokenizer performance and memory for GGUF/HF models.")
    parser.add_argument("--models", default="", help="Comma-separated list of model paths or hf:<id> identifiers")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models directory to scan for .gguf files")
    parser.add_argument("--samples-dir", type=Path, default=Path("File_Samples_For_Tests"), help="Directory containing sample text/pdf files")
    parser.add_argument("--iterations", type=int, default=1, help="Number of repeated tokenization passes over all files")
    args = parser.parse_args()

    models = collect_model_files(args.models, args.models_dir)
    if not models:
        raise ValueError("No models found. Provide --models or set --models-dir with GGUF models.")

    samples = collect_sample_files(args.samples_dir)
    if not samples:
        raise ValueError("No sample files found in samples directory.")

    run_stress(models, samples, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
