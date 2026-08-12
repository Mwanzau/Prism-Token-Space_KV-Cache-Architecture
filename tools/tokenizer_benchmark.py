"""Simple tokenizer benchmark helper.

Usage:
  PYTHONPATH=. python3 tools/tokenizer_benchmark.py --gguf models/.. --file File_Samples_For_Tests/large.txt --iterations 5

Measures tokens/sec for the configured tokenizer and prints summary.
"""
import argparse
import time
from pathlib import Path

from prism_protocol.src.token_space.tokenizer_adapter import get_tokenizer_adapter


def benchmark_text(tokenizer, text: str, iterations: int = 3) -> dict:
    # warmup
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    # timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        _ = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    runs = iterations
    tokens_processed = total_tokens * runs
    return {
        "iterations": runs,
        "total_tokens_once": total_tokens,
        "tokens_processed": tokens_processed,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_processed / elapsed if elapsed > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer benchmark helper")
    parser.add_argument("--gguf", type=Path, required=True, help="GGUF model file (or other supported tokenizer resource)")
    parser.add_argument("--file", type=Path, required=True, help="Text file to tokenize for benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of repeated tokenization runs")
    args = parser.parse_args()

    if not args.gguf.exists():
        print(f"GGUF/tokenizer resource not found: {args.gguf}")
        return 2
    if not args.file.exists():
        print(f"Input file not found: {args.file}")
        return 2

    tokenizer = get_tokenizer_adapter(str(args.gguf))
    text = args.file.read_text(encoding="utf-8")

    print(f"Warmup + running {args.iterations} iterations...")
    stats = benchmark_text(tokenizer, text, iterations=args.iterations)

    print("\nBenchmark results:")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Tokens per single run: {stats['total_tokens_once']}")
    print(f"  Total tokens processed: {stats['tokens_processed']}")
    print(f"  Elapsed (s): {stats['elapsed_s']:.4f}")
    print(f"  Throughput (tokens/sec): {stats['tokens_per_sec']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
