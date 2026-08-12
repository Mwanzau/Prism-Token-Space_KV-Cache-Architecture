**Prism Token-Space / KV-Cache Architecture**

Overview
-
Prism is a lightweight offline token-space and key-value cache architecture designed for edge RAG workflows. It provides a compact append-only `.prism` binary artifact format, multi-tokenizer adapters (GGUF / Hugging Face / tokenizers), document ingestion utilities, and tools for tokenizer stress-testing and benchmarking.

Highlights
-
- Append-only `.prism` artifacts with an atomic tail-index for incremental ingestion
- Support for GGUF token extraction, HF `transformers`, and `tokenizers` formats
- Document importers with chunking and sentence-level QA provenance
- Stress-test tooling to validate tokenizer throughput and memory usage

Quickstart
-
Prerequisites
- Python 3.10+ (recommended)
- Git
- Optional: `transformers` / `tokenizers` packages for HuggingFace or tokenizer.json use

Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run tests
```bash
source .venv/bin/activate
python -m pytest -q
```

Build a `.prism` from a text document (example)
```bash
source .venv/bin/activate
python -m prism_protocol.src.token_space.importer \
	--input File_Samples_For_Tests/The\ black\ barque.txt \
	--model models/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
	--output output/test.prism --chunk-size 128 --append
```

Run the demo QA CLI
```bash
source .venv/bin/activate
python run_rag_demo.py qa --prism output/test.prism --query "What animal is mentioned?"
```

Stress-test tokenizers (tokenize all samples with GGUF/HF models)
```bash
source .venv/bin/activate
PYTHONPATH=. python tools/stress_test_gguf.py --models-dir models --samples-dir File_Samples_For_Tests --iterations 1
```

Files not included in repository
-
Large binary artifacts such as model files and sample corpora are intentionally excluded. The following folders are git-ignored:

- `models/` — place your GGUF or model artifacts here locally
- `File_Samples_For_Tests/` — large sample documents used for benchmarking

Contributing
-
Please open issues or pull requests on GitHub. If you want to reproduce experiments, add smaller sample files or share scripts that point to local model artifacts.

License
-
MIT — see the `LICENSE` file (if present) or add one before redistributing.
