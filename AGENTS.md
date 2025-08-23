# Repository Guidelines

## Project Structure & Module Organization
- `deepcompressor/`: core Python package.
  - `app/llm` and `app/diffusion`: entrypoints and configs for PTQ flows.
  - `quantizer/`, `nn/`, `backend/`, `utils/`: quantization logic, model structs, kernels, helpers.
  - `csrc/`: native/CUDA sources (if needed by backends).
- `examples/llm` and `examples/diffusion`: runnable configs, scripts, and docs.
- `assets/`: figures and images for docs.

## Build, Test, and Development Commands
- Create env and install deps:
  - `conda env create -f environment.yml`
  - `poetry install`
- Lint and format (Ruff):
  - `poetry run ruff check .`
  - `poetry run ruff format .`
- Run LLM PTQ locally (example):
  - `CUDA_VISIBLE_DEVICES=0 poetry run python -m deepcompressor.app.llm.ptq examples/llm/configs/awq.yaml --model-name llama-3-8b --model-path /dev/shm/Meta-Llama-3-8B --save-model true`
  - Outputs: logs under the configured run dir; quantized checkpoints and a copy under `/dev/shm/sonpt/calibrated_models/<model_name>`.
- Build package: `poetry build`

## Coding Style & Naming Conventions
- Python ≥ 3.10; type hints encouraged.
- Indentation: 4 spaces; max line length: 120; quotes: double. Enforced by Ruff (see `pyproject.toml`).
- Imports: grouped and sorted (`ruff` rule set includes `I`).
- Filenames: `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.

## Testing Guidelines
- This repo uses runnable examples as smoke tests. Validate changes by running configs in `examples/llm` or `examples/diffusion` and checking saved artifacts and metrics.
- If you add unit tests, prefer `pytest`, place under `tests/`, and name files `test_*.py`. Ensure `poetry run ruff check .` passes before opening a PR.

## Commit & Pull Request Guidelines
- Commits: concise, imperative tense; optionally prefix scope or impact (e.g., `[minor]`, `[major]`). Reference files or modules touched.
- PRs: include a clear description, linked issues, reproduction steps, and before/after results (logs, sample outputs, or screenshots). Note any model/config changes and update `README.md` or example configs when relevant.
- CI hygiene: run lint/format and at least one example config locally before requesting review.

## Security & Configuration Tips
- Large models and caches use `/dev/shm`; ensure sufficient shared memory. Set `CUDA_VISIBLE_DEVICES` to select GPUs.
- Do not commit datasets, model weights, or large binaries; rely on paths like `--model-path /path/to/model`.
- Keep secrets out of configs; prefer environment variables and local paths.
