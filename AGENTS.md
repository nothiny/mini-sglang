# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives under `python/minisgl/`. Keep changes close to their subsystem: model definitions in `models/`, serving entry points in `server/`, request execution in `scheduler/` and `engine/`, KV-cache implementations in `kvcache/`, and attention backends in `attention/`. Native C++/CUDA sources and Triton kernels are under `python/minisgl/kernel/`. Tests are grouped by concern in `tests/core/`, `tests/kernel/`, and `tests/misc/`. Use `benchmark/offline/` and `benchmark/online/` for performance scripts, `docs/` for design documentation, and `assets/` for documentation images.

## Build, Test, and Development Commands

- `uv venv --python=3.12 && source .venv/bin/activate` creates the recommended environment.
- `uv pip install -e ".[dev]"` installs Mini-SGLang and contributor tools in editable mode.
- `pytest` runs the configured suite and writes terminal plus HTML coverage reports.
- `pre-commit run --all-files` applies repository checks, including Black, Ruff, and clang-format.
- `mypy python/minisgl` runs the strict type checks configured in `pyproject.toml`.
- `python -m minisgl --model Qwen/Qwen3-0.6B` starts the OpenAI-compatible server locally.
- `docker build -t minisgl .` builds the CUDA-enabled container.

## Coding Style & Naming Conventions

Use four-space indentation, Python type annotations, and a 100-character line limit. Black owns formatting; Ruff checks imports, Pyflakes errors, warnings, and comprehension style. Name modules, functions, and variables `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Preserve existing subsystem abstractions rather than adding cross-package shortcuts. Run pre-commit before submitting; native C++ and CUDA files are formatted by clang-format.

## Testing Guidelines

Pytest discovers `test_*.py`/`*_test.py`, `Test*` classes, and `test_*` functions. Add focused regression tests in the matching test directory. Coverage is collected for `minisgl`; no minimum threshold is declared, but new logic should exercise success and failure paths. Kernel, distributed, and end-to-end tests may require Linux, an NVIDIA GPU, a matching CUDA toolkit, model access, or multiple GPUs; record the exact hardware and command when those tests cannot run everywhere.

## Commit & Pull Request Guidelines

Recent commits favor imperative subjects prefixed with `[Fix]`, `[Feature]`, or `[Minor]`, for example `[Fix] Stabilize decode batch request order across TP ranks`. Keep each commit scoped to one concern. Pull requests should explain the problem and solution, link relevant issues, list validation commands and hardware, and include before/after benchmark results for performance-sensitive changes. Call out API, model-compatibility, configuration, or documentation impacts explicitly.
