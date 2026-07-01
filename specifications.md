# Specifications

## Metadata
- Project: inference_vllm
- Repository: /home/nbizon/Thèse/Projects/inference_vllm
- Main branch: main
- Last updated: 2026-07-01
- Status: active

## Scope
- In scope: offline vLLM inference for reachability benchmark datasets.
- Out of scope: dataset generation and downstream result analysis.

## Explicit Requirements

### REQ-0001
- Status: implemented
- Source: User request, 2026-07-01
- Description: Implement an inference script in this repository adapted to the reachability reasoning-budget sweep task from `LLM_benchmark/src/inference/run_reasoning_budget_sweep_2.py`.
- Acceptance criteria:
  - The script reads `system.txt` and `reachability_questions.txt` from experiment directories.
  - The script evaluates each question for every configured reasoning budget.
  - The script writes `all_results.txt` with `seq`, `distance`, `reasoning_budget`, `Q`, `R`, and `A` fields.
- Traceability:
  - Code: run_inference.py::run_experiment
  - Tests: python_compile::run_inference.py
  - Docs: README.md
  - ADR:
  - Commits: commit::HEAD

## Non-Functional Requirements

### NFR-0001
- Category: performance
- Status: implemented
- Source: User request and `/home/nbizon/Téléchargements/vllm_0_24_guide_codex.md`
- Description: The runner should be optimized for long-context vLLM batch inference without reducing model compatibility.
- Acceptance criteria:
  - vLLM is initialized once per process.
  - Prompts are rendered and token-counted before generation.
  - Prefix caching and chunked prefill are configurable and enabled by default.
  - Output token budgets are strictly bounded.
  - Incompatible or overlong prompts fail cleanly before generation.
- Traceability:
  - Code: run_inference.py::build_llm
  - Code: run_inference.py::prepare_experiments
  - Code: run_inference.py::validate_task_fits_context
  - Tests: python_compile::run_inference.py
  - Docs: README.md
  - ADR:
  - Commits: commit::HEAD

### NFR-0002
- Category: performance
- Status: implemented
- Source: User request and `/home/nbizon/Téléchargements/vllm_gpu_time_codex_instructions.md`
- Description: The reasoning-budget sweep should avoid repeated long-context prefill work when deterministic generation makes budget prefixes reusable.
- Acceptance criteria:
  - The default sweep mode generates reasoning once at the largest budget and derives smaller reasoning budgets by token-prefix truncation.
  - The reuse mode is rejected when `temperature` is not `0.0`.
  - The independent per-budget path remains available as a validation baseline.
  - Final answers are generated in descending budget order to benefit from prefix caching.
- Traceability:
  - Code: run_inference.py::generate_budget_sweep_reuse_longest
  - Code: run_inference.py::decode_token_prefix
  - Code: run_inference.py::run_experiment
  - Tests: python_compile::run_inference.py
  - Docs: README.md
  - ADR:
  - Commits: commit::HEAD

## Assumptions

### ASM-0001
- Status: proposed
- Source: Codex
- Description: The benchmark task can use a compatibility two-pass offline vLLM flow for reasoning then final answer, because offline vLLM text generation is broadly supported across model families.
- Reason: The OpenAI-compatible server path exposes model-specific reasoning metadata for some models, while the offline path provides better batch efficiency and avoids server lifecycle overhead.
- Risk if wrong: A target model may require a server-specific reasoning parser to expose hidden reasoning fields.
- Validation plan: Run a small dataset split and compare output shape and accuracy against the existing `run_reasoning_budget_sweep_2.py` for the same model.
- Traceability:
  - Code: run_inference.py::generate_reasoning_then_answer
  - Tests:
  - Docs:
  - ADR:
  - Commits: commit::HEAD

## Open Questions

## Architecture Decisions

## Traceability Index
- REQ-0001 -> run_inference.py::run_experiment -> python_compile::run_inference.py -> commit::HEAD
- NFR-0001 -> run_inference.py::build_llm -> python_compile::run_inference.py -> commit::HEAD
- NFR-0002 -> run_inference.py::generate_budget_sweep_reuse_longest -> python_compile::run_inference.py -> commit::HEAD

## Change Log
- 2026-07-01 - Initial offline vLLM reachability inference runner.
- 2026-07-01 - Added deterministic reuse-longest budget sweep to reduce repeated long-context GPU prefill.
