#!/usr/bin/env python3
"""Offline vLLM runner for reachability reasoning-budget sweeps."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams


RE_CONTEXT_DIR = re.compile(r"^context-(\d+)_")
RE_RESULT_BLOCK = re.compile(
    r"^=== seq=(?P<seq_id>-?\d+), distance=(?P<distance>-?\d+), reasoning_budget=(?P<reasoning_budget>-?\d+) ===\n"
    r"Q : (?P<question>.*?)\n"
    r"R : (?P<reasoning>.*?)\n"
    r"A : (?P<answer>.*?)(?=\n=== seq=|\Z)",
    re.MULTILINE | re.DOTALL,
)

MODEL_MAX_LEN_KEYS = {
    "max_position_embeddings",
    "n_positions",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_sequence_length",
    "max_seq_length",
    "context_length",
    "max_context_length",
    "sliding_window",
    "original_max_position_embeddings",
}

MAX_LEN_SENTINEL = 10**12
DEFAULT_REASONING_BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_SAFETY_MARGIN_TOKENS = 32

logger = logging.getLogger("inference_vllm")


@dataclass(frozen=True)
class GPUInfo:
    names: tuple[str, ...]
    total_memory_gb: float
    free_memory_gb: float | None
    count: int
    profile: str

    @property
    def label(self) -> str:
        return ", ".join(sorted(set(self.names))) if self.names else "unknown"


@dataclass(frozen=True)
class PromptTask:
    seq_id: int
    distance: int
    question: str
    messages: list[dict[str, str]]
    prompt_text: str
    prompt_tokens: int


@dataclass(frozen=True)
class RawExperiment:
    path: pathlib.Path
    system_prompt: str
    question_lines: tuple[str, ...]


@dataclass(frozen=True)
class BudgetTask:
    prompt: PromptTask
    reasoning_budget: int

    @property
    def work_tokens(self) -> int:
        return self.prompt.prompt_tokens + self.reasoning_budget


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Experiment:
    path: pathlib.Path
    tasks: list[PromptTask]

    @property
    def max_prompt_tokens(self) -> int:
        return max((task.prompt_tokens for task in self.tasks), default=0)


@dataclass(frozen=True)
class PromptStats:
    count: int
    p50: int
    p90: int
    p95: int
    p99: int
    max_prompt: int


@dataclass(frozen=True)
class RuntimeSettings:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    max_cudagraph_capture_size: int | None


def setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def read_text(path: pathlib.Path) -> str:
    for encoding in ("utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_question_line(line: str) -> tuple[int, str]:
    match = re.match(r"^\s*(-?\d+)\s+(.*)$", line.strip())
    if not match:
        return 0, line.strip()
    distance = int(match.group(1))
    question = re.sub(r"(?:\s+-?\d+)+\s*$", "", match.group(2).strip()).strip()
    return distance, question


def extract_context_from_dirname(dirname: str) -> int | None:
    match = RE_CONTEXT_DIR.match(dirname)
    return int(match.group(1)) if match else None


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return int(round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction))


def prompt_stats(values: list[int]) -> PromptStats:
    return PromptStats(
        count=len(values),
        p50=percentile(values, 0.50),
        p90=percentile(values, 0.90),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        max_prompt=max(values, default=0),
    )


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in re.split(r"[,\s]+", raw.strip()):
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("Reasoning budgets must be positive integers.")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("At least one reasoning budget is required.")
    return values


def bool_auto(value: str, *, auto: bool) -> bool:
    normalized = value.lower()
    if normalized == "auto":
        return auto
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise ValueError(f"Unsupported boolean mode: {value}")


def sanitize_model_len(value: Any) -> int | None:
    if not isinstance(value, int) or value <= 0 or value >= MAX_LEN_SENTINEL:
        return None
    return value


def collect_context_candidates(node: Any) -> list[int]:
    values: list[int] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in MODEL_MAX_LEN_KEYS:
                sanitized = sanitize_model_len(value)
                if sanitized is not None:
                    values.append(sanitized)
            values.extend(collect_context_candidates(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(collect_context_candidates(value))
    return values


def infer_model_context(model: str, trust_remote_code: bool) -> int | None:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception as exc:
        logger.warning("Could not inspect model config for %s: %s", model, exc)
        return None
    candidates = collect_context_candidates(config.to_dict())
    return max(candidates) if candidates else None


def load_tokenizer(args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )


def visible_cuda_device_count(default: int) -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if not visible or visible.strip() in {"", "-1"}:
        return default
    return len([part for part in visible.split(",") if part.strip()])


def gpu_profile(name: str) -> str:
    lowered = name.lower()
    if "b200" in lowered or "gb200" in lowered:
        return "blackwell"
    if "h200" in lowered:
        return "h200"
    if "h100" in lowered:
        return "h100"
    if "a100" in lowered:
        return "a100"
    if "l40" in lowered or "l4" in lowered:
        return "l40_l4"
    if "v100" in lowered:
        return "v100"
    return "generic"


def detect_gpu_info() -> GPUInfo:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Could not query nvidia-smi; using conservative defaults: %s", exc)
        count = visible_cuda_device_count(1)
        return GPUInfo((), 80.0, None, max(1, count), "generic")

    names: list[str] = []
    total: list[float] = []
    free: list[float] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        names.append(parts[0])
        try:
            total.append(float(parts[1]) / 1024.0)
            free.append(float(parts[2]) / 1024.0)
        except ValueError:
            continue

    visible = visible_cuda_device_count(len(names) or 1)
    names = names[:visible] or names
    total = total[:visible] or total
    free = free[:visible] or free
    profiles = [gpu_profile(name) for name in names]
    priority = {"generic": 0, "v100": 1, "l40_l4": 2, "a100": 3, "h100": 4, "h200": 5, "blackwell": 6}
    profile = min(profiles, key=lambda item: priority.get(item, 0)) if profiles else "generic"
    return GPUInfo(tuple(names), min(total) if total else 80.0, min(free) if free else None, max(1, visible), profile)


def model_family(model: str) -> str:
    lowered = model.lower()
    if "qwen3" in lowered:
        return "qwen3"
    if "deepseek-v3.1" in lowered:
        return "deepseek_v3_1"
    if "deepseek" in lowered:
        return "deepseek"
    if "gpt-oss" in lowered:
        return "gpt_oss"
    if "gemma-4" in lowered or "gemma4" in lowered:
        return "gemma4"
    if "nemotron" in lowered:
        return "nemotron"
    if "glm-4.5" in lowered or "glm45" in lowered:
        return "glm45"
    if "granite" in lowered:
        return "granite"
    return "generic"


def thinking_template_kwargs(model: str, enable_thinking: bool) -> dict[str, Any]:
    family = model_family(model)
    if enable_thinking:
        if family in {"gemma4", "qwen3", "nemotron", "glm45"}:
            return {"enable_thinking": True}
        if family in {"deepseek_v3_1", "granite"}:
            return {"thinking": True}
        return {}
    if family in {"gemma4", "qwen3", "nemotron", "glm45"}:
        return {"enable_thinking": False}
    if family in {"deepseek_v3_1", "granite"}:
        return {"thinking": False}
    return {}


def build_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]


def render_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    model: str,
    enable_thinking: bool,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> str:
    kwargs = thinking_template_kwargs(model, enable_thinking)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        **kwargs,
    )


def count_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def load_raw_experiment(path: pathlib.Path) -> RawExperiment | None:
    system_path = path / "system.txt"
    questions_path = path / "reachability_questions.txt"
    if not system_path.is_file() or not questions_path.is_file():
        return None

    system_prompt = read_text(system_path)
    lines = [line for line in read_text(questions_path).splitlines() if line.strip()]
    if not lines:
        return None
    return RawExperiment(
        path=path,
        system_prompt=system_prompt,
        question_lines=tuple(lines),
    )


def materialize_experiment(raw: RawExperiment, tokenizer, args) -> Experiment | None:
    tasks: list[PromptTask] = []
    for seq_id, line in enumerate(raw.question_lines):
        distance, question = parse_question_line(line)
        messages = build_messages(raw.system_prompt, question)
        prompt_text = render_chat_prompt(
            tokenizer,
            messages,
            model=args.model,
            enable_thinking=args.enable_thinking,
            add_generation_prompt=True,
        )
        tasks.append(
            PromptTask(
                seq_id=seq_id,
                distance=distance,
                question=question,
                messages=messages,
                prompt_text=prompt_text,
                prompt_tokens=count_tokens(tokenizer, prompt_text),
            )
        )
    if not tasks:
        return None
    return Experiment(path=raw.path, tasks=tasks)


def iter_experiment_dirs(work_dir: pathlib.Path, contexts: set[int]) -> list[pathlib.Path]:
    context_dirs = sorted(
        path
        for path in work_dir.iterdir()
        if path.is_dir()
        and (context := extract_context_from_dirname(path.name)) is not None
        and (not contexts or context in contexts)
    )
    if not context_dirs:
        raise RuntimeError(f"No context-* experiment directory found in {work_dir}")
    return [
        child
        for context_dir in context_dirs
        for child in sorted(path for path in context_dir.iterdir() if path.is_dir())
    ]


def load_raw_experiments(paths: list[pathlib.Path], workers: int) -> list[RawExperiment]:
    if workers <= 1:
        raw = [load_raw_experiment(path) for path in tqdm(paths, desc="Loading data", unit="exp")]
        return [item for item in raw if item is not None]

    experiments: list[RawExperiment] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_raw_experiment, path) for path in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data", unit="exp"):
            item = future.result()
            if item is not None:
                experiments.append(item)
    experiments.sort(key=lambda item: item.path.name)
    return experiments


def materialize_experiments(raw_experiments: list[RawExperiment], tokenizer, args) -> list[Experiment]:
    workers = max(1, args.tokenize_workers)
    if workers <= 1:
        experiments = [
            materialize_experiment(raw, tokenizer, args)
            for raw in tqdm(raw_experiments, desc="Tokenizing prompts", unit="exp")
        ]
    else:
        experiments = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(materialize_experiment, raw, tokenizer, args)
                for raw in raw_experiments
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Tokenizing prompts",
                unit="exp",
            ):
                experiments.append(future.result())

    ready = [item for item in experiments if item is not None]
    if not ready:
        raise RuntimeError("No non-empty experiments were found.")
    ready.sort(key=lambda item: item.max_prompt_tokens, reverse=True)
    return ready


def prepare_experiments(tokenizer, args, raw_experiments: list[RawExperiment] | None = None) -> list[Experiment]:
    contexts = set(args.context or [])
    if raw_experiments is None:
        paths = iter_experiment_dirs(args.work_dir, contexts)
        raw_experiments = load_raw_experiments(paths, max(1, args.data_workers))
    return materialize_experiments(raw_experiments, tokenizer, args)


def resolve_runtime_settings(
    args,
    stats: PromptStats,
    gpu: GPUInfo,
    model_context: int | None = None,
) -> RuntimeSettings:
    max_budget = max(args.reasoning_budgets)
    output_room = max_budget + args.answer_max_tokens + args.safety_margin_tokens
    needed_len = round_up_to_multiple(stats.max_prompt + output_room, 1024 if stats.max_prompt < 65_536 else 4096)
    if model_context is None:
        model_context = infer_model_context(args.model, args.trust_remote_code)

    requested = args.max_model_len.lower()
    if requested in {"needed", "fit", "prompt"}:
        max_model_len = min(needed_len, model_context) if model_context else needed_len
    elif requested == "model":
        max_model_len = model_context or needed_len
    else:
        max_model_len = int(args.max_model_len)

    if max_model_len <= 0:
        raise ValueError("--max-model-len must resolve to a positive integer.")

    enable_prefix_caching = bool_auto(args.prefix_caching, auto=True)
    enable_chunked_prefill = bool_auto(args.chunked_prefill, auto=stats.p95 >= 2048)

    if args.max_num_seqs:
        max_num_seqs = args.max_num_seqs
    elif stats.p95 >= 128_000:
        max_num_seqs = 1
    elif stats.p95 >= 64_000:
        max_num_seqs = 2
    elif stats.p95 >= 16_000:
        max_num_seqs = 4 if args.optimization_profile != "latency" else 2
    else:
        max_num_seqs = 16 if args.optimization_profile != "latency" else 4

    if args.max_num_batched_tokens:
        max_num_batched_tokens = args.max_num_batched_tokens
    elif stats.p95 >= 128_000:
        max_num_batched_tokens = 65_536
    elif stats.p95 >= 16_000:
        max_num_batched_tokens = 16_384
    else:
        max_num_batched_tokens = 8192
    if args.optimization_profile == "latency" and stats.p95 < 128_000:
        max_num_batched_tokens = min(max_num_batched_tokens, 16_384)

    if args.gpu_memory_utilization is not None:
        gpu_memory_utilization = args.gpu_memory_utilization
    elif gpu.total_memory_gb >= 80:
        gpu_memory_utilization = 0.92 if args.optimization_profile == "latency" else 0.95
    elif gpu.total_memory_gb >= 48:
        gpu_memory_utilization = 0.90
    else:
        gpu_memory_utilization = 0.86

    if args.max_cudagraph_capture_size is not None:
        capture_size = args.max_cudagraph_capture_size
    elif stats.p95 >= 65_536:
        capture_size = 512
    elif stats.p95 >= 16_384:
        capture_size = 1024
    else:
        capture_size = 2048

    return RuntimeSettings(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=round(gpu_memory_utilization, 3),
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_cudagraph_capture_size=capture_size,
    )


def startup_worker_count(requested: int, default_cap: int = 8) -> int:
    if requested > 0:
        return requested
    return max(1, min(os.cpu_count() or 1, default_cap))


def is_int_string(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def can_overlap_model_loading(args) -> bool:
    if args.startup_overlap_model == "off":
        return False
    if args.startup_overlap_model == "on":
        return True
    return (
        is_int_string(args.max_model_len)
        and args.max_num_seqs > 0
        and args.max_num_batched_tokens > 0
        and args.gpu_memory_utilization is not None
        and args.prefix_caching != "auto"
        and args.chunked_prefill != "auto"
        and args.max_cudagraph_capture_size is not None
    )


def resolve_overlap_runtime_settings(args) -> RuntimeSettings:
    if not is_int_string(args.max_model_len):
        raise ValueError(
            "--startup-overlap-model requires an integer --max-model-len so vLLM can start before prompt stats are known."
        )
    if args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError(
            "--startup-overlap-model requires explicit --max-num-seqs and --max-num-batched-tokens."
        )
    if args.gpu_memory_utilization is None:
        raise ValueError("--startup-overlap-model requires explicit --gpu-memory-utilization.")
    if args.prefix_caching == "auto" or args.chunked_prefill == "auto":
        raise ValueError(
            "--startup-overlap-model requires --prefix-caching on/off and --chunked-prefill on/off."
        )
    if args.max_cudagraph_capture_size is None:
        raise ValueError("--startup-overlap-model requires explicit --max-cudagraph-capture-size.")

    return RuntimeSettings(
        max_model_len=int(args.max_model_len),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=round(args.gpu_memory_utilization, 3),
        enable_prefix_caching=bool_auto(args.prefix_caching, auto=True),
        enable_chunked_prefill=bool_auto(args.chunked_prefill, auto=True),
        max_cudagraph_capture_size=args.max_cudagraph_capture_size,
    )


def build_llm(args, settings: RuntimeSettings) -> LLM:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dtype": args.dtype,
        "max_model_len": settings.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": settings.gpu_memory_utilization,
        "enable_prefix_caching": settings.enable_prefix_caching,
        "enable_chunked_prefill": settings.enable_chunked_prefill,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "max_num_seqs": settings.max_num_seqs,
        "generation_config": "vllm",
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.tokenizer_mode:
        kwargs["tokenizer_mode"] = args.tokenizer_mode
    if settings.max_cudagraph_capture_size is not None and settings.max_cudagraph_capture_size > 0:
        kwargs["max_cudagraph_capture_size"] = settings.max_cudagraph_capture_size

    logger.info("Initializing vLLM once with: %s", json.dumps({k: str(v) for k, v in kwargs.items()}, sort_keys=True))
    started = time.perf_counter()
    llm = LLM(**kwargs)
    logger.info("vLLM initialized in %.2fs", time.perf_counter() - started)
    return llm


def clean_reasoning(text: str) -> str:
    return (text or "").replace("<think>", "").replace("</think>", "").strip()


def reasoning_message(reasoning: str) -> dict[str, str]:
    return {"role": "assistant", "content": f"<think>\n{clean_reasoning(reasoning)}\n</think>\n\n"}


def validate_task_fits_context(task: BudgetTask, args, settings: RuntimeSettings) -> None:
    required = task.prompt.prompt_tokens + task.reasoning_budget + args.answer_max_tokens + args.safety_margin_tokens
    if required > settings.max_model_len:
        raise ValueError(
            f"Prompt too long for configured context: seq={task.prompt.seq_id}, "
            f"required={required}, max_model_len={settings.max_model_len}"
        )


def sampling(max_tokens: int, args) -> SamplingParams:
    return SamplingParams(
        max_tokens=max(1, int(max_tokens)),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        stop=args.stop or None,
    )


def generate_outputs(llm: LLM, prompts: list[str], params: SamplingParams) -> list[GeneratedText]:
    outputs = llm.generate(prompts, params, use_tqdm=False)
    generated: list[GeneratedText] = []
    for output in outputs:
        if not output.outputs:
            generated.append(GeneratedText("", tuple()))
            continue
        completion = output.outputs[0]
        token_ids = tuple(getattr(completion, "token_ids", ()) or ())
        generated.append(GeneratedText((completion.text or "").strip(), token_ids))
    return generated


def generate_texts(llm: LLM, prompts: list[str], params: SamplingParams) -> list[str]:
    return [output.text for output in generate_outputs(llm, prompts, params)]


def decode_token_prefix(
    tokenizer,
    token_ids: tuple[int, ...],
    max_tokens: int,
    fallback_text: str,
) -> str:
    if token_ids:
        return tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
    return fallback_text.strip()


def generate_reasoning_then_answer(
    *,
    llm: LLM,
    tokenizer,
    tasks: list[BudgetTask],
    args,
    settings: RuntimeSettings,
) -> list[tuple[BudgetTask, str, str]]:
    for task in tasks:
        validate_task_fits_context(task, args, settings)

    reasoning_prompts = [task.prompt.prompt_text for task in tasks]
    reasonings = generate_texts(llm, reasoning_prompts, sampling(tasks[0].reasoning_budget, args))

    answer_prompts: list[str] = []
    for task, reasoning in zip(tasks, reasonings, strict=True):
        messages = list(task.prompt.messages)
        messages.append(reasoning_message(reasoning))
        answer_prompts.append(
            render_chat_prompt(
                tokenizer,
                messages,
                model=args.model,
                enable_thinking=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        )

    answers = generate_texts(llm, answer_prompts, sampling(args.answer_max_tokens, args))
    return [
        (task, clean_reasoning(reasoning), answer.strip())
        for task, reasoning, answer in zip(tasks, reasonings, answers, strict=True)
    ]


def build_answer_prompt(tokenizer, prompt: PromptTask, reasoning: str, args) -> str:
    messages = list(prompt.messages)
    messages.append(reasoning_message(reasoning))
    return render_chat_prompt(
        tokenizer,
        messages,
        model=args.model,
        enable_thinking=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )


def generate_budget_sweep_reuse_longest(
    *,
    llm: LLM,
    tokenizer,
    prompt: PromptTask,
    budgets: list[int],
    args,
    settings: RuntimeSettings,
) -> list[tuple[int, str, str]]:
    max_budget = max(budgets)
    validate_task_fits_context(BudgetTask(prompt, max_budget), args, settings)

    max_reasoning = generate_outputs(
        llm,
        [prompt.prompt_text],
        sampling(max_budget, args),
    )[0]

    reasoning_by_budget = {
        budget: clean_reasoning(
            decode_token_prefix(tokenizer, max_reasoning.token_ids, budget, max_reasoning.text)
        )
        for budget in budgets
    }

    answer_jobs = [
        (
            budget,
            build_answer_prompt(tokenizer, prompt, reasoning_by_budget[budget], args),
        )
        for budget in sorted(budgets, reverse=True)
    ]
    if args.answer_generation_mode == "sequential":
        answers = [
            generate_outputs(llm, [answer_prompt], sampling(args.answer_max_tokens, args))[0]
            for _, answer_prompt in answer_jobs
        ]
    else:
        answers = generate_outputs(
            llm,
            [answer_prompt for _, answer_prompt in answer_jobs],
            sampling(args.answer_max_tokens, args),
        )
    answer_by_budget = {
        budget: answer.text.strip()
        for (budget, _), answer in zip(answer_jobs, answers, strict=True)
    }

    return [
        (budget, reasoning_by_budget[budget], answer_by_budget.get(budget, ""))
        for budget in budgets
    ]


def write_results(output_dir: pathlib.Path, results: list[tuple[int, int, int, str, str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for seq_id, distance, budget, question, reasoning, answer in results:
        reasoning = reasoning.strip() or "[NO_REASONING_RETURNED]"
        answer = answer.strip() or "[NO_ANSWER_RETURNED]"
        lines.append(
            f"=== seq={seq_id}, distance={distance}, reasoning_budget={budget} ===\n"
            f"Q : {question}\n"
            f"R : {reasoning}\n"
            f"A : {answer}\n\n"
        )
    (output_dir / "all_results.txt").write_text("".join(lines), encoding="utf-8")


def load_existing_results(output_dir: pathlib.Path) -> dict[tuple[int, int], tuple[int, int, int, str, str, str]]:
    path = output_dir / "all_results.txt"
    if not path.is_file():
        return {}
    text = read_text(path)
    results: dict[tuple[int, int], tuple[int, int, int, str, str, str]] = {}
    for match in RE_RESULT_BLOCK.finditer(text):
        seq_id = int(match.group("seq_id"))
        budget = int(match.group("reasoning_budget"))
        results[(seq_id, budget)] = (
            seq_id,
            int(match.group("distance")),
            budget,
            match.group("question").rstrip("\n"),
            match.group("reasoning").rstrip("\n"),
            match.group("answer").rstrip("\n"),
        )
    return results


def ordered_budget_tasks(experiment: Experiment, budgets: list[int], mode: str) -> list[BudgetTask]:
    tasks = [BudgetTask(prompt, budget) for prompt in experiment.tasks for budget in budgets]
    if mode == "off":
        return tasks
    if mode == "longest_first":
        return sorted(tasks, key=lambda task: (task.prompt.prompt_tokens, task.reasoning_budget), reverse=True)
    if mode == "largest_work_first":
        return sorted(tasks, key=lambda task: (task.work_tokens, task.prompt.prompt_tokens), reverse=True)
    raise ValueError(f"Unsupported schedule mode: {mode}")


def iter_budget_batches(tasks: list[BudgetTask], batch_size: int) -> list[list[BudgetTask]]:
    if batch_size <= 1:
        return [[task] for task in tasks]

    batches: list[list[BudgetTask]] = []
    current: list[BudgetTask] = []
    current_budget: int | None = None
    for task in tasks:
        if (
            current
            and (task.reasoning_budget != current_budget or len(current) >= batch_size)
        ):
            batches.append(current)
            current = []
        current.append(task)
        current_budget = task.reasoning_budget
    if current:
        batches.append(current)
    return batches


def run_experiment(llm: LLM, tokenizer, experiment: Experiment, args, settings: RuntimeSettings, global_pbar: tqdm) -> None:
    output_dir = args.output_root / experiment.path.name
    existing = load_existing_results(output_dir) if args.resume else {}
    results_by_key = dict(existing)

    if args.budget_sweep_mode == "reuse_longest":
        pending_prompts: list[tuple[PromptTask, list[int]]] = []
        for prompt in experiment.tasks:
            missing_budgets = [
                budget
                for budget in args.reasoning_budgets
                if (prompt.seq_id, budget) not in existing
            ]
            if missing_budgets:
                pending_prompts.append((prompt, missing_budgets))

        pending_prompts.sort(key=lambda item: item[0].prompt_tokens, reverse=True)
        if not pending_prompts:
            write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))
            return

        pending_runs = sum(len(budgets) for _, budgets in pending_prompts)
        logger.info(
            "Running %s: %s pending runs via reuse_longest over %s prompts",
            experiment.path.name,
            pending_runs,
            len(pending_prompts),
        )
        for prompt, budgets in pending_prompts:
            try:
                sweep_results = generate_budget_sweep_reuse_longest(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    budgets=budgets,
                    args=args,
                    settings=settings,
                )
            except Exception:
                logger.exception(
                    "Prompt sweep failed in %s seq=%s; recording request errors",
                    experiment.path.name,
                    prompt.seq_id,
                )
                sweep_results = [
                    (budget, "[REQUEST_ERROR] sweep failed", "[REQUEST_ERROR] sweep failed")
                    for budget in budgets
                ]

            for budget, reasoning, answer in sweep_results:
                results_by_key[(prompt.seq_id, budget)] = (
                    prompt.seq_id,
                    prompt.distance,
                    budget,
                    prompt.question,
                    reasoning,
                    answer,
                )
                global_pbar.update(1)

            if args.autosave_every and len(results_by_key) % args.autosave_every < len(budgets):
                write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))

        write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))
        return

    tasks = [
        task
        for task in ordered_budget_tasks(experiment, args.reasoning_budgets, args.schedule)
        if (task.prompt.seq_id, task.reasoning_budget) not in existing
    ]
    if not tasks:
        write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))
        return

    logger.info("Running %s: %s pending runs", experiment.path.name, len(tasks))
    for batch in iter_budget_batches(tasks, args.batch_size):
        try:
            batch_results = generate_reasoning_then_answer(
                llm=llm,
                tokenizer=tokenizer,
                tasks=batch,
                args=args,
                settings=settings,
            )
        except Exception:
            logger.exception("Batch failed in %s; recording request errors", experiment.path.name)
            batch_results = [(task, "[REQUEST_ERROR] batch failed", "[REQUEST_ERROR] batch failed") for task in batch]

        for task, reasoning, answer in batch_results:
            prompt = task.prompt
            results_by_key[(prompt.seq_id, task.reasoning_budget)] = (
                prompt.seq_id,
                prompt.distance,
                task.reasoning_budget,
                prompt.question,
                reasoning,
                answer,
            )
            global_pbar.update(1)

        if args.autosave_every and len(results_by_key) % args.autosave_every < len(batch):
            write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))

    write_results(output_dir, ordered_results(experiment, args.reasoning_budgets, results_by_key))


def ordered_results(
    experiment: Experiment,
    budgets: list[int],
    results: dict[tuple[int, int], tuple[int, int, int, str, str, str]],
) -> list[tuple[int, int, int, str, str, str]]:
    ordered: list[tuple[int, int, int, str, str, str]] = []
    for prompt in experiment.tasks:
        for budget in budgets:
            item = results.get((prompt.seq_id, budget))
            if item is not None:
                ordered.append(item)
    return ordered


def main(args) -> None:
    random.seed(args.seed)
    args.reasoning_budgets = parse_int_list(args.reasoning_budgets)
    if args.budget_sweep_mode == "reuse_longest" and args.temperature != 0.0:
        raise ValueError(
            "--budget-sweep-mode reuse_longest requires --temperature 0.0 for exact deterministic prefix reuse."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    args.data_workers = startup_worker_count(args.data_workers)
    args.tokenize_workers = startup_worker_count(args.tokenize_workers)
    contexts = set(args.context or [])
    experiment_paths = iter_experiment_dirs(args.work_dir, contexts)

    startup_started = time.perf_counter()
    llm_future = None
    overlap_settings: RuntimeSettings | None = None
    with ThreadPoolExecutor(max_workers=max(4, args.startup_workers)) as executor:
        raw_future = executor.submit(load_raw_experiments, experiment_paths, args.data_workers)
        tokenizer_future = executor.submit(load_tokenizer, args)
        gpu_future = executor.submit(detect_gpu_info)
        model_context_future = executor.submit(
            infer_model_context,
            args.model,
            args.trust_remote_code,
        )

        if can_overlap_model_loading(args):
            overlap_settings = resolve_overlap_runtime_settings(args)
            logger.info(
                "Starting vLLM model loading in parallel with data preparation. "
                "Settings must be explicit in this mode: %s",
                overlap_settings,
            )
            llm_future = executor.submit(build_llm, args, overlap_settings)
        else:
            logger.info(
                "Model loading will start after prompt stats are known. "
                "Use --startup-overlap-model on with explicit vLLM settings to overlap it."
            )

        tokenizer_started = time.perf_counter()
        tokenizer = tokenizer_future.result()
        logger.info("Tokenizer loaded in %.2fs", time.perf_counter() - tokenizer_started)

        raw_started = time.perf_counter()
        raw_experiments = raw_future.result()
        logger.info(
            "Loaded %s raw experiments in %.2fs with %s data workers",
            len(raw_experiments),
            time.perf_counter() - raw_started,
            args.data_workers,
        )

        prep_started = time.perf_counter()
        experiments = prepare_experiments(tokenizer, args, raw_experiments)
        model_context = model_context_future.result()
        gpu = gpu_future.result()

    all_prompt_lengths = [task.prompt_tokens for experiment in experiments for task in experiment.tasks]
    stats = prompt_stats(all_prompt_lengths)
    logger.info(
        "Prepared %s experiments / %s prompts in %.2fs | p50=%s p95=%s p99=%s max=%s",
        len(experiments),
        len(all_prompt_lengths),
        time.perf_counter() - prep_started,
        stats.p50,
        stats.p95,
        stats.p99,
        stats.max_prompt,
    )
    logger.info(
        "GPU=%s profile=%s count=%s min_total=%.1fGB",
        gpu.label,
        gpu.profile,
        gpu.count,
        gpu.total_memory_gb,
    )

    if overlap_settings is not None:
        settings = overlap_settings
        required_len = stats.max_prompt + max(args.reasoning_budgets) + args.answer_max_tokens + args.safety_margin_tokens
        if required_len > settings.max_model_len:
            raise ValueError(
                f"Configured --max-model-len={settings.max_model_len} is too small for observed prompts; "
                f"required at least {required_len}."
            )
    else:
        settings = resolve_runtime_settings(args, stats, gpu, model_context=model_context)
    logger.info("Runtime settings: %s", settings)
    if llm_future is not None:
        llm = llm_future.result()
    else:
        llm = build_llm(args, settings)
    logger.info("Startup completed in %.2fs", time.perf_counter() - startup_started)

    total_runs = sum(len(exp.tasks) for exp in experiments) * len(args.reasoning_budgets)
    with tqdm(total=total_runs, desc="All runs", unit="run") as pbar:
        for experiment in experiments:
            run_experiment(llm, tokenizer, experiment, args, settings, pbar)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--context", type=int, nargs="*", default=None)
    parser.add_argument("--reasoning-budgets", default=",".join(str(v) for v in DEFAULT_REASONING_BUDGETS))
    parser.add_argument(
        "--budget-sweep-mode",
        choices=["independent", "reuse_longest"],
        default="reuse_longest",
        help=(
            "reuse_longest generates reasoning once at the largest budget, then token-truncates "
            "that deterministic generation for smaller budgets."
        ),
    )
    parser.add_argument(
        "--answer-generation-mode",
        choices=["sequential", "batched"],
        default="sequential",
        help=(
            "How reuse_longest generates final answers. Sequential preserves deterministic "
            "stability best and still benefits from prefix cache; batched can be faster."
        ),
    )
    parser.add_argument("--answer-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"], default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", default="needed", help="'needed', 'model', or an integer")
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--prefix-caching", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--chunked-prefill", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
    parser.add_argument("--tokenizer-mode", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--optimization-profile", choices=["throughput", "latency"], default="latency")
    parser.add_argument("--schedule", choices=["off", "longest_first", "largest_work_first"], default="largest_work_first")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="Workers for parallel system/question file loading. 0 auto-tunes from CPU count.",
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=0,
        help="Workers for prompt rendering and token counting. 0 auto-tunes from CPU count.",
    )
    parser.add_argument(
        "--startup-workers",
        type=int,
        default=4,
        help="Thread pool size for startup tasks such as data loading, tokenizer loading, GPU probing, and config probing.",
    )
    parser.add_argument(
        "--startup-overlap-model",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Start vLLM model loading while data/tokenization work is still running. Auto only "
            "does this when all vLLM sizing options are explicit; on requires explicit settings."
        ),
    )
    parser.add_argument("--autosave-every", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--safety-margin-tokens", type=int, default=DEFAULT_SAFETY_MARGIN_TOKENS)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    setup_logging(parsed_args.debug)
    main(parsed_args)
