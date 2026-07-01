#!/usr/bin/env python3
"""Start a vLLM OpenAI-compatible server and run basic dataset examples."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RE_CONTEXT_DIR = re.compile(r"^context-(\d+)_")
logger = logging.getLogger("vllm_server_basic")


@dataclass(frozen=True)
class Example:
    seq_id: int
    distance: int
    question: str


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


def parse_question_line(seq_id: int, line: str) -> Example:
    match = re.match(r"^\s*(-?\d+)\s+(.*)$", line.strip())
    if not match:
        return Example(seq_id=seq_id, distance=0, question=line.strip())
    question = re.sub(r"(?:\s+-?\d+)+\s*$", "", match.group(2).strip()).strip()
    return Example(seq_id=seq_id, distance=int(match.group(1)), question=question)


def context_value(path: pathlib.Path) -> int | None:
    match = RE_CONTEXT_DIR.match(path.name)
    return int(match.group(1)) if match else None


def find_experiment_dir(work_dir: pathlib.Path, context: int | None) -> pathlib.Path:
    if (work_dir / "system.txt").is_file() and (work_dir / "reachability_questions.txt").is_file():
        return work_dir

    context_dirs = sorted(
        path
        for path in work_dir.iterdir()
        if path.is_dir()
        and (value := context_value(path)) is not None
        and (context is None or value == context)
    )
    for context_dir in context_dirs:
        for child in sorted(path for path in context_dir.iterdir() if path.is_dir()):
            if (child / "system.txt").is_file() and (child / "reachability_questions.txt").is_file():
                return child

    suffix = f" for context={context}" if context is not None else ""
    raise RuntimeError(f"No experiment directory with system.txt and reachability_questions.txt found in {work_dir}{suffix}.")


def load_examples(experiment_dir: pathlib.Path, limit: int) -> tuple[str, list[Example]]:
    system_prompt = read_text(experiment_dir / "system.txt")
    lines = [
        line
        for line in read_text(experiment_dir / "reachability_questions.txt").splitlines()
        if line.strip()
    ]
    examples = [parse_question_line(seq_id, line) for seq_id, line in enumerate(lines[:limit])]
    if not examples:
        raise RuntimeError(f"No examples found in {experiment_dir / 'reachability_questions.txt'}.")
    return system_prompt, examples


def build_vllm_command(args) -> list[str]:
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--generation-config",
        "vllm",
    ]

    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    for item in args.vllm_arg or []:
        cmd.extend(item.split("=", 1) if "=" in item else [item])
    return cmd


def start_server(args) -> subprocess.Popen:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(args.log_file, "a", encoding="utf-8")
    cmd = build_vllm_command(args)
    logger.info("Starting vLLM server:")
    logger.info("  %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
    )
    proc._vllm_log_handle = log_handle  # type: ignore[attr-defined]
    logger.info("vLLM PID=%s log=%s", proc.pid, args.log_file)
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("Stopping vLLM server PID=%s", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not stop after SIGTERM; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=30)
    finally:
        log_handle = getattr(proc, "_vllm_log_handle", None)
        if log_handle is not None:
            log_handle.close()


def request_json(url: str, api_key: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    method = "GET"
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, api_key: str, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}.")
        try:
            request_json(f"{base_url}/models", api_key, timeout=10)
            logger.info("vLLM server is ready")
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"vLLM server did not become ready within {timeout_s}s. Last error: {last_error}")


def run_one_chat(
    *,
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    example: Example,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = request_json(
        f"{base_url}/chat/completions",
        api_key,
        payload=payload,
        timeout=timeout,
    )
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args) -> None:
    experiment_dir = find_experiment_dir(args.work_dir, args.context)
    system_prompt, examples = load_examples(experiment_dir, args.limit)
    logger.info("Using examples from %s", experiment_dir)

    base_url = f"http://{args.host}:{args.port}/v1"
    proc: subprocess.Popen | None = None
    try:
        proc = start_server(args)
        wait_until_ready(base_url, args.api_key, proc, args.server_timeout)

        rows: list[dict[str, Any]] = []
        for example in examples:
            started = time.perf_counter()
            answer = run_one_chat(
                base_url=base_url,
                model=args.served_model_name,
                api_key=args.api_key,
                system_prompt=system_prompt,
                example=example,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.request_timeout,
            )
            elapsed = time.perf_counter() - started
            row = {
                "seq_id": example.seq_id,
                "distance": example.distance,
                "question": example.question,
                "answer": answer,
                "latency_s": round(elapsed, 3),
            }
            rows.append(row)
            print(
                f"\n=== seq={example.seq_id}, distance={example.distance}, latency={elapsed:.3f}s ===\n"
                f"Q: {example.question}\n"
                f"A: {answer}\n",
                flush=True,
            )

        if args.output_jsonl:
            write_jsonl(args.output_jsonl, rows)
            logger.info("Wrote %s", args.output_jsonl)
    finally:
        if not args.keep_server:
            stop_server(proc)
        elif proc is not None:
            logger.info("Keeping vLLM server running at %s with PID=%s", base_url, proc.pid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="local-model")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output-jsonl", type=pathlib.Path, default=None)

    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--log-file", type=pathlib.Path, default=pathlib.Path("outputs/vllm_server_basic.log"))

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        help="Extra vLLM CLI argument. Repeat as needed, e.g. --vllm-arg=--max-num-seqs=1.",
    )

    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    setup_logging(parsed_args.debug)
    try:
        main(parsed_args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)
