from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from together import AsyncTogether


QUESTION_FILE = Path(__file__).with_name("questions.jsonl")
RESULT_DIR = Path(__file__).with_name("results")

REFERENCE_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3.7-Max",
    "moonshotai/Kimi-K2.7-Code",
    "deepseek-ai/DeepSeek-V4-Pro",
]

AGGREGATOR_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

AGGREGATOR_SYSTEM_PROMPT = """You have been provided with a set of responses
from various models to the latest user query. Synthesize them into one
high-quality response. Critically evaluate the supplied information because
some responses may be incomplete, biased, or incorrect. Produce a refined,
accurate, comprehensive, and well-structured answer.

Responses from models:"""


def load_question(question_id: str) -> dict[str, Any]:
    for line in QUESTION_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        item = json.loads(line)

        if item["id"] == question_id:
            return item

    raise ValueError(f"没有找到题目：{question_id}")


def format_previous_responses(
    results: list[dict[str, Any]],
) -> str:
    return "\n".join(
        f"{index + 1}. {result['content']}"
        for index, result in enumerate(results)
    )


def build_messages(
    prompt: str,
    previous_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    if not previous_results:
        return [{"role": "user", "content": prompt}]

    return [
        {
            "role": "system",
            "content": (
                AGGREGATOR_SYSTEM_PROMPT
                + "\n"
                + format_previous_responses(previous_results)
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


async def collect_stream_response(
    client: AsyncTogether,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    print_stream: bool = False,
) -> dict[str, Any]:
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=max_tokens,
        stream=True,
    )

    content_parts: list[str] = []
    reasoning_chars = 0
    finish_reason = None

    async for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)

        content = getattr(delta, "content", None) if delta else None
        reasoning = getattr(delta, "reasoning", None) if delta else None
        current_finish_reason = getattr(choice, "finish_reason", None)

        if content:
            content_parts.append(content)

            if print_stream:
                print(content, end="", flush=True)

        if reasoning:
            reasoning_chars += len(str(reasoning))

        if current_finish_reason is not None:
            finish_reason = str(current_finish_reason)

    answer = "".join(content_parts).strip()

    if not answer:
        raise RuntimeError(
            f"{model} 返回空 content；"
            f"reasoning_chars={reasoning_chars}, "
            f"finish_reason={finish_reason}"
        )

    return {
        "model": model,
        "content": answer,
        "reasoning_chars": reasoning_chars,
        "finish_reason": finish_reason,
    }


async def call_model(
    client: AsyncTogether,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    stage: str,
    print_stream: bool = False,
) -> dict[str, Any]:
    retry_token_limit = 8192
    max_attempts = 4

    attempt_records: list[dict[str, Any]] = []
    total_start_time = time.perf_counter()

    for attempt_number in range(1, max_attempts + 1):
        request_max_tokens = max_tokens
        attempt_start_time = time.perf_counter()

        try:
            result = await collect_stream_response(
                client,
                model=model,
                messages=messages,
                max_tokens=request_max_tokens,
                print_stream=print_stream,
            )

            attempt_duration = round(
                time.perf_counter() - attempt_start_time,
                2,
            )

            attempt_record = {
                "attempt": attempt_number,
                "max_tokens": request_max_tokens,
                "duration_seconds": attempt_duration,
                "finish_reason": result.get("finish_reason"),
                "reasoning_chars": result.get("reasoning_chars", 0),
                "content_chars": len(result.get("content", "")),
                "api_success": True,
                "status_code": None,
                "error_type": None,
                "error_message": None,
            }

            if (
                result["finish_reason"] == "length"
                and request_max_tokens < retry_token_limit
            ):
                attempt_record["outcome"] = "truncated_retry"
                attempt_records.append(attempt_record)

                max_tokens = min(
                    request_max_tokens * 2,
                    retry_token_limit,
                )

                print(
                    f"{stage} 输出被截断，将 max_tokens "
                    f"从 {request_max_tokens} 提高到 "
                    f"{max_tokens} 后重试……"
                )
                continue

            if result["finish_reason"] == "length":
                attempt_record["outcome"] = "truncated_at_limit"
            else:
                attempt_record["outcome"] = "completed"

            attempt_records.append(attempt_record)

            result["stage"] = stage
            result["duration_seconds"] = round(
                time.perf_counter() - total_start_time,
                2,
            )
            result["final_attempt_duration_seconds"] = (
                attempt_duration
            )
            result["api_attempt_duration_seconds"] = round(
                sum(
                    record["duration_seconds"]
                    for record in attempt_records
                ),
                2,
            )
            result["logical_request_count"] = 1
            result["actual_api_request_count"] = len(
                attempt_records
            )
            result["retry_count"] = max(
                0,
                len(attempt_records) - 1,
            )
            result["attempts"] = attempt_records
            result["final_max_tokens"] = request_max_tokens

            return result

        except Exception as exc:
            attempt_duration = round(
                time.perf_counter() - attempt_start_time,
                2,
            )

            status_code = getattr(exc, "status_code", None)
            error_text = str(exc)

            attempt_record = {
                "attempt": attempt_number,
                "max_tokens": request_max_tokens,
                "duration_seconds": attempt_duration,
                "finish_reason": None,
                "reasoning_chars": None,
                "content_chars": 0,
                "api_success": False,
                "status_code": status_code,
                "error_type": type(exc).__name__,
                "error_message": error_text,
                "outcome": "failed",
            }

            if (
                "返回空 content" in error_text
                and "finish_reason=length" in error_text
                and request_max_tokens < retry_token_limit
            ):
                attempt_record["finish_reason"] = "length"
                attempt_record["outcome"] = (
                    "empty_content_length_retry"
                )
                attempt_records.append(attempt_record)

                max_tokens = min(
                    request_max_tokens * 2,
                    retry_token_limit,
                )

                print(
                    f"{stage} 的 reasoning 占满输出预算，"
                    f"将 max_tokens 从 {request_max_tokens} "
                    f"提高到 {max_tokens} 后重试……"
                )
                continue

            if status_code == 429 and attempt_number < max_attempts:
                attempt_record["outcome"] = "rate_limit_retry"
                attempt_records.append(attempt_record)

                delay = 2 ** (attempt_number - 1)
                print(
                    f"{stage} 遇到限流，"
                    f"{delay} 秒后重试……"
                )

                await asyncio.sleep(delay)
                continue

            attempt_records.append(attempt_record)

            raise RuntimeError(
                f"{stage} 调用失败：{model}；"
                f"{type(exc).__name__}: {exc}；"
                f"actual_api_requests={len(attempt_records)}；"
                f"retries={max(0, len(attempt_records) - 1)}"
            ) from exc

    raise RuntimeError(
        f"{stage} 重试耗尽：{model}；"
        f"actual_api_requests={len(attempt_records)}；"
        f"retries={max(0, len(attempt_records) - 1)}"
    )


async def run_reference_layer(
    client: AsyncTogether,
    *,
    prompt: str,
    layer_number: int,
    previous_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    print(f"\n正在运行 reference layer {layer_number}……")

    messages = build_messages(prompt, previous_results)

    raw_results = await asyncio.gather(
        *[
            call_model(
                client,
                model=model,
                messages=messages,
                max_tokens=(
                    1024 if layer_number == 1 else 2048
                ),
                stage=f"reference-layer-{layer_number}",
            )
            for model in REFERENCE_MODELS
        ],
        return_exceptions=True,
    )

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, BaseException]] = []

    for model, item in zip(REFERENCE_MODELS, raw_results):
        if isinstance(item, BaseException):
            failures.append((model, item))
        else:
            results.append(item)

    for result in results:
        print(
            f"✓ {result['model']} | "
            f"{result['duration_seconds']} 秒 | "
            f"finish={result['finish_reason']}"
        )

    if failures:
        for model, error in failures:
            print(f"✗ {model}: {error}")

        raise RuntimeError(
            f"reference layer {layer_number} 存在失败模型"
        )

    return results


async def run_experiment(
    question: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    api_key = os.getenv("TOGETHER_API_KEY")

    if not api_key:
        raise RuntimeError("未检测到 TOGETHER_API_KEY")

    client = AsyncTogether(api_key=api_key)
    prompt = question["prompt"]
    total_start = time.perf_counter()
    layers: list[dict[str, Any]] = []

    if mode == "single":
        print("\n正在运行 Single Model……\n")

        final_result = await call_model(
            client,
            model=AGGREGATOR_MODEL,
            messages=build_messages(prompt),
            max_tokens=2048,
            stage="single",
            print_stream=True,
        )

        request_count = 1

    else:
        reference_layer_count = (
            1 if mode == "two-layer" else 2
        )
        previous_results = None

        for layer_number in range(1, reference_layer_count + 1):
            previous_results = await run_reference_layer(
                client,
                prompt=prompt,
                layer_number=layer_number,
                previous_results=previous_results,
            )

            layers.append(
                {
                    "layer_number": layer_number,
                    "responses": previous_results,
                }
            )

        print("\n正在运行最终 Aggregator……\n")

        final_result = await call_model(
            client,
            model=AGGREGATOR_MODEL,
            messages=build_messages(prompt, previous_results),
            max_tokens=2048,
            stage="aggregator",
            print_stream=True,
        )

        request_count = 5 if mode == "two-layer" else 9

    print()

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": question["id"],
        "category": question["category"],
        "mode": mode,
        "prompt": prompt,
        "reference_answer": question.get("reference_answer"),
        "key_points": question.get("key_points", []),
        "reference_models": REFERENCE_MODELS if mode != "single" else [],
        "aggregator_model": AGGREGATOR_MODEL,
        "request_count": request_count,
        "layers": layers,
        "final_result": final_result,
        "total_duration_seconds": round(
            time.perf_counter() - total_start,
            2,
        ),
    }


def save_result(record: dict[str, Any]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULT_DIR / (
        f"{record['question_id']}_{record['mode']}_{timestamp}.jsonl"
    )

    output_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["single", "two-layer", "three-layer"],
    )
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    question = load_question(args.question_id)

    request_counts = {
        "single": 1,
        "two-layer": 5,
        "three-layer": 9,
    }

    if args.dry_run:
        print("Dry run：不会发送 API 请求")
        print("题目:", question["id"])
        print("类别:", question["category"])
        print("模式:", args.mode)
        print("预计请求数:", request_counts[args.mode])
        print("Aggregator:", AGGREGATOR_MODEL)
        return

    result = asyncio.run(run_experiment(question, args.mode))
    output_path = save_result(result)

    print("\n实验完成")
    print("模式:", result["mode"])
    print("请求数:", result["request_count"])
    print("总耗时:", result["total_duration_seconds"], "秒")
    print("结果文件:", output_path)


if __name__ == "__main__":
    main()
