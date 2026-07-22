from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from together import AsyncTogether


BASE_DIR = Path(__file__).resolve().parent
QUESTION_FILE = BASE_DIR / "data" / "question.jsonl"
RESULT_DIR = BASE_DIR / "results"

# 复用前面已经验证过的模型调用和重试逻辑。
MOA_EVAL_DIR = BASE_DIR.parent / "moa_eval"
sys.path.insert(0, str(MOA_EVAL_DIR))

from run_eval import (  # noqa: E402
    AGGREGATOR_MODEL,
    AGGREGATOR_SYSTEM_PROMPT,
    REFERENCE_MODELS,
    call_model,
)


def load_question(question_id: int) -> dict[str, Any]:
    for line in QUESTION_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        question = json.loads(line)

        if int(question["question_id"]) == question_id:
            return question

    raise ValueError(f"未找到 question_id={question_id}")


def format_reference_responses(
    results: list[dict[str, Any]],
) -> str:
    return "\n".join(
        f"{index + 1}. {result['content']}"
        for index, result in enumerate(results)
    )


def build_messages(
    history: list[dict[str, str]],
    current_prompt: str,
    previous_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    if previous_results:
        messages.append(
            {
                "role": "system",
                "content": (
                    AGGREGATOR_SYSTEM_PROMPT
                    + "\n"
                    + format_reference_responses(previous_results)
                ),
            }
        )

    # 保存用户实际看到的多轮对话。
    for turn in history:
        messages.append(
            {
                "role": "user",
                "content": turn["user"],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": turn["assistant"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": current_prompt,
        }
    )

    return messages


async def run_reference_layer(
    client: AsyncTogether,
    *,
    history: list[dict[str, str]],
    prompt: str,
    turn_number: int,
    layer_number: int,
    previous_results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    print(
        f"\n正在运行 turn {turn_number} "
        f"reference layer {layer_number}……"
    )

    messages = build_messages(
        history,
        prompt,
        previous_results,
    )

    max_tokens = 2048 if layer_number == 1 else 4096

    raw_results = await asyncio.gather(
        *[
            call_model(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stage=(
                    f"turn-{turn_number}-"
                    f"reference-layer-{layer_number}"
                ),
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
            f"turn {turn_number} layer {layer_number} "
            "存在失败模型"
        )

    return results


async def run_mt_bench_question(
    question: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    api_key = os.getenv("TOGETHER_API_KEY")

    if not api_key:
        raise RuntimeError("未检测到 TOGETHER_API_KEY")

    client = AsyncTogether(api_key=api_key)

    history: list[dict[str, str]] = []
    turn_records: list[dict[str, Any]] = []

    total_logical_requests = 0
    total_actual_api_requests = 0
    total_retries = 0
    total_api_attempt_duration = 0.0

    total_start = time.perf_counter()

    for turn_number, prompt in enumerate(
        question["turns"],
        start=1,
    ):
        turn_start = time.perf_counter()

        print("\n" + "=" * 70)
        print(
            f"Question {question['question_id']} | "
            f"Turn {turn_number}"
        )
        print("=" * 70)
        print("用户问题:", prompt)

        layers: list[dict[str, Any]] = []

        # 保存这一轮所有模型调用结果，
        # 包括各层 Reference Models 和最终模型。
        turn_call_results: list[dict[str, Any]] = []

        if mode == "single":
            print("\n正在运行 Single Model……\n")

            final_result = await call_model(
                client,
                model=AGGREGATOR_MODEL,
                messages=build_messages(history, prompt),
                max_tokens=4096,
                stage=f"turn-{turn_number}-single",
                print_stream=True,
            )

            turn_call_results.append(final_result)

        else:
            reference_layer_count = (
                1 if mode == "two-layer" else 2
            )

            previous_results = None

            for layer_number in range(
                1,
                reference_layer_count + 1,
            ):
                layer_results = await run_reference_layer(
                    client,
                    history=history,
                    prompt=prompt,
                    turn_number=turn_number,
                    layer_number=layer_number,
                    previous_results=previous_results,
                )

                turn_call_results.extend(layer_results)

                layers.append(
                    {
                        "layer_number": layer_number,
                        "responses": layer_results,
                    }
                )

                previous_results = layer_results

            print(
                f"\n正在运行 turn {turn_number} "
                "最终 Aggregator……\n"
            )

            final_result = await call_model(
                client,
                model=AGGREGATOR_MODEL,
                messages=build_messages(
                    history,
                    prompt,
                    previous_results,
                ),
                max_tokens=4096,
                stage=f"turn-{turn_number}-aggregator",
                print_stream=True,
            )

            turn_call_results.append(final_result)

        print()

        turn_logical_requests = sum(
            result.get("logical_request_count", 1)
            for result in turn_call_results
        )

        turn_actual_api_requests = sum(
            result.get("actual_api_request_count", 1)
            for result in turn_call_results
        )

        turn_retries = sum(
            result.get("retry_count", 0)
            for result in turn_call_results
        )

        turn_api_attempt_duration = round(
            sum(
                result.get(
                    "api_attempt_duration_seconds",
                    result.get("duration_seconds", 0.0),
                )
                for result in turn_call_results
            ),
            2,
        )

        turn_duration = round(
            time.perf_counter() - turn_start,
            2,
        )

        print(
            f"Turn {turn_number} 统计："
            f"logical={turn_logical_requests}，"
            f"actual_api={turn_actual_api_requests}，"
            f"retries={turn_retries}，"
            f"wall_time={turn_duration} 秒"
        )

        history.append(
            {
                "user": prompt,
                "assistant": final_result["content"],
            }
        )

        total_logical_requests += turn_logical_requests
        total_actual_api_requests += turn_actual_api_requests
        total_retries += turn_retries
        total_api_attempt_duration += turn_api_attempt_duration

        turn_records.append(
            {
                "turn_number": turn_number,
                "prompt": prompt,

                # 保留旧字段，含义改为逻辑请求数。
                "request_count": turn_logical_requests,

                "logical_request_count": (
                    turn_logical_requests
                ),
                "actual_api_request_count": (
                    turn_actual_api_requests
                ),
                "retry_count": turn_retries,
                "api_attempt_duration_seconds": (
                    turn_api_attempt_duration
                ),
                "turn_duration_seconds": turn_duration,

                "layers": layers,
                "final_result": final_result,
            }
        )

    total_duration = round(
        time.perf_counter() - total_start,
        2,
    )

    print("\n" + "-" * 70)
    print(
        "整题统计："
        f"logical={total_logical_requests}，"
        f"actual_api={total_actual_api_requests}，"
        f"retries={total_retries}，"
        f"total_time={total_duration} 秒"
    )
    print("-" * 70)

    return {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "question_id": question["question_id"],
        "category": question["category"],
        "mode": mode,
        "reference_models": (
            REFERENCE_MODELS if mode != "single" else []
        ),
        "aggregator_model": AGGREGATOR_MODEL,

        # 保留旧字段，避免旧脚本读取时报错。
        "request_count": total_logical_requests,

        "logical_request_count": total_logical_requests,
        "actual_api_request_count": (
            total_actual_api_requests
        ),
        "retry_count": total_retries,
        "api_attempt_duration_seconds": round(
            total_api_attempt_duration,
            2,
        ),

        "turns": turn_records,
        "conversation_history": history,
        "total_duration_seconds": total_duration,
    }


def save_result(record: dict[str, Any]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = RESULT_DIR / (
        f"q{record['question_id']}_"
        f"{record['mode']}_{timestamp}.jsonl"
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
        choices=[
            "single",
            "two-layer",
            "three-layer",
        ],
    )

    parser.add_argument(
        "--question-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()
    question = load_question(args.question_id)

    request_counts = {
        "single": 2,
        "two-layer": 10,
        "three-layer": 18,
    }

    if args.dry_run:
        print("Dry run：不会发送 API 请求")
        print("question_id:", question["question_id"])
        print("category:", question["category"])
        print("turn 数量:", len(question["turns"]))
        print("模式:", args.mode)
        print("预计请求数:", request_counts[args.mode])

        for index, turn in enumerate(
            question["turns"],
            start=1,
        ):
            print(f"turn {index}:", turn)

        return

    record = asyncio.run(
        run_mt_bench_question(question, args.mode)
    )

    output_path = save_result(record)

    print("\nMT-Bench pilot 实验完成")
    print("question_id:", record["question_id"])
    print("模式:", record["mode"])
    print("总请求数:", record["request_count"])
    print("总耗时:", record["total_duration_seconds"], "秒")
    print("结果文件:", output_path)


if __name__ == "__main__":
    main()
