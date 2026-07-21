# Mixture-of-Agents quickstart adapted for current Together Serverless models

import asyncio
import os
import time
from typing import Any

from together import AsyncTogether


api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise RuntimeError(
        "未检测到 TOGETHER_API_KEY，请先在当前终端设置环境变量。"
    )

async_client = AsyncTogether(api_key=api_key)

user_prompt = "What are 3 fun things to do in SF?"

reference_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3.7-Max",
    "moonshotai/Kimi-K2.7-Code",
    "pearl-ai/gemma-4-31b-it",
]

aggregator_model = "Qwen/Qwen3.7-Max"

aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


async def collect_stream_response(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    print_stream: bool = False,
) -> dict[str, Any]:
    """调用一个流式模型，并收集最终 content。"""

    stream = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
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
            f"{model} 返回了空 content；"
            f"reasoning_chars={reasoning_chars}, "
            f"finish_reason={finish_reason}"
        )

    return {
        "model": model,
        "content": answer,
        "reasoning_chars": reasoning_chars,
        "finish_reason": finish_reason,
    }


async def run_reference(model: str) -> dict[str, Any]:
    """运行一个 reference model，只对 429 错误进行重试。"""

    for attempt in range(4):
        try:
            start_time = time.perf_counter()

            result = await collect_stream_response(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                temperature=0.7,
                max_tokens=512,
            )

            result["duration_seconds"] = round(
                time.perf_counter() - start_time,
                2,
            )

            return result

        except Exception as exc:
            status_code = getattr(exc, "status_code", None)

            if status_code == 429 and attempt < 3:
                sleep_time = 2 ** attempt
                print(
                    f"{model} 遇到限流，"
                    f"{sleep_time} 秒后重试……"
                )
                await asyncio.sleep(sleep_time)
                continue

            raise RuntimeError(
                f"Reference model 调用失败：{model}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise RuntimeError(f"Reference model 重试耗尽：{model}")


async def main() -> None:
    print("正在并行调用 4 个 reference models……")

    raw_results = await asyncio.gather(
        *[run_reference(model) for model in reference_models],
        return_exceptions=True,
    )

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, BaseException]] = []

    for model, item in zip(reference_models, raw_results):
        if isinstance(item, BaseException):
            failures.append((model, item))
        else:
            results.append(item)

    for result in results:
        print(
            f"✓ {result['model']}\n"
            f"  content 字符数: {len(result['content'])}\n"
            f"  reasoning 字符数: {result['reasoning_chars']}\n"
            f"  耗时: {result['duration_seconds']} 秒"
        )

    if failures:
        print("\n以下 reference model 调用失败：")

        for model, error in failures:
            print(f"✗ {model}")
            print(f"  {type(error).__name__}: {error}")

        raise RuntimeError(
            "并非所有 reference model 都成功，已停止 aggregator 调用。"
        )

    numbered_responses = "\n".join(
        f"{index + 1}. {result['content']}"
        for index, result in enumerate(results)
    )

    aggregator_messages = [
        {
            "role": "system",
            "content": (
                aggregator_system_prompt
                + "\n"
                + numbered_responses
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    print("\n正在调用 aggregator……")
    print("\n=== Aggregated response ===\n")

    start_time = time.perf_counter()

    final_result = await collect_stream_response(
        model=aggregator_model,
        messages=aggregator_messages,
        temperature=0.7,
        max_tokens=2048,
        print_stream=True,
    )

    duration = round(time.perf_counter() - start_time, 2)

    print()
    print("\n=== Aggregator metadata ===")
    print("模型:", final_result["model"])
    print("reasoning 字符数:", final_result["reasoning_chars"])
    print("结束原因:", final_result["finish_reason"])
    print("耗时:", duration, "秒")


if __name__ == "__main__":
    asyncio.run(main())
