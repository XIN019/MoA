import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from together import Together


MT_BENCH_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    MT_BENCH_DIR
    / "judge"
    / "data"
    / "judge_inputs.jsonl"
)

RESULT_DIR = (
    MT_BENCH_DIR
    / "results"
    / "judge"
)

DEFAULT_JUDGE_MODEL = "zai-org/GLM-5.2"


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question_id": {
            "type": "integer",
        },
        "category": {
            "type": "string",
        },
        "evaluations": {
            "type": "object",
            "properties": {
                label: {
                    "type": "object",
                    "properties": {
                        "correctness": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "instruction_following": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "relevance": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "clarity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "overall_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "major_error": {
                            "type": "boolean",
                        },
                        "major_error_description": {
                            "type": "string",
                        },
                        "rationale": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "correctness",
                        "instruction_following",
                        "relevance",
                        "clarity",
                        "overall_score",
                        "major_error",
                        "major_error_description",
                        "rationale",
                    ],
                    "additionalProperties": False,
                }
                for label in ["A", "B", "C"]
            },
            "required": ["A", "B", "C"],
            "additionalProperties": False,
        },
        "ranking": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["A", "B", "C"],
            },
            "minItems": 3,
            "maxItems": 3,
            "uniqueItems": True,
        },
        "ranking_reason": {
            "type": "string",
        },
    },
    "required": [
        "question_id",
        "category",
        "evaluations",
        "ranking",
        "ranking_reason",
    ],
    "additionalProperties": False,
}


JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator for MT-Bench model responses.

You will receive:
1. A two-turn user conversation.
2. Three anonymous candidate conversations: Answer A, Answer B, and Answer C.

Evaluate each answer independently. Do not infer which system produced an
answer. Do not prefer an answer because it is longer or more detailed.

Evaluation dimensions, each scored from 1 to 10:

- correctness:
  Factual, logical, mathematical, and technical correctness.
- instruction_following:
  Compliance with all explicit and implicit user requirements.
- relevance:
  Directness and usefulness for the user's request.
- clarity:
  Organization, readability, and appropriate level of detail.
- overall_score:
  Holistic answer quality.

Major error definition:
A substantive error that makes an important part of the response incorrect,
unusable, misleading, or non-executable.

For coding questions:
- Inspect function signatures and their call sites.
- Check whether required arguments are passed.
- Check imports, variable names, multiprocessing pickling constraints,
  likely runtime exceptions, and whether the proposed code can execute.
- Do not assume code is correct merely because its explanation sounds good.

For multi-turn questions:
- Evaluate both turns.
- Check whether turn 2 correctly uses the context established in turn 1.
- Treat contradiction between turns as a quality problem.

Return valid JSON only. Do not use Markdown fences.
""".strip()


def load_judge_inputs() -> list[dict[str, Any]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"未找到 Judge 输入文件：{INPUT_FILE}\n"
            "请先运行 prepare_judge_inputs.py"
        )

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        INPUT_FILE.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Judge 输入第 {line_number} 行 JSON 无效：{exc}"
            ) from exc

    return records


def select_record(
    records: list[dict[str, Any]],
    question_id: int,
) -> dict[str, Any]:
    for record in records:
        if int(record["question_id"]) == question_id:
            return record

    available = sorted(
        int(record["question_id"])
        for record in records
    )

    raise ValueError(
        f"未找到 question_id={question_id}；"
        f"当前可用题号：{available}"
    )


def format_candidate(
    label: str,
    candidate: dict[str, Any],
) -> str:
    parts = [f"Answer {label}"]

    for answer_turn in candidate.get("answers", []):
        turn_number = answer_turn["turn_number"]
        answer = answer_turn.get(
            "assistant_answer",
            "",
        )

        parts.append(
            f"\nAssistant response for turn {turn_number}:\n"
            f"{answer}"
        )

    return "\n".join(parts)


def build_judge_prompt(
    record: dict[str, Any],
) -> str:
    user_parts: list[str] = []

    for index, user_turn in enumerate(
        record["user_turns"],
        start=1,
    ):
        user_parts.append(
            f"User turn {index}:\n{user_turn}"
        )

    candidate_parts = [
        format_candidate(
            label,
            record["answers"][label],
        )
        for label in ["A", "B", "C"]
    ]

    output_contract = {
        "question_id": int(record["question_id"]),
        "category": str(record.get("category", "")),
        "evaluations": {
            "A": {
                "correctness": "integer 1-10",
                "instruction_following": "integer 1-10",
                "relevance": "integer 1-10",
                "clarity": "integer 1-10",
                "overall_score": "integer 1-10",
                "major_error": "boolean",
                "major_error_description": "string",
                "rationale": "string",
            },
            "B": {
                "correctness": "integer 1-10",
                "instruction_following": "integer 1-10",
                "relevance": "integer 1-10",
                "clarity": "integer 1-10",
                "overall_score": "integer 1-10",
                "major_error": "boolean",
                "major_error_description": "string",
                "rationale": "string",
            },
            "C": {
                "correctness": "integer 1-10",
                "instruction_following": "integer 1-10",
                "relevance": "integer 1-10",
                "clarity": "integer 1-10",
                "overall_score": "integer 1-10",
                "major_error": "boolean",
                "major_error_description": "string",
                "rationale": "string",
            },
        },
        "ranking": [
            "best answer label",
            "second-best answer label",
            "worst answer label",
        ],
        "ranking_reason": "string",
    }

    return (
        f"Question ID: {record['question_id']}\n"
        f"Category: {record.get('category', '')}\n\n"
        "USER CONVERSATION\n"
        "=================\n"
        + "\n\n".join(user_parts)
        + "\n\n"
        "ANONYMOUS CANDIDATE ANSWERS\n"
        "===========================\n"
        + "\n\n".join(candidate_parts)
        + "\n\n"
        "REQUIRED JSON STRUCTURE\n"
        "=======================\n"
        + json.dumps(
            output_contract,
            ensure_ascii=False,
            indent=2,
        )
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Judge 输出中未找到有效 JSON 对象"
            )

        result = json.loads(
            cleaned[start:end + 1]
        )

    if not isinstance(result, dict):
        raise ValueError("Judge 输出不是 JSON 对象")

    return result


def validate_score(
    value: Any,
    *,
    label: str,
    field: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(
            f"Answer {label} 的 {field} 必须是整数"
        )

    if not 1 <= value <= 10:
        raise ValueError(
            f"Answer {label} 的 {field} 超出 1-10"
        )


def validate_judgment(
    judgment: dict[str, Any],
) -> None:
    evaluations = judgment.get("evaluations")

    if not isinstance(evaluations, dict):
        raise ValueError("缺少 evaluations 对象")

    score_fields = [
        "correctness",
        "instruction_following",
        "relevance",
        "clarity",
        "overall_score",
    ]

    for label in ["A", "B", "C"]:
        evaluation = evaluations.get(label)

        if not isinstance(evaluation, dict):
            raise ValueError(
                f"缺少 Answer {label} 的评分"
            )

        for field in score_fields:
            validate_score(
                evaluation.get(field),
                label=label,
                field=field,
            )

        if not isinstance(
            evaluation.get("major_error"),
            bool,
        ):
            raise ValueError(
                f"Answer {label} 的 major_error "
                "必须是布尔值"
            )

        for field in [
            "major_error_description",
            "rationale",
        ]:
            if not isinstance(
                evaluation.get(field),
                str,
            ):
                raise ValueError(
                    f"Answer {label} 的 {field} "
                    "必须是字符串"
                )

    ranking = judgment.get("ranking")

    if (
        not isinstance(ranking, list)
        or len(ranking) != 3
        or set(ranking) != {"A", "B", "C"}
    ):
        raise ValueError(
            "ranking 必须是 A、B、C 的完整排列"
        )

    if not isinstance(
        judgment.get("ranking_reason"),
        str,
    ):
        raise ValueError(
            "ranking_reason 必须是字符串"
        )


def call_judge(
    client: Together,
    *,
    model: str,
    prompt: str,
) -> tuple[dict[str, Any], str, float, int]:
    last_error: Exception | None = None

    for attempt in range(1, 4):
        start_time = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=4096,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mt_bench_judgment",
                        "schema": JUDGE_RESPONSE_SCHEMA,
                    },
                },
            )

            duration = round(
                time.perf_counter() - start_time,
                2,
            )

            content = (
                response.choices[0]
                .message.content
            )

            if not content:
                raise ValueError(
                    "Judge 返回空 content"
                )

            judgment = extract_json(content)
            validate_judgment(judgment)

            return (
                judgment,
                content,
                duration,
                attempt,
            )

        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            print(
                f"Judge 第 {attempt} 次调用失败："
                f"{type(exc).__name__}: {exc}"
            )

            if (
                status_code == 400
                or "model_not_available" in error_text
                or "non-serverless model" in error_text
            ):
                raise RuntimeError(
                    "Judge 请求不可重试："
                    f"{error_text}"
                ) from exc

            if attempt < 3:
                delay = 2 ** (attempt - 1)
                print(f"{delay} 秒后重试……")
                time.sleep(delay)

    raise RuntimeError(
        f"Judge 重试耗尽：{last_error}"
    )


def save_result(
    *,
    question_record: dict[str, Any],
    model: str,
    judgment: dict[str, Any],
    raw_response: str,
    duration_seconds: float,
    actual_api_request_count: int,
) -> Path:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = RESULT_DIR / (
        f"q{question_record['question_id']}_"
        f"judge_{timestamp}.json"
    )

    result = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "question_id": int(
            question_record["question_id"]
        ),
        "category": question_record.get(
            "category",
            "",
        ),
        "judge_model": model,
        "anonymous_labels": ["A", "B", "C"],
        "duration_seconds": duration_seconds,
        "actual_api_request_count": (
            actual_api_request_count
        ),
        "judgment": judgment,
        "raw_response": raw_response,
    }

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def print_summary(
    judgment: dict[str, Any],
) -> None:
    print("\n匿名评分结果：")

    for label in ["A", "B", "C"]:
        evaluation = judgment[
            "evaluations"
        ][label]

        print(
            f"Answer {label}: "
            f"overall={evaluation['overall_score']}, "
            f"correctness={evaluation['correctness']}, "
            f"major_error={evaluation['major_error']}"
        )

        if evaluation["major_error"]:
            print(
                "  关键错误：",
                evaluation[
                    "major_error_description"
                ],
            )

    print(
        "\n匿名排名：",
        " > ".join(judgment["ranking"]),
    )
    print(
        "排名理由：",
        judgment["ranking_reason"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "JUDGE_MODEL",
            DEFAULT_JUDGE_MODEL,
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    api_key = os.getenv("TOGETHER_API_KEY")

    if not api_key and not args.dry_run:
        raise RuntimeError(
            "未检测到 TOGETHER_API_KEY"
        )

    records = load_judge_inputs()
    record = select_record(
        records,
        args.question_id,
    )

    prompt = build_judge_prompt(record)

    print("Question ID:", record["question_id"])
    print("Category:", record.get("category", ""))
    print("Judge model:", args.judge_model)
    print("Prompt characters:", len(prompt))

    for label in ["A", "B", "C"]:
        answer_chars = sum(
            len(turn["assistant_answer"])
            for turn in record[
                "answers"
            ][label]["answers"]
        )

        print(
            f"Answer {label} characters:",
            answer_chars,
        )

    if args.dry_run:
        print(
            "\nDry-run 完成："
            "没有调用 Judge API。"
        )
        return

    client = Together(api_key=api_key)

    judgment, raw_response, duration, attempts = (
        call_judge(
            client,
            model=args.judge_model,
            prompt=prompt,
        )
    )

    output_path = save_result(
        question_record=record,
        model=args.judge_model,
        judgment=judgment,
        raw_response=raw_response,
        duration_seconds=duration,
        actual_api_request_count=attempts,
    )

    print_summary(judgment)

    print("\nJudge 调用完成")
    print("API 请求次数:", attempts)
    print("耗时:", duration, "秒")
    print("结果文件:", output_path)


if __name__ == "__main__":
    main()
