import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from together import Together

_JUDGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_JUDGE_DIR))

from run_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    call_judge,
    print_summary,
)

MT_BENCH_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = MT_BENCH_DIR / "judge" / "data" / "turn_judge"
DEFAULT_RESULT_DIR = MT_BENCH_DIR / "results" / "judge_turn"

TURN_SYSTEM_PROMPT = (
    """\
You are an impartial evaluator for MT-Bench model responses.

You will receive:
1. A user conversation.
2. Three anonymous candidate answers: Answer A, Answer B, and Answer C.

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
- Evaluate only the specified turn.
- When evaluating turn 2, check whether it correctly uses the context
  established in turn 1.
- Treat contradiction between turns as a quality problem.

You are evaluating a specific turn of a multi-turn conversation.
The field judged_turn_number in your output MUST match the turn
number specified in the input. Score ONLY the target turn's
responses, but use prior turn context to assess correctness,
consistency, and instruction-following.

Return valid JSON only. Do not use Markdown fences.
"""
)


def find_input_file(input_dir: Path, question_id: int, turn_number: int) -> Path:
    if input_dir.is_file():
        return input_dir

    candidate = input_dir / f"q{question_id}_turn{turn_number}_judge_input.jsonl"
    if candidate.exists():
        return candidate

    legacy = input_dir / f"turn{turn_number}_judge_inputs.jsonl"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"未找到 Turn Judge 输入文件：q{question_id}_turn{turn_number}_judge_input.jsonl\n"
        f"已检查目录：{input_dir}\n请先运行 prepare_turn_judge_inputs.py"
    )


def load_inputs(input_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        input_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Turn judge 输入第 {line_number} 行 JSON 无效：{exc}"
            ) from exc
    return records


def select_record(records: list[dict[str, Any]], question_id: int) -> dict[str, Any]:
    for rec in records:
        if int(rec["question_id"]) == question_id:
            return rec
    available = sorted(int(r["question_id"]) for r in records)
    raise ValueError(
        f"未找到 question_id={question_id}；当前可用题号：{available}"
    )


def result_exists(output_dir: Path, question_id: int, turn_number: int) -> bool:
    return any(output_dir.glob(f"q{question_id}_turn{turn_number}_judge_*.json"))


def save_result(
    *,
    output_dir: Path,
    question_record: dict[str, Any],
    model: str,
    judgment: dict[str, Any],
    raw_response: str,
    duration_seconds: float,
    actual_api_request_count: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    turn_number = question_record["turn_number"]
    qid = question_record["question_id"]

    output_path = output_dir / (
        f"q{qid}_turn{turn_number}_judge_{timestamp}.json"
    )

    answer_key_file = question_record.get("answer_key_file", "")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": int(qid),
        "category": question_record.get("category", ""),
        "judge_model": model,
        "anonymous_labels": ["A", "B", "C"],
        "judged_turn_number": turn_number,
        "duration_seconds": duration_seconds,
        "actual_api_request_count": actual_api_request_count,
        "answer_key_file": answer_key_file,
        "judgment": judgment,
        "raw_response": raw_response,
    }

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", type=int, required=True)
    parser.add_argument("--turn-number", type=int, required=True, choices=[1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULT_DIR

    if args.skip_existing and result_exists(output_dir, args.question_id, args.turn_number):
        print(f"跳过已有结果：Q{args.question_id} Turn {args.turn_number}")
        return

    input_dir = Path(args.input_file) if args.input_file else DEFAULT_INPUT_DIR
    input_file = find_input_file(input_dir, args.question_id, args.turn_number)

    records = load_inputs(input_file)
    record = select_record(records, args.question_id)

    prompt = record["prompt_text"]

    print("Question ID:", record["question_id"])
    print("Judged turn:", record["turn_number"])
    print("Category:", record.get("category", ""))
    print("Judge model:", os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    print("Prompt characters:", len(prompt))

    for label in ["A", "B", "C"]:
        answer_len = len(
            record["answers"][label].get("target_answer", "")
        )
        print(f"Answer {label} turn {args.turn_number} chars:", answer_len)

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_path = output_dir / (
            f"q{record['question_id']}_turn{args.turn_number}"
            f"_prompt_{timestamp}.txt"
        )
        prompt_path.write_text(prompt, encoding="utf-8")

        print(f"\nDry-run 完成：没有调用 Judge API。")
        print(f"Judge prompt 已保存至：{prompt_path}")
        return

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 TOGETHER_API_KEY")

    client = Together(api_key=api_key)

    judgment, raw_response, duration, attempts = call_judge(
        client,
        model=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        prompt=prompt,
    )

    output_path = save_result(
        output_dir=output_dir,
        question_record=record,
        model=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        judgment=judgment,
        raw_response=raw_response,
        duration_seconds=duration,
        actual_api_request_count=attempts,
    )

    print_summary(judgment)
    print("\nTurn Judge 调用完成")
    print("API 请求次数:", attempts)
    print("耗时:", duration, "秒")
    print("结果文件:", output_path)


if __name__ == "__main__":
    main()
