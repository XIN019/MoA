import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

_JUDGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_JUDGE_DIR))

from prepare_judge_inputs import (  # noqa: E402
    LABELS,
    MODES,
    find_latest_result,
    load_jsonl_record,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "results"

DEFAULT_OUTPUT_DIR = BASE_DIR / "judge" / "data" / "turn_judge"
DEFAULT_RANDOM_SEED = 20260722


def stable_question_seed(seed: int, question_id: int) -> int:
    value = f"{seed}:{question_id}".encode("utf-8")
    digest = hashlib.sha256(value).hexdigest()
    return int(digest[:16], 16)


def extract_turn_answers(record: dict[str, Any], turn_number: int) -> dict[str, Any]:
    turns = record.get("turns", [])
    result: dict[str, Any] = {}

    for turn in turns:
        tn = int(turn["turn_number"])
        if tn == 1:
            result["turn1_answer"] = turn.get("final_result", {}).get("content", "")
        if tn == turn_number:
            answer = turn.get("final_result", {}).get("content", "")
            result["target_answer"] = answer
            result.setdefault("turn1_answer", "")

    return result


def build_turn1_prompt(record: dict[str, Any]) -> str:
    user_turn = record["user_turns"][0]

    parts = [
        f"Question ID: {record['question_id']}",
        f"Category: {record.get('category', '')}",
        "",
        "USER CONVERSATION (Turn 1)",
        "=" * 27,
        f"User turn 1:\n{user_turn}",
        "",
        "ANONYMOUS CANDIDATE ANSWERS (Turn 1)",
        "=" * 36,
        "Evaluate ONLY the Turn 1 responses below. Score each independently.",
        "",
    ]

    for label in LABELS:
        answer_text = record["answers"][label]["turn1_answer"]
        parts.append(f"Answer {label}:\n{answer_text}")
        parts.append("")

    parts.extend([
        "REQUIRED JSON STRUCTURE",
        "=" * 23,
        json.dumps(_output_contract(record), ensure_ascii=False, indent=2),
    ])

    return "\n".join(parts)


def build_turn2_prompt(record: dict[str, Any]) -> str:
    user_turns = record["user_turns"]
    turn1_user = user_turns[0]
    turn2_user = user_turns[1]

    parts = [
        f"Question ID: {record['question_id']}",
        f"Category: {record.get('category', '')}",
        "",
        "USER CONVERSATION (Full Context)",
        "=" * 31,
        f"User turn 1:\n{turn1_user}",
        "",
        f"User turn 2:\n{turn2_user}",
        "",
        "ANONYMOUS CANDIDATE ANSWERS (Full Context)",
        "=" * 44,
        "Evaluate ONLY the Turn 2 responses below. However you MUST consider the",
        "Turn 1 context (user question + each answer's own Turn 1 response) when",
        "judging Turn 2. Check for consistency, contradiction, follow-up correctness,",
        "and whether Turn 2 properly addresses the instruction change.",
        "",
    ]

    for label in LABELS:
        turn1_answer = record["answers"][label]["turn1_answer"]
        turn2_answer = record["answers"][label]["target_answer"]

        parts.append(
            f"Answer {label}:\n"
            f"\nAssistant response for turn 1:\n{turn1_answer}\n"
            f"\nAssistant response for turn 2:\n{turn2_answer}"
        )
        parts.append("")

    parts.extend([
        "REQUIRED JSON STRUCTURE",
        "=" * 23,
        json.dumps(_output_contract(record), ensure_ascii=False, indent=2),
    ])

    return "\n".join(parts)


def _output_contract(record: dict[str, Any]) -> dict[str, Any]:
    question_id = int(record["question_id"])
    turn_number = record["turn_number"]

    evals: dict[str, dict[str, str]] = {}
    for label in LABELS:
        evals[label] = {
            "correctness": "integer 1-10",
            "instruction_following": "integer 1-10",
            "relevance": "integer 1-10",
            "clarity": "integer 1-10",
            "overall_score": "integer 1-10",
            "major_error": "boolean",
            "major_error_description": "string",
            "rationale": "string",
        }

    return {
        "question_id": question_id,
        "category": str(record.get("category", "")),
        "judged_turn_number": turn_number,
        "evaluations": evals,
        "ranking": [
            "best answer label",
            "second-best answer label",
            "worst answer label",
        ],
        "ranking_reason": "string",
    }


def discover_complete_questions() -> list[int]:
    question_modes: dict[int, set[str]] = {}
    for path in RESULT_DIR.glob("q*_*.jsonl"):
        try:
            record = load_jsonl_record(path)
            question_id = int(record["question_id"])
            mode = str(record["mode"]).strip()
            question_modes.setdefault(question_id, set()).add(mode)
        except Exception:
            pass

    complete = sorted(
        qid for qid, modes in question_modes.items()
        if set(MODES).issubset(modes)
    )
    return complete


def generate_answer_key(seed: int, question_id: int) -> dict[str, Any]:
    shuffled = MODES.copy()
    rng = random.Random(stable_question_seed(seed, question_id))
    rng.shuffle(shuffled)

    label_to_mode = dict(zip(LABELS, shuffled))
    mode_to_label = {mode: lbl for lbl, mode in label_to_mode.items()}

    return {
        "label_to_mode": label_to_mode,
        "mode_to_label": mode_to_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", type=int, action="append", dest="question_ids",
                        help="Question ID (repeatable)")
    parser.add_argument("--turn-number", type=int, required=True, choices=[1, 2])
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    turn_number = args.turn_number

    if args.question_ids is None:
        question_ids = discover_complete_questions()
        print(f"未指定题号，自动发现 {len(question_ids)} 道完整题目：{question_ids}")
    else:
        question_ids = args.question_ids

    for question_id in question_ids:
        mode_records: dict[str, dict[str, Any]] = {}
        category = ""
        user_turns: list[str] = []

        for mode in MODES:
            path, record = find_latest_result(question_id, mode)
            mode_records[mode] = record

            if not category:
                category = str(record.get("category", ""))
            if not user_turns:
                user_turns = [
                    str(turn["prompt"])
                    for turn in record.get("turns", [])
                ][:2]

        key = generate_answer_key(seed, question_id)
        label_to_mode = key["label_to_mode"]

        anonymous_answers: dict[str, dict[str, Any]] = {}
        for label, mode in label_to_mode.items():
            extracted = extract_turn_answers(mode_records[mode], turn_number)
            anonymous_answers[label] = extracted

        answer_key_filename = f"q{question_id}_turn{turn_number}_answer_key.json"
        answer_key_data = {
            "random_seed": seed,
            "turn_number": turn_number,
            "question_id": question_id,
            "label_to_mode": label_to_mode,
            "mode_to_label": key["mode_to_label"],
        }

        answer_key_path = output_dir / answer_key_filename
        answer_key_path.write_text(
            json.dumps(answer_key_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        judge_record = {
            "question_id": question_id,
            "category": category,
            "turn_number": turn_number,
            "user_turns": user_turns,
            "answers": anonymous_answers,
            "answer_key_file": answer_key_filename,
        }

        if turn_number == 1:
            judge_record["prompt_text"] = build_turn1_prompt(judge_record)
        else:
            judge_record["prompt_text"] = build_turn2_prompt(judge_record)

        input_filename = f"q{question_id}_turn{turn_number}_judge_input.jsonl"
        input_path = output_dir / input_filename
        input_path.write_text(
            json.dumps(judge_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(
            f"Q{question_id} Turn {turn_number}: "
            f"input={input_filename}, key={answer_key_filename} | "
            f"A={label_to_mode['A']}, B={label_to_mode['B']}, C={label_to_mode['C']}"
        )

    print(f"\nOutput dir: {output_dir}")
    print(f"Questions prepared: {len(question_ids)}")


if __name__ == "__main__":
    main()
