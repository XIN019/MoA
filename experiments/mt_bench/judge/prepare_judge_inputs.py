import hashlib
import json
import random
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "judge" / "data"

JUDGE_INPUT_FILE = OUTPUT_DIR / "judge_inputs.jsonl"
ANSWER_KEY_FILE = OUTPUT_DIR / "judge_answer_key.json"

MODES = [
    "single",
    "two-layer",
    "three-layer",
]

LABELS = [
    "A",
    "B",
    "C",
]

RANDOM_SEED = 20260722


def load_jsonl_record(path: Path) -> dict[str, Any]:
    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            return json.loads(line)

    raise ValueError(f"结果文件为空：{path}")


def find_latest_result(
    question_id: int,
    mode: str,
) -> tuple[Path, dict[str, Any]]:
    pattern = f"q{question_id}_{mode}_*.jsonl"

    candidates = sorted(
        RESULT_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"未找到 question={question_id}, mode={mode}"
        )

    path = candidates[0]
    return path, load_jsonl_record(path)


def discover_complete_questions() -> list[int]:
    question_modes: dict[int, set[str]] = {}

    for path in RESULT_DIR.glob("q*_*.jsonl"):
        try:
            record = load_jsonl_record(path)

            question_id = int(
                record["question_id"]
            )
            mode = str(
                record["mode"]
            ).strip()

            question_modes.setdefault(
                question_id,
                set(),
            ).add(mode)

        except Exception as exc:
            print(
                f"跳过无法读取的结果文件："
                f"{path.name}: {exc}"
            )

    print("发现的题目及模式：")

    for question_id in sorted(question_modes):
        modes = question_modes[question_id]
        print(
            f"  Question {question_id}: "
            f"{sorted(modes)}"
        )

    complete_questions = [
        question_id
        for question_id, modes
        in question_modes.items()
        if set(MODES).issubset(modes)
    ]

    print(
        "三种模式均完整的题目：",
        sorted(complete_questions),
    )

    return sorted(complete_questions)


def extract_conversation(
    record: dict[str, Any],
) -> dict[str, Any]:
    turns = record.get("turns", [])

    return {
        "answers": [
            {
                "turn_number": int(
                    turn["turn_number"]
                ),
                "assistant_answer": turn.get(
                    "final_result",
                    {},
                ).get(
                    "content",
                    "",
                ),
            }
            for turn in turns
        ],
        "source_file": record.get(
            "_source_file",
            "",
        ),
    }


def stable_question_seed(question_id: int) -> int:
    value = f"{RANDOM_SEED}:{question_id}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(value).hexdigest()

    return int(digest[:16], 16)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    question_ids = discover_complete_questions()

    if not question_ids:
        raise RuntimeError(
            "没有找到同时包含三种模式的题目"
        )

    judge_records: list[dict[str, Any]] = []
    answer_key: dict[str, Any] = {
        "random_seed": RANDOM_SEED,
        "questions": {},
    }

    for question_id in question_ids:
        mode_records: dict[str, dict[str, Any]] = {}

        category = ""
        user_turns: list[str] = []
        source_files: dict[str, str] = {}

        for mode in MODES:
            path, record = find_latest_result(
                question_id,
                mode,
            )

            record["_source_file"] = path.name
            mode_records[mode] = record
            source_files[mode] = path.name

            if not category:
                category = str(
                    record.get("category", "")
                )

            if not user_turns:
                user_turns = [
                    str(turn.get("prompt", ""))
                    for turn in record.get(
                        "turns",
                        [],
                    )
                ]

        shuffled_modes = MODES.copy()

        question_rng = random.Random(
            stable_question_seed(question_id)
        )
        question_rng.shuffle(shuffled_modes)

        label_to_mode = dict(
            zip(LABELS, shuffled_modes)
        )

        anonymous_answers: dict[
            str,
            dict[str, Any],
        ] = {}

        for label, mode in label_to_mode.items():
            anonymous_answers[label] = (
                extract_conversation(
                    mode_records[mode]
                )
            )

            # Judge 文件中不暴露原始文件名。
            anonymous_answers[label].pop(
                "source_file",
                None,
            )

        judge_record = {
            "question_id": question_id,
            "category": category,
            "user_turns": user_turns,
            "answers": anonymous_answers,
        }

        judge_records.append(judge_record)

        answer_key["questions"][str(question_id)] = {
            "label_to_mode": label_to_mode,
            "mode_to_label": {
                mode: label
                for label, mode
                in label_to_mode.items()
            },
            "source_files": source_files,
        }

    with JUDGE_INPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in judge_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    ANSWER_KEY_FILE.write_text(
        json.dumps(
            answer_key,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"完整题目数：{len(judge_records)}"
    )
    print(
        f"Judge 输入：{JUDGE_INPUT_FILE}"
    )
    print(
        f"答案映射：{ANSWER_KEY_FILE}"
    )

    for record in judge_records:
        question_id = str(record["question_id"])
        mapping = answer_key[
            "questions"
        ][question_id]["label_to_mode"]

        print(
            f"Question {question_id}: "
            f"A/B/C 已匿名化；"
            f"真实映射仅保存在 answer key"
        )


if __name__ == "__main__":
    main()
