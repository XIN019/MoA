import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
RUNNER_FILE = Path(__file__).resolve().parent / "run_turn_judge.py"
DEFAULT_RESULT_DIR = BASE_DIR / "results" / "judge_turn"
BATCH_LOG_DIR = DEFAULT_RESULT_DIR / "batch_logs"

DEFAULT_QUESTION_IDS = [
    81, 82, 91, 92,
    101, 102, 111, 112,
    121, 122, 131, 132,
    141, 142, 151, 152,
]


def result_exists(output_dir: Path, question_id: int, turn_number: int) -> bool:
    return any(output_dir.glob(f"q{question_id}_turn{turn_number}_judge_*.json"))


def run_one(
    question_id: int,
    turn_number: int,
    *,
    dry_run: bool,
    input_dir: str,
    output_dir: str,
    skip_existing: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER_FILE),
        "--question-id", str(question_id),
        "--turn-number", str(turn_number),
        "--input-file", input_dir,
        "--output-dir", output_dir,
    ]

    if dry_run:
        command.append("--dry-run")

    print("\n" + "=" * 72)
    print(f"Q{question_id} | Turn {turn_number}")
    print("=" * 72)

    if skip_existing and result_exists(Path(output_dir), question_id, turn_number):
        print(f"跳过已有结果：Q{question_id} Turn {turn_number}")
        return {
            "question_id": question_id,
            "turn_number": turn_number,
            "status": "skipped_existing",
            "return_code": None,
        }

    completed = subprocess.run(command, cwd=BASE_DIR, check=False)

    status = "success" if completed.returncode == 0 else "failed"

    return {
        "question_id": question_id,
        "turn_number": turn_number,
        "status": status,
        "return_code": completed.returncode,
    }


def save_batch_log(records: list[dict[str, Any]], output_dir: Path) -> Path:
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = BATCH_LOG_DIR / f"turn_batch_{timestamp}.json"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "records": records,
        "success_count": sum(r["status"] == "success" for r in records),
        "failed_count": sum(r["status"] == "failed" for r in records),
        "skipped_count": sum(r["status"] == "skipped_existing" for r in records),
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-ids", type=int, nargs="+", default=DEFAULT_QUESTION_IDS)
    parser.add_argument("--turns", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULT_DIR
    input_dir = args.input_dir if args.input_dir else str(
        BASE_DIR / "judge" / "data" / "turn_judge"
    )

    records: list[dict[str, Any]] = []

    for question_id in args.question_ids:
        for turn_number in args.turns:
            record = run_one(
                question_id=question_id,
                turn_number=turn_number,
                dry_run=args.dry_run,
                input_dir=input_dir,
                output_dir=str(output_dir),
                skip_existing=args.skip_existing,
            )
            records.append(record)

            if record["status"] == "failed":
                print(
                    f"运行失败，继续后续任务："
                    f"Q{question_id} Turn {turn_number}"
                )

    log_path = save_batch_log(records, output_dir)

    success = sum(r["status"] == "success" for r in records)
    failed = sum(r["status"] == "failed" for r in records)
    skipped = sum(r["status"] == "skipped_existing" for r in records)

    print("\n" + "-" * 72)
    print("批量运行结束")
    print(f"成功：{success}  失败：{failed}  跳过：{skipped}")
    print("批次日志：", log_path)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
