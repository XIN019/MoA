import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUNNER_FILE = BASE_DIR / "run_mt_bench.py"
RESULT_DIR = BASE_DIR / "results"
BATCH_LOG_DIR = RESULT_DIR / "batch_logs"

DEFAULT_QUESTION_IDS = [91, 131, 141, 151]
DEFAULT_MODES = ["single", "two-layer", "three-layer"]


def result_exists(question_id: int, mode: str) -> bool:
    """判断某道题的某种模式是否已有成功保存的结果。"""
    pattern = f"q{question_id}_{mode}_*.jsonl"
    return any(RESULT_DIR.glob(pattern))


def run_one(
    question_id: int,
    mode: str,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUNNER_FILE),
        "--mode",
        mode,
        "--question-id",
        str(question_id),
    ]

    print("\n" + "=" * 72)
    print(f"Question {question_id} | Mode {mode}")
    print("=" * 72)
    print("命令：", " ".join(command))

    if dry_run:
        return {
            "question_id": question_id,
            "mode": mode,
            "status": "dry_run",
            "return_code": None,
        }

    completed = subprocess.run(
        command,
        cwd=BASE_DIR.parents[1],
        check=False,
    )

    status = (
        "success"
        if completed.returncode == 0
        else "failed"
    )

    return {
        "question_id": question_id,
        "mode": mode,
        "status": status,
        "return_code": completed.returncode,
    }


def save_batch_log(
    records: list[dict[str, Any]],
) -> Path:
    BATCH_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        BATCH_LOG_DIR
        / f"pilot_batch_{timestamp}.json"
    )

    payload = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "records": records,
        "success_count": sum(
            record["status"] == "success"
            for record in records
        ),
        "failed_count": sum(
            record["status"] == "failed"
            for record in records
        ),
        "skipped_count": sum(
            record["status"] == "skipped_existing"
            for record in records
        ),
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "批量运行 MT-Bench pilot，"
            "支持跳过已有结果和失败后继续。"
        )
    )

    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="+",
        default=DEFAULT_QUESTION_IDS,
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=DEFAULT_MODES,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有结果也重新运行",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示计划，不调用 API",
    )

    args = parser.parse_args()

    records: list[dict[str, Any]] = []

    print("计划题号：", args.question_ids)
    print("计划模式：", args.modes)
    print("强制重跑：", args.force)
    print("Dry-run：", args.dry_run)

    for question_id in args.question_ids:
        for mode in args.modes:
            if (
                not args.force
                and result_exists(question_id, mode)
            ):
                print(
                    f"跳过已有结果："
                    f"Question {question_id}, {mode}"
                )

                records.append(
                    {
                        "question_id": question_id,
                        "mode": mode,
                        "status": "skipped_existing",
                        "return_code": None,
                    }
                )
                continue

            record = run_one(
                question_id,
                mode,
                dry_run=args.dry_run,
            )

            records.append(record)

            if record["status"] == "failed":
                print(
                    f"运行失败，但继续后续任务："
                    f"Question {question_id}, {mode}"
                )

    log_path = save_batch_log(records)

    print("\n" + "-" * 72)
    print("批量运行结束")
    print(
        "成功：",
        sum(
            record["status"] == "success"
            for record in records
        ),
    )
    print(
        "失败：",
        sum(
            record["status"] == "failed"
            for record in records
        ),
    )
    print(
        "跳过：",
        sum(
            record["status"] == "skipped_existing"
            for record in records
        ),
    )
    print("批次日志：", log_path)


if __name__ == "__main__":
    main()