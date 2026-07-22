import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

QUESTION_CSV = RESULT_DIR / "question_results.csv"
MODE_CSV = RESULT_DIR / "mode_summary.csv"
REPORT_MD = RESULT_DIR / "pilot_summary.md"

MODE_ORDER = {
    "single": 0,
    "two-layer": 1,
    "three-layer": 2,
}


def load_record(path: Path) -> dict[str, Any]:
    """读取单条 JSONL 实验结果。"""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)

    raise ValueError(f"结果文件为空：{path}")


def select_latest_results() -> list[tuple[Path, dict[str, Any]]]:
    """
    每个 question_id 和 mode 只保留最新一次运行结果。
    """
    latest: dict[
        tuple[int, str],
        tuple[float, Path, dict[str, Any]],
    ] = {}

    for path in RESULT_DIR.glob("q*_*.jsonl"):
        try:
            record = load_record(path)

            question_id = int(record["question_id"])
            mode = str(record["mode"])
            key = (question_id, mode)

            modified_time = path.stat().st_mtime

            previous = latest.get(key)

            if previous is None or modified_time > previous[0]:
                latest[key] = (
                    modified_time,
                    path,
                    record,
                )

        except Exception as exc:
            print(f"跳过无法解析的文件 {path.name}: {exc}")

    selected = [
        (path, record)
        for _, path, record in latest.values()
    ]

    selected.sort(
        key=lambda item: (
            int(item[1]["question_id"]),
            MODE_ORDER.get(str(item[1]["mode"]), 99),
        )
    )

    return selected


def safe_round(
    value: float | int | None,
    digits: int = 2,
) -> float | str:
    if value is None:
        return ""

    return round(float(value), digits)


def build_question_rows(
    selected: list[tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path, record in selected:
        turns = record.get("turns", [])

        final_output_chars = sum(
            len(
                turn.get(
                    "final_result",
                    {},
                ).get("content", "")
            )
            for turn in turns
        )

        logical_requests = record.get(
            "logical_request_count",
            record.get("request_count"),
        )

        actual_api_requests = record.get(
            "actual_api_request_count"
        )

        retry_count = record.get("retry_count")

        stats_version = (
            "detailed"
            if actual_api_requests is not None
            else "legacy"
        )

        rows.append(
            {
                "question_id": int(record["question_id"]),
                "category": record.get("category", ""),
                "mode": record["mode"],
                "turn_count": len(turns),
                "logical_request_count": logical_requests,
                "actual_api_request_count": (
                    actual_api_requests
                    if actual_api_requests is not None
                    else ""
                ),
                "retry_count": (
                    retry_count
                    if retry_count is not None
                    else ""
                ),
                "wall_time_seconds": safe_round(
                    record.get("total_duration_seconds")
                ),
                "api_attempt_time_seconds": safe_round(
                    record.get(
                        "api_attempt_duration_seconds"
                    )
                ),
                "final_output_chars": final_output_chars,
                "stats_version": stats_version,
                "timestamp_utc": record.get(
                    "timestamp_utc",
                    "",
                ),
                "source_file": path.name,
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("没有可写入的结果")

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def average_known(
    rows: list[dict[str, Any]],
    field: str,
) -> float | str:
    values = [
        float(row[field])
        for row in rows
        if row[field] != ""
    ]

    if not values:
        return ""

    return round(mean(values), 2)


def build_mode_rows(
    question_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in question_rows:
        grouped[str(row["mode"])].append(row)

    mode_rows: list[dict[str, Any]] = []

    for mode in sorted(
        grouped,
        key=lambda item: MODE_ORDER.get(item, 99),
    ):
        rows = grouped[mode]

        known_retry_values = [
            int(row["retry_count"])
            for row in rows
            if row["retry_count"] != ""
        ]

        categories = sorted(
            {
                str(row["category"])
                for row in rows
                if row["category"]
            }
        )

        mode_rows.append(
            {
                "mode": mode,
                "question_count": len(rows),
                "categories": ",".join(categories),
                "avg_wall_time_seconds": average_known(
                    rows,
                    "wall_time_seconds",
                ),
                "avg_logical_requests": average_known(
                    rows,
                    "logical_request_count",
                ),
                "avg_actual_api_requests_detailed_only": average_known(
                    rows,
                    "actual_api_request_count",
                ),
                "total_known_retries": (
                    sum(known_retry_values)
                    if known_retry_values
                    else ""
                ),
                "avg_final_output_chars": average_known(
                    rows,
                    "final_output_chars",
                ),
                "detailed_stats_count": sum(
                    row["stats_version"] == "detailed"
                    for row in rows
                ),
                "legacy_stats_count": sum(
                    row["stats_version"] == "legacy"
                    for row in rows
                ),
            }
        )

    return mode_rows


def markdown_value(value: Any) -> str:
    if value == "":
        return "N/A"

    return str(value)


def write_markdown_report(
    question_rows: list[dict[str, Any]],
    mode_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# MT-Bench Pilot 汇总",
        "",
        "## 模式级汇总",
        "",
        (
            "| 模式 | 题目数 | 平均墙钟耗时 | "
            "平均逻辑请求 | 平均实际请求 | "
            "已知重试数 | 详细统计数 | 旧统计数 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in mode_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(row["mode"]),
                    markdown_value(row["question_count"]),
                    markdown_value(
                        row["avg_wall_time_seconds"]
                    ),
                    markdown_value(
                        row["avg_logical_requests"]
                    ),
                    markdown_value(
                        row["avg_actual_api_requests_detailed_only"]
                    ),
                    markdown_value(
                        row["total_known_retries"]
                    ),
                    markdown_value(
                        row["detailed_stats_count"]
                    ),
                    markdown_value(
                        row["legacy_stats_count"]
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 题目级结果",
            "",
            (
                "| 题号 | 类别 | 模式 | 逻辑请求 | "
                "实际请求 | 重试 | 墙钟耗时 | "
                "输出字符数 | 统计版本 |"
            ),
            (
                "|---:|---|---|---:|---:|---:|"
                "---:|---:|---|"
            ),
        ]
    )

    for row in question_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_value(row["question_id"]),
                    markdown_value(row["category"]),
                    markdown_value(row["mode"]),
                    markdown_value(
                        row["logical_request_count"]
                    ),
                    markdown_value(
                        row["actual_api_request_count"]
                    ),
                    markdown_value(row["retry_count"]),
                    markdown_value(
                        row["wall_time_seconds"]
                    ),
                    markdown_value(
                        row["final_output_chars"]
                    ),
                    markdown_value(row["stats_version"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 统计说明",
            "",
            (
                "- `detailed`：包含逻辑请求数、真实 API "
                "请求数和重试次数。"
            ),
            (
                "- `legacy`：修改统计代码前产生的旧结果，"
                "只能确认逻辑请求数和总耗时。"
            ),
            (
                "- 并行 Reference Models 会导致所有 API "
                "尝试耗时之和大于实际墙钟耗时。"
            ),
            (
                "- 当前表格只统计运行成本，不代表回答质量。"
            ),
            "",
        ]
    )

    REPORT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_latest_results()

    if not selected:
        raise RuntimeError(
            f"未在 {RESULT_DIR} 中找到实验结果"
        )

    question_rows = build_question_rows(selected)
    mode_rows = build_mode_rows(question_rows)

    write_csv(QUESTION_CSV, question_rows)
    write_csv(MODE_CSV, mode_rows)
    write_markdown_report(question_rows, mode_rows)

    print(f"选取最新结果数：{len(selected)}")
    print(f"题目级汇总：{QUESTION_CSV}")
    print(f"模式级汇总：{MODE_CSV}")
    print(f"Markdown 报告：{REPORT_MD}")

    print("\n模式级结果：")

    for row in mode_rows:
        print(
            f"{row['mode']}: "
            f"questions={row['question_count']}, "
            f"avg_wall={row['avg_wall_time_seconds']}, "
            f"avg_logical={row['avg_logical_requests']}, "
            f"avg_actual={row['avg_actual_api_requests']}, "
            f"detailed={row['detailed_stats_count']}, "
            f"legacy={row['legacy_stats_count']}"
        )


if __name__ == "__main__":
    main()
