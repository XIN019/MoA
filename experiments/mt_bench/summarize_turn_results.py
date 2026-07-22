from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

OUTPUT_CSV = RESULT_DIR / "turn_question_results.csv"
SUMMARY_CSV = RESULT_DIR / "turn_mode_summary.csv"
REPORT_FILE = RESULT_DIR / "turn_summary.md"

QUESTION_IDS = [
    81, 82,
    91, 92,
    101, 102,
    111, 112,
    121, 122,
    131, 132,
    141, 142,
    151, 152,
]

MODES = [
    "single",
    "two-layer",
    "three-layer",
]

MODE_ORDER = {
    "single": 0,
    "two-layer": 1,
    "three-layer": 2,
}


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
) -> Path:
    candidates = sorted(
        RESULT_DIR.glob(
            f"q{question_id}_{mode}_*.jsonl"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"缺少 Question {question_id}, mode={mode}"
        )

    return candidates[0]


def optional_number(
    value: Any,
) -> int | float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    return None


def format_number(
    value: Any,
    digits: int = 2,
) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, int):
        return str(value)

    return str(round(float(value), digits))


def available_mean(
    values: list[int | float | None],
) -> float | None:
    available = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available:
        return None

    return round(mean(available), 2)


def available_median(
    values: list[int | float | None],
) -> float | None:
    available = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available:
        return None

    return round(median(available), 2)


def main() -> None:
    rows: list[dict[str, Any]] = []

    for question_id in QUESTION_IDS:
        for mode in MODES:
            path = find_latest_result(
                question_id,
                mode,
            )
            record = load_jsonl_record(path)

            category = str(
                record.get("category", "")
            )

            for turn in record.get("turns", []):
                turn_number = int(
                    turn["turn_number"]
                )

                final_result = turn.get(
                    "final_result",
                    {},
                )

                content = str(
                    final_result.get(
                        "content",
                        "",
                    )
                )

                logical_requests = optional_number(
                    turn.get(
                        "logical_request_count"
                    )
                )
                actual_requests = optional_number(
                    turn.get(
                        "actual_api_request_count"
                    )
                )
                retry_count = optional_number(
                    turn.get("retry_count")
                )
                wall_time = None

                candidate_sources = [
                    turn,
                    turn.get("statistics", {}),
                ]

                candidate_fields = [
                    "wall_time_seconds",
                    "wall_time",
                    "turn_duration_seconds",
                    "total_duration_seconds",
                    "duration_seconds",
                ]

                for source in candidate_sources:
                    if not isinstance(source, dict):
                        continue

                    for field_name in candidate_fields:
                        candidate = optional_number(
                            source.get(field_name)
                        )

                        if candidate is not None:
                            wall_time = candidate
                            break

                    if wall_time is not None:
                        break

                is_detailed = (
                    actual_requests is not None
                    and retry_count is not None
                )

                is_timing_enabled = "wall_time_seconds" in turn and turn["wall_time_seconds"] is not None

                rows.append(
                    {
                        "question_id": question_id,
                        "category": category,
                        "mode": mode,
                        "turn_number": turn_number,
                        "logical_requests": logical_requests,
                        "actual_api_requests": actual_requests,
                        "retry_count": retry_count,
                        "wall_time_seconds": wall_time,
                        "output_chars": len(content),
                        "is_detailed": is_detailed,
                        "is_timing_enabled": is_timing_enabled,
                        "source_file": path.name,
                    }
                )

    rows.sort(
        key=lambda row: (
            row["question_id"],
            MODE_ORDER[row["mode"]],
            row["turn_number"],
        )
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
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

    grouped: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["mode"],
                row["turn_number"],
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []

    for mode in MODES:
        for turn_number in [1, 2]:
            items = grouped[
                (mode, turn_number)
            ]

            retries = [
                item["retry_count"]
                for item in items
                if item["retry_count"] is not None
            ]

            summary_rows.append(
                {
                    "mode": mode,
                    "turn_number": turn_number,
                    "question_count": len(items),
                    "avg_wall_time_seconds": (
                        available_mean(
                            [
                                item[
                                    "wall_time_seconds"
                                ]
                                for item in items
                            ]
                        )
                    ),
                    "median_wall_time_seconds": (
                        available_median(
                            [
                                item[
                                    "wall_time_seconds"
                                ]
                                for item in items
                            ]
                        )
                    ),
                    "avg_logical_requests": (
                        available_mean(
                            [
                                item[
                                    "logical_requests"
                                ]
                                for item in items
                            ]
                        )
                    ),
                    "avg_actual_api_requests": (
                        available_mean(
                            [
                                item[
                                    "actual_api_requests"
                                ]
                                for item in items
                            ]
                        )
                    ),
                    "known_retry_count": (
                        int(sum(retries))
                        if retries
                        else None
                    ),
                    "avg_output_chars": round(
                        mean(
                            item["output_chars"]
                            for item in items
                        ),
                        2,
                    ),
                    "detailed_count": sum(
                        item["is_detailed"]
                        for item in items
                    ),
                    "legacy_count": sum(
                        not item["is_detailed"]
                        for item in items
                    ),
                    "timing_enabled_count": sum(
                        item["is_timing_enabled"]
                        for item in items
                    ),
                }
            )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# MT-Bench Turn-level 成本汇总",
        "",
        "## 模式与轮次汇总",
        "",
        (
            "| 模式 | 轮次 | 题目数 | "
            "平均耗时 | 中位数耗时 | "
            "平均逻辑请求 | 平均实际请求 | "
            "已知重试 | 平均输出字符数 | "
            "有详细统计 | 有耗时记录 |"
        ),
        (
            "|---|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ]

    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    str(row["turn_number"]),
                    str(row["question_count"]),
                    format_number(
                        row[
                            "avg_wall_time_seconds"
                        ]
                    ),
                    format_number(
                        row[
                            "median_wall_time_seconds"
                        ]
                    ),
                    format_number(
                        row[
                            "avg_logical_requests"
                        ]
                    ),
                    format_number(
                        row[
                            "avg_actual_api_requests"
                        ]
                    ),
                    format_number(
                        row[
                            "known_retry_count"
                        ]
                    ),
                    format_number(
                        row["avg_output_chars"]
                    ),
                    str(row["detailed_count"]),
                    str(row["timing_enabled_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            (
                "- Turn 1 和 Turn 2 的耗时、请求数、"
                "重试数均从已有结果文件提取。"
            ),
            (
                "- **historical/legacy**：缺少 actual_api_request_count "
                "或 retry_count，对应列显示 N/A。"
            ),
            (
                "- **detailed**：包含完整的 "
                "actual_api_request_count 和 retry_count。"
            ),
            (
                "- **timing-enabled**：包含分轮 wall_time_seconds。"
                "新记录可以同时为 detailed 和 timing-enabled；"
                "历史 legacy 记录缺少分轮耗时，继续显示 N/A。"
            ),
            (
                "- 当前报告只统计运行成本；已有 Judge "
                "结果是整题评分，不能直接拆分到单轮。"
            ),
            "",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"题目级文件：{OUTPUT_CSV}")
    print(f"汇总文件：{SUMMARY_CSV}")
    print(f"Markdown 报告：{REPORT_FILE}")


if __name__ == "__main__":
    main()