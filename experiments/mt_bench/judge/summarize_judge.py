import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MT_BENCH_DIR = Path(__file__).resolve().parents[1]

JUDGE_RESULT_DIR = (
    MT_BENCH_DIR
    / "results"
    / "judge"
)

ANSWER_KEY_FILE = (
    MT_BENCH_DIR
    / "judge"
    / "data"
    / "judge_answer_key.json"
)

OUTPUT_CSV = (
    JUDGE_RESULT_DIR
    / "judge_question_results.csv"
)

MODE_SUMMARY_CSV = (
    JUDGE_RESULT_DIR
    / "judge_mode_summary.csv"
)

REPORT_FILE = (
    JUDGE_RESULT_DIR
    / "judge_summary.md"
)

MODE_ORDER = {
    "single": 0,
    "two-layer": 1,
    "three-layer": 2,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def select_latest_results() -> dict[int, dict[str, Any]]:
    latest: dict[
        int,
        tuple[float, dict[str, Any]],
    ] = {}

    for path in JUDGE_RESULT_DIR.glob(
        "q*_judge_*.json"
    ):
        try:
            record = load_json(path)
            question_id = int(record["question_id"])
            modified_time = path.stat().st_mtime

            previous = latest.get(question_id)

            if (
                previous is None
                or modified_time > previous[0]
            ):
                record["_source_file"] = path.name
                latest[question_id] = (
                    modified_time,
                    record,
                )

        except Exception as exc:
            print(
                f"跳过 {path.name}: {exc}"
            )

    return {
        question_id: record
        for question_id, (_, record)
        in latest.items()
    }


def main() -> None:
    if not ANSWER_KEY_FILE.exists():
        raise FileNotFoundError(
            f"未找到答案映射：{ANSWER_KEY_FILE}"
        )

    answer_key = load_json(ANSWER_KEY_FILE)
    judge_results = select_latest_results()

    if not judge_results:
        raise RuntimeError(
            "没有找到 Judge 结果"
        )

    rows: list[dict[str, Any]] = []

    for question_id in sorted(judge_results):
        result = judge_results[question_id]

        mapping = answer_key[
            "questions"
        ][str(question_id)]["label_to_mode"]

        evaluations = result[
            "judgment"
        ]["evaluations"]

        ranking = result[
            "judgment"
        ]["ranking"]

        mode_ranking = [
            mapping[label]
            for label in ranking
        ]

        for label in ["A", "B", "C"]:
            evaluation = evaluations[label]
            mode = mapping[label]

            rows.append(
                {
                    "question_id": question_id,
                    "category": result.get(
                        "category",
                        "",
                    ),
                    "mode": mode,
                    "anonymous_label": label,
                    "correctness": evaluation[
                        "correctness"
                    ],
                    "instruction_following": evaluation[
                        "instruction_following"
                    ],
                    "relevance": evaluation[
                        "relevance"
                    ],
                    "clarity": evaluation[
                        "clarity"
                    ],
                    "overall_score": evaluation[
                        "overall_score"
                    ],
                    "major_error": evaluation[
                        "major_error"
                    ],
                    "major_error_description": evaluation[
                        "major_error_description"
                    ],
                    "mode_ranking": " > ".join(
                        mode_ranking
                    ),
                    "judge_model": result[
                        "judge_model"
                    ],
                    "judge_duration_seconds": result[
                        "duration_seconds"
                    ],
                    "source_file": result[
                        "_source_file"
                    ],
                }
            )

    rows.sort(
        key=lambda row: (
            row["question_id"],
            MODE_ORDER.get(
                row["mode"],
                99,
            ),
        )
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
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        grouped[row["mode"]].append(row)

    mode_rows: list[dict[str, Any]] = []

    for mode in sorted(
        grouped,
        key=lambda item: MODE_ORDER.get(
            item,
            99,
        ),
    ):
        mode_items = grouped[mode]

        mode_rows.append(
            {
                "mode": mode,
                "question_count": len(mode_items),
                "avg_correctness": round(
                    mean(
                        item["correctness"]
                        for item in mode_items
                    ),
                    2,
                ),
                "avg_instruction_following": round(
                    mean(
                        item["instruction_following"]
                        for item in mode_items
                    ),
                    2,
                ),
                "avg_relevance": round(
                    mean(
                        item["relevance"]
                        for item in mode_items
                    ),
                    2,
                ),
                "avg_clarity": round(
                    mean(
                        item["clarity"]
                        for item in mode_items
                    ),
                    2,
                ),
                "avg_overall_score": round(
                    mean(
                        item["overall_score"]
                        for item in mode_items
                    ),
                    2,
                ),
                "major_error_count": sum(
                    bool(item["major_error"])
                    for item in mode_items
                ),
            }
        )

    with MODE_SUMMARY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                mode_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(mode_rows)

    lines = [
        "# MT-Bench Pilot Judge 汇总",
        "",
        "## 模式级结果",
        "",
        (
            "| 模式 | 题目数 | 正确性 | "
            "指令遵循 | 相关性 | 清晰度 | "
            "总体分数 | 关键错误数 |"
        ),
        (
            "|---|---:|---:|---:|---:|"
            "---:|---:|---:|"
        ),
    ]

    for row in mode_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["mode"]),
                    str(row["question_count"]),
                    str(row["avg_correctness"]),
                    str(
                        row[
                            "avg_instruction_following"
                        ]
                    ),
                    str(row["avg_relevance"]),
                    str(row["avg_clarity"]),
                    str(row["avg_overall_score"]),
                    str(row["major_error_count"]),
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
                "| 题号 | 类别 | 模式 | Overall | "
                "Correctness | Major error | 排名 |"
            ),
            (
                "|---:|---|---|---:|---:|---|---|"
            ),
        ]
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["question_id"]),
                    str(row["category"]),
                    str(row["mode"]),
                    str(row["overall_score"]),
                    str(row["correctness"]),
                    str(row["major_error"]),
                    str(row["mode_ranking"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 注意",
            "",
            (
                "- 当前仅包含少量 pilot 题目，"
                "不能视为正式 Benchmark 结论。"
            ),
            (
                "- Coding 题应结合代码执行结果，"
                "不能只依赖 LLM Judge。"
            ),
            (
                "- Judge 分数可能受答案长度和表达风格影响。"
            ),
            "",
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"题目级结果：{OUTPUT_CSV}"
    )
    print(
        f"模式级结果：{MODE_SUMMARY_CSV}"
    )
    print(
        f"Markdown 报告：{REPORT_FILE}"
    )

    print("\n模式级 Judge 结果：")

    for row in mode_rows:
        print(
            f"{row['mode']}: "
            f"overall={row['avg_overall_score']}, "
            f"correctness={row['avg_correctness']}, "
            f"major_errors={row['major_error_count']}"
        )


if __name__ == "__main__":
    main()
