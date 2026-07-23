import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

MT_BENCH_DIR = Path(__file__).resolve().parents[1]

JUDGE_RESULT_DIR = MT_BENCH_DIR / "results" / "judge_turn"
ANSWER_KEY_DIR = MT_BENCH_DIR / "judge" / "data" / "turn_judge"

OUTPUT_CSV = JUDGE_RESULT_DIR / "turn_judge_question_results.csv"
MODE_SUMMARY_CSV = JUDGE_RESULT_DIR / "turn_judge_mode_summary.csv"
REPORT_FILE = JUDGE_RESULT_DIR / "turn_judge_summary.md"

MODE_ORDER = {"single": 0, "two-layer": 1, "three-layer": 2}

SCORE_FIELDS = [
    "correctness",
    "instruction_following",
    "relevance",
    "clarity",
    "overall_score",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_latest_results() -> dict[tuple[int, int], dict[str, Any]]:
    latest: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for path in JUDGE_RESULT_DIR.glob("q*_turn*_judge_*.json"):
        try:
            record = load_json(path)
            question_id = int(record["question_id"])
            turn_number = int(record["judged_turn_number"])
            mtime = path.stat().st_mtime

            key = (question_id, turn_number)
            prev = latest.get(key)
            if prev is None or mtime > prev[0]:
                record["_source_file"] = path.name
                latest[key] = (mtime, record)
        except Exception as exc:
            print(f"跳过 {path.name}: {exc}")

    return {k: v for k, (_, v) in latest.items()}


def load_answer_keys(key_dir: Path) -> dict[int, dict[str, dict[str, str]]]:
    keys: dict[int, dict[str, dict[str, str]]] = {}

    # Per-question answer key files (new format)
    for path in key_dir.glob("q*_turn*_answer_key.json"):
        try:
            data = load_json(path)
            turn_number = data["turn_number"]
            question_id = data["question_id"]
            keys.setdefault(question_id, {})[str(turn_number)] = data["label_to_mode"]
        except Exception as exc:
            print(f"跳过 {path.name}: {exc}")

    # Legacy batch answer key files
    for path in key_dir.glob("turn*_judge_answer_key.json"):
        try:
            data = load_json(path)
            turn_number = data["turn_number"]
            for qid_str, qdata in data.get("questions", {}).items():
                qid = int(qid_str)
                if str(turn_number) not in keys.get(qid, {}):
                    keys.setdefault(qid, {})[str(turn_number)] = qdata["label_to_mode"]
        except Exception as exc:
            print(f"跳过 {path.name}: {exc}")

    return keys


def main() -> None:
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--results-dir", type=str, default=None)
    _parser.add_argument("--answer-key-dir", type=str, default=None)
    _parser.add_argument("--output-dir", type=str, default=None)
    _args = _parser.parse_args()

    global JUDGE_RESULT_DIR, ANSWER_KEY_DIR, OUTPUT_CSV, MODE_SUMMARY_CSV, REPORT_FILE

    if _args.results_dir:
        JUDGE_RESULT_DIR = Path(_args.results_dir)
    if _args.answer_key_dir:
        ANSWER_KEY_DIR = Path(_args.answer_key_dir)
    if _args.output_dir:
        out = Path(_args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV = out / "turn_judge_question_results.csv"
        MODE_SUMMARY_CSV = out / "turn_judge_mode_summary.csv"
        REPORT_FILE = out / "turn_judge_summary.md"

    judge_results = select_latest_results()
    answer_keys = load_answer_keys(ANSWER_KEY_DIR)

    if not judge_results:
        raise RuntimeError("没有找到 Turn Judge 结果文件")

    rows: list[dict[str, Any]] = []

    for (question_id, turn_number), result in sorted(judge_results.items()):
        key_info = answer_keys.get(question_id, {}).get(str(turn_number), {})
        evaluations = result["judgment"]["evaluations"]
        ranking = result["judgment"]["ranking"]

        mode_ranking = [key_info.get(label, f"?({label})") for label in ranking]

        for label in ["A", "B", "C"]:
            evaluation = evaluations[label]
            mode = key_info.get(label, f"?({label})")

            rows.append({
                "question_id": question_id,
                "turn_number": turn_number,
                "category": result.get("category", ""),
                "mode": mode,
                "anonymous_label": label,
                "correctness": evaluation["correctness"],
                "instruction_following": evaluation["instruction_following"],
                "relevance": evaluation["relevance"],
                "clarity": evaluation["clarity"],
                "overall_score": evaluation["overall_score"],
                "major_error": evaluation["major_error"],
                "turn_ranking": " > ".join(mode_ranking),
                "judge_model": result["judge_model"],
                "judge_duration_seconds": result["duration_seconds"],
                "source_file": result["_source_file"],
            })

    rows.sort(key=lambda r: (
        r["question_id"],
        r["turn_number"],
        MODE_ORDER.get(r["mode"], 99),
    ))

    JUDGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # -- mode-level summary per turn --
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["turn_number"])].append(row)

    mode_rows: list[dict[str, Any]] = []
    for mode in ["single", "two-layer", "three-layer"]:
        for turn_number in [1, 2]:
            items = grouped[(mode, turn_number)]
            if not items:
                continue

            mode_rows.append({
                "mode": mode,
                "turn_number": turn_number,
                "question_count": len(items),
                "avg_correctness": round(
                    mean(item["correctness"] for item in items), 2
                ),
                "avg_instruction_following": round(
                    mean(item["instruction_following"] for item in items), 2
                ),
                "avg_relevance": round(
                    mean(item["relevance"] for item in items), 2
                ),
                "avg_clarity": round(
                    mean(item["clarity"] for item in items), 2
                ),
                "avg_overall_score": round(
                    mean(item["overall_score"] for item in items), 2
                ),
                "major_error_count": sum(
                    bool(item["major_error"]) for item in items
                ),
            })

    with MODE_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_rows)

    # -- report --
    lines = [
        "# MT-Bench Turn-Level Judge 汇总",
        "",
        "## 模式 × 轮次汇总",
        "",
        "| 模式 | 轮次 | 题目数 | Overall | Correctness |"
        " Instr.Follow | Relevance | Clarity | Major Err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in mode_rows:
        lines.append(
            "| "
            + " | ".join(map(str, [
                row["mode"], row["turn_number"], row["question_count"],
                row["avg_overall_score"], row["avg_correctness"],
                row["avg_instruction_following"], row["avg_relevance"],
                row["avg_clarity"], row["major_error_count"],
            ]))
            + " |"
        )

    # Turn 2 − Turn 1 delta
    lines.extend([
        "",
        "## Turn 2 − Turn 1 Overall 差值",
        "",
        "| 模式 | Turn 1 Overall | Turn 2 Overall | Delta |",
        "|---|---:|---:|---:|",
    ])

    for mode in ["single", "two-layer", "three-layer"]:
        t1 = mode_rows_by_key(mode_rows, mode, 1)
        t2 = mode_rows_by_key(mode_rows, mode, 2)
        if t1 and t2:
            ov1 = t1["avg_overall_score"] if t1 else None
            ov2 = t2["avg_overall_score"] if t2 else None
            if ov1 is not None and ov2 is not None:
                lines.append(
                    f"| {mode} | {ov1} | {ov2} | {round(ov2 - ov1, 2)} |"
                )

    # per-question per-mode scores
    lines.extend([
        "",
        "## 每题 × 模式分数",
        "",
        "| QID | 类别 | 模式 | Turn | Overall | Major Err |",
        "|---:|---|---|---:|---:|---:|",
    ])

    for row in rows:
        lines.append(
            "| "
            + " | ".join(map(str, [
                row["question_id"], row["category"], row["mode"],
                row["turn_number"], row["overall_score"],
                row["major_error"],
            ]))
            + " |"
        )

    # per-question ranking
    question_ranking: dict[int, dict[int, dict[str, list[str]]]] = \
        defaultdict(lambda: defaultdict(dict))
    for row in rows:
        qid = row["question_id"]
        tn = row["turn_number"]
        question_ranking[qid][tn] = row["turn_ranking"]

    lines.extend([
        "",
        "## 每题匿名排名",
        "",
        "| QID | Turn 1 Ranking | Turn 2 Ranking |",
        "|---:|---|---|",
    ])

    for qid in sorted(question_ranking):
        r1 = question_ranking[qid].get(1, "")
        r2 = question_ranking[qid].get(2, "")
        if isinstance(r1, str) and isinstance(r2, str):
            lines.append(f"| {qid} | {r1} | {r2} |")
        elif isinstance(r1, str):
            lines.append(f"| {qid} | {r1} | N/A |")
        else:
            lines.append(f"| {qid} | N/A | {r2} |")

    lines.extend([
        "",
        "## 注意",
        "",
        "- Turn 1 和 Turn 2 分别独立评测，Judge 仅对指定轮次打分。",
        "- Turn 2 评分包含完整第一轮上下文检查。",
        "- 匿名映射与整题 Judge 使用相同的可复现 seed。",
        "- 当前仅包含 pilot 题目，不能视为正式 Benchmark 结论。",
        "",
    ])

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"题目级 CSV：{OUTPUT_CSV}")
    print(f"模式汇总 CSV：{MODE_SUMMARY_CSV}")
    print(f"Markdown 报告：{REPORT_FILE}")

    print("\n模式 × 轮次 汇总：")
    for row in mode_rows:
        print(
            f"  {row['mode']} Turn {row['turn_number']}: "
            f"overall={row['avg_overall_score']}, "
            f"N={row['question_count']}, "
            f"major_errors={row['major_error_count']}"
        )


def mode_rows_by_key(
    mode_rows: list[dict[str, Any]], mode: str, turn_number: int
) -> dict[str, Any] | None:
    for row in mode_rows:
        if row["mode"] == mode and row["turn_number"] == turn_number:
            return row
    return None


if __name__ == "__main__":
    main()
