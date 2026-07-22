import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULT_DIR / "q121_three_layer_trace.md"


def load_latest_result() -> tuple[Path, dict[str, Any]]:
    files = sorted(
        RESULT_DIR.glob("q121_three-layer_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise RuntimeError("没有找到 q121 Three-layer 结果")

    path = files[0]

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            return path, json.loads(line)

    raise RuntimeError(f"结果文件为空：{path}")


def code_fence(text: str) -> str:
    return f"```text\n{text}\n```"


def main() -> None:
    source_path, record = load_latest_result()

    lines = [
        "# Question 121 Three-layer 输出追踪",
        "",
        f"源文件：`{source_path.name}`",
        "",
    ]

    for turn in record.get("turns", []):
        turn_number = turn["turn_number"]

        lines.extend(
            [
                f"## Turn {turn_number}",
                "",
                "### 用户问题",
                "",
                code_fence(turn.get("prompt", "")),
                "",
            ]
        )

        for layer in turn.get("layers", []):
            layer_number = layer["layer_number"]

            lines.extend(
                [
                    f"### Reference Layer {layer_number}",
                    "",
                ]
            )

            for index, response in enumerate(
                layer.get("responses", []),
                start=1,
            ):
                model = response.get(
                    "model",
                    f"model-{index}",
                )

                lines.extend(
                    [
                        f"#### {index}. {model}",
                        "",
                        code_fence(
                            response.get("content", "")
                        ),
                        "",
                    ]
                )

        final_result = turn.get("final_result", {})

        lines.extend(
            [
                "### Final Aggregator",
                "",
                code_fence(
                    final_result.get("content", "")
                ),
                "",
            ]
        )

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"源结果：{source_path}")
    print(f"追踪报告：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
