from __future__ import annotations

import csv
import json
from pathlib import Path

from common import ROOT


PHASE2 = ROOT / "experiments" / "dim_384_vs_768" / "phase2"
RAW = PHASE2 / "results" / "raw"
AUDIT = PHASE2 / "audit"
ANNOTATION = PHASE2 / "annotation"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_kappa(first: list[int], second: list[int]) -> float:
    if not first:
        return 0.0
    categories = range(4)
    observed = sum(1 - abs(a - b) / 3 for a, b in zip(first, second)) / len(first)
    first_counts = [first.count(category) for category in categories]
    second_counts = [second.count(category) for category in categories]
    expected = sum(
        (first_counts[a] / len(first)) * (second_counts[b] / len(second)) * (1 - abs(a - b) / 3)
        for a in categories for b in categories
    )
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


def kappa_section() -> str:
    path = ANNOTATION / "annotation_2.csv"
    if not path.exists():
        return "Chưa có `annotation_2.csv`; Cohen's kappa có trọng số chưa được tính."
    first = {row["item_id"]: row for row in read_csv(ANNOTATION / "annotation_sheet.csv")}
    second = {row["item_id"]: row for row in read_csv(path)}
    common = sorted(set(first) & set(second))
    pairs = [
        (int(first[item]["label"]), int(second[item]["label"]))
        for item in common
        if first[item]["label"] in {"0", "1", "2", "3"} and second[item]["label"] in {"0", "1", "2", "3"}
    ]
    return f"Cohen's kappa có trọng số: `{weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]):.6f}` trên `{len(pairs)}` mục chung."


def main() -> None:
    stats = json.loads((RAW / "paired_stats.json").read_text(encoding="utf-8"))
    overlap = read_csv(RAW / "overlap_at10.csv")
    counts = json.loads((AUDIT / "annotation_counts.json").read_text(encoding="utf-8"))
    discrepancy = json.loads((AUDIT / "readme_discrepancy.json").read_text(encoding="utf-8"))
    overlap_summary = {}
    for method in ("dense", "bm25_dense"):
        values = [float(row["overlap_at_10"]) for row in overlap if row["method"] == method]
        overlap_summary[method] = sum(values) / len(values)
    lines = [
        "# Báo cáo Phase 2: phân tích exact và chấm mù",
        "",
        "## Bước 0 - Điểm lệch với README",
        "",
        f"- README ghi nDCG@10 BM25+Dense: `{discrepancy['readme_expected_bm25_dense_ndcg_at_10']}`.",
        f"- Đường cơ sở tái tạo: `{discrepancy['phase1_recreated_bm25_dense_ndcg_at_10']}`.",
        "- Bằng chứng:",
        *[f"  - {item}" for item in discrepancy["evidence"]],
        f"- Kết luận: {discrepancy['conclusion']}",
        "",
        "## Phần A - Exact dense search",
        "",
        "Dense được xếp hạng bằng tích vô hướng cosine trên toàn bộ 8.670 vector đã lưu; không dùng HNSW gần đúng. Đoạn đúng được xác định minh bạch bằng cùng quy tắc `relevant_terms` của evaluator hiện có trên toàn kho, vì golden dataset không chứa chunk ID độc lập.",
        "",
        "| Chỉ số ghép cặp | Mean delta (768-384 hoặc lợi thế hạng) | CI95% | Tốt hơn | Xấu hơn | Không đổi | Kiểm định | p |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for metric, item in stats.items():
        direction = "768-384" if metric in {"mrr_full_corpus", "mean_relevant_z"} else "384-768 (hạng thấp hơn là tốt hơn)"
        lines.append(
            f"| {metric} ({direction}) | {item['mean_delta']:.6f} | [{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] | "
            f"{item['improved']} | {item['worsened']} | {item['unchanged']} | {item['test']['name']} | {item['test']['p_value']:.6f} |"
        )
    lines += [
        "",
        "### Overlap@10",
        "",
        f"- Dense-only: trung bình `{overlap_summary['dense']:.6f}`.",
        f"- BM25+Dense: trung bình `{overlap_summary['bm25_dense']:.6f}`.",
        "- Chi tiết từng truy vấn nằm trong `results/raw/overlap_at10.csv`.",
        "",
        "## Phần B - Bảng chấm thủ công mù",
        "",
        f"- Bản đầy đủ: `{counts['all_items']}` mục tại `annotation/annotation_sheet.csv`.",
        f"- Bản rút gọn khác biệt: `{counts['short_items']}` mục tại `annotation/annotation_sheet_short.csv`.",
        "- Vì bản đầy đủ vượt 150 mục, báo cáo cung cấp cả hai bản để người dùng chọn; chưa tự động chọn bản nào.",
        "- Không có cột mô hình, hạng, điểm số hoặc collection trong bảng chấm.",
        "- Khóa ánh xạ được tách riêng tại `annotation/KEY_do_not_open.json`.",
        "",
        "### Rubric",
        "",
        "- 0 = không liên quan; 1 = cùng chủ đề nhưng không trả lời; 2 = trả lời một phần; 3 = trả lời trực tiếp, đúng công ty, đúng kỳ, đúng chỉ tiêu.",
        "- `wrong_entity_or_period=1` nếu đúng chủ đề nhưng sai công ty hoặc sai kỳ.",
        "",
        kappa_section(),
        "",
        "## Phần C - Phân tích sau khi có nhãn người",
        "",
        "Chưa chạy. Đang chờ người chấm điền `label` và `wrong_entity_or_period`; không tự điền nhãn.",
        "",
        "## Giới hạn",
        "",
        "- Chỉ có 25 truy vấn, chạy trên một máy.",
        "- Phân tích exact dùng quy tắc relevance từ `relevant_terms`, không phải ground truth chunk-level độc lập.",
        "- Bảng chấm mới chỉ gồm các đoạn thuộc top-10 của các hệ thống.",
        "- Kết quả nhãn người chưa tồn tại; do đó chưa thể kết luận về nDCG theo nhãn người, Spearman hay mức bỏ sót của golden set.",
        "",
        "## Kết luận tạm thời",
        "",
        "Các CI chứa 0 phải được đọc là không đủ bằng chứng về khác biệt. Phần C chỉ được thực hiện sau khi người dùng xác nhận đã điền xong nhãn.",
    ]
    (PHASE2 / "REPORT_PHASE2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
