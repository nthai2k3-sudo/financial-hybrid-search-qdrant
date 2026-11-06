from __future__ import annotations

import csv
import json
from pathlib import Path

from common import AUDIT, EXPERIMENT, RAW


def main() -> None:
    summary = list(csv.DictReader((RAW.parent / "summary.csv").open(encoding="utf-8")))
    integrity = json.loads((AUDIT / "integrity.json").read_text(encoding="utf-8"))
    comparison = json.loads((AUDIT / "comparison_checks.json").read_text(encoding="utf-8"))
    baseline = json.loads((AUDIT / "baseline_reproduction.json").read_text(encoding="utf-8"))
    cross = json.loads((AUDIT / "cross_check.json").read_text(encoding="utf-8"))
    build = json.loads((AUDIT / "build_768_result.json").read_text(encoding="utf-8"))
    latency_path = AUDIT / "latency.json"
    lines = [
        "# Báo cáo thí nghiệm BGE 384 vs BGE 768",
        "",
        "## Mục tiêu và giả thuyết",
        "",
        "So sánh có đối chứng `BAAI/bge-small-en-v1.5` (384) với `BAAI/bge-base-en-v1.5` (768) trên cùng dữ liệu và 25 truy vấn. H0: không có khác biệt. Chỉ kết luận có ý nghĩa khi CI bootstrap ghép cặp 95% không chứa 0.",
        "",
        "## Dấu vết và kiểm tra toàn vẹn",
        "",
        f"- Dense 768 embedding seconds: `{build['embedding_seconds']}`.",
        f"- Số điểm nguồn/đích: `{build['source_count']}` / `{build['target_count']}`.",
        f"- Tất cả kiểm tra toàn vẹn PASS: `{all(integrity['checks'].values())}`.",
        f"- BM25 và sparse giống hệt: `{comparison['bm25_and_sparse_identical']}`.",
        f"- Kiểm tra chéo Recall/Precision độc lập: `{cross['all_match']}`.",
        "",
        "## Kết quả tổng hợp",
        "",
        "| Cấu hình | Chỉ số | 384 | 768 | Chênh lệch (768-384) | Chênh lệch % | CI 95% | Tốt hơn | Xấu hơn | Không đổi |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        ci = f"[{float(row['ci95_low']):.6f}, {float(row['ci95_high']):.6f}]"
        pct = "" if row["delta_percent"] in ("", "None") else f"{float(row['delta_percent']):.4f}%"
        lines.append(
            f"| {row['method']} | {row['metric']} | {float(row['mean_384']):.6f} | {float(row['mean_768']):.6f} | {float(row['delta_768_minus_384']):.6f} | {pct} | {ci} | {row['improved_queries']} | {row['worsened_queries']} | {row['unchanged_queries']} |"
        )
    lines += [
        "",
        "## Tái tạo số trong README",
        "",
        "Các số dưới đây được so sánh với số đã in trong README; chúng không được dùng để chỉnh sửa kết quả.",
        "",
        "```json",
        json.dumps(baseline, indent=2),
        "```",
        "",
        "## Độ trễ và tài nguyên",
        "",
        _latency_section(latency_path),
        "",
        "## Phân tích lỗi",
        "",
        _error_analysis(),
        "",
        "## Giới hạn",
        "",
        "- Bộ test chỉ có 25 truy vấn và khoảng 75 đoạn đúng theo mô tả thí nghiệm; một đoạn tương đương khoảng 1,3 điểm phần trăm Recall.",
        "- Chạy trên một máy và một lần dựng chỉ mục cho mỗi mô hình.",
        "- Bộ vàng không có chunk ID đúng độc lập; evaluator hiện có suy ra relevant từ `relevant_terms` trên chính kết quả trả về. Vì vậy đây là phép tái hiện evaluator hiện có, không phải đánh giá ground truth chunk-level độc lập.",
        "- Công suất thống kê thấp với `n=25`; CI chứa 0 được báo là không đủ bằng chứng.",
        "- Độ trễ trong lần chạy này chỉ có tổng end-to-end nếu mã hiện có không cung cấp hai timer thành phần.",
        "",
        "## Kết luận",
        "",
        "Kết luận chỉ dựa trên chênh lệch và CI trong bảng trên; không mặc định 768 tốt hơn. Các dòng có CI chứa 0 phải được đọc là không đủ bằng chứng về khác biệt.",
    ]
    (EXPERIMENT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _latency_section(path: Path) -> str:
    if not path.exists():
        return "Chưa có file `audit/latency.json`; phép đo độ trễ chưa hoàn tất."
    latency = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"- Số lần lặp mỗi truy vấn: `{latency['repetitions_per_query']}`.",
        "",
        "| Collection | Cấu hình | n | Encoder p50/p95/p99 (ms) | Qdrant p50/p95/p99 (ms) | Tổng p50/p95/p99 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, values in sorted(latency["summary"].items()):
        collection, method = key.split("_", 1)
        lines.append(
            f"| {collection} | {method} | {values['n']} | "
            f"{values['encoding_p50_ms']:.3f}/{values['encoding_p95_ms']:.3f}/{values['encoding_p99_ms']:.3f} | "
            f"{values['qdrant_p50_ms']:.3f}/{values['qdrant_p95_ms']:.3f}/{values['qdrant_p99_ms']:.3f} | "
            f"{values['total_p50_ms']:.3f}/{values['total_p95_ms']:.3f}/{values['total_p99_ms']:.3f} |"
        )
    return "\n".join(lines)


def _error_analysis() -> str:
    left = json.loads((RAW / "384.json").read_text(encoding="utf-8"))
    right = json.loads((RAW / "768.json").read_text(encoding="utf-8"))
    right_by_id = {q["id"]: q for q in right["queries"]}
    lines = []
    for query in left["queries"]:
        other = right_by_id[query["id"]]
        for method in query["methods"]:
            a = query["methods"][method]["results"]
            b = other["methods"][method]["results"]
            if [x["chunk_id"] for x in a] == [x["chunk_id"] for x in b]:
                continue
            terms = query["relevant_terms"]
            def relevant(result):
                return sum(t.lower() in result["text"].lower() for t in terms) / max(len(terms) * 0.5, 1.0) >= 0.5
            a_rank = {x["chunk_id"]: i + 1 for i, x in enumerate(a)}
            b_rank = {x["chunk_id"]: i + 1 for i, x in enumerate(b)}
            added = [x for x in b if relevant(x) and x["chunk_id"] not in a_rank]
            lost = [x for x in a if relevant(x) and x["chunk_id"] not in b_rank]
            lines.append(f"- `{query['id']}` / `{method}`: added relevant={len(added)}, lost relevant={len(lost)}")
            for label, items, ranks in [("added", added, b_rank), ("lost", lost, a_rank)]:
                for item in items:
                    lines.append(f"  - {label} rank={ranks[item['chunk_id']]} id={item['chunk_id']}: {item['text'][:200].replace(chr(10), ' ')}")
    return "\n".join(lines) if lines else "Không có truy vấn khác biệt."


if __name__ == "__main__":
    main()
