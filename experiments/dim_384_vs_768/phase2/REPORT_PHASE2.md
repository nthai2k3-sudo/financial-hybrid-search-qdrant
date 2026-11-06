# Báo cáo Phase 2: phân tích exact và chấm mù

## Bước 0 - Điểm lệch với README

- README ghi nDCG@10 BM25+Dense: `0.8631`.
- Đường cơ sở tái tạo: `0.812707307419045`.
- Bằng chứng:
  - benchmark.py computes relevant_ids only from the returned result list using relevant_terms; it does not have independent ground-truth chunk IDs.
  - metrics.py computes recall denominator from that result-derived relevant_ids and nDCG ideal gains from the same result-derived relevance_map.
  - Phase 1 integrity confirms 8670 points; README historical figures refer to a different dataset/state (README describes 8609 chunks).
- Kết luận: Có bằng chứng về khác evaluator/điều kiện dữ liệu; không thể quy toàn bộ chênh lệch cho một nguyên nhân duy nhất.

## Phần A - Exact dense search

Dense được xếp hạng bằng tích vô hướng cosine trên toàn bộ 8.670 vector đã lưu; không dùng HNSW gần đúng. Đoạn đúng được xác định minh bạch bằng cùng quy tắc `relevant_terms` của evaluator hiện có trên toàn kho, vì golden dataset không chứa chunk ID độc lập.

| Chỉ số ghép cặp | Mean delta (768-384 hoặc lợi thế hạng) | CI95% | Tốt hơn | Xấu hơn | Không đổi | Kiểm định | p |
|---|---:|---|---:|---:|---:|---|---:|
| first_relevant_rank (384-768 (hạng thấp hơn là tốt hơn)) | -5.500000 | [-19.863636, 3.272727] | 7 | 7 | 8 | Exact sign test | 1.000000 |
| mean_relevant_z (768-384) | 0.272721 | [0.134710, 0.411144] | 17 | 5 | 0 | Exact sign test | 0.016901 |
| median_relevant_rank (384-768 (hạng thấp hơn là tốt hơn)) | 31.977273 | [-4.500000, 75.388636] | 14 | 8 | 0 | Exact sign test | 0.286279 |
| mrr_full_corpus (768-384) | 0.000398 | [-0.122236, 0.130869] | 7 | 7 | 11 | Exact sign test | 1.000000 |

### Overlap@10

- Dense-only: trung bình `0.480000`.
- BM25+Dense: trung bình `0.704000`.
- Chi tiết từng truy vấn nằm trong `results/raw/overlap_at10.csv`.

## Phần B - Bảng chấm thủ công mù

- Bản đầy đủ: `464` mục tại `annotation/annotation_sheet.csv`.
- Bản rút gọn khác biệt: `214` mục tại `annotation/annotation_sheet_short.csv`.
- Vì bản đầy đủ vượt 150 mục, báo cáo cung cấp cả hai bản để người dùng chọn; chưa tự động chọn bản nào.
- Không có cột mô hình, hạng, điểm số hoặc collection trong bảng chấm.
- Khóa ánh xạ được tách riêng tại `annotation/KEY_do_not_open.json`.

### Rubric

- 0 = không liên quan; 1 = cùng chủ đề nhưng không trả lời; 2 = trả lời một phần; 3 = trả lời trực tiếp, đúng công ty, đúng kỳ, đúng chỉ tiêu.
- `wrong_entity_or_period=1` nếu đúng chủ đề nhưng sai công ty hoặc sai kỳ.

Chưa có `annotation_2.csv`; Cohen's kappa có trọng số chưa được tính.

## Phần C - Phân tích sau khi có nhãn người

Chưa chạy. Đang chờ người chấm điền `label` và `wrong_entity_or_period`; không tự điền nhãn.

## Giới hạn

- Chỉ có 25 truy vấn, chạy trên một máy.
- Phân tích exact dùng quy tắc relevance từ `relevant_terms`, không phải ground truth chunk-level độc lập.
- Bảng chấm mới chỉ gồm các đoạn thuộc top-10 của các hệ thống.
- Kết quả nhãn người chưa tồn tại; do đó chưa thể kết luận về nDCG theo nhãn người, Spearman hay mức bỏ sót của golden set.

## Kết luận tạm thời

Các CI chứa 0 phải được đọc là không đủ bằng chứng về khác biệt. Phần C chỉ được thực hiện sau khi người dùng xác nhận đã điền xong nhãn.