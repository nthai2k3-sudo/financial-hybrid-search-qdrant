# Preregistration: BGE Small 384 vs BGE Base 768

## Mục tiêu

Đo lường có đối chứng sự khác biệt giữa `BAAI/bge-small-en-v1.5` (384 chiều)
và `BAAI/bge-base-en-v1.5` (768 chiều) trên cùng collection dữ liệu tài chính,
cùng 25 truy vấn vàng, cùng tham số truy hồi và cùng mã đánh giá hiện có.

Mục tiêu là đo trung thực cả cải thiện, suy giảm và kết quả không đổi; không giả
định trước mô hình 768 chiều tốt hơn.

## Giả thuyết đã khai báo trước

- **H0:** Không có khác biệt giữa cấu hình dense 384 và dense 768.
- **H1:** Có khác biệt giữa hai cấu hình.
- Chỉ kết luận “tốt hơn/xấu hơn có ý nghĩa” khi khoảng tin cậy bootstrap ghép cặp
  95% của chênh lệch (768 trừ 384) không chứa 0.
- Khi khoảng tin cậy chứa 0, kết luận là **không đủ bằng chứng về khác biệt**,
  bất kể p-value riêng lẻ.

## Chỉ số và điều kiện cố định

- **Chỉ số chính:** nDCG@10 của cấu hình `bm25_dense`.
- **Chỉ số phụ:** Recall@10, MRR, Precision@10, Hit Rate@10 của từng cấu hình;
  Recall@1, Recall@3, Recall@5, Recall@10, Recall@20 cho `dense` và
  `bm25_dense`; độ trễ và dung lượng.
- Cấu hình: `bm25`, `dense`, `bm25_dense`, `sparse`, `sparse_dense`, `rerank`.
- `K=10` cho bảng chỉ số chính/phụ; kết quả thô lưu top-20.
- Giữ nguyên RRF constant `60`, prefetch limit `50`, chunking, payload, bộ lọc,
  reranker và bộ truy vấn.
- Chỉ thay đúng mô hình dense và số chiều. Vector `bm25` và `sparse` được sao
  chép nguyên trạng sang collection 768.
- Không recreate, ghi đè hoặc sửa collection 384; không sửa
  `evaluation/golden_dataset.yaml`.
- Bootstrap ghép cặp: 10.000 lần, seed cố định, lấy mẫu theo 25 truy vấn.

## Quy tắc dừng và báo cáo

Nếu không tải được đúng `BAAI/bge-base-en-v1.5`, không đọc được collection 384,
không đạt kiểm tra toàn vẹn, hoặc collection 768 đã tồn tại, thí nghiệm phải
dừng và báo nguyên nhân; không thay mô hình, dữ liệu hay tham số.

Mọi số liệu báo cáo phải được sinh từ JSON/CSV kết quả thô. Báo cáo phải nêu cả
kết quả xấu đi và không đổi, cùng giới hạn công suất thống kê thấp do `n=25`.
