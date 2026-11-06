# Plan: BGE 384 vs BGE 768

## Phạm vi Bước 0

Đây là khảo sát chỉ đọc. Tại bước này chưa tạo collection mới, chưa sao chép
snapshot, chưa ghi Qdrant và chưa chạy embedding 768.

## Trạng thái hiện tại

- Repository commit khảo sát: `16682dd42ca9cd4887ae74935cb3526dce580c49`.
- Qdrant đang chạy bằng **embedded local storage trên đĩa**, không phải Docker:
  `config/default.yaml` đặt `prefer_local_embedded: true`, đường dẫn tương đối
  `qdrant_data`.
- API `/api/status` xác nhận `storage_mode=embedded`, collection hiện tại
  `financial_docs`, trạng thái `online`, `points_count=8670`.
- README ghi 8.609 chunks, nhưng collection thực tế hiện có 8.670 points. Khi
  chạy thí nghiệm, số đọc trực tiếp từ collection sẽ là nguồn sự thật.
- File metadata hiện tại xác nhận collection `financial_docs` có:
  - named dense vector `dense`: 384 chiều, cosine;
  - named sparse vector `bm25`: modifier IDF;
  - named sparse vector `sparse`: modifier IDF.
- Không dùng dữ liệu `qdrant_data_empty_backup/` cho thí nghiệm.

## Dữ liệu và chuỗi đầu vào embedding

`DocumentChunker.chunk_document()` tạo tiền tố ngữ cảnh chính xác:

```text
[Company (Ticker) | DocumentType FiscalPeriod | Section: SectionName]

```

Sau đó nối với nội dung chunk. Trong `QdrantIndexManager.index_chunks()`, dense
document embedding nhận `batch_texts = [c.text for c in batch_chunks]`; vì vậy
chuỗi đưa vào encoder dense là toàn bộ `payload["text"]`, bao gồm context header
và phần nội dung chunk. Payload lưu cùng chính chuỗi đó trong trường `text`, cùng
`text_hash`.

Document embedding không thêm query instruction. Query embedding trong
`DenseEmbeddingGenerator.embed_query()` thêm chính xác tiền tố sau khi model name
có chứa `bge`:

```text
Represent this sentence for searching relevant passages: {query}
```

Do đó 384 và 768 sẽ dùng cùng quy tắc query prefix; chỉ model name thay đổi.

BM25 và SPLADE dùng query nguyên bản, không thêm BGE prefix. Vector `bm25` và
`sparse` của collection 768 sẽ được sao chép byte/giá trị từ collection 384,
không tính lại.

## Truy hồi và tham số cố định

- `dense`: query named vector `dense`, `limit=k`.
- `bm25`: query named sparse vector `bm25`, `limit=k`.
- `sparse`: query named sparse vector `sparse`, `limit=k`.
- `bm25_dense`: hai prefetch (`bm25`, `dense`) cùng `prefetch_k=50`, hợp nhất
  server-side RRF với constant theo cấu hình `60`, trả `k`.
- `sparse_dense`: tương tự với `sparse` và `dense`.
- `rerank`: lấy `sparse_dense` với `max(k*3, 20)` ứng viên rồi áp dụng
  `NeuralReranker`.

`NeuralReranker` không gọi vector/model encoder; tên model
`colbert-ir/colbertv2.0` chỉ là metadata, còn điểm rerank dùng text, token overlap,
trích xuất câu/bảng, metric bonus, intent, năm và recency.

## Cách tính chỉ số hiện có

Trong `src/hybrid_search/evaluation/metrics.py`:

- Recall@K = số relevant ID trong top-K chia tổng số relevant ID.
- Precision@K = số relevant ID trong top-K chia K.
- MRR = nghịch đảo hạng của relevant đầu tiên.
- Hit Rate@K = 1 nếu top-K có ít nhất một relevant, ngược lại 0.
- nDCG@K dùng điểm relevance theo ID, chiết khấu bằng `log2(rank+1)` và chuẩn
  hóa theo ideal DCG.

Trong `benchmark.py`, mỗi truy vấn chạy từng method `num_latency_runs` lần; chỉ
lần chạy cuối được dùng để lấy ID và tính chỉ số. Relevance được suy ra từ
`relevant_terms`: mỗi term xuất hiện trong text được tính điểm coverage, và
chunk có điểm `>=0.5` trở thành relevant. `relevance_map` chỉ được xây từ các
result trả về ở lần chạy cuối.

## Bộ truy vấn vàng và điểm mơ hồ cần xác nhận

`evaluation/golden_dataset.yaml` có đúng 25 truy vấn, thuộc 5 category:
`exact_terminology`, `semantic_questions`, `mixed_queries`,
`rare_terminology`, `long_natural_language`; tổng cộng 106 relevant terms.

File vàng không chứa danh sách chunk ID đúng cố định, chỉ chứa
`relevant_terms`. Vì vậy benchmark hiện tại không có ground truth chunk-level
độc lập: mẫu relevant và cả mẫu số Recall được suy ra từ các kết quả mỗi method.
Điều này có thể làm kết quả phụ thuộc collection/mô hình và khiến Recall không
phải Recall theo tập đúng cố định. Đây là điểm mơ hồ được phát hiện trước khi
chạy; không tự thay đổi `golden_dataset.yaml` hoặc `metrics.py`. Bản chạy sau sẽ
ghi rõ nó tái hiện đúng evaluator hiện có và tách kiểm tra chéo tối giản theo
cùng semantics.

README cũng có các số benchmark cũ (ví dụ 8.609 chunks và Recall@10 0.9600),
nhưng chúng chưa được xem là số liệu tái tạo. Bước kiểm tra sẽ chạy lại baseline
384 và dùng số tái tạo nếu khác.

## FastEmbed

`TextEmbedding.list_supported_models()` trong môi trường hiện tại liệt kê cả:

- `BAAI/bge-small-en-v1.5`, dimension 384, nguồn ONNX Qdrant.
- `BAAI/bge-base-en-v1.5`, dimension 768, nguồn ONNX Qdrant.

Vì vậy đúng model 768 được hỗ trợ; thí nghiệm vẫn phải dừng nếu bước tải/khởi tạo
model thực tế thất bại.

## Kế hoạch sau khi được xác nhận

1. Sao lưu collection 384 và ghi checksum/path vào `audit/`.
2. Dựng collection mới `financial_docs_bge768` chỉ nếu tên chưa tồn tại.
3. Scroll toàn bộ points, giữ nguyên ID/payload/bm25/sparse, tính lại dense 768 từ
   đúng payload text theo batch có checkpoint.
4. Chạy toàn bộ kiểm tra toàn vẹn trước khi benchmark.
5. Chạy baseline 384 và 768 với cùng 25 truy vấn, lưu raw top-20.
6. Đối chiếu evaluator, thống kê ghép cặp, độ trễ/tài nguyên, phân tích lỗi và
   sinh `REPORT.md`/`summary.csv` tự động từ kết quả thô.

## Câu hỏi cần xác nhận trước khi ghi Qdrant

Bạn xác nhận tiếp tục với evaluator hiện có như mô tả ở trên (relevance suy ra
từ `relevant_terms` trên chính kết quả trả về), hay muốn dừng để bổ sung một
ground truth chunk-ID độc lập? Theo yêu cầu hiện tại, mình sẽ không tự thay đổi
evaluator và chỉ bắt đầu Bước 1-8 sau khi bạn xác nhận.
