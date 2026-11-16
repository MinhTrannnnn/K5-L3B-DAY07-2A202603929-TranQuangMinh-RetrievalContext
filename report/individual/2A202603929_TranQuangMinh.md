# Báo cáo Day 7 cơ bản — local

## Kết quả đã chạy để nộp

| Phương pháp | Dev Hit@1 | Dev Hit@3 | Test Hit@1 | Test Hit@3 |
|---|---|---|---|---|
| Dense | 8/15 | 13/15 | 3/5 | 4/5 |
| BM25 | 9/15 | 14/15 | 4/5 | 5/5 |
| Hybrid RRF | 7/15 | 13/15 | 3/5 | 5/5 |

Các kết quả này nằm trong `results/local/basic/`, dùng `benchmark/basic_v2_cases.json`.
Đã chạy 51 lượt agent trên dev và 14 lượt trên test. Các nhãn nội dung cần đối chiếu nguồn.
Sau đó controller được chỉnh để tách schema search/answer và ngăn answer rỗng chỉ có citation.
Lệnh query đã kiểm tra lại với StandardScaler. Toàn bộ benchmark chưa chạy lại sau bản sửa cuối,
do người học yêu cầu chốt ngay để nộp. Những metric trên là kết quả của lần chạy đã lưu.

Trần Quang Minh · 2A202603929 · L3B

## Dữ liệu và cấu hình

Corpus kế thừa Day 6 là Scikit-learn User Guide 1.9, 72 đơn vị nguồn.
Dùng 332 chunk của chiến lược B. Chunk, source ID, page ID và metadata được giữ
theo đầu ra Day 6. Vector thật đã có được nạp lại từ cache vào PostgreSQL/pgvector.

Embedding BAAI/bge-base-en-v1.5 chạy CPU local, 768 chiều, revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. Query instruction và normalization
giống Day 6. Store dùng `lab6_chunks`, experiment `9b408e4e3b1de0b1`, strategy B,
cosine exact search. Model được load offline từ cache.

Agent dùng Qwen2.5 3B qua Ollama local. Mặc định một lần search, top-k 6,
context budget 3000 token bao gồm metadata trước khi rút gọn nội dung gửi model.
Có thể cấu hình nhiều lần search. Model chọn query và action qua JSON schema.
Các tool result chỉ được dùng làm dữ liệu. ID S1/S2 được gắn bởi chương trình từ
chunk thật, rồi chuyển về citation chunk ID khi xuất kết quả. ID ngoài context bị từ chối.
Citation hỗ trợ nội dung cần được đối chiếu nguồn riêng.

## Retrieval và thử nghiệm

Ba phương pháp dùng cùng chunk B, corpus, embedding và bộ câu hỏi.

| Cấu hình | Cách tìm | Giả thuyết |
|---|---|---|
| Dense baseline | BGE query + pgvector cosine | Tái sử dụng hệ thống Day 6 |
| BM25, cải thiện 1 | Ranking từ khóa | Tên API và thuật ngữ chính xác có thể cải thiện ranking |
| Hybrid RRF, cải thiện 2 | Dense và BM25 top-20, RRF hằng số 60 | Kết hợp hai ranking có thể tăng coverage |

Chọn trên dev theo Evidence Hit@3, rồi Hit@1, rồi thời gian agent.
Các cấu hình mới đồng thời là hai thử nghiệm cải thiện, có trace đo riêng.
Kết quả có thể không tăng điểm và vẫn được báo đầy đủ.

Benchmark `benchmark/local_basic_cases.json` gồm 15 dev có đáp án + 2 dev thiếu evidence,
5 test có đáp án + 2 test thiếu evidence. Các câu test v2 dùng các facts khác bộ test
cơ bản cũ, được tạo trước lần chạy pipeline local. Tài liệu nguồn vẫn được dùng chung
giữa dev và test, nên chưa thể suy rộng ngoài corpus này.

Kết quả thực tế ở `results/local/final/dev/summary.json` và
`results/local/final/test/summary.json`. Trace từng câu gồm ranking, tool call,
context, answer, citation, token và thời gian. Lựa chọn trước test nằm ở
`results/local/final/selection.json` với fingerprint model/controller/dữ liệu.

## Đánh giá và giới hạn

Hit@1 và Evidence Hit@3 dùng đủ quote, đúng nguồn và page. Context Evidence cũng
kiểm tra quote trên context thực tế đã gửi model. Đây là phép đo tự động cần đối chiếu.
Các nhãn context 0/1/2, answer 0/1/2, citation support và abstention nằm trong
`reviews_pending.json`, cần người học đọc và xác nhận sau khi xem nguồn.

Các lỗi dev cần xem gồm q10 và q12 của dense. Ở Day 6 và lần retrieval ban đầu,
dense tìm các đoạn gần nghĩa nhưng thiếu full evidence quote trong top-3.
BM25 có thể tìm đúng thuật ngữ và quote hơn. RRF có thể đẩy gold xuống khi hai
retriever ưu tiên các đoạn khác nhau. Cần đọc trace local để chấm câu trả lời thực tế.

Model 3B có thể bỏ sót điều kiện hoặc thêm giải thích vượt evidence. Guard citation
chỉ bảo đảm ID có trong context. Cấu hình một lần search ưu tiên thời gian chạy,
cần nhiều lượt hơn cho câu hỏi không đủ evidence trong lần tìm đầu.

25 unit tests đã qua sau khi chuyển local, gồm guard endpoint loopback,
ánh xạ citation và từ chối ID chưa truy xuất.
Chi phí gọi model local là 0 USD. CPU/GPU và điện năng chưa quy đổi thành tiền.

## Cách chạy

Xem `docs/README.md`. Lệnh query từ thư mục gốc Day 7.

```powershell
./.venv313/Scripts/python.exe -m src query "How can I disable centering in StandardScaler?" --method bm25
./.venv313/Scripts/python.exe scripts/run_basic_local.py
```
