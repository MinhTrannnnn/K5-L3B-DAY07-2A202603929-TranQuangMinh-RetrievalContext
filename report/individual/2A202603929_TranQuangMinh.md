# Báo cáo Day 7 — Agentic RAG cơ bản, local

Trần Quang Minh · 2A202603929 · L3B

Đây là **báo cáo cá nhân duy nhất** của bản nộp. Kết quả chính thức nằm trong
`results/local/final/`. Các thư mục kết quả khác là lịch sử thử nghiệm.

## Dữ liệu và benchmark

Nguồn: sáu trang Scikit-learn User Guide 1.9, 72 đơn vị nguồn HTML, 332 chunk chiến lược B
từ Day 6. Giữ nguyên source URL, page ID, metadata. Provenance và giấy phép BSD-3-Clause:
`data/day6/data/manifest.json`, `data/day6/data/LICENSE-scikit-learn.txt`.
Snapshot kèm units, chunks, cấu hình và store gốc để tái lập khi không có repo Day 6 bên cạnh.
Một số ký tự lỗi mã hóa ở nguồn được giữ nguyên, có thể ảnh hưởng tokenization.

Embedding **BAAI/bge-base-en-v1.5**, revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`,
768 chiều, vector chuẩn hóa, chạy CPU local. Query instruction:
`Represent this sentence for searching relevant passages: `.
PostgreSQL 16 + pgvector 0.8.7, cosine exact search, localhost:55433,
bảng `lab6_chunks`, experiment `9b408e4e3b1de0b1`, strategy B.
Agent **qwen2.5:3b qua Ollama localhost**, nhiệt độ 0; không dùng API key trả phí.

Benchmark `benchmark/local_basic_cases.json`: dev **15 câu có đáp án + 2 câu thiếu evidence**;
test **5 câu có đáp án + 2 câu thiếu evidence**. Test L01–L07 được lập trước lần chạy test này,
thay các bộ test cũ đã tham gia sửa controller. Không đưa gold vào agent/search query.
Chọn cấu hình theo Evidence Hit@3 trên dev, rồi Hit@1, rồi thời gian agent thấp hơn.
`results/local/final/selection.json` khóa fingerprint corpus, benchmark, model và controller trước test.
Không chỉnh controller theo kết quả L01–L07 sau khi chạy.

Câu có đáp án chứa nguồn, page, gold quote. Câu thiếu evidence hỏi RAM trên mọi dataset,
ngày phát hành tương lai hoặc thời gian/accuracy của thí nghiệm riêng. Phạm vi units công khai
không chứa cấu hình, dữ liệu, nhãn và phép đo riêng đó; không kết luận thiếu từ top-k đơn thuần.

## Retrieval và thử nghiệm

Cùng corpus, index và benchmark. Lưu ranking query gốc riêng với query agent viết lại.
Top-k 6; context budget 3000 token gồm metadata trước khi rút gọn prompt; loại trùng context.
Hybrid kết hợp top-20 dense/BM25 bằng RRF hằng số 60. Không có cross-encoder rerank.
Filter metadata theo case được áp dụng giống nhau giữa các phương pháp, lưu trong trace.

| Cấu hình | Thay đổi và giả thuyết |
|---|---|
| Dense baseline | BGE + pgvector cosine, tái sử dụng index Day 6 |
| BM25 — cải thiện 1 | Từ khóa phân biệt tên API, option và thuật ngữ chính xác |
| Hybrid RRF — cải thiện 2 | Kết hợp hai ranking để tăng coverage |

Hai cấu hình mới đồng thời là hai thử nghiệm cải thiện; không giả định mọi thử nghiệm đều tăng điểm.

| Method | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Tổng thời gian agent / phí API |
|---|---|---|---|---|---|
| Dense | dev | 8/15 (53.3%) | 13/15 (86.7%) | 13/15 (86.7%) | 65.09 s / 0 USD |
| BM25 | dev | 9/15 (60.0%) | 14/15 (93.3%) | 13/15 (86.7%) | 45.58 s / 0 USD |
| Hybrid | dev | 7/15 (46.7%) | 13/15 (86.7%) | 14/15 (93.3%) | 44.62 s / 0 USD |
| Dense | test | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 28.39 s / 0 USD |
| BM25 | test | 2/5 (40.0%) | 5/5 (100.0%) | 5/5 (100.0%) | 21.78 s / 0 USD |
| Hybrid | test | 2/5 (40.0%) | 3/5 (60.0%) | 0/0 (không áp dụng) | Không chạy agent |

Mẫu số chỉ gồm câu có đáp án. Hit yêu cầu nguyên quote sau chuẩn hóa khoảng trắng, đúng source/page.
Context metric dùng cùng phép đo trên context thật sau query rewrite. Đoạn thay thế đúng vẫn có thể
bị tính miss theo gold; nhãn đọc nguồn được báo riêng. Thời gian là tổng agent, gồm viết query,
search và trả lời; thời gian toàn lượt đánh giá được lưu trong summary.json.
Hybrid test chỉ đo retrieval vì cấu hình chọn là BM25.

**Chọn BM25 trên dev:** Hit@3 và Hit@1 cao nhất; agent nhanh hơn dense. Hybrid có context
coverage tốt hơn nhưng không thắng tiêu chí đã công bố. Test BM25 tăng Hit@3 từ 3/5 lên 5/5,
đồng thời Hit@1 giảm từ 3/5 xuống 2/5. Không đổi tiêu chí sau khi xem test.

### Hai lỗi dev, trước/sau và giới hạn

1. **q10, chia train/test theo nhóm:** dense thiếu gold GroupKFold trong top-3/context theo
   phép đo quote. Giả thuyết: nhiều khái niệm chia nhóm gần nghĩa gây nhiễu. BM25 đưa gold
   vào top-3 và context, agent trả GroupKFold. Hybrid vẫn miss gold top-3.
   Dense trả GroupShuffleSplit, cũng có thể hợp cách hỏi rộng; đây là miss gold chuẩn,
   không kết luận mọi đáp án khác GroupKFold đều sai. Cần mở rộng gold thay thế nếu tiếp tục.
2. **q12, GridSearchCV/RandomizedSearchCV:** dense miss quote chuẩn ở top-1/3;
   BM25 đưa đoạn đối chiếu hai API lên top-1, hybrid đưa vào top-3.
   BM25 trả đúng exhaustive combinations và sampling. Dense có đoạn thay thế đủ trả lời đúng,
   nên miss quote không đồng nghĩa sai answer. BM25 có lợi với tên API chính xác.

## Agent và đánh giá

Model thực sự viết search query; chương trình gọi retrieval tool và đóng gói context;
model chọn answer/abstain. Mặc định tối đa **1 search/query**, có thể cấu hình 2–3.
Dừng khi trả lời, từ chối hoặc hết lượt. Alias S1/S2 được ánh xạ về chunk ID thật.
Chặn citation ngoài context và answer rỗng chỉ có citation. JSON lỗi được thử lại một lần;
tool/model error được lưu riêng. Endpoint giới hạn loopback. ID hợp lệ không tự chứng minh citation đúng.

Đã đọc **14 lượt test**: 5 có đáp án và 2 thiếu evidence cho mỗi cấu hình dense/BM25.
Nhãn do **Codex đối chiếu source units gốc**, chưa được sinh viên xác nhận thủ công;
không trình bày là human review độc lập. Mỗi context chunk được đối chiếu với toàn văn unit gốc.
Nhãn và lý do: `results/local/final/test/reviews.json`; nguồn đối chiếu: `review_sources.json`;
tổng hợp: `review_summary.json`. `reviews_pending.json` chỉ là mẫu trống runner tạo.

Context: 0 không evidence, 1 thiếu điều kiện/nguồn, 2 đủ evidence và nguồn.
Answer: 0 sai/không đủ, 1 đúng một phần hoặc có bổ sung chưa chứng minh, 2 đúng đủ.
Citation chấm từng ID theo nội dung hỗ trợ. Câu thiếu evidence không chấm answer/context 0/1/2.
Abstention chấm chặt: cần thực sự từ chối, không chỉ trả thông tin gần chủ đề.

| Method/test | Tool use | Search | Answer số mức 0/1/2; TB | Citation hỗ trợ | Phản hồi có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense | 7/7 (100%) | 1/query | 1/1/3; 1.40/2 | 4/9 (44.4%) | 7/7 (100%) | 0/2 (0%) | 0/5 (0%) |
| BM25 | 7/7 (100%) | 1/query | 1/0/4; 1.60/2 | 4/8 (50%) | 6/7 (85.7%) | 1/2 (50%) | 0/5 (0%) |

Context đọc nguồn, số mức 0/1/2: dense **0/1/4**; BM25 **0/0/5**.
Dense L02 có example `return_estimator=True` đủ trả lời dù không khớp nguyên gold quote;
vì vậy context đọc nguồn mức 2 là 4/5, cao hơn metric tự động 3/5.

### Nhãn từng câu test

Query, tool call, ranking, context, answer, citation đầy đủ nằm trong `test/traces.json`,
tra theo method/query_id. Lý do chi tiết từng citation nằm trong `test/reviews.json`.

| Method | Query | Context | Answer | Citation hỗ trợ/tổng | Abstention | Lý do |
|---|---|---|---|---|---|---|
| Dense | L01 | 2 | 1 | 0/1 | N/A | Đúng phần fit không làm gì; thêm ý fit khởi tạo object không có trong nguồn |
| Dense | L02 | 2 | 2 | 1/1 | N/A | Example đặt return_estimator=True, đáp án gọi đúng tên option |
| Dense | L03 | 1 | 0 | 0/1 | N/A | Context chỉ nói có thể trả indices; answer gọi tên hàm thay vì option |
| Dense | L04 | 2 | 2 | 2/2 | N/A | Hai nguồn hỗ trợ MinMaxScaler/MaxAbsScaler và cách scale |
| Dense | L05 | 2 | 2 | 1/1 | N/A | Đối chiếu input/output example: -1 thành 0 |
| Dense | L06 | N/A | N/A | 0/2 | Sai | Có nói không xác định chính xác nhưng action answered và thêm giả định riêng |
| Dense | L07 | N/A | N/A | 0/1 | Sai | Trả mô tả Binarizer thay vì từ chối accuracy không biết |
| BM25 | L01 | 2 | 2 | 1/1 | N/A | Nguồn nói fit không làm gì vì mỗi mẫu độc lập |
| BM25 | L02 | 2 | 2 | 1/1 | N/A | Đúng return_estimator=True và nguồn trực tiếp |
| BM25 | L03 | 2 | 0 | 0/2 | N/A | Có return_indices=True trong context nhưng model chọn sai return_train_score |
| BM25 | L04 | 2 | 2 | 1/1 | N/A | Nguồn chỉ rõ hai transformer đúng thứ tự |
| BM25 | L05 | 2 | 2 | 1/1 | N/A | Example chứng minh -1 thành 0 |
| BM25 | L06 | N/A | N/A | 0/0 | Đúng | Action abstain, nói thiếu dữ liệu thời gian chạy riêng, không bịa số |
| BM25 | L07 | N/A | N/A | 0/2 | Sai | Không từ chối rõ, thêm ý fitting/evaluation không được hai citation hỗ trợ |

**51 agent dev + 14 agent test, 0 lỗi thực thi.** Phí API lần chạy local **0 USD**;
điện năng/phần cứng chưa quy đổi tiền. Chi phí thử cloud trước đó là lịch sử riêng,
không nằm trong con số 0 USD này. Trace lưu token usage và thời gian từng lượt.

## Tái lập, kiểm thử và giới hạn

Hướng dẫn đầy đủ: `docs/README.md`. `scripts/prepare_basic_local.py` tải BGE revision cố định
miễn phí một lần; sau đó inference offline. Snapshot `data/day6/` tái tạo database độc lập.
Không đóng gói model weights, `.env` hay API key. Clone mới cần cài dependencies, tải model,
khởi động Docker và Ollama.

```powershell
./.venv313/Scripts/python.exe -m src query "How can I disable centering in StandardScaler?"
./.venv313/Scripts/python.exe scripts/run_basic_local.py
./.venv313/Scripts/python.exe -m unittest discover -s tests -v
```

**25 kiểm thử code đã qua**; test mock kiểm tra logic, tách biệt benchmark model thật.
Đã kiểm tra restore 332 chunk bằng snapshot khi giả lập không có repo Day 6 bên cạnh.
Giới hạn: corpus nhỏ, HTML tiếng Anh, model 3B nhầm option dù context đúng (L03), abstention
chưa tốt (L07), citation chưa hỗ trợ đầy đủ. Không tuyên bố độ tin cậy production hoặc điểm rubric.

## Nâng cao

Không yêu cầu chấm bonus ở bản nộp này. NIST/WDR và OCR/vision trước đây là thử nghiệm lịch sử;
chưa chứng minh đủ điều kiện tài liệu/benchmark/90%. Không tính vào đánh giá cơ bản hiện tại.
