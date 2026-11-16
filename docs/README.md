# Day 7 cơ bản chạy local

Chạy từ thư mục gốc repo Day 7.

```powershell
./.venv313/Scripts/python.exe -m src query "How can I disable centering in StandardScaler?" --method bm25
```

Agent dùng Qwen2.5 3B qua Ollama. Dense dùng BGE-base-en-v1.5 local, 768 chiều,
revision và query instruction của Day 6. PostgreSQL/pgvector lưu 332 chunk B.
Mỗi query lưu answer, citation và trace trong `results/local/queries/`.

## Khởi động

Nếu Ollama đã chạy ở system tray, tiếp tục các lệnh dưới.

```powershell
py -3.13 -m venv .venv313
./.venv313/Scripts/python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:3b
./.venv313/Scripts/python.exe scripts/prepare_basic_local.py
docker compose -f data/day6/starter/docker-compose.yml -p day6_student_starter up -d --wait
./.venv313/Scripts/python.exe -m src ingest --corpus day6
./.venv313/Scripts/python.exe -m src doctor
```

`.env` dùng `OLLAMA_URL=http://127.0.0.1:11434` và `OLLAMA_MODEL=qwen2.5:3b`.
Code giới hạn endpoint ở loopback. Chi phí API local được ghi là 0 USD.
CPU/GPU và điện năng chưa được quy đổi thành tiền.

Ingest dùng lại chunk B và cache vector thật của Day 6 để nạp database.
BGE chạy offline từ cache model đã tải. Nếu không có repo Day 6 bên cạnh,
code dùng snapshot `data/day6/` kèm units, chunks, cấu hình, store gốc và giấy phép.
Lần đầu chạy script prepare để tải BGE miễn phí; không cần cache riêng của người nộp.

## Retrieval và benchmark

```powershell
./.venv313/Scripts/python.exe -m src query "What does RepeatedKFold do?" --method dense
./.venv313/Scripts/python.exe -m src query "What does RepeatedKFold do?" --method bm25
./.venv313/Scripts/python.exe -m src query "What does RepeatedKFold do?" --method hybrid
./.venv313/Scripts/python.exe scripts/run_basic_local.py
```

Agent tạo query bằng model, gọi search rồi chọn answer hoặc abstain.
Mặc định một lượt search để chạy nhanh. Có thể tăng bằng `--max-rounds 2` hoặc 3.
ID ngắn S1/S2 giúp model trích dẫn, sau đó được ánh xạ về chunk ID thật.
Code định dạng citation khi model đã cung cấp danh sách ID hợp lệ.
Cần đối chiếu nguồn để xác nhận citation hỗ trợ mọi khẳng định.

`benchmark/local_basic_cases.json` có 17 dev và 7 test. Mỗi split có 2 câu thiếu evidence.
Dev chạy ba phương pháp. Chọn theo Evidence Hit@3, rồi Hit@1, rồi thời gian agent.
Test chạy dense baseline và cấu hình chọn trên cùng tập. Fingerprint khóa model và controller.

Kết quả chính thức ở `results/local/final/`. Nhãn đối chiếu nguồn cho cả 14 lượt test
ở `test/reviews.json`, tổng hợp ở `test/review_summary.json`. Nhãn do Codex hỗ trợ,
chưa có xác nhận thủ công của sinh viên. `reviews_pending.json` chỉ là mẫu trống runner tạo.

```powershell
$env:PYTHONUTF8 = '1'
./.venv313/Scripts/python.exe -m unittest discover -s tests -v
```
