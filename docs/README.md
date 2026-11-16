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
ollama pull qwen2.5:3b
docker compose -f ../../Day6/K5-L3B-DAY06-TranQuangMinh-2A202603929-DataIngestion/starter/docker-compose.yml -p day6_student_starter up -d --wait
./.venv313/Scripts/python.exe -m src ingest --corpus day6
./.venv313/Scripts/python.exe -m src doctor
```

`.env` dùng `OLLAMA_URL=http://127.0.0.1:11434` và `OLLAMA_MODEL=qwen2.5:3b`.
Code giới hạn endpoint ở loopback. Chi phí API local được ghi là 0 USD.
CPU/GPU và điện năng chưa được quy đổi thành tiền.

Ingest dùng lại chunk B và cache vector thật của Day 6 để nạp database.
BGE chạy offline từ cache model đã tải. Repo Day 6 nằm cạnh Day 7 theo cấu trúc workspace.

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

Kết quả local ở `results/local/final/`. `reviews_pending.json` là mẫu chấm context,
answer, citation và abstention sau khi đối chiếu nguồn.

```powershell
$env:PYTHONUTF8 = '1'
./.venv313/Scripts/python.exe -m unittest discover -s tests -v
```
