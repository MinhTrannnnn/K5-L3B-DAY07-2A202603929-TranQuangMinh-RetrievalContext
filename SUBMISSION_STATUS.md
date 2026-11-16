# Bản nộp cơ bản local

- Agent Qwen2.5 3B chạy Ollama tại localhost.
- BGE local 768 chiều và 332 chunk B của Day 6, PostgreSQL/pgvector.
- Dense, BM25 và hybrid RRF đã triển khai và có kết quả so sánh.
- Kết quả lần chạy local đã hoàn tất ở `results/local/basic/`.
- Demo lệnh query sau bản sửa controller ở `results/local/submission_demo.json`.
- Code mới cần đối chiếu nhãn nội dung/citation thủ công. Bộ benchmark đầy đủ chưa chạy lại sau hotfix cuối.
- Báo cáo hiện tại là `report/individual/2A202603929_TranQuangMinh.md`.

```powershell
./.venv313/Scripts/python.exe -m src query "How can I disable centering in StandardScaler?"
```
