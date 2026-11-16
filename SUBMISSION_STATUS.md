# Bản nộp Day 7 cơ bản — local

- Một báo cáo cá nhân: `report/individual/2A202603929_TranQuangMinh.md`.
- 332 chunk B, BGE 768 chiều, PostgreSQL/pgvector; snapshot kèm nguồn và license.
- Qwen2.5 3B/Ollama local, không dùng API key trả phí.
- Dense/BM25/hybrid; 51 agent dev + 14 agent test, không có lỗi thực thi.
- Benchmark 24 câu: 17 dev, 7 test; chốt BM25 trên dev trước test.
- BM25 test Evidence Hit@3 5/5, dense 3/5. Không đồng nghĩa answer đúng 100%.
- Có nhãn và lý do cho toàn bộ 14 lượt test, đối chiếu source units gốc.
- Nhãn do Codex hỗ trợ, chưa được sinh viên xác nhận thủ công; ghi trung thực lỗi model.
- 25 unit tests đã qua; đã kiểm tra restore bằng snapshot độc lập.
- Tái lập: `docs/README.md`. Kết quả chính thức: `results/local/final/`.
- Không yêu cầu chấm bonus nâng cao trong bản này.
