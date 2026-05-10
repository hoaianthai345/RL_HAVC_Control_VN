# Kế hoạch thực hiện nghiên cứu  
**Đề tài:** *Edge–Cloud Distributed Reinforcement Learning for Real-Time HVAC Control*  
**Mục tiêu:** thiết kế + đánh giá kiến trúc RL phân tán Edge–Cloud cho điều khiển HVAC thời gian thực, cân bằng **độ trễ suy luận**, **tiết kiệm năng lượng**, và **thoải mái nhiệt**; có cơ chế **continuous learning** qua cập nhật policy định kỳ.

---

## 0) Giả định phạm vi (để triển khai được trong 1 học kỳ)
- Đánh giá chủ yếu trên mô phỏng (HOT testbed / môi trường mô phỏng tương đương); có thể “giả lập edge” bằng máy tính cá nhân với giới hạn CPU để đo độ trễ.
- “Edge” = dịch vụ suy luận + safety layer + logging; “Cloud/HPC” = pipeline train/eval + chọn policy + phát hành phiên bản policy.
- Có tối thiểu 1–3 cấu hình tòa nhà + 2–3 vùng khí hậu để kiểm tra tính tổng quát (có thể mở rộng nếu kịp).

---

## 1) Câu hỏi nghiên cứu & tiêu chí thành công
### RQ1 (Latency vs Performance)
- Edge inference có giảm độ trễ đáng kể so với cloud inference mà vẫn đạt hiệu quả điều khiển tương đương?
**Tiêu chí:** báo cáo p50/p95 inference latency; so sánh năng lượng + comfort + reward.

### RQ2 (Adaptation)
- Retrain định kỳ trên cloud/HPC có giúp robust hơn khi điều kiện thay đổi?
**Tiêu chí:** so sánh trước/sau distribution shift (ví dụ thay đổi lịch occupancy, thời tiết, setpoint mục tiêu).

### RQ3 (Generalization)
- Train phân tán trên nhiều bối cảnh tòa nhà/khí hậu có giúp transfer tốt hơn?
**Tiêu chí:** zero-shot / few-shot fine-tune trên bối cảnh mới; đo “transfer efficiency”.

---

## 2) Thiết kế hệ thống (deliverable kiến trúc)
### 2.1 Edge layer (online loop)
- Thu thập sensor (T_in, T_out, humidity, occupancy, energy, time features)
- Tiền xử lý state + chuẩn hóa
- Suy luận policy `a_t = π(s_t)` (PPO policy)
- Safety constraints (hard constraints): giới hạn setpoint, tốc độ thay đổi, deadband tối thiểu, v.v.
- Logging (state, action, reward components, constraint violations, timestamp)
- Kênh nhận policy update (versioned artifact, checksum, rollback)

### 2.2 Cloud/HPC layer (offline loop)
- Thu thập dữ liệu từ edge (batch)
- Tạo kịch bản mô phỏng / domain randomization
- Distributed training (PPO) + hyperparam sweep (giới hạn)
- Evaluation theo nhiều bối cảnh + chọn policy tốt nhất
- Phát hành policy về edge theo chu kỳ (ví dụ mỗi N giờ/ngày)

---

## 3) Kế hoạch thực hiện theo mốc (10 tuần gợi ý, bắt đầu từ 2026-04-21)
> Nếu bạn có deadline khác, mình sẽ căn lại timeline theo ngày cụ thể.

### Tuần 1 (2026-04-21 → 2026-04-27): Chốt bài toán + setup
- Chốt mô hình MDP: state/action/reward, timestep, constraint
- Chọn bộ baseline tối thiểu (rule-based, static RL, cloud inference)
- Setup môi trường chạy mô phỏng + pipeline chạy 1 episode end-to-end
**Deliverables:** tài liệu “Spec” 1–2 trang + chạy được 1 baseline đơn giản.

### Tuần 2: Baselines + metric instrumentation
- Cài/viết rule-based controller (ASHRAE-like) + thu thập metric
- Thiết kế chuẩn logging + format kết quả (CSV/Parquet) + seed control
**Deliverables:** baseline report nhỏ + script tái chạy thí nghiệm.

### Tuần 3–4: RL training (PPO) trên cloud/HPC (single-node trước)
- PPO training ổn định (reward shaping, normalization, early stopping)
- Tối thiểu 3 seed; lưu checkpoint; eval theo tập kịch bản cố định
**Deliverables:** policy PPO “static” + bảng so sánh với baseline.

### Tuần 5: Phân tán huấn luyện (distributed training)
- Chọn 1 hướng:
  - Ray/RLlib (dễ scale + sweep), hoặc
  - SB3 + vector env + multiprocessing, hoặc
  - Slurm job array (nếu có cụm HPC)
- Xác định “multi-context training” (nhiều tòa nhà/khí hậu) và lịch train
**Deliverables:** pipeline chạy được nhiều worker + tổng hợp kết quả.

### Tuần 6: Edge inference service + đo độ trễ
- Đóng gói policy (TorchScript/ONNX) + runtime nhẹ
- API control loop (gRPC/REST) + safety layer
- Đo latency (p50/p95) trong 3 chế độ: edge-local, cloud-remote, edge-with-update-check
**Deliverables:** báo cáo latency + tài liệu triển khai edge.

### Tuần 7: Edge–Cloud update loop (continuous learning)
- Chu kỳ: (log → upload → retrain → evaluate → deploy)
- Versioning: policy registry + rollback
- Thí nghiệm “periodic retraining” theo lịch (ví dụ mỗi 24h mô phỏng)
**Deliverables:** chứng minh update loop hoạt động + biểu đồ learning/adaptation.

### Tuần 8: Thí nghiệm chính + ablation
- So sánh: rule-based vs static RL vs cloud inference vs edge-only RL vs edge–cloud RL
- Ablation: tần suất update, kích thước model, safety on/off (nếu an toàn)
**Deliverables:** bảng/biểu đồ kết quả chính (energy, comfort, reward, latency).

### Tuần 9: Robustness & generalization
- Kiểm tra distribution shift + transfer (train multi-context → test context mới)
- Báo cáo độ bền (variance theo seed + confidence intervals)
**Deliverables:** phần kết quả RQ2/RQ3.

### Tuần 10: Viết báo cáo + slide + tái lập (reproducibility)
- Chốt cấu hình thí nghiệm + “runbook” để chạy lại
- Viết paper/report + chuẩn hóa hình/ bảng + demo (nếu có)
**Deliverables:** báo cáo cuối + mã nguồn/thí nghiệm tái lập.

---

## 4) Danh mục artefact cần có (để báo cáo thuyết phục)
- `spec.md`: định nghĩa MDP, reward, constraints, update schedule
- `baselines/`: rule-based + centralized cloud inference mock
- `train/`: PPO training config + distributed runner
- `edge/`: inference service + safety + logging + policy loader
- `eval/`: scripts tổng hợp metric + vẽ biểu đồ + thống kê
- `results/`: file kết quả theo seed + context + phiên bản policy

---

## 5) Rủi ro thường gặp & cách giảm
- **Reward không ổn định / comfort bị phá:** bắt đầu với reward đơn giản, thêm penalty theo từng bước; kiểm soát action bounds.
- **Sim chậm:** giảm horizon/timestep, dùng vector env, cache weather/scenario, chạy distributed.
- **So sánh không công bằng:** khóa seed, dùng cùng tập scenario test, log đầy đủ config.
- **Edge latency không khác biệt rõ:** đo theo p95; cố tình giới hạn CPU/quantization; thử ONNX/TorchScript.

---

## 6) Thông tin cần bạn xác nhận (để “khóa” kế hoạch)
- Deadline nộp (ngày/tháng/năm) và định dạng yêu cầu (báo cáo/paper, slide, demo)?
- Bạn có quyền truy cập cụm HPC/Slurm không, hay chỉ chạy local?
- Bạn muốn scope tối thiểu: 1 tòa nhà + 1 khí hậu, hay nhiều hơn?

