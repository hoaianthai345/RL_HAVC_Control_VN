# Sổ tay chạy thí nghiệm HVAC edge–cloud trên máy chủ GPU

Tài liệu này bổ sung và làm **nguồn tham chiếu chi tiết** cho chạy trên máy chủ có NVIDIA GPU và/hoặc cụm Slurm/HPC. Hướng dẫn ngắn gọn và đóng gói có thêm tại [`README_CLOUD_EXPERIMENTS.md`](README_CLOUD_EXPERIMENTS.md) và [`cloud_run_guide_vi.md`](../docs/cloud_run_guide_vi.md).

**Đề tài khung:** reinforcement learning có nhận thức độ trễ (latency-aware) trong kiến trúc edge–cloud, đánh giá điều khiển HVAC dưới dịch chuyển phân phối (thời tiết, tòa nhà, lịch chiếm dụng), với chỉ số điều khiển (năng lượng, comfort, reward), hệ thống (p50/p95 latency), và (trong phiên bản paper đầy đủ) thích nghi/chuyển giao.

---

## 1. Phạm vi pipeline hiện tại và tính sẵn sàng chạy

| Thành phần | Mô tả | Sẵn sàng báo cáo trong paper? |
|------------|--------|-------------------------------|
| **Chế độ `dummy`** | Môi trường gymnasium giả lập trong [`src/envs.py`](src/envs.py); kiểm tra end-to-end script, baseline, latency, train PPO, tổng hợp CSV | **Không.** Chỉ dùng để **thẩm định pipeline** và logic mã |
| **Chế độ `hot`** | Môi trường thật qua factory `HOT_ENV_FACTORY`; cần adapter do bạn triển khai (xem [`src/hot_adapter_template.py`](src/hot_adapter_template.py)) | **Có**, sau khi adapter ổn định, HOT/EnergyPlus và bối cảnh khớp manuscript |

**Kết luận vận hành:** Pipeline **dummy** sẵn sàng chạy trên GPU server *sau khi* cài Python + PyTorch (CUDA nếu train PPO trên GPU). Pipeline **hot** chỉ sẵn sàng khi `create_env(...)` trong adapter trả về object gymnasium tương thích và dữ liệu/mô phỏng HOT khả dụng.

---

## 2. Điều kiện tiên quyết

- **Hệ điều hành:** Linux trên server/HPC (khuyến nghị; macOS có thể dùng cho dummy, paper thường cần Linux + HOT).
- **Python:** 3.10 trở lên (khuyến nghị; kiểm tra `python3 --version`).
- **Driver GPU:** `nvidia-smi` hiển thị GPU và driver tương thích với bản PyTorch CUDA bạn cài.
- **CUDA / PyTorch:** File [`requirements.txt`](requirements.txt) khai báo `torch>=2.2` **không** cố định bản wheel GPU. Trên server, nên cài PyTorch theo hướng dẫn chính thức cho phiên bản CUDA của cụm (ví dụ CUDA 12.x), rồi cài các gói còn lại.

Ví dụ kiểm tra sau cài đặt (trong virtualenv đã kích hoạt):

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Stable-Baselines3 mặc định dùng `device="auto"` khi huấn luyện; nếu PyTorch thấy CUDA, policy sẽ chạy trên GPU. Script [`src/train_ppo.py`](src/train_ppo.py) hỗ trợ `--device auto|cuda|cpu` để ghi rõ trong log thí nghiệm.

---

## 3. Đóng gói, tải lên server và cài đặt

Từ máy có mã nguồn (thư mục workspace):

```bash
bash experiments_cloud/scripts/package_for_cloud.sh
```

Tạo `dist/eaai_hvac_cloud_bundle.tar.gz`. Đưa file lên server (`scp`, rsync, hoặc storage HPC), giải nén:

```bash
tar -xzf eaai_hvac_cloud_bundle.tar.gz
cd experiments_cloud
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
```

**Lưu ý:** Nếu cụm yêu cầu module `cuda` hoặc `toolchain`, nạp module **trước** khi tạo venv/cài `torch` có CUDA.

Sau `bootstrap_cloud.sh`, nếu cần ghi đè PyTorch bản GPU (thường làm **một lần** sau bootstrap):

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

(Thay `cu124` bằng đuôi phù hợp với cụm; xác nhận trên [pytorch.org](https://pytorch.org).)

---

## 4. Tái hiện (reproducibility)

Các yếu tố ảnh hưởng trực tiếp tới so sánh công bằng giữa phương pháp:

- [`configs/experiment_grid.yaml`](configs/experiment_grid.yaml): `seeds`, `episode_steps`, `methods`, `latency`, `training`, `mode`.
- [`configs/contexts_min.yaml`](configs/contexts_min.yaml): danh sách `contexts` (id, dummy profile, metadata HOT, tùy chọn `transfer_bucket: source|target` cho tỷ số chuyển giao trong tổng hợp).

**Nguyên tắc:** Mọi thay đổi so với bản đính kèm nên được commit/ghi chú trong phụ lục thí nghiệm. Báo cáo nên nêu **phiên bản commit**, **phiên bản Python/torch/gymnasium/sb3**, và **GPU model + driver**.

---

## 5. Thứ tự chạy khuyến nghị trên server

1. **Smoke test** (nhanh, dummy):

   ```bash
   bash scripts/run_smoke.sh
   ```

2. **Baseline đầy đủ (dummy hoặc hot):**

   ```bash
   bash scripts/run_baselines.sh dummy
   # hoặc: export HOT_ENV_FACTORY="my_hot_adapter:create_env" && bash scripts/run_baselines.sh hot
   ```

3. **Đo latency:**

   ```bash
   bash scripts/run_latency.sh dummy
   ```

4. **Huấn luyện PPO** — lặp theo từng `(context_index, seed)`; ví dụ dummy, 4 context (0–3), seed 1:

   ```bash
   for i in 0 1 2 3; do
     bash scripts/run_train_ppo.sh dummy "$i" 1
   done
   ```

   Hoặc **multi-context một job** (một SB3 `DummyVecEnv` trên mỗi dòng trong [`contexts_min.yaml`](configs/contexts_min.yaml)):

   ```bash
   MULTI_CONTEXT=1 bash scripts/run_train_ppo.sh dummy 0 1
   ```

   Thêm `--device cuda` (hoặc `cpu`) nếu gọi trực tiếp module (xem `python -m src.train_ppo --help`).

5. **Đánh giá với policy đã train:** truyền `--policy-path` vào [`src/experiment_runner.py`](src/experiment_runner.py) cho các method cần load checkpoint SB3 (`ppo_static`, `ppo_multicontext`, `edge_only` — chỉ liên quan khi thiết kế thí nghiệm dùng cùng policy).

6. **Full matrix + tổng hợp:**

   ```bash
   bash scripts/run_matrix.sh dummy
   python -m src.summarize_results --input-dir results/raw --output-dir results/summary
   ```

Kết quả tóm tắt (sau `summarize_results`): [`results/summary/control_summary.csv`](results/summary/control_summary.csv) (bao gồm `deployment_score` và bản ghép latency `deployment_score_joint` nếu có file latency thô), [`results/summary/latency_summary.csv`](results/summary/latency_summary.csv), và khi có đủ rollout: [`transfer_metrics.csv`](results/summary/transfer_metrics.csv), [`transfer_efficiency_by_bucket.csv`](results/summary/transfer_efficiency_by_bucket.csv) (cần trường `transfer_bucket` trong context), [`adaptation_summary.csv`](results/summary/adaptation_summary.csv), [`deployment_score_by_context.csv`](results/summary/deployment_score_by_context.csv).

---

## 6. Slurm / HPC

- **Rollout chỉ (baseline/control, không huấn luyện):** dùng [`slurm/run_array.sbatch`](slurm/run_array.sbatch) — cấu hình **CPU** (không xin GPU) vì [`experiment_runner`](src/experiment_runner.py) không dùng GPU.
- **Huấn luyện PPO trên GPU:** dùng [`slurm/run_train_array.sbatch`](slurm/run_train_array.sbatch) — xin `#SBATCH --gres=gpu:1`, mỗi task `SLURM_ARRAY_TASK_ID` tương ứng `--context-index`. Có thể đặt `SEED`, `MODE`, `DEVICE` trong môi trường trước `sbatch`.

```bash
export MODE=dummy SEED=1 DEVICE=cuda
sbatch slurm/run_train_array.sbatch
```

Submit rollout:

```bash
export MODE=dummy
sbatch slurm/run_array.sbatch
```

---

## 7. HOT: bước tối thiểu để có bằng chứng học thuật “thật”

1. Cài toolchain HOT và EnergyPlus theo tài liệu HOT/BUILDINGS.
2. Sao chép [`src/hot_adapter_template.py`](src/hot_adapter_template.py) thành ví dụ `my_hot_adapter.py`, triển khai `create_env(context, seed, episode_steps)` ánh xạ các trường trong [`contexts_min.yaml`](configs/contexts_min.yaml) sang mô phỏng HOT.
3. Xuất:

   ```bash
   export PYTHONPATH="$PWD:$PYTHONPATH"
   export HOT_ENV_FACTORY="my_hot_adapter:create_env"
   ```

4. Chạy lại `run_baselines.sh hot`, `run_latency.sh hot`, `run_train_ppo.sh hot ...`, và `run_matrix.sh hot`.

Đừng đưa số liệu **dummy** vào bảng kết quả chính của manuscript.

---

## 8. Checklist bằng chứng tối thiểu (đối chiếu EAAI / kế hoạch thực thi)

Theo [`eaai_journal_execution_plan.md`](../docs/eaai_journal_execution_plan.md), mức **tối thiểu** gồm:

- Ít nhất **4** bối cảnh building–weather (file context hiện có 4 id dummy/HOT metadata).
- Ít nhất **3 seeds** (đã cấu hình trong `experiment_grid.yaml`).
- So sánh: static, ASHRAE-style rule, RL tĩnh, multi-context RL, cloud-only có trễ, edge-only, phương pháp edge–cloud đề xuất (ở mức code: xem **`methods`** trong YAML).
- p50 và p95 latency (edge vs cloud) — [`src/latency.py`](src/latency.py) dùng ước lượng phân vị tuyến tính trên mẫu (`numpy.percentile`).
- Năng lượng và comfort kèm khoảng tin cậy (summarize hiện dùng khoảng 95% dạng chuẩn hóa từ sai số chuẩn mẫu cho mean).

Đọc và ghi trong paper **rõ ràng** phần nào là heuristic proxy và phần nào là policy SB3 có checkpoint — xem mục 9.

---

## 9. Rủi ro paper đã xử lý trong mã và các hạng mục còn mở

**Đã có trong pipeline (để report minh bạch và hỗ trợ RQ):**

- Mỗi dòng rollout chứa `policy_backend`, `trained_policy_loaded`, [`deployment_score`](src/deployment.py) (tách năng lượng/vi phạm comfort/độ “lắc”; trọng số lấy từ [`policy_update.deployment_score_weights`](configs/experiment_grid.yaml)). Khi chạy [`summarize_results.py`](src/summarize_results.py) với cả control và latency thô, bổ sung `deployment_score_with_latency` và cột tổng hợp `deployment_score_joint` trong `control_summary`.
- Cảnh báo khi các method kiểu RL chạy **không** có `--policy-path`; `edge_cloud_adaptive` **không** nhận `--policy-path` (chỉ heuristic bias; xem [`controller_tags.py`](src/controller_tags.py)).
- Tổng hợp: `energy_regret_vs_oracle_kwh`, worst-case theo context (`transfer_metrics.csv`), tỷ số năng lượng target/source nếu có `transfer_bucket` (`transfer_efficiency_by_bucket.csv`), và `adaptation_gain` static so với adaptive (`adaptation_summary.csv`) — định nghĩa trong [`paper_metrics.py`](src/paper_metrics.py).
- Huấn luyện **multi-context** trong một process: [`train_ppo.py --multi-context`](src/train_ppo.py) hoặc `MULTI_CONTEXT=1 bash scripts/run_train_ppo.sh`.
- Docstring trong [`envs.DummyHVACEnv`](src/envs.py) nhắc `energy_kwh` là **proxy** từng bước.

**Vẫn nên nêu trong Limitations / Future work (chưa có module độc lập):**

- **Cổng triển khai tự động:** `max_comfort_regression` và tiêu chí “chấp nhận policy mới” chỉ có helper [`comfort_regression_violates`](src/deployment.py); chưa có pipeline so sánh hai checkpoint và rollback tự động.
- **Vòng log → retrain → deploy**, xuất TorchScript/ONNX, ablation guard chi tiết như trong [`eaai_journal_execution_plan.md`](../docs/eaai_journal_execution_plan.md): mục tiêu thiết kế, chưa được lấp đầy bằng các service riêng trong repo này.

**Lưu ý khi viết bảng điều khiển:** oracle regret dùng metric **năng lượng** làm chuẩn chính — nếu paper ưu tiên comfort hoặc đa mục tiêu, diễn giải hoặc bổ sung hàm agregation tương ứng trong phụ lục thí nghiệm.

---

## 10. Tải kết quả về và liên kết manuscript

```bash
scp -r user@HOST:~/experiments_cloud/results/raw ./backup_raw
scp -r user@HOST:~/experiments_cloud/results/summary ./backup_summary
scp -r user@HOST:~/experiments_cloud/artifacts/policies ./backup_policies
```

Đồ họa/bảng trong [`Project/manuscript_eaai/main.tex`](../manuscript_eaai/main.tex): cập nhật bằng số từ CSV sau khi chạy HOT và đông băng phiên bản mã/config.

---

*Tài liệu này phản ánh trạng thái codebase tại thời điểm biên soạn; khi chức năng mới được thêm, hãy cập nhật mục 8–9 tương ứng.*
