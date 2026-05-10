# Implementation Plan: Vietnam Multi-City HVAC RL

## Mục tiêu
Train và evaluate edge-cloud adaptive RL trên 4 thành phố Việt Nam:
- **Source (train)**: HCMC (0A), Can Tho (0A)
- **Target (transfer)**: Da Nang (1A), Hanoi (2A)

---

## Tổng quan file đã có

```
experiments_cloud/
  configs/
    experiment_grid.yaml          ← training hyperparams
    contexts_vietnam_multicity.yaml ← 10 contexts (4 cities) ✓ DONE
  data/vietnam_hot/
    weather/
      VNM_SVN_Ho.Chi.Minh...TMYx.epw   ✓
      VNM_SVN_Can.Tho...TMYx.epw        ✓
      VNM_CVN_Da.Nang...TMYx.epw        ✓
      VNM_NVN_Hanoi...TMYx.epw          ✓
      weather/real_base/VNM_SVN_Ho.Chi.Minh...AMY_20XX.epw  ✓ (2020-2024)
    buildings/
      processed/base/*_HoChiMinh.epJSON  ✓ (13 archetypes)
  src/
    envs.py             ← DummyHVACEnv + hot_adapter stub
    train_ppo.py        ← PPO training
    controllers.py      ← static, ashrae, ppo controllers
    experiment_runner.py
    latency.py
    summarize_results.py
```

---

## Phase 0: Validate pipeline — dummy mode (local, ngay bây giờ)

**Thời gian**: 30 phút  
**Mục đích**: xác nhận contexts mới chạy được trước khi đụng đến EnergyPlus

### Bước 0.1 — Cập nhật experiment_grid.yaml

```yaml
# configs/experiment_grid.yaml — thêm/sửa:
mode: dummy
contexts_file: configs/contexts_vietnam_multicity.yaml  # thêm dòng này
episode_steps: 288
seeds: [1, 2, 3]
```

### Bước 0.2 — Smoke test với Vietnam contexts

```bash
cd experiments_cloud
source .venv/bin/activate

# Test pipeline end-to-end (dummy mode, 3 methods, 1 seed)
python -m src.experiment_runner \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_vietnam_multicity.yaml \
  --mode dummy \
  --methods static ashrae edge_cloud_adaptive \
  --seeds 1 \
  --episode-steps 32 \
  --output-dir results/raw/vietnam_smoke \
  --run-name vietnam_smoke
```

### Bước 0.3 — Smoke train PPO

```bash
python -m src.train_ppo \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_vietnam_multicity.yaml \
  --mode dummy --context-index 0 --seed 1
```

**Pass criteria**: không có lỗi, có file CSV trong `results/raw/vietnam_smoke/`

---

## Phase 1: HOT Adapter — kết nối EnergyPlus (cần server H100)

**Thời gian**: 2-3 ngày  
**Mục đích**: thay DummyHVACEnv bằng EnergyPlus simulation thật

### Bước 1.1 — Cài Sinergym trên server

```bash
pip install sinergym[extras]
# Sinergym tự cài EnergyPlus 24.1 hoặc cài thủ công:
# sudo apt-get install energyplus=24.1.0
```

### Bước 1.2 — Implement `src/hot_adapter_vietnam.py`

File này implement hàm `create_env(context, seed, episode_steps)` theo đúng interface của `hot_adapter_template.py`.

**Input từ context YAML**:
```yaml
hot:
  epw_file:   data/vietnam_hot/weather/VNM_NVN_Hanoi...TMYx.epw
  model_file: data/vietnam_hot/buildings/processed/base/OfficeSmall__STD2013/OfficeSmall_STD2013_HoChiMinh.epJSON
```

**Implement**:
```python
# src/hot_adapter_vietnam.py
import sinergym
from sinergym.utils.wrappers import NormalizeObservation

def create_env(context, seed, episode_steps):
    epw  = context["hot"]["epw_file"]
    idf  = context["hot"]["model_file"]   # epJSON → cần convert hoặc dùng Sinergym trực tiếp
    env  = sinergym.make(
        "Eplus-office-mixed-continuous-v1",
        weather_file=epw,
        idf_file=idf,
        episode_length=episode_steps,
        seed=seed,
    )
    return NormalizeObservation(env)
```

**Lưu ý**: Sinergym dùng `.idf`, HOT dùng `.epJSON` — cần một bước convert:
```bash
energyplus --convert-only OfficeSmall_STD2013_HoChiMinh.epJSON
# → sinh ra OfficeSmall_STD2013_HoChiMinh.idf
```

Hoặc dùng Sinergym >= 3.5 hỗ trợ epJSON trực tiếp.

### Bước 1.3 — Set environment variable và test

```bash
export HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Chạy 1 episode test
python -c "
from src.hot_adapter_vietnam import create_env
import yaml
ctx = yaml.safe_load(open('configs/contexts_vietnam_multicity.yaml'))['contexts'][0]
env = create_env(ctx, seed=1, episode_steps=96)
obs, _ = env.reset()
print('obs shape:', obs.shape)
obs, rew, done, trunc, info = env.step(env.action_space.sample())
print('reward:', rew, 'info:', info)
env.close()
"
```

**Pass criteria**: episode chạy được, `info` có `energy_kwh` và `comfort_violation`

---

## Phase 2: Training — trên H100 (cần cloud server)

**Thời gian**: 8-12 giờ GPU  
**Mục đích**: train PPO trên source cities, evaluate trên target cities

### Bước 2.1 — Cập nhật training config

```yaml
# configs/experiment_grid.yaml
training:
  total_timesteps: 500000   # từ 20000 lên 500k cho hot mode
  n_steps: 256
  batch_size: 256
  learning_rate: 0.0003
  device: cuda
```

### Bước 2.2 — Train single-context PPO (source cities)

```bash
# Train trên HCMC OfficeSmall (context-index 0)
DEVICE=cuda HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env" \
  bash scripts/run_train_ppo.sh hot 0 1   # mode, context_index, seed

# Train trên HCMC OfficeSmall seed 2, 3
for SEED in 2 3; do
  DEVICE=cuda HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env" \
    bash scripts/run_train_ppo.sh hot 0 $SEED &
done
wait
```

### Bước 2.3 — Train multi-context PPO (source cities chung)

```bash
MULTI_CONTEXT=1 DEVICE=cuda HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env" \
  bash scripts/run_train_ppo.sh hot
```

### Bước 2.4 — Run all baselines

```bash
HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env" \
  bash scripts/run_baselines.sh hot
```

### Bước 2.5 — Run latency experiments

```bash
bash scripts/run_latency.sh hot   # đo p50/p95 latency edge vs cloud
```

---

## Phase 3: Transfer Evaluation

**Thời gian**: 2-4 giờ  
**Mục đích**: evaluate policy trained trên source → apply lên target cities

### Bước 3.1 — Evaluate trained policy trên target cities

```bash
# Load policy trained trên HCMC, eval trên Hanoi (context-index 8 trong multicity yaml)
python -m src.experiment_runner \
  --config configs/experiment_grid.yaml \
  --contexts configs/contexts_vietnam_multicity.yaml \
  --mode hot \
  --methods ppo_multicontext edge_cloud_adaptive \
  --context-indices 6 7 8 9 \   # Da Nang và Hanoi contexts
  --seeds 1 2 3 \
  --policy-path artifacts/policies/ppo_multicontext_best.zip \
  --output-dir results/raw/transfer_eval
```

### Bước 3.2 — Transfer efficiency metrics

```bash
python -m src.summarize_results \
  --input-dir results/raw/transfer_eval \
  --output-dir results/summary/transfer \
  --transfer-source hcmc cantho \
  --transfer-target danang hanoi
```

---

## Phase 4: AMY Weather Shift Test (adaptation gain)

**Mục đích**: test policy trên real weather (AMY 2022-2024) → đo adaptation gain

```bash
# Tạo contexts với AMY weather thay TMYx
python scripts/gen_amy_contexts.py \
  --base-contexts configs/contexts_vietnam_multicity.yaml \
  --amy-years 2022 2023 2024 \
  --city hcmc \
  --out configs/contexts_vietnam_amy.yaml

# Eval với AMY
HOT_ENV_FACTORY="src.hot_adapter_vietnam:create_env" \
python -m src.experiment_runner \
  --contexts configs/contexts_vietnam_amy.yaml \
  --mode hot --methods edge_cloud_adaptive --seeds 1 2 3 \
  --output-dir results/raw/amy_shift
```

---

## Phase 5: Compile kết quả → manuscript

```bash
python -m src.summarize_results \
  --input-dir results/raw \
  --output-dir results/summary/final

# CSV files cần điền vào manuscript:
# results/summary/final/control_summary.csv     → Table 3
# results/summary/final/transfer_metrics.csv    → Table 4
# results/summary/final/latency_summary.csv     → Table 5
```

---

## File cần viết mới (chưa có)

| File | Nội dung | Phase |
|------|----------|-------|
| `src/hot_adapter_vietnam.py` | EnergyPlus env via Sinergym + Vietnam EPW | 1 |
| `scripts/convert_epjson.sh` | Batch convert epJSON → idf nếu cần | 1 |
| `scripts/gen_amy_contexts.py` | Generate AMY weather contexts | 4 |
| `configs/contexts_vietnam_amy.yaml` | AMY contexts cho HCMC 2022-2024 | 4 |

## File cần sửa

| File | Thay đổi | Phase |
|------|----------|-------|
| `configs/experiment_grid.yaml` | `total_timesteps: 500000`, thêm `device: cuda` | 2 |
| `configs/contexts_vietnam_multicity.yaml` | Đã có ✓ | - |
| `src/envs.py` | Không sửa — `make_env("hot")` gọi `HOT_ENV_FACTORY` đã đúng | - |

---

## Thứ tự thực hiện

```
Tuần 1 (local):
  Phase 0 → validate pipeline dummy mode với Vietnam contexts

Tuần 2 (cloud server setup):
  Phase 1 → implement + test hot_adapter_vietnam.py
             chạy 1 episode EnergyPlus + Vietnam EPW thành công

Tuần 3 (H100 training):
  Phase 2 → train PPO 500k steps × 3 seeds × source cities
  Phase 3 → transfer eval lên Da Nang + Hanoi

Tuần 4 (analysis + write):
  Phase 4 → AMY weather shift test
  Phase 5 → compile CSV → điền kết quả vào manuscript
```

---

## Chi phí ước tính (sau tối ưu)

| Phase | Giờ H100 | Chi phí (67,925 VND/h) |
|-------|----------|------------------------|
| Phase 2 (training) | ~8h | ~543k VND |
| Phase 3 (eval)     | ~3h | ~204k VND |
| Phase 4 (AMY)      | ~2h | ~136k VND |
| **Tổng**           | **~13h** | **~883k VND** |
