# Huong dan chay thi nghiem tren cloud cho bai EAAI HVAC

**Huong dan day du tren server GPU/HPC (tieng Viet, checklist paper + gap hoc thuat):** [GPU_SERVER_RUNBOOK_VI.md](GPU_SERVER_RUNBOOK_VI.md).

Thu muc nay la goi thi nghiem cloud-ready cho bai:

**Latency-Aware Edge-Cloud Reinforcement Learning for Transferable Heating, Ventilation, and Air Conditioning Control**

Muc tieu:

- chay baseline HVAC control;
- do latency edge/cloud;
- train/evaluate Proximal Policy Optimization;
- tong hop bang ket qua de dua vao manuscript EAAI;
- dong goi duoc de upload len cloud GPU hoac cum HPC/Slurm.

## 1. Cau truc thu muc

```text
experiments_cloud/
  configs/
    contexts_min.yaml
    experiment_grid.yaml
  scripts/
    bootstrap_cloud.sh
    package_for_cloud.sh
    run_baselines.sh
    run_latency.sh
    run_matrix.sh
    run_smoke.sh
    run_train_ppo.sh
  slurm/
    run_array.sbatch
    run_train_array.sbatch
  GPU_SERVER_RUNBOOK_VI.md
  src/
    controllers.py
    envs.py
    experiment_runner.py
    latency.py
    metrics.py
    summarize_results.py
    train_ppo.py
  artifacts/
  logs/
  results/
  requirements.txt
```

## 2. Hai che do chay

### Che do `dummy`

Dung de kiem tra pipeline tren laptop/cloud ma chua can tai HOT/EnergyPlus.

- Co moi truong HVAC gia lap nho.
- Chay duoc baseline, latency, summary.
- Ket qua chi dung de debug pipeline, khong dung de viet paper.

### Che do `hot`

Dung de chay thi nghiem that tren HOT dataset/testbed.

- Can tai HOT dataset `BuildingBench/HOT`.
- Can cai EnergyPlus 24.1+ neu HOT adapter yeu cau.
- Can cung cap factory tao environment qua bien moi truong `HOT_ENV_FACTORY`.

Vi du:

```bash
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
```

Trong do `my_hot_adapter.py` nam tren `PYTHONPATH` va co ham:

```python
def create_env(context, seed, episode_steps):
    ...
    return env
```

`env` nen tuong thich Gymnasium: co `reset(seed=...)`, `step(action)`, `action_space`, `observation_space`.

## 3. Dong goi de upload len cloud

Chay tu root workspace:

```bash
bash experiments_cloud/scripts/package_for_cloud.sh
```

File tao ra:

```text
dist/eaai_hvac_cloud_bundle.tar.gz
```

Upload file nay len cloud bang `scp`, Google Drive, Hugging Face Space, AWS S3, hoac cong cu cua cum HPC.

## 4. Cai dat tren cloud

Sau khi upload len cloud:

```bash
tar -xzf eaai_hvac_cloud_bundle.tar.gz
cd experiments_cloud
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
```

Kiem tra smoke test:

```bash
bash scripts/run_smoke.sh
```

Neu thanh cong, ban se thay file ket qua trong:

```text
results/raw/
results/summary/
```

## 5. Chay baseline toi thieu

Che do dummy:

```bash
bash scripts/run_baselines.sh dummy
```

Che do HOT:

```bash
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
bash scripts/run_baselines.sh hot
```

Baseline mac dinh:

- static
- ashrae
- ppo_static
- ppo_multicontext
- edge_only
- cloud_only
- edge_cloud_adaptive

Luu y: trong `dummy`, cac controller PPO chi la heuristic proxy de kiem tra pipeline. Trong `hot`, can train/load policy that truoc khi dua ket qua vao paper.

## 6. Chay latency experiment

```bash
bash scripts/run_latency.sh dummy
```

Ket qua latency nam o:

```text
results/raw/latency_*.csv
```

Metric can dua vao paper:

- p50 inference latency;
- p95 inference latency;
- p50/p95 end-to-end latency;
- so sanh edge-local va cloud-remote.

## 7. Train PPO tren cloud GPU

Smoke training voi dummy environment:

```bash
bash scripts/run_train_ppo.sh dummy
```

Chay voi HOT:

```bash
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
bash scripts/run_train_ppo.sh hot
```

Policy duoc luu vao:

```text
artifacts/policies/
```

De chay that cho paper, tang `total_timesteps` trong `configs/experiment_grid.yaml`.

## 8. Chay full matrix

```bash
bash scripts/run_matrix.sh dummy
```

Voi HOT:

```bash
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
bash scripts/run_matrix.sh hot
```

Sau khi chay xong, tong hop:

```bash
python -m src.summarize_results --input-dir results/raw --output-dir results/summary
```

## 9. Chay tren Slurm/HPC

Sua cac dong `#SBATCH` trong:

```text
slurm/run_array.sbatch          # rollout/baseline CPU, khong GPU
slurm/run_train_array.sbatch    # train PPO, xin GPU
```

Submit rollout:

```bash
sbatch slurm/run_array.sbatch
```

Submit train PPO (vi du bien moi truong):

```bash
export MODE=dummy SEED=1 DEVICE=cuda
sbatch slurm/run_train_array.sbatch
```

Moi array task chay mot context index (`SLURM_ARRAY_TASK_ID`). Ket qua rollout ghi ve `results/raw/`; policy train ghi ve `artifacts/policies/`.

## 10. Ket qua can dong bang cho manuscript

Sau `python -m src.summarize_results` (tu dong khi chay cac script `run_*`):

- [`results/summary/control_summary.csv`](results/summary/control_summary.csv): `_mean` / `_ci95` cho energy_kwh, comfort_violation_rate, mean_temperature_deviation, cumulative_reward, action_instability, deployment_score (neu co trong raw), deployment_score_joint (neu co latency tho de ghep).
- [`results/summary/latency_summary.csv`](results/summary/latency_summary.csv): latence p50/p95 theo context/method.
- [`results/summary/transfer_metrics.csv`](results/summary/transfer_metrics.csv): mean_energy_regret_vs_oracle_kwh, worst_context_regret_energy_kwh_* (oracle theo nang luong cung mode/context/seed).
- [`results/summary/transfer_efficiency_by_bucket.csv`](results/summary/transfer_efficiency_by_bucket.csv): can `transfer_bucket: source|target` trong [`configs/contexts_min.yaml`](configs/contexts_min.yaml).
- [`results/summary/adaptation_summary.csv`](results/summary/adaptation_summary.csv): adaptation_gain_energy_kwh_* (static minus edge_cloud_adaptive).
- [`results/summary/deployment_score_by_context.csv`](results/summary/deployment_score_by_context.csv): diem trien khai co kem p95 latency neu co file latency tho.

Raw rollout con co `policy_backend`, `trained_policy_loaded` de giai thich baseline heuristic vs SB3.

## 11. Quy tac de ket qua du manh cho EAAI

Toi thieu:

- 4 building-weather contexts;
- 3 seeds;
- static + ASHRAE-style + static RL + proposed edge-cloud adaptive RL;
- p50/p95 latency;
- energy va comfort co confidence interval.

Manh hon:

- 12 contexts;
- 5 seeds;
- them Soft Actor-Critic hoac Twin Delayed Deep Deterministic Policy Gradient;
- ablation safety guard, update frequency, single-context vs multi-context training.

## 12. Thu tu chay khuyen nghi

1. `bash scripts/run_smoke.sh`
2. `bash scripts/run_baselines.sh dummy`
3. `bash scripts/run_latency.sh dummy`
4. Noi adapter HOT va chay `bash scripts/run_baselines.sh hot`
5. Train PPO that bang `bash scripts/run_train_ppo.sh hot`
6. Chay full matrix bang `bash scripts/run_matrix.sh hot`
7. Tong hop CSV va dua so vao `Project/manuscript_eaai/main.tex`

