# Huong dan nhanh: dua thi nghiem EAAI HVAC len cloud de chay

Thu muc thi nghiem da setup:

```text
experiments_cloud/
```

Bundle da tao san:

```text
dist/eaai_hvac_cloud_bundle.tar.gz
```

## 1. Dong goi lai neu co thay doi

Tu root workspace:

```bash
bash experiments_cloud/scripts/package_for_cloud.sh
```

## 2. Upload len cloud

Vi du dung `scp`:

```bash
scp dist/eaai_hvac_cloud_bundle.tar.gz user@CLOUD_IP:~/
```

Hoac upload bang giao dien RunPod, Vast.ai, Google Drive, Kaggle, Colab, hoac storage cua cum HPC.

## 3. Giai nen va cai moi truong tren cloud

```bash
tar -xzf eaai_hvac_cloud_bundle.tar.gz
cd experiments_cloud
bash scripts/bootstrap_cloud.sh
source .venv/bin/activate
```

`bootstrap_cloud.sh` se:

- tao Python virtual environment;
- cai dependencies tu `requirements.txt`;
- tao cac thu muc `results/`, `artifacts/`, `logs/`;
- chay mot smoke test nho.

## 4. Kiem tra pipeline

```bash
bash scripts/run_smoke.sh
```

Neu thanh cong, kiem tra:

```bash
ls results/raw
ls results/summary
```

## 5. Chay baseline dummy de test may

```bash
bash scripts/run_baselines.sh dummy
bash scripts/run_latency.sh dummy
```

Ket qua dummy chi de test pipeline, khong dung lam ket qua paper.

## 6. Noi HOT dataset/testbed that

Can cai HOT toolkit va EnergyPlus theo huong dan cua HOT. Sau do tao file adapter rieng, vi du `my_hot_adapter.py`, co ham:

```python
def create_env(context, seed, episode_steps):
    # import HOT environment/toolkit tai day
    # doc context tu configs/contexts_min.yaml
    # tra ve Gymnasium-compatible env
    return env
```

Sau do export:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export HOT_ENV_FACTORY="my_hot_adapter:create_env"
```

Chay HOT mode:

```bash
bash scripts/run_baselines.sh hot
bash scripts/run_latency.sh hot
bash scripts/run_train_ppo.sh hot
bash scripts/run_matrix.sh hot
```

## 7. Chay tren Slurm/HPC

```bash
sbatch slurm/run_array.sbatch
```

Neu can doi so GPU/CPU/time, sua cac dong `#SBATCH` trong:

```text
experiments_cloud/slurm/run_array.sbatch
```

## 8. Tai ket qua ve may

Tu may local:

```bash
scp -r user@CLOUD_IP:~/experiments_cloud/results Project/cloud_results
scp -r user@CLOUD_IP:~/experiments_cloud/artifacts Project/cloud_artifacts
```

## 9. File can dua vao manuscript

Sau khi chay:

```text
results/summary/control_summary.csv
results/summary/latency_summary.csv
```

Dua so lieu vao:

```text
Project/manuscript_eaai/main.tex
```

Bang paper can it nhat:

- energy_kwh_mean va confidence interval;
- comfort_violation_rate_mean va confidence interval;
- cumulative_reward_mean va confidence interval;
- p50_latency_ms_mean;
- p95_latency_ms_mean.

## 10. Cau hinh quan trong

Sua context thi nghiem:

```text
experiments_cloud/configs/contexts_min.yaml
```

Sua seeds, episode length, latency, training timesteps:

```text
experiments_cloud/configs/experiment_grid.yaml
```

De paper du manh cho EAAI, nen chay toi thieu:

- 4 contexts;
- 3 seeds;
- static, ASHRAE-style, static reinforcement learning, edge-only, cloud-only, proposed edge-cloud adaptive;
- latency p50/p95;
- confidence interval.

