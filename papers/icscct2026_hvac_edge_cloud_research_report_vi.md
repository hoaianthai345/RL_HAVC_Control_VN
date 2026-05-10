# Bản thảo báo cáo nghiên cứu theo cấu trúc ICSCCT-2026 / Springer LNNS

**Tiêu đề đề xuất:** Điều khiển HVAC bằng học tăng cường biên–đám mây có xét độ trễ cho các tòa nhà bền vững  
**English title for submission:** Latency-Aware Edge–Cloud Reinforcement Learning for Sustainable HVAC Control  
**Hội nghị mục tiêu:** The International Conference on Sustainable Computing and Communication Technologies — ICSCCT 2026  
**Track phù hợp:** Sustainable AI and data analytics; sustainable technology applications; energy management using ICT  
**Trạng thái kết quả:** **CHƯA CÓ KẾT QUẢ THỰC NGHIỆM HỢP LỆ ĐỂ ĐIỀN**. Các phần kết quả và visualization được để trống theo yêu cầu.

> Ghi chú chuẩn nộp bài: trang ICSCCT yêu cầu bài định dạng theo Springer; trang Publications cho biết bài được chấp nhận, trình bày và đăng ký sẽ xuất bản trong Springer Lecture Notes in Networks and Systems. Bản này là bản thảo nội dung tiếng Việt theo cấu trúc LNNS; trước khi nộp cần chuyển sang tiếng Anh và đưa vào template Springer chính thức.

---

## Abstract

Heating, ventilation, and air-conditioning (HVAC) systems account for a substantial share of building energy demand, making intelligent HVAC control an important component of sustainable computing and smart-building operation. Reinforcement learning (RL) can optimize sequential control policies, but many RL-based HVAC studies emphasize simulation rewards while under-reporting deployment constraints such as inference latency, cloud connectivity, policy staleness, and transfer across heterogeneous building/weather contexts. This paper proposes a latency-aware edge–cloud RL framework for sustainable HVAC control. The edge layer performs low-latency inference, safety-constrained action filtering, and telemetry collection, while the cloud/HPC layer performs multi-context training, validation, and policy update scheduling. The experimental protocol compares static setpoint control, ASHRAE-style rules, trained PPO policies, edge-only deployment, cloud-only inference, and adaptive edge–cloud updating across multiple building/weather contexts. Energy use, comfort violation, action stability, transfer regret, adaptation gain, and p50/p95 inference latency are used as evaluation metrics. **No experimental results are reported yet; this draft defines the methodology and placeholders for later measured results.**

**Keywords:** sustainable computing; HVAC control; reinforcement learning; edge computing; cloud computing; smart buildings; latency-aware AI

---

## 1. Introduction

Buildings consume energy continuously to maintain thermal comfort, and HVAC operation is one of the main controllable loads in modern facilities. Improving HVAC control therefore contributes directly to sustainable computing and communication technologies: sensing, edge inference, cloud training, and data-driven decision support can reduce energy waste while preserving occupant comfort.

Reinforcement learning has become a promising approach because an RL agent can learn a control policy from interaction data rather than relying only on fixed setpoint schedules or manually tuned rules. However, HVAC control is a cyber-physical problem, not only an offline optimization benchmark. A deployable controller must satisfy at least four requirements:

1. **Energy efficiency:** reduce HVAC energy consumption under realistic weather and occupancy patterns.
2. **Comfort preservation:** avoid thermal discomfort and large zone-temperature deviations.
3. **Real-time operation:** generate actions within acceptable latency bounds.
4. **Transfer and adaptation:** remain useful when building envelope, climate, occupancy, or equipment behavior shifts.

Cloud-only inference simplifies model management but can be vulnerable to network latency and outages. Edge-only deployment lowers latency but can produce stale policies when operating conditions change. This motivates an edge–cloud split: the edge executes a validated policy close to the building, while the cloud/HPC layer retrains and validates candidate policies using larger compute capacity.

This paper studies that architecture for sustainable HVAC control and defines an experiment plan that can be executed with the current `experiments_cloud` codebase after several reproducibility fixes.

### 1.1 Contributions

The intended contributions are:

1. A latency-aware edge–cloud architecture for RL-based HVAC control.
2. A reproducible experiment protocol covering energy, comfort, transfer, adaptation, and inference latency.
3. A codebase execution plan for evaluating static, rule-based, PPO, edge-only, cloud-only, and adaptive edge–cloud controllers.
4. A results-interpretation template for writing the final ICSCCT paper once experiments are completed.

---

## 2. Related Work

### 2.1 RL for HVAC and building control

HVAC control can be modeled as a Markov decision process in which the controller observes building states and chooses setpoint actions. PPO is a widely used policy-gradient algorithm for continuous-control problems due to its clipped objective and practical training stability. In HVAC, PPO-like methods are attractive because setpoints are continuous or quasi-continuous and the controller must optimize long-horizon energy/comfort trade-offs.

### 2.2 Transfer learning in building-control benchmarks

Real buildings differ in construction, thermal mass, occupancy, equipment capacity, weather exposure, and operating schedules. A policy trained on one building/weather context may fail when deployed elsewhere. Public benchmarks such as HOT are relevant because they enable systematic transfer studies across many building scenarios instead of reporting results on a single hand-picked simulator.

### 2.3 Edge–cloud AI for sustainable cyber-physical systems

Edge–cloud AI separates latency-critical inference from compute-intensive training. In an HVAC context, edge deployment can run every control interval with low latency, while cloud/HPC infrastructure can periodically retrain policies on logged trajectories, evaluate candidates, and deploy updates. This structure aligns with sustainable computing goals because it uses communication and compute resources selectively rather than requiring continuous cloud-inference dependence.

---

## 3. System Model and Problem Formulation

### 3.1 Building-control Markov decision process

For each building context \(c\), HVAC control is modeled as:

\[
\mathcal{M}_c = \langle \mathcal{S}, \mathcal{A}, P_c, R_c, \gamma \rangle,
\]

where \(\mathcal{S}\) is the state space, \(\mathcal{A}\) is the action space, \(P_c\) is the context-conditioned transition function, \(R_c\) is the reward function, and \(\gamma\) is the discount factor.

A representative state vector is:

\[
s_t = [T^{in}_t, T^{out}_t, H_t, O_t, E_{t-1}, \sin(2\pi h_t/24), \cos(2\pi h_t/24)],
\]

where \(T^{in}\) is indoor temperature, \(T^{out}\) is outdoor temperature, \(H\) is humidity, \(O\) is occupancy or schedule intensity, \(E\) is recent HVAC energy, and \(h_t\) is hour of day.

The action is a pair of heating and cooling setpoints:

\[
a_t = [T^{heat}_{sp,t}, T^{cool}_{sp,t}].
\]

The reward should penalize energy use, comfort deviation, unsafe actions, and abrupt setpoint changes:

\[
r_t = -\left(w_e E_t + w_c D_t + w_v V_t + w_s \lVert a_t-a_{t-1}\rVert_1\right),
\]

where \(D_t\) is temperature deviation from the comfort band and \(V_t\) is a comfort-violation indicator.

### 3.2 Deployment-aware objective

The final evaluation should not rank controllers only by cumulative reward. A deployment-aware score is defined as:

\[
J = \alpha \bar{E} + \beta \bar{C} + \eta L_{95} + \lambda \bar{A},
\]

where \(\bar{E}\) is normalized energy use, \(\bar{C}\) is comfort-violation rate, \(L_{95}\) is p95 inference or end-to-end latency, and \(\bar{A}\) is mean action instability. Lower \(J\) is better. Before final experiments, all terms must be normalized or justified to avoid arbitrary weighting.

---

## 4. Proposed Edge–Cloud RL Framework

### 4.1 Edge layer

The edge layer executes the real-time control loop:

1. receive sensor and context observations;
2. normalize the observation vector;
3. run local policy inference;
4. apply safety constraints to setpoints;
5. send validated action to HVAC actuator or simulator;
6. log observation, action, reward components, latency, and safety events.

The edge layer is responsible for low-latency operation and fail-safe fallback. If the learned policy is unavailable, unsafe, or stale, the edge should fall back to a rule-based controller.

### 4.2 Cloud/HPC layer

The cloud or HPC layer performs:

1. multi-context PPO training;
2. offline evaluation of candidate checkpoints;
3. transfer analysis across source and target contexts;
4. latency-aware deployment scoring;
5. policy versioning and rollback;
6. periodic update scheduling.

### 4.3 Policy update rule

A candidate policy \(\pi'\) should replace incumbent policy \(\pi\) only if:

\[
J(\pi') < J(\pi) - \epsilon
\]

and comfort regression is bounded:

\[
\bar{C}(\pi') \leq \bar{C}(\pi) + \delta.
\]

This prevents energy-only improvements that substantially worsen comfort.

---

## 5. Methodology

### 5.1 Experimental contexts

The codebase currently defines four minimal contexts:

| ID | Building | Climate / shift | Transfer bucket |
|---|---|---|---|
| `s1_small_office_mild_tmy` | OfficeSmall | mild TMY | source |
| `s2_small_office_mild_amy` | OfficeSmall | mild variable weather | source |
| `s3_medium_office_hot_humid` | OfficeMedium | hot-humid building/climate shift | target |
| `s4_school_cold_occupancy_shift` | PrimarySchool | cold climate + occupancy shift | target |

These contexts are sufficient for smoke testing but weak for a final paper. The final study should expand to at least 12 contexts and 5 random seeds if compute permits.

### 5.2 Compared methods

The final experiment should compare:

1. **Static setpoint:** fixed heating/cooling setpoints.
2. **ASHRAE-style rule controller:** threshold-based setpoint adjustment.
3. **PPO-static:** PPO trained on one source context.
4. **PPO-multicontext:** PPO trained across source contexts.
5. **Edge-only:** trained policy executed locally without cloud adaptation.
6. **Cloud-only:** remote inference with simulated or measured network delay.
7. **Edge–cloud adaptive:** edge inference with cloud/HPC retraining and gated policy updates.

Important: the current codebase uses heuristic stand-ins for several PPO-named methods unless a trained policy checkpoint is supplied. Final paper experiments must not report those fallback outputs as RL results.

### 5.3 Metrics

The following metrics should be reported:

| Metric | Definition | Direction |
|---|---|---|
| Energy | Total or normalized HVAC energy per episode | lower better |
| Comfort violation rate | Fraction of steps outside comfort band | lower better |
| Mean temperature deviation | Mean distance outside comfort band | lower better |
| Cumulative reward | Sum of reward over an episode | higher better |
| Action instability | Mean action change magnitude | lower better |
| p50/p95 latency | Median and tail inference latency | lower better |
| Transfer regret | Energy gap against best method in same context/seed | lower better |
| Adaptation gain | Static-control energy minus adaptive-control energy | higher better if comfort is not worse |
| Deployment score | Normalized composite of energy, comfort, latency, instability | lower better |

### 5.4 Statistical reporting

For each metric, report mean and 95% confidence interval across seeds:

\[
CI_{95} = 1.96 \frac{s}{\sqrt{n}},
\]

where \(s\) is sample standard deviation and \(n\) is the number of seeds. If \(n<3\), mark the result as preliminary.

---

## 6. Results

**TODO: run experiments before filling this section. No experimental results are available yet.**

### 6.1 Control performance

_To be filled after running validated HOT or calibrated simulation experiments._

| Method | Energy mean ± CI | Comfort violation mean ± CI | Temp. deviation mean ± CI | Reward mean ± CI |
|---|---:|---:|---:|---:|
| Static | TODO | TODO | TODO | TODO |
| ASHRAE-style | TODO | TODO | TODO | TODO |
| PPO-static | TODO | TODO | TODO | TODO |
| PPO-multicontext | TODO | TODO | TODO | TODO |
| Edge-only | TODO | TODO | TODO | TODO |
| Cloud-only | TODO | TODO | TODO | TODO |
| Edge–cloud adaptive | TODO | TODO | TODO | TODO |

### 6.2 Latency performance

_To be filled after latency experiments._

| Method | p50 latency | p95 latency | Max latency | Notes |
|---|---:|---:|---:|---|
| Edge-only | TODO | TODO | TODO | TODO |
| Cloud-only | TODO | TODO | TODO | TODO |
| Edge–cloud adaptive | TODO | TODO | TODO | TODO |

### 6.3 Transfer and adaptation

_To be filled after source/target experiments._

| Method | Source energy | Target energy | Target/source ratio | Regret | Adaptation gain |
|---|---:|---:|---:|---:|---:|
| Static | TODO | TODO | TODO | TODO | TODO |
| PPO-static | TODO | TODO | TODO | TODO | TODO |
| PPO-multicontext | TODO | TODO | TODO | TODO | TODO |
| Edge–cloud adaptive | TODO | TODO | TODO | TODO | TODO |

---

## 7. Visualization

**TODO: create figures after experiments. No visualization is available yet.**

Planned figures:

1. **Figure 1 — Edge–cloud control architecture:** sensor/edge/cloud/HPC/policy-update flow.
2. **Figure 2 — Energy vs comfort trade-off:** scatter plot by method, context, and seed.
3. **Figure 3 — p50/p95 latency comparison:** grouped bar chart for edge-only, cloud-only, and adaptive modes.
4. **Figure 4 — Transfer regret by target context:** bar chart with CI.
5. **Figure 5 — Adaptation gain under shift:** before/after or static-vs-adaptive comparison.
6. **Figure 6 — Example episode trace:** indoor temperature, comfort band, and setpoints over time.

---

## 8. Discussion Template for Later Results

After results are available, write the discussion using the following logic:

1. **Energy–comfort trade-off:** identify whether energy reduction is achieved without comfort degradation.
2. **Latency:** compare edge vs cloud p95 latency and discuss whether cloud delay materially affects deployment score.
3. **Transfer:** check whether multicontext training lowers regret on target contexts compared with single-context PPO.
4. **Adaptation:** evaluate whether edge–cloud updates improve target-context performance relative to static or edge-only policies.
5. **Failure cases:** report contexts where the proposed method fails or worsens comfort.
6. **Sustainability implication:** translate control improvements into cautious energy-efficiency implications without over-claiming beyond measured data.

---

## 9. Codebase Experiment Plan

This plan is based on the code audit of `experiments_cloud`.

### Phase 0 — Reproducibility fixes before experiments

**Goal:** prevent invalid or contaminated results.

1. **Run-scoped outputs**
   - Change raw output structure from `results/raw/*.csv` to `results/raw/<run_name>/*.csv`.
   - Change summary input to require an explicit run directory or manifest.
   - Write `results/summary/<run_name>/manifest.json` listing all raw files, commit hash, config hash, command, and timestamp.

2. **Policy provenance guard**
   - Fail if methods `ppo_static`, `ppo_multicontext`, or `edge_only` run without `--policy-path`, unless `--allow-heuristic-proxy` is set.
   - Add `policy_path`, `policy_sha256`, `training_run_name`, `total_timesteps`, `seed`, and `trained_policy_loaded` to every result row.

3. **HOT adapter validation**
   - Implement concrete `HOT_ENV_FACTORY` adapter.
   - Add one short HOT smoke episode.
   - Record HOT dataset version, building ID, weather file, and EnergyPlus version.

4. **Latency fairness fix**
   - Recreate or reset the environment for each `(context, method)` in `src/latency.py`.
   - Prefer a fixed observation trace for latency-only measurements.

5. **Trace logging**
   - Add optional per-step CSV traces: `results/traces/<run_name>/<context>_<seed>_<method>.csv`.

### Phase 1 — Local dummy validation

**Purpose:** verify pipeline only; do not use results in paper.

Commands after fixes:

```bash
cd experiments_cloud
source .venv/bin/activate
python -m compileall -q src
bash scripts/run_smoke.sh
```

Expected validation:

- summaries include only files from the smoke run;
- warnings appear if heuristic proxies are used;
- trace files are written when enabled;
- no stale CSVs are included.

### Phase 2 — PPO training runs

Train separate and multicontext PPO policies.

```bash
# Single-context PPO
python -m src.train_ppo \
  --mode hot \
  --context-index 0 \
  --seed 1 \
  --total-timesteps 200000 \
  --run-name ppo_static_ctx0_seed1

# Multicontext PPO
MULTI_CONTEXT=1 python -m src.train_ppo \
  --mode hot \
  --seed 1 \
  --total-timesteps 500000 \
  --run-name ppo_multicontext_seed1 \
  --multi-context
```

Repeat for seeds `[1, 2, 3, 4, 5]` if compute permits.

### Phase 3 — Control evaluation matrix

Evaluate all methods across all contexts/seeds with trained checkpoints.

```bash
python -m src.experiment_runner \
  --mode hot \
  --methods static ashrae ppo_static ppo_multicontext edge_only cloud_only edge_cloud_adaptive \
  --seeds 1 2 3 4 5 \
  --episode-steps 288 \
  --output-dir results/raw/final_hot_control \
  --run-name final_hot_control \
  --policy-path artifacts/policies/<validated_policy>.zip
```

If separate checkpoints are needed per method, add method-specific policy-path configuration rather than a single global path.

### Phase 4 — Latency evaluation

Run latency in two modes:

1. **Controller-only latency:** no simulated network delay, measures local inference overhead.
2. **Deployment latency:** edge delay and cloud delay configured to reflect measured network conditions.

```bash
python -m src.latency \
  --mode hot \
  --methods edge_only cloud_only edge_cloud_adaptive \
  --repetitions 1000 \
  --warmup 100 \
  --cloud-delay-ms 80 \
  --output-dir results/raw/final_hot_latency \
  --run-name final_hot_latency
```

### Phase 5 — Summary and visualization generation

```bash
python -m src.summarize_results \
  --input-dir results/raw/final_hot_control \
  --output-dir results/summary/final_hot

python -m src.summarize_results \
  --input-dir results/raw/final_hot_latency \
  --output-dir results/summary/final_hot_latency
```

Then generate planned figures from validated summary CSVs.

### Phase 6 — Write final results and comments

After results are generated:

1. Fill Section 6 tables with measured means and CIs.
2. Fill Section 7 figures.
3. Write discussion using Section 8 template.
4. Add a limitations paragraph distinguishing HOT/simulation results from real deployment.
5. Run a final reproducibility audit: commands, configs, hashes, raw files, and generated figures.

---

## 10. Threats to Validity

1. **Simulation-to-reality gap:** HOT/EnergyPlus results may not capture all real HVAC actuator delays, sensor noise, and occupancy uncertainty.
2. **Policy naming risk:** if heuristic fallbacks remain enabled, results may be misinterpreted as learned RL results.
3. **Metric weighting:** deployment score requires normalization and sensitivity analysis.
4. **Latency realism:** simulated cloud delay must be replaced or supplemented with measured network latency for deployment claims.
5. **Context coverage:** four contexts are insufficient for strong generalization claims; more buildings/weather conditions are needed.

---

## 11. Conclusion

This draft defines a sustainable HVAC-control study suitable for the ICSCCT-2026 theme of sustainable computing and communication technologies. The proposed edge–cloud RL framework separates low-latency control from cloud/HPC retraining and introduces a deployment-aware evaluation protocol. At this stage, no quantitative result is reported. The next step is to fix the experiment pipeline, connect a validated HOT/EnergyPlus adapter, train real PPO policies, run the full control and latency matrix, and then fill the results, visualization, and discussion sections with measured evidence.

---

## References and sources

1. ICSCCT 2026 — Paper Submissions. https://icscct.com/paper-submissions/  
2. ICSCCT 2026 — Publications. https://icscct.com/publications/  
3. ICSCCT 2026 — Call for Papers. https://icscct.com/call-for-papers/  
4. Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347  
5. Berkes, A., Bengio, Y., Rolnick, D., Vakalis, D. “A HOT Dataset: 150,000 Buildings for HVAC Operations Transfer Research.” ACM BuildSys 2025. Dataset/source listed in local references: https://huggingface.co/datasets/BuildingBench/HOT  
6. Crawley, D. B. et al. “EnergyPlus: Creating a New-Generation Building Energy Simulation Program.” Energy and Buildings, 2001. DOI: 10.1016/S0378-7788(00)00114-6
