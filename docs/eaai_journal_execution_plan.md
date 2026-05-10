# EAAI Journal Execution Plan

Target journal: Engineering Applications of Artificial Intelligence (Elsevier)

Working topic: Latency-Aware Edge-Cloud Reinforcement Learning for Transferable HVAC Control under Weather and Building Distribution Shifts

## 1. Journal fit and submission constraints

Engineering Applications of Artificial Intelligence is a suitable target only if the paper is positioned as an applied AI contribution for a real engineering control problem, not as a generic reinforcement learning architecture.

Current fit:

- Engineering application: real-time HVAC control in buildings.
- AI contribution: distributed/adaptive reinforcement learning with deployment-aware policy execution.
- Engineering evidence: energy use, thermal comfort, latency, robustness, transferability.
- Reproducibility basis: HOT dataset/testbed, public building-weather contexts, fixed seeds, released code/configs.

Critical EAAI requirements to respect:

- The abstract must clearly state both the AI contribution and the engineering application.
- Undefined acronyms must not appear in the title or abstract.
- The manuscript must be single-column.
- Maximum manuscript length is 50 pages and file size must be below 100 MB.
- The journal uses double-anonymized review, so title page and anonymized manuscript must be separate.
- Data/code availability must be stated; research data should be deposited, cited, and linked.

## 2. Recommended paper angle

Do not submit the current proposal as an architecture-only paper. The publishable angle should be:

> A deployment-aware reinforcement learning framework that separates low-latency edge control from cloud/HPC policy adaptation, and evaluates the energy-comfort-latency trade-off across public HVAC transfer-learning scenarios.

Proposed final title options:

1. Latency-Aware Edge-Cloud Reinforcement Learning for Transferable HVAC Control
2. Deployment-Aware Reinforcement Learning for Edge-Cloud HVAC Control under Distribution Shift
3. Edge-Cloud Adaptive Reinforcement Learning for Real-Time Building Climate Control

Recommended title: **Latency-Aware Edge-Cloud Reinforcement Learning for Transferable HVAC Control**

Reason: concise, includes the AI method, distributed deployment idea, and engineering application. Avoid "HVAC" in the final title only if the journal enforces no undefined acronyms in title; safer expanded version:

**Latency-Aware Edge-Cloud Reinforcement Learning for Transferable Heating, Ventilation, and Air Conditioning Control**

## 3. Core research gap

The HOT paper establishes a large-scale transfer-learning dataset, similarity framework, and baseline transfer experiments. It does not fully answer deployment questions:

- How should a learned HVAC controller be split between latency-critical edge inference and compute-heavy cloud training?
- What control quality is lost or gained when inference is moved from cloud to edge?
- How often should policies be updated under realistic weather/building distribution shifts?
- Does multi-context training plus periodic cloud retraining improve transfer while preserving real-time latency?

This paper should target that gap.

## 4. Contribution claims

The paper should make 3 to 4 claims, each backed by quantitative evidence.

Contribution 1: A deployment-aware edge-cloud reinforcement learning architecture for HVAC control.

- Edge: state preprocessing, policy inference, safety guard, action dispatch, logging.
- Cloud/HPC: multi-context training, policy evaluation, policy selection, versioned deployment.
- Evidence: end-to-end control loop and policy update pipeline.

Contribution 2: A reproducible benchmark protocol for energy-comfort-latency evaluation.

- Uses HOT public contexts.
- Reports both control metrics and system metrics.
- Evidence: fixed scenario matrix, seeds, scripts, data statement.

Contribution 3: Quantitative analysis of edge-cloud trade-offs.

- Compares edge-local inference, cloud-remote inference, static edge deployment, and adaptive edge-cloud deployment.
- Evidence: p50/p95 latency, energy use, comfort violation, cumulative reward.

Contribution 4: Robustness and adaptation under distribution shift.

- Tests TMY-to-AMY, cross-location, cross-building, and occupancy-shift scenarios.
- Evidence: adaptation gain, transfer efficiency, worst-context regret, statistical confidence intervals.

## 5. Experimental design

### 5.1 Scenario matrix

Minimum viable matrix:

- Buildings: 4 archetypes from HOT
  - Small office
  - Medium office
  - Retail
  - School or hospital if compute permits
- Climate/weather:
  - Seattle or equivalent mild climate
  - Hot-humid climate
  - Cold climate
  - TMY weather and AMY real weather where available
- Operating shifts:
  - Original occupancy schedule
  - Shifted occupancy schedule
  - Optional: changed comfort bounds or setpoint preferences
- Seeds:
  - Minimum: 3 seeds
  - Stronger journal version: 5 seeds

If compute is limited, prioritize more seeds and fewer contexts over many contexts with one seed.

### 5.2 Methods to compare

Required baselines:

1. Static setpoint controller
2. ASHRAE-style rule-based controller
3. Static single-context Proximal Policy Optimization policy
4. Multi-context Proximal Policy Optimization policy without adaptation
5. Cloud-only inference policy with simulated or measured network delay
6. Edge-only inference policy without retraining
7. Proposed edge-cloud adaptive reinforcement learning

Recommended stronger baseline if time allows:

- Soft Actor-Critic or Twin Delayed Deep Deterministic Policy Gradient as a second continuous-control reinforcement learning baseline.

### 5.3 Proposed method

The proposed method should be named clearly and conservatively, for example:

**ECARL: Edge-Cloud Adaptive Reinforcement Learning**

Main components:

- Multi-context policy training on source building-weather contexts.
- Edge-local policy execution using TorchScript or Open Neural Network Exchange runtime.
- Safety guard for valid heating/cooling setpoints and action-rate limits.
- Periodic cloud retraining from logged trajectories and simulated domain-randomized contexts.
- Policy selection based on validation score:

```text
score = alpha * normalized_energy
      + beta  * comfort_violation_rate
      + gamma * p95_latency
      + delta * action_instability
```

- Policy deployment only if validation score improves and comfort constraints are not degraded.

### 5.4 Metrics

Control metrics:

- Energy use intensity in kWh/m2.
- Total HVAC energy in kWh.
- Thermal comfort violation rate: percentage of timesteps outside comfort band.
- Mean and worst-zone temperature deviation.
- Cumulative reward.
- Action instability: average absolute setpoint change per control step.

System metrics:

- p50 and p95 inference latency.
- End-to-end control-loop latency.
- Policy artifact size.
- Training wall-clock time.
- Policy update time.
- Rollback frequency if any candidate policy fails validation.

Transfer/adaptation metrics:

- Transfer efficiency relative to source-context performance.
- Adaptation gain before versus after policy update.
- Worst-context regret relative to the best policy for that context.
- Performance variance across seeds.

Statistical reporting:

- Report mean plus 95% bootstrap confidence interval.
- Use paired tests where the same contexts/seeds are evaluated across methods.
- Always show per-context results, not only averages.

## 6. Ablation studies

Minimum ablations:

1. Safety guard on versus off.
2. Single-context training versus multi-context training.
3. No update versus periodic update.
4. Edge inference runtime: PyTorch eager versus TorchScript or ONNX.

Optional ablations:

- Update frequency: every 1, 3, 7 simulated days.
- Policy size: small, medium, large actor network.
- Similarity-guided source selection versus random source selection.

## 7. Figures and tables

Required figures:

1. Edge-cloud architecture diagram.
2. Experimental scenario matrix.
3. Energy-comfort Pareto chart.
4. Latency distribution plot: edge versus cloud.
5. Adaptation under distribution shift over time.
6. Per-context robustness heatmap.

Required tables:

1. Dataset/scenario summary.
2. Method comparison: energy, comfort, reward, p95 latency.
3. Ablation results.
4. Training/deployment cost summary.

## 8. Manuscript structure

1. Introduction
   - HVAC energy and control motivation.
   - Gap between simulation-trained reinforcement learning and real-time deployment.
   - Why edge-cloud split is needed.
   - Contributions.

2. Related Work
   - Reinforcement learning for HVAC control.
   - Transfer learning and multi-building generalization.
   - Edge/cloud artificial intelligence for cyber-physical control.
   - Deployment-aware evaluation of intelligent control systems.

3. Problem Formulation
   - Building-control Markov decision process.
   - State, action, reward, constraints.
   - Distribution-shift and transfer setting.

4. Proposed Edge-Cloud Adaptive Reinforcement Learning Framework
   - Edge control loop.
   - Cloud/HPC retraining loop.
   - Policy validation and deployment protocol.
   - Safety guard.

5. Experimental Setup
   - HOT dataset/testbed.
   - Scenario matrix.
   - Baselines.
   - Implementation details.
   - Metrics and statistical testing.

6. Results
   - Control performance.
   - Latency and deployment performance.
   - Robustness and adaptation.
   - Ablation studies.

7. Discussion
   - Engineering interpretation.
   - Deployment implications.
   - Failure cases.
   - Limitations.

8. Conclusion
   - Main findings.
   - Practical implications.
   - Future work.

## 9. Draft abstract template

Heating, ventilation, and air conditioning systems are major energy consumers in buildings, and reinforcement learning has shown promise for optimizing the trade-off between energy use and thermal comfort. However, many reinforcement learning controllers are evaluated mainly in simulation and do not explicitly address real-time deployment constraints such as inference latency, policy staleness, and adaptation under weather or occupancy shifts. This paper proposes an edge-cloud adaptive reinforcement learning framework for transferable heating, ventilation, and air conditioning control. The edge layer performs low-latency policy inference, safety-constrained action validation, and operational logging, while the cloud or high-performance computing layer performs multi-context retraining, policy evaluation, and versioned deployment. Using public building-control scenarios from the HOT testbed, we compare the proposed framework against static setpoint control, ASHRAE-style rule-based control, static reinforcement learning, cloud-only inference, and edge-only deployment. The evaluation reports energy use, thermal comfort violations, cumulative reward, transfer efficiency, p50/p95 inference latency, and adaptation gains under distribution shift. The results are expected to show that edge-local inference can substantially reduce control-loop latency while periodic cloud retraining improves robustness to changing operating conditions. The study provides a reproducible deployment-aware benchmark protocol for intelligent building control and demonstrates how real-time engineering constraints can be integrated into reinforcement learning evaluation.

The final abstract must be updated with real numeric results before submission.

## 10. Draft highlights

Each highlight must be 85 characters or fewer for Elsevier submission.

- Edge-cloud reinforcement learning enables low-latency HVAC control.
- Public HOT scenarios support reproducible transfer evaluation.
- Policy updates improve robustness under weather distribution shifts.
- Latency, comfort, and energy are evaluated in one benchmark protocol.

## 11. Keywords

Use 4 to 6 keywords:

- Reinforcement learning
- Building control
- Edge computing
- Transfer learning
- Cyber-physical systems
- Energy efficiency

## 12. Acceptance-risk checklist

High-risk issues before submission:

- No real experiments: desk rejection risk is high.
- Only one building/context: too weak for Q1/Q2.
- Only PPO versus rule-based baseline: likely insufficient.
- No latency measurement: weakens the edge-cloud claim.
- No code/data statement: conflicts with journal reproducibility expectations.
- Abstract says "novel architecture" but results only show simulation reward: weak contribution.
- Undefined acronyms in title/abstract: explicit desk-reject risk.
- Manuscript not anonymized: review-process violation.

Minimum publishable evidence:

- At least 4 building-weather contexts.
- At least 3 random seeds.
- At least 4 baselines including rule-based and static reinforcement learning.
- p50/p95 latency comparison for edge versus cloud.
- Energy and comfort results with confidence intervals.
- Adaptation experiment under at least one realistic shift.
- Public repository or archived artifact with configs and result tables.

Strong journal evidence:

- 12 or more building-weather contexts.
- 5 random seeds.
- A second reinforcement learning baseline such as Soft Actor-Critic.
- Multi-context and similarity-guided source-selection ablation.
- Open-source reproducibility package with scripts and anonymized data links.

## 13. Eight-week execution plan

Week 1: Reproducibility setup

- Download/prepare HOT subset.
- Run one HOT environment episode end-to-end.
- Implement result logging schema.
- Confirm state/action/reward definitions.

Week 2: Baselines

- Run static and ASHRAE-style controllers.
- Train first single-context PPO policy.
- Produce first table: energy, comfort, reward.

Week 3: Multi-context training

- Train PPO across multiple building-weather contexts.
- Evaluate zero-shot transfer.
- Add seed control and confidence interval scripts.

Week 4: Edge inference and latency

- Export policy to TorchScript or ONNX.
- Implement edge inference wrapper and safety guard.
- Measure p50/p95 latency locally and with simulated cloud delay.

Week 5: Edge-cloud adaptation loop

- Implement log-to-retrain-to-deploy workflow.
- Add policy versioning and validation gate.
- Run first distribution-shift experiment.

Week 6: Full experiment matrix

- Run all baselines and proposed method across selected contexts.
- Run ablations: safety, update, multi-context training, runtime format.
- Freeze result tables.

Week 7: Manuscript drafting

- Write Introduction, Method, Experiment Setup, Results.
- Generate all figures and tables.
- Draft data/code availability statement.

Week 8: Journal packaging

- Convert to single-column Elsevier format.
- Prepare anonymized manuscript and separate title page.
- Prepare highlights file and graphical abstract.
- Final internal review against EAAI checklist.

## 14. Repository structure to create

Recommended project structure:

```text
Project/
  manuscript/
    main.tex
    references.bib
    highlights.docx or highlights.txt
    title_page.tex
    anonymized_main.tex
  experiments/
    configs/
    scripts/
    results/
    figures/
  src/
    controllers/
    edge/
    training/
    evaluation/
  artifacts/
    policies/
    logs/
  docs/
    data_statement.md
    reproducibility.md
```

## 15. Immediate next actions

1. Confirm compute resources: local only, cloud GPU, or university HPC/Slurm.
2. Confirm whether HOT code/data can be downloaded and run in the current environment.
3. Select the first four contexts for the minimum experiment matrix.
4. Implement baseline result table before writing the full manuscript.
5. Replace all expected-results language with actual numbers before submission.

