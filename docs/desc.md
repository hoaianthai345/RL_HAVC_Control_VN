# Edge–Cloud Distributed Reinforcement Learning for Real-Time HVAC Control

## 1. Introduction

Heating, Ventilation, and Air Conditioning (HVAC) systems account for a significant portion of global energy consumption in buildings. Recent advances in Reinforcement Learning (RL) have demonstrated strong potential in optimizing HVAC control policies, particularly in simulated environments. However, deploying RL-based controllers in real-world systems introduces critical challenges related to latency, scalability, and adaptability.

Most existing approaches focus on offline training and evaluation using simulation platforms such as EnergyPlus. While effective in controlled settings, these methods do not address the constraints of real-time deployment, where decisions must be made within strict latency bounds and under continuously changing environmental conditions.

This research proposes an **Edge–Cloud Distributed Reinforcement Learning architecture** that bridges the gap between simulation-based optimization and real-time operational control.

---

## 2. Problem Statement

Current RL-based HVAC control systems suffer from a fundamental trade-off:

- **Real-time control requires low-latency inference**, which favors deployment on edge devices close to the building.
- **Effective learning and adaptation require large-scale computation**, which is only feasible on cloud or High-Performance Computing (HPC) infrastructure.

Existing systems typically fail to balance these requirements, leading to either:

- high latency (cloud-based inference), or  
- poor adaptability (static edge deployment).

---

## 3. Research Objectives

The main objective of this research is to design and evaluate a **distributed HVAC control system** that:

1. Enables **low-latency real-time control** via edge-based RL inference  
2. Supports **scalable policy training and adaptation** using cloud/HPC resources  
3. Maintains a balance between **energy efficiency** and **thermal comfort**  
4. Facilitates **continuous learning** from real-world operational data  

---

## 4. Proposed Architecture

### 4.1 System Overview

The proposed system consists of two main components:

#### Edge Layer
- Collects real-time sensor data
- Performs state preprocessing
- Executes RL policy inference
- Applies safety constraints
- Sends control signals to HVAC actuators

#### Cloud/HPC Layer
- Aggregates operational data from edge devices
- Performs distributed retraining of RL policies
- Runs large-scale simulations across multiple building contexts
- Evaluates and selects optimal policies
- Deploys updated policies back to edge devices

---

### 4.2 Architecture Diagram
```

Sensors → Edge Controller → HVAC System → Environment  
↑ ↓  
| |  
Policy Update Data Logging  
| |  
Cloud / HPC Layer (Training, Simulation, Evaluation)

```

---

## 5. System Pipeline

### 5.1 Online Control Loop (Edge)

1. Sensor data acquisition  
2. State preprocessing  
3. Policy inference:  
```

a_t = π(s_t)

```
4. Safety validation  
5. HVAC actuation  
6. Outcome observation  
7. Logging for future training  

---

### 5.2 Offline Training Loop (Cloud/HPC)

1. Data aggregation from edge devices  
2. Data preprocessing and scenario generation  
3. Distributed RL training using simulation environments  
4. Policy evaluation against baselines  
5. Selection of optimal policy  
6. Deployment to edge  

---

## 6. Research Questions

- **RQ1:** Can edge-based inference significantly reduce latency while maintaining control performance?  
- **RQ2:** Does periodic retraining on cloud/HPC improve robustness under changing conditions?  
- **RQ3:** Can distributed learning across multiple building contexts improve generalization and transferability?  

---

## 7. Methodology

### 7.1 Environment and Dataset

This research will leverage the **HOT (HVAC Optimization Testbed)** dataset, which provides a large-scale simulation framework with diverse building configurations, climate conditions, and operational scenarios.

### 7.2 Reinforcement Learning Framework

- Algorithm: Proximal Policy Optimization (PPO)  
- State representation:
- Indoor temperature  
- Outdoor temperature  
- Humidity  
- Occupancy  
- Energy consumption  
- Time features  

- Action space:
- Heating setpoint  
- Cooling setpoint  
- Deadband adjustment  

- Reward function:
```

r_t = w_e * r_energy + w_c * r_comfort + w_s * r_stability

```

---

## 8. Experimental Design

### 8.1 Baselines

- Rule-based controller (ASHRAE standard)  
- Static RL policy (no retraining)  
- Centralized cloud-based RL control  

### 8.2 Proposed Systems

- Edge-only RL  
- Edge–Cloud Distributed RL (proposed)  

---

### 8.3 Evaluation Metrics

#### Control Performance
- Energy consumption  
- Thermal comfort violations  
- Cumulative reward  

#### System Performance
- Inference latency (edge vs cloud)  
- Training time  
- Update frequency  

#### Adaptation Performance
- Transfer efficiency  
- Robustness across different climates  
- Performance under distribution shift  

---

## 9. Expected Contributions

1. A novel **Edge–Cloud distributed RL architecture** for HVAC control  
2. A scalable training pipeline leveraging **HPC infrastructure**  
3. A framework for **continuous adaptation** using real-world operational data  
4. Empirical evaluation on large-scale simulation environments (HOT)  

---

## 10. Significance

This research demonstrates that HVAC control should not be treated as a standalone optimization problem, but as a **distributed adaptive system** where:

- the **edge ensures responsiveness**, and  
- the **cloud/HPC enables scalable learning and adaptation**  

This paradigm can be extended to other cyber-physical systems requiring both real-time control and large-scale learning.

---

## 11. Future Work

- Integration of similarity-based transfer learning  
- Multi-building cooperative control  
- Uncertainty-aware RL for safer deployment  
- Real-world pilot deployment  

---

## 12. Conclusion

This proposal introduces a distributed reinforcement learning framework for real-time HVAC control that combines the strengths of edge computing and HPC-based training. By bridging the gap between simulation and deployment, the proposed system aims to achieve both responsiveness and scalability in intelligent building control systems.
