# Kế hoạch viết lại bài — EAAI Vietnam HVAC RL

## Trạng thái hiện tại (2026-05-08)

- Thí nghiệm đang chạy overnight (EnergyPlus thật, 4 thành phố VN)
- Manuscript đã có structure đầy đủ, tất cả Results section còn placeholder
- 4 figures đã tạo (PNG + PDF)
- references.bib: 17 entries

---

## Figures đã tạo

| File | Nội dung | Vị trí trong bài |
|------|----------|------------------|
| `fig1_architecture.pdf` | Edge-Cloud RL Architecture | Section 4 (Framework) |
| `fig2_vietnam_map.pdf`  | Vietnam 4-city map + climate zones | Section 5 (Exp Setup) |
| `fig3_policy_update.pdf`| Policy Update Protocol flowchart | Section 4.3 |
| `fig4_transfer_scenarios.pdf` | Transfer scenarios V1-V5 | Section 5.2 |

---

## References cần thêm vào main.tex

Thêm `\cite{}` cho các references mới vào đúng chỗ:

| Cite key | Vị trí |
|----------|--------|
| `\cite{vazquez2019rl}` | Related Work — RL for Building |
| `\cite{wang2020review}` | Related Work — RL for Building |
| `\cite{yu2021review}` | Related Work — Multi-agent HVAC |
| `\cite{fang2023transfer}` | Related Work — Transfer Learning |
| `\cite{kadamala2024transfer}` | Related Work — Transfer Learning |
| `\cite{jimenez2021sinergym}` | Exp Setup — simulation environment |
| `\cite{stable_baselines3}` | Exp Setup — PPO implementation |
| `\cite{shi2016edge}` | Related Work — Edge Computing |
| `\cite{dena2021renewable}` | Intro — Vietnam cooling |
| `\cite{deru2011doe}` | Exp Setup — DOE reference buildings |
| `\cite{ashrae55}` | Problem Form — comfort bounds |
| `\cite{ashrae901}` | Exp Setup — building standards |

---

## Checklist viết bài — theo thứ tự ưu tiên

### Bước 1: Sau khi có kết quả thí nghiệm (sáng mai)

**Section 6.1 — Control Performance Table**
- [ ] Điền energy (kWh/m²) cho 7 methods × 4 cities
- [ ] Điền comfort violation rate (%)
- [ ] Điền cumulative reward
- [ ] Điền action instability
- [ ] Thêm confidence intervals (± 95% CI, 3 seeds)
- Script: `python -m src.summarize_results → results/summary/vietnam_final/control_summary.csv`

**Section 6.2 — Cross-City Transfer Table**
- [ ] TE(c) cho HCMC → Can Tho, HCMC → Da Nang, HCMC → Hanoi
- [ ] Bảng TE theo method (single-ctx PPO vs multi-ctx PPO vs edge-cloud)
- Script: `results/summary/vietnam_final/transfer_metrics.csv`

**Section 6.3 — Latency Table**
- [ ] p50/p95 edge inference (ms)
- [ ] p50/p95 cloud inference (ms, simulated 80ms delay)
- [ ] Policy artifact size (MB)
- Script: `results/summary/vietnam_final/latency_summary.csv`

**Section 6.4 — Adaptation (AMY weather shift)**
- [ ] Energy delta after 1 update cycle
- [ ] Adaptation gain per city

### Bước 2: Cập nhật text sections

**Abstract** — điền 3-4 số thực tế:
```
"The proposed framework reduces energy use intensity by X% in HCMC 
and achieves transfer efficiency of Y% when deployed in Hanoi 
with p95 edge inference latency below Z ms."
```

**Introduction** — thêm:
- [ ] Câu số liệu cụ thể về Vietnam (64.68 MT CO₂eq từ cooling năm 2022)
- [ ] Cite `\cite{dena2021renewable}` cho cooling in HCMC + Hanoi

**Section 5.1 (Exp Setup)** — thêm:
- [ ] Cite Sinergym `\cite{jimenez2021sinergym}` — simulation platform
- [ ] Cite SB3 `\cite{stable_baselines3}` — PPO implementation
- [ ] Cite DOE reference buildings `\cite{deru2011doe}`

**Section 7 (Discussion)** — viết sau khi có kết quả:
- [ ] Tại sao HCMC → Hanoi transfer khó hơn HCMC → Can Tho?
- [ ] Latency edge < 5ms vs cloud 80ms — có đáng giá không?
- [ ] Limitations: building model calibrated for zone 0A, dùng cho zone 2A

### Bước 3: Thêm figures vào LaTeX

Thêm vào main.tex sau khi figures hoàn chỉnh:

```latex
% Sau Section 4 intro
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{figures/fig1_architecture.pdf}
  \caption{Edge-cloud adaptive RL framework for transferable HVAC control.
           The edge layer executes low-latency policy inference and safety
           validation; the cloud layer performs multi-city retraining and
           versioned policy deployment.}
  \label{fig:architecture}
\end{figure}

% Trong Section 5 (Exp Setup)
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.65\textwidth]{figures/fig2_vietnam_map.pdf}
  \caption{Four Vietnamese cities used in the evaluation, spanning three
           ASHRAE climate zones. Square markers = source cities (training);
           triangle markers = target cities (transfer evaluation).}
  \label{fig:map}
\end{figure}

% Trong Section 4.3
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.7\textwidth]{figures/fig3_policy_update.pdf}
  \caption{Policy update protocol. A candidate policy trained on source
           cities is deployed to the edge only when it improves the
           deployment score without exceeding the comfort regression threshold.}
  \label{fig:protocol}
\end{figure}
```

### Bước 4: Ablation studies

Sau khi có kết quả, viết Section 6.5:
- [ ] Safety guard ON vs OFF (comfort violation rate)
- [ ] Single-city vs multi-city training (transfer efficiency)
- [ ] No update vs periodic update (adaptation gain)
- [ ] Edge eager vs TorchScript (latency reduction)

---

## Thứ tự viết (timeline)

```
Sáng mai (sau overnight run):
  1. Đọc control_summary.csv, transfer_metrics.csv, latency_summary.csv
  2. Điền Tables 3, 4, 5 trong main.tex
  3. Cập nhật abstract với số thực tế
  4. Viết Discussion dựa trên kết quả

Ngày 2:
  5. Thêm figures vào LaTeX
  6. Thêm cite keys còn thiếu
  7. Viết Conclusion đầy đủ
  8. Chạy pdflatex để kiểm tra compile

Ngày 3:
  9. Ablation analysis (nếu cần thêm runs)
  10. Proofreading
  11. Check submission requirements (EAAI)
```

---

## Thông tin submission (EAAI)

- Venue: Engineering Applications of Artificial Intelligence (Elsevier)
- Format: elsarticle, double-blind
- Figures: PDF vector preferred
- References: numbered, elsarticle-num style
- Data availability: code + data + policy artifacts → anonymized repo
