# Hướng dẫn dựng pipeline figures cho paper Elsevier (EAAI)

> Bối cảnh: manuscript `manuscript_eaai/main_revised.tex`. Fig5–8 (số liệu) dùng matplotlib OK; fig1, fig3, fig4 (pipeline/architecture) cần dựng lại.

---

## 1. Chẩn đoán hiện trạng

### Fig 1 — Architecture (`make_fig1_architecture.py`)
- Dùng quá nhiều màu (cam/đỏ/xanh dương/xanh lá đậm/đỏ đậm) + 3 nền màu khác → "PowerPoint vibe".
- Mỗi box có 2 tầng text khác cỡ → hierarchy rối.
- Mũi tên "Push Policy Update" cong cắt qua panel Network → red flag thiết kế.
- Legend ở dưới = band-aid cho việc dùng quá nhiều màu.
- Title in đậm trong hình → trùng với `\caption{}`.

### Fig 3 — Policy update
- 2 mũi tên dashed cong từ 2 lá ngược lên đỉnh, cắt qua diamond → rối.
- Cả nhánh Yes và No đều trỏ lên cùng node mà không có node "Next interval" trung gian → logic ngầm.
- Box có pastel fill → "consumer slide" thay vì academic.

### Vấn đề chung
- `FancyBboxPatch` + manual coordinates trong matplotlib khó duy trì.
- Stroke width không nhất quán.
- Mũi tên không gắn đúng cạnh box.

---

## 2. Chọn công cụ

| Công cụ | Khi nào dùng |
|---|---|
| **TikZ/PGF** | Lựa chọn số 1 cho fig1, fig3 |
| **draw.io** (diagrams.net) | Khi cần tốc độ, vd fig4 transfer scenarios |
| **Inkscape** | Tinh chỉnh SVG xuất ra |
| ~~Excalidraw~~ | KHÔNG dùng — hand-drawn vibe không hợp paper formal |
| **matplotlib** | Chỉ cho biểu đồ số liệu (fig5–8) |

---

## 3. Nguyên tắc thiết kế

1. Tối đa 3 màu (kể cả đen). Lý tưởng: đen + 1 accent.
2. 1 font family, 2 cỡ chữ. Dùng `\sffamily` trong figure.
3. Stroke width nhất quán: box `0.5pt`, arrow `0.7pt`.
4. Không gradient / shadow / 3D.
5. Không title trong hình — title đi vào `\caption{}`.
6. Không mũi tên cắt nhau — nếu cắt thì redraw layout.
7. Box căn theo grid (TikZ `node distance`).
8. Output PDF (vector), không PNG.
9. Whitespace là bạn — đừng nhồi metadata.
10. B&W test: print grayscale vẫn đọc được.

---

## 4. Template TikZ

### 4.1 Fig 1 — Architecture

File `figures/tikz/fig1.tex`:

```latex
\documentclass[border=4pt,tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,fit,calc,backgrounds}
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >={Stealth[length=2.2mm,width=1.6mm]},
    box/.style    ={draw,rounded corners=2pt,minimum width=34mm,
                    minimum height=10mm,align=center,line width=0.5pt,fill=white},
    decision/.style={draw,diamond,aspect=2,align=center,inner sep=1pt,
                    line width=0.5pt,fill=white},
    flow/.style   ={->,line width=0.7pt,draw=black!75},
    deploy/.style ={->,line width=0.7pt,draw=black!75,dashed},
    panel/.style  ={draw=black!30,line width=0.4pt,rounded corners=2pt,inner sep=4mm},
    lbl/.style    ={font=\sffamily\scriptsize\itshape,inner sep=1pt}]

  % EDGE column
  \node[box] (sens) {Building sensors};
  \node[box,below=6mm of sens] (inf)  {Policy inference\\\scriptsize PPO actor};
  \node[box,below=6mm of inf]  (safe) {Safety guard};
  \node[box,below=6mm of safe] (act)  {HVAC actuator};

  % CLOUD column
  \node[box,right=44mm of sens] (agg) {Trajectory aggregator};
  \node[box,below=6mm of agg]   (tr)  {Multi-context PPO trainer};
  \node[box,below=6mm of tr]    (ev)  {Policy evaluator};
  \node[decision,below=6mm of ev] (dec) {$D < D_0$\\$\Delta V \le \epsilon$?};

  \draw[flow] (sens)--(inf); \draw[flow] (inf)--(safe); \draw[flow] (safe)--(act);
  \draw[flow] (inf.east) -- node[lbl,above]{trajectory log} (agg.west);
  \draw[flow] (agg)--(tr); \draw[flow] (tr)--(ev); \draw[flow] (ev)--(dec);
  \draw[deploy] (dec.west) -- ++(-10mm,0)
                |- node[lbl,above,pos=0.25]{deploy if accepted} (inf.east);

  \begin{scope}[on background layer]
    \node[panel,fit=(sens)(act),
          label={[font=\sffamily\bfseries\small]above:Edge layer}] {};
    \node[panel,fit=(agg)(dec),
          label={[font=\sffamily\bfseries\small]above:Cloud / HPC layer}] {};
  \end{scope}
\end{tikzpicture}
\end{document}
```

### 4.2 Fig 3 — Policy Update

File `figures/tikz/fig3.tex`:

```latex
\documentclass[border=4pt,tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,calc}
\begin{document}
\begin{tikzpicture}[
    font=\sffamily\small,
    >={Stealth[length=2.2mm,width=1.6mm]},
    node distance=7mm and 14mm,
    step/.style    ={draw,rounded corners=2pt,minimum width=46mm,
                     minimum height=9mm,align=center,line width=0.5pt},
    decision/.style={draw,diamond,aspect=2.2,align=center,inner sep=0pt,line width=0.5pt},
    flow/.style    ={->,line width=0.7pt}]
  \node[step] (run) {Edge policy running\\\scriptsize interval $\approx$ 24\,h};
  \node[step,below=of run] (up) {Upload trajectories};
  \node[step,below=of up]  (tr) {Multi-context PPO training\\\scriptsize HCMC, Can Tho};
  \node[step,below=of tr]  (ev) {Evaluate on held-out city\\\scriptsize Da Nang / Hanoi};
  \node[decision,below=of ev] (d) {$D_{\text{cand}} < D_{\text{cur}}$\\$\Delta V \le \epsilon$?};
  \node[step,below left=12mm and 4mm of d]  (yes) {Deploy candidate};
  \node[step,below right=12mm and 4mm of d] (no)  {Retain current};

  \draw[flow] (run)--(up); \draw[flow] (up)--(tr);
  \draw[flow] (tr)--(ev);  \draw[flow] (ev)--(d);
  \draw[flow] (d) -| node[pos=0.25,above,font=\sffamily\scriptsize]{yes} (yes);
  \draw[flow] (d) -| node[pos=0.25,above,font=\sffamily\scriptsize]{no}  (no);

  \draw[flow,dashed] (yes.south) -- ++(0,-6mm) -|
        node[pos=0.25,below,font=\sffamily\scriptsize\itshape]{next interval}
        ($(run.west)+(-8mm,0)$) -- (run.west);
  \draw[flow,dashed] (no.south) -- ++(0,-6mm) -|
        ($(run.east)+(8mm,0)$) -- (run.east);
\end{tikzpicture}
\end{document}
```

### 4.3 Fig 4 — Transfer scenarios (draw.io)

1. [app.diagrams.net](https://app.diagrams.net) → New → Blank.
2. 4 ellipse: `HCMC (0A)`, `Can Tho (0A)`, `Da Nang (1A)`, `Hanoi (2A)`.
3. Đặt theo trục Bắc–Nam (Hà Nội trên cùng).
4. 5 mũi tên với label `V1`–`V5` nghiêng dọc theo arrow.
5. Màu: solid black; nếu cần phân biệt loại shift, dùng dashed cho cross-zone.
6. Export → PDF, Crop, Transparent background → lưu vào `figures/fig4_transfer_scenarios.pdf`.

---

## 5. Workflow

```bash
mkdir -p figures/tikz
cd figures/tikz
# soạn fig1.tex, fig3.tex theo template
pdflatex -interaction=nonstopmode fig1.tex
pdfcrop fig1.pdf ../fig1_architecture.pdf
pdflatex -interaction=nonstopmode fig3.tex
pdfcrop fig3.pdf ../fig3_policy_update.pdf
```

`main_revised.tex` đã trỏ tới `figures/fig1_architecture.pdf` và `figures/fig3_policy_update.pdf` → không cần đổi gì trong manuscript.

---

## 5b. Lưu ý kỹ thuật TikZ (lesson learned)

- **Đừng đặt tên style là `step`**: `step` là reserved key của TikZ (dùng cho grid drawing). Nếu khai báo `step/.style={...}` thì style không được đăng ký, dẫn đến `\node[step]` mất viền box và xếp text dồn lên 1 dòng. Dùng tên khác như `task`, `proc`, `stage`.
- **Trong node text, dùng `\\` để xuống dòng** kết hợp với `align=center` *hoặc* `text width=...`. Nếu cần dòng phụ cỡ nhỏ:
  ```latex
  \node[task] {Tiêu đề chính\\[1pt]{\scriptsize chú thích phụ}};
  ```
  Đặt `\scriptsize` trong `{...}` để giới hạn scope.
- **Mũi tên L-shape không cắt nhau**: dùng `(d.west) -| (yes.north)` thay vì `(d) -- (yes)`.
- **`on background layer`** để vẽ panel/khung sau khi nodes đã định vị → khung tự ôm trọn theo `fit=(node1)(node2)`.
- **Đường return không cắt diagram**: cho dashed arrows đi vòng ra ngoài 2 cạnh trái/phải.

---

## 6. Checklist trước submit

- [ ] PDF zoom 400% còn sắc nét (vector test)
- [ ] Font trùng family với body text
- [ ] ≤ 3 màu trên toàn bộ paper
- [ ] Không title bên trong hình
- [ ] Không có mũi tên cắt nhau
- [ ] Caption đầy đủ, đứng độc lập đọc được
- [ ] Print grayscale vẫn đọc được
