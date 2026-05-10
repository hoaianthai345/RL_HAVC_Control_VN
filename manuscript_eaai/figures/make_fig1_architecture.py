"""
Fig 1: Edge-Cloud Adaptive RL Architecture  (v2 — cleaned layout)
3-panel layout: EDGE LAYER | NETWORK | CLOUD/HPC LAYER
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_EDGE_BG   = "#E3F2FD"
C_NET_BG    = "#F5F5F5"
C_CLOUD_BG  = "#E8F5E9"
C_EDGE_BOX  = "#1976D2"
C_CLOUD_BOX = "#388E3C"
C_SENSOR    = "#E65100"
C_SAFETY    = "#C62828"
C_NET_BOX   = "#757575"
C_WHITE     = "white"
C_ARROW     = "#37474F"
C_UP        = "#1565C0"
C_DOWN      = "#2E7D32"

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")

# ── Panel backgrounds ─────────────────────────────────────────────────────────
def panel(x0, y0, w, h, color, label):
    r = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor="#BDBDBD",
                       linewidth=1.8, zorder=0)
    ax.add_patch(r)
    ax.text(x0 + w/2, y0 + h - 0.22, label,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#212121", zorder=1)

panel(0.2,  0.3, 5.0, 7.4, C_EDGE_BG,  "EDGE LAYER")
panel(5.4,  0.3, 4.2, 7.4, C_NET_BG,   "NETWORK")
panel(9.8,  0.3, 4.9, 7.4, C_CLOUD_BG, "CLOUD / HPC LAYER")

# ── Helper: rounded box ───────────────────────────────────────────────────────
def box(x, y, w, h, line1, line2=None, color=C_EDGE_BOX, fs=8.5):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor="white",
                       linewidth=1.4, zorder=2)
    ax.add_patch(r)
    dy = 0.15 if line2 else 0
    ax.text(x, y + dy, line1, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=C_WHITE, zorder=3)
    if line2:
        ax.text(x, y - 0.2, line2, ha="center", va="center",
                fontsize=6.8, color=C_WHITE, alpha=0.88, zorder=3)

# ── Helper: straight arrow ────────────────────────────────────────────────────
def arr(x0, y0, x1, y1, color=C_ARROW, lw=1.6, label="", lpos="right"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14,
                                connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx = 0.12 if lpos == "right" else -0.12
        ha = "left" if lpos == "right" else "right"
        ax.text(mx + dx, my, label, ha=ha, va="center",
                fontsize=6.8, color=color, style="italic", zorder=4)

# ── EDGE LAYER  (centre x = 2.7) ──────────────────────────────────────────────
EX = 2.7
ew, eh = 3.8, 0.6

box(EX, 6.6,  ew, eh, "Building Sensors",
    "Temperature · Humidity · Occupancy", C_SENSOR)
box(EX, 5.3,  ew, eh, "Policy Inference",
    "PPO · TorchScript (eager → scripted)", C_EDGE_BOX)
box(EX, 4.0,  ew, eh, "Safety Guard",
    "Comfort constraint enforcement", C_SAFETY)
box(EX, 2.7,  ew, eh, "HVAC Actuator",
    "Cooling / Heating setpoint command", C_EDGE_BOX)

# vertical arrows inside edge
arr(EX, 6.3,  EX, 5.6)
arr(EX, 5.0,  EX, 4.3)
arr(EX, 3.7,  EX, 3.0)

# Trajectory log side label (inside edge panel, right side)
ax.annotate("", xy=(4.45, 5.3), xytext=(EX + ew/2, 5.3),
            arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=1.1,
                            mutation_scale=11))
ax.text(4.55, 5.3, "Trajectory\nlog →", ha="left", va="center",
        fontsize=6.5, color="#37474F", style="italic")

# ── NETWORK  (centre x = 7.5) ─────────────────────────────────────────────────
NX = 7.5
nw, nh = 3.4, 0.72

# Upload box
box(NX, 5.3, nw, nh, "Upload Trajectories", "p50 ≈ 80 ms  |  compressed batch",
    C_NET_BOX, fs=8)
# Push policy box
box(NX, 3.7, nw, nh, "Push Policy Update", "p50 < 5 ms  |  versioned artifact",
    C_NET_BOX, fs=8)

# Edge → Upload (horizontal)
arr(EX + ew/2, 5.3, NX - nw/2, 5.3, color=C_UP, lw=1.8)
# Push Policy → Edge (horizontal, below)
arr(NX - nw/2, 3.7, EX + ew/2, 3.7, color=C_DOWN, lw=1.8)

# Upload → Cloud
arr(NX + nw/2, 5.3, 9.8 + 0.05, 5.3, color=C_UP, lw=1.8)
# Cloud → Push Policy
arr(9.8 + 0.05, 3.7, NX + nw/2, 3.7, color=C_DOWN, lw=1.8)

# ── CLOUD / HPC  (centre x = 12.25) ──────────────────────────────────────────
CX = 12.25
cw, ch = 4.2, 0.6

box(CX, 6.6, cw, ch, "Trajectory Aggregator",
    "cities: HCMC · Cần Thơ · Đà Nẵng · Hà Nội", C_CLOUD_BOX)
box(CX, 5.3, cw, ch, "Multi-City PPO Trainer",
    "source contexts: HCMC · Cần Thơ", C_CLOUD_BOX)
box(CX, 4.0, cw, ch, "Policy Evaluator",
    "held-out city rollout · score D", C_CLOUD_BOX)

arr(CX, 6.3, CX, 5.6)
arr(CX, 5.0, CX, 4.3)
arr(CX, 3.7, CX, 3.1)

# Decision diamond
diam_x, diam_y, dw, dh = CX, 2.55, 2.1, 0.8
diamond = plt.Polygon(
    [(diam_x,        diam_y + dh/2),
     (diam_x + dw/2, diam_y),
     (diam_x,        diam_y - dh/2),
     (diam_x - dw/2, diam_y)],
    closed=True, facecolor="#F57F17", edgecolor="white", lw=1.3, zorder=2)
ax.add_patch(diamond)
ax.text(diam_x, diam_y + 0.1, "Deploy?",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="white", zorder=3)
ax.text(diam_x, diam_y - 0.18, "D < θ   ΔV ≤ ε",
        ha="center", va="center", fontsize=6.5, color="white", zorder=3)

# Accept / Reject
box(CX - 1.6, 1.45, 1.8, 0.52, "Accept & Deploy", color="#1B5E20", fs=8)
box(CX + 1.6, 1.45, 1.8, 0.52, "Reject / Retain", color="#B71C1C", fs=8)

ax.annotate("", xy=(CX - 1.6, 1.71), xytext=(CX - dw/2 + 0.1, 2.55),
            arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.5,
                            mutation_scale=13,
                            connectionstyle="arc3,rad=0"))
ax.text(CX - dw/2 - 0.08, 2.1, "Yes", ha="right", va="center",
        fontsize=7.5, color="#2E7D32", fontweight="bold")

ax.annotate("", xy=(CX + 1.6, 1.71), xytext=(CX + dw/2 - 0.1, 2.55),
            arrowprops=dict(arrowstyle="-|>", color="#B71C1C", lw=1.5,
                            mutation_scale=13,
                            connectionstyle="arc3,rad=0"))
ax.text(CX + dw/2 + 0.08, 2.1, "No", ha="left", va="center",
        fontsize=7.5, color="#B71C1C", fontweight="bold")

# Accepted policy feedback: bottom of Accept → edge (curved path under panels)
ax.annotate("",
    xy=(EX + ew/2, 3.7),
    xytext=(CX - 1.6, 1.19),
    arrowprops=dict(
        arrowstyle="-|>", color=C_DOWN, lw=1.5, mutation_scale=13,
        connectionstyle="arc3,rad=-0.25"))
ax.text(7.5, 0.75, "Deployed Policy  (versioned rollout)",
        ha="center", va="center", fontsize=7, color=C_DOWN,
        style="italic", fontweight="bold")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 7.82,
        "Edge-Cloud Adaptive RL Framework for Transferable HVAC Control",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#0D47A1")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=C_SENSOR,    label="Sensor / Actuator"),
    mpatches.Patch(facecolor=C_EDGE_BOX,  label="Edge compute"),
    mpatches.Patch(facecolor=C_SAFETY,    label="Safety constraint"),
    mpatches.Patch(facecolor=C_CLOUD_BOX, label="Cloud compute"),
    mpatches.Patch(facecolor="#F57F17",   label="Deployment gate"),
    mpatches.Patch(facecolor="#1B5E20",   label="Deploy path"),
    mpatches.Patch(facecolor="#B71C1C",   label="Reject path"),
]
ax.legend(handles=handles, loc="lower center", ncol=7,
          fontsize=7, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.02),
          handlelength=1.1, handleheight=0.85)

plt.tight_layout(pad=0.4)

OUT = "/Users/anhoaithai/Documents/AHT/1. PROJECTS/RL HVAC HPC/manuscript_eaai/figures/fig1_architecture"
plt.savefig(OUT + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
print("Saved fig1_architecture  v2")

from PIL import Image
img = Image.open(OUT + ".png")
print(f"Size: {img.size}")
