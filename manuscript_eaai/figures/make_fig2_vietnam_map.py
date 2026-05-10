"""
Fig 2 — Vietnam Multi-City Evaluation Map
Uses the Vecteezy EPS silhouette as base, overlays ASHRAE climate zones
and city markers in matplotlib.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

# ── Load silhouette ───────────────────────────────────────────────────────────
BASE = ("/Users/anhoaithai/Documents/AHT/1. PROJECTS/RL HVAC HPC/"
        "manuscript_eaai/figures/vn_map_base.png")
img  = Image.open(BASE).convert("RGBA")
arr  = np.array(img)   # (8319, 8319, 4)

# Vietnam territory mask: dark pixels with non-zero alpha
mask = (arr[:,:,0] < 80) & (arr[:,:,3] > 100)  # True = land

# ── Pixel ↔ latitude mapping ──────────────────────────────────────────────────
# Silhouette spans rows 1063–7272.
# Top = northernmost Vietnam ~23.4°N; bottom = Ca Mau ~8.4°N
R_TOP, R_BOT = 1063, 7272
LAT_TOP, LAT_BOT = 23.4, 8.4   # °N

def lat2row(lat):
    return R_TOP + (LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * (R_BOT - R_TOP)

# ASHRAE zone boundaries (approximate lat thresholds)
# 0A: < 13°N  |  1A: 13–18°N  |  2A: > 18°N
ROW_0A1A = int(lat2row(13.0))   # ~6400
ROW_1A2A = int(lat2row(18.0))   # ~3500

# Silhouette col bounds
C_MIN, C_MAX = 2675, 5660
H, W = arr.shape[:2]

# ── Build RGBA overlay image ───────────────────────────────────────────────────
# Start transparent; paint zone colour onto land pixels
overlay = np.zeros((H, W, 4), dtype=np.uint8)

ALPHA = 200   # zone fill opacity (0–255)
# Zone 2A (north, Warm Humid) — teal/green
COL_2A = np.array([56, 142, 60,  ALPHA], dtype=np.uint8)   # #388E3C
# Zone 1A (centre, Hot Humid) — amber
COL_1A = np.array([251, 192, 45, ALPHA], dtype=np.uint8)   # #FBC02D
# Zone 0A (south, Extremely Hot) — coral/red
COL_0A = np.array([239, 108, 91, ALPHA], dtype=np.uint8)   # softer coral-red

for r in range(H):
    row_mask = mask[r]
    if not row_mask.any():
        continue
    if r <= ROW_1A2A:
        overlay[r, row_mask] = COL_2A
    elif r <= ROW_0A1A:
        overlay[r, row_mask] = COL_1A
    else:
        overlay[r, row_mask] = COL_0A

# White background for display
bg = np.ones((H, W, 4), dtype=np.uint8) * 255
bg[:,:,3] = 255

# Compose: bg → overlay → black outline
def alpha_composite(dst, src):
    src_a = src[:,:,3:4].astype(np.float32) / 255.0
    dst_a = dst[:,:,3:4].astype(np.float32) / 255.0
    out_a = src_a + dst_a * (1 - src_a)
    out_rgb = (src[:,:,:3].astype(np.float32) * src_a +
               dst[:,:,:3].astype(np.float32) * dst_a * (1 - src_a))
    out = np.zeros_like(dst)
    nz = out_a[:,:,0] > 0
    out[:,:,:3][nz] = (out_rgb[nz] / out_a[:nz.sum() if False else ..., np.newaxis
                                            if False else 0:1][nz]
                       ).clip(0,255).astype(np.uint8)
    out[:,:,3] = (out_a[:,:,0] * 255).astype(np.uint8)
    return out

# Simple composite (overlay on white)
composed = bg.copy().astype(np.float32)
oa = overlay[:,:,3:4].astype(np.float32) / 255.0
composed[:,:,:3] = (overlay[:,:,:3].astype(np.float32) * oa +
                    bg[:,:,:3].astype(np.float32) * (1 - oa))

# Add thin outline: find edge pixels (land pixel with at least one non-land neighbor)
from scipy.ndimage import binary_erosion
interior = binary_erosion(mask, iterations=6)
edge = mask & ~interior
composed[edge, :3] = [30, 30, 30]   # dark outline only on border
composed = composed.clip(0, 255).astype(np.uint8)

# Crop to silhouette extent + padding
PAD = 300
r0 = max(0, R_TOP - PAD)
r1 = min(H, R_BOT + PAD)
c0 = max(0, C_MIN - PAD)
c1 = min(W, C_MAX + PAD)
crop = composed[r0:r1, c0:c1]

# ── matplotlib figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 11))

# Shift the silhouette image left + up (positive values move map in those
# directions) without moving the city dots, so the dots align with their
# actual geographic positions on the silhouette.
MAP_SHIFT_LEFT = 280   # px shifted to the left
MAP_SHIFT_UP   = 100   # px shifted upward
H_crop, W_crop = crop.shape[:2]
ax.imshow(
    crop, origin="upper",
    extent=(-MAP_SHIFT_LEFT, W_crop - MAP_SHIFT_LEFT,
            H_crop - MAP_SHIFT_UP, -MAP_SHIFT_UP),
)
# Keep the original city-dot coordinate range visible
ax.set_xlim(0, W_crop)
ax.set_ylim(H_crop, 0)
ax.axis("off")

# Helpers — anything painted on the silhouette (zone fills already shift with
# the image; zone boundary lines and zone labels need the same explicit shift
# so the 3-region partition follows the silhouette).
def shift_xy(x, y):
    return x - MAP_SHIFT_LEFT, y - MAP_SHIFT_UP

def shift_y(y):
    return y - MAP_SHIFT_UP

# Coordinate helpers (pixel in crop image)
def rc(lat, lon_frac=0.5):
    """lat in °N → (x, y) in crop pixel coords."""
    r = lat2row(lat) - r0
    # lon_frac: 0=left edge, 1=right edge of silhouette at that lat
    c = (C_MIN + (C_MAX - C_MIN) * lon_frac) - c0
    return c, r

# ── Zone boundary lines ───────────────────────────────────────────────────────
# Boundaries describe latitude bands of the silhouette image; they must
# travel with the silhouette under the global map shift.
for row_bound, lat_val in [(ROW_1A2A, 18.0), (ROW_0A1A, 13.0)]:
    y = shift_y(row_bound - r0)
    ax.axhline(y=y, color="white", lw=1.5, linestyle="--", alpha=0.7, zorder=3)

# ── Zone labels ───────────────────────────────────────────────────────────────
zone_labels = [
    (21.5, 0.08, "Zone 2A", "Warm Humid", "#1B5E20"),
    (15.5, 0.88, "Zone 1A", "Hot Humid",  "#7B6000"),
    (10.2, 0.08, "Zone 0A", "Extremely\nHot Humid", "#B71C1C"),
]
for lat, lf, zname, zdesc, zcolor in zone_labels:
    cx, cy = rc(lat, lf)
    cx, cy = shift_xy(cx, cy)
    ax.text(cx, cy, f"{zname}\n{zdesc}", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=zcolor, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=zcolor, alpha=0.82, linewidth=1.2))

# ── City coordinates (approximate pixel positions) ────────────────────────────
# Fine-tune: row = lat2row(lat), col fraction calibrated to actual city position
cities = {
    "Hà Nội":     (21.03, 0.60, "square", "#1565C0", "source"),  # actually target
    "Đà Nẵng":   (16.06, 0.92, "triangle", "#E65100", "target"),
    "HCMC\n(Hồ Chí Minh)": (10.82, 0.75, "square", "#C62828", "source"),
    "Cần Thơ":   (10.03, 0.52, "square",   "#C62828", "source"),
}

# Actually per the paper: HCMC + Can Tho = source; Da Nang + Hanoi = target
cities = {
    "Hà Nội":              (21.03, 0.60, "^", "#1565C0", "target"),
    "Đà Nẵng":            (16.06, 0.90, "^", "#E65100", "target"),
    "Hồ Chí Minh (HCMC)": (10.82, 0.70, "s", "#B71C1C", "source"),
    "Cần Thơ":             (10.03, 0.45, "s", "#B71C1C", "source"),
}

LOCATION_ARROW_SHIFT_X = -700

def annotation_xy(x, y):
    return x + LOCATION_ARROW_SHIFT_X, y

MSIZE = 120
for name, (lat, lf, marker, color, role) in cities.items():
    cx, cy = rc(lat, lf)
    cx, cy = annotation_xy(cx, cy)
    ax.scatter(cx, cy, s=MSIZE, marker=marker, color=color,
               edgecolors="white", linewidths=1.8, zorder=7)
    # label offset
    dx = 60 if lf < 0.7 else -60
    ha = "left" if lf < 0.7 else "right"
    ax.text(cx + dx, cy, name, ha=ha, va="center",
            fontsize=8, fontweight="bold", color=color, zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.85, linewidth=1))

# ── Transfer arrows ───────────────────────────────────────────────────────────
# HCMC → Da Nang  (0A → 1A)
hcmc_c, hcmc_r   = rc(10.82, 0.70)
danang_c, danang_r = rc(16.06, 0.90)
hanoi_c, hanoi_r  = rc(21.03, 0.60)

ax.annotate("",
    xy=annotation_xy(danang_c, danang_r), xytext=annotation_xy(hcmc_c, hcmc_r),
    arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.0,
                    mutation_scale=16, connectionstyle="arc3,rad=-0.25"),
    zorder=5)
# label at 1/3 of the way up, shifted right outside silhouette
# label at right edge of crop image
img_w = c1 - c0
ax.text(img_w - 30, (hcmc_r * 0.65 + danang_r * 0.35),
        "Transfer\n0A → 1A", ha="right", va="center",
        fontsize=7.2, color="#E65100", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#E65100", alpha=0.88, linewidth=1.2))

ax.annotate("",
    xy=annotation_xy(hanoi_c, hanoi_r), xytext=annotation_xy(hcmc_c, hcmc_r),
    arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2.0,
                    mutation_scale=16, connectionstyle="arc3,rad=-0.22"),
    zorder=5)
ax.text(img_w - 30, (hcmc_r * 0.45 + hanoi_r * 0.55),
        "Transfer\n0A → 2A", ha="right", va="center",
        fontsize=7.2, color="#1565C0", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#1565C0", alpha=0.88, linewidth=1.2))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor="#E53935", alpha=0.8, label="Zone 0A — Extremely Hot Humid"),
    mpatches.Patch(facecolor="#FBC02D", alpha=0.8, label="Zone 1A — Hot Humid"),
    mpatches.Patch(facecolor="#388E3C", alpha=0.8, label="Zone 2A — Warm Humid"),
    plt.scatter([], [], s=90, marker="s", color="#B71C1C",
                edgecolors="white", lw=1.5, label="Source city (training)"),
    plt.scatter([], [], s=90, marker="^", color="#1565C0",
                edgecolors="white", lw=1.5, label="Target city (transfer eval)"),
]
ax.legend(handles=legend_handles, loc="lower left",
          fontsize=7.5, framealpha=0.92, edgecolor="#BDBDBD",
          bbox_to_anchor=(0.01, 0.01),
          handlelength=1.4, handleheight=1.2)

ax.set_title("Vietnam Multi-City Evaluation\n4 Cities · 3 ASHRAE Climate Zones",
             fontsize=11, fontweight="bold", color="#1A237E", pad=10)

plt.tight_layout(pad=0.5)

OUT = "/Users/anhoaithai/Documents/AHT/1. PROJECTS/RL HVAC HPC/manuscript_eaai/figures/fig2_vietnam_map"
plt.savefig(OUT + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=180, bbox_inches="tight")
print("Saved fig2_vietnam_map  v2")

from PIL import Image as _I
_img = _I.open(OUT + ".png")
print(f"Size: {_img.size}")
