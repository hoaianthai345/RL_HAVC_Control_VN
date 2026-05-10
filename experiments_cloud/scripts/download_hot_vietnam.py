#!/usr/bin/env python3
"""
Download Vietnam-only subset from HOT dataset and audit coverage.
Run from experiments_cloud/:
    python scripts/download_hot_vietnam.py
"""
from __future__ import annotations
import os, sys, textwrap
from pathlib import Path
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download, list_repo_files

TOKEN     = os.environ.get("HF_TOKEN", "")
REPO_ID   = "BuildingBench/HOT"
REPO_TYPE = "dataset"
OUT_DIR   = Path("data/vietnam_hot")
YAML_OUT  = Path("configs/contexts_vietnam.yaml")

# Building types phù hợp với bối cảnh Việt Nam (dùng sau khi biết tên cột)
# Tên archetype đúng theo HOT dataset
PRIORITY_TYPES = [
    "OfficeSmall", "OfficeMedium", "OfficeLarge",
    "HotelSmall",  "HotelLarge",
    "Hospital",    "OutPatientHealthCare",
    "SchoolPrimary", "SchoolSecondary",
    "ApartmentHighRise", "ApartmentMidRise",
    "RetailStandalone",  "Warehouse",
]

HUB_KW = dict(repo_id=REPO_ID, repo_type=REPO_TYPE, token=TOKEN)

def dl(filename: str, local_dir: Path | None = None) -> Path:
    kw = {**HUB_KW, "filename": filename}
    if local_dir:
        kw["local_dir"] = str(local_dir)
    return Path(hf_hub_download(**kw))

def sec(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# ── 1. list files ─────────────────────────────────────────────────────────────
sec("1. Files in repo")
all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE, token=TOKEN))
epw_files  = [f for f in all_files if f.endswith(".epw")]
csv_files  = [f for f in all_files if f.endswith(".csv")]
json_files = [f for f in all_files if f.endswith(".epJSON") or f.endswith(".json")]
print(f"Total: {len(all_files):,}  |  EPW: {len(epw_files)}  |  epJSON: {len(json_files)}  |  CSV: {len(csv_files)}")

# ── 2. Vietnam EPW — lọc chính xác bằng prefix VNM ──────────────────────────
sec("2. Vietnam EPW files (exact: VNM prefix)")
vn_epw_base = [f for f in epw_files if Path(f).name.startswith("VNM_")]
print(f"Found {len(vn_epw_base)} Vietnam EPW file(s):")
for f in sorted(vn_epw_base):
    print(f"  {f}")

# ── 3. Load metadata tables ────────────────────────────────────────────────────
sec("3. Loading metadata tables")
combos    = pd.read_csv(dl("tables/building_weather_combinations_all.csv"))
buildings = pd.read_csv(dl("tables/buildings.csv"))
weather   = pd.read_csv(dl("weather/tables/all_weather.csv"))

print(f"\nweather  shape: {weather.shape}")
print(f"weather  cols : {weather.columns.tolist()}")
print(f"\nbuildings shape: {buildings.shape}")
print(f"buildings cols : {buildings.columns.tolist()}")
print(f"\ncombos   shape : {combos.shape}")
print(f"combos   cols  : {combos.columns.tolist()}")

# ── 4. Tìm Vietnam trong weather table ────────────────────────────────────────
sec("4. Vietnam in weather table (country == VNM)")

# Tìm cột country
country_col = next((c for c in weather.columns if "country" in c.lower()), None)
print(f"Country column: {country_col}")
if country_col:
    print(f"Unique countries: {sorted(weather[country_col].unique())}")

vn_weather = weather[weather[country_col] == "VNM"] if country_col else pd.DataFrame()
print(f"\nVietnam weather rows ({len(vn_weather)}):")
print(vn_weather.to_string())

# Lấy weather IDs cho VNM
id_col = next((c for c in weather.columns if c.lower() in ["id","weather_id","index"]), None)
if id_col is None and weather.index.name:
    id_col = weather.index.name
vn_weather_ids = set(vn_weather[id_col].tolist()) if id_col and not vn_weather.empty else set()
print(f"\nVietnam weather IDs: {vn_weather_ids}  (column: {id_col})")

# ── 5. Tại sao chỉ có HCMC? ────────────────────────────────────────────────────
sec("5. Tại sao HOT chỉ có HCMC cho Việt Nam?")
cz_col  = next((c for c in weather.columns if "zone" in c.lower() and "code" in c.lower()), None)
if not cz_col:
    cz_col = next((c for c in weather.columns if "zone" in c.lower()), None)
print(f"Climate zone column: {cz_col}")

if cz_col and country_col:
    all_zones = weather[[country_col, cz_col]].drop_duplicates().sort_values(cz_col)
    print("\nTất cả 76 locations theo climate zone:")
    print(all_zones.to_string())

    if not vn_weather.empty and cz_col in vn_weather.columns:
        vn_zone = vn_weather[cz_col].iloc[0]
        peers = weather[weather[cz_col] == vn_zone][[country_col, cz_col]]
        print(f"\nCác thành phố cùng zone với HCMC ({vn_zone}):")
        print(peers.to_string())
        print(textwrap.dedent(f"""
        ┌─────────────────────────────────────────────────────────────────────┐
        │  HOT chọn 1 đại diện / ASHRAE climate zone.                        │
        │  HCMC (zone {str(vn_zone):<5}) = "Extremely Hot Humid" → đại diện VN duy nhất │
        │                                                                     │
        │  Hanoi   → zone 2A (Warm Humid)   → HOT chọn đại diện KHÁC        │
        │  Da Nang → zone 1A (Hot Humid)    → HOT chọn đại diện KHÁC        │
        │                                                                     │
        │  Để thêm Hanoi/Da Nang: cần tải EPW từ climate.onebuilding.org     │
        │  rồi chạy EnergyPlus với các file đó (không qua HOT).              │
        └─────────────────────────────────────────────────────────────────────┘"""))

# ── 6. Join combos → buildings → lọc VN ──────────────────────────────────────
sec("6. Lọc combinations cho HCMC (join building types)")

# Tìm join key giữa combos và weather
w_key_in_combos = next((c for c in combos.columns
    if any(k in c.lower() for k in ["weather_id","weather id","wid","location_id"])), None)
print(f"Weather join key in combos: {w_key_in_combos}")
print(f"Combos sample:\n{combos.head(3).to_string()}")

# Filter combos cho VN — dùng nhiều cách vì weather_id có thể là "base_18" (string)
vn_combos = pd.DataFrame()

# Cách 1: weather_country trực tiếp (chính xác nhất)
if "weather_country" in combos.columns:
    vn_combos = combos[combos["weather_country"] == "VNM"]
    print(f"Filtered by weather_country=='VNM': {len(vn_combos)} rows")

# Cách 2: city == "hochiminh"
if vn_combos.empty and "city" in combos.columns:
    vn_combos = combos[combos["city"].astype(str).str.lower().str.contains("hochiminh|ho.chi.minh", regex=True, na=False)]
    print(f"Filtered by city: {len(vn_combos)} rows")

# Cách 3: weather_id chứa "18" (e.g. "base_18")
if vn_combos.empty and w_key_in_combos:
    vn_combos = combos[combos[w_key_in_combos].astype(str).str.contains("_18$|^18$", regex=True, na=False)]
    print(f"Filtered by weather_id ~18: {len(vn_combos)} rows")

print(f"\nVietnam combinations: {len(vn_combos)}")

# ── 7. Join với buildings để lấy building type name ───────────────────────────
sec("7. Building types cho HCMC (sau khi join)")

# Tìm join key giữa combos và buildings
b_key_in_combos   = next((c for c in combos.columns
    if any(k in c.lower() for k in ["building_id","bid","building id"])), None)
b_key_in_buildings = next((c for c in buildings.columns
    if any(k in c.lower() for k in ["building_id","bid","id"])), None)
type_col = next((c for c in buildings.columns
    if any(k in c.lower() for k in ["type","geometry","archetype","building_type",
                                    "typology","name","category"])), None)

print(f"buildings key  : {b_key_in_buildings}")
print(f"combos bld key : {b_key_in_combos}")
print(f"type col       : {type_col}")
print(f"\nbuildings sample:\n{buildings.head(5).to_string()}")

# vn_combos đã có building_archetype và building_file_path — dùng trực tiếp
vn_merged = vn_combos.copy()

if not vn_merged.empty:
    arch_col  = "building_archetype"   # tên trong combos
    fpath_col = "building_file_path"   # path epJSON trong combos

    type_summary = (vn_merged.groupby(arch_col).size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False))
    print(f"\nBuilding types for Vietnam/HCMC ({len(vn_merged)} combos):")
    print(type_summary.to_string(index=False))

    for col in ["thermal_scenario","occupancy_schedule","weather_type"]:
        vals = sorted(vn_merged[col].dropna().unique()) if col in vn_merged.columns else "n/a"
        print(f"  {col}: {vals}")
else:
    arch_col  = "building_archetype"
    fpath_col = "building_file_path"
    print("Vietnam combinations empty")

# ── 8. Download VN EPW files ──────────────────────────────────────────────────
sec("8. Downloading Vietnam EPW files (VNM prefix only)")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "weather").mkdir(exist_ok=True)
(OUT_DIR / "buildings").mkdir(exist_ok=True)

downloaded_epw: list[str] = []
for epw in sorted(vn_epw_base):
    try:
        local = dl(epw, OUT_DIR / "weather")
        sz = local.stat().st_size / 1e6
        print(f"  OK  {local.name}  ({sz:.1f} MB)")
        downloaded_epw.append(epw)
    except Exception as e:
        print(f"  ERR {epw}: {e}")

# ── 9. Download building models ────────────────────────────────────────────────
sec("9. Downloading building models (priority types)")

model_path_col = next((c for c in buildings.columns
    if any(k in c.lower() for k in ["file","path","epjson","filepath","model"])), None)
print(f"Model path column in buildings: {model_path_col}")
print(f"All buildings columns: {buildings.columns.tolist()}")

downloaded_models: list[dict] = []

if not vn_merged.empty:
    # Lọc priority archetypes, 1 combination/archetype (base weather, standard occupancy, default thermal)
    base_combos = vn_merged[
        (vn_merged["weather_type"] == "base") &
        (vn_merged.get("building_variation", pd.Series(["base"]*len(vn_merged))) == "base")
    ] if "weather_type" in vn_merged.columns else vn_merged

    vn_priority = base_combos[base_combos[arch_col].isin(PRIORITY_TYPES)]
    targets = vn_priority.drop_duplicates(subset=[arch_col])
    print(f"Targets to download: {len(targets)} HCMC building models")

    for _, row in targets.iterrows():
        path = str(row.get(fpath_col, "")).strip()
        if not path or path == "nan":
            continue
        try:
            local = dl(path, OUT_DIR / "buildings")
            sz = local.stat().st_size / 1e3
            print(f"  OK  {row[arch_col]:<25}  {sz:.0f} KB  ← {Path(path).name}")
            downloaded_models.append({
                "type":      row[arch_col],
                "thermal":   "default",
                "repo_path": path,
                "local":     f"data/vietnam_hot/buildings/{Path(path).name}",
            })
        except Exception as e:
            print(f"  ERR {path}: {e}")

# ── 10. Generate contexts_vietnam.yaml ────────────────────────────────────────
sec("10. Generating contexts_vietnam.yaml")

EPW_TMY = next((e for e in downloaded_epw if "TMYx" in e or "TMY" in e), "")
EPW_AMY_YEARS = sorted([e for e in downloaded_epw if "AMY" in e])

contexts = []
buckets  = ["source", "source", "target", "target", "target", "target"]

for i, m in enumerate(downloaded_models[:6]):
    bname   = m["type"]
    thermal = m.get("thermal", "default")
    env_f   = 1.3 if thermal == "low_performance" else 1.0

    ctx = {
        "id": f"vn_hcmc_{bname.lower().replace(' ','_')}_{thermal}",
        "transfer_bucket": buckets[i],
        "building": bname,
        "climate_zone": "0A",
        "location": "HoChiMinh_Vietnam",
        "weather_type": "TMYx",
        "occupancy_schedule": "standard",
        "thermal_scenario": thermal,
        "shift_type": "none",
        "hot": {
            "epw_tmy":  EPW_TMY,
            "epw_amy_years": EPW_AMY_YEARS,
            "model_file": m["repo_path"],
            "epw_local":  f"data/vietnam_hot/weather/{Path(EPW_TMY).name}" if EPW_TMY else "",
            "model_local": m["local"],
        },
        "dummy": {
            "climate_profile": "hot_humid",
            "envelope_factor": env_f,
            "occupancy_scale": 1.0,
        },
    }
    contexts.append(ctx)

YAML_OUT.parent.mkdir(exist_ok=True)
with open(YAML_OUT, "w", encoding="utf-8") as f:
    yaml.dump({"contexts": contexts}, f, allow_unicode=True,
              default_flow_style=False, sort_keys=False)

print(f"Saved: {YAML_OUT}  ({len(contexts)} contexts)")
for c in contexts:
    print(f"  [{c['transfer_bucket']:6}]  {c['id']}")

# ── Summary ────────────────────────────────────────────────────────────────────
sec("SUMMARY")
total_sz = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
print(f"Data folder   : {OUT_DIR}/  ({total_sz/1e6:.1f} MB)")
print(f"Vietnam EPW   : {len(downloaded_epw)} files")
print(f"  TMY  : {[Path(e).name for e in downloaded_epw if 'TMY' in e]}")
print(f"  AMY  : {[Path(e).name for e in downloaded_epw if 'AMY' in e]}")
print(f"Building models: {len(downloaded_models)}")
print(f"Contexts YAML : {YAML_OUT}")
print("""
Tại sao chỉ có HCMC?
  HOT chọn 1 thành phố đại diện cho mỗi ASHRAE climate zone.
  Vietnam chỉ có 1 zone được chọn: Zone 0A (Extremely Hot Humid) = HCMC.
  Hanoi (Zone 2A) và Da Nang (Zone 1A) không được chọn vì zone đó đã có
  đại diện từ quốc gia khác. Để nghiên cứu VN đầy đủ, cần bổ sung EPW
  Hanoi/Da Nang từ climate.onebuilding.org riêng biệt.
""")
