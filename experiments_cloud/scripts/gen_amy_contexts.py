#!/usr/bin/env python3
"""
Generate AMY (Actual Meteorological Year) eval contexts from a base contexts YAML.

Swaps the TMYx EPW for AMY EPW files (e.g. 2022, 2023, 2024) so we can measure
how trained policies hold up under year-specific weather. Currently only HCMC
has AMY EPW available locally, so by default we only emit HCMC × buildings × years.

Usage:
    python scripts/gen_amy_contexts.py \
        --base-contexts configs/contexts_vietnam_multicity.yaml \
        --amy-years 2022 2023 2024 \
        --city hcmc \
        --out configs/contexts_vietnam_amy.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
AMY_DIR = ROOT / "data/vietnam_hot/weather/weather/real_base"

CITY_TO_STATION = {
    "hcmc": "VNM_SVN_Ho.Chi.Minh-Tan.Son.Nhat.Intl.AP.489000",
}


def amy_epw_path(city: str, year: int) -> Path:
    station = CITY_TO_STATION.get(city.lower())
    if station is None:
        raise ValueError(f"No AMY station mapping for city={city}")
    return AMY_DIR / f"{station}_AMY_{year}.epw"


def make_amy_variant(base: dict, year: int) -> dict:
    """Return a deep-copied context with AMY year substituted in EPW + id."""
    ctx = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    new_id = f"{base['id']}_amy{year}"
    ctx["id"] = new_id
    ctx["weather_type"] = f"AMY_{year}"
    ctx["shift_type"] = f"amy_{year}"
    # Override EPW; keep model file as-is
    hot = dict(base.get("hot", {}))
    epw = amy_epw_path("hcmc", year)
    hot["epw_file"] = str(epw.relative_to(ROOT))
    ctx["hot"] = hot
    return ctx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-contexts", required=True)
    p.add_argument("--amy-years", nargs="+", type=int, required=True)
    p.add_argument("--city", default="hcmc",
                   help="City filter — only base contexts whose location starts with this are kept.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base_path = Path(args.base_contexts)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    with base_path.open() as f:
        base = yaml.safe_load(f)

    city_norm = args.city.lower().replace("_", "")
    src_ctxs = [c for c in base["contexts"] if city_norm in c["id"].lower()]
    if not src_ctxs:
        sys.exit(f"No base contexts matched city='{args.city}'")

    # Verify EPWs exist for every requested year
    missing = []
    for year in args.amy_years:
        ep = amy_epw_path(args.city, year)
        if not ep.exists():
            missing.append(str(ep))
    if missing:
        sys.exit("Missing AMY EPW files:\n  " + "\n  ".join(missing))

    out_ctxs = []
    for ctx in src_ctxs:
        for year in args.amy_years:
            out_ctxs.append(make_amy_variant(ctx, year))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump({"contexts": out_ctxs}, f, sort_keys=False)

    print(f"Wrote {len(out_ctxs)} AMY contexts → {out_path}")
    for c in out_ctxs:
        print(f"  - {c['id']}  EPW={Path(c['hot']['epw_file']).name}")


if __name__ == "__main__":
    main()
