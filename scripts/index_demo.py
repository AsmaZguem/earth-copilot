"""Step 1 check, no LLM: area + dates -> least cloudy Sentinel-2 scene -> index -> PNG.

Default area: Lac Tremblant (Quebec), water + forest, so NDVI should be negative on the lake
and high on the forest.

    uv run python scripts/index_demo.py
    uv run python scripts/index_demo.py --index MNDWI --start 2024-07-01 --end 2024-08-31
"""

import argparse
import json

from earth_copilot.geo.indices import bands_for, compute_index
from earth_copilot.geo.render import save_png
from earth_copilot.geo.stac import bbox_area_km2, load_reflectance, search_scenes

DEFAULT_BBOX = (-74.62, 46.18, -74.54, 46.26)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", default="2024-06-01")
    parser.add_argument("--end", default="2024-08-31")
    parser.add_argument("--index", default="NDVI")
    args = parser.parse_args()
    bbox = tuple(args.bbox)

    print(f"Area: {bbox_area_km2(bbox):.1f} km²")
    scenes = search_scenes(bbox, args.start, args.end)
    if not scenes:
        raise SystemExit("No scene under the cloud limit for these dates.")
    item = scenes[0]
    print(f"{len(scenes)} scenes found; using {item.id} "
          f"({item.datetime:%Y-%m-%d}, {item.properties['eo:cloud_cover']:.1f}% clouds)")

    ds = load_reflectance(item, bands_for(args.index), bbox)
    index = compute_index(args.index, ds)
    print(f"{index.name}: mean {float(index.mean()):.3f}, "
          f"min {float(index.min()):.3f}, max {float(index.max()):.3f}")

    out = f"outputs/{index.name.lower()}_{item.datetime:%Y%m%d}.png"
    corners = save_png(index, out)
    print(f"PNG: {out}")
    print(f"MapLibre coordinates: {json.dumps(corners)}")


if __name__ == "__main__":
    main()
