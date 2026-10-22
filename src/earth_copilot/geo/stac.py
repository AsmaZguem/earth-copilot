"""Find and load Sentinel-2 L2A scenes from the free Element84 Earth Search STAC API."""

import math

import odc.stac
import pystac
import xarray as xr
from pystac_client import Client

from earth_copilot.config import settings

COLLECTION = "sentinel-2-l2a"

# Scene Classification Layer values to drop: cloud shadow, cloud medium/high probability, cirrus
SCL_MASK = [3, 8, 9, 10]

BBox = tuple[float, float, float, float]  # west, south, east, north (lon/lat)

odc.stac.configure_rio(cloud_defaults=True)


def bbox_area_km2(bbox: BBox) -> float:
    west, south, east, north = bbox
    mid_lat = math.radians((south + north) / 2)
    width = (east - west) * 111.32 * math.cos(mid_lat)
    height = (north - south) * 110.57
    return abs(width * height)


def search_scenes(
    bbox: BBox, start: str, end: str, max_cloud: float | None = None, limit: int = 20
) -> list[pystac.Item]:
    """Scenes covering bbox between start and end (YYYY-MM-DD), least cloudy first."""
    area = bbox_area_km2(bbox)
    if area > settings.max_area_km2:
        raise ValueError(f"Area is {area:.0f} km², limit is {settings.max_area_km2:.0f} km²")

    max_cloud = settings.max_cloud_cover if max_cloud is None else max_cloud
    client = Client.open(settings.stac_url)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    return sorted(search.items(), key=lambda item: item.properties["eo:cloud_cover"])


def load_reflectance(
    item: pystac.Item, bands: list[str], bbox: BBox, resolution: int = 10
) -> xr.Dataset:
    """Surface reflectance (0-1) for the given bands, with clouds and no-data set to NaN."""
    ds = odc.stac.load([item], bands=[*bands, "scl"], bbox=bbox, resolution=resolution)
    ds = ds.isel(time=0)
    clear = ~ds["scl"].isin(SCL_MASK)

    # Earth Search may already have removed the baseline-04.00+ offset from the pixel values while
    # still listing it in raster:bands; applying it again would push dark pixels below zero.
    offset_applied = item.properties.get("earthsearch:boa_offset_applied", False)

    out = {}
    for band in bands:
        raster = item.assets[band].extra_fields.get("raster:bands", [{}])[0]
        scale = raster.get("scale", 1e-4)
        offset = 0.0 if offset_applied else raster.get("offset", 0.0)
        valid = (ds[band] != 0) & clear
        out[band] = (ds[band].where(valid) * scale + offset).astype("float32")
    return xr.Dataset(out)
