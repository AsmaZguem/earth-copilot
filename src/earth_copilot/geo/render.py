"""Render an index raster to a transparent PNG that MapLibre can overlay as an image source."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import odc.geo.xr  # noqa: F401  (registers the .odc accessor)
import xarray as xr


def save_png(
    da: xr.DataArray, path: str | Path, cmap: str = "RdYlGn", vmin: float = -1, vmax: float = 1
) -> list[list[float]]:
    """Write the PNG and return its corners in lon/lat, in MapLibre order (TL, TR, BR, BL)."""
    da = da.odc.reproject("EPSG:4326")

    colormap = matplotlib.colormaps[cmap].copy()
    colormap.set_bad(alpha=0)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = colormap(norm(np.ma.masked_invalid(da.values)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, rgba)

    left, bottom, right, top = da.odc.geobox.boundingbox
    return [[left, top], [right, top], [right, bottom], [left, bottom]]
