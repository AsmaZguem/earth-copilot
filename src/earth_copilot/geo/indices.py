"""Spectral indices. A small built-in set for now; step 2 moves index choice to the RAG catalogue."""

import xarray as xr

# Index name -> (Earth Search band names, formula)
INDICES = {
    "NDVI": (["nir", "red"], lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"])),
    "NDWI": (["green", "nir"], lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"])),
    "MNDWI": (
        ["green", "swir16"],
        lambda b: (b["green"] - b["swir16"]) / (b["green"] + b["swir16"]),
    ),
}


def bands_for(index: str) -> list[str]:
    return INDICES[index.upper()][0]


def compute_index(index: str, ds: xr.Dataset) -> xr.DataArray:
    _, formula = INDICES[index.upper()]
    result = formula(ds).clip(-1, 1)
    result.name = index.upper()
    return result
