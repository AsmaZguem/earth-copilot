"""Awesome Spectral Indices catalogue: downloaded once, each index turned into a citable document."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from earth_copilot.config import settings

BASE_URL = (
    "https://raw.githubusercontent.com/awesome-spectral-indices/awesome-spectral-indices/main/output"
)
FILES = ["spectral-indices-dict.json", "bands.json"]


@dataclass
class IndexDoc:
    short_name: str
    long_name: str
    domain: str
    formula: str
    bands: list[str]  # catalogue codes (G, S1, ...) and constants (L, g, ...)
    band_names: list[str]  # readable names of the real bands only
    platforms: list[str]
    reference: str

    @property
    def sentinel2(self) -> bool:
        return "Sentinel-2" in self.platforms

    def text(self) -> str:
        """What gets embedded and shown to the LLM."""
        return (
            f"{self.short_name}: {self.long_name}. Application domain: {self.domain}. "
            f"Bands: {', '.join(self.band_names)}. Formula: {self.formula}. "
            f"Platforms: {', '.join(self.platforms)}."
        )

    def payload(self) -> dict:
        return {**asdict(self), "sentinel2": self.sentinel2, "text": self.text()}


def download(force: bool = False) -> Path:
    folder = Path(settings.catalogue_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = folder / name
        if force or not path.exists():
            response = httpx.get(f"{BASE_URL}/{name}", timeout=60, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
    return folder


def load_catalogue() -> list[IndexDoc]:
    folder = download()
    indices = json.loads((folder / FILES[0]).read_text(encoding="utf-8"))["SpectralIndices"]
    bands = json.loads((folder / FILES[1]).read_text(encoding="utf-8"))

    return [
        IndexDoc(
            short_name=entry["short_name"],
            long_name=entry["long_name"],
            domain=entry["application_domain"],
            formula=entry["formula"],
            bands=entry["bands"],
            band_names=[bands[code]["long_name"] for code in entry["bands"] if code in bands],
            platforms=entry["platforms"],
            reference=entry["reference"],
        )
        for entry in indices.values()
    ]
