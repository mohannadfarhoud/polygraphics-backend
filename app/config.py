from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path
    output_dir_name: str = "output"
    masked_dir_name: str = "masked"
    masks_dir_name: str = "masks"
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    poisson_depth: int = 9
    poisson_density_quantile: float = 0.02
    decimation_target_triangles: int = 120_000
    cdn_base_url: str = "http://127.0.0.1:8000/output"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / self.output_dir_name

    @property
    def masked_dir(self) -> Path:
        return self.root_dir / self.masked_dir_name

    @property
    def masks_dir(self) -> Path:
        return self.root_dir / self.masks_dir_name

