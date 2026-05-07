from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RuntimeSettings(BaseModel):
    # `dust3r`/`colmap` produce a meshed `.glb`; `gaussian_splatting` produces a `.ply` (3DGS).
    reconstruction_backend: Literal["dust3r", "colmap", "gaussian_splatting"] = "dust3r"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    sam_checkpoint_path: str | None = None
    sam_model_type: str = "vit_h"
    dust3r_repo_path: str | None = None
    dust3r_checkpoint_path: str | None = None
    colmap_binary_path: str | None = None
    # Gaussian Splatting (https://github.com/graphdeco-inria/gaussian-splatting)
    gs_repo_path: str | None = None
    gs_python_executable: str | None = None  # leave null to use the API's Python
    gs_init_source: Literal["colmap", "dust3r"] = "colmap"
    gs_iterations: int = Field(default=7000, ge=100, le=60000)
    gs_sh_degree: int = Field(default=3, ge=0, le=4)
    gs_resolution: int = Field(default=-1, ge=-1, le=8192)  # -1 = original
    meshing_method: Literal["poisson", "bpa"] = "poisson"
    output_dir_name: str = "output"
    masked_dir_name: str = "masked"
    nb_neighbors: int = Field(default=20, ge=1)
    std_ratio: float = Field(default=2.0, gt=0)
    poisson_depth: int = Field(default=9, ge=4, le=14)
    poisson_density_quantile: float = Field(default=0.02, ge=0.0, le=1.0)
    decimation_target_triangles: int = Field(default=120_000, ge=1000)
    # Public URL prefix for generated model files (must match where this API serves /output/…).
    cdn_base_url: str = "http://127.0.0.1:8000/output"
    max_images: int = Field(default=100, ge=2, le=1000)
    max_image_side: int = Field(default=1024, ge=128, le=8192)
    # When True: fake masks/points/GS PLY (demo only). When False: real checkpoints + packages required.
    allow_placeholder_pipeline: bool = False


class SettingsStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.path = self.root_dir / "config" / "runtime_settings.json"

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            settings = RuntimeSettings()
            self.save(settings)
            return settings

        data = json.loads(self.path.read_text(encoding="utf-8"))
        return RuntimeSettings.model_validate(data)

    def save(self, settings: RuntimeSettings) -> RuntimeSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        return settings

