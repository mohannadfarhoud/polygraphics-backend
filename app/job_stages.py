"""Stable, machine-readable values written to ``JobRecord.stage`` while a job runs.

The UI should switch on the ``base`` of the stage string (everything before the
first space). The optional suffix in parentheses is for humans (e.g.
``"phase_1_segmentation (3/27)"``).
"""

from __future__ import annotations

from typing import Final

# Lifecycle markers (not phases of the protocol, but emitted in `stage`).
STAGE_STARTING: Final = "starting"
STAGE_EXPORTING: Final = "exporting"
STAGE_COMPLETED: Final = "completed"

# Pipeline protocol — phases 1..5.
STAGE_PHASE_1_SEGMENTATION: Final = "phase_1_segmentation"
STAGE_PHASE_2_ALIGNMENT: Final = "phase_2_alignment"
STAGE_PHASE_3_SANITIZATION: Final = "phase_3_sanitization"
STAGE_PHASE_4_COLMAP_BRIDGE: Final = "phase_4_colmap_bridge"   # neural seed → COLMAP text for GS/train.py reader
STAGE_PHASE_4_COLMAP_SCENE: Final = "phase_4_colmap_scene"     # deprecated — kept for progress JSON compat
STAGE_PHASE_5_GAUSSIAN_SPLATTING: Final = "phase_5_gaussian_splatting"

# Mesh-only finishing step (only emitted on the .glb backend).
STAGE_MESHING: Final = "meshing"
STAGE_VERTEX_COLOR_TRANSFER: Final = "vertex_color_transfer"
STAGE_PHOTO_VERTEX_BAKE: Final = "photo_vertex_bake"
STAGE_MESH_CLEANUP: Final = "mesh_cleanup"
STAGE_COLOR_AUTOBALANCE: Final = "color_autobalance"
STAGE_COMPARE_MESH_PREVIEW: Final = "compare_mesh_preview"

ALL_STAGES: Final[tuple[str, ...]] = (
    STAGE_STARTING,
    STAGE_PHASE_1_SEGMENTATION,
    STAGE_PHASE_2_ALIGNMENT,
    STAGE_PHASE_3_SANITIZATION,
    STAGE_PHASE_4_COLMAP_BRIDGE,
    STAGE_PHASE_4_COLMAP_SCENE,
    STAGE_PHASE_5_GAUSSIAN_SPLATTING,
    STAGE_MESHING,
    STAGE_VERTEX_COLOR_TRANSFER,
    STAGE_PHOTO_VERTEX_BAKE,
    STAGE_MESH_CLEANUP,
    STAGE_COLOR_AUTOBALANCE,
    STAGE_COMPARE_MESH_PREVIEW,
    STAGE_EXPORTING,
    STAGE_COMPLETED,
)
