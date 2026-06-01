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

# Pipeline protocol — phases 0..5.
STAGE_PHASE_0_INPUT_CURATION: Final = "phase_0_input_curation"
# Backward-compat alias for older clients.
STAGE_PHASE_0_QUALITY_FILTER: Final = "phase_0_quality_filter"
STAGE_PHASE_1_SEGMENTATION: Final = "phase_1_segmentation"
STAGE_PHASE_DEPTH_NORMALIZATION: Final = "phase_depth_normalization"
STAGE_PHASE_2_ALIGNMENT: Final = "phase_2_alignment"
STAGE_PHASE_3_SANITIZATION: Final = "phase_3_sanitization"
STAGE_PHASE_4_COLMAP_BRIDGE: Final = "phase_4_colmap_bridge"   # neural seed → COLMAP text for GS/train.py reader
STAGE_PHASE_4_COLMAP_SCENE: Final = "phase_4_colmap_scene"     # deprecated — kept for progress JSON compat
STAGE_PHASE_5_GAUSSIAN_SPLATTING: Final = "phase_5_gaussian_splatting"
STAGE_PHASE_CONFIDENCE_ROUTING: Final = "phase_confidence_routing"
STAGE_PHASE_PRIOR_GENERATION: Final = "phase_prior_generation"
STAGE_PHASE_PRIOR_REFINEMENT: Final = "phase_prior_refinement"

# Mesh-only finishing step (only emitted on the .glb backend).
STAGE_MESHING: Final = "meshing"
STAGE_PHASE_VIDEO_EXTRACTION: Final = "phase_video_extraction"
STAGE_PHASE_VIDEO_FRAME_SELECTION: Final = "phase_video_frame_selection"
STAGE_SHAPE_CLASSIFICATION: Final = "shape_classification"
STAGE_SHAPE_TEMPLATE_CORRECTION: Final = "shape_template_correction"
STAGE_VERTEX_COLOR_TRANSFER: Final = "vertex_color_transfer"
STAGE_PHOTO_VERTEX_BAKE: Final = "photo_vertex_bake"
STAGE_PHASE_SURFACE_REGION_EXTRACT: Final = "phase_surface_region_extract"
STAGE_PHASE_SURFACE_REGION_PROJECTION: Final = "phase_surface_region_projection"
STAGE_PHASE_SURFACE_REGION_BLEND: Final = "phase_surface_region_blend"
STAGE_MESH_CLEANUP: Final = "mesh_cleanup"
STAGE_COLOR_AUTOBALANCE: Final = "color_autobalance"
STAGE_COMPARE_MESH_PREVIEW: Final = "compare_mesh_preview"

ALL_STAGES: Final[tuple[str, ...]] = (
    STAGE_STARTING,
    STAGE_PHASE_0_INPUT_CURATION,
    STAGE_PHASE_0_QUALITY_FILTER,
    STAGE_PHASE_1_SEGMENTATION,
    STAGE_PHASE_DEPTH_NORMALIZATION,
    STAGE_PHASE_2_ALIGNMENT,
    STAGE_PHASE_3_SANITIZATION,
    STAGE_PHASE_4_COLMAP_BRIDGE,
    STAGE_PHASE_4_COLMAP_SCENE,
    STAGE_PHASE_5_GAUSSIAN_SPLATTING,
    STAGE_PHASE_VIDEO_EXTRACTION,
    STAGE_PHASE_VIDEO_FRAME_SELECTION,
    STAGE_PHASE_CONFIDENCE_ROUTING,
    STAGE_PHASE_PRIOR_GENERATION,
    STAGE_PHASE_PRIOR_REFINEMENT,
    STAGE_MESHING,
    STAGE_SHAPE_CLASSIFICATION,
    STAGE_SHAPE_TEMPLATE_CORRECTION,
    STAGE_VERTEX_COLOR_TRANSFER,
    STAGE_PHOTO_VERTEX_BAKE,
    STAGE_PHASE_SURFACE_REGION_EXTRACT,
    STAGE_PHASE_SURFACE_REGION_PROJECTION,
    STAGE_PHASE_SURFACE_REGION_BLEND,
    STAGE_MESH_CLEANUP,
    STAGE_COLOR_AUTOBALANCE,
    STAGE_COMPARE_MESH_PREVIEW,
    STAGE_EXPORTING,
    STAGE_COMPLETED,
)
