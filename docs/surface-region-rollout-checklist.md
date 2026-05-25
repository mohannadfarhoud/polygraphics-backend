# Surface-Region Validation and Rollout

## Automated gates

1. Backend unit/integration:
   - `python -m unittest discover -s tests -p "test_*.py"`
2. Android compile gate:
   - `.\gradlew.bat :app:compileDebugEnvDebugKotlin`
3. API contract smoke (manual curl):
   - `GET /job-stages` includes:
     - `phase_surface_region_extract`
     - `phase_surface_region_projection`
     - `phase_surface_region_blend`
   - `GET /jobs/{id}` returns optional fields when available:
     - `dominant_surface_regions_detected`
     - `per_view_region_confidence`
     - `region_projection_coverage`
     - `texture_route_taken`
     - `texture_quality_reason`
     - `texture_report_url`

## Real-device validation matrix

- Devices: at least 3 tiers (mid, upper-mid, flagship).
- Objects:
  - shiny/reflective body
  - low-texture matte object
  - symmetric object (left-right ambiguity)
- Lighting:
  - diffuse daylight
  - indoor mixed lighting
  - low-light with noise

## Acceptance gates

- Visual seam artifacts reduced by >= 15% vs `mapanything` baseline.
- Dominant-surface sharpness improved by >= 10% vs baseline.
- Failures that miss region confidence/coverage must route to:
  - `mapanything` or
  - `surface_region+mapanything_fill`
- API diagnostics must expose route + reason for every processed job.

## Rollout

1. Internal test: enable `surface_region_texture_enabled=true` for internal jobs only.
2. Staging: enable with default settings and review telemetry:
   - median `mesh_area_ratio`
   - fallback rate to `mapanything`
   - user-visible failure rate
3. Production canary: 10% traffic, then 50%, then 100% after two clean windows.
