"""Write a vivid vertex-colored GLB for viewer / pipeline sanity checks.

Run from repo root (needs trimesh + numpy from requirements.txt):

    .venv\\Scripts\\python.exe scripts\\make_colorful_demo_glb.py

Output: output/demo_colorful_rainbow.glb
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_colorful_rainbow.glb"

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    # Hue from longitude + latitude for a saturated rainbow shell
    xy = verts[:, :2]
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    z = verts[:, 2]
    u = (ang / (2 * np.pi) + 0.5) % 1.0
    v = (z - z.min()) / max(z.max() - z.min(), 1e-9)
    h = (u + 0.35 * v) % 1.0
    s = np.full_like(h, 0.92, dtype=np.float64)
    v_bright = np.full_like(h, 0.95, dtype=np.float64)
    # HSV to RGB (v = v_bright)
    i = np.floor(h * 6).astype(np.int32) % 6
    f = h * 6 - np.floor(h * 6)
    p = v_bright * (1 - s)
    q = v_bright * (1 - s * f)
    t = v_bright * (1 - s * (1 - f))
    rgb = np.zeros((len(h), 3), dtype=np.float64)
    for k in range(6):
        m = i == k
        if k == 0:
            rgb[m] = np.column_stack([v_bright[m], t[m], p[m]])
        elif k == 1:
            rgb[m] = np.column_stack([q[m], v_bright[m], p[m]])
        elif k == 2:
            rgb[m] = np.column_stack([p[m], v_bright[m], t[m]])
        elif k == 3:
            rgb[m] = np.column_stack([p[m], q[m], v_bright[m]])
        elif k == 4:
            rgb[m] = np.column_stack([t[m], p[m], v_bright[m]])
        else:
            rgb[m] = np.column_stack([v_bright[m], p[m], q[m]])

    rgba = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rgba = np.column_stack([rgba, np.full((len(rgba), 1), 255, dtype=np.uint8)])
    mesh.visual.vertex_colors = rgba
    mesh.export(str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
