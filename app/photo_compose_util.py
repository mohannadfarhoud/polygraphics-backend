from __future__ import annotations


def placement_pixels(
    placement_x: float,
    placement_y: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    px = int(round(placement_x * image_width))
    py = int(round(placement_y * image_height))
    px = max(0, min(image_width - 1, px))
    py = max(0, min(image_height - 1, py))
    return px, py
