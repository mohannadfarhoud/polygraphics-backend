#!/usr/bin/env python3
"""Batch background removal with local rembg models (no API key required)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _supported_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    s = value.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("bg color must be a 6-char hex RGB string, e.g. 000000")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch local background removal with rembg")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="u2net",
        help="rembg model name (u2net, u2netp, isnet-general-use, birefnet-general, ...)",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default=None,
        help="Optional solid background color hex (e.g. 000000). Omit for transparent PNG.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    try:
        from rembg import new_session, remove
    except ImportError as exc:
        raise SystemExit("rembg not installed. Install with: pip install rembg") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow missing. Install with: pip install pillow") from exc
    from io import BytesIO

    session = new_session(args.model)
    paths = _supported_images(input_dir)
    if not paths:
        raise SystemExit(f"No supported images found under: {input_dir}")

    bg_rgb = _hex_to_rgb(args.bg_color) if args.bg_color else None
    ok = 0
    print(f"[rembg] model={args.model} | processing {len(paths)} image(s)")
    for idx, src in enumerate(paths, start=1):
        try:
            out_bytes = remove(src.read_bytes(), session=session)
            dest = output_dir / f"{src.stem}.png"
            if bg_rgb is None:
                dest.write_bytes(out_bytes)
            else:
                # Re-open from bytes to preserve alpha output regardless of input format.
                rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")
                bg = Image.new("RGB", rgba.size, bg_rgb)
                bg.paste(rgba, mask=rgba.getchannel("A"))
                bg.save(dest, format="PNG")
            ok += 1
            print(f"[{idx}/{len(paths)}] ok  {src.name} -> {dest.name}")
        except Exception as exc:
            print(f"[{idx}/{len(paths)}] err {src.name}: {exc}")

    print(f"[rembg] done: {ok}/{len(paths)} successful | output={output_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

