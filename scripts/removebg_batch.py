#!/usr/bin/env python3
"""Batch background removal via remove.bg (removebg.py client).

This script is for A/B testing against SAM outputs:
  - Input: full-frame photos
  - Output: isolated PNGs under --output-dir

Requires:
  pip install removebg
  set REMOVE_BG_API_KEY=<your remove.bg API key>
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


def _supported_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _call_removebg(
    *,
    client,
    src: Path,
    dest: Path,
    size: str,
    bg_color: str | None,
) -> None:
    """Call removebg.py with signature-compatible kwargs.

    removebg.py has changed signatures across versions, so this uses reflection.
    """
    fn = client.remove_background_from_img_file
    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs: dict[str, object] = {}

    if "size" in params:
        kwargs["size"] = size
    if "bg_color" in params and bg_color:
        kwargs["bg_color"] = bg_color

    # Known argument names across releases.
    output_keys = ("output_file_name", "output_path", "out_file", "out_path")
    out_key = next((k for k in output_keys if k in params), None)
    if out_key is not None:
        kwargs[out_key] = str(dest)
        fn(str(src), **kwargs)
        return

    # Older variants can emit <stem>_no_bg.png near CWD / source.
    fn(str(src), **kwargs)
    candidates = [
        src.parent / f"{src.stem}_no_bg.png",
        Path.cwd() / f"{src.stem}_no_bg.png",
    ]
    for c in candidates:
        if c.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(c), str(dest))
            return
    raise RuntimeError(
        "removebg.py finished but output file was not found. "
        "Try upgrading removebg and re-run."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch remove backgrounds with remove.bg API")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing source photos")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for isolated PNG outputs")
    parser.add_argument("--api-key", type=str, default=None, help="remove.bg API key (or use REMOVE_BG_API_KEY env)")
    parser.add_argument(
        "--size",
        type=str,
        default="auto",
        choices=("auto", "preview", "small", "medium", "hd", "4k", "full"),
        help="remove.bg output size profile",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default=None,
        help="Optional hex color (e.g. FFFFFF) for solid background. Omit to keep transparent PNG.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.getenv("REMOVE_BG_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set REMOVE_BG_API_KEY.")

    try:
        from removebg import RemoveBg
    except ImportError as exc:
        raise SystemExit("removebg package missing. Install with: pip install removebg") from exc

    paths = _supported_images(input_dir)
    if not paths:
        raise SystemExit(f"No supported images found under: {input_dir}")

    log_path = output_dir / "removebg_error.log"
    client = RemoveBg(api_key, str(log_path))

    print(f"[removebg] processing {len(paths)} image(s)")
    ok = 0
    for idx, src in enumerate(paths, start=1):
        dest = output_dir / f"{src.stem}.png"
        try:
            _call_removebg(client=client, src=src, dest=dest, size=args.size, bg_color=args.bg_color)
            ok += 1
            print(f"[{idx}/{len(paths)}] ok  {src.name} -> {dest.name}")
        except Exception as exc:
            print(f"[{idx}/{len(paths)}] err {src.name}: {exc}")
    print(f"[removebg] done: {ok}/{len(paths)} successful | output={output_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

