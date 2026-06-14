"""AI image editing for photo try-on compose (Gemini 2.5 Flash Image)."""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 2000


def draw_placement_marker(
    face_path: Path,
    *,
    px: int,
    py: int,
    out_path: Path,
) -> Path:
    """Copy face image with a visible marker dot at the earlobe target."""
    from PIL import Image, ImageDraw

    img = Image.open(face_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(6, min(img.width, img.height) // 80)
    draw.ellipse((px - r, py - r, px + r, py + r), fill=(0, 200, 120), outline=(255, 255, 255), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=92)
    return out_path


def _build_prompt(
    *,
    px: int,
    py: int,
    image_width: int,
    image_height: int,
    placement_side: str,
    user_prompt: str | None,
) -> str:
    extra = (user_prompt or "").strip()
    if len(extra) > MAX_PROMPT_CHARS:
        extra = extra[:MAX_PROMPT_CHARS]
    extra_block = f"\n{extra}" if extra else ""
    return (
        "You are a professional jewelry photo retoucher.\n\n"
        "Image 1: A person's face photo (side or front profile). "
        "A marker dot shows where the earring should attach on the earlobe.\n"
        "Image 2: Product photo of the earrings.\n\n"
        "Task: Edit image 1 so the person is wearing the earrings from image 2. "
        "The earring hook must attach at the marker dot on the earlobe. "
        "Match the earring design exactly (shape, metal, stones). "
        "Keep the person's face, skin, hair, expression, clothing, and background unchanged. "
        "Match the lighting and shadows of the original photo. Photorealistic result.\n\n"
        f"Placement hint: pixel ({px}, {py}) in a {image_width}x{image_height} image.\n"
        f"Ear side: {placement_side}."
        f"{extra_block}"
    )


def _dev_mock_enabled() -> bool:
    return os.getenv("PHOTO_COMPOSE_DEV_MOCK", "").strip().lower() in ("1", "true", "yes")


def _gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def compose_photorealistic(
    *,
    marked_face_path: Path,
    earring_path: Path,
    result_path: Path,
    px: int,
    py: int,
    image_width: int,
    image_height: int,
    placement_side: str,
    user_prompt: str | None,
) -> str | None:
    """Run image editing; save JPEG to ``result_path``. Returns optional revised prompt."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(
        px=px,
        py=py,
        image_width=image_width,
        image_height=image_height,
        placement_side=placement_side,
        user_prompt=user_prompt,
    )

    if _dev_mock_enabled() and not _gemini_configured():
        shutil.copy2(marked_face_path, result_path)
        return "dev-mock: copied marked face (set GEMINI_API_KEY for real compose)"

    if not _gemini_configured():
        raise RuntimeError("GEMINI_API_KEY is not configured on the API server")

    model = os.getenv("PHOTO_COMPOSE_MODEL", "gemini-2.5-flash-image").strip()
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package is not installed. pip install google-genai"
        ) from exc

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    face_bytes = marked_face_path.read_bytes()
    earring_bytes = earring_path.read_bytes()
    face_mime = "image/jpeg" if marked_face_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    ear_mime = "image/jpeg" if earring_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    contents = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=face_bytes, mime_type=face_mime),
        types.Part.from_bytes(data=earring_bytes, mime_type=ear_mime),
    ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    revised: str | None = None
    saved = False
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if getattr(part, "text", None):
                revised = (revised or "") + part.text
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                from PIL import Image

                raw = inline.data
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if img.width != image_width or img.height != image_height:
                    img = img.resize((image_width, image_height), Image.Resampling.LANCZOS)
                img.save(result_path, format="JPEG", quality=92)
                saved = True
                break

    if not saved:
        raise RuntimeError("Gemini returned no image in the response")

    return revised
