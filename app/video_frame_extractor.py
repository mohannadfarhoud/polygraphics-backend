"""Video frame extraction and quality scoring for the InstantMesh pipeline.

Uses FFmpeg (subprocess) to extract frames at a configurable rate, then scores
each frame with a combined quality metric so the reconstruction stage receives
the best possible single input image.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".avi", ".mkv"})


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FrameScore:
    path: Path
    sharpness: float      # 0..1 (Laplacian variance / 500, capped at 1)
    object_size: float    # 0..1 (foreground area ratio)
    center_alignment: float  # 0..1 (1 = centroid at image center)
    exposure: float       # 0..1 (1 = perfect mid-gray)
    total: float          # weighted composite score


@dataclass
class ExtractionResult:
    video_path: Path
    frames_dir: Path
    all_frames: list[Path] = field(default_factory=list)
    kept_frames: list[Path] = field(default_factory=list)
    scored_frames: list[FrameScore] = field(default_factory=list)
    best_frame: Path | None = None
    best_frame_score: float = 0.0
    total_extracted: int = 0
    total_kept: int = 0
    fps_used: float = 2.0


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str | None:
    """Return path to ffmpeg executable, or None if not found."""
    return shutil.which("ffmpeg")


def get_video_duration(video_path: Path, ffmpeg_bin: str = "ffmpeg") -> float | None:
    """Return video duration in seconds using ffprobe, or None on failure."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # Fallback: try to detect via ffmpeg stderr (less reliable)
        return None
    try:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            return float(proc.stdout.strip())
    except Exception:
        pass
    return None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 2.0,
    ffmpeg_bin: str | None = None,
    timeout_seconds: int = 300,
) -> list[Path]:
    """Extract frames from video at the given FPS using FFmpeg.

    Returns list of extracted frame paths sorted by name.
    """
    bin_ = ffmpeg_bin or find_ffmpeg()
    if not bin_:
        raise RuntimeError(
            "FFmpeg not found on PATH. Install FFmpeg and ensure it is accessible "
            "(https://ffmpeg.org/download.html)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove any existing frames first so stale files don't pollute the list.
    for old in output_dir.glob("frame_*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass

    fps_str = f"{fps:.4f}"
    cmd = [
        bin_,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps_str}",
        "-q:v", "2",              # high-quality JPEG
        str(output_dir / "frame_%04d.jpg"),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"FFmpeg timed out after {timeout_seconds}s while extracting frames from {video_path.name}"
        ) from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-20:]
        raise RuntimeError(
            f"FFmpeg failed (exit {proc.returncode}) extracting frames from {video_path.name}.\n"
            + "\n".join(tail)
        )

    frames = sorted(output_dir.glob("frame_*.jpg"))
    _log.info("Extracted %d frames from %s at %.1f fps", len(frames), video_path.name, fps)
    return frames


# ---------------------------------------------------------------------------
# Per-frame quality scoring (same formula as single-image AI input selection)
# ---------------------------------------------------------------------------

def _score_frame(frame_path: Path, masked_path: Path | None = None) -> FrameScore:
    """Compute per-frame quality score.

    If a masked (segmented) version exists, uses it for object_size and
    center_alignment. Otherwise falls back to full-frame estimates.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for frame scoring.") from exc

    src = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if src is None or src.size == 0:
        return FrameScore(
            path=frame_path,
            sharpness=0.0, object_size=0.0,
            center_alignment=0.0, exposure=0.0, total=0.0,
        )

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # --- Sharpness (Laplacian variance, capped at 1) ---
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, lap_var / 500.0)

    # --- Exposure (1 = perfect mid-gray 128, 0 = pure black or pure white) ---
    mean_luma = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(mean_luma - 128.0) / 128.0)

    # --- Object size + center alignment from mask (or full frame) ---
    if masked_path is not None:
        msk = cv2.imread(str(masked_path), cv2.IMREAD_COLOR)
    else:
        msk = None

    if msk is not None and msk.size > 0:
        if msk.shape[:2] != (h, w):
            msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        fg = np.any(msk > 8, axis=2)
        fg_count = float(np.count_nonzero(fg))
        total_px = float(h * w)
        object_size = min(1.0, fg_count / max(1.0, total_px))
        if fg_count > 0:
            ys, xs = np.where(fg)
            cx = float(xs.mean()) / max(1.0, float(w - 1))
            cy = float(ys.mean()) / max(1.0, float(h - 1))
            dist = float(np.hypot(cx - 0.5, cy - 0.5))
            center_alignment = max(0.0, 1.0 - dist / 0.5)
        else:
            center_alignment = 0.0
    else:
        # No mask: use a weak heuristic — assume object fills ~50% of frame
        object_size = 0.5
        center_alignment = 0.5

    total = (
        0.4 * sharpness
        + 0.3 * object_size
        + 0.2 * center_alignment
        + 0.1 * exposure
    )

    return FrameScore(
        path=frame_path,
        sharpness=sharpness,
        object_size=object_size,
        center_alignment=center_alignment,
        exposure=exposure,
        total=total,
    )


def _is_acceptable(
    score: FrameScore,
    *,
    min_sharpness: float = 0.04,
    min_exposure: float = 0.10,
) -> bool:
    """Return False for clearly bad frames (blurry or extreme exposure)."""
    if score.sharpness < min_sharpness:
        return False
    if score.exposure < min_exposure:
        return False
    return True


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def run_video_extraction(
    job_id: str,
    video_path: Path,
    upload_dir: Path,
    *,
    fps: float = 2.0,
    min_sharpness: float = 0.04,
    min_exposure: float = 0.10,
    max_frames: int = 60,
    max_duration_seconds: int = 30,
    ffmpeg_bin: str | None = None,
    ffmpeg_timeout: int = 300,
) -> ExtractionResult:
    """Full video-to-best-frame pipeline.

    1. Validate video duration (warn; don't fail).
    2. Extract frames with FFmpeg.
    3. Score and filter frames.
    4. Return ExtractionResult with best_frame set.
    """
    frames_dir = upload_dir / job_id / "frames"
    result = ExtractionResult(
        video_path=video_path,
        frames_dir=frames_dir,
        fps_used=fps,
    )

    # Duration check (advisory)
    duration = get_video_duration(video_path)
    if duration is not None and duration > max_duration_seconds:
        _log.warning(
            "job=%s: video duration %.1fs exceeds recommended max %ds; "
            "extraction may take a while and many frames will be generated.",
            job_id, duration, max_duration_seconds,
        )

    # Frame extraction
    all_frames = extract_frames(
        video_path,
        frames_dir,
        fps=fps,
        ffmpeg_bin=ffmpeg_bin,
        timeout_seconds=ffmpeg_timeout,
    )
    result.all_frames = all_frames
    result.total_extracted = len(all_frames)

    if not all_frames:
        raise RuntimeError(
            f"FFmpeg extracted 0 frames from {video_path.name}. "
            "Check that the video file is not corrupted."
        )

    # Subsample if too many frames
    if len(all_frames) > max_frames:
        step = len(all_frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        all_frames = [all_frames[i] for i in indices]
        _log.info("job=%s: subsampled %d frames → %d", job_id, result.total_extracted, len(all_frames))

    # Score all frames (no mask yet — SAM runs later)
    scored: list[FrameScore] = []
    for fp in all_frames:
        try:
            s = _score_frame(fp)
            scored.append(s)
        except Exception as exc:
            _log.warning("job=%s: failed to score frame %s: %s", job_id, fp.name, exc)

    if not scored:
        raise RuntimeError("All frames failed quality scoring.")

    # Filter bad frames; keep at least 5 (fall back to top-5 by score)
    kept = [s for s in scored if _is_acceptable(s, min_sharpness=min_sharpness, min_exposure=min_exposure)]
    if len(kept) < 5:
        kept = sorted(scored, key=lambda s: s.total, reverse=True)[:max(5, len(scored))]

    kept_sorted = sorted(kept, key=lambda s: s.total, reverse=True)
    result.scored_frames = kept_sorted
    result.kept_frames = [s.path for s in kept_sorted]
    result.total_kept = len(result.kept_frames)

    # Best frame = highest composite score
    result.best_frame = kept_sorted[0].path
    result.best_frame_score = kept_sorted[0].total

    _log.info(
        "job=%s: extracted=%d, kept=%d, best=%s (score=%.3f)",
        job_id, result.total_extracted, result.total_kept,
        result.best_frame.name, result.best_frame_score,
    )
    return result


def find_video_in_upload_dir(upload_dir: Path, job_id: str) -> Path | None:
    """Return the uploaded video file for a job, or None if no video was uploaded."""
    job_dir = upload_dir / job_id
    if not job_dir.is_dir():
        return None
    for ext in (".mp4", ".mov", ".webm", ".avi", ".mkv"):
        for candidate in (
            job_dir / f"input_video{ext}",
            job_dir / f"video{ext}",
        ):
            if candidate.is_file():
                return candidate
        # Also accept any input_000.* video upload
        for p in job_dir.glob(f"input_*{ext}"):
            if p.is_file():
                return p
    return None
