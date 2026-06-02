#!/usr/bin/env python3
"""Write openapi.json from the FastAPI app schema (run from repo root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.main import app  # noqa: E402


def main() -> None:
    schema = app.openapi()
    out = _REPO / "openapi.json"
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
