"""Verify POST /isolation/datasets/{id}/pairs accepts one before+after without indices."""
from __future__ import annotations

import json
import struct
import zlib
from urllib import request


# Nginx strips /polygraph; uvicorn serves bare paths locally.
BASE = "http://127.0.0.1:8000"


def _png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _json(method: str, path: str, data: dict | None = None, headers: dict | None = None) -> dict:
    hdrs = dict(headers or {})
    body = None if data is None else json.dumps(data).encode()
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    req = request.Request(BASE + path, data=body, headers=hdrs, method=method)
    with request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def _multipart(path: str, *, headers: dict, files: dict[str, tuple[str, bytes, str]]) -> dict:
    boundary = "----polybound"
    parts: list[bytes] = []
    for name, (fname, content, ctype) in files.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n"
            ).encode()
        )
        parts.append(content)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    hdrs = dict(headers)
    hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = request.Request(BASE + path, data=b"".join(parts), headers=hdrs, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    login = _json("POST", "/auth/login", {"email": "trainer", "password": "devtek2026"})
    token = login["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    ds = _json("POST", "/isolation/datasets", {"name": "single-couple-check"}, headers=auth)
    dataset_id = ds["dataset_id"]
    up = _multipart(
        f"/isolation/datasets/{dataset_id}/pairs",
        headers=auth,
        files={
            "before": ("b.png", _png(8, 8, (10, 20, 30)), "image/png"),
            "after": ("a.png", _png(8, 8, (255, 255, 255)), "image/png"),
        },
    )
    assert up.get("uploaded") == 1, up
    assert up.get("pair_indices") == [0], up
    print("single_couple_ok", dataset_id, up)


if __name__ == "__main__":
    main()
