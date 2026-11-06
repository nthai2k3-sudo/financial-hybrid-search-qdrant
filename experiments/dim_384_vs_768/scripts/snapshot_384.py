from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import AUDIT, ROOT, sha256_file, system_metadata


def main() -> None:
    source = ROOT / "qdrant_data"
    snapshot = AUDIT / "snapshot_384_qdrant_data"
    if not source.exists():
        raise FileNotFoundError(source)
    if snapshot.exists():
        raise FileExistsError(f"Snapshot already exists: {snapshot}")
    AUDIT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, snapshot)
    manifest = {
        "source": str(source),
        "snapshot": str(snapshot),
        "metadata": system_metadata(),
        "files": [],
    }
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            manifest["files"].append(
                {
                    "path": str(path.relative_to(snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    (AUDIT / "snapshot_384.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
