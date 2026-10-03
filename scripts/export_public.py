"""Copy the explicit, hash-verified public manifest to a new directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def export(root, target):
    manifest = root / "PUBLIC_MANIFEST.json"
    records = json.loads(manifest.read_text())["files"]
    if target.exists():
        raise FileExistsError(f"Use a new empty export destination: {target}")
    verified = []
    for item in records:
        rel = Path(item["path"])
        src = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or root not in src.parents:
            raise ValueError(f"Unsafe manifest path: {rel}")
        if hashlib.sha256(src.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Changed since release manifest: {rel}; regenerate after review")
        verified.append((src, rel))
    target.mkdir(parents=True)
    for src, rel in verified:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(manifest, target / manifest.name)
    return len(verified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(f"Exported {export(Path(__file__).resolve().parents[1], args.target.resolve())} verified files")
