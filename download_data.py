"""Download public Seattle snapshots; no key or private data is required."""

import argparse
import gzip
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "streets": "https://sidewalk-sea.cs.washington.edu/v3/api/streets?filetype=geojson",
    "clusters": "https://sidewalk-sea.cs.washington.edu/v3/api/labelClusters?filetype=geojson",
    "places": "https://sidewalk-sea.cs.washington.edu/v3/api/places?filetype=geojson",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--import-dir", type=Path, help="Import already downloaded originals")
    args = parser.parse_args()
    target = ROOT / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in SOURCES.items():
        out = target / f"{name}.geojson.gz"
        if out.exists() and not args.refresh:
            print(f"Using snapshot: {out.name}")
            continue
        if args.import_dir:
            original = args.import_dir / f"{name}.geojson"
            content = original.read_bytes()
            retrieved = datetime.fromtimestamp(original.stat().st_mtime, timezone.utc)
        else:
            # Large API exports may take several minutes. This is an explicit
            # total deadline as well as a per-read timeout.
            started = time.monotonic()
            parts = []
            with urlopen(url, timeout=55) as response:
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Export exceeded 10 minutes; retry later.")
                    parts.append(chunk)
            content = b"".join(parts)
            retrieved = datetime.now(timezone.utc)
        data = json.loads(content)
        if data.get("type") != "FeatureCollection" or not data.get("features"):
            raise ValueError(f"{name}: empty or unexpected API response")
        with out.open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                compressed.write(content)
        manifest[name] = {
            "url": url,
            "retrieved_at_utc": retrieved.isoformat(),
            "raw_sha256": hashlib.sha256(content).hexdigest(),
            "gzip_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "features": len(data["features"]),
            "raw_bytes": len(content),
            "license": "CC0 (annotations)" if name == "clusters" else "ODbL (OSM-derived)",
        }
        print(f"{name}: {len(data['features']):,} features; {out.stat().st_size:,} bytes gzip")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # A refresh changes the dataset; rebuild downstream outputs afterwards.
    cache = ROOT / "data" / "processed"
    if args.refresh and cache.exists():
        shutil.rmtree(cache)


if __name__ == "__main__":
    main()
