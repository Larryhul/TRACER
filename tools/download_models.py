"""Download the release model archive and verify every extracted checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile


def download(output, url=None):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "provenance/model_release.json").read_text(encoding="utf-8"))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / spec["filename"]
        with urllib.request.urlopen(url or spec["url"], timeout=120) as response, archive.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Model archive checksum mismatch")
        with zipfile.ZipFile(archive) as bundle:
            for item in spec["files"]:
                name = item["name"]
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Invalid checkpoint member path")
                data = bundle.read(name)
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Checkpoint checksum mismatch: " + name)
                destination = output / name
                if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                    raise FileExistsError("A different checkpoint already exists: " + str(destination))
                destination.write_bytes(data)
                print("Verified:", destination)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/external/checkpoints")
    parser.add_argument("--url", help="Optional mirror URL; the recorded checksum is always enforced")
    args = parser.parse_args()
    download(args.output, args.url)
