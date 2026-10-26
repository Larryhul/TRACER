"""Download and validate the public Bi3 JSON release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_URL = (
    "https://www.dropbox.com/scl/fo/ix27cqjcknemh01u9i2f3/"
    "ANbzit4ZiK9FsnAVKGiWai0?rlkey=b4pg0n0m1wjf3j5d6g8z3eqkb&"
    "st=ol52lejh&dl=1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_paths(root: Path) -> list[Path]:
    return sorted(root.glob("jsons/*/*/*.json"))


def _validate_archive(archive: Path) -> None:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Bi3 download is not a ZIP archive: {archive}")


def _extract_archive(archive: Path, destination: Path, overwrite: bool, merge: bool) -> dict:
    existing = _json_paths(destination)
    if existing and not overwrite and not merge:
        return {
            "status": "already_present",
            "destination": str(destination),
            "json_count": len(existing),
        }
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bi3_extract_", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            selected = {}
            for member in handle.infolist():
                path = PurePosixPath(member.filename)
                if ".." in path.parts or "\\" in member.filename:
                    raise ValueError(f"unsafe archive path: {member.filename}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"archive symlinks are not supported: {member.filename}")
                if member.is_dir():
                    continue
                parts = path.parts
                if path.is_absolute():
                    parts = parts[1:]
                if "jsons" in parts:
                    relative = PurePosixPath(*parts[parts.index("jsons"):])
                elif len(parts) >= 3 and parts[0] in ("um", "laas"):
                    relative = PurePosixPath("jsons", *parts)
                else:
                    continue
                if len(relative.parts) != 4 or relative.parts[1] not in ("um", "laas") or relative.suffix != ".json":
                    continue
                if relative in selected:
                    raise ValueError(f"duplicate JSON path in archive: {relative}")
                selected[relative] = member
            if not selected:
                raise FileNotFoundError("no jsons/{um,laas}/*/*.json files found in archive")
            required_bytes = sum(member.file_size for member in selected.values())
            if required_bytes + 64 * 1024 * 1024 > shutil.disk_usage(destination.parent).free:
                raise OSError("insufficient free space for Bi3 JSON extraction")
            for relative, member in selected.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        target = destination / "jsons"
        staged = staging / "jsons"
        if merge and target.exists():
            merge_files = []
            for source in sorted(staged.rglob("*.json")):
                relative = source.relative_to(staged)
                target_file = target / relative
                if target_file.exists():
                    if _sha256(source) != _sha256(target_file):
                        raise ValueError(f"conflicting JSON path during merge: {relative}")
                    continue
                merge_files.append((source, target_file))
            for source, target_file in merge_files:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_file)
        else:
            backup = staging / "previous_jsons"
            if target.exists():
                if not overwrite:
                    raise FileExistsError(f"{target} exists; pass --overwrite to replace JSON data")
                target.rename(backup)
            try:
                staged.rename(target)
            except OSError:
                if backup.exists():
                    backup.rename(target)
                raise
    json_paths = _json_paths(destination)
    return {
        "status": "installed",
        "destination": str(destination),
        "json_count": len(json_paths),
    }


def install_bi3_archive(
    archive: Path,
    destination: Path,
    overwrite: bool = False,
    merge: bool = False,
    source_url: str | None = None,
) -> dict:
    destination = destination.resolve()
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    _validate_archive(archive)
    result = _extract_archive(archive, destination, overwrite, merge)
    if result["status"] == "already_present":
        return result
    result.update(
        {
            "source_archive": str(archive),
            "archive_sha256": _sha256(archive),
            "expected_layout": "jsons/{um,laas}/{experiment}/*.json",
            "merge": merge,
        }
    )
    if source_url is not None:
        result["source_url"] = source_url
    (destination / "download_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def download_bi3(
    url: str,
    destination: Path,
    overwrite: bool = False,
    timeout: int = 120,
) -> dict:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    existing = _json_paths(destination)
    if existing and not overwrite:
        return {
            "status": "already_present",
            "destination": str(destination),
            "json_count": len(existing),
        }

    with tempfile.TemporaryDirectory(prefix="bi3_download_") as temporary:
        archive = Path(temporary) / "bi3_shared.zip"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ICRA2027-real-response-preparation/1.0"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        manifest = install_bi3_archive(archive, destination, overwrite, source_url=url)
    manifest["status"] = "downloaded"
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("raw_data/external/bi3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.archive is not None:
        result = install_bi3_archive(
            args.archive,
            args.destination,
            overwrite=args.overwrite,
            merge=args.merge,
            source_url=args.url,
        )
    else:
        result = download_bi3(args.url, args.destination, args.overwrite, args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
