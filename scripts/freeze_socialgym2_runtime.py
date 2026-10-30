"""Archive the shared external runtime patch and its binary/source hashes."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


PATCHED_FILES = (
    "submodules/pedsim_ros/src/pedsim_ros/3rdparty/libpedsim/include/pedsim/ped_agent.h",
    "submodules/pedsim_ros/src/pedsim_ros/3rdparty/libpedsim/src/ped_agent.cpp",
    "submodules/graph_navigation/src/navigation/social_main.cc",
    "submodules/graph_navigation/src/navigation/navigation.cc",
    "submodules/ut_multirobot_sim/src/simulator/simulator_main.cpp",
    "submodules/ut_multirobot_sim/src/simulator/simulator.cpp",
    "submodules/pedsim_ros/src/pedsim_ros/pedsim_srvs/srv/StepPedsim.srv",
    "submodules/pedsim_ros/src/pedsim_ros/pedsim_simulator/src/simulator.cpp",
    "submodules/pedsim_ros/src/pedsim_ros/pedsim_simulator/src/scene.cpp",
    "submodules/pedsim_ros/src/pedsim_ros/pedsim_simulator/include/pedsim_simulator/simulator.h",
    "submodules/pedsim_ros/src/pedsim_ros/pedsim_simulator/include/pedsim_simulator/scene.h",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--vfs", required=True, type=Path)
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    patch = []
    sources = {}
    for relative in PATCHED_FILES:
        original = args.originals / relative
        patched = args.source / relative
        cached = args.vfs / "home/rosdev/social_gym" / relative
        if digest(patched) != digest(cached):
            raise ValueError(f"build source differs from declared checkout: {relative}")
        patch.extend(difflib.unified_diff(original.read_text(encoding="utf-8").splitlines(keepends=True),
                                        patched.read_text(encoding="utf-8").splitlines(keepends=True),
                                        fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        sources[relative] = {"original_sha256": digest(original), "patched_sha256": digest(patched)}
    patch_path = args.output_dir / "runtime.patch"
    patch_path.write_text("".join(patch), encoding="utf-8")
    binary_paths = [
        args.vfs / "home/rosdev/social_gym/submodules/pedsim_ros/devel/lib/libpedsim.so",
        args.vfs / "home/rosdev/social_gym/submodules/ut_multirobot_sim/bin/simulator",
        args.vfs / "home/rosdev/social_gym/submodules/pedsim_ros/devel/lib/pedsim_simulator/pedsim_simulator",
        args.vfs / "home/rosdev/social_gym/submodules/graph_navigation/bin/social_nav",
    ]
    result = {
        "protocol": "socialgym2-joint-synchronous-runtime-v5",
        "boundary": "shared upstream runtime with opt-in reset/clock/RNG and multi-robot state-transport patches; not unmodified official reproduction",
        "source_root": str(args.source.resolve()),
        "sources": sources,
        "binaries": {str(path.resolve()): digest(path) for path in binary_paths},
        "patch_sha256": digest(patch_path),
        "applies_equally_to_all_policies": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
