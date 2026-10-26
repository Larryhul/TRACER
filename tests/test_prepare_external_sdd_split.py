from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sage.experiments.prepare_external_sdd_split import prepare


class PrepareExternalSDDSplitTest(unittest.TestCase):
    def test_scene_split_is_disjoint_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "annotations"
            for scene, video in (("bookstore", "video0"), ("gates", "video1"), ("nexus", "video0")):
                path = root / scene / video
                path.mkdir(parents=True)
                (path / "annotations.txt").write_text(
                    '0 0 0 1 1 0 0 0 0 "Pedestrian"\n',
                    encoding="utf-8",
                )
            output = Path(temporary) / "split"
            manifest = prepare(root.parent, output, train_scenes=2)
            self.assertEqual(len(manifest["train_scene_ids"]), 2)
            self.assertEqual(len(manifest["test_scene_ids"]), 1)
            self.assertTrue(
                set(manifest["train_scene_ids"]).isdisjoint(manifest["test_scene_ids"])
            )
            self.assertTrue((output / "train_trajnet.pkl").exists())
            self.assertTrue((output / "test_trajnet.pkl").exists())


if __name__ == "__main__":
    unittest.main()
