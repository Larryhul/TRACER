import errno
import importlib
import sys

from tracer.compat import loads_legacy_pickle
from tracer.filesystem import link_or_copy


def test_archived_class_paths_load_without_global_aliases():
    before = sys.modules.get("sage")
    restored = loads_legacy_pickle(b"csage.models.sage_model\nSAGEConfig\n.")
    expected = importlib.import_module("tracer.models.sage_model").SAGEConfig
    assert restored is expected
    assert sys.modules.get("sage") is before


def test_file_export_works_without_symlink_privileges(tmp_path, monkeypatch):
    source = tmp_path / "original.bin"
    target = tmp_path / "snapshot.bin"
    source.write_bytes(b"immutable planner input")

    def denied(*args, **kwargs):
        raise OSError(errno.EPERM, "symlinks unavailable")

    monkeypatch.setattr("tracer.filesystem.os.symlink", denied)
    link_or_copy(source, target)
    assert source.read_bytes() == target.read_bytes()
