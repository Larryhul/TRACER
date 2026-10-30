"""Read trusted historical pickles without registering global module aliases."""
from __future__ import annotations

import io

import dill


class LegacyUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        prefix, _, suffix = module.partition(".")
        if prefix == "sage":
            module = "tracer" + ("." + suffix if suffix else "")
        elif prefix in {"models", "dataset", "environment", "evaluation", "utils"}:
            module = "tracer.legacy." + module
        return super().find_class(module, name)


def load_legacy_pickle(stream, **kwargs):
    """Load an archived dataset with its original class paths remapped locally."""
    return LegacyUnpickler(stream, **kwargs).load()


def loads_legacy_pickle(data, **kwargs):
    return load_legacy_pickle(io.BytesIO(data), **kwargs)
