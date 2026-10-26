import json
from pathlib import Path

from sage.belief.correlated_emission import CorrelatedEmission
from sage.experiments.formal_manifest import sha256_file


def load_emission_noise(path, emission_checkpoint, raw, evaluation_data_seed):
    from sage.experiments.calibrate_component_memory import component_protocol_id

    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("schema") != "component-correlated-emission-v1" or record.get("status") != "COMPLETED" or record.get("changed_inputs"):
        raise ValueError("completed unchanged correlated emission artifact required")
    if record["emission_checkpoint_sha256"] != sha256_file(Path(emission_checkpoint)):
        raise ValueError("correlated emission checkpoint mismatch")
    if record["component_protocol_id"] != component_protocol_id(raw, True):
        raise ValueError("correlated emission execution protocol mismatch")
    if evaluation_data_seed in record["fitting_data_seeds"]:
        raise ValueError("correlated emission fitting and evaluation scenes must be separate")
    return CorrelatedEmission(**record["parameters"])
