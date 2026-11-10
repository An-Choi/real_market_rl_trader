from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.merge_research_shards import merge
from experiments.run_ablation import write_json


def shard(root, condition):
    output = root / condition
    output.mkdir(parents=True)
    name, code = "agent/src/example.py", b"value = 1\n"
    source_hash = hashlib.sha256(name.encode() + code).hexdigest()
    archive = output / "source_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        entry = tarfile.TarInfo(name)
        entry.size = len(code)
        handle.addfile(entry, io.BytesIO(code))
    manifest = {"source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "code": {"python_source_sha256": source_hash}, "config": {"gamma": .99},
        "folds": [{"index": 1}], "data_fingerprints": {"EXAMPLE": "dataset"},
        "packages": {"torch": "example"}, "minimum_coverage": .95, "selection": "validation",
        "arguments": {"output": str(output), "conditions": condition, "seeds": "42", "bar_budget": 100}}
    write_json(output / "study_manifest.json", manifest)
    write_json(output / "completion.json", {"completed": True, "runs": 1, "test_evaluated": False})
    write_json(output / "results.json", [{"fold": 1, "condition": condition, "seed": 42, "roundtrip_verified": True}])
    bundle = output / "fold_01" / condition / "seed_42"
    bundle.mkdir(parents=True)
    for filename in ("research_model.zip", "normalization.json"):
        (bundle / filename).write_bytes(b"example")
    write_json(bundle / "research_manifest.json", {"condition": condition, "seed": 42, "fold": {"index": 1},
        "code": manifest["code"], "sha256": {filename: hashlib.sha256(b"example").hexdigest()
            for filename in ("research_model.zip", "normalization.json")}})
    return output


def test_merge_preserves_all_trials_and_archive_provenance(tmp_path):
    source = tmp_path / "shards"
    shard(source, "R")
    shard(source, "H")
    output = tmp_path / "merged"
    merge(source, output, ["R", "H"])
    assert json.loads((output / "completion.json").read_text())["runs"] == 2
    assert len(list(output.rglob("research_model.zip"))) == 2
    assert json.loads((output / "study_manifest.json").read_text())["arguments"]["conditions"] == "R,H"


def test_merge_rejects_changed_model_before_creating_output(tmp_path):
    source = tmp_path / "shards"
    directory = shard(source, "R")
    next(directory.rglob("research_model.zip")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        merge(source, tmp_path / "merged", ["R"])
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_missing_condition_and_mismatched_data(tmp_path):
    source = tmp_path / "shards"
    shard(source, "R")
    with pytest.raises(ValueError, match="missing or extra"):
        merge(source, tmp_path / "merged", ["R", "H"])
    directory = shard(source, "H")
    manifest = json.loads((directory / "study_manifest.json").read_text())
    manifest["data_fingerprints"]["EXAMPLE"] = "other-dataset"
    write_json(directory / "study_manifest.json", manifest)
    with pytest.raises(ValueError, match="data_fingerprints"):
        merge(source, tmp_path / "merged", ["R", "H"])
