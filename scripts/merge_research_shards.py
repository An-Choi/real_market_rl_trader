"""Validate and merge independent condition runs without selecting or retraining."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT, ROOT / "agent/src", ROOT / "env/src"):
    sys.path.insert(0, str(source))

from experiments.run_ablation import write_json


def verify_archive(directory, manifest):
    path = directory / "source_snapshot.tar.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["source_archive_sha256"]:
        raise ValueError("source archive checksum mismatch")
    digest = hashlib.sha256()
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for area in ("agent/src", "env/src", "evaluation", "experiments"):
            for name in sorted(name for name in members if name.startswith(area + "/") and name.endswith(".py")):
                digest.update(name.encode())
                digest.update(archive.extractfile(members[name]).read())
    if digest.hexdigest() != manifest["code"]["python_source_sha256"]:
        raise ValueError("archived code does not match training fingerprint")


def merge(shards, output, conditions):
    shards, output = Path(shards).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("merged output exists; do not overwrite")
    directories = sorted(path for path in shards.iterdir() if path.is_dir())
    studies, results, sources = [], [], {}
    for directory in directories:
        completion = json.loads((directory / "completion.json").read_text())
        manifest = json.loads((directory / "study_manifest.json").read_text())
        if not completion["completed"] or completion["test_evaluated"]:
            raise ValueError("all shards must be complete validation-only experiments")
        verify_archive(directory, manifest)
        shard_results = json.loads((directory / "results.json").read_text())
        if len(shard_results) != completion["runs"]:
            raise ValueError("shard run count mismatch")
        for result in shard_results:
            key = (result["fold"], result["condition"], result["seed"])
            if key in sources or result["condition"] not in conditions or not result["roundtrip_verified"]:
                raise ValueError("duplicate, unknown or unverified trial")
            fold, condition, seed = key
            bundle_dir = directory / f"fold_{fold:02d}" / condition / f"seed_{seed}"
            bundle = json.loads((bundle_dir / "research_manifest.json").read_text())
            if (bundle["condition"], bundle["seed"], bundle["fold"]["index"]) != (condition, seed, fold):
                raise ValueError("artifact does not match its trial")
            if bundle["code"]["python_source_sha256"] != manifest["code"]["python_source_sha256"]:
                raise ValueError("artifact code differs from study")
            for name in ("research_model.zip", "normalization.json"):
                if hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest() != bundle["sha256"][name]:
                    raise ValueError("artifact checksum mismatch before merge")
            sources[key] = directory
        studies.append((directory, manifest))
        results.extend(shard_results)
    if not studies:
        raise ValueError("no completed shards")
    first_directory, first = studies[0]
    comparable_args = lambda args: {key: value for key, value in args.items() if key not in ("output", "conditions", "seeds")}
    for _, manifest in studies[1:]:
        for field in ("config", "folds", "data_fingerprints", "packages", "minimum_coverage", "selection", "code"):
            if manifest[field] != first[field]:
                raise ValueError(f"incompatible shard {field}")
        if comparable_args(manifest["arguments"]) != comparable_args(first["arguments"]):
            raise ValueError("incompatible experiment budgets or arguments")
    seeds = sorted({int(seed) for _,manifest in studies for seed in manifest["arguments"]["seeds"].split(",")})
    expected = {(fold["index"], condition, seed) for fold in first["folds"]
                for condition in conditions for seed in seeds}
    if set(sources) != expected:
        raise ValueError("missing or extra trial in merged experiment")
    for path in first_directory.glob("fold_*_baselines.json"):
        payload = json.loads(path.read_text())
        for directory, _ in studies[1:]:
            if json.loads((directory / path.name).read_text()) != payload:
                raise ValueError("baseline mismatch between shards")
    output.mkdir(parents=True)
    for key, directory in sources.items():
        fold, condition, seed = key
        relative = Path(f"fold_{fold:02d}") / condition / f"seed_{seed}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(directory / relative, destination)
    for path in first_directory.glob("fold_*_baselines.json"):
        shutil.copy2(path, output / path.name)
    shutil.copy2(first_directory / "source_snapshot.tar.gz", output / "source_snapshot.tar.gz")
    merger = Path(__file__)
    shutil.copy2(merger, output / "merger_source.py")
    combined = copy.deepcopy(first)
    combined["arguments"].update(output=str(output), conditions=",".join(conditions), seeds=",".join(map(str,seeds)))
    combined["execution"] = {"type": "parallel independent condition shards; no retraining or selection",
        "shards": [{"directory": str(directory), "manifest_sha256": hashlib.sha256(
            (directory / "study_manifest.json").read_bytes()).hexdigest()} for directory, _ in studies],
        "merger_sha256": hashlib.sha256(merger.read_bytes()).hexdigest()}
    results.sort(key=lambda run: (run["fold"], conditions.index(run["condition"]), run["seed"]))
    write_json(output / "study_manifest.json", combined)
    write_json(output / "results.json", results)
    write_json(output / "completion.json", {"completed": True, "runs": len(results),
        "research_only": True, "test_evaluated": False, "execution": "parallel independent shards"})
    print(json.dumps({"merged_runs": len(results), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default="Q,R,E,F,G,H")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    if not conditions or len(set(conditions)) != len(conditions) or any(not c.isalpha() or not c.isupper() for c in conditions):
        raise SystemExit("conditions must be unique uppercase names")
    merge(args.shards, args.output, conditions)
