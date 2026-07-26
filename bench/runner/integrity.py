"""Canonical input bindings for benchmark results and release gates."""

import hashlib
import json
import os


CANONICALIZATION = "json-sorted-v1"


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def effective_suite_document(suite_name: str, probes: list[dict]) -> dict:
    return {
        "schema": 1,
        "suite": suite_name,
        "probes": probes,
    }


def result_bindings(recipe: dict, suite_name: str, probes: list[dict],
                    model_registry: dict, *, release: str | None = None,
                    box: dict | None = None) -> dict:
    names = [
        recipe["model"],
        *recipe.get("requires", {}).get("extra_models", []),
    ]
    models = {}
    for name in names:
        spec = model_registry[name]
        models[name] = {
            "repo": spec["hf_repo"],
            "revision": spec["revision"],
        }
    suite_document = effective_suite_document(suite_name, probes)
    bindings = {
        "canonicalization": CANONICALIZATION,
        "recipe_sha256": canonical_sha256(recipe),
        "suite_sha256": canonical_sha256(suite_document),
        "recipe_snapshot": recipe,
        "suite_snapshot": suite_document,
        "models": models,
    }
    if release is not None:
        bindings["release"] = release
    if box is not None:
        bindings["box"] = box["id"]
        bindings["box_sha256"] = canonical_sha256(box)
    return bindings


def self_binding_errors(inputs: dict) -> list[str]:
    errors = []
    if inputs.get("canonicalization") != CANONICALIZATION:
        errors.append("unknown canonicalization")
    recipe = inputs.get("recipe_snapshot")
    suite = inputs.get("suite_snapshot")
    if recipe is None or canonical_sha256(recipe) != inputs.get("recipe_sha256"):
        errors.append("recipe snapshot/hash mismatch")
    if suite is None or canonical_sha256(suite) != inputs.get("suite_sha256"):
        errors.append("suite snapshot/hash mismatch")
    return errors


def strict_gate_errors(result: dict, expected_suite: str,
                       expected_inputs: dict,
                       expected_runtime: dict | None = None) -> list[str]:
    checks = [
        (result.get("schema") == 2, "schema is not 2"),
        (result.get("complete") is True, "result is incomplete"),
        (result.get("provenance") == "live", "result is not live"),
        (result.get("status") == "ok", "result status is not ok"),
        (result.get("suite") == expected_suite, "suite mismatch"),
        (result.get("inputs") == expected_inputs, "input binding drift"),
    ]
    fingerprint = result.get("env_fingerprint") or {}
    vllm_tree = fingerprint.get("vllm_tree") or {}
    runtime = fingerprint.get("runtime")
    runtime_identity_ok = (
        bool(fingerprint.get("docker_image_id"))
        if runtime == "docker"
        else bool(vllm_tree.get("sha"))
    )
    checks.extend([
        (not fingerprint.get("moet_dirty"), "publication tree is dirty"),
        (not vllm_tree.get("dirty", False), "vLLM tree is dirty"),
        (bool(fingerprint.get("patch_sha256")), "patch hash missing"),
        (runtime_identity_ok, "runtime image/source identity missing"),
    ])
    errors = [message for ok, message in checks if not ok]
    if not expected_runtime:
        errors.append("tracked runtime manifest missing for release")
        return errors
    required_manifest_keys = {
        "patch_sha256", "cubins", "moet_sha", "model_manifests",
        "image_id" if runtime == "docker" else "vllm_sha",
    }
    missing_keys = required_manifest_keys - set(expected_runtime)
    if missing_keys:
        errors.append(
            f"runtime manifest missing keys {sorted(missing_keys)}")
    actual_runtime = {
        "patch_sha256": fingerprint.get("patch_sha256"),
        "cubins": fingerprint.get("cubins"),
        "moet_sha": fingerprint.get("moet_sha"),
        "vllm_sha": vllm_tree.get("sha"),
        "image_id": fingerprint.get("docker_image_id"),
        "model_manifests": (
            (result.get("inputs") or {}).get("model_manifests")),
    }
    for key, expected in expected_runtime.items():
        if actual_runtime.get(key) != expected:
            errors.append(f"runtime {key} mismatch")
    return errors


def artifact_path(result_path: str, artifact_name: str) -> str:
    """Resolve an artifact adjacent to a box result directory, contained."""
    root = os.path.realpath(os.path.join(os.path.dirname(result_path),
                                         "artifacts"))
    path = os.path.realpath(os.path.join(root, artifact_name))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(f"artifact escapes result directory: {artifact_name}")
    return path
