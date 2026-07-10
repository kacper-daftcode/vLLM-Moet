"""Environment fingerprint recorded into every result: what exactly ran.

Cross-release comparisons are only honest when the pins are visible — the
project's history is full of behaviour that tracked DeepGEMM/flashinfer/driver
versions, not code."""

import hashlib
import os
import subprocess

from common import REPO_DIR, run


def _git(repo, *args):
    r = run(["git", "-C", repo, *args])
    return r.stdout.strip() if r.returncode == 0 else None


def _sha256_file(path, short=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:short]


def _cubins_fingerprint(cubin_dir):
    if not cubin_dir or not os.path.isdir(cubin_dir):
        return None
    h = hashlib.sha256()
    for fn in sorted(os.listdir(cubin_dir)):
        p = os.path.join(cubin_dir, fn)
        if os.path.isfile(p):
            h.update(fn.encode())
            h.update(_sha256_file(p, 64).encode())
    return h.hexdigest()[:12]


def _pkg_versions(venv, names):
    py = os.path.join(venv, "bin", "python")
    code = ("from importlib.metadata import version\n"
            "import json,sys\n"
            "out={}\n"
            f"for n in {names!r}:\n"
            "    try: out[n]=version(n)\n"
            "    except Exception: out[n]=None\n"
            "print(json.dumps(out))")
    r = run([py, "-c", code], timeout=60)
    if r.returncode != 0:
        return {}
    import json
    return json.loads(r.stdout.strip())


def _vllm_tree(venv):
    """Locate the (editable) vllm checkout without importing vllm."""
    py = os.path.join(venv, "bin", "python")
    code = ("import importlib.util as u; s=u.find_spec('vllm'); "
            "print(s.origin or '')")
    r = run([py, "-c", code], timeout=60)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    d = os.path.dirname(r.stdout.strip())          # .../vllm/__init__.py
    repo = os.path.dirname(d)
    if not os.path.isdir(os.path.join(repo, ".git")):
        return None
    return {
        "path": repo,
        "sha": _git(repo, "rev-parse", "--short=12", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
    }


def collect(box, serve_env):
    smi = run(["nvidia-smi",
               "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader"])
    gpus = [ln.strip() for ln in smi.stdout.splitlines() if ln.strip()]
    patch = os.path.join(REPO_DIR, "patch", "vllm-moet-v0.24.0.patch")
    info = {
        "box": box["id"],
        "gpus": gpus,
        "driver": gpus[0].rsplit(",", 1)[-1].strip() if gpus else None,
        "moet_sha": _git(REPO_DIR, "rev-parse", "--short=12", "HEAD"),
        "moet_dirty": bool(_git(REPO_DIR, "status", "--porcelain")),
        "patch_sha256": _sha256_file(patch) if os.path.exists(patch) else None,
        "cubins": _cubins_fingerprint(serve_env.get("VLLM_MOE_W2_CUBIT_DIR")),
        "runtime": box.get("runtime", "venv"),
    }
    if box.get("runtime", "venv") == "venv":
        info["packages"] = _pkg_versions(
            box["venv"], ["vllm", "torch", "flashinfer-python", "triton"])
        info["vllm_tree"] = _vllm_tree(box["venv"])
    else:
        info["docker_image"] = box.get("docker_image")
        r = run(["docker", "image", "inspect",
                 "--format", "{{.Id}}", box.get("docker_image", "")],
                timeout=30)
        info["docker_image_id"] = (r.stdout.strip()[:19]
                                   if r.returncode == 0 else None)
    return info
