#!/usr/bin/env python3
"""Recipe launcher — the entrypoint of the vllm-moet-recipes image.

Given a recipe id (see --list), it: checks the GPUs, downloads the model(s)
from HuggingFace into $MODELS_DIR if missing (resumable; big checkpoints are
announced with their size first), applies the recipe's env knobs, and execs
`vllm serve` with the recipe's exact arguments. The recipe IS the supported,
benchmarked configuration — bench/results/ in the repo holds the numbers this
exact config produced.

  docker run --rm --gpus all --network host --ipc host --shm-size 64g \
    -v /srv/models:/models -e HF_TOKEN=... \
    vllm-moet-recipes:v024  kimi-k2.7-code-nvfp4/pro6000x4-tp4-256k

Conventions:
  - $MODELS_DIR (default /models): weights live at $MODELS_DIR/<ModelName>.
  - $PORT (default 8000): the API port (--network host recommended).
  - Any env you set on the container OVERRIDES the recipe's knob of the same
    name (e.g. -e VLLM_MOE_W2_DELTA_GB=0), and host quirks like
    -e NCCL_P2P_DISABLE=1 pass straight through.
  - $PLANES_CACHE: mount a directory (-v /srv/planes:/planes -e
    PLANES_CACHE=/planes) to cache the built 2-bit planes across restarts
    (big: can approach the checkpoint size; cuts reload time ~40%). Without
    it the planes-cache knob is dropped and planes rebuild at load.
  - Arguments after `--` are appended to vllm serve (e.g. -- --api-key ...).
  - $SKIP_GPU_CHECK=1 skips the GPU count/VRAM preflight.

Runs anywhere python3+pyyaml exist; --print shows the exec without running
(that is also how the repo's CI smoke-tests it).
"""

import argparse
import os
import re
import shlex
import subprocess
import sys

import yaml

HOME = os.environ.get("VLLM_MOET_HOME", "/opt/vllm-moet")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")


def die(msg):
    print(f"serve_recipe: ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def list_recipes():
    root = os.path.join(HOME, "recipes")
    out = []
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if os.path.isdir(mdir):
            out += [f"{model}/{fn[:-5]}" for fn in sorted(os.listdir(mdir))
                    if fn.endswith(".yaml")]
    return out


def gpu_inventory():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    gpus = []
    for ln in r.stdout.splitlines():
        if "," in ln:
            name, mem = ln.rsplit(",", 1)
            gpus.append((name.strip(), int(mem) // 1024))
    return gpus


def check_gpus(recipe):
    if os.environ.get("SKIP_GPU_CHECK") == "1":
        return
    gpus = gpu_inventory()
    if gpus is None:
        print("serve_recipe: nvidia-smi unavailable — skipping GPU preflight")
        return
    req = recipe.get("requires") or {}
    need_n = req.get("gpus", 1)
    need_vram = req.get("vram_gb", 0)
    if len(gpus) < need_n:
        die(f"recipe needs {need_n} GPU(s), container sees {len(gpus)} "
            f"(--gpus all? SKIP_GPU_CHECK=1 to override)")
    small = [f"{n} ({v} GiB)" for n, v in gpus[:need_n] if v + 1 < need_vram]
    if small:
        die(f"recipe needs {need_vram} GiB VRAM/GPU; too small: "
            f"{', '.join(small)} (SKIP_GPU_CHECK=1 to override)")


def ensure_model(name, registry, do_print):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(os.path.join(path, "config.json")) \
            and not os.environ.get("FORCE_DOWNLOAD"):
        print(f"serve_recipe: {name}: present at {path}")
        return path
    if name not in registry:
        die(f"model {name} not in models.yaml registry")
    repo = registry[name]["hf_repo"]
    size = registry[name].get("approx_gb", "?")
    print(f"serve_recipe: {name}: downloading {repo} (~{size} GB) "
          f"-> {path}", flush=True)
    if do_print:
        return path
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, local_dir=path)
    except ImportError:
        die("huggingface_hub not installed")
    except Exception as e:  # noqa: BLE001 — auth/network/disk: user-facing
        if "hf_transfer" in str(e):
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, local_dir=path)
        else:
            die(f"download of {repo} failed: {e}\n"
                "  (gated repo? pass -e HF_TOKEN=...; disk full? the "
                f"volume behind {MODELS_DIR} needs ~{size} GB)")
    return path


_PLACEHOLDER = re.compile(r"\{(planes_cache|model:[\w.\-]+)\}")


def resolve(s):
    """Container-side placeholder resolution. Returns None if the string
    needs a resource this container does not have (knob is then dropped)."""
    dropped = False

    def sub(m):
        nonlocal dropped
        key = m.group(1)
        if key == "planes_cache":
            v = os.environ.get("PLANES_CACHE")
            if not v:
                dropped = True
                return ""
            return v
        return os.path.join(MODELS_DIR, key.split(":", 1)[1])

    out = _PLACEHOLDER.sub(sub, s)
    return None if dropped else out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", nargs="?", default=os.environ.get("RECIPE"),
                    help="recipe id, e.g. glm-5.2-nvfp4/pro6000x4-tp4-mtp "
                         "(or RECIPE env)")
    ap.add_argument("--list", action="store_true", help="list recipes and exit")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="show the resolved env + command without running")
    ap.add_argument("extra", nargs="*",
                    help="extra vllm serve args after --")
    args = ap.parse_args()

    if args.list or not args.recipe:
        print("available recipes:")
        for rid in list_recipes():
            r = load_yaml(os.path.join(HOME, "recipes", rid + ".yaml"))
            print(f"  {rid:55s} {r.get('summary', '')}")
        if not args.list:
            die("no recipe given (positional arg or RECIPE env)")
        return

    rpath = os.path.join(HOME, "recipes", args.recipe + ".yaml")
    if not os.path.exists(rpath):
        die(f"unknown recipe {args.recipe!r} (see --list)")
    recipe = load_yaml(rpath)
    registry = load_yaml(os.path.join(HOME, "models.yaml"))["models"]

    check_gpus(recipe)

    model_path = ensure_model(recipe["model"], registry, args.do_print)
    for extra in (recipe.get("requires") or {}).get("extra_models", []):
        ensure_model(extra, registry, args.do_print)

    # recipe knobs: container env (docker -e) wins over the recipe value
    applied, overridden, dropped = {}, {}, []
    for k, v in (recipe.get("env") or {}).items():
        if k in os.environ:
            overridden[k] = os.environ[k]
            continue
        rv = resolve(str(v))
        if rv is None:
            dropped.append(k)
            continue
        os.environ[k] = rv
        applied[k] = rv

    port = os.environ.get("PORT", "8000")
    argv = ["vllm", "serve", model_path,
            "--served-model-name", recipe["served_name"], "--port", port]
    for a in recipe.get("serve_args") or []:
        a = resolve(a)
        if a is None:
            continue
        flag, _, rest = a.partition(" ")
        argv += [flag, rest] if rest else [flag]
    argv += args.extra

    print(f"serve_recipe: {args.recipe} — {recipe.get('summary', '')}")
    for k, v in applied.items():
        print(f"  env {k}={v}")
    for k, v in overridden.items():
        print(f"  env {k}={v}   (container override)")
    for k in dropped:
        print(f"  env {k} dropped (no PLANES_CACHE mounted)" if "PLANES" in k
              else f"  env {k} dropped (unresolved)")
    print("  exec: " + " ".join(shlex.quote(x) for x in argv), flush=True)

    if args.do_print:
        return
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
