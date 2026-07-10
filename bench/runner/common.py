"""Shared plumbing: YAML config loading, placeholder resolution, serve-command
assembly, result paths. Only dependency beyond the stdlib is PyYAML."""

import json
import os
import re
import shlex
import subprocess
import time

import yaml

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BENCH_DIR)

DEFAULT_PORT = 8123
DEFAULT_LOAD_TIMEOUT_S = 2400


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_matrix():
    return _load_yaml(os.path.join(BENCH_DIR, "matrix.yaml"))


def load_box(path):
    box = _load_yaml(path)
    box.setdefault("env", {})
    box.setdefault("models", {})
    box.setdefault("planes_cache", {})
    box.setdefault("log_dir", "/tmp/vllm-moet-bench")
    return box


def load_suite(name):
    return _load_yaml(os.path.join(BENCH_DIR, "suites", f"{name}.yaml"))


def recipe_path(recipe_id):
    return os.path.join(BENCH_DIR, "recipes", recipe_id + ".yaml")


def load_recipe(recipe_id):
    r = _load_yaml(recipe_path(recipe_id))
    if r.get("id") != recipe_id:
        raise ValueError(f"{recipe_id}: file id field says {r.get('id')!r}")
    r.setdefault("env", {})
    r.setdefault("serve_args", [])
    r.setdefault("suite_params", {})
    r.setdefault("requires", {})
    r.setdefault("warmup", {})
    return r


def list_recipe_ids():
    root = os.path.join(BENCH_DIR, "recipes")
    out = []
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith(".yaml"):
                out.append(f"{model}/{fn[:-5]}")
    return out


def compat(recipe, box):
    """None if the box can run the recipe, else a human-readable skip reason."""
    req = recipe["requires"]
    g = box.get("gpus", {})
    if g.get("count", 0) < req.get("gpus", 1):
        return f"needs {req.get('gpus')} GPUs, box has {g.get('count', 0)}"
    if g.get("vram_gb", 0) < req.get("vram_gb", 0):
        return f"needs {req.get('vram_gb')} GB VRAM/GPU, box has {g.get('vram_gb')}"
    if box.get("host_ram_gb", 0) < req.get("host_ram_gb", 0):
        return f"needs {req.get('host_ram_gb')} GB host RAM"
    for m in [recipe["model"], *req.get("extra_models", [])]:
        if m not in box["models"]:
            return f"model {m} not on this box"
        if not os.path.exists(box["models"][m]):
            return f"model path missing: {box['models'][m]}"
    return None


_PLACEHOLDER = re.compile(r"\{(planes_cache|model(?::[\w.\-]+)?)\}")


def _resolve_str(s, recipe, box):
    def sub(m):
        key = m.group(1)
        if key == "planes_cache":
            v = box["planes_cache"].get(recipe["model"])
            if v is None:
                raise KeyError("planes_cache")
            return v
        name = key.split(":", 1)[1] if ":" in key else recipe["model"]
        if name not in box["models"]:
            raise KeyError(f"model {name} not in box")
        return box["models"][name]

    return _PLACEHOLDER.sub(sub, s)


def build_env(recipe, box, gpus):
    """Process env for the server: inherited + box + recipe (resolved)."""
    env = dict(os.environ)
    merged = {**box["env"], **recipe["env"]}
    pp = merged.pop("PATH_PREPEND", None)
    lp = merged.pop("LD_LIBRARY_PATH_PREPEND", None)
    for k, v in merged.items():
        try:
            env[k] = _resolve_str(str(v), recipe, box)
        except KeyError:
            # optional resource missing on this box (e.g. no planes cache):
            # drop the knob rather than fail — the recipe works without it.
            env.pop(k, None)
    if pp:
        env["PATH"] = pp + ":" + env.get("PATH", "")
    if lp:
        env["LD_LIBRARY_PATH"] = lp + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


def declared_env(recipe, box):
    """The knobs (box+recipe) — recorded into the result JSON. In docker
    mode recipe values stay unresolved (the in-container launcher resolves
    them; its applied-env lines are in the server log)."""
    out = {}
    for k, v in {**box["env"], **recipe["env"]}.items():
        if k in ("PATH_PREPEND", "LD_LIBRARY_PATH_PREPEND"):
            continue
        if box.get("runtime", "venv") == "docker":
            out[k] = str(v)
            continue
        try:
            out[k] = _resolve_str(str(v), recipe, box)
        except KeyError:
            out[k] = None  # dropped on this box
    return out


def build_serve_cmd(recipe, box, port, gpus=None, recipe_file=None,
                    extra_binds=()):
    """The server argv for this recipe on this box.

    venv runtime: the vllm serve command itself.
    docker runtime: a `docker run` of the RECIPES image (the exact artifact
    customers get). Local weights mount over /models (the launcher's
    download step no-ops) and the checkout's recipe/models.yaml mount over
    the baked-in copies, so recipe edits and sweeps need no image rebuild.
    `recipe_file` overrides the recipe YAML the container sees (sweeps);
    `extra_binds` are extra docker -v specs (runtime knob files)."""
    if box.get("runtime", "venv") == "docker":
        return _docker_serve_cmd(recipe, box, port, gpus or [],
                                 recipe_file, extra_binds)
    vllm = os.path.join(box["venv"], "bin", "vllm")
    cmd = [vllm, "serve", box["models"][recipe["model"]],
           "--served-model-name", recipe["served_name"],
           "--port", str(port)]
    for arg in recipe["serve_args"]:
        arg = _resolve_str(arg, recipe, box)
        parts = arg.split(" ", 1)
        cmd.extend(parts if len(parts) == 1 else [parts[0], parts[1]])
    return cmd


# Venv-only knobs that must never leak into the image (it ships its own
# toolchain); everything else in box env is a host quirk the container needs.
_VENV_ONLY_ENV = ("PATH_PREPEND", "LD_LIBRARY_PATH_PREPEND",
                  "CUDA_HOME", "FLASHINFER_CUDA_ARCH_LIST",
                  "VLLM_MOE_W2_CUBIT_DIR")


def _docker_serve_cmd(recipe, box, port, gpus, recipe_file, extra_binds):
    image = box.get("docker_image")
    if not image:
        raise ValueError(f"box {box['id']}: runtime docker needs docker_image")
    rid = recipe["id"]
    dev = ",".join(str(g) for g in gpus)
    cmd = ["docker", "run", "--rm", "--name", container_name(port),
           "--gpus", f"device={dev}",
           "--network", "host", "--ipc", "host",
           "--shm-size", box.get("shm_size", "64g"),
           "-e", f"PORT={port}",
           "-v", f"{recipe_file or recipe_path(rid)}"
                 f":/opt/vllm-moet/recipes/{rid}.yaml:ro",
           "-v", f"{os.path.join(BENCH_DIR, 'models.yaml')}"
                 ":/opt/vllm-moet/models.yaml:ro"]
    names = [recipe["model"], *recipe["requires"].get("extra_models", [])]
    for name in names:
        cmd += ["-v", f"{box['models'][name]}:/models/{name}:ro"]
    pc = box["planes_cache"].get(recipe["model"])
    if pc and os.path.isdir(pc):
        cmd += ["-v", f"{pc}:/planes", "-e", "PLANES_CACHE=/planes"]
    for bind in extra_binds:
        cmd += ["-v", bind]
    for k, v in box["env"].items():
        if k not in _VENV_ONLY_ENV:
            cmd += ["-e", f"{k}={v}"]
    cmd += [image, rid]
    return cmd


def container_name(port):
    return f"vllm-moet-bench-{port}"


def serve_cmd_display(cmd, env_knobs):
    knobs = " ".join(f"{k}={shlex.quote(str(v))}"
                     for k, v in sorted(env_knobs.items()) if v is not None)
    # group "--flag value" / "-e K=V" pairs back onto one display line
    lines, i = [], 0
    while i < len(cmd):
        tok = shlex.quote(cmd[i])
        if cmd[i].startswith("-") and i + 1 < len(cmd) \
                and not cmd[i + 1].startswith("-"):
            tok += " " + shlex.quote(cmd[i + 1])
            i += 1
        lines.append(tok)
        i += 1
    head = " ".join(lines[:2])                    # vllm serve <model>
    if len(lines) > 2 and not lines[2].startswith("'--"):
        head += " " + lines[2]
        rest = lines[3:]
    else:
        rest = lines[2:]
    return (knobs + " \\\n" if knobs else "") + " \\\n  ".join([head] + rest)


def result_path(release, box_id, recipe_id):
    d = os.path.join(BENCH_DIR, "results", release, box_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, recipe_id.replace("/", "__") + ".json")


def write_result(path, result):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def iter_results(release):
    root = os.path.join(BENCH_DIR, "results", release)
    if not os.path.isdir(root):
        return
    for box_id in sorted(os.listdir(root)):
        bdir = os.path.join(root, box_id)
        if not os.path.isdir(bdir):
            continue
        for fn in sorted(os.listdir(bdir)):
            if fn.endswith(".json"):
                with open(os.path.join(bdir, fn)) as f:
                    yield box_id, json.load(f)


def merge_suite(suite, recipe):
    """Suite probe list with the recipe's per-probe overrides applied.

    Only suites with allow_recipe_overrides (the release suite) take recipe
    suite_params — `quick` stays deliberately small for smoke runs."""
    params = recipe.get("suite_params", {}) \
        if suite.get("allow_recipe_overrides") else {}
    out = []
    for probe in suite["probes"]:
        p = dict(probe)
        p.update(params.get(p["kind"], {}))
        out.append(p)
    return out


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
