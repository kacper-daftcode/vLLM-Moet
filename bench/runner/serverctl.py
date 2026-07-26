"""Start/stop a vllm serve process for one recipe and watch it become healthy.

The server is its own session (process group) so worker subprocesses die with
it; teardown waits for the benched GPUs to actually release memory before the
next recipe starts."""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request

from common import run


class ServerFailed(Exception):
    def __init__(self, msg, log_tail):
        super().__init__(msg)
        self.log_tail = log_tail


def gpu_mem_used_mib(gpus):
    r = run(["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"])
    used = {}
    for ln in r.stdout.splitlines():
        if "," in ln:
            i, m = ln.split(",")
            used[int(i)] = int(m)
    return {g: used.get(g, 0) for g in gpus}


def tail(log_path, lines=80):
    try:
        with open(log_path, errors="ignore") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return ""


class Server:
    def __init__(self, cmd, env, log_path, port, gpus, container=None,
                 expected_model=None):
        self.cmd, self.env, self.log_path = cmd, env, log_path
        self.port, self.gpus = port, gpus
        self.container = container          # docker runtime: container name
        self.expected_model = expected_model
        self.proc = None
        self.load_time_s = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s):
        # Refuse an occupied port before truncating logs or spawning a second
        # server. Otherwise any existing /health endpoint could be mistaken
        # for the process this run intended to measure.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.port))
        except OSError as e:
            raise ServerFailed(
                f"benchmark port {self.port} is already occupied", str(e)
            ) from e
        finally:
            probe.close()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.cmd, env=self.env, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise ServerFailed(
                    f"server exited rc={self.proc.returncode} during load",
                    tail(self.log_path))
            try:
                with urllib.request.urlopen(self.base + "/health",
                                            timeout=5) as r:
                    if r.status == 200:
                        if self.expected_model is not None:
                            actual = self.model_id()
                            if actual != self.expected_model:
                                self.stop()
                                raise ServerFailed(
                                    "healthy endpoint serves unexpected "
                                    f"model {actual!r}, expected "
                                    f"{self.expected_model!r}",
                                    tail(self.log_path))
                        self.load_time_s = round(time.time() - t0, 1)
                        return
            except ServerFailed:
                raise
            except Exception:
                pass
            time.sleep(10)
        self.stop()
        raise ServerFailed(f"not healthy after {timeout_s}s",
                           tail(self.log_path))

    def model_id(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            return json.load(r)["data"][0]["id"]

    def _gpu_mem_used_mib(self):
        return sum(gpu_mem_used_mib(self.gpus).values())

    def stop(self, drain_timeout_s=180):
        if self.proc is None:
            return
        if self.container:
            # docker runtime: stop the container by name (SIGKILLing the
            # attached client would orphan it), then let --rm reap it.
            run(["docker", "stop", "-t", "60", self.container], timeout=90)
            t0 = time.time()
            while time.time() - t0 < 60 and self.proc.poll() is None:
                time.sleep(1)
        pgid = None
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            pass
        for sig, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 30),
                           (signal.SIGKILL, 30)):
            if self.proc.poll() is not None:
                break
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    self.proc.send_signal(sig)
            except ProcessLookupError:
                break
            t0 = time.time()
            while time.time() - t0 < grace and self.proc.poll() is None:
                time.sleep(1)
        # wait for VRAM to come back before the next recipe boots
        t0 = time.time()
        while time.time() - t0 < drain_timeout_s:
            if self._gpu_mem_used_mib() < 2048:
                break
            time.sleep(5)
        self.proc = None
