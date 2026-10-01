from __future__ import annotations

import json
import os
import base64
import shutil
import socket
import subprocess
import time
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

try:
    import paramiko  # type: ignore
except Exception:  # pragma: no cover - exercised only in offline WSL envs.
    paramiko = None

from .backends import ModelResponse, RemoteHTTPTextBackend
from .config import AppConfig


@dataclass(frozen=True)
class RemoteConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    python_path: str
    remote_root: str


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    gpu_index: int
    port: int
    model_path: str


@dataclass(frozen=True)
class RemoteTunnelHandle:
    pid: int
    local_port: int
    mode: str


@dataclass(frozen=True)
class SelectorRemoteBackends:
    generator_backend: RemoteHTTPTextBackend | RemoteLoopbackTextBackend | BridgeHTTPTextBackend | None
    judge_backends: list[RemoteHTTPTextBackend | RemoteLoopbackTextBackend | BridgeHTTPTextBackend]
    runtime_metrics: dict[str, Any]


def use_serial_single_gpu_mode() -> bool:
    return os.environ.get("POC_SERIAL_SINGLE_GPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def use_dual_hotswap_mode() -> bool:
    return os.environ.get("POC_DUAL_HOTSWAP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _connection_from_prefix(
    config: AppConfig,
    *,
    prefix: str,
    default_host: str,
    default_user: str,
    default_python: str,
    default_root: str,
) -> RemoteConnectionConfig:
    password = os.environ.get(f"{prefix}_PASSWORD")
    if not password:
        raise RuntimeError(f"{prefix}_PASSWORD is not set")
    host = os.environ.get(f"{prefix}_HOST", default_host).strip()
    user = os.environ.get(f"{prefix}_USER", default_user).strip()
    if not host:
        raise RuntimeError(f"{prefix}_HOST is not set")
    if not user:
        raise RuntimeError(f"{prefix}_USER is not set")
    return RemoteConnectionConfig(
        host=host,
        port=int(os.environ.get(f"{prefix}_PORT", "22")),
        user=user,
        password=password,
        python_path=os.environ.get(f"{prefix}_PYTHON", default_python),
        remote_root=os.environ.get(f"{prefix}_ROOT", default_root),
    )


def default_remote_connection(config: AppConfig) -> RemoteConnectionConfig:
    return _connection_from_prefix(
        config,
        prefix="POC_REMOTE",
        default_host=config.runtime.remote_host,
        default_user=config.runtime.remote_user,
        default_python=config.runtime.remote_python,
        default_root=config.runtime.remote_root,
    )


def generator_model_path(config: AppConfig) -> str:
    return os.environ.get(
        "POC_GENERATOR_MODEL_PATH",
        config.runtime.generator_model_path,
    )


def _gpu_index_from_env(env_name: str, default: int) -> int:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return default
    return int(value)


def judge_b_remote_connection(config: AppConfig) -> RemoteConnectionConfig | None:
    password = os.environ.get("POC_JUDGE_B_REMOTE_PASSWORD")
    host = os.environ.get("POC_JUDGE_B_REMOTE_HOST")
    if not password or not host:
        return None
    return _connection_from_prefix(
        config,
        prefix="POC_JUDGE_B_REMOTE",
        default_host=config.runtime.remote_host,
        default_user=config.runtime.remote_user,
        default_python=config.runtime.remote_python,
        default_root=config.runtime.remote_root,
    )


class RemoteManager:
    def __init__(self, connection: RemoteConnectionConfig) -> None:
        self.connection = connection
        self._client: Any | None = None
        self._bridge_mode = paramiko is None
        self._sshpass_mode = self._bridge_mode and shutil.which("sshpass") is not None

    def __enter__(self) -> "RemoteManager":
        if self._bridge_mode:
            return self
        last_error: Exception | None = None
        for attempt in range(5):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=self.connection.host,
                    port=self.connection.port,
                    username=self.connection.user,
                    password=self.connection.password,
                    timeout=15,
                    banner_timeout=15,
                    auth_timeout=15,
                    look_for_keys=False,
                    allow_agent=False,
                )
                self._client = client
                return self
            except Exception as exc:
                last_error = exc
                try:
                    client.close()
                except Exception:
                    pass
                if attempt >= 4:
                    break
                time.sleep(min(2 + attempt, 5))
        if last_error is not None:
            raise last_error
        raise RuntimeError("remote client is not connected")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> paramiko.SSHClient:
        if self._bridge_mode:
            raise RuntimeError("bridge mode does not expose a live client")
        if self._client is None:
            raise RuntimeError("remote client is not connected")
        return self._client

    def _windows_python(self) -> str:
        configured_root = os.environ.get("POC_WINDOWS_UV_PYTHON_ROOT", "").strip()
        uv_root = Path(configured_root) if configured_root else self._default_windows_uv_root()
        preferred = uv_root / "cpython-3.11.14-windows-x86_64-none" / "python.exe"
        if preferred.exists():
            return str(preferred)
        candidates = sorted(
            path for path in uv_root.glob("*/python.exe") if "cpython-" in str(path)
        )
        for candidate in candidates:
            probe = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    "import importlib.util; print(importlib.util.find_spec('paramiko') is not None)",
                ],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0 and probe.stdout.strip() == "True":
                return str(candidate)
        raise RuntimeError("could not locate a Windows Python with paramiko installed")

    def _default_windows_uv_root(self) -> Path:
        proc = subprocess.run(
            ["cmd.exe", "/C", "echo %APPDATA%"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(
                "set POC_WINDOWS_UV_PYTHON_ROOT to the uv Python directory"
            )
        appdata_win = proc.stdout.strip()
        appdata_wsl = subprocess.run(
            ["wslpath", "-u", appdata_win],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return Path(appdata_wsl) / "uv" / "python"

    def _windows_path(self, path: Path) -> str:
        proc = subprocess.run(
            ["wslpath", "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    def _sshpass_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["SSHPASS"] = self.connection.password
        return env

    def _ssh_target(self) -> str:
        return f"{self.connection.user}@{self.connection.host}"

    def _ssh_base_command(self) -> list[str]:
        return [
            "sshpass",
            "-e",
            "ssh",
            "-p",
            str(self.connection.port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/tmp/pairwise_bpp_known_hosts",
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            self._ssh_target(),
        ]

    def _sshpass_run(self, command: str, *, timeout: float) -> tuple[str, str]:
        try:
            proc = subprocess.run(
                [*self._ssh_base_command(), command],
                capture_output=True,
                env=self._sshpass_env(),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            proc = subprocess.run(
                [*self._ssh_base_command(), command],
                capture_output=True,
                env=self._sshpass_env(),
                timeout=timeout,
                check=False,
            )
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode == 255:
            raise RuntimeError(f"sshpass ssh failed: {stderr or stdout}")
        return stdout, stderr

    def _bridge_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        bridge_path = self._windows_path(
            Path(__file__).resolve().with_name("windows_paramiko_bridge.py")
        )
        temp_dir = Path(__file__).resolve().parent / ".bridge_tmp"
        temp_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            dir=temp_dir,
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            temp_path = Path(handle.name)
        try:
            command = [
                self._windows_python(),
                bridge_path,
                self._windows_path(temp_path),
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=False,
            )
            stdout_text = proc.stdout.decode("utf-8", errors="replace")
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"bridge failed rc={proc.returncode} stdout={stdout_text!r} stderr={stderr_text!r}"
                )
            return json.loads(stdout_text)
        finally:
            temp_path.unlink(missing_ok=True)

    def run(self, command: str, *, timeout: float = 60.0) -> tuple[str, str]:
        if self._sshpass_mode:
            return self._sshpass_run(command, timeout=timeout)
        if self._bridge_mode:
            payload = {
                "action": "run",
                "host": self.connection.host,
                "port": self.connection.port,
                "user": self.connection.user,
                "password": self.connection.password,
                "command": command,
                "timeout": timeout,
            }
            result = self._bridge_call(payload)
            return result.get("stdout", ""), result.get("stderr", "")
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        return (
            stdout.read().decode("utf-8", errors="replace").strip(),
            stderr.read().decode("utf-8", errors="replace").strip(),
        )

    def run_detached(self, command: str, *, timeout: float = 15.0) -> None:
        if self._sshpass_mode:
            proc = subprocess.Popen(
                [*self._ssh_base_command(), command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._sshpass_env(),
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return
            if proc.returncode == 255:
                raise RuntimeError("sshpass detached ssh failed")
            return
        if self._bridge_mode:
            self._bridge_call(
                {
                    "action": "run_detached",
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "user": self.connection.user,
                    "password": self.connection.password,
                    "command": command,
                    "timeout": timeout,
                }
            )
            return
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        time.sleep(1.0)
        stdout.channel.close()
        stderr.channel.close()

    def launch_tunnel(
        self,
        *,
        remote_port: int,
        remote_host: str = "127.0.0.1",
        local_port: int = 0,
        bind_host: str = "0.0.0.0",
        ready_timeout: float = 30.0,
    ) -> RemoteTunnelHandle:
        if self._sshpass_mode:
            if local_port <= 0:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("127.0.0.1", 0))
                    local_port = int(sock.getsockname()[1])
            forward = f"{bind_host}:{local_port}:{remote_host}:{remote_port}"
            process = subprocess.Popen(
                [
                    *self._ssh_base_command(),
                    "-N",
                    "-L",
                    forward,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._sshpass_env(),
            )
            deadline = time.time() + ready_timeout
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"ssh tunnel exited early with rc={process.returncode}")
                if _probe_http_port("127.0.0.1", local_port):
                    return RemoteTunnelHandle(
                        pid=int(process.pid),
                        local_port=local_port,
                        mode="sshpass_tunnel",
                    )
                time.sleep(0.25)
            process.terminate()
            raise RuntimeError("ssh tunnel did not become ready")
        if not self._bridge_mode:
            raise RuntimeError("persistent_tunnel_requires_bridge_mode")
        payload = {
            "action": "launch_tunnel",
            "tunnel_payload": {
                "host": self.connection.host,
                "port": self.connection.port,
                "user": self.connection.user,
                "password": self.connection.password,
                "remote_host": remote_host,
                "remote_port": remote_port,
                "local_port": local_port,
                "bind_host": bind_host,
            },
            "ready_timeout": ready_timeout,
        }
        result = self._bridge_call(payload)
        return RemoteTunnelHandle(
            pid=int(result["pid"]),
            local_port=int(result["local_port"]),
            mode="ssh_tunnel",
        )

    def kill_process(self, pid: int) -> None:
        if self._sshpass_mode:
            subprocess.run(["kill", str(int(pid))], check=False)
            return
        if self._bridge_mode:
            self._bridge_call({"action": "kill_process", "pid": int(pid)})
            return
        raise RuntimeError("kill_process_requires_bridge_mode")

    def ensure_remote_root(self) -> None:
        self.run(f"mkdir -p {self.connection.remote_root}/logs {self.connection.remote_root}/pids")

    def upload_runtime_files(self, local_repo_root: Path) -> None:
        self.ensure_remote_root()
        remote_script = f"{self.connection.remote_root}/remote_model_server.py"
        local_script = local_repo_root / "src" / "pairwise_bpp_poc" / "remote_model_server.py"
        self.upload_file(local_script, remote_script)

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        if self._sshpass_mode:
            proc = subprocess.run(
                [
                    "sshpass",
                    "-e",
                    "scp",
                    "-P",
                    str(self.connection.port),
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/tmp/pairwise_bpp_known_hosts",
                    "-o",
                    "PreferredAuthentications=password",
                    "-o",
                    "PubkeyAuthentication=no",
                    str(local_path),
                    f"{self._ssh_target()}:{remote_path}",
                ],
                capture_output=True,
                env=self._sshpass_env(),
                check=False,
            )
            if proc.returncode != 0:
                stdout = proc.stdout.decode("utf-8", errors="replace")
                stderr = proc.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"sshpass scp failed rc={proc.returncode}: {stdout}{stderr}")
            return
        if self._bridge_mode:
            self._bridge_call(
                {
                    "action": "upload",
                    "host": self.connection.host,
                    "port": self.connection.port,
                    "user": self.connection.user,
                    "password": self.connection.password,
                    "local_path": self._windows_path(local_path),
                    "remote_path": remote_path,
                }
            )
            return
        with self.client.open_sftp() as sftp:
            sftp.put(str(local_path), remote_path)

    def gpu_health(self) -> dict[str, Any]:
        summary_cmd = (
            "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,"
            "memory.used,memory.total,power.draw,power.limit,fan.speed "
            "--format=csv,noheader,nounits"
        )
        processes_cmd = (
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
            "--format=csv,noheader,nounits"
        )
        summary, summary_err = self.run(summary_cmd)
        processes, process_err = self.run(processes_cmd)
        return {
            "summary": summary.splitlines() if summary else [],
            "processes": processes.splitlines() if processes else [],
            "summary_error": summary_err,
            "process_error": process_err,
        }

    def kill_gpu_processes(self, gpu_index: int) -> list[int]:
        gpu_uuid_out, _ = self.run(
            "nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader,nounits"
        )
        uuid_map = {}
        for line in gpu_uuid_out.splitlines():
            index_text, uuid = [part.strip() for part in line.split(",", 1)]
            uuid_map[int(index_text)] = uuid
        target_uuid = uuid_map[gpu_index]
        proc_out, _ = self.run(
            "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
        )
        killed: list[int] = []
        for line in proc_out.splitlines():
            gpu_uuid, pid_text = [part.strip() for part in line.split(",", 1)]
            if gpu_uuid == target_uuid:
                pid = int(pid_text)
                self.run(f"kill -TERM {pid} || true")
                killed.append(pid)
        return killed

    def start_worker(self, spec: WorkerSpec) -> None:
        self.ensure_remote_root()
        remote_script = f"{self.connection.remote_root}/remote_model_server.py"
        pid_path = f"{self.connection.remote_root}/pids/{spec.name}.pid"
        log_path = f"{self.connection.remote_root}/logs/{spec.name}.log"
        command = (
            f"cd {self.connection.remote_root} && "
            f"rm -f {pid_path} && "
            f"CUDA_VISIBLE_DEVICES={spec.gpu_index} PYTHONUNBUFFERED=1 "
            f"setsid -f {self.connection.python_path} -u "
            f"{remote_script} --model-path {spec.model_path} --device cuda:0 --port {spec.port} "
            f"< /dev/null > {log_path} 2>&1; "
            f"sleep 1; "
            "pid=$(ps -eo pid,comm,args | grep remote_model_server | grep python | grep -v grep "
            "| sed -n 's/^ *\\([0-9][0-9]*\\) .*/\\1/p' | head -n 1); "
            f"if [ -n \"$pid\" ]; then echo $pid > {pid_path}; fi"
        )
        self.run_detached(command, timeout=15.0)

    def stop_worker(self, name: str, port: int) -> None:
        pid_path = f"{self.connection.remote_root}/pids/{name}.pid"
        self.run(
            f"if [ -f {pid_path} ]; then "
            f"pid=$(cat {pid_path}); "
            f"kill -TERM $pid || true; "
            f"sleep 2; "
            f"kill -KILL $pid || true; "
            f"rm -f {pid_path}; "
            f"fi"
        )
        if os.environ.get("POC_REMOTE_ALLOW_PROCESS_SWEEP", "").strip().lower() in {"1", "true", "yes", "on"}:
            self.run(f"fuser -k {port}/tcp || true")
            self.run(f"pkill -f '[r]emote_model_server.py .* --port {port}' || true")

    def wait_for_http(self, port: int, *, timeout_seconds: float = 180.0) -> bool:
        deadline = time.time() + timeout_seconds
        last_status = ""
        last_error = ""
        while time.time() < deadline:
            stdout, _ = self.run(
                f"{self.connection.python_path} - <<'PY'\n"
                f"import json, urllib.request\n"
                f"try:\n"
                f"    with urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=3) as resp:\n"
                f"        data = json.loads(resp.read().decode('utf-8'))\n"
                f"        print(data.get('status', ''))\n"
                f"        print(data.get('error', ''))\n"
                f"except Exception as exc:\n"
                f"    print('')\n"
                f"    print(type(exc).__name__ + ': ' + str(exc))\n"
                f"PY",
                timeout=10.0,
            )
            lines = stdout.splitlines()
            last_status = lines[0].strip() if lines else ""
            last_error = lines[1].strip() if len(lines) > 1 else ""
            if last_status == "ok":
                return True
            if last_status == "error":
                return False
            time.sleep(2.0)
        return False


def remote_worker_specs(config: AppConfig) -> list[WorkerSpec]:
    return [
        WorkerSpec(
            name="generator_gpu0",
            gpu_index=_gpu_index_from_env("POC_GENERATOR_GPU_INDEX", 0),
            port=config.runtime.generator_port,
            model_path=generator_model_path(config),
        ),
        WorkerSpec(
            name="judge_a_gpu0",
            gpu_index=_gpu_index_from_env("POC_JUDGE_A_GPU_INDEX", 0),
            port=config.runtime.judge_a_port,
            model_path=config.runtime.judge_a_model_path,
        ),
        WorkerSpec(
            name="judge_b_gpu1",
            gpu_index=_gpu_index_from_env("POC_JUDGE_B_GPU_INDEX", 1),
            port=config.runtime.judge_b_port,
            model_path=os.environ.get(
                "POC_JUDGE_B_MODEL_PATH",
                config.runtime.judge_b_model_path,
            ),
        ),
    ]


def _probe_http_port(host: str, port: int, *, timeout_seconds: float = 1.5) -> bool:
    try:
        with urllib_request.urlopen(
            f"http://{host}:{port}/health",
            timeout=timeout_seconds,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("status") == "ok"
    except Exception:
        return False


def _wsl_windows_host() -> str:
    resolv_path = Path("/etc/resolv.conf")
    if resolv_path.exists():
        for line in resolv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("nameserver "):
                return line.split()[1]
    return "127.0.0.1"


def _choose_transport_mode(manager: RemoteManager, *, host: str, port: int) -> str:
    if _probe_http_port(host, port):
        return "direct_http"
    if manager._bridge_mode:
        return "ssh_tunnel"
    return "legacy_loopback"


class SelectorRemoteSession:
    def __init__(self, config: AppConfig, *, task: str) -> None:
        self.config = config
        self.task = task
        self.primary_connection = default_remote_connection(config)
        self.judge_b_connection = judge_b_remote_connection(config)
        self._stack = ExitStack()
        self._primary_manager: RemoteManager | None = None
        self._judge_b_manager: RemoteManager | None = None
        self._tunnels: list[tuple[RemoteManager, RemoteTunnelHandle]] = []
        self._started_workers: list[tuple[RemoteManager, WorkerSpec]] = []
        self._transport_modes: list[str] = []
        self._worker_start_count: dict[str, int] = {}
        self._worker_health_wait_seconds = 0.0

    def __enter__(self) -> SelectorRemoteBackends:
        self._primary_manager = self._stack.enter_context(RemoteManager(self.primary_connection))
        self._primary_manager.upload_runtime_files(self.config.runtime.root_dir)
        if self.judge_b_connection is not None:
            self._judge_b_manager = self._stack.enter_context(RemoteManager(self.judge_b_connection))
            self._judge_b_manager.upload_runtime_files(self.config.runtime.root_dir)
        specs = {spec.name: spec for spec in remote_worker_specs(self.config)}
        generator_needed = self.task != "jsplib"
        if generator_needed:
            self._start_worker(self._primary_manager, specs["generator_gpu0"])
        self._start_worker(self._primary_manager, specs["judge_a_gpu0"])
        if self._judge_b_manager is None:
            self._start_worker(self._primary_manager, specs["judge_b_gpu1"])
        else:
            self._start_worker(
                self._judge_b_manager,
                WorkerSpec(
                    name=specs["judge_b_gpu1"].name,
                    gpu_index=0,
                    port=specs["judge_b_gpu1"].port,
                    model_path=specs["judge_b_gpu1"].model_path,
                ),
            )
        generator_backend = (
            self._make_backend(
                self._primary_manager,
                specs["generator_gpu0"],
                name="generator",
                model_id=generator_model_path(self.config),
            )
            if generator_needed
            else None
        )
        judge_a_backend = self._make_backend(
            self._primary_manager,
            specs["judge_a_gpu0"],
            name="judge_a",
            model_id=self.config.runtime.judge_a_model_path,
        )
        judge_b_spec = specs["judge_b_gpu1"]
        judge_b_backend = self._make_backend(
            self._judge_b_manager or self._primary_manager,
            WorkerSpec(
                name=judge_b_spec.name,
                gpu_index=0 if self._judge_b_manager is not None else judge_b_spec.gpu_index,
                port=judge_b_spec.port,
                model_path=judge_b_spec.model_path,
            ),
            name="judge_b",
            model_id=os.environ.get(
                "POC_JUDGE_B_MODEL_PATH",
                self.config.runtime.judge_b_model_path,
            ),
        )
        return SelectorRemoteBackends(
            generator_backend=generator_backend,
            judge_backends=[judge_a_backend, judge_b_backend],
            runtime_metrics={
                "transport_mode": self._transport_mode_value(),
                "worker_start_count": dict(self._worker_start_count),
                "worker_health_wait_seconds": round(self._worker_health_wait_seconds, 6),
            },
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        for manager, handle in reversed(self._tunnels):
            try:
                manager.kill_process(handle.pid)
            except Exception:
                pass
        for manager, spec in reversed(self._started_workers):
            try:
                manager.stop_worker(spec.name, spec.port)
            except Exception:
                pass
        self._stack.close()

    def _transport_mode_value(self) -> str:
        unique = {mode for mode in self._transport_modes if mode}
        if not unique:
            return "legacy_loopback"
        if len(unique) == 1:
            return next(iter(unique))
        return "mixed"

    def _start_worker(self, manager: RemoteManager, spec: WorkerSpec) -> None:
        manager.stop_worker(spec.name, spec.port)
        manager.start_worker(spec)
        started = time.perf_counter()
        if not manager.wait_for_http(spec.port, timeout_seconds=600.0):
            raise RuntimeError(f"{spec.name} failed health check")
        self._worker_health_wait_seconds += time.perf_counter() - started
        self._worker_start_count[spec.name] = self._worker_start_count.get(spec.name, 0) + 1
        self._started_workers.append((manager, spec))

    def _make_backend(
        self,
        manager: RemoteManager,
        spec: WorkerSpec,
        *,
        name: str,
        model_id: str,
    ) -> RemoteHTTPTextBackend | RemoteLoopbackTextBackend | BridgeHTTPTextBackend:
        host = manager.connection.host
        transport_mode = _choose_transport_mode(manager, host=host, port=spec.port)
        self._transport_modes.append(transport_mode)
        if transport_mode == "direct_http":
            return RemoteHTTPTextBackend(
                name=name,
                model_id=model_id,
                url=f"http://{host}:{spec.port}",
                timeout_seconds=180.0,
            )
        if transport_mode == "ssh_tunnel":
            tunnel = manager.launch_tunnel(remote_port=spec.port)
            self._tunnels.append((manager, tunnel))
            return BridgeHTTPTextBackend(
                manager=manager,
                name=name,
                model_id=model_id,
                url=f"http://127.0.0.1:{tunnel.local_port}",
                timeout_seconds=180.0,
            )
        return RemoteLoopbackTextBackend(
            connection=manager.connection,
            name=name,
            model_id=model_id,
            remote_port=spec.port,
            timeout_seconds=180.0,
        )


def selector_remote_session(config: AppConfig, *, task: str) -> SelectorRemoteSession:
    return SelectorRemoteSession(config, task=task)


class RemoteLoopbackTextBackend:
    def __init__(
        self,
        *,
        connection: RemoteConnectionConfig,
        name: str,
        model_id: str,
        remote_port: int,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.connection = connection
        self.name = name
        self.model_id = model_id
        self.remote_port = remote_port
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> ModelResponse:
        payload = base64.b64encode(
            json.dumps(
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                }
            ).encode("utf-8")
        ).decode("ascii")
        command = (
            f"{self.connection.python_path} - <<'PY'\n"
            "import base64, sys, urllib.error, urllib.request\n"
            f"payload = base64.b64decode('{payload}')\n"
            f"req = urllib.request.Request('http://127.0.0.1:{self.remote_port}/generate', data=payload, method='POST', headers={{'Content-Type': 'application/json'}})\n"
            "try:\n"
            f"    print(urllib.request.urlopen(req, timeout={self.timeout_seconds}).read().decode('utf-8'))\n"
            "except urllib.error.HTTPError as exc:\n"
            "    sys.stderr.write(exc.read().decode('utf-8', errors='replace'))\n"
            "    raise\n"
            "PY"
        )
        started = time.perf_counter()
        with RemoteManager(self.connection) as manager:
            stdout, stderr = manager.run(command, timeout=self.timeout_seconds + 30.0)
        if stderr:
            raise RuntimeError(stderr)
        body = json.loads(stdout)
        return ModelResponse(
            text=body["text"],
            latency_seconds=round(time.perf_counter() - started, 6),
            raw=body,
        )

    def health(self) -> dict[str, Any]:
        command = (
            f"{self.connection.python_path} - <<'PY'\n"
            "import urllib.request\n"
            f"print(urllib.request.urlopen('http://127.0.0.1:{self.remote_port}/health', timeout=10).read().decode('utf-8'))\n"
            "PY"
        )
        with RemoteManager(self.connection) as manager:
            stdout, stderr = manager.run(command, timeout=20.0)
        if stderr:
            raise RuntimeError(stderr)
        return json.loads(stdout)


class BridgeHTTPTextBackend:
    def __init__(
        self,
        *,
        manager: RemoteManager,
        name: str,
        model_id: str,
        url: str,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.manager = manager
        self.name = name
        self.model_id = model_id
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _bridge_json_request(self, payload: dict[str, Any], *, retries: int = 1) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self.manager._bridge_call(payload)
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> ModelResponse:
        started = time.perf_counter()
        body = self._bridge_json_request(
            {
                "action": "http_post_json",
                "url": self.url + "/generate",
                "payload": {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                },
                "timeout": self.timeout_seconds,
            },
            retries=1,
        )
        return ModelResponse(
            text=body["text"],
            latency_seconds=round(time.perf_counter() - started, 6),
            raw=body,
        )

    def health(self) -> dict[str, Any]:
        return self._bridge_json_request(
            {
                "action": "http_get_json",
                "url": self.url + "/health",
                "timeout": 15.0,
            },
            retries=1,
        )


class SingleGPUWorkerController:
    def __init__(self, connection: RemoteConnectionConfig) -> None:
        self.connection = connection
        self._active: tuple[str, int] | None = None

    def ensure_worker(self, spec: WorkerSpec) -> None:
        active = (spec.name, spec.port)
        if self._active == active:
            with RemoteManager(self.connection) as manager:
                if manager.wait_for_http(spec.port, timeout_seconds=10.0):
                    return
                manager.stop_worker(spec.name, spec.port)
                manager.start_worker(spec)
                if not manager.wait_for_http(spec.port):
                    raise RuntimeError(f"{spec.name} failed health check")
            self._active = active
            return
        with RemoteManager(self.connection) as manager:
            if self._active is not None:
                manager.stop_worker(*self._active)
            manager.stop_worker(spec.name, spec.port)
            manager.start_worker(spec)
            if not manager.wait_for_http(spec.port):
                raise RuntimeError(f"{spec.name} failed health check")
        self._active = active

    def stop_active(self) -> None:
        if self._active is None:
            return
        with RemoteManager(self.connection) as manager:
            manager.stop_worker(*self._active)
        self._active = None


class HotSwapRemoteTextBackend(RemoteLoopbackTextBackend):
    def __init__(
        self,
        *,
        connection: RemoteConnectionConfig,
        controller: SingleGPUWorkerController,
        spec: WorkerSpec,
        name: str,
        model_id: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        super().__init__(
            connection=connection,
            name=name,
            model_id=model_id,
            remote_port=spec.port,
            timeout_seconds=timeout_seconds,
        )
        self.controller = controller
        self.spec = spec

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        max_new_tokens: int = 256,
    ) -> ModelResponse:
        self.controller.ensure_worker(self.spec)
        return super().generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )

    def health(self) -> dict[str, Any]:
        self.controller.ensure_worker(self.spec)
        return super().health()


def build_remote_backends(
    config: AppConfig,
    *,
    loopback: bool = True,
    orchestration_mode: str = "default",
) -> tuple[RemoteHTTPTextBackend | RemoteLoopbackTextBackend, list[RemoteHTTPTextBackend | RemoteLoopbackTextBackend]]:
    host = os.environ.get("POC_REMOTE_HOST", config.runtime.remote_host)
    if loopback:
        connection = default_remote_connection(config)
        serial_single_gpu = (
            orchestration_mode == "serial_single_gpu"
            or (orchestration_mode == "default" and use_serial_single_gpu_mode())
        )
        dual_hotswap = (
            orchestration_mode == "dual_hotswap"
            or (orchestration_mode == "default" and use_dual_hotswap_mode())
        )
        judge_b_model_path = os.environ.get(
            "POC_JUDGE_B_MODEL_PATH",
            config.runtime.judge_b_model_path,
        )
        if serial_single_gpu:
            controller = SingleGPUWorkerController(connection)
            single_gpu_index = _gpu_index_from_env("POC_GENERATOR_GPU_INDEX", 0)
            generator_spec = WorkerSpec(
                name="generator_gpu0",
                gpu_index=single_gpu_index,
                port=config.runtime.generator_port,
                model_path=generator_model_path(config),
            )
            judge_a_spec = WorkerSpec(
                name="judge_a_gpu0",
                gpu_index=_gpu_index_from_env("POC_JUDGE_A_GPU_INDEX", single_gpu_index),
                port=config.runtime.judge_a_port,
                model_path=config.runtime.judge_a_model_path,
            )
            judge_b_spec = WorkerSpec(
                name="judge_b_gpu1",
                gpu_index=_gpu_index_from_env("POC_JUDGE_B_GPU_INDEX", single_gpu_index),
                port=config.runtime.judge_b_port,
                model_path=judge_b_model_path,
            )
            generator = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller,
                spec=generator_spec,
                name="generator",
                model_id=generator_model_path(config),
                timeout_seconds=180.0,
            )
            judge_a = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller,
                spec=judge_a_spec,
                name="judge_a",
                model_id=config.runtime.judge_a_model_path,
                timeout_seconds=180.0,
            )
            judge_b = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller,
                spec=judge_b_spec,
                name="judge_b",
                model_id=judge_b_model_path,
                timeout_seconds=180.0,
            )
        elif dual_hotswap:
            controller_gpu0 = SingleGPUWorkerController(connection)
            controller_gpu1 = SingleGPUWorkerController(connection)
            generator_spec = WorkerSpec(
                name="generator_gpu0",
                gpu_index=_gpu_index_from_env("POC_GENERATOR_GPU_INDEX", 0),
                port=config.runtime.generator_port,
                model_path=generator_model_path(config),
            )
            judge_a_spec = WorkerSpec(
                name="judge_a_gpu0",
                gpu_index=_gpu_index_from_env("POC_JUDGE_A_GPU_INDEX", 0),
                port=config.runtime.judge_a_port,
                model_path=config.runtime.judge_a_model_path,
            )
            judge_b_spec = WorkerSpec(
                name="judge_b_gpu1",
                gpu_index=_gpu_index_from_env("POC_JUDGE_B_GPU_INDEX", 1),
                port=config.runtime.judge_b_port,
                model_path=judge_b_model_path,
            )
            generator = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller_gpu0,
                spec=generator_spec,
                name="generator",
                model_id=generator_model_path(config),
                timeout_seconds=180.0,
            )
            judge_a = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller_gpu0,
                spec=judge_a_spec,
                name="judge_a",
                model_id=config.runtime.judge_a_model_path,
                timeout_seconds=180.0,
            )
            judge_b = HotSwapRemoteTextBackend(
                connection=connection,
                controller=controller_gpu1,
                spec=judge_b_spec,
                name="judge_b",
                model_id=judge_b_model_path,
                timeout_seconds=180.0,
            )
        else:
            generator = RemoteLoopbackTextBackend(
                connection=connection,
                name="generator",
                model_id=generator_model_path(config),
                remote_port=config.runtime.generator_port,
                timeout_seconds=180.0,
            )
            judge_a = RemoteLoopbackTextBackend(
                connection=connection,
                name="judge_a",
                model_id=config.runtime.judge_a_model_path,
                remote_port=config.runtime.judge_a_port,
                timeout_seconds=180.0,
            )
            judge_b = RemoteLoopbackTextBackend(
                connection=judge_b_remote_connection(config) or connection,
                name="judge_b",
                model_id=judge_b_model_path,
                remote_port=config.runtime.judge_b_port,
                timeout_seconds=180.0,
            )
    else:
        generator = RemoteHTTPTextBackend(
            name="generator",
            model_id=generator_model_path(config),
            url=f"http://{host}:{config.runtime.generator_port}",
            timeout_seconds=180.0,
        )
        judge_a = RemoteHTTPTextBackend(
            name="judge_a",
            model_id=config.runtime.judge_a_model_path,
            url=f"http://{host}:{config.runtime.judge_a_port}",
            timeout_seconds=180.0,
        )
        judge_b = RemoteHTTPTextBackend(
            name="judge_b",
            model_id=config.runtime.judge_b_model_path,
            url=f"http://{host}:{config.runtime.judge_b_port}",
            timeout_seconds=180.0,
        )
    return generator, [judge_a, judge_b]
