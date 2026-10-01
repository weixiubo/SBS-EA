from __future__ import annotations

import json
import sys
import base64
import os
import subprocess
import tempfile
from pathlib import Path
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

import paramiko


def main() -> None:
    arg = sys.argv[1]
    payload = json.loads(base64.b64decode(arg).decode("utf-8")) if not Path(arg).exists() else json.loads(Path(arg).read_text(encoding="utf-8"))
    action = payload["action"]
    if action == "launch_tunnel":
        tunnel_script = Path(__file__).with_name("windows_paramiko_tunnel.py")
        temp_dir = Path(tempfile.mkdtemp(prefix="pairwise-bpp-tunnel-"))
        ready_path = temp_dir / "ready.json"
        tunnel_payload = dict(payload["tunnel_payload"])
        tunnel_payload["ready_path"] = str(ready_path)
        payload_path = temp_dir / "payload.json"
        payload_path.write_text(json.dumps(tunnel_payload, ensure_ascii=False), encoding="utf-8")
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        proc = subprocess.Popen(
            [sys.executable, str(tunnel_script), str(payload_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            cwd=str(tunnel_script.parent),
        )
        ready_timeout = float(payload.get("ready_timeout", 30.0))
        deadline = time.time() + ready_timeout
        ready_payload = None
        while time.time() < deadline:
            if ready_path.exists():
                ready_payload = json.loads(ready_path.read_text(encoding="utf-8"))
                break
            if proc.poll() is not None:
                raise RuntimeError(f"tunnel process exited early rc={proc.returncode}")
            time.sleep(0.2)
        if ready_payload is None:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            raise RuntimeError("tunnel_ready_timeout")
        print(json.dumps({"ok": True, "pid": proc.pid, "local_port": int(ready_payload["local_port"])}, ensure_ascii=False))
        try:
            payload_path.unlink(missing_ok=True)
            ready_path.unlink(missing_ok=True)
            os.rmdir(temp_dir)
        except Exception:
            pass
        return
    if action == "kill_process":
        pid = int(payload["pid"])
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        print(
            json.dumps(
                {
                    "ok": completed.returncode == 0,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                    "returncode": completed.returncode,
                },
                ensure_ascii=False,
            )
        )
        return
    if action == "http_get_json":
        with urllib_request.urlopen(payload["url"], timeout=float(payload.get("timeout", 30.0))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print(json.dumps(body, ensure_ascii=False))
        return
    if action == "http_post_json":
        req = urllib_request.Request(
            payload["url"],
            data=json.dumps(payload["payload"], ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(req, timeout=float(payload.get("timeout", 30.0))) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
        print(json.dumps(body, ensure_ascii=False))
        return
    connect_timeout = float(payload.get("connect_timeout", 30.0))
    connect_retries = int(payload.get("connect_retries", 3))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_error: Exception | None = None
    for attempt in range(connect_retries):
        try:
            client.connect(
                hostname=payload["host"],
                port=int(payload.get("port", 22)),
                username=payload["user"],
                password=payload["password"],
                timeout=connect_timeout,
                banner_timeout=connect_timeout,
                auth_timeout=connect_timeout,
                look_for_keys=False,
                allow_agent=False,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= connect_retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    try:
        if action == "run":
            stdin, stdout, stderr = client.exec_command(
                payload["command"],
                timeout=float(payload.get("timeout", 60.0)),
            )
            result = {
                "stdout": stdout.read().decode("utf-8", errors="replace").strip(),
                "stderr": stderr.read().decode("utf-8", errors="replace").strip(),
            }
        elif action == "run_detached":
            stdin, stdout, stderr = client.exec_command(
                payload["command"],
                timeout=float(payload.get("timeout", 15.0)),
            )
            time.sleep(1.0)
            stdout.channel.close()
            stderr.channel.close()
            result = {"ok": True}
        elif action == "upload":
            with client.open_sftp() as sftp:
                sftp.put(payload["local_path"], payload["remote_path"])
            result = {"ok": True}
        else:
            raise ValueError(f"unsupported action: {action}")
    finally:
        client.close()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
