from __future__ import annotations

import json
import select
import socket
import sys
import threading
import time
from pathlib import Path

import paramiko


def _pump(left: socket.socket, right) -> None:
    sockets = [left, right]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], 1.0)
            if not readable:
                continue
            for current in readable:
                other = right if current is left else left
                data = current.recv(32768)
                if not data:
                    return
                other.sendall(data)
    finally:
        try:
            left.close()
        except Exception:
            pass
        try:
            right.close()
        except Exception:
            pass


def _handle_client(client: socket.socket, transport: paramiko.Transport, remote_host: str, remote_port: int) -> None:
    channel = transport.open_channel(
        "direct-tcpip",
        (remote_host, remote_port),
        client.getsockname(),
    )
    _pump(client, channel)


def main() -> None:
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    ready_path = Path(payload["ready_path"])
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

    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("ssh_transport_unavailable")
    transport.set_keepalive(30)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((str(payload.get("bind_host", "127.0.0.1")), int(payload.get("local_port", 0))))
    listener.listen(64)
    local_port = listener.getsockname()[1]
    ready_path.write_text(json.dumps({"local_port": local_port}), encoding="utf-8")

    try:
        while True:
            client_socket, _address = listener.accept()
            thread = threading.Thread(
                target=_handle_client,
                args=(
                    client_socket,
                    transport,
                    str(payload.get("remote_host", "127.0.0.1")),
                    int(payload["remote_port"]),
                ),
                daemon=True,
            )
            thread.start()
    finally:
        try:
            listener.close()
        finally:
            client.close()


if __name__ == "__main__":
    main()
