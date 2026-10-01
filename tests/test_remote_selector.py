from __future__ import annotations

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.remote import (
    BridgeHTTPTextBackend,
    RemoteConnectionConfig,
    RemoteTunnelHandle,
    _choose_transport_mode,
    selector_remote_session,
)


def test_choose_transport_mode_prefers_direct_http(monkeypatch):
    class DummyManager:
        _bridge_mode = True

    monkeypatch.setattr("pairwise_bpp_poc.remote._probe_http_port", lambda *args, **kwargs: True)
    assert _choose_transport_mode(DummyManager(), host="10.0.0.1", port=18081) == "direct_http"


def test_choose_transport_mode_falls_back_to_ssh_tunnel(monkeypatch):
    class DummyManager:
        _bridge_mode = True

    monkeypatch.setattr("pairwise_bpp_poc.remote._probe_http_port", lambda *args, **kwargs: False)
    assert _choose_transport_mode(DummyManager(), host="10.0.0.1", port=18081) == "ssh_tunnel"


def test_selector_remote_session_jsplib_skips_generator(monkeypatch, tmp_path):
    config = default_config(tmp_path)
    fake_connection = RemoteConnectionConfig(
        host="10.0.0.1",
        port=22,
        user="tester",
        password="secret",
        python_path="/usr/bin/python3",
        remote_root="/tmp/pairwise-bpp-runtime",
    )
    started: list[str] = []

    def fake_enter(self):
        return self

    def fake_exit(self, exc_type, exc, tb):
        return None

    def fake_upload(self, _root):
        return None

    def fake_stop(self, name, port):
        return None

    def fake_start(self, spec):
        started.append(spec.name)

    def fake_wait(self, port, timeout_seconds=180.0):
        return True

    def fake_tunnel(self, *, remote_port, remote_host="127.0.0.1", local_port=0, ready_timeout=30.0):
        return RemoteTunnelHandle(pid=remote_port, local_port=19000 + remote_port, mode="ssh_tunnel")

    monkeypatch.setattr("pairwise_bpp_poc.remote.default_remote_connection", lambda config: fake_connection)
    monkeypatch.setattr("pairwise_bpp_poc.remote.judge_b_remote_connection", lambda config: None)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.__enter__", fake_enter)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.__exit__", fake_exit)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.upload_runtime_files", fake_upload)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.stop_worker", fake_stop)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.start_worker", fake_start)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.wait_for_http", fake_wait)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.launch_tunnel", fake_tunnel)
    monkeypatch.setattr("pairwise_bpp_poc.remote.RemoteManager.kill_process", lambda self, pid: None)
    monkeypatch.setattr("pairwise_bpp_poc.remote._probe_http_port", lambda *args, **kwargs: False)

    with selector_remote_session(config, task="jsplib") as session:
        assert session.generator_backend is None
        assert all(isinstance(backend, BridgeHTTPTextBackend) for backend in session.judge_backends)
        assert session.runtime_metrics["transport_mode"] == "ssh_tunnel"
        assert set(session.runtime_metrics["worker_start_count"]) == {"judge_a_gpu0", "judge_b_gpu1"}

    assert started == ["judge_a_gpu0", "judge_b_gpu1"]


def test_bridge_http_backend_retries_once_on_timeout():
    calls = {"count": 0}

    class DummyManager:
        def _bridge_call(self, payload):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("temporary timeout")
            if payload["action"] == "http_post_json":
                return {"text": "{\"ok\": true}"}
            return {"status": "ok"}

    backend = BridgeHTTPTextBackend(
        manager=DummyManager(),
        name="judge_a",
        model_id="mock",
        url="http://127.0.0.1:19001",
    )

    response = backend.generate("hello", system_prompt="system")
    assert response.text == "{\"ok\": true}"
    assert calls["count"] == 2
