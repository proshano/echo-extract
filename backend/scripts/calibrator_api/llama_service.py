from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib.request import Request, urlopen


DEFAULT_MODEL_DIRS = [
    Path.home() / "models",
    Path.home() / "llama-models",
    Path.home() / ".local" / "share" / "llama.cpp" / "models",
]
SLOW_MODEL_DIRS = [Path.home() / ".cache" / "huggingface" / "hub"]
MODEL_SCAN_TIME_BUDGET_SECONDS = 2.0


class LlamaServiceError(ValueError):
    """Raised when llama-server management fails."""


def normalize_base_url(base_url: str) -> str:
    value = base_url.strip()
    if not value:
        raise LlamaServiceError("llama_url must be non-empty.")
    return value.rstrip("/")


def fetch_server_models(llama_url: str, *, timeout_seconds: float = 4.0) -> List[str]:
    base_url = normalize_base_url(llama_url)
    endpoint = f"{base_url}/v1/models"
    request = Request(endpoint, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload_raw = response.read()
    except urllib_error.HTTPError as exc:
        raise LlamaServiceError(f"Model endpoint returned HTTP {exc.code}.") from exc
    except urllib_error.URLError as exc:
        raise LlamaServiceError(f"Could not reach model endpoint: {exc.reason}.") from exc
    except TimeoutError as exc:
        raise LlamaServiceError("Timed out while fetching server models.") from exc

    try:
        payload = json.loads(payload_raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise LlamaServiceError("Model endpoint returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise LlamaServiceError("Model endpoint returned an unexpected payload shape.")

    entries = payload.get("data", [])
    if not isinstance(entries, list):
        raise LlamaServiceError("Model endpoint payload is missing a 'data' array.")

    models: List[str] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)

    return models


class LlamaServerManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._model_path = ""
        self._port: Optional[int] = None
        self._ctx_size: Optional[int] = None
        self._started_at_unix: Optional[int] = None
        self._logs: Deque[str] = deque(maxlen=300)

    def get_binary_path(self) -> str:
        return shutil.which("llama-server") or ""

    def list_local_models(self, *, include_slow_dirs: bool = False) -> Tuple[List[str], bool]:
        search_dirs = list(DEFAULT_MODEL_DIRS)
        if include_slow_dirs:
            search_dirs.extend(SLOW_MODEL_DIRS)

        env_dir = os.environ.get("LLM_MODELS_DIR", "").strip()
        if env_dir:
            search_dirs.insert(0, Path(env_dir))

        seen = set()
        models: List[str] = []
        timed_out = False
        started = time.perf_counter()

        for base_dir in search_dirs:
            if not base_dir.exists():
                continue

            for root_dir, _, files in os.walk(base_dir):
                if not include_slow_dirs and (time.perf_counter() - started) > MODEL_SCAN_TIME_BUDGET_SECONDS:
                    timed_out = True
                    break

                for file_name in files:
                    if not file_name.lower().endswith(".gguf"):
                        continue
                    resolved = str((Path(root_dir) / file_name).resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    models.append(resolved)

            if timed_out:
                break

        models.sort()
        return models, timed_out

    def _append_log(self, message: str) -> None:
        with self._lock:
            self._logs.append(message)

    def _stream_logs(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            text = line.rstrip()
            if text:
                self._append_log(text)

    def _is_managed_process_running(self) -> bool:
        with self._lock:
            process = self._process
            return process is not None and process.poll() is None

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return

        used_process_group = False
        process_group_id: Optional[int] = None

        if os.name != "nt":
            try:
                process_group_id = os.getpgid(process.pid)
            except (OSError, ProcessLookupError):
                process_group_id = None

            if process_group_id is not None and process_group_id == process.pid:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                    used_process_group = True
                except ProcessLookupError:
                    return
                except OSError as exc:
                    self._append_log(f"[managed] Could not signal process group: {exc}. Falling back to terminate().")

        if not used_process_group:
            try:
                process.terminate()
            except OSError:
                return

        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass

        if used_process_group and process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                return
            except OSError as exc:
                self._append_log(f"[managed] Could not force-kill process group: {exc}. Falling back to kill().")
                try:
                    process.kill()
                except OSError:
                    return
        else:
            try:
                process.kill()
            except OSError:
                return

        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._append_log("[managed] Warning: llama-server did not exit after force kill.")

    def start(self, *, model_path: str, port: int, ctx_size: int = 8192) -> Dict[str, object]:
        model_value = model_path.strip()
        if not model_value:
            raise LlamaServiceError("model_path must be non-empty.")

        model_file = Path(model_value).expanduser()
        if not model_file.exists():
            raise LlamaServiceError(f"Model file not found: {model_file}")

        binary = self.get_binary_path()
        if not binary:
            raise LlamaServiceError("llama-server was not found in PATH.")

        if self._is_managed_process_running():
            raise LlamaServiceError("A managed llama-server is already running. Stop it first.")

        command = [binary, "-m", str(model_file), "--port", str(port), "--ctx-size", str(ctx_size)]
        try:
            popen_kwargs: Dict[str, object] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
            }
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if creation_flags:
                    popen_kwargs["creationflags"] = creation_flags
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **popen_kwargs)
        except OSError as exc:
            raise LlamaServiceError(f"Failed to start llama-server: {exc}") from exc

        with self._lock:
            self._process = process
            self._model_path = str(model_file)
            self._port = int(port)
            self._ctx_size = int(ctx_size)
            self._started_at_unix = int(time.time())
            self._logs.clear()
            self._logs.append(f"[managed] Started: {' '.join(command)}")

        thread = threading.Thread(target=self._stream_logs, args=(process,), daemon=True)
        thread.start()

        for _ in range(15):
            if process.poll() is not None:
                break
            time.sleep(0.1)

        if process.poll() is not None:
            logs = self.get_logs_tail(limit=20)
            with self._lock:
                self._process = None
            joined_logs = "\n".join(logs[-8:]).strip()
            if joined_logs:
                raise LlamaServiceError(f"llama-server exited immediately.\n{joined_logs}")
            raise LlamaServiceError("llama-server exited immediately.")

        return self.get_process_info()

    def ensure_running(self, *, model_path: str, port: int, ctx_size: int = 8192) -> Dict[str, object]:
        model_file = Path(model_path.strip()).expanduser()
        if not model_file.exists():
            raise LlamaServiceError(f"Model file not found: {model_file}")

        resolved_model = str(model_file.resolve())
        desired_port = int(port)
        desired_ctx_size = int(ctx_size)

        with self._lock:
            process = self._process
            is_running = process is not None and process.poll() is None
            same_model = self._model_path == resolved_model
            same_port = self._port == desired_port
            same_ctx = self._ctx_size == desired_ctx_size

        if is_running and same_model and same_port and same_ctx:
            info = self.get_process_info()
            info["changed"] = False
            return info

        if is_running:
            self.stop()

        info = self.start(model_path=resolved_model, port=desired_port, ctx_size=desired_ctx_size)
        info["changed"] = True
        return info

    def stop(self) -> Dict[str, object]:
        with self._lock:
            process = self._process

        if process is None or process.poll() is not None:
            self._append_log("[managed] Stop requested, but no managed process is running.")
            return self.get_process_info()

        self._terminate_process(process)

        self._append_log("[managed] llama-server stopped.")
        with self._lock:
            self._process = None

        return self.get_process_info()

    def get_logs_tail(self, *, limit: int = 40) -> List[str]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._logs)[-limit:]

    def get_process_info(self) -> Dict[str, object]:
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            return {
                "process_running": running,
                "managed_model_path": self._model_path,
                "managed_port": self._port,
                "managed_ctx_size": self._ctx_size,
                "started_at_unix": self._started_at_unix,
            }

    def status(self, *, llama_url: str) -> Dict[str, object]:
        info = self.get_process_info()
        base_url = normalize_base_url(llama_url)

        reachable = False
        connect_error = ""
        server_models: List[str] = []
        try:
            server_models = fetch_server_models(base_url, timeout_seconds=2.5)
            reachable = True
        except LlamaServiceError as exc:
            connect_error = str(exc)

        info.update(
            {
                "llama_url": base_url,
                "binary_path": self.get_binary_path(),
                "reachable": reachable,
                "server_models": server_models,
                "connect_error": connect_error,
                "logs_tail": self.get_logs_tail(limit=40),
            }
        )
        return info


llama_server_manager = LlamaServerManager()
