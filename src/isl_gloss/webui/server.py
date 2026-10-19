"""A tiny local web server for the interactive pipeline tester (standard library only, no new dependencies).

Safety choices:
* Binds to the loopback address only; a non-loopback host is refused.
* Rejects requests whose Host or Origin header is not this local server, so a web page on another site cannot make your browser
  spend your API quota or read results.
* POST bodies must be JSON and are size-limited; only one LLM run at a time.
* The API key is read server-side from the environment/.env and never sent to the page; any key value is scrubbed from responses.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from ..youtube.errors import IngestError
from . import demo
from .youtube_runner import YouTubeRunner

STATIC = Path(__file__).resolve().parent / "static"
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
MAX_BODY = 64 * 1024
DRAIN_LIMIT = 1024 * 1024  # how much of a rejected request's body is read and discarded before replying
_RUN_LOCK = threading.Lock()


def _hostname(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("["):
        return value.split("]")[0] + "]"
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class Handler(BaseHTTPRequestHandler):
    server_version = "ISLPipelineTester"
    env_file = ".env"
    data_dir = "data/demo"
    jobs_dir = "data/jobs"
    runner: YouTubeRunner = None  # set by make_server

    def log_message(self, fmt, *args):  # request lines only; never bodies (they contain the user's text)
        pass

    # --- helpers ---------------------------------------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                                    "connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, demo.scrub(json.dumps(obj, ensure_ascii=False)).encode("utf-8"), "application/json; charset=utf-8")

    def _allowed(self) -> bool:
        if _hostname(self.headers.get("Host", "")) not in LOOPBACK:
            return False
        origin = self.headers.get("Origin")
        return not origin or _hostname(urlsplit(origin).netloc) in LOOPBACK

    # --- routes ----------------------------------------------------------------------------------------------
    def do_GET(self):
        if not self._allowed():
            return self._json(403, {"error": "This tester only answers requests from this computer."})
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self._json(200, demo.status(self.env_file))
        if path == "/api/youtube/status":
            job_id = (parse_qs(urlsplit(self.path).query).get("job_id") or [""])[0]
            if not demo.JOB_ID.match(job_id):
                return self._json(400, {"error": "That is not a valid video job id."})
            return self._json(200, self.runner.status(job_id))
        self._json(404, {"error": "not found"})

    def _reject(self, code: int, obj: dict) -> None:
        """Answer a refused POST. The request body is read and thrown away first (bounded): replying and closing while the client is
        still sending makes Windows reset the connection, and the client would then see a network error instead of this message."""
        try:
            left = min(max(int(self.headers.get("Content-Length", "0")), 0), DRAIN_LIMIT)
        except ValueError:
            left = 0
        while left > 0:
            chunk = self.rfile.read(min(left, 65536))
            if not chunk:
                break
            left -= len(chunk)
        self.close_connection = True
        self._json(code, obj)

    def do_POST(self):
        if not self._allowed():
            return self._reject(403, {"error": "This tester only answers requests from this computer."})
        path = urlsplit(self.path).path
        if path not in ("/api/analyze", "/api/translate", "/api/youtube/start"):
            return self._reject(404, {"error": "not found"})
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return self._reject(415, {"error": "Content-Type must be application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return self._reject(413, {"error": "request body too large"})
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            text = body.get("text")
            opts = {k: body.get(k) for k in ("prompt_version", "reasoning_effort", "call_llm", "use_cache", "job_id", "max_sentences") if k in body}
        except (ValueError, AttributeError):
            return self._json(400, {"error": "request body is not valid JSON"})
        try:
            if path == "/api/youtube/start":
                try:
                    return self._json(200, self.runner.start(body.get("url"), body.get("model") or None, bool(body.get("force"))))
                except IngestError as exc:  # a bad link or a busy server: a readable message, not a stack trace
                    return self._json(400 if exc.stage in ("validation",) else 500, {"error": exc.message, "hint": exc.hint, "kind": exc.kind})
            if path == "/api/analyze":
                return self._json(200, demo.analyze(text, opts, self.jobs_dir))
            if not _RUN_LOCK.acquire(blocking=False):
                return self._json(429, {"error": "Another run is still in progress. Wait for it to finish."})
            try:
                return self._json(200, demo.translate(text, opts, self.env_file, self.data_dir, jobs_dir=self.jobs_dir))
            finally:
                _RUN_LOCK.release()
        except demo.DemoError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # a bug must show up as a readable message, not a hung page
            self._json(500, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})


def make_server(port: int = 8765, host: str = "127.0.0.1", env_file: str = ".env", data_dir: str = "data/demo", jobs_dir: str = "data/jobs",
                max_video_minutes: float = 30.0, runner: Optional[YouTubeRunner] = None) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise ValueError("The tester only binds to the local loopback address (127.0.0.1) so that nobody else on the network can use your API key.")

    class Bound(Handler):
        pass

    Bound.env_file, Bound.data_dir, Bound.jobs_dir = env_file, data_dir, jobs_dir
    Bound.runner = runner or YouTubeRunner(jobs_dir, max_video_minutes)
    srv = ThreadingHTTPServer((host, port), Bound)
    srv.daemon_threads = True
    return srv


def serve(port: int = 8765, env_file: str = ".env", data_dir: str = "data/demo", host: str = "127.0.0.1", jobs_dir: str = "data/jobs",
          max_video_minutes: float = 30.0) -> None:
    srv = make_server(port, host, env_file, data_dir, jobs_dir, max_video_minutes)
    print(f"ISL pipeline tester running at http://127.0.0.1:{srv.server_address[1]}/  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
