import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from isl_gloss.llm import LLMResponse
from isl_gloss.models import SourceChunk
from isl_gloss.prompt import CONTEXT_HEADER


def chunk(cid, text, **kw):
    return SourceChunk(chunk_id=cid, text=text, **kw)


def parse_user(user: str):
    """Split the user message into (context sentences, [(label, sentence), ...]).
    Prompt 6.0 lines look like "[sentence_id] text" (label is the id string); 5.0 lines like "1. text" (label is an int)."""
    context, body = [], user
    if user.startswith(CONTEXT_HEADER):
        head, _, body = user.partition("\n\n")
        context = head.split("\n")[1:]
    if "\n\nFollowing speech" in body:
        body = body.split("\n\nFollowing speech")[0]
    items = []
    for line in body.split("\n"):
        m = re.match(r"\[([^\]]+)\] (.*)$", line)
        if m:
            items.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"(\d+)\. (.*)$", line)
        if m:
            items.append((int(m.group(1)), m.group(2)))
    return context, items


def results_json(rows):
    return json.dumps({"results": rows})


def make_row(label, text, tokens, needs=False, parts=None, notes="", echo=True, status=None):
    """One reply row in the format that matches the label: id string -> prompt 6.0 row, int -> prompt 5.0 row."""
    if isinstance(label, str):
        return {"sentence_id": label, "source_text": text if echo else "", "candidate_gloss": list(tokens),
                "uncertain_parts": list(parts or []), "uncertainty_notes": notes, "needs_human_review": needs}
    return {"n": label, "gloss": list(tokens), "status": status or ("uncertain" if needs else "ok"),
            "uncertain_parts": list(parts or []), "notes": notes}


def echo_reply(user: str) -> str:
    """What a well-behaved model would return: upper-cased words of each sentence as the gloss (either prompt format)."""
    _, items = parse_user(user)
    return results_json([make_row(label, text, re.findall(r"[A-Za-z0-9]+", text.upper())) for label, text in items])


def reply_from(fn):
    """Scripted reply: fn(label, text) -> list of tokens (or a ready-made row dict) for every sentence in the request."""
    def reply(user):
        rows = []
        for label, text in parse_user(user)[1]:
            got = fn(label, text)
            rows.append(got if isinstance(got, dict) else make_row(label, text, got))
        return results_json(rows)
    return reply


def label_of(chunk_id: str, position: int = 1):
    """The label the model is given for a sentence under the default prompt (the real id)."""
    return chunk_id


class FakeClient:
    """Scripted LLM client. Each entry in `script` is a callable(user) -> str, an Exception to raise, or a str.
    When the script runs out it echoes."""

    model = "fake-model"

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []
        self.requests_made = 0

    def complete(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        self.requests_made += 1
        step = self.script.pop(0) if self.script else echo_reply
        if isinstance(step, Exception):
            raise step
        text = step(user) if callable(step) else step
        return LLMResponse(text=text, model=self.model, finish_reason="STOP")


class MockGemini:
    """A real HTTP server speaking just enough of the generateContent protocol."""

    def __init__(self, script=None):
        self.script = list(script or [])  # entries: (status, body_dict_or_str) ; empty => echo success
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if outer.script:
                    status, payload = outer.script.pop(0)
                else:
                    user = body["contents"][0]["parts"][0]["text"]
                    status, payload = 200, {
                        "candidates": [{"content": {"parts": [{"text": echo_reply(user)}]}, "finishReason": "STOP"}],
                        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
                        "modelVersion": "mock-1",
                    }
                raw = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw.encode("utf-8"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1beta"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class MockGroq:
    """A real HTTP server speaking just enough of Groq's OpenAI-compatible chat-completions protocol.
    script entries: (status, body_dict_or_str) or (status, body, {header: value}); empty script => echo success."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                headers = {}
                if outer.script:
                    entry = outer.script.pop(0)
                    status, payload = entry[0], entry[1]
                    headers = entry[2] if len(entry) > 2 else {}
                else:
                    user = body["messages"][-1]["content"]
                    status, payload = 200, {
                        "id": "chatcmpl-test", "object": "chat.completion", "model": "openai/gpt-oss-120b",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": echo_reply(user)},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    }
                raw = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw.encode("utf-8"))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/openai/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
