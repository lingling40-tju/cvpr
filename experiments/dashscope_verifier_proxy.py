"""Keep the original local parser while judging events with DashScope.

No credential or image is logged. The service binds to loopback only.
"""

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STATUS = {"Y": "completed", "N": "not_completed", "U": "uncertain"}
RETRY_CODES = {429, 500, 502, 503, 504}


def json_request(url, body, timeout, headers=None):
    request = Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class Judge:
    def __init__(self, model, api_url, key_file, parser_url, strict_transition=False,
                 reasoning_effort=None):
        self.model = model
        self.api_url = api_url.rstrip("/") + "/chat/completions"
        self.key = key_file.read_text().strip()
        assert self.key.startswith("sk-")
        self.parser_url = parser_url.rstrip("/")
        self.strict_transition = strict_transition
        self.reasoning_effort = reasoning_effort
        self.lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(8)
        self.counts = {name: 0 for name in (
            "parser_requests", "parser_errors", "verify_requests", "motion_gate_rejects",
            "stop_gate_rejects", "api_calls", "api_errors", "malformed_responses",
            "completed", "not_completed", "uncertain", "input_tokens", "output_tokens",
            "api_latency_ms",
        )}

    def add(self, **values):
        with self.lock:
            for name, value in values.items():
                self.counts[name] += value

    def stats(self):
        with self.lock:
            return {"model": self.model, "strict_transition": self.strict_transition,
                    "reasoning_effort": self.reasoning_effort, **self.counts}

    def parser_ok(self):
        try:
            with urlopen(self.parser_url + "/health", timeout=2) as response:
                return response.status == 200
        except (URLError, TimeoutError):
            return False

    def parse(self, instruction):
        self.add(parser_requests=1)
        try:
            result = json_request(self.parser_url + "/parse", {"instruction": instruction}, 120)
            if "error" in result:
                raise RuntimeError("local parser returned an error")
            return result
        except Exception:
            self.add(parser_errors=1)
            raise

    def complete(self, prompt, before, after):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt + "\nImage A:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + before}},
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + after}},
            ]}],
            "temperature": 0,
            "max_tokens": 256 if self.reasoning_effort else 16,
            "enable_thinking": bool(self.reasoning_effort),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Authorization": "Bearer " + self.key}
        with self.semaphore:
            for attempt in range(4):
                started = time.monotonic()
                self.add(api_calls=1)
                try:
                    result = json_request(self.api_url, body, 120, headers)
                    elapsed = round((time.monotonic() - started) * 1000)
                    usage = result.get("usage") or {}
                    self.add(
                        api_latency_ms=elapsed,
                        input_tokens=int(usage.get("prompt_tokens", 0)),
                        output_tokens=int(usage.get("completion_tokens", 0)),
                    )
                    return result["choices"][0]["message"].get("content") or "", usage
                except HTTPError as exc:
                    self.add(api_errors=1)
                    if exc.code not in RETRY_CODES or attempt == 3:
                        raise RuntimeError(f"DashScope HTTP {exc.code}") from None
                except (URLError, TimeoutError) as exc:
                    self.add(api_errors=1)
                    if attempt == 3:
                        raise RuntimeError(f"DashScope network {type(exc).__name__}") from None
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def verify(self, event, before, after, actions, displacement, vertical_delta):
        self.add(verify_requests=1)
        typ = event.get("type")
        action_text = ", ".join(actions).lower()
        if typ in {"cross", "enter", "pass"} and (
            displacement < 0.10 or "move forward" not in action_text
        ):
            self.add(motion_gate_rejects=1, not_completed=1)
            return {"status": "not_completed", "evidence": "motion gate", "model": self.model}
        if typ == "descend" and vertical_delta > -0.08:
            self.add(motion_gate_rejects=1, not_completed=1)
            return {"status": "not_completed", "evidence": "motion gate", "model": self.model}
        if typ == "stop_near" and "stop" not in action_text:
            self.add(stop_gate_rejects=1, not_completed=1)
            return {"status": "not_completed", "evidence": "STOP gate", "model": self.model}
        strict_clause = (
            "For Y, require clear evidence that the event was not fulfilled in image A "
            "and became fulfilled in image B. If the event was already fulfilled in A, "
            "the target is still ahead in B, or the transition cannot be localized, "
            "answer U unless non-completion is clear (then N). "
        ) if self.strict_transition else ""
        prompt = (
            "You are auditing one executed navigation event. Image A is BEFORE the actions; "
            "image B is AFTER them. Decide whether the specified event actually became true "
            "during these actions. Merely seeing a doorway or object does not prove crossing, "
            "entering, or passing it. If the images cannot establish the event, answer U. "
            "Answer exactly one letter and nothing else: Y means clearly completed, "
            "N means clearly not completed, U means uncertain. "
        ) + strict_clause + (
            f"Event type: {typ}. Event target: {event.get('target')}. "
            f"Instruction phrase: {event.get('source_text')}. "
            f"Executed actions: {action_text}. Displacement: {displacement:.2f} meters. "
            f"Vertical change: {vertical_delta:.2f} meters."
        )
        raw, usage = self.complete(prompt, before, after)
        label = raw.strip().upper()
        if not re.fullmatch(r"[YNU]", label):
            self.add(malformed_responses=1)
            label = "U"
        status = STATUS[label]
        self.add(**{status: 1})
        return {"status": status, "evidence": "forced-choice DashScope label",
                "raw": raw, "model": self.model, "usage": usage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.8-max-0902")
    parser.add_argument("--api-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--parser-url", default="http://127.0.0.1:5003")
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--strict-transition", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"))
    args = parser.parse_args()
    judge = Judge(args.model, args.api_url, args.key_file, args.parser_url,
                  strict_transition=args.strict_transition,
                  reasoning_effort=args.reasoning_effort)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *values):
            pass

        def respond(self, body, code=200):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                return self.respond({"status": "ok", "model": judge.model,
                                     "strict_transition": judge.strict_transition,
                                     "reasoning_effort": judge.reasoning_effort,
                                     "parser_ok": judge.parser_ok()})
            if self.path == "/stats":
                return self.respond(judge.stats())
            self.send_error(404)

        def do_POST(self):
            if self.path not in {"/parse", "/verify"}:
                return self.send_error(404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 5_000_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/parse":
                    result = judge.parse(str(body["instruction"]))
                else:
                    result = judge.verify(
                        body["event"], body["before"], body["after"], body["actions"],
                        float(body["displacement"]), float(body["vertical_delta"]),
                    )
                self.respond(result)
            except Exception as exc:
                self.respond({"error": type(exc).__name__, "message": str(exc)}, 500)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
