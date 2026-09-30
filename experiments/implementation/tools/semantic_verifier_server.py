"""Small, frozen Qwen3-VL service for a semantic-reward pilot.

The service is deliberately local-only. It returns categorical judgments; its
model-generated explanation is for auditing, never interpreted as confidence.
"""

import argparse
import base64
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


EVENT_TYPES = {"face_relation", "cross", "enter", "pass", "descend", "stop_near"}
VERB_PATTERNS = {
    "face_relation": r"\bleft\b.*\bright\b|\bright\b.*\bleft\b",
    "cross": r"\b(through|cross|exit through|out (of|the) door)\b",
    "enter": r"\b(enter|into|go in|walk in|step in)\b",
    "pass": r"\b(pass|past)\b",
    "descend": r"\b(down|descend)\b",
    "stop_near": r"\b(stop|wait|stand|pause)\b",
}


def extract_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("model did not return a JSON object")
    return json.loads(match.group(0))


def decode_image(value):
    image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    image.thumbnail((448, 448))
    return image


class Judge:
    def __init__(self, model_path):
        self.model_name = model_path.rstrip("/").rsplit("/", 1)[-1]
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True
        ).eval()
        self.lock = threading.Lock()
        self.parse_cache = {}

    def generate(self, content, max_new_tokens=180):
        messages = [{"role": "user", "content": content}]
        with self.lock:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens
                )
            trimmed = output[:, inputs["input_ids"].shape[-1]:]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def parse(self, instruction):
        if instruction in self.parse_cache:
            return self.parse_cache[instruction]
        prompt = (
            "Convert this English navigation instruction into a JSON object with an "
            "'events' array. Use at most 3 ordered, visually verifiable events. "
            "Each event has 'type', 'target', and 'source_text'. Allowed types: "
            "face_relation, cross, enter, pass, descend, stop_near. "
            "For face_relation, target describes the two objects and their left/right relation. "
            "source_text must be an exact substring of the instruction. "
            "Do not invent landmarks, objects, or events. Skip vague or unverifiable clauses. "
            "Return JSON only. Instruction: " + instruction
        )
        raw = self.generate([{"type": "text", "text": prompt}])
        events = []
        try:
            data = extract_json(raw)
            for item in data.get("events", [])[:3]:
                if not isinstance(item, dict):
                    continue
                typ, target, span = item.get("type"), item.get("target"), item.get("source_text")
                if typ not in EVENT_TYPES or not isinstance(target, str) or not target.strip():
                    continue
                if not isinstance(span, str) or span.casefold() not in instruction.casefold():
                    continue
                if not re.search(VERB_PATTERNS[typ], span.casefold()):
                    continue
                if typ == "descend" and not re.search(r"\b(stair|stairs|staircase|steps)\b", target.casefold()):
                    continue
                target_words = set(re.findall(r"[a-z]+", target.casefold())) - {
                    "the", "a", "an", "of", "to", "and", "left", "right", "near", "side"
                }
                source_words = set(re.findall(r"[a-z]+", span.casefold()))
                if target_words and not target_words.intersection(source_words):
                    continue
                events.append({"type": typ, "target": target.strip(), "source_text": span})
        except (ValueError, TypeError, AttributeError):
            pass
        result = {"events": events, "raw": raw, "model": self.model_name}
        self.parse_cache[instruction] = result
        return result

    def verify(self, event, before, after, actions, displacement, vertical_delta):
        typ = event.get("type")
        action_text = ", ".join(actions).lower()
        if typ in {"cross", "enter", "pass"} and (
            displacement < 0.10 or "move forward" not in action_text
        ):
            return {"status": "not_completed", "evidence": "motion gate"}
        if typ == "descend" and vertical_delta > -0.08:
            return {"status": "not_completed", "evidence": "motion gate"}
        if typ == "stop_near" and "stop" not in action_text:
            return {"status": "not_completed", "evidence": "STOP gate"}
        prompt = (
            "You are auditing one executed navigation event. Image A is BEFORE the actions; "
            "image B is AFTER them. Decide whether the specified event actually became true "
            "during these actions. Merely seeing a doorway or object does not prove crossing, "
            "entering, or passing it. If the images cannot establish the event, answer U. "
            "Answer exactly one letter and nothing else: Y means clearly completed, "
            "N means clearly not completed, U means uncertain. "
            f"Event type: {typ}. Event target: {event.get('target')}. "
            f"Instruction phrase: {event.get('source_text')}. "
            f"Executed actions: {action_text}. Displacement: {displacement:.2f} meters. "
            f"Vertical change: {vertical_delta:.2f} meters."
        )
        raw = self.generate([
            {"type": "text", "text": prompt + "\nImage A:"},
            {"type": "image", "image": decode_image(before)},
            {"type": "text", "text": "Image B:"},
            {"type": "image", "image": decode_image(after)},
        ], max_new_tokens=4)
        label = raw.strip().upper()
        status = {"Y": "completed", "N": "not_completed", "U": "uncertain"}.get(label, "uncertain")
        return {"status": status, "evidence": "forced-choice VLM label", "raw": raw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()
    judge = Judge(args.model_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.respond({"status": "ok"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 5_000_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/parse":
                    result = judge.parse(str(body["instruction"]))
                elif self.path == "/verify":
                    result = judge.verify(
                        body["event"], body["before"], body["after"],
                        body["actions"], float(body["displacement"]),
                        float(body["vertical_delta"])
                    )
                else:
                    self.send_error(404)
                    return
                self.respond(result)
            except Exception as exc:
                self.respond({"error": type(exc).__name__, "message": str(exc)}, 500)

        def respond(self, data, code=200):
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
