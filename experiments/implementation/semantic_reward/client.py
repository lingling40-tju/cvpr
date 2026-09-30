"""Local-only HTTP client for the frozen semantic event judge."""

import base64
import io
import json
from urllib.request import Request, urlopen


def encode_image(image):
    image = image.copy().convert("RGB")
    image.thumbnail((448, 448))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=78)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class SemanticClient:
    def __init__(self, base_url, timeout=90):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path, body):
        request = Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result)
        return result

    def parse(self, instruction):
        return self._post("/parse", {"instruction": instruction})

    def verify(self, event, before, after, actions, displacement, vertical_delta):
        return self._post("/verify", {
            "event": event,
            "before": encode_image(before),
            "after": encode_image(after),
            "actions": actions,
            "displacement": displacement,
            "vertical_delta": vertical_delta,
        })
