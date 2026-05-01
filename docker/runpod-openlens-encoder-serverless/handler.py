from __future__ import annotations

import runpod

from openlens.serverless_encoder import handle_event


def handler(event):
    return handle_event(event)


runpod.serverless.start({"handler": handler})
