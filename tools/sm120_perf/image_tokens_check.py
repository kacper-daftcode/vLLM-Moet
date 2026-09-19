#!/usr/bin/env python3
"""Image token count of a served DeepSeek-V4.1 stack vs the checkpoint's reference image processor.

The formula below is `inference/image_processor.py` from the DeepSeek-V4.1-Flash checkpoint
(patch 14, downsample 3, max 1024 tokens, min 544x544 pixels): tokens = n_llm_h * (n_llm_w + 1) + 2.
Synthetic images of several sizes are sent through /v1/chat/completions and the server's prompt
token count (minus the text-only count) is compared with the formula.

usage: image_tokens_check.py BASE MODEL [API_KEY]
"""
import base64
import io
import json
import math
import sys
import urllib.request

from PIL import Image, ImageDraw

base, model, key = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
P, DS, MAX_TOK, MIN_PIX = 14, 3, 1024, 544 * 544


def num_tokens(h, w):
    return h * (w + 1) + 2


def llm_grid(bh, bw):
    return math.ceil((bh // P) / DS), math.ceil((bw // P) / DS)


def solve(height, width):
    r = height / width
    max_w = math.sqrt((MAX_TOK - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = P * DS
    if max_w < 1:
        return (MAX_TOK - 2) // 2 * cell, cell
    if max_h < 1:
        return cell, (MAX_TOK - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / P) * P, math.floor(width * beta / P) * P


def ref(width, height):
    if 0 < width * height < MIN_PIX:
        ratio = (MIN_PIX / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    bw, bh = math.ceil(width / P) * P, math.ceil(height / P) * P
    h, w = llm_grid(bh, bw)
    if num_tokens(h, w) > MAX_TOK:
        bh, bw = solve(height, width)
        h, w = llm_grid(bh, bw)
    return num_tokens(h, w)


def chat(content):
    body = dict(model=model, messages=[dict(role="user", content=content)], max_tokens=4, temperature=0,
                chat_template_kwargs={"thinking": False})
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["usage"]


text = "Describe the image in one word."
t0 = chat(text)["prompt_tokens"]
for (w, h) in [(3024, 588), (1024, 1024), (640, 480), (1920, 1080), (300, 200), (4000, 3000)]:
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).ellipse((w // 4, h // 4, w // 2, h // 2), fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    u = chat([{"type": "text", "text": text},
              {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}])
    det = (u.get("prompt_tokens_details") or {}).get("image_tokens")
    print(f"{w}x{h}: prompt {u['prompt_tokens']} (text-only {t0}) -> image tokens ~{u['prompt_tokens'] - t0}"
          f" (server detail: {det}); reference {ref(w, h)}")
