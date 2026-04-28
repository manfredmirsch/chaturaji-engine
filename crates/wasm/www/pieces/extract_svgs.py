#!/usr/bin/env python3
import re
import base64
import os

css_file = os.path.join(os.path.dirname(__file__), "pieces.css")
out_dir = os.path.join(os.path.dirname(__file__), "svg")
os.makedirs(out_dir, exist_ok=True)

with open(css_file, "r", encoding="utf-8") as f:
    css = f.read()

# Match selector block + base64 data together
# Each block looks like:
#   .piece[data-piece="X"][data-color="Y"]:... {
#     background-image: url(data:image/svg+xml;base64,<DATA>);
#   }
pattern = re.compile(
    r'\.piece\[data-piece="([^"]+)"\]\[data-color="([^"]+)"\][^{]*\{[^}]*'
    r'background-image:\s*url\(data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)\)',
    re.DOTALL
)

seen = {}
count = 0

for m in pattern.finditer(css):
    piece, color, b64 = m.group(1), m.group(2), m.group(3)
    key = (piece, color)
    if key in seen:
        continue
    seen[key] = True

    svg_bytes = base64.b64decode(b64)
    # sanitize piece name for filesystem
    safe_piece = piece.replace("/", "_").replace("\\", "_")
    filename = f"{safe_piece}_{color}.svg"
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(svg_bytes)
    count += 1

print(f"Extracted {count} SVG files to {out_dir}/")
