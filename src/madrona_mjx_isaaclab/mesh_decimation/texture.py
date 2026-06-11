from __future__ import annotations

from pathlib import Path


def downsample_texture(src, max_edge: int, out_path=None):
    from PIL import Image

    if isinstance(src, (str, Path)):
        img = Image.open(str(src))
    else:
        img = src

    if img.width <= max_edge and img.height <= max_edge:
        if out_path is not None:
            img.save(str(out_path))
        return img

    ratio = min(max_edge / img.width, max_edge / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if out_path is not None:
        img.save(str(out_path))

    return img
