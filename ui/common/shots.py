"""Screenshot helpers for the measurements: downscale and palette-quantise PNGs to stay below a size limit."""
import io
from pathlib import Path

from PIL import Image


def save_small(png: bytes | Image.Image, path: Path, max_kb: int = 290, width: int = 1100) -> int:
    """Write a PNG of at most max_kb (downscaled to `width`, then 256 / 128 / 64 colours); returns bytes written."""
    img = png if isinstance(png, Image.Image) else Image.open(io.BytesIO(png))
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for colours in (None, 256, 128, 64, 32):
        im = img if colours is None else img.quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        if buf.tell() <= max_kb * 1024 or colours == 32:
            path.write_bytes(buf.getvalue())
            return buf.tell()
    return 0
