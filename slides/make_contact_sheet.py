"""Create a contact sheet from rendered slide PNGs."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PREVIEW_DIR = Path(__file__).resolve().parent / "preview"
OUTPUT = PREVIEW_DIR / "contact-sheet.png"


def main() -> None:
    slide_paths = sorted(PREVIEW_DIR.glob("slide-*.png"))
    if not slide_paths:
        raise RuntimeError("No slide preview images found.")

    thumb_w = 420
    thumb_h = 236
    pad_x = 24
    pad_y = 28
    cols = 3
    rows = (len(slide_paths) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad_x, rows * thumb_h + (rows + 1) * pad_y),
        "white",
    )

    for idx, path in enumerate(slide_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = pad_x + (idx % cols) * (thumb_w + pad_x)
        y = pad_y + (idx // cols) * (thumb_h + pad_y)
        sheet.paste(image, (x, y))

    sheet.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

