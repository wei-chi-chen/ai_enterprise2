import json
import math
import re
from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw, ImageFont, ImageFilter


class AutonomousImageForge:
    """Render a model-authored, self-contained illustration as a high-resolution PNG."""

    @staticmethod
    def quality_size(iterations: int) -> int:
        return min(2048, 1024 + max(0, iterations - 1) * 128)

    @staticmethod
    def validate_png(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, ValueError):
            return False

    @staticmethod
    def _color(value, default):
        value = str(value or default).strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", value):
            return default
        return value

    @staticmethod
    def _font(size: int, bold: bool = False):
        candidates = [
            r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def render(self, target_dir: Path, specification: Dict, iterations: int = 1) -> Dict[str, str]:
        target_dir = Path(target_dir)
        if target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing image delivery: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

        size = self.quality_size(iterations)
        top = self._color(specification.get("background_top"), "#0f172a")
        bottom = self._color(specification.get("background_bottom"), "#312e81")
        image = Image.new("RGB", (size, size))
        pixels = image.load()
        top_rgb = Image.new("RGB", (1, 1), top).getpixel((0, 0))
        bottom_rgb = Image.new("RGB", (1, 1), bottom).getpixel((0, 0))
        background_draw = ImageDraw.Draw(image)
        for y in range(size):
            blend = y / max(1, size - 1)
            color = tuple(round(top_rgb[i] * (1 - blend) + bottom_rgb[i] * blend) for i in range(3))
            background_draw.line((0, y, size, y), fill=color)

        draw = ImageDraw.Draw(image, "RGBA")
        shapes = specification.get("shapes", [])
        if not isinstance(shapes, list):
            shapes = []
        for index, shape in enumerate(shapes[:48]):
            if not isinstance(shape, dict):
                continue
            kind = str(shape.get("type", "circle")).lower()
            color = self._color(shape.get("color"), ["#38bdf8", "#f472b6", "#facc15"][index % 3])
            x = max(0.0, min(1.0, float(shape.get("x", 0.5))))
            y = max(0.0, min(1.0, float(shape.get("y", 0.5))))
            width = max(0.01, min(1.0, float(shape.get("width", 0.2))))
            height = max(0.01, min(1.0, float(shape.get("height", width))))
            box = (
                round((x - width / 2) * size),
                round((y - height / 2) * size),
                round((x + width / 2) * size),
                round((y + height / 2) * size),
            )
            points = shape.get("points", [])
            vector_points = []
            if isinstance(points, list):
                for point in points[:128]:
                    if (
                        isinstance(point, (list, tuple)) and len(point) >= 2
                        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point[:2])
                    ):
                        vector_points.append((
                            round(max(0.0, min(1.0, point[0])) * size),
                            round(max(0.0, min(1.0, point[1])) * size),
                        ))
            if kind in {"triangle", "star"} and not vector_points:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                radius_x, radius_y = max(1, (box[2] - box[0]) / 2), max(1, (box[3] - box[1]) / 2)
                count = 3 if kind == "triangle" else 10
                vector_points = [
                    (
                        round(cx + math.cos(-math.pi / 2 + math.pi * index / (count / 2)) * radius_x
                              * (0.45 if kind == "star" and index % 2 else 1)),
                        round(cy + math.sin(-math.pi / 2 + math.pi * index / (count / 2)) * radius_y
                              * (0.45 if kind == "star" and index % 2 else 1)),
                    )
                    for index in range(count)
                ]
            if kind == "path" and len(vector_points) < 2:
                continue
            mask = Image.new("L", (size, size))
            mask_draw = ImageDraw.Draw(mask)
            if kind in {"ellipse", "circle"}:
                mask_draw.ellipse(box, fill=255)
            elif kind in {"rect", "rectangle"}:
                mask_draw.rounded_rectangle(box, radius=size // 48, fill=255)
            elif kind in {"triangle", "star", "polygon", "path", "freeform"} and len(vector_points) >= 3:
                mask_draw.polygon(vector_points, fill=255)
            elif kind == "line" and len(vector_points) >= 2:
                mask_draw.line(vector_points, fill=255, width=max(2, size // 128), joint="curve")

            if shape.get("gradient_end") and kind != "line":
                end_color = self._color(shape["gradient_end"], color)
                start_rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
                end_rgb = Image.new("RGB", (1, 1), end_color).getpixel((0, 0))
                bounds = mask.getbbox()
                if bounds:
                    left, top, right, bottom = bounds
                    gradient = Image.new("RGB", (right - left, bottom - top))
                    gradient_draw = ImageDraw.Draw(gradient)
                    for y_pos in range(top, bottom):
                        blend = (y_pos - top) / max(1, bottom - top - 1)
                        shade = tuple(
                            round(start_rgb[channel] * (1 - blend) + end_rgb[channel] * blend)
                            for channel in range(3)
                        )
                        gradient_draw.line(
                            (0, y_pos - top, right - left, y_pos - top), fill=shade
                        )
                    image.paste(gradient, (left, top), mask.crop(bounds))
            else:
                image.paste(color, (0, 0, size, size), mask)

            if shape.get("glow") and kind != "line":
                glow_mask = mask.filter(ImageFilter.GaussianBlur(max(2, size // 32)))
                glow = Image.new("RGBA", (size, size), color=color)
                glow.putalpha(glow_mask.point(lambda alpha: alpha // 3))
                image.paste(glow, (0, 0), glow)
            if kind == "line":
                line_points = vector_points if len(vector_points) >= 2 else [
                    (box[0], box[1]), (box[2], box[3])
                ]
                draw.line(line_points, fill=color, width=max(2, size // 128), joint="curve")
            elif kind in {"ellipse", "circle"}:
                draw.ellipse(box, outline="#ffffffb0", width=max(1, size // 512))
            elif kind in {"rect", "rectangle"}:
                draw.rounded_rectangle(box, radius=size // 48, outline="#ffffffa0", width=max(1, size // 512))
            elif len(vector_points) >= 2:
                draw.line(vector_points + ([vector_points[0]] if shape.get("closed", True) else []),
                          fill="#ffffffb0", width=max(1, size // 512), joint="curve")

        title = str(specification.get("title", "")).strip()[:100]
        subtitle = str(specification.get("subtitle", "")).strip()[:240]
        if title or subtitle:
            draw.rectangle((0, size * 0.76, size, size), fill="#020617d9")
            draw.text((size * 0.07, size * 0.79), title, font=self._font(size // 18, True), fill="#ffffff")
            draw.text((size * 0.07, size * 0.89), subtitle, font=self._font(size // 40), fill="#dbeafe")

        image_path = target_dir / "image.png"
        image.save(image_path, "PNG", optimize=True)
        (target_dir / "art_direction.json").write_text(
            json.dumps(specification, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"image": str(image_path), "specification": str(target_dir / "art_direction.json")}
