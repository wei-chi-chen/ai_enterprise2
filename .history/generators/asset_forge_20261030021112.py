import math
import wave
import struct
import random
from pathlib import Path
from PIL import Image, ImageDraw

class AutonomousAssetForge:
    """自主素材生成引擎：自動合成 2D 像素角色、地圖磚塊與 WAV 遊戲音效"""

    @staticmethod
    def generate_sprite_png(save_path: Path, color_hex: str = "#38bdf8", shape: str = "hero", size: int = 64):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if shape == "hero":
            draw.rectangle([size*0.25, size*0.2, size*0.75, size*0.8], fill=color_hex, outline="#ffffff", width=2)
            draw.ellipse([size*0.35, size*0.3, size*0.45, size*0.4], fill="#ffffff")
            draw.ellipse([size*0.55, size*0.3, size*0.65, size*0.4], fill="#ffffff")
        elif shape == "enemy":
            draw.polygon([(size*0.5, size*0.1), (size*0.9, size*0.85), (size*0.1, size*0.85)], fill=color_hex, outline="#ff0000")
        elif shape == "coin":
            draw.ellipse([size*0.2, size*0.2, size*0.8, size*0.8], fill="#facc15", outline="#ca8a04", width=3)
        else:
            draw.rounded_rectangle([4, 4, size-4, size-4], radius=8, fill=color_hex)

        img.save(save_path, "PNG")
        return str(save_path)

    @staticmethod
    def generate_sfx_wav(save_path: Path, sfx_type: str = "laser", duration: float = 0.25):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 44100
        n_samples = int(sample_rate * duration)

        with wave.open(str(save_path), "w") as wav_file:
            wav_file.setparams((1, 2, sample_rate, n_samples, "NONE", "not compressed"))
            for i in range(n_samples):
                t = i / sample_rate
                if sfx_type == "laser":
                    freq = 880.0 - (600.0 * (t / duration))
                    val = 0.5 * math.sin(2 * math.pi * freq * t)
                elif sfx_type == "coin":
                    freq = 987.77 if t < duration / 2 else 1318.51
                    val = 0.5 * math.sin(2 * math.pi * freq * t)
                elif sfx_type == "explosion":
                    val = random.uniform(-0.6, 0.6) * (1.0 - (t / duration))
                else:  # jump
                    freq = 300.0 + (400.0 * (t / duration))
                    val = 0.5 * math.sin(2 * math.pi * freq * t)

                packed = struct.pack("<h", int(max(min(val, 1.0), -1.0) * 32767.0))
                wav_file.writeframesraw(packed)
        return str(save_path)

    def fulfill_asset_manifest(self, target_dir: Path, manifest: list) -> list:
        """根據模型開出的素材清單自動生成所有檔案"""
        created = []
        for item in manifest:
            atype = item.get("type", "sprite")
            fname = item.get("filename", "asset.png")
            fpath = target_dir / fname
            if atype == "sfx":
                self.generate_sfx_wav(fpath, sfx_type=item.get("style", "laser"))
            else:
                self.generate_sprite_png(
                    fpath,
                    color_hex=item.get("color", "#38bdf8"),
                    shape=item.get("style", "hero")
                )
            created.append(fname)
        return created