import math
import wave
import struct
import random
from pathlib import Path
from PIL import Image, ImageDraw

class AutonomousAssetForge:
    """自主素材生成引擎：自動合成 2D 像素角色、堡壘、砲台與 WAV 音效"""

    @staticmethod
    def generate_sprite_png(save_path: Path, color_hex: str = "#38bdf8", shape: str = "hero", size: int = 64):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if not color_hex.startswith("#") or len(color_hex) not in [4, 7]:
            color_hex = "#38bdf8"

        if "barricade" in shape or "base" in shape or "tower" in shape:
            draw.rectangle([size*0.15, size*0.1, size*0.85, size*0.9], fill=color_hex, outline="#e2e8f0", width=3)
            draw.rectangle([size*0.3, size*0.5, size*0.7, size*0.9], fill="#1e293b")
        elif "enemy" in shape or "boss" in shape:
            draw.polygon([(size*0.5, size*0.08), (size*0.92, size*0.88), (size*0.08, size*0.88)], fill=color_hex, outline="#ffffff")
            draw.ellipse([size*0.35, size*0.45, size*0.45, size*0.55], fill="#000000")
            draw.ellipse([size*0.55, size*0.45, size*0.65, size*0.55], fill="#000000")
        elif "turret" in shape or "cannon" in shape:
            draw.ellipse([size*0.2, size*0.3, size*0.8, size*0.9], fill=color_hex, outline="#ffffff", width=2)
            draw.rectangle([size*0.5, size*0.45, size*0.95, size*0.6], fill="#94a3b8")
        elif "coin" in shape or "gold" in shape:
            draw.ellipse([size*0.2, size*0.2, size*0.8, size*0.8], fill="#facc15", outline="#ca8a04", width=3)
        else:
            draw.rectangle([size*0.25, size*0.2, size*0.75, size*0.85], fill=color_hex, outline="#ffffff", width=2)
            draw.ellipse([size*0.35, size*0.3, size*0.45, size*0.4], fill="#ffffff")
            draw.ellipse([size*0.55, size*0.3, size*0.65, size*0.4], fill="#ffffff")

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
                if "laser" in sfx_type or "shoot" in sfx_type or "attack" in sfx_type:
                    freq = 880.0 - (600.0 * (t / duration))
                    val = 0.5 * math.sin(2 * math.pi * freq * t)
                elif "coin" in sfx_type or "gold" in sfx_type or "evolve" in sfx_type:
                    freq = 987.77 if t < duration / 2 else 1318.51
                    val = 0.5 * math.sin(2 * math.pi * freq * t)
                elif "explosion" in sfx_type or "hit" in sfx_type or "super" in sfx_type:
                    val = random.uniform(-0.6, 0.6) * (1.0 - (t / duration))
                else:
                    freq = 320.0 + (380.0 * (t / duration))
                    val = 0.5 * math.sin(2 * math.pi * freq * t)

                packed = struct.pack("<h", int(max(min(val, 1.0), -1.0) * 32767.0))
                wav_file.writeframesraw(packed)
        return str(save_path)

    def fulfill_asset_manifest(self, target_dir: Path, manifest: list) -> list:
        created = []
        for item in manifest:
            atype = str(item.get("type", "sprite")).lower()
            fname = Path(item.get("filename", "asset.png")).name
            fpath = target_dir / fname
            if atype in ["sfx", "audio", "sound"] or fname.endswith((".wav", ".mp3")):
                if fname.endswith(".mp3"):
                    fname = fname.replace(".mp3", ".wav")
                    fpath = target_dir / fname
                self.generate_sfx_wav(fpath, sfx_type=str(item.get("style", fname)))
            else:
                self.generate_sprite_png(
                    fpath,
                    color_hex=str(item.get("color", "#38bdf8")),
                    shape=str(item.get("style", fname))
                )
            created.append(fname)
        return created