import html
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import random
import shutil
import struct
import subprocess
import tempfile
import textwrap
import wave
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg


class AutonomousFilmForge:
    """Render a local, captioned motion-graphics short film from a screenplay."""

    WIDTH = 960
    HEIGHT = 540
    FPS = 24
    MAX_SCENES = 12
    FRAME_WORKERS = 4

    @staticmethod
    def fallback_screenplay(title: str, premise: str) -> Dict:
        premise = premise.strip()[:400] or title
        is_chinese = any("\u4e00" <= char <= "\u9fff" for char in premise)
        story = (
            [
                ("故事的開端", f"以「{premise}」為靈感的朦朧開場。", f"每個故事都從一個問題開始：{premise}", "mysterious"),
                ("迎面而來的挑戰", "旅人獨自面對遼闊而陌生的世界。", "前方的路，比任何人想像的都更艱難。", "tense"),
                ("命運的轉折", "風暴漸歇，遠方出現一道微光。", "在最黑暗的時刻，新的選擇終於浮現。", "hopeful"),
                ("嶄新的地平線", "晨光灑落地平線，世界逐漸甦醒。", "一個微小的行動，也能改變明天。", "epic"),
            ]
            if is_chinese
            else [
                ("The Beginning", f"An atmospheric opening inspired by: {premise}", f"Every story begins with a question: {premise}", "mysterious"),
                ("The Challenge", "A lone traveler faces a vast, unfamiliar landscape.", "The path ahead is harder than anyone imagined.", "tense"),
                ("The Turning Point", "A distant light appears as the storm begins to break.", "In the darkest moment, a different choice becomes possible.", "hopeful"),
                ("A New Horizon", "The first light of dawn spreads across the horizon.", "One small act can change the shape of tomorrow.", "epic"),
            ]
        )
        return {
            "title": title,
            "logline": premise,
            "scenes": [
                {"heading": heading, "visual": visual, "narration": narration, "mood": mood, "duration_sec": 4}
                for heading, visual, narration, mood in story
            ],
        }

    @staticmethod
    def quality_profile(iterations: int) -> Dict[str, int]:
        if iterations >= 5:
            return {"width": 1920, "height": 1080, "fps": 30, "crf": 17}
        if iterations >= 3:
            return {"width": 1280, "height": 720, "fps": 24, "crf": 18}
        return {"width": 960, "height": 540, "fps": 24, "crf": 20}

    def render_film(self, target_dir: Path, film: Dict, iterations: int = 1) -> Dict[str, str]:
        scenes = self._normalize_scenes(film.get("scenes", []))
        if not scenes:
            raise ValueError("A film needs at least one scene.")

        if target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing film delivery: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        quality = self.quality_profile(iterations)
        output_size = (quality["width"], quality["height"])
        title = str(film.get("title") or "Untitled Film").strip()[:160]
        logline = str(film.get("logline") or "").strip()[:1000]
        poster_path = target_dir / "poster.png"
        storyboard_path = target_dir / "storyboard.png"
        screenplay_path = target_dir / "screenplay.md"
        soundtrack_path = target_dir / "soundtrack.wav"
        narration_path = target_dir / "narration.wav"
        movie_path = target_dir / "film.mp4"
        silent_path = target_dir / ".film_silent.mp4"
        mux_path = target_dir / ".film_final.mp4"
        mixed_audio_path = target_dir / ".film_mixed.wav"

        has_narration = self._synthesize_narration(narration_path, scenes)
        total_duration = sum(scene["duration_sec"] for scene in scenes)
        total_frames = total_duration * quality["fps"]
        backgrounds = [self._draw_scene(scene, index, title) for index, scene in enumerate(scenes)]
        poster = backgrounds[0].copy()
        poster_draw = ImageDraw.Draw(poster, "RGBA")
        poster_draw.rectangle((0, 0, self.WIDTH, 105), fill=(2, 6, 16, 180))
        poster_draw.text((42, 24), textwrap.shorten(title, width=48, placeholder="…"),
                         font=self._font(32, True), fill=(249, 246, 238, 255))
        if logline:
            poster_draw.text((44, 70), textwrap.shorten(logline, width=84, placeholder="…"),
                             font=self._font(16), fill=(205, 218, 231, 255))
        if output_size != poster.size:
            poster = poster.resize(output_size, Image.Resampling.LANCZOS)
        poster.save(poster_path, "PNG", optimize=True)
        self._write_storyboard(storyboard_path, scenes, backgrounds, title)
        screenplay_path.write_text(self._write_screenplay(title, logline, scenes), encoding="utf-8")
        self._write_soundtrack(soundtrack_path, scenes, total_frames / quality["fps"])

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{output_size[0]}x{output_size[1]}",
            "-framerate", str(quality["fps"]), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(quality["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(silent_path)
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self._write_ordered_frames(
                process.stdin, scenes, backgrounds, output_size, quality["fps"]
            )
            process.stdin.close()
            error = process.stderr.read().decode("utf-8", errors="replace")
            process.stderr.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Could not encode the film video: {error.strip()}")

            audio_path = soundtrack_path
            if has_narration:
                mix = subprocess.run(
                    [
                        ffmpeg, "-y", "-loglevel", "error",
                        "-i", str(soundtrack_path), "-i", str(narration_path),
                        "-filter_complex",
                        "[0:a]volume=0.28[music];[1:a]volume=1.0[voice];"
                        "[music][voice]amix=inputs=2:duration=first:dropout_transition=2,"
                        "alimiter=limit=0.95[a]",
                        "-map", "[a]", "-c:a", "pcm_s16le", str(mixed_audio_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if mix.returncode:
                    raise RuntimeError(f"Could not mix the film narration: {mix.stderr.strip()}")
                audio_path = mixed_audio_path
            mux_command = [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", str(silent_path), "-i", str(audio_path),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "160k", "-shortest",
                "-movflags", "+faststart", str(mux_path)
            ]
            mux = subprocess.run(mux_command, capture_output=True, text=True, timeout=180)
            if mux.returncode:
                raise RuntimeError(f"Could not add the film soundtrack: {mux.stderr.strip()}")
            mux_path.replace(movie_path)
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            silent_path.unlink(missing_ok=True)
            mux_path.unlink(missing_ok=True)
            mixed_audio_path.unlink(missing_ok=True)

        self._write_viewer(target_dir, title, logline, scenes, has_narration)
        return {
            "video": str(movie_path),
            "viewer": str(target_dir / "index.html"),
            "poster": str(poster_path),
            "storyboard": str(storyboard_path),
            "screenplay": str(screenplay_path),
            "soundtrack": str(soundtrack_path),
            "narration": str(narration_path) if has_narration else "",
            "duration_sec": str(total_frames // quality["fps"]),
            "width": str(output_size[0]),
            "height": str(output_size[1]),
            "fps": str(quality["fps"]),
        }

    def _render_frame_bytes(
        self, background: Image.Image, scene: Dict, scene_index: int,
        local_frame: int, frame_count: int, output_size,
    ) -> bytes:
        image = self._render_frame(background, scene, scene_index, local_frame, frame_count)
        if output_size != image.size:
            image = image.resize(output_size, Image.Resampling.LANCZOS)
        return image.tobytes()

    def _write_ordered_frames(self, output, scenes, backgrounds, output_size, fps):
        jobs = (
            (scene_index, scene, local_frame, scene["duration_sec"] * fps)
            for scene_index, scene in enumerate(scenes)
            for local_frame in range(scene["duration_sec"] * fps)
        )
        total_frames = sum(scene["duration_sec"] * fps for scene in scenes)
        workers = min(self.FRAME_WORKERS, total_frames)
        if workers <= 1:
            for scene_index, scene, local_frame, frame_count in jobs:
                output.write(self._render_frame_bytes(
                    backgrounds[scene_index], scene, scene_index, local_frame,
                    frame_count, output_size,
                ))
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = deque()

            def submit_next():
                try:
                    scene_index, scene, local_frame, frame_count = next(jobs)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._render_frame_bytes, backgrounds[scene_index], scene,
                    scene_index, local_frame, frame_count, output_size,
                )
                pending.append(future)
                return True

            for _ in range(min(total_frames, workers * 2)):
                submit_next()
            while pending:
                output.write(pending.popleft().result())
                submit_next()

    @staticmethod
    def _synthesize_narration(path: Path, scenes: List[Dict]) -> bool:
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        narration_scenes = [
            (index, str(scene.get("narration", "")).strip())
            for index, scene in enumerate(scenes)
            if str(scene.get("narration", "")).strip()
        ]
        if not powershell or not narration_scenes:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        script = r"""
$ErrorActionPreference = 'Stop'
$raw = [Console]::In.ReadToEnd()
$request = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($raw)) | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $synth.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'zh*' } | Select-Object -First 1
if (-not $voice) { $voice = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | Select-Object -First 1 }
if (-not $voice) { exit 3 }
$synth.SelectVoice($voice.VoiceInfo.Name)
$synth.Rate = 0
$synth.Volume = 100
foreach ($scene in $request.scenes) {
    $synth.SetOutputToWaveFile($scene.path)
    $synth.Speak($scene.text)
    $synth.SetOutputToNull()
}
$synth.Dispose()
"""
        with tempfile.TemporaryDirectory(prefix="film_narration_", dir=str(path.parent)) as work_dir:
            work = Path(work_dir)
            requests = [
                {
                    "path": str((work / f"scene_{index:02d}.wav").resolve()),
                    "text": text,
                }
                for index, text in narration_scenes
            ]
            payload = base64.b64encode(json.dumps({
                "scenes": requests,
            }, ensure_ascii=False).encode("utf-8")).decode("ascii")
            try:
                result = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if result.returncode:
                    return False
                durations = {}
                for (scene_index, _), request in zip(narration_scenes, requests):
                    clip = Path(request["path"])
                    if not clip.is_file() or clip.stat().st_size <= 44:
                        return False
                    with wave.open(str(clip), "rb") as audio:
                        durations[scene_index] = audio.getnframes() / audio.getframerate()
                aligned_scene_durations = [
                    max(
                        scene["duration_sec"],
                        math.ceil(durations[index] + 0.35) if index in durations else 0,
                    )
                    for index, scene in enumerate(scenes)
                ]

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                inputs = []
                filters = []
                offset = 0.0
                audio_labels = []
                request_by_scene = {
                    scene_index: request
                    for (scene_index, _), request in zip(narration_scenes, requests)
                }
                for scene_index, scene_duration in enumerate(aligned_scene_durations):
                    request = request_by_scene.get(scene_index)
                    if request is not None:
                        input_index = len(inputs)
                        inputs.append(request["path"])
                        label = f"voice{input_index}"
                        filters.append(
                            f"[{input_index}:a]adelay={round(offset * 1000)}:all=1[{label}]"
                        )
                        audio_labels.append(f"[{label}]")
                    offset += scene_duration
                if not inputs:
                    return False
                if len(inputs) == 1:
                    filters.append(f"{audio_labels[0]}alimiter=limit=0.95[a]")
                else:
                    mix_inputs = "".join(audio_labels)
                    filters.append(
                        f"{mix_inputs}amix=inputs={len(inputs)}:duration=longest:normalize=0,"
                        "alimiter=limit=0.95[a]"
                    )
                command = [ffmpeg, "-y", "-loglevel", "error"]
                for clip in inputs:
                    command.extend(["-i", clip])
                command.extend([
                    "-filter_complex", ";".join(filters),
                    "-map", "[a]", "-t", f"{offset:.3f}",
                    "-c:a", "pcm_s16le", str(path),
                ])
                mux = subprocess.run(
                    command, capture_output=True, text=True, timeout=180
                )
                if mux.returncode or not path.is_file() or path.stat().st_size <= 44:
                    path.unlink(missing_ok=True)
                    return False
                for scene, duration in zip(scenes, aligned_scene_durations):
                    scene["duration_sec"] = duration
                return True
            except (OSError, subprocess.SubprocessError, wave.Error, ZeroDivisionError):
                path.unlink(missing_ok=True)
                return False

    @classmethod
    def _normalize_scenes(cls, scenes: List[Dict]) -> List[Dict]:
        if not isinstance(scenes, list):
            raise ValueError("The screenplay scenes must be a list.")
        normalized = []
        for scene in scenes[:cls.MAX_SCENES]:
            if not isinstance(scene, dict):
                continue
            try:
                duration = int(scene.get("duration_sec", 4))
            except (TypeError, ValueError, OverflowError):
                duration = 4
            normalized.append({
                "heading": str(scene.get("heading") or f"Scene {len(normalized) + 1}")[:100],
                "visual": str(scene.get("visual") or scene.get("setting") or "")[:500],
                "narration": str(scene.get("narration") or scene.get("dialogue") or "")[:600],
                "mood": str(scene.get("mood") or "hopeful").lower()[:40],
                "duration_sec": max(2, min(12, duration)),
            })
        return normalized

    @staticmethod
    @lru_cache(maxsize=32)
    def _font(size: int, bold: bool = False):
        candidates = [
            r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\seguisb.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _draw_scene(self, scene: Dict, index: int, title: str) -> Image.Image:
        palettes = {
            "hopeful": ((12, 31, 63), (83, 145, 177), (239, 177, 104)),
            "dark": ((8, 10, 28), (50, 42, 75), (170, 80, 85)),
            "tense": ((25, 16, 36), (108, 48, 62), (238, 122, 72)),
            "sad": ((16, 29, 49), (82, 108, 139), (179, 171, 159)),
            "romantic": ((58, 31, 76), (175, 98, 125), (255, 190, 155)),
            "mysterious": ((5, 23, 38), (31, 89, 104), (147, 195, 176)),
            "epic": ((11, 24, 49), (56, 103, 151), (247, 185, 87)),
        }
        mood = next((name for name in palettes if name in scene["mood"]), "hopeful")
        top, bottom, accent = palettes[mood]
        image = Image.new("RGB", (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(image)
        for y in range(self.HEIGHT):
            ratio = y / max(1, self.HEIGHT - 1)
            color = tuple(int(top[c] * (1 - ratio) + bottom[c] * ratio) for c in range(3))
            draw.line((0, y, self.WIDTH, y), fill=color)

        rng = random.Random(index * 982451653 + len(scene["visual"]))
        night = mood in {"dark", "mysterious"}
        atmosphere = Image.new("RGBA", (self.WIDTH, self.HEIGHT))
        atmosphere_draw = ImageDraw.Draw(atmosphere)
        for cloud_index in range(7):
            cloud_x = rng.randrange(-90, self.WIDTH)
            cloud_y = rng.randrange(70, 300)
            cloud_width = rng.randrange(100, 260)
            alpha = 18 if night else 28
            cloud_color = (*accent, alpha)
            atmosphere_draw.ellipse(
                (cloud_x, cloud_y, cloud_x + cloud_width, cloud_y + rng.randrange(25, 70)),
                fill=cloud_color,
            )
        atmosphere = atmosphere.filter(ImageFilter.GaussianBlur(24))
        image = Image.alpha_composite(image.convert("RGBA"), atmosphere).convert("RGB")
        draw = ImageDraw.Draw(image)
        for _ in range(80):
            x, y = rng.randrange(self.WIDTH), rng.randrange(20, 340)
            brightness = rng.randrange(80, 210)
            draw.ellipse((x, y, x + 2, y + 2), fill=(brightness, brightness, brightness))

        sun_x = int(self.WIDTH * (0.72 if index % 2 == 0 else 0.28))
        sun_y = int(self.HEIGHT * 0.29)
        sun_radius = 66 if not night else 40
        glow = Image.new("RGBA", (self.WIDTH, self.HEIGHT))
        glow_draw = ImageDraw.Draw(glow)
        for layer in range(5, 0, -1):
            radius = sun_radius * (1 + layer * 0.55)
            glow_draw.ellipse(
                (sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius),
                fill=(*accent, max(5, 12 - layer)),
            )
        glow = glow.filter(ImageFilter.GaussianBlur(28))
        image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ray in range(18):
            angle = math.tau * ray / 18
            outer_x = sun_x + math.cos(angle) * 260
            outer_y = sun_y + math.sin(angle) * 230
            draw.line((sun_x, sun_y, outer_x, outer_y), fill=tuple(
                int(channel * 0.35) for channel in accent
            ), width=1)
        draw.ellipse(
            (sun_x - sun_radius, sun_y - sun_radius, sun_x + sun_radius, sun_y + sun_radius),
            fill=(accent[0], accent[1], accent[2])
        )

        description = (scene["visual"] + " " + scene["heading"]).lower()
        if any(word in description for word in (
            "space", "planet", "galaxy", "太空", "星球", "宇宙", "星際"
        )):
            self._draw_hills(draw, accent, index)
            planet_x = int(self.WIDTH * (0.22 if index % 2 == 0 else 0.78))
            planet_y = int(self.HEIGHT * 0.4)
            planet_radius = 45 + (index * 13) % 38
            draw.ellipse(
                (planet_x - planet_radius, planet_y - planet_radius,
                 planet_x + planet_radius, planet_y + planet_radius),
                fill=accent,
            )
            draw.arc(
                (planet_x - planet_radius * 1.7, planet_y - planet_radius * 0.45,
                 planet_x + planet_radius * 1.7, planet_y + planet_radius * 0.45),
                190, 350, fill=(210, 222, 241), width=3,
            )
        elif any(word in description for word in ("city", "street", "building", "城市", "街道", "大樓", "都市")):
            self._draw_city(draw, rng, accent)
        elif any(word in description for word in ("sea", "ocean", "water", "海", "船", "湖")):
            draw.polygon([(0, 360), (self.WIDTH, 335), (self.WIDTH, 445), (0, 455)], fill=(25, 75, 103))
            for line in range(10):
                x = rng.randrange(self.WIDTH)
                y = rng.randrange(365, 445)
                draw.line((x, y, x + rng.randrange(20, 100), y), fill=(76, 134, 154), width=2)
            self._draw_hills(draw, accent, 1)
        else:
            self._draw_hills(draw, accent, index)
            if any(word in description for word in ("forest", "wood", "樹", "森林", "山林")):
                self._draw_forest(draw, index)

        self._draw_silhouette(draw, index)
        foreground = Image.new("RGBA", (self.WIDTH, self.HEIGHT))
        foreground_draw = ImageDraw.Draw(foreground)
        foreground_draw.polygon(
            [(0, 500), (110, 457), (245, 490), (390, 448), (540, 500),
             (700, 458), (840, 492), (960, 450), (960, 540), (0, 540)],
            fill=(8, 15, 25, 125),
        )
        foreground = foreground.filter(ImageFilter.GaussianBlur(3))
        image = Image.alpha_composite(image.convert("RGBA"), foreground).convert("RGB")
        return image

    def _draw_hills(self, draw: ImageDraw.ImageDraw, accent, index: int):
        draw.polygon(
            [(0, 345), (105, 235 + index % 3 * 20), (250, 342), (400, 205),
             (580, 355), (745, 225), (960, 350), (960, 540), (0, 540)],
            fill=(35, 57, 84)
        )
        draw.polygon(
            [(0, 405), (165, 315), (320, 420), (510, 300),
             (720, 420), (850, 330), (960, 390), (960, 540), (0, 540)],
            fill=(24, 43, 62)
        )
        draw.polygon([(0, 440), (960, 408), (960, 540), (0, 540)], fill=(16, 30, 43))
        draw.line((0, 438, 960, 405), fill=accent, width=2)

    def _draw_city(self, draw: ImageDraw.ImageDraw, rng: random.Random, accent):
        x = 0
        while x < self.WIDTH:
            width = rng.randrange(38, 105)
            height = rng.randrange(90, 240)
            top = 420 - height
            draw.rectangle((x, top, x + width, 455), fill=(18, 30, 51))
            for wx in range(x + 9, x + width - 5, 17):
                for wy in range(top + 14, 440, 23):
                    if rng.random() > 0.35:
                        draw.rectangle((wx, wy, wx + 5, wy + 8), fill=accent)
            x += width + 5
        draw.rectangle((0, 455, self.WIDTH, self.HEIGHT), fill=(12, 22, 38))

    def _draw_forest(self, draw: ImageDraw.ImageDraw, index: int):
        for x in range(20, self.WIDTH, 85):
            height = 65 + ((x * 7 + index * 31) % 90)
            draw.polygon(
                [(x, 438 - height), (x - 30, 438), (x + 31, 438)],
                fill=(11, 35, 38)
            )
            draw.rectangle((x - 3, 425, x + 3, 468), fill=(32, 30, 27))

    def _draw_silhouette(self, draw: ImageDraw.ImageDraw, index: int):
        x = 235 if index % 2 == 0 else 725
        ground = 450
        scale = 1.0 + (index % 3) * 0.08
        r = int(20 * scale)
        draw.ellipse((x - r, ground - 144 * scale, x + r, ground - 104 * scale), fill=(9, 15, 25))
        draw.polygon(
            [(x - 27 * scale, ground - 107 * scale), (x + 27 * scale, ground - 107 * scale),
             (x + 43 * scale, ground - 37 * scale), (x + 14 * scale, ground - 35 * scale),
             (x, ground - 70 * scale), (x - 14 * scale, ground - 35 * scale),
             (x - 43 * scale, ground - 37 * scale)],
            fill=(9, 15, 25)
        )
        draw.line((x - 15, ground - 70, x - 43, ground - 15), fill=(9, 15, 25), width=12)
        draw.line((x + 15, ground - 70, x + 42, ground - 20), fill=(9, 15, 25), width=12)

    def _render_frame(self, background: Image.Image, scene: Dict, scene_index: int, frame: int, frame_count: int) -> Image.Image:
        progress = frame / max(1, frame_count)
        zoom = 1.0 + 0.035 * progress
        scaled = background.resize(
            (int(self.WIDTH * zoom), int(self.HEIGHT * zoom)),
            Image.Resampling.BICUBIC
        )
        max_x = scaled.width - self.WIDTH
        max_y = scaled.height - self.HEIGHT
        left = int(max_x * progress)
        top = int(max_y * progress * 0.4)
        image = scaled.crop((left, top, left + self.WIDTH, top + self.HEIGHT))
        draw = ImageDraw.Draw(image, "RGBA")
        drift = int(progress * 18)
        draw.ellipse((720 - drift, 62, 740 - drift, 82), fill=(255, 255, 255, 150))
        for particle in range(7):
            x = (particle * 137 + frame * (particle % 3 + 1) * 2) % self.WIDTH
            y = 110 + particle * 37
            draw.line((x, y, x + 12, y - 5), fill=(255, 212, 150, 100), width=2)

        draw.rectangle((0, 0, self.WIDTH, 54), fill=(3, 8, 19, 170))
        label = f"{scene_index + 1:02d} / {scene['heading']}"
        draw.text((42, 16), label, font=self._font(20, True), fill=(245, 239, 225, 255))

        panel_top = self.HEIGHT - 150
        draw.rectangle((0, panel_top, self.WIDTH, self.HEIGHT), fill=(2, 6, 16, 205))
        text = scene["narration"] or scene["visual"] or scene["heading"]
        lines = textwrap.wrap(text, width=64)[:3]
        font = self._font(24)
        for line_index, line in enumerate(lines):
            draw.text((45, panel_top + 20 + line_index * 34), line, font=font, fill=(249, 246, 238, 255))
        if scene["visual"]:
            visual = textwrap.shorten(scene["visual"], width=92, placeholder="…")
            draw.text((45, panel_top + 119), visual, font=self._font(14), fill=(183, 199, 215, 255))

        fade_frames = min(12, frame_count // 4)
        fade = min(1.0, frame / max(1, fade_frames), (frame_count - 1 - frame) / max(1, fade_frames))
        if fade < 1:
            draw.rectangle((0, 0, self.WIDTH, self.HEIGHT), fill=(0, 0, 0, int(255 * (1 - fade))))
        return image

    def _write_storyboard(self, path: Path, scenes: List[Dict], backgrounds: List[Image.Image], title: str):
        columns, thumb_w, thumb_h, row_h = 3, 300, 169, 215
        rows = math.ceil(len(scenes) / columns)
        sheet = Image.new("RGB", (columns * thumb_w + 80, rows * row_h + 105), (12, 19, 31))
        draw = ImageDraw.Draw(sheet)
        draw.text((32, 22), title[:100], font=self._font(26, True), fill=(242, 242, 242))
        for index, (scene, background) in enumerate(zip(scenes, backgrounds)):
            x = 25 + (index % columns) * (thumb_w + 15)
            y = 72 + (index // columns) * row_h
            thumb = background.resize((thumb_w, thumb_h))
            sheet.paste(thumb, (x, y))
            draw.text((x, y + thumb_h + 5), f"{index + 1:02d}. {scene['heading'][:32]}", font=self._font(16, True), fill=(245, 245, 245))
            caption = textwrap.shorten(scene["narration"] or scene["visual"], width=38, placeholder="…")
            draw.text((x, y + thumb_h + 28), caption, font=self._font(12), fill=(174, 193, 212))
        sheet.save(path, "PNG")

    @staticmethod
    def _write_screenplay(title: str, logline: str, scenes: List[Dict]) -> str:
        parts = [f"# {title}", "", f"> {logline}" if logline else "", "", "## Screenplay"]
        for index, scene in enumerate(scenes, 1):
            parts.extend([
                f"### Scene {index}: {scene['heading']}",
                f"- **Visual:** {scene['visual'] or '—'}",
                f"- **Mood:** {scene['mood']}",
                f"- **Duration:** {scene['duration_sec']} seconds",
                f"- **Narration / dialogue:** {scene['narration'] or '—'}",
                "",
            ])
        return "\n".join(parts)

    def _write_soundtrack(self, path: Path, scenes: List[Dict], duration: float):
        sample_rate = 22050
        chord_roots = [110.0, 130.81, 146.83, 164.81, 196.0, 174.61]
        with wave.open(str(path), "wb") as output:
            total_samples = int(sample_rate * duration)
            output.setparams((1, 2, sample_rate, total_samples, "NONE", "not compressed"))
            chunk = bytearray()
            sample = 0
            for scene_index, scene in enumerate(scenes):
                root = chord_roots[scene_index % len(chord_roots)]
                for _ in range(scene["duration_sec"] * sample_rate):
                    t = sample / sample_rate
                    envelope = min(1.0, t / 1.2, (duration - t) / 1.5)
                    value = envelope * (
                        0.16 * math.sin(2 * math.pi * root * t)
                        + 0.10 * math.sin(2 * math.pi * root * 1.25 * t)
                        + 0.08 * math.sin(2 * math.pi * root * 1.5 * t)
                    )
                    chunk.extend(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
                    sample += 1
                    if len(chunk) >= 32768:
                        output.writeframesraw(chunk)
                        chunk.clear()
            if chunk:
                output.writeframesraw(chunk)

    @staticmethod
    def _write_viewer(
        target_dir: Path, title: str, logline: str, scenes: List[Dict], has_narration: bool
    ):
        safe_title = html.escape(title)
        safe_logline = html.escape(logline)
        narration_link = (
            '<a id="narration-label" href="narration.wav">語音旁白</a>'
            if has_narration else ""
        )
        scene_list = "".join(
            f"<li>{html.escape(scene['heading'])}</li>" for scene in scenes
        )
        viewer = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#080d18;color:#f1f5f9;font:16px/1.5 'Segoe UI',sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px}}h1{{margin:.2em 0}}p{{color:#aab8ca}}
video{{display:block;width:100%;max-height:75vh;background:#000;border:1px solid #334155}}
a{{color:#7dd3fc}}.links{{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}}
</style>
</head>
<body><main>
<p id="film-kicker">本機製作短片</p><h1>{safe_title}</h1><p>{safe_logline}</p>
<video controls playsinline preload="metadata" poster="poster.png">
<source src="film.mp4" type="video/mp4">Your browser cannot play this video.
</video>
<nav class="links"><a id="download-label" href="film.mp4" download>下載 MP4</a>
<a id="screenplay-label" href="screenplay.md">劇本</a><a id="storyboard-label" href="storyboard.png">分鏡圖</a>
<a id="soundtrack-label" href="soundtrack.wav">配樂</a>
{narration_link}</nav>
<h2 id="scenes-label">場景</h2><ol>{scene_list}</ol>
</main><script>
const filmLabels = {{
  zh: ['本機製作短片', '下載 MP4', '劇本', '分鏡圖', '配樂', '場景', '語音旁白'],
  en: ['LOCAL SHORT FILM', 'Download MP4', 'Screenplay', 'Storyboard', 'Soundtrack', 'Scenes', 'Voice-over']
}};
function applyFilmLanguage(language) {{
  const labels = filmLabels[language === 'en' ? 'en' : 'zh'];
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-Hant';
  ['film-kicker', 'download-label', 'screenplay-label', 'storyboard-label', 'soundtrack-label', 'scenes-label']
    .forEach((id, index) => document.getElementById(id).textContent = labels[index]);
  const narration = document.getElementById('narration-label');
  if (narration) narration.textContent = labels[6];
}}
let preferredFilmLanguage = 'zh';
try {{ preferredFilmLanguage = localStorage.getItem('omni-language') || 'zh'; }}
catch (error) {{ if (error?.name !== 'SecurityError') throw error; }}
applyFilmLanguage(preferredFilmLanguage);
window.addEventListener('message', event => {{
  if (event.source === window.parent && event.data?.type === 'set-language') {{
    applyFilmLanguage(event.data.language);
  }}
}});
</script></body></html>"""
        (target_dir / "index.html").write_text(viewer, encoding="utf-8")
