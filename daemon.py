import json
import time
import re
import html
import os
import threading
import tempfile
import shutil
import traceback
import ast
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Callable, Optional, Dict, List
import psutil

from config import TASK_ZONE, DELIVERY_ZONE, HISTORY_ZONE, FEEDBACK_DIR, CONTROL_FILE, BACKLOG_FILE, LESSONS_FILE
from core.model_pool import DynamicModelPool
from generators.asset_forge import AutonomousAssetForge
from generators.film_forge import AutonomousFilmForge
from generators.image_forge import AutonomousImageForge
from execution.sandbox_runner import SandboxRunner
from execution.game_shell import assemble_game_html
import pandas as pd

class TaskTerminatedException(Exception):
    pass

class OmniEnterpriseDaemon:
    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self.pool = DynamicModelPool()
        self.forge = AutonomousAssetForge()
        self.film_forge = AutonomousFilmForge()
        self.image_forge = AutonomousImageForge()
        self.sandbox = SandboxRunner()
        self.broadcast_fn = broadcast_fn
        self.backlog_lock = threading.Lock()
        self.queue_condition = threading.Condition()
        self._queue_restart_requested = False
        self._interrupt_lock = threading.Lock()
        self._interrupt_events = {}
        self._task_context = threading.local()
        self._task_runtime_lock = threading.Lock()
        self._task_runtime = {}

        # 啟動時將上次中斷的 processing 任務自動恢復為 pending (確保未完成任務可立即被看到並續跑)
        self._recover_unfinished_tasks_on_boot()

        self.daemon_active = False
        self.daemon_start_ts: Optional[float] = None
        self.daemon_paused_accum: float = 0.0
        self.task_start_ts: Optional[float] = None
        self.task_paused_accum: float = 0.0
        self.task_prior_duration_sec: float = 0.0
        self.pause_begin_ts: Optional[float] = None

        self.current_task_id = "-"
        self.current_task_stage = "Idle (Ready to Resume)"
        self.current_task_progress_pct = 0

        self._timer_thread = threading.Thread(target=self._background_ticker, daemon=True)
        self._timer_thread.start()

    def _thread_value(self, key, default=None):
        return getattr(self._task_context, key, default)

    def _ensure_task_runtime(self):
        if not hasattr(self, "_task_context"):
            self._task_context = threading.local()
        if not hasattr(self, "_task_runtime_lock"):
            self._task_runtime_lock = threading.Lock()
        if not hasattr(self, "_task_runtime"):
            self._task_runtime = {}
        if not hasattr(self, "_interrupt_lock"):
            self._interrupt_lock = threading.Lock()
        if not hasattr(self, "_interrupt_events"):
            self._interrupt_events = {}

    def _set_thread_value(self, key, value):
        self._ensure_task_runtime()
        setattr(self._task_context, key, value)
        task_id = getattr(self._task_context, "current_task_id", "-")
        if task_id == "-" or not hasattr(self, "_task_runtime_lock"):
            return
        with self._task_runtime_lock:
            runtime = self._task_runtime.get(task_id)
            if runtime is not None:
                field = {
                    "current_task_stage": "stage",
                    "current_task_progress_pct": "progress_pct",
                    "task_start_ts": "started_at",
                }.get(key)
                if field:
                    runtime[field] = value

    @property
    def current_task_id(self):
        return self._thread_value("current_task_id", "-")

    @current_task_id.setter
    def current_task_id(self, value):
        self._set_thread_value("current_task_id", value)

    @property
    def current_task_stage(self):
        return self._thread_value("current_task_stage", "Idle")

    @current_task_stage.setter
    def current_task_stage(self, value):
        self._set_thread_value("current_task_stage", value)

    @property
    def current_task_progress_pct(self):
        return self._thread_value("current_task_progress_pct", 0)

    @current_task_progress_pct.setter
    def current_task_progress_pct(self, value):
        self._set_thread_value("current_task_progress_pct", value)

    @property
    def task_start_ts(self):
        task_id = self.current_task_id
        if task_id != "-" and hasattr(self, "_task_runtime_lock"):
            with self._task_runtime_lock:
                runtime = self._task_runtime.get(task_id)
                if runtime is not None:
                    return runtime["started_at"]
        return self._thread_value("task_start_ts")

    @task_start_ts.setter
    def task_start_ts(self, value):
        self._set_thread_value("task_start_ts", value)

    @property
    def task_paused_accum(self):
        task_id = self.current_task_id
        if task_id != "-" and hasattr(self, "_task_runtime_lock"):
            with self._task_runtime_lock:
                runtime = self._task_runtime.get(task_id)
                if runtime is not None:
                    return runtime["paused_accum"]
        return self._thread_value("task_paused_accum", 0.0)

    @task_paused_accum.setter
    def task_paused_accum(self, value):
        self._set_thread_value("task_paused_accum", value)

    @property
    def task_prior_duration_sec(self):
        return self._thread_value("task_prior_duration_sec", 0.0)

    @task_prior_duration_sec.setter
    def task_prior_duration_sec(self, value):
        self._set_thread_value("task_prior_duration_sec", value)

    def _active_task_snapshot(self):
        with self._task_runtime_lock:
            return {
                task_id: dict(runtime)
                for task_id, runtime in self._task_runtime.items()
            }

    @staticmethod
    def _active_task_telemetry(active_tasks: Dict, now: Optional[float] = None) -> List[Dict]:
        wall_now = time.time() if now is None else now
        entries = []
        for task_id, runtime in active_tasks.items():
            active_pause = (
                wall_now - runtime["pause_started"]
                if runtime.get("pause_started") is not None else 0.0
            )
            elapsed_sec = max(0, int(
                wall_now - runtime["started_at"]
                - runtime["paused_accum"] - active_pause
                + float(runtime.get("prior_duration_sec", 0.0))
            ))
            entries.append({
                "task_id": task_id,
                "title": runtime.get("title", ""),
                "stage": runtime["stage"],
                "progress_pct": runtime["progress_pct"],
                "elapsed_sec": elapsed_sec,
            })
        return entries

    def _generation_interrupt(self, task_id: str) -> threading.Event:
        self._ensure_task_runtime()
        with self._interrupt_lock:
            return self._interrupt_events.setdefault(task_id, threading.Event())

    def _generate(self, *args, **kwargs):
        task_id = kwargs.get("task_id")
        if task_id is None and len(args) > 3:
            task_id = args[3]
        if task_id is None:
            task_id = self.current_task_id
        kwargs.setdefault("interrupt_event", self._generation_interrupt(str(task_id)))
        return self.pool.generate(*args, **kwargs)

    def _run_python_script(self, script_path: Path) -> Dict:
        result = self.sandbox.run_python_script(script_path)
        if not result.get("success") and result.get("infrastructure_error"):
            raise RuntimeError(
                f"Python sandbox is unavailable or misconfigured: {result.get('stderr', '')}"
            )
        return result

    def _recover_unfinished_tasks_on_boot(self):
        with self.backlog_lock:
            if not BACKLOG_FILE.exists():
                return
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                changed = False
                for t in tasks:
                    if t.get("status") == "processing":
                        t["status"] = "pending"
                        changed = True
                if changed:
                    self._save_task_state(BACKLOG_FILE, tasks)
            except (OSError, ValueError, TypeError) as exc:
                self.emit_log("SYS", f"無法恢復未完成任務佇列：{exc}")

    def get_all_tasks(self) -> List[Dict]:
        with self.backlog_lock:
            if not BACKLOG_FILE.exists():
                return []
            try:
                return json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
            except Exception:
                return []

    def _get_accurate_elapsed(self) -> tuple[int, int]:
        now = time.time()
        active_pause_delta = (now - self.pause_begin_ts) if self.pause_begin_ts else 0.0

        if self.daemon_start_ts is not None:
            total_sec = max(0, int(now - self.daemon_start_ts - self.daemon_paused_accum - active_pause_delta))
        else:
            total_sec = 0

        if self.task_start_ts is not None and self.current_task_id != "-":
            task_sec = max(0, int(now - self.task_start_ts - self.task_paused_accum - active_pause_delta))
        else:
            task_sec = 0

        return total_sec, task_sec

    def _background_ticker(self):
        last_memory_sync = 0.0
        last_queue_sync = 0.0
        mem_info = {"used_gb": 0.0, "running_models": []}
        q_stats = {"completed": 0, "total": 0, "pct": 0}
        tasks = []
        while True:
            time.sleep(2.0)
            ctrl = self.get_control_state()
            state = ctrl.get("state", "IDLE")
            total_sec, task_sec = self._get_accurate_elapsed()
            active_tasks = self._active_task_snapshot()
            primary_task_id = next(iter(active_tasks), "-")
            primary_task = active_tasks.get(primary_task_id, {})
            active_task_entries = self._active_task_telemetry(active_tasks)
            if active_task_entries:
                task_sec = active_task_entries[0]["elapsed_sec"]

            now = time.monotonic()
            if now - last_memory_sync >= 5.0:
                mem_info = self.pool.sync_realtime_memory()
                last_memory_sync = now
            if now - last_queue_sync >= 4.0:
                tasks = self.get_all_tasks()
                q_stats = self.get_queue_progress(tasks)
                last_queue_sync = now
            budget = self.pool.get_hardware_budget()

            if self.broadcast_fn:
                self.broadcast_fn({
                    "type": "telemetry",
                    "state": state,
                    "total_elapsed_sec": total_sec,
                    "task_elapsed_sec": task_sec,
                    "estimated_total_sec": self.estimate_total_runtime_sec(
                        total_sec,
                        task_sec,
                        tasks=tasks,
                        active_task=(
                            {
                                **primary_task,
                                "task_id": primary_task_id,
                                "elapsed_sec": task_sec,
                            }
                            if primary_task else None
                        ),
                    ),
                    "current_task_id": primary_task_id,
                    "current_stage": primary_task.get("stage", "Idle"),
                    "task_progress_pct": primary_task.get("progress_pct", 0),
                    "queue_completed": q_stats["completed"],
                    "queue_total": q_stats["total"],
                    "queue_pct": q_stats["pct"],
                    "used_memory_gb": mem_info["used_gb"],
                    "budget_gb": budget,
                    "model_error": self.pool.last_error,
                    "memory_error": self.pool.last_memory_error,
                    "running_models": mem_info["running_models"],
                    "active_workers": self.pool.get_active_worker_progress(),
                    "active_tasks": active_task_entries,
                    "tasks": tasks
                })

    def get_control_state(self) -> Dict:
        if CONTROL_FILE.exists():
            try:
                return json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"state": "IDLE", "pause_requested": False, "stop_requested": False}

    def set_control_state(self, state: str, pause_req: bool = False, stop_req: bool = False):
        self._ensure_task_runtime()
        now = time.time()
        with self._task_runtime_lock:
            for runtime in self._task_runtime.values():
                if pause_req and runtime["pause_started"] is None:
                    runtime["pause_started"] = now
                elif not pause_req and runtime["pause_started"] is not None:
                    runtime["paused_accum"] += max(0.0, now - runtime["pause_started"])
                    runtime["pause_started"] = None
        if pause_req or stop_req:
            with self._interrupt_lock:
                for interrupt_event in self._interrupt_events.values():
                    interrupt_event.set()
        elif not stop_req:
            with self._interrupt_lock:
                for interrupt_event in self._interrupt_events.values():
                    interrupt_event.clear()
        if pause_req and self.pause_begin_ts is None:
            self.pause_begin_ts = now
        elif not pause_req and self.pause_begin_ts is not None:
            paused_duration = now - self.pause_begin_ts
            self.daemon_paused_accum += paused_duration
            self.pause_begin_ts = None

        CONTROL_FILE.write_text(json.dumps({
            "state": state,
            "pause_requested": pause_req,
            "stop_requested": stop_req
        }, indent=2), encoding="utf-8")
        with self.queue_condition:
            self.queue_condition.notify_all()

    def get_queue_progress(self, tasks: Optional[List[Dict]] = None) -> Dict:
        if tasks is None:
            tasks = self.get_all_tasks()
        total = len(tasks)
        if total == 0:
            return {"completed": 0, "total": 0, "pct": 0}
        comp = sum(1 for t in tasks if t.get("status") in ["completed", "failed"])
        return {"completed": comp, "total": total, "pct": int((comp / total) * 100)}

    def emit_log(self, task_id: str, msg: str, event_type: str = "log", extra: dict = None):
        if self.broadcast_fn:
            self.broadcast_fn({
                "time": time.strftime("%H:%M:%S"),
                "type": event_type,
                "task_id": task_id,
                "message": msg,
                "extra": extra or {}
            })

    def checkpoint(self, task_id: str, stage_name: str = None, progress_pct: int = None) -> str:
        self._ensure_task_runtime()
        if self._thread_value("current_task_id", "-") != task_id:
            self.current_task_id = task_id
            with self._task_runtime_lock:
                runtime = self._task_runtime.get(task_id)
                initial_stage = runtime.get("stage") if runtime else "Starting"
                initial_progress = runtime.get("progress_pct", 0) if runtime else 0
            self.current_task_stage = initial_stage
            self.current_task_progress_pct = initial_progress
        if stage_name:
            self.current_task_stage = stage_name
        if progress_pct is not None:
            self.current_task_progress_pct = progress_pct
        self._persist_task_checkpoint(task_id, stage_name, progress_pct)

        ctrl = self.get_control_state()
        if ctrl.get("stop_requested"):
            raise TaskTerminatedException(f"Task {task_id} terminated by user.")

        if ctrl.get("pause_requested"):
            self.set_control_state("PAUSED", pause_req=True, stop_req=False)
            self.update_task_status(task_id, "paused")
            self.emit_log(task_id, "⏸ 系統已進入暫停狀態（目前已生成之進度已保留，可調整記憶體上限或注入意見後按『繼續』）...")
            while True:
                time.sleep(0.5)
                c = self.get_control_state()
                if c.get("stop_requested"):
                    raise TaskTerminatedException(f"Task {task_id} terminated while paused.")
                if not c.get("pause_requested"):
                    self.set_control_state("RUNNING", pause_req=False, stop_req=False)
                    self.update_task_status(task_id, "processing")
                    new_budget = self.pool.get_hardware_budget()
                    self.emit_log(task_id, f"⏯ 系統已無損接續運行！當前套用記憶體上限: {new_budget} GB")
                    break

        return self.consume_user_feedback(task_id)

    def _persist_task_checkpoint(
        self, task_id: str, stage_name: Optional[str] = None,
        progress_pct: Optional[int] = None,
    ):
        with self.backlog_lock:
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                for task in tasks:
                    if task.get("task_id") == task_id:
                        task["resume_stage"] = stage_name or self.current_task_stage
                        task["progress_pct"] = (
                            self.current_task_progress_pct if progress_pct is None else progress_pct
                        )
                        task["checkpoint_at"] = time.time()
                        break
                self._save_task_state(BACKLOG_FILE, tasks)
            except (OSError, ValueError, TypeError) as exc:
                self.emit_log(task_id, f"⚠ 無法將目前階段寫入任務佇列：{exc}")

    @staticmethod
    def _save_task_state(path: Path, state: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.tmp")
        staged.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.replace(path)

    @staticmethod
    def _task_delivery_dir(category: str, task_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
            raise ValueError(f"Invalid task ID for delivery path: {task_id!r}")
        directory = DELIVERY_ZONE / category / task_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _write_delivery_text(path: Path, content: str) -> bool:
        return OmniEnterpriseDaemon._publish_delivery_bytes(path, content.encode("utf-8"))

    @staticmethod
    def _publish_delivery_bytes(path: Path, content: bytes) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                dir=path.parent, delete=False
            ) as staged:
                temp_path = Path(staged.name)
                staged.write(content)
                staged.flush()
                os.fsync(staged.fileno())
            os.link(temp_path, path)
            return True
        except FileExistsError:
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _publish_staged_directory(staged: Path, destination: Path):
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite existing delivery: {destination}"
                )
            destination.rmdir()
        staged.rename(destination)

    @staticmethod
    def _infer_task_routes(title: str, description: str, attached_file: str = "") -> Dict[str, bool]:
        text = f"{title} {description}".casefold()
        image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        suffix = Path(attached_file).suffix.casefold() if attached_file else ""
        has_image = suffix in image_suffixes
        game = any(term in text for term in [
            "遊戲", "game", "rpg", "推塔", "主堡", "gameplay"
        ])
        film = OmniEnterpriseDaemon._is_film_request(title, description)
        image_creation = any(term in text for term in [
            "圖片創建", "創建圖片", "生成圖片", "製作圖片", "畫圖", "繪製",
            "插畫", "illustration", "create an image", "generate an image",
            "create image", "generate image", "draw a", "draw an", "artwork", "logo design"
        ])
        spreadsheet = Path(attached_file).suffix.casefold() in {".csv", ".xlsx", ".xls"}
        data_cleaning = spreadsheet and any(term in text for term in [
            "清理", "整理", "轉換", "去重", "clean", "transform", "deduplicate", "normalize"
        ])
        explicit_software = any(term in text for term in [
            "程式", "程式碼", "軟體", "系統開發", "網站", "應用程式",
            "爬蟲", "crawler", "scraper", "web scraping", "web scraper",
            "python", "javascript", "code", "software", "application", "website",
            "build an app", "build a tool", "automation", "api"
        ])
        return {
            "needs_vision_analysis": has_image,
            "needs_data_cleaning": data_cleaning,
            "needs_game_development": game,
            "needs_film_production": film,
            "needs_image_creation": image_creation,
            "needs_software_project": explicit_software and not (game or film or image_creation),
            "needs_direct_qa_report": True,
        }

    def consume_user_feedback(self, task_id: str) -> str:
        fb_file = FEEDBACK_DIR / f"{task_id}.json"
        if not fb_file.exists():
            return ""
        try:
            notes = json.loads(fb_file.read_text(encoding="utf-8"))
            if not notes:
                return ""
            fb_file.write_text("[]", encoding="utf-8")
            joined = "\n".join([f"- [{n['time']}] {n['text']}" for n in notes])
            self.emit_log(task_id, f"💡 已成功讀取並套用使用者最新意見：\n{joined}")
            return f"\n【使用者最新追加指導意見（最高優先級，必須嚴格實作）】：\n{joined}\n"
        except Exception:
            return ""

    @staticmethod
    def _lesson_keywords(text: str) -> set[str]:
        normalized = text.casefold()
        words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
        chinese = re.findall(r"[\u4e00-\u9fff]+", normalized)
        for phrase in chinese:
            words.update(phrase[index:index + 2] for index in range(len(phrase) - 1))
            if len(phrase) <= 4:
                words.add(phrase)
        return words

    def get_lessons(self, title: str = "", description: str = "") -> str:
        try:
            data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                return ""
            task_terms = self._lesson_keywords(f"{title} {description}")
            ranked = []
            for index, item in enumerate(data):
                if not isinstance(item, dict) or not isinstance(item.get("rule"), str):
                    continue
                rule = item["rule"].strip()
                if not rule:
                    continue
                overlap = task_terms & self._lesson_keywords(rule)
                if overlap:
                    ranked.append((len(overlap), index, rule))
            relevant = [
                rule for _, _, rule in sorted(ranked, key=lambda row: (row[0], row[1]), reverse=True)[:5]
            ]
            if not relevant:
                return ""
            return "【與本任務相關的歷史避坑經驗】：\n" + "\n".join(f"- {rule}" for rule in relevant)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return ""

    @staticmethod
    def _strip_code_fence(code: str, language: str = "python") -> str:
        return (
            code.replace(f"```{language}", "")
            .replace("```", "")
            .strip()
        )

    @staticmethod
    def _apply_exact_patches(source: str, patch_response: str) -> Optional[str]:
        if not patch_response.strip() or not any(
            marker in patch_response
            for marker in ("<<<PATCH:", "<<<FIND>>>", "<<<REPLACE>>>", "<<<END_PATCH>>>")
        ):
            return source
        patches = re.findall(
            r"<<<FIND>>>(.*?)<<<REPLACE>>>(.*?)<<<END_PATCH>>>",
            patch_response,
            re.DOTALL,
        )
        if not patches:
            return None
        updated = source
        for old, new in patches:
            old = old.strip("\r\n")
            new = new.strip("\r\n")
            old_lines = old.splitlines()
            source_lines = updated.splitlines()
            normalized_old = [line.strip() for line in old_lines]
            if not old_lines:
                return None
            matches = [
                start for start in range(len(source_lines) - len(old_lines) + 1)
                if [line.strip() for line in source_lines[start:start + len(old_lines)]]
                == normalized_old
            ]
            if len(matches) != 1:
                return None
            start = matches[0]
            matched = source_lines[start:start + len(old_lines)]
            old_indent = len(old_lines[0]) - len(old_lines[0].lstrip())
            actual_indent = len(matched[0]) - len(matched[0].lstrip())
            replacement_lines = new.splitlines()
            adjusted = []
            for line in replacement_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indent = max(0, indent + actual_indent - old_indent)
                    adjusted.append((" " * indent) + line.lstrip())
                else:
                    adjusted.append("")
            source_lines[start:start + len(old_lines)] = adjusted
            newline = "\r\n" if "\r\n" in updated else "\n"
            updated = newline.join(source_lines)
            if source.endswith(("\n", "\r")):
                updated += newline
        return updated

    @staticmethod
    def _is_valid_python_output(code: str) -> bool:
        cleaned = OmniEnterpriseDaemon._strip_code_fence(code)
        return SandboxRunner.is_valid_python(cleaned)

    @staticmethod
    def _is_valid_javascript_output(code: str) -> bool:
        cleaned = code.replace("```javascript", "").replace("```js", "").replace("```", "").strip()
        return SandboxRunner.is_valid_javascript(cleaned)

    @staticmethod
    def _normalize_game_asset_manifest(payload) -> list:
        items = payload.get("assets", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            return []
        normalized = []
        seen = set()
        for item in items[:24]:
            if not isinstance(item, dict):
                continue
            asset_type = str(item.get("type", "sprite")).casefold()
            extension = ".wav" if asset_type in {"audio", "sound", "sfx"} else ".png"
            supplied_name = str(item.get("filename", ""))
            filename = Path(supplied_name).name
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", filename)
                or filename != supplied_name
                or not filename.casefold().endswith(extension)
                or filename.casefold() in seen
            ):
                continue
            seen.add(filename.casefold())
            color = str(item.get("color", "#38bdf8"))
            if not re.fullmatch(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?", color):
                color = "#38bdf8"
            normalized.append({
                "type": "sfx" if extension == ".wav" else "sprite",
                "filename": filename,
                "style": str(item.get("style", "abstract"))[:80],
                "color": color,
            })
        return normalized

    @staticmethod
    def _game_validation_guidance(validation_error: str) -> str:
        if not validation_error:
            return ""
        return (
            f"\n上一輪 Node.js 沙盒驗證失敗：\n{validation_error[:3000]}\n"
            "請先修復上述錯誤，尤其是錯誤訊息指出的模組；確認每個模組都以正確 ID "
            "註冊 window.GameModules，並實作 init(api)。"
        )

    @staticmethod
    def _select_escalation_model(ranked_models: list, tier: int) -> dict:
        if not ranked_models:
            raise ValueError("No compatible escalation models are available.")
        index = max(0, len(ranked_models) - max(1, tier) - 1)
        return ranked_models[index]

    def _is_valid_python_patch(self, filename: str, source: str, response: str) -> bool:
        if not response.strip() or not any(
            marker in response
            for marker in ("<<<PATCH:", "<<<FIND>>>", "<<<REPLACE>>>", "<<<END_PATCH>>>")
        ):
            return True
        match = re.search(
            rf"<<<PATCH:\s*{re.escape(filename)}\s*>>>(.*?)(?=<<<PATCH:|$)",
            response,
            re.DOTALL,
        )
        if not match:
            return False
        patch_body = match.group(1)
        updated = self._apply_exact_patches(source, patch_body)
        return updated is not None and self.sandbox.is_valid_python(updated)

    def _are_valid_python_patches(
        self, project_dir: Path, allowed_files: set[str], response: str
    ) -> bool:
        if not response.strip() or not any(
            marker in response
            for marker in ("<<<PATCH:", "<<<FIND>>>", "<<<REPLACE>>>", "<<<END_PATCH>>>")
        ):
            return True
        patches = re.findall(
            r"<<<PATCH:\s*([^>]+)>>>(.*?)(?=<<<PATCH:|$)",
            response,
            re.DOTALL,
        )
        if not patches:
            return False
        grouped = {}
        for filename, patch in patches:
            filename = filename.strip()
            if (
                filename not in allowed_files
                or Path(filename).name != filename
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py", filename)
            ):
                return False
            grouped.setdefault(filename, []).append(patch)
        for filename, file_patches in grouped.items():
            path = project_dir / filename
            if not path.is_file():
                return False
            source = path.read_text(encoding="utf-8")
            for patch in file_patches:
                updated = self._apply_exact_patches(source, patch)
                if updated is None:
                    return False
                source = updated
            if not self.sandbox.is_valid_python(source):
                return False
        return True

    def _spawn_subtasks(self, parent_task: Dict, proposed: object) -> List[str]:
        if not isinstance(proposed, list) or not proposed:
            return []
        if len(proposed) > 12:
            raise ValueError("An architecture plan may spawn at most 12 subtasks.")

        normalized = []
        keys = set()
        for item in proposed:
            if not isinstance(item, dict):
                raise ValueError("Every DAG subtask must be a JSON object.")
            key = str(item.get("key", "")).strip()
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            dependencies = item.get("depends_on", [])
            priority = item.get("priority", 0)
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", key)
                or key in keys
                or not title
                or not description
                or not isinstance(dependencies, list)
                or not isinstance(priority, int)
                or isinstance(priority, bool)
                or not -100 <= priority <= 100
            ):
                raise ValueError(f"Invalid DAG subtask specification for key {key!r}.")
            keys.add(key)
            normalized.append({
                "key": key,
                "title": title[:200],
                "description": description[:20000],
                "depends_on": dependencies,
                "priority": priority,
            })

        with self.backlog_lock:
            tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
            parent = next(
                (task for task in tasks if task.get("task_id") == parent_task["task_id"]),
                None,
            )
            if parent is None:
                raise ValueError(f"Parent task {parent_task['task_id']} is missing from the backlog.")
            existing_children = {
                task.get("subtask_key"): task
                for task in tasks
                if task.get("parent_task_id") == parent["task_id"]
                and task.get("subtask_key")
            }
            by_key = {
                key: {
                    "depends_on": [
                        next(
                            (
                                child_key for child_key, child in existing_children.items()
                                if child.get("task_id") == dependency_id
                            ),
                            None,
                        )
                        for dependency_id in child.get("depends_on", [])
                    ]
                }
                for key, child in existing_children.items()
            }
            by_key.update({item["key"]: item for item in normalized})
            for item in normalized:
                if item["key"] in existing_children:
                    raise ValueError(f"Subtask key {item['key']!r} already exists.")
                if any(
                    not isinstance(dependency, str) or dependency not in by_key
                    for dependency in item["depends_on"]
                ):
                    raise ValueError(f"Unknown DAG dependency in subtask {item['key']!r}.")
                if item["key"] in item["depends_on"]:
                    raise ValueError(f"Subtask {item['key']!r} cannot depend on itself.")
            visited = set()
            visiting = set()

            def visit(key):
                if key in visiting:
                    raise ValueError("The architecture model returned a cyclic subtask dependency graph.")
                if key in visited:
                    return
                visiting.add(key)
                for dependency in by_key[key]["depends_on"]:
                    if dependency is None:
                        continue
                    visit(dependency)
                visiting.remove(key)
                visited.add(key)

            for key in by_key:
                visit(key)

            next_number = max(
                (
                    int(match.group(1))
                    for task in tasks
                    if (match := re.fullmatch(r"TASK-(\d{3,})", str(task.get("task_id", ""))))
                ),
                default=0,
            ) + 1
            task_ids = {
                key: child["task_id"] for key, child in existing_children.items()
            }
            for item in normalized:
                task_ids[item["key"]] = f"TASK-{next_number:03d}"
                next_number += 1

            for item in normalized:
                task_id = task_ids[item["key"]]
                tasks.append({
                    "task_id": task_id,
                    "title": item["title"],
                    "description": (
                        f"Parent project: {parent.get('title', '')}\n"
                        f"Subtask objective: {item['description']}\n"
                        "Scope restriction: complete only this subtask; do not execute "
                        "unrelated parent-project pipelines."
                    )[:20000],
                    "iterations": int(parent.get("iterations", 3)),
                    "attached_file": "",
                    "status": "waiting_dependencies" if item["depends_on"] else "pending",
                    "priority": item["priority"],
                    "parent_task_id": parent["task_id"],
                    "subtask_key": item["key"],
                    "depends_on": [task_ids[key] for key in item["depends_on"]],
                })
            parent["subtask_ids"] = list(dict.fromkeys(task_ids.values()))
            parent["depends_on"] = list(parent["subtask_ids"])
            self._save_task_state(BACKLOG_FILE, tasks)
        with self.queue_condition:
            self.queue_condition.notify_all()
        return list(task_ids.values())

    def _refresh_dependency_states(self):
        changed = False
        blocked_tasks = []
        with self.backlog_lock:
            tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
            states = {task.get("task_id"): task.get("status") for task in tasks}
            dependents = {}
            for task in tasks:
                for dependency in task.get("depends_on", []):
                    dependents.setdefault(dependency, []).append(task)
            failed = [
                task_id for task_id, status in states.items()
                if status in {"failed", "blocked"}
            ]
            cursor = 0
            while cursor < len(failed):
                failed_id = failed[cursor]
                cursor += 1
                for dependent in dependents.get(failed_id, []):
                    if dependent.get("status") not in {"pending", "waiting_dependencies"}:
                        continue
                    dependent["status"] = "blocked"
                    dependent["blocked_reason"] = (
                        f"Prerequisite task {failed_id} failed or was blocked."
                    )
                    states[dependent.get("task_id")] = "blocked"
                    failed.append(dependent.get("task_id"))
                    blocked_tasks.append((dependent.get("task_id"), failed_id))
                    changed = True
            for task in tasks:
                status = task.get("status")
                if status not in {"pending", "waiting_dependencies", "blocked"}:
                    continue
                dependencies = task.get("depends_on", [])
                if not dependencies:
                    if status in {"waiting_dependencies", "blocked"}:
                        task["status"] = "pending"
                        task.pop("blocked_reason", None)
                        states[task.get("task_id")] = "pending"
                        changed = True
                    continue
                dependency_states = [
                    states.get(dependency) for dependency in dependencies
                ]
                if all(state == "completed" for state in dependency_states):
                    if status != "pending":
                        task["status"] = "pending"
                        task.pop("blocked_reason", None)
                        states[task.get("task_id")] = "pending"
                        changed = True
                elif status == "blocked" and not any(
                    state in {"failed", "blocked"} for state in dependency_states
                ):
                    task["status"] = "waiting_dependencies"
                    task.pop("blocked_reason", None)
                    states[task.get("task_id")] = "waiting_dependencies"
                    changed = True
                elif task.get("status") == "pending":
                    task["status"] = "waiting_dependencies"
                    states[task.get("task_id")] = "waiting_dependencies"
                    changed = True
            if changed:
                self._save_task_state(BACKLOG_FILE, tasks)
        for task_id, prerequisite_id in blocked_tasks:
            self.emit_log(
                task_id,
                f"⛔ 任務因前置任務 {prerequisite_id} 失敗而封鎖；此狀態已級聯至下游相依任務。",
            )
        return changed

    def _set_task_fields(self, task_id: str, **fields):
        with self.backlog_lock:
            tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
            task = next((item for item in tasks if item.get("task_id") == task_id), None)
            if task is None:
                raise ValueError(f"Task {task_id} is missing from the backlog.")
            task.update(fields)
            self._save_task_state(BACKLOG_FILE, tasks)
            return dict(task)

    def _collect_subtask_deliverables(self, parent_task: Dict, project_dir: Path) -> List[str]:
        child_ids = parent_task.get("subtask_ids", [])
        if not isinstance(child_ids, list) or not child_ids:
            return []
        tasks = {task.get("task_id"): task for task in self.get_all_tasks()}
        completed_ids = [
            task_id for task_id in child_ids
            if tasks.get(task_id, {}).get("status") == "completed"
        ]
        if len(completed_ids) != len(child_ids):
            return []

        collected = []
        for task_id in completed_ids:
            child = tasks[task_id]
            key = str(child.get("subtask_key") or task_id)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", key):
                raise ValueError(f"Invalid subtask key in backlog: {key!r}")
            source_dir = DELIVERY_ZONE / "projects" / task_id
            if not source_dir.is_dir():
                continue
            destination = project_dir / "subtasks" / key
            for source in source_dir.rglob("*"):
                if source.is_symlink() or not source.is_file():
                    continue
                relative = source.relative_to(source_dir)
                target = destination / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                collected.append((Path("subtasks") / key / relative).as_posix())
        return sorted(set(collected))

    @staticmethod
    def _load_subtask_python_sources(
        project_dir: Path, inherited_files: List[str]
    ) -> Dict[str, str]:
        return {
            filename: (project_dir / filename).read_text(encoding="utf-8")
            for filename in inherited_files
            if Path(filename).suffix.casefold() == ".py"
            and (project_dir / filename).is_file()
        }

    @staticmethod
    def _should_check_runtime_dag(task: Dict) -> bool:
        return not task.get("runtime_dag_checked") and not task.get("subtask_ids")

    @staticmethod
    def _build_repo_map(generated_files: Dict[str, str]) -> Dict[str, List[str]]:
        repo_map = {}
        for filename, source in generated_files.items():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            symbols = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append(node.name)
            if symbols:
                repo_map[filename] = symbols
        return repo_map

    @staticmethod
    def _archive_failure_audit(task: Dict, error: Exception, stack_trace: str) -> Path:
        task_id = str(task.get("task_id", "unknown"))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", task_id):
            raise ValueError(f"Invalid task ID for failure audit: {task_id!r}")
        audit_root = HISTORY_ZONE / "failure_audits"
        audit_path = audit_root / (
            f"{task_id}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}.json"
        )
        audit_root.mkdir(parents=True, exist_ok=True)
        project_dir = TASK_ZONE / task_id / "project"
        copied_files = []
        if project_dir.is_dir():
            snapshot_dir = audit_root / f"{audit_path.stem}_code"
            for source in project_dir.rglob("*"):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(project_dir)
                target = snapshot_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied_files.append(relative.as_posix())
        payload = {
            "task": {
                key: task.get(key)
                for key in (
                    "task_id", "title", "description", "iterations",
                    "attached_file", "parent_task_id", "subtask_key",
                )
                if key in task
            },
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": task.get("resume_stage"),
            "error_type": type(error).__name__,
            "error": str(error),
            "stack_trace": stack_trace,
            "project_snapshot": copied_files,
        }
        OmniEnterpriseDaemon._save_task_state(audit_path, payload)
        return audit_path

    @staticmethod
    def _profile_tabular_attachment(path: Path) -> str:
        suffix = path.suffix.casefold()
        def summarize_batches(batches, sheet_name=None):
            column_types = {}
            missing_counts = {}
            sample_rows = []
            rows_profiled = 0
            process = psutil.Process()
            baseline_rss = process.memory_info().rss
            for frame in batches:
                if frame is None or frame.empty:
                    continue
                if len(frame.columns) > 1000:
                    raise ValueError("Tabular profiling is limited to 1000 columns per sheet.")
                if process.memory_info().rss - baseline_rss > 512 * 1024 * 1024:
                    raise MemoryError("Tabular profiling exceeded its 512 MiB memory-growth limit.")
                rows_profiled += len(frame)
                for column in frame.columns:
                    key = str(column)
                    column_types.setdefault(key, str(frame[column].dtype))
                    missing_counts[key] = missing_counts.get(key, 0) + int(frame[column].isna().sum())
                if len(sample_rows) < 5:
                    remaining = 5 - len(sample_rows)
                    sample_rows.extend(json.loads(
                        frame.head(remaining).to_json(
                            orient="records", force_ascii=False, date_format="iso"
                        )
                    ))
            columns = [
                {
                    "name": name,
                    "dtype": column_types[name],
                    "missing": missing_counts.get(name, 0),
                    "missing_pct": round(
                        missing_counts.get(name, 0) * 100 / rows_profiled, 1
                    ) if rows_profiled else 0.0,
                }
                for name in column_types
            ]
            profile = {
                "rows_profiled": rows_profiled,
                "rows_profiled_note": (
                    "first 10000 rows per sheet at most; missing counts are for this profiled sample"
                ),
                "columns": columns,
                "sample_rows": sample_rows,
            }
            if sheet_name is not None:
                profile["sheet"] = sheet_name
            return profile

        if suffix == ".csv":
            last_decode_error = None
            for encoding in ("utf-8-sig", "cp950", "big5"):
                try:
                    chunks = pd.read_csv(
                        path, encoding=encoding, chunksize=500, nrows=10000,
                        low_memory=True,
                    )
                    profile = summarize_batches(chunks)
                    break
                except UnicodeDecodeError as exc:
                    last_decode_error = exc
            else:
                raise ValueError(
                    f"Could not decode CSV attachment {path.name}: {last_decode_error}"
                )
            result = {"file": path.name, **profile}
        elif suffix in {".xlsx", ".xls"}:
            sheets = []
            with pd.ExcelFile(path) as workbook:
                for sheet_name in workbook.sheet_names:
                    frame = pd.read_excel(
                        workbook, sheet_name=sheet_name, nrows=10000
                    )
                    sheets.append(summarize_batches([frame], sheet_name))
            result = {"file": path.name, "sheets": sheets}
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as parquet
            except ImportError as exc:
                raise ValueError(
                    "Parquet profiling requires the optional pyarrow package."
                ) from exc
            parquet_file = parquet.ParquetFile(path)
            def parquet_batches():
                rows_remaining = 10000
                for batch in parquet_file.iter_batches(batch_size=500):
                    if rows_remaining <= 0:
                        break
                    limited = batch.slice(0, rows_remaining)
                    yield limited.to_pandas()
                    rows_remaining -= limited.num_rows

            batches = parquet_batches()
            result = {"file": path.name, **summarize_batches(batches)}
        else:
            raise ValueError(f"Unsupported tabular attachment type: {suffix or '(no extension)'}")
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    def record_lesson(self, task_id: str, rule: str):
        if not rule.strip():
            return
        if LESSONS_FILE.is_file():
            data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
        else:
            data = []
        if not isinstance(data, list):
            raise ValueError(f"Lessons file must contain a JSON list: {LESSONS_FILE}")
        normalized_rule = rule.strip()[:2000]
        if any(
            isinstance(item, dict)
            and item.get("task_id") == task_id
            and item.get("rule") == normalized_rule
            for item in data
        ):
            return
        data.append({
            "task_id": task_id,
            "rule": normalized_rule,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(data) > 500:
            data = data[-500:]
        self._save_task_state(LESSONS_FILE, data)

    def update_task_status(self, task_id: str, status: str):
        elapsed_status = status in {
            "completed", "failed", "pending", "paused", "waiting_dependencies",
        }
        with self.backlog_lock:
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                for t in tasks:
                    if t.get("task_id") == task_id:
                        t["status"] = status
                        if status == "processing":
                            t.setdefault("started_at", time.time())
                            t["run_started_at"] = time.time()
                        elif elapsed_status:
                            started = t.pop("run_started_at", None)
                            if started is not None:
                                t["duration_sec"] = round(
                                    float(t.get("duration_sec", 0) or 0)
                                    + max(0.0, time.time() - float(started)),
                                    1,
                                )
                            if status in {"completed", "failed"}:
                                t["finished_at"] = time.time()
                self._save_task_state(BACKLOG_FILE, tasks)
            except (OSError, ValueError, TypeError) as exc:
                self.emit_log(task_id, f"⚠ 無法原子更新任務狀態為 {status}：{exc}")

    def estimate_total_runtime_sec(
        self,
        total_elapsed_sec: int,
        task_elapsed_sec: int,
        tasks: Optional[List[Dict]] = None,
        active_task: Optional[Dict] = None,
    ) -> Optional[int]:
        if tasks is None:
            tasks = self.get_all_tasks()
        active_task = active_task or {}
        active_task_id = active_task.get("task_id", self.current_task_id)
        active_elapsed = active_task.get("elapsed_sec", task_elapsed_sec)
        active_started_at = active_task.get("started_at", self.task_start_ts)
        completed_durations = [
            float(task["duration_sec"])
            for task in tasks
            if task.get("status") == "completed"
            and isinstance(task.get("duration_sec"), (int, float))
            and task["duration_sec"] > 0
        ]
        average_duration = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations else None
        )
        pending_count = sum(
            task.get("status") in {"pending", "paused", "waiting_dependencies"}
            and task.get("task_id") != active_task_id
            for task in tasks
        )

        has_active_task = active_task_id != "-" and active_started_at is not None
        if (has_active_task or pending_count) and average_duration is None:
            return None
        if average_duration is None:
            return total_elapsed_sec

        current_remaining = (
            max(0.0, average_duration - active_elapsed)
            if has_active_task else 0.0
        )
        remaining = current_remaining + pending_count * average_duration
        return max(0, round(total_elapsed_sec + remaining))

    def peek_pending_tasks(self) -> List[Dict]:
        tasks = self.get_all_tasks()
        completed = {
            task.get("task_id") for task in tasks if task.get("status") == "completed"
        }
        return [
            task for task in tasks
            if task.get("status") == "pending"
            and all(dependency in completed for dependency in task.get("depends_on", []))
        ]

    # ================= 全能任務執行主流程（貪婪塞滿記憶體 + 全模型協同） =================
    def execute_omnipotent_task(self, task: dict):
        self._ensure_task_runtime()
        task_id = task["task_id"]
        title = task["title"]
        desc = task["description"]
        attached_file = task.get("attached_file", "")
        iterations = int(task.get("iterations", 3))

        self.current_task_id = task_id
        started_at = time.time()
        self.task_start_ts = started_at
        self.task_paused_accum = 0.0
        self.task_prior_duration_sec = float(task.get("duration_sec", 0) or 0)
        with self._task_runtime_lock:
            self._task_runtime[task_id] = {
                "title": title,
                "started_at": started_at,
                "prior_duration_sec": self.task_prior_duration_sec,
                "paused_accum": 0.0,
                "pause_started": None,
                "stage": "Starting",
                "progress_pct": 0,
            }
        self.update_task_status(task_id, "processing")

        try:
            final_report = DELIVERY_ZONE / "reports" / task_id / "final_delivery.md"
            if final_report.is_file():
                completion_marker = f"<!-- TASK-COMPLETE: {task_id} -->"
                if completion_marker not in final_report.read_text(encoding="utf-8"):
                    raise RuntimeError(
                        f"An incomplete final report already exists at {final_report}; "
                        "it was preserved and needs manual review before retrying."
                    )
                self.update_task_status(task_id, "completed")
                self.current_task_progress_pct = 100
                self.current_task_stage = "Completed (Recovered)"
                self.emit_log(
                    task_id,
                    "✔ 偵測到完整最終交付報告；已恢復任務完成狀態，未重新產生或覆寫交付檔。",
                    event_type="completed",
                )
                return

            fb = self.checkpoint(task_id, "Stage 1: 總架構師自主路由與任務拆解", 8)
            accumulated_ctx = (
                f"任務標題：{title}\n使用者需求：{desc}\n附件檔案：{attached_file}\n"
                f"{self.get_lessons(title, desc)}\n{fb}"
            )
            inferred_routes = self._infer_task_routes(title, desc, attached_file)

            self.emit_log(task_id, "🧠 [階段 1] 總架構師正在自主分析需求並決定執行管線與 Context 視窗...")
            router_prompt = f"""請分析以下需求與附件，自主決定需要啟動哪些功能管線（可複選）。
本機需求分類線索：{json.dumps(inferred_routes, ensure_ascii=False)}
{accumulated_ctx}

輸出嚴格 JSON 格式：
{{
  "suggested_num_ctx": 8192,
  "needs_vision_analysis": false,
  "needs_data_cleaning": false,
  "needs_game_development": false,
  "needs_film_production": false,
  "needs_image_creation": false,
  "needs_software_project": false,
  "needs_direct_qa_report": true,
  "subtasks": [],
  "reasoning": "規劃說明"
}}
複雜全端或企業級軟體需求，請在 subtasks 提供可執行的 DAG 子任務；每項格式為
{{"key":"database","title":"資料庫 Schema","description":"具體可交付範圍","depends_on":[],"priority":0}}。
depends_on 使用其他子任務的 key；不得形成循環，依賴完成後可並行的工作應設為同一批可執行節點。
簡單需求或非軟體任務輸出空陣列。不要將一般問答誤判為軟體專案。爬蟲、資料工具、網站、程式與自動化應使用軟體管線；圖片創作使用圖片管線；一般研究、建議或分析只需報告。各管線可以複選，但只啟動需求明確要求或必要的管線。"""
            router_ctx = max(4096, self.pool.decide_dynamic_ctx(router_prompt))
            route_raw = self._generate(
                role="architect", prompt=router_prompt, task_id=task_id,
                action_desc="自主判斷任務管線與 Context", format_json=True, num_ctx=router_ctx
            )
            try:
                plan_flags = json.loads(route_raw)
                if not isinstance(plan_flags, dict):
                    raise ValueError("The task router response must be a JSON object.")
            except Exception:
                plan_flags = {
                    "suggested_num_ctx": 8192,
                    "needs_vision_analysis": bool(attached_file and Path(attached_file).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]),
                    "needs_data_cleaning": inferred_routes["needs_data_cleaning"],
                    "needs_game_development": inferred_routes["needs_game_development"],
                    "needs_film_production": self._is_film_request(title, desc),
                    "needs_image_creation": inferred_routes["needs_image_creation"],
                    "needs_software_project": inferred_routes["needs_software_project"],
                    "needs_direct_qa_report": True,
                    "reasoning": "依需求路由至必要的任務管線"
                }

            try:
                decision_routes = self.pool.classify_task_routes(title, desc, attached_file)
            except RuntimeError as exc:
                self.emit_log(task_id, f"⚠ 決策模型路由暫不可用，保留架構模型規劃並繼續：{exc}")
                decision_routes = {}
            for decision_key, plan_key in [
                ("vision", "needs_vision_analysis"),
                ("data_cleaning", "needs_data_cleaning"),
                ("game", "needs_game_development"),
                ("film", "needs_film_production"),
                ("software", "needs_software_project"),
            ]:
                if decision_routes.get(decision_key):
                    plan_flags[plan_key] = True
            for route_key, is_needed in inferred_routes.items():
                if is_needed:
                    plan_flags[route_key] = True

            film_requested = self._is_film_request(title, desc)
            if film_requested:
                plan_flags["needs_film_production"] = True
                explicit_software = any(word in (title + desc).lower() for word in [
                    "software project", "python module", "web app", "build an application",
                    "軟體專案", "程式專案", "系統開發", "網站開發"
                ])
                explicit_game = any(word in (title + desc).lower() for word in [
                    "game development", "create a game", "make a game", "遊戲開發", "製作遊戲"
                ])
                if not explicit_software:
                    plan_flags["needs_software_project"] = False
                if not explicit_game:
                    plan_flags["needs_game_development"] = False

            suggested_ctx = max(4096, int(plan_flags.get("suggested_num_ctx", router_ctx)))
            context_capacity = self.pool.context_capacity("architect")
            decided_ctx = min(suggested_ctx, context_capacity) if context_capacity else router_ctx
            decided_ctx = max(4096, decided_ctx)
            self.emit_log(task_id, f"✔ 自主規劃完成 (動態 Ctx: {decided_ctx})：{plan_flags.get('reasoning', '')}")
            if plan_flags.get("needs_software_project") and not task.get("subtask_ids"):
                spawned_ids = self._spawn_subtasks(task, plan_flags.get("subtasks", []))
                if spawned_ids:
                    self.update_task_status(task_id, "waiting_dependencies")
                    self.emit_log(
                        task_id,
                        f"🧩 已建立 DAG 子任務 {', '.join(spawned_ids)}；前置相依完成後將自動續行整合。",
                    )
                    return

            # 2. 視覺影像分析管線
            if plan_flags.get("needs_vision_analysis") and attached_file and Path(attached_file).exists():
                vision_path = self._task_delivery_dir("reports", task_id) / "vision_analysis.md"
                if vision_path.is_file():
                    self.emit_log(task_id, "📦 偵測到既有視覺分析結果，保留原檔並用於續跑。")
                    accumulated_ctx += f"\n【已完成的視覺解析報告】：\n{vision_path.read_text(encoding='utf-8')}\n"
                else:
                    fb = self.checkpoint(task_id, "Stage 2: 多模態視覺感知解析", 20)
                    accumulated_ctx += fb
                    self.emit_log(task_id, "👁 [視覺管線] 啟動視覺專員模型解析影像...")
                    vis_res = self._generate(
                        role="vision",
                        prompt=f"請根據需求（{desc}）深度解析此圖片之數據、結構與細節：",
                        task_id=task_id, action_desc="視覺多模態解析", images=[attached_file], num_ctx=decided_ctx
                    )
                    accumulated_ctx += f"\n【視覺解析報告】：\n{vis_res}\n"
                    if not self._write_delivery_text(vision_path, vis_res):
                        self.emit_log(task_id, "📦 保留既有視覺分析報告，不覆寫交付檔。")

            # 3. 資料清理管線
            if plan_flags.get("needs_data_cleaning"):
                fb = self.checkpoint(task_id, "Stage 3: 自動資料清理與沙盒驗證", 32)
                accumulated_ctx += fb
                clean_dir = self._task_delivery_dir("cleaned_data", task_id)
                out_csv = clean_dir / "cleaned.csv"
                if out_csv.is_file():
                    self.emit_log(task_id, "📦 偵測到既有清理結果，保留原檔，不重新覆寫。")
                else:
                    work_dir = TASK_ZONE / task_id
                    work_dir.mkdir(parents=True, exist_ok=True)
                    staged_csv = work_dir / "cleaned.csv"
                    attachment_profile = ""
                    if attached_file and Path(attached_file).is_file():
                        attachment_profile = self._profile_tabular_attachment(Path(attached_file))
                    clean_prompt = f"""{accumulated_ctx}
附件表格實際結構（欄位名稱、dtype、缺失統計與前五筆樣本；缺失統計以最多前 10000 列估算）：
{attachment_profile or "沒有上傳可讀取的表格附件。"}
請只依上述實際欄位名稱撰寫純 Python 腳本處理資料，輸出至 `{staged_csv.as_posix()}`，末端包含 assert 驗證。只輸出純 Python 代碼。"""
                    script_p = work_dir / "data_job.py"
                    data_work_state = work_dir / "data_progress.json"
                    best_csv_path = work_dir / "best_cleaned.csv"
                    best_csv = None
                    best_code = ""
                    completed_attempt = 0
                    if data_work_state.is_file() and best_csv_path.is_file():
                        try:
                            saved_state = json.loads(data_work_state.read_text(encoding="utf-8"))
                            completed_attempt = max(0, int(saved_state.get("completed_attempt", 0)))
                            best_code = str(saved_state.get("best_code", ""))
                            best_csv = best_csv_path.read_bytes()
                            clean_prompt += (
                                "\n已完成上一輪並通過資料驗證，請在保留輸出契約下繼續改善。"
                                f"\n目前程式：\n{best_code[:6000]}"
                            )
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            self.emit_log(task_id, "⚠ 資料清理續跑狀態無效，將重新執行驗證流程。")
                            completed_attempt = 0
                            best_code = ""
                            best_csv = None
                    last_attempt_succeeded = completed_attempt >= iterations
                    for attempt in range(completed_attempt + 1, iterations + 1):
                        self.checkpoint(task_id)
                        c_raw = self._generate(
                            role="coder", prompt=clean_prompt, task_id=task_id,
                            action_desc=f"資料清理實作與驗證 ({attempt}/{iterations})",
                            num_ctx=decided_ctx, output_validator=self._is_valid_python_output,
                        )
                        script_p.write_text(c_raw.replace("```python", "").replace("```", "").strip(), encoding="utf-8")
                        staged_csv.unlink(missing_ok=True)
                        res = self._run_python_script(script_p)
                        if res["success"] and staged_csv.is_file():
                            best_csv = staged_csv.read_bytes()
                            best_code = script_p.read_text(encoding="utf-8")
                            best_csv_path.write_bytes(best_csv)
                            completed_attempt = attempt
                            last_attempt_succeeded = True
                            self._save_task_state(data_work_state, {
                                "completed_attempt": completed_attempt,
                                "best_code": best_code,
                            })
                            clean_prompt += (
                                f"\n目前已驗證通過。第 {attempt}/{iterations} 輪請在保留輸出契約下，"
                                "進一步改善資料品質、缺值處理、欄位正規化、重複資料與可追溯性。"
                                f"\n上一輪程式：\n{best_code[:6000]}"
                            )
                        else:
                            last_attempt_succeeded = False
                            failure = res["stderr"] or "The output file was not created."
                            self.emit_log(
                                task_id,
                                f"⚠ 資料清理第 {attempt}/{iterations} 輪未通過驗證；"
                                f"保留最近一次通過驗證的成果。{failure[:500]}",
                            )
                            clean_prompt += (
                                f"\n驗證錯誤（第 {attempt} 輪）：\n"
                                f"{failure}"
                            )
                    if best_csv is not None:
                        if not self._publish_delivery_bytes(out_csv, best_csv):
                            self.emit_log(task_id, "📦 偵測到既有清理結果，保留原檔，不覆寫。")
                        if last_attempt_succeeded:
                            self.emit_log(
                                task_id,
                                f"✔ 資料清理第 {completed_attempt}/{iterations} 輪完成並通過驗證。"
                                f"產出：{out_csv.name}"
                            )
                        else:
                            self.emit_log(
                                task_id,
                                f"⚠ 資料清理最近一輪未通過；已交付第 {completed_attempt} 輪"
                                f"最近通過驗證的成果：{out_csv.name}"
                            )
                    else:
                        raise RuntimeError(
                            f"Data-cleaning output did not pass validation: {res.get('stderr', '')}"
                        )

            if plan_flags.get("needs_image_creation"):
                image_dir = DELIVERY_ZONE / "illustrations" / task_id
                image_path = image_dir / "image.png"
                if image_path.is_file():
                    if not self.image_forge.validate_png(image_path):
                        raise RuntimeError(
                            f"An invalid image delivery already exists and will not be overwritten: {image_path}"
                        )
                    self.emit_log(task_id, "📦 偵測到既有圖片交付，保留原檔並載入成果。")
                else:
                    self.checkpoint(task_id, "Stage 3.5: 多輪圖片設計與高解析度繪製", 38)
                    spec = None
                    image_work_state = TASK_ZONE / task_id / "image_progress.json"
                    completed_round = 0
                    if image_work_state.is_file():
                        try:
                            saved_state = json.loads(image_work_state.read_text(encoding="utf-8"))
                            saved_spec = saved_state.get("spec")
                            if isinstance(saved_spec, dict) and isinstance(saved_spec.get("shapes"), list):
                                spec = saved_spec
                                completed_round = max(0, int(saved_state.get("completed_round", 0)))
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            self.emit_log(task_id, "⚠ 圖片設計續跑資料無效，將從最後可用設計重新演化。")
                    for evo in range(completed_round + 1, iterations + 1):
                        fb = self.checkpoint(
                            task_id,
                            f"圖片設計演化 ({evo}/{iterations})",
                            38 + int(evo / iterations * 6),
                        )
                        refinement = (
                            f"\n上一輪設計規格：{json.dumps(spec, ensure_ascii=False)}\n"
                            "請保留原始主題與核心構圖，在此基礎上增加更有辨識度的層次、色彩對比、"
                            "前中後景與細節，不要只重述上一版。"
                            if spec else "請從零設計完整、原創且構圖清楚的插畫。"
                        )
                        prompt = f"""依使用者需求為本機 PNG 插畫撰寫一份視覺規格 JSON。
需求：{title}\n{desc[:3000]}{refinement}
只輸出 JSON，格式為：
{{"title":"短標題","subtitle":"簡短副標","background_top":"#112233","background_bottom":"#445566",
"shapes":[{{"type":"circle|ellipse|rect|triangle|star|polygon|path|line","x":0.5,"y":0.5,"width":0.2,"height":0.2,"color":"#RRGGBB","gradient_end":"#RRGGBB","glow":true,"closed":true,"points":[[0.1,0.2],[0.2,0.4],[0.3,0.2]]}}]}}
形狀位置、尺寸與向量路徑點使用 0 到 1 的比例；提供 12 到 36 個分層向量物件，包含自訂多邊形／曲線路徑、漸層填色、發光及前中後景；使用色彩和構圖描述，不要輸出程式碼。"""
                        raw = self._generate(
                            role="architect", prompt=prompt, task_id=task_id,
                            action_desc=f"圖片構圖與美術品質改善 ({evo}/{iterations})",
                            format_json=True, num_ctx=decided_ctx,
                        )
                        parsed = json.loads(raw)
                        if not isinstance(parsed, dict) or not isinstance(parsed.get("shapes"), list):
                            raise ValueError("The image design model returned an invalid illustration specification.")
                        spec = parsed
                        completed_round = evo
                        self._save_task_state(image_work_state, {
                            "completed_round": completed_round,
                            "spec": spec,
                        })
                    if spec is None:
                        raise RuntimeError("No valid image design is available to render.")
                    with tempfile.TemporaryDirectory(
                        prefix=f"{task_id}_image_", dir=str(TASK_ZONE)
                    ) as staging:
                        staged_dir = Path(staging) / "illustration"
                        staged_result = self.image_forge.render(
                            staged_dir, spec, iterations=iterations
                        )
                        image_dir.parent.mkdir(parents=True, exist_ok=True)
                        self._publish_staged_directory(staged_dir, image_dir)
                    image_result = {
                        "image": str(image_path),
                        "specification": str(image_dir / Path(staged_result["specification"]).name),
                    }
                    self.emit_log(
                        task_id,
                        f"✔ 高解析度圖片已產出：{Path(image_result['image']).name} ({self.image_forge.quality_size(iterations)}×{self.image_forge.quality_size(iterations)})",
                        event_type="image_preview",
                        extra={"image_url": f"/delivery/illustrations/{task_id}/image.png"},
                    )

            # 4. 遊戲製作管線（貪婪塞滿 40GB 記憶體，同時派出 N 個不同模型並行開發）
            if plan_flags.get("needs_game_development"):
                fb = self.checkpoint(task_id, "Stage 4: 遊戲素材合成與全模型並行開發", 45)
                accumulated_ctx += fb
                game_dir = DELIVERY_ZONE / "games" / task_id
                assets_dir = game_dir / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)
                game_work_state = TASK_ZONE / task_id / "game_progress.json"
                saved_state = {}
                if game_work_state.is_file():
                    try:
                        candidate_state = json.loads(game_work_state.read_text(encoding="utf-8"))
                        if isinstance(candidate_state, dict):
                            saved_state = candidate_state
                    except (OSError, json.JSONDecodeError):
                        self.emit_log(task_id, "⚠ 遊戲續跑狀態無效，將使用可用的既有素材重新開始。")
                manifest = saved_state.get("asset_manifest")
                if isinstance(manifest, list):
                    manifest = self._normalize_game_asset_manifest({"assets": manifest})
                else:
                    asset_prompt = f"""依照使用者的遊戲需求，列出真正有助於玩法或視覺呈現的本機程序化素材；不需要的素材請省略，允許空清單。
需求：{desc[:3000]}
只輸出 JSON：{{"assets":[{{"type":"sprite 或 sfx","filename":"描述性英文檔名.png 或 .wav","style":"素材外觀或聲音描述","color":"#RRGGBB"}}]}}
最多 16 個，檔名不可含路徑；不要使用固定角色、兵種或特定玩法的預設清單。"""
                    asset_raw = self._generate(
                        role="architect", prompt=asset_prompt, task_id=task_id,
                        action_desc="依遊戲需求規劃素材清單",
                        format_json=True, num_ctx=decided_ctx,
                    )
                    try:
                        manifest = self._normalize_game_asset_manifest(json.loads(asset_raw))
                    except (json.JSONDecodeError, TypeError):
                        manifest = []
                        self.emit_log(task_id, "⚠ 遊戲素材規格無效，將以 Canvas 程序化繪圖繼續。")
                    saved_state["asset_manifest"] = manifest
                    self._save_task_state(game_work_state, saved_state)
                created_assets = self.forge.fulfill_asset_manifest(
                    assets_dir,
                    manifest,
                    sprite_size=min(512, 64 + (iterations - 1) * 24),
                )
                if len(created_assets) != len(manifest):
                    raise RuntimeError("The game asset manifest could not be completed.")
                self.emit_log(
                    task_id,
                    f"✔ 遊戲素材已備妥：{len(created_assets)} 個檔案（最高素材尺寸 {min(512, 64 + (iterations - 1) * 24)}px）",
                    event_type="game_assets",
                    extra={"assets": created_assets},
                )
                asset_context = json.dumps([
                    {
                        "name": name,
                        "url": f"assets/{name}",
                        "type": "audio" if name.casefold().endswith(".wav") else "image",
                    }
                    for name in created_assets
                ], ensure_ascii=False)

                game_module_defaults = [
                    ("Gameplay_Core", "實作符合使用者要求的核心玩法、狀態轉移、勝負或完成條件；不得預設遊戲類型。"),
                    ("Player_Interaction", "實作適合本遊戲的鍵盤、滑鼠、觸控或互動控制，並使用共用遊戲介面。"),
                    ("World_Content", "實作使用者要求的關卡、棋盤、物件、角色、題目或內容資料。"),
                    ("Visual_Presentation", "實作符合遊戲主題的 Canvas 畫面、動畫、回饋及響應式呈現。"),
                    ("Audio_Feedback", "提供遊戲狀態、音效、計分或可讀性回饋，避免依賴未隨交付的資源。"),
                    ("Accessibility_QA", "實作重新開始、操作說明、無障礙與邊界情況處理，並檢查模組整合。")
                ]

                completed_round = 0
                if (
                    (TASK_ZONE / f"{task_id}_game_candidate.html").is_file()
                    and (TASK_ZONE / f"{task_id}_game_bundle.js").is_file()
                ):
                    try:
                        completed_round = max(0, int(saved_state.get("completed_round", 0)))
                    except (TypeError, ValueError):
                        self.emit_log(task_id, "⚠ 遊戲續跑輪數無效，將從最後可用版本重新演化。")
                previous_modules = saved_state.get("modules", {})
                if not isinstance(previous_modules, dict):
                    previous_modules = {}
                validation_feedback = str(saved_state.get("validation_feedback", ""))
                try:
                    attempted_round = max(
                        completed_round,
                        int(saved_state.get("attempted_round", completed_round)),
                    )
                except (TypeError, ValueError):
                    attempted_round = completed_round
                    self.emit_log(task_id, "⚠ 遊戲續跑驗證輪數無效，將從最後通過的版本繼續。")
                for evo in range(attempted_round + 1, iterations + 1):
                    html_path = game_dir / "index.html"
                    if html_path.is_file():
                        existing_validation = self.sandbox.validate_html_game(html_path)
                        if existing_validation["success"]:
                            self.emit_log(task_id, "📦 偵測到既有有效遊戲成果；保留原檔，不重新覆寫。")
                            break
                        raise FileExistsError(
                            f"An invalid game delivery already exists and will not be overwritten: {html_path}"
                        )
                    fb = self.checkpoint(task_id, f"Stage 4.{evo}: 滿載多模型並行遊戲演化 ({evo}/{iterations})", 45 + int((evo / iterations) * 28))
                    accumulated_ctx += fb

                    # 使用目前預算內可運行的不同模型建立並行工作槽。
                    swarm_models = self.pool.plan_greedy_swarm(num_ctx=decided_ctx)
                    swarm_names = [m["name"] for m in swarm_models]
                    total_swarm_gb = round(sum(m["slot_memory_gb"] for m in swarm_models), 1)
                    self.emit_log(
                        task_id,
                        f"🔥 [Round {evo}/{iterations} 記憶體滿載調度] 已規劃同時啟動 {len(swarm_models)} 個模型槽位 (預估總載入: {total_swarm_gb} GB / 上限 {self.pool.get_hardware_budget()} GB)：{swarm_names}"
                    )

                    previous_bundle_path = TASK_ZONE / f"{task_id}_game_bundle.js"
                    previous_bundle = previous_bundle_path.read_text(encoding="utf-8") if previous_bundle_path.exists() else ""
                    module_prompt = f"""請將以下遊戲需求拆解為恰好 {len(swarm_models)} 個互不重複、可並行實作的 JavaScript 子模組。
使用者需求：{desc[:4000]}
目前版本（若有）:\n{previous_bundle[:6000]}
這是第 {evo}/{iterations} 輪。若已有版本，需提出具體新功能、修正或視覺／可用性改善，保留所有已有功能，不可只換名稱或重複輸出原內容。
候選職責：{json.dumps(game_module_defaults, ensure_ascii=False)}
輸出嚴格 JSON：{{"modules":[{{"title":"短英文識別名稱","instruction":"具體、獨立的實作規格"}}]}}
第一個模組必須負責初始化遊戲狀態、主要玩法和可遊玩循環。其他模組須依需求擴充不同系統；不得套用預設推塔、兵種、時代或主堡等玩法。
各模組都必須遵循以下共用 JavaScript 契約：以 `window.GameModules = window.GameModules || {{}}; window.GameModules["game-N"] = {{init(api), update(dt, api), render(api), onKeyDown(event, api), onPointerDown(event, api), resize(api), reset(api)}}` 註冊；`api` 提供 `canvas`、`context`、可持久化的 `state`、`setStatus(message)`、`addControl(label, callback)`、`reset()`。每個模組至少實作 `init(api)`，其餘方法可省略。不得使用未定義的全域變數或直接覆寫其他模組。
每個模組必須涵蓋不同需求；不得少於或多於 {len(swarm_models)} 個。"""
                    module_raw = self._generate(
                        role="architect", prompt=module_prompt, task_id=task_id,
                        action_desc=f"將遊戲需求切分為 {len(swarm_models)} 個並行模組",
                        format_json=True, num_ctx=decided_ctx
                    )
                    try:
                        module_payload = json.loads(module_raw)
                        proposed_modules = module_payload.get("modules", []) if isinstance(module_payload, dict) else []
                    except (json.JSONDecodeError, TypeError):
                        proposed_modules = []
                    game_submodules = self._normalize_parallel_work_items(
                        proposed_modules,
                        len(swarm_models),
                        game_module_defaults,
                    )

                    def _run_game_submodule(idx: int, mod_title: str, mod_spec: str, target_model: str) -> tuple[int, str]:
                        self.checkpoint(task_id)
                        module_id = f"game-{idx + 1}"
                        previous_module = str(previous_modules.get(module_id, ""))
                        p = f"""請使用純 JavaScript 實作遊戲子模組【{mod_title}】，模組 ID 為 `{module_id}`。
任務背景：{desc[:800]}
{fb}
所有模組共用契約：`window.GameModules["{module_id}"] = {{ init(api), update(dt, api), render(api), onKeyDown(event, api), onPointerDown(event, api), resize(api), reset(api) }}`；必須實際註冊自己的物件並實作 `init(api)`。`api` 提供 `canvas`、`context`、通用 `state`、`assets`、`assetUrls`、`loadAsset(name)`、`setStatus(message)`、`addControl(label, callback)`、`reset()`。可用素材：{asset_context}。不得假設存在 RTS 狀態或預設遊戲規則。
本模組負責：{mod_spec}
上一輪此模組程式碼（空白代表首次實作）：\n{previous_module[:8000]}
{self._game_validation_guidance(validation_feedback)}
本輪品質目標：第 {evo}/{iterations} 輪；在不破壞既有行為下增加需求覆蓋、錯誤處理、平衡、效能或無障礙品質。
要求：完整保留所有共用公開介面；直接輸出可執行的純 JavaScript 代碼，不加 HTML 或 Markdown。"""
                        res = self._generate(
                            role="coder", prompt=p, task_id=task_id,
                            action_desc=f"並行開發 {mod_title} (R{evo})",
                            num_ctx=decided_ctx, preferred_model=target_model,
                            output_validator=self._is_valid_javascript_output,
                        )
                        return idx, res.replace("```javascript", "").replace("```js", "").replace("```", "").strip()

                    js_parts = {}
                    with ThreadPoolExecutor(max_workers=len(game_submodules)) as ex:
                        futures = []
                        for i, (m_title, m_spec) in enumerate(game_submodules):
                            assigned_m = swarm_models[i]["name"]
                            futures.append(ex.submit(_run_game_submodule, i, m_title, m_spec, assigned_m))
                        for fut in as_completed(futures):
                            module_index, module_code = fut.result()
                            js_parts[module_index] = module_code

                    final_html = self._assemble_playable_game_html(
                        title, created_assets,
                        "\n\n".join(js_parts[index] for index in sorted(js_parts))
                    )
                    previous_bundle_path.write_text(
                        "\n\n".join(js_parts[index] for index in sorted(js_parts)),
                        encoding="utf-8",
                    )
                    candidate_path = TASK_ZONE / f"{task_id}_game_candidate.html"
                    temp_html = candidate_path.with_name(f"{task_id}_game_candidate.tmp.html")
                    temp_html.write_text(final_html, encoding="utf-8")
                    validation = self.sandbox.validate_html_game(temp_html)
                    if not validation["success"]:
                        temp_html.unlink(missing_ok=True)
                        validation_feedback = str(
                            validation.get("stderr") or "Unknown game validation failure."
                        )
                        previous_modules = {
                            f"game-{index + 1}": code for index, code in js_parts.items()
                        }
                        self._save_task_state(game_work_state, {
                            **saved_state,
                            "completed_round": completed_round,
                            "attempted_round": evo,
                            "modules": previous_modules,
                            "validation_feedback": validation_feedback,
                        })
                        if candidate_path.is_file():
                            self.emit_log(
                                task_id,
                                f"⚠ 第 {evo} 輪遊戲未通過驗證，保留先前通過的版本並於下一輪修復：{validation_feedback}"
                            )
                        else:
                            self.emit_log(
                                task_id,
                                f"⚠ 第 {evo} 輪遊戲未通過驗證，將把錯誤交給下一輪修復：{validation_feedback}"
                            )
                        continue
                    temp_html.replace(candidate_path)
                    completed_round = evo
                    attempted_round = evo
                    previous_modules = {
                        f"game-{index + 1}": code for index, code in js_parts.items()
                    }
                    saved_state = {
                        **saved_state,
                        "completed_round": completed_round,
                        "attempted_round": attempted_round,
                        "modules": previous_modules,
                        "validation_feedback": "",
                    }
                    validation_feedback = ""
                    self._save_task_state(game_work_state, saved_state)
                    self.emit_log(task_id, f"✔ 遊戲第 {evo}/{iterations} 輪已通過結構驗證，已保留為下一輪改善基礎。")

                html_path = game_dir / "index.html"
                if not html_path.exists():
                    candidate_path = TASK_ZONE / f"{task_id}_game_candidate.html"
                    if candidate_path.is_file():
                        if self._publish_delivery_bytes(html_path, candidate_path.read_bytes()):
                            self.emit_log(
                                task_id, f"🎮 遊戲完成 {iterations} 輪演化並交付！",
                                event_type="game_preview",
                                extra={"preview_url": f"/delivery/games/{task_id}/index.html?t={int(time.time())}"}
                            )
                    else:
                        raise RuntimeError(
                            f"No validated game draft is available for delivery: {candidate_path}"
                        )

            if plan_flags.get("needs_film_production"):
                self.checkpoint(task_id, "Stage 5: 電影劇本、分鏡與動畫影片製作", 75)
                self.emit_log(task_id, "🎬 [電影製作] 正在根據需求編寫分場劇本與視覺腳本...")
                film_dir = DELIVERY_ZONE / "films" / task_id
                existing_film = film_dir / "film.mp4"
                film_work_state = TASK_ZONE / task_id / "film_progress.json"
                completed_round = 0
                if film_dir.exists() and any(film_dir.iterdir()):
                    if existing_film.is_file() and (film_dir / "index.html").is_file():
                        self.emit_log(task_id, "📦 偵測到既有完整電影交付，保留原檔，不重新覆寫。")
                        film_spec = None
                    else:
                        raise FileExistsError(
                            f"An incomplete film delivery already exists and will not be overwritten: {film_dir}"
                        )
                else:
                    film_spec = {}
                    if film_work_state.is_file():
                        try:
                            saved_state = json.loads(film_work_state.read_text(encoding="utf-8"))
                            saved_spec = saved_state.get("spec")
                            if isinstance(saved_spec, dict) and isinstance(saved_spec.get("scenes"), list):
                                film_spec = saved_spec
                                completed_round = max(0, int(saved_state.get("completed_round", 0)))
                        except (OSError, json.JSONDecodeError, TypeError, ValueError):
                            self.emit_log(task_id, "⚠ 電影續跑資料無效，將從最後可用劇本重新演化。")
                film_prompt = f"""請為以下需求設計一部可製作成 2D 動態圖像短片的原創電影。
使用者需求：
{accumulated_ctx[:5000]}

輸出嚴格 JSON，不要 Markdown，格式如下：
{{
  "title": "片名",
  "logline": "一句話故事概要",
  "scenes": [
    {{"heading": "場景標題", "visual": "可視化場景描述（英文或中文皆可）", "narration": "該場旁白或對白", "mood": "hopeful", "duration_sec": 4}}
  ]
}}
請提供 4 至 8 個有起承轉合的場景，每場旁白精簡；mood 可用 hopeful、dark、tense、sad、romantic、mysterious、epic。"""
                if film_spec is not None:
                    if not film_spec:
                        film_raw = self._generate(
                            role="architect", prompt=film_prompt, task_id=task_id,
                            action_desc="撰寫電影劇本與分鏡", format_json=True, num_ctx=decided_ctx
                        )
                        try:
                            film_spec = json.loads(film_raw)
                            if (
                                not isinstance(film_spec, dict)
                                or not isinstance(film_spec.get("scenes"), list)
                                or not any(isinstance(scene, dict) for scene in film_spec["scenes"])
                            ):
                                raise ValueError("The screenplay did not contain any valid scenes.")
                        except (json.JSONDecodeError, ValueError) as exc:
                            self.emit_log(task_id, f"⚠ 電影劇本 JSON 格式無效，使用內建分鏡完成可播放草稿：{exc}")
                            film_spec = self.film_forge.fallback_screenplay(title, desc)

                        film_spec["title"] = str(film_spec.get("title") or title)
                        film_spec["scenes"] = self.film_forge._normalize_scenes(film_spec["scenes"])
                        self._save_task_state(film_work_state, {
                            "completed_round": 0,
                            "spec": film_spec,
                        })
                    film_models = self.pool.plan_greedy_swarm(num_ctx=decided_ctx)
                    for evo in range(completed_round + 1, iterations + 1):
                        self.checkpoint(
                            task_id,
                            f"電影劇本與分鏡演化 ({evo}/{iterations})",
                            75 + int(evo / iterations * 3),
                        )
                        film_spec["scenes"] = self._refine_film_scenes(
                            task_id, desc, film_spec, film_models, evo, iterations, decided_ctx
                        )
                        completed_round = evo
                        self._save_task_state(film_work_state, {
                            "completed_round": completed_round,
                            "spec": film_spec,
                        })

                    film_dir.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix=f"{task_id}_film_", dir=str(TASK_ZONE)) as staging:
                        staged_dir = Path(staging) / "film"
                        film_result = self.film_forge.render_film(
                            staged_dir, film_spec, iterations=iterations
                        )
                        self._publish_staged_directory(staged_dir, film_dir)
                    self.emit_log(
                        task_id,
                        (
                            f"✔ 電影短片已輸出（{film_result['duration_sec']} 秒，{iterations} 輪演化）："
                            "film.mp4、劇本、分鏡與原創配樂"
                            + ("及電腦語音旁白。" if film_result.get("narration") else
                               "。目前系統沒有可用的 Windows 語音，影片保留字幕與配樂。")
                        ),
                        event_type="film_preview",
                        extra={
                            "preview_url": f"/delivery/films/{task_id}/index.html?t={int(time.time())}",
                            "video_url": f"/delivery/films/{task_id}/film.mp4"
                        }
                    )

            # 6. 多模組軟體專案開發（根據記憶體預算動態拆分並行模組）
            project_delivery_dir = DELIVERY_ZONE / "projects" / task_id
            if (
                plan_flags.get("needs_software_project")
                and project_delivery_dir.exists()
                and any(project_delivery_dir.iterdir())
            ):
                existing_tests = project_delivery_dir / "test_suite.py"
                if not existing_tests.is_file():
                    raise FileExistsError(
                        f"An incomplete project delivery already exists and will not be overwritten: {project_delivery_dir}"
                    )
                existing_test_result = self._run_python_script(existing_tests)
                if not existing_test_result["success"]:
                    raise RuntimeError(
                        f"Existing project delivery failed its tests and was preserved: "
                        f"{existing_test_result['stderr']}"
                    )
                self.emit_log(task_id, "📦 偵測到既有且通過測試的軟體專案交付，保留原檔並略過重建。")
                plan_flags["needs_software_project"] = False
            if plan_flags.get("needs_software_project"):
                if self._should_check_runtime_dag(task):
                    runtime_plan_raw = self._generate(
                        role="architect",
                        prompt=f"""請根據目前已完成的分析及使用者需求，重新檢查接下來的軟體交付是否有尚未處理、可獨立並行的工作。
只輸出需要在主專案之前或期間完成的新增子任務；已由既有任務負責或已完成的範圍不可重複建立。依賴使用既有子任務 key 或本次輸出的 key。
目前子任務：{json.dumps(task.get('subtask_ids', []), ensure_ascii=False)}
目前分析：\n{accumulated_ctx[:5000]}
輸出 JSON：{{"subtasks":[{{"key":"integration","title":"API integration","description":"Concrete deliverable","depends_on":[],"priority":0}}]}}
沒有新增工作則輸出空陣列。""",
                        task_id=task_id,
                        action_desc="執行期重新評估 DAG 子任務",
                        format_json=True,
                        num_ctx=decided_ctx,
                    )
                    try:
                        runtime_plan = json.loads(runtime_plan_raw)
                        proposed_runtime = (
                            runtime_plan.get("subtasks", [])
                            if isinstance(runtime_plan, dict) else []
                        )
                        if not isinstance(proposed_runtime, list):
                            proposed_runtime = []
                    except (json.JSONDecodeError, TypeError):
                        proposed_runtime = []
                        self.emit_log(task_id, "⚠ 執行期 DAG 規劃回應無效，沿用目前工作分解。")
                    task = self._set_task_fields(task_id, runtime_dag_checked=True)
                    runtime_ids = self._spawn_subtasks(task, proposed_runtime)
                    if runtime_ids:
                        task = self._set_task_fields(task_id, status="waiting_dependencies")
                        self.emit_log(
                            task_id,
                            f"🧩 執行期依分析結果新增 DAG 子任務：{', '.join(runtime_ids)}。",
                        )
                        return
                fb = self.checkpoint(task_id, "Stage 6: 滿載多模型並行專案開發與 TDD 驗收", 78)
                accumulated_ctx += fb
                proj_dir = TASK_ZONE / task_id / "project"
                proj_dir.mkdir(parents=True, exist_ok=True)
                inherited_subtask_files = self._collect_subtask_deliverables(task, proj_dir)
                project_state_path = TASK_ZONE / task_id / "project_progress.json"
                project_state = {}
                if project_state_path.is_file():
                    try:
                        loaded_state = json.loads(project_state_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_state, dict):
                            project_state = loaded_state
                    except (OSError, json.JSONDecodeError):
                        self.emit_log(task_id, "⚠ 軟體專案 checkpoint 無法讀取，將保留原始碼並重新建立規格。")

                swarm_models = self.pool.plan_greedy_swarm(num_ctx=decided_ctx)
                num_parallel = len(swarm_models)

                spec_prompt = f"""{accumulated_ctx}
已完成子任務交付（必須檢視並整合，勿重複從零實作）：\n{json.dumps(inherited_subtask_files, ensure_ascii=False)}
這些檔案已位於專案的 `subtasks/` 目錄；請在架構規格與 additional_files 中明確引用、包裝或整合既有模組。
請將使用者需求拆解為恰好 {num_parallel} 個互不重複、可獨立並行開發的 Python 核心模組檔案，以發揮預算內所有不同模型的算力。
依責任邊界切分功能、資料模型、輸入驗證、錯誤處理及整合，不要只按檔案數重複拆分；各模組需清楚描述介面與相依性。
若需求涉及全端或部署，額外在 `additional_files` 產生可交付的專案支援檔，例如 TypeScript/React/Vue、HTML/CSS/JavaScript、SQL migration、package.json 與 Dockerfile；不需要的項目輸出空陣列。每項使用相對路徑及完整文字內容。
                先定義跨檔案共享介面：在 `interface_stubs` 中提供可通過 Python AST 的 `interfaces.py` 原始碼，包含共用公開 Protocol／資料類別、函式簽名、參數與回傳型別；各模組 instruction 必須使用一致的名稱、簽名與 import，且不得自行改寫契約。
                輸出嚴格 JSON：
                {{
                  "interface_stubs": "可執行的 interfaces.py 原始碼字串",
                  "module_contracts": [
                    {{"filename":"module_1.py","public_symbols":"公開類別、函式名稱、簽名、import 路徑與呼叫範例"}},
                    {{"filename":"module_2.py","public_symbols":"公開類別、函式名稱、簽名、import 路徑與呼叫範例"}}
                  ],
                  "additional_files": [
                    {{"path":"frontend/src/App.tsx","content":"完整原始碼"}},
                    {{"path":"migrations/001_init.sql","content":"CREATE TABLE ...;"}},
                    {{"path":"Dockerfile","content":"FROM ..."}}
                  ],
                  "files": [
                    {{"filename": "module_1.py", "instruction": "詳細類別與函式邏輯及需遵守的公開簽名"}},
                    {{"filename": "module_2.py", "instruction": "詳細類別與函式邏輯及需遵守的公開簽名"}}
                  ]
                }}
                files 必須恰好有 {num_parallel} 個，module_contracts 必須為每個模組定義跨檔案公開符號契約。"""
                checkpoint_spec = project_state.get("spec")
                if (
                    project_state.get("phase") in {
                        "spec_ready", "modules_ready", "tests_ready", "tdd_complete",
                        "quality_round", "project_complete",
                    }
                    and isinstance(checkpoint_spec, dict)
                ):
                    proj_spec = checkpoint_spec
                    self.emit_log(task_id, "📦 正在從軟體專案 checkpoint 還原模組規格與已產生程式碼。")
                else:
                    spec_raw = self._generate(
                        role="architect", prompt=spec_prompt, task_id=task_id,
                        action_desc=f"拆解 {num_parallel} 個並行模組契約",
                        format_json=True, num_ctx=decided_ctx,
                    )
                    try:
                        proj_spec = json.loads(spec_raw)
                    except (json.JSONDecodeError, TypeError):
                        proj_spec = {}
                file_list = proj_spec.get("files", []) if isinstance(proj_spec, dict) else []
                interface_code = (
                    proj_spec.get("interface_stubs")
                    if isinstance(proj_spec, dict) else None
                )
                if not isinstance(interface_code, str) or not self.sandbox.is_valid_python(interface_code):
                    raise ValueError(
                        "The architecture model did not provide valid Python interface stubs."
                    )
                safe_file_list = []
                seen_filenames = set()
                for item in file_list if isinstance(file_list, list) else []:
                    if not isinstance(item, dict):
                        continue
                    filename = Path(str(item.get("filename", ""))).name
                    if (
                        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py", filename)
                        or filename in {"test_suite.py", "interfaces.py"}
                        or filename in seen_filenames
                    ):
                        continue
                    seen_filenames.add(filename)
                    safe_file_list.append({
                        "filename": filename,
                        "instruction": str(item.get("instruction", ""))[:4000]
                    })
                    if len(safe_file_list) >= num_parallel:
                        break
                if not safe_file_list:
                    safe_file_list = [{
                        "filename": "core_engine.py",
                        "instruction": desc[:4000],
                    }]
                actual_parallel = min(num_parallel, len(safe_file_list))
                file_list = self._normalize_parallel_work_items(
                    safe_file_list,
                    actual_parallel,
                    [],
                    filenames=True,
                )
                additional_files = self._normalize_project_artifacts(
                    proj_spec.get("additional_files", [])
                    if isinstance(proj_spec, dict) else []
                )
                proj_spec["additional_files"] = additional_files
                artifact_paths = []
                for artifact in additional_files:
                    artifact_path = proj_dir / artifact["path"]
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    artifact_path.write_text(artifact["content"], encoding="utf-8")
                    artifact_paths.append(artifact["path"])
                artifact_validation = self.sandbox.validate_project_files(
                    proj_dir, artifact_paths
                )
                if not artifact_validation["success"]:
                    raise ValueError(
                        f"Project support-file validation failed: {artifact_validation['stderr']}"
                    )
                module_contracts = (
                    proj_spec.get("module_contracts", [])
                    if isinstance(proj_spec, dict) else []
                )
                contract_text = json.dumps(module_contracts, ensure_ascii=False)
                for item in file_list:
                    if "跨模組公開符號契約" not in item["instruction"]:
                        item["instruction"] = (
                            f'{item["instruction"]}\n跨模組公開符號契約（不得自行更名或猜測）：'
                            f"{contract_text}"
                        )
                if len(safe_file_list) < num_parallel:
                    self.emit_log(
                        task_id,
                        f"ℹ 架構師規劃 {len(safe_file_list)} 個模組，將只派發這些已定義的模組，不建立重複補丁工作。"
                    )

                proj_spec["files"] = file_list
                interface_path = proj_dir / "interfaces.py"
                required_files = [interface_path, *(proj_dir / item["filename"] for item in file_list)]
                checkpoint_phase = project_state.get("phase")
                resuming_project = (
                    checkpoint_phase in {
                        "spec_ready", "modules_ready", "tests_ready", "tdd_complete",
                        "quality_round", "project_complete",
                    }
                    and all(path.is_file() for path in required_files)
                    and self.sandbox.is_valid_python(interface_path.read_text(encoding="utf-8"))
                    and all(
                        self.sandbox.is_valid_python(path.read_text(encoding="utf-8"))
                        for path in required_files[1:]
                    )
                )
                generated_files = {"interfaces.py": interface_code}
                if resuming_project:
                    interface_code = interface_path.read_text(encoding="utf-8")
                    generated_files.update({
                        item["filename"]: (proj_dir / item["filename"]).read_text(encoding="utf-8")
                        for item in file_list
                    })
                else:
                    self.emit_log(task_id, f"⚡ [滿載多模型並行] 同時調動 {len(swarm_models)} 個模型槽位並行實作 {len(file_list)} 個 Python 模組...")
                    interface_path.write_text(interface_code, encoding="utf-8")
                    project_state = {
                        **project_state,
                        "phase": "spec_ready",
                        "spec": proj_spec,
                        "completed_quality_round": 1,
                    }
                    self._save_task_state(project_state_path, project_state)
                    for item in file_list:
                        existing_module = proj_dir / item["filename"]
                        if existing_module.is_file():
                            existing_code = existing_module.read_text(encoding="utf-8")
                            if self.sandbox.is_valid_python(existing_code):
                                generated_files[item["filename"]] = existing_code
                generated_files.update(
                    self._load_subtask_python_sources(proj_dir, inherited_subtask_files)
                )

                def _parallel_py_coder(idx: int, f_item: dict, model_name: str):
                    self.checkpoint(task_id)
                    fn = f_item["filename"]
                    p = f"""請實作 Python 模組 `{fn}`：
規格要求：{f_item['instruction']}
共享介面存根 `interfaces.py`（不可變更名稱、型別或函式簽名）：
{interface_code}
相依模組可依存根指定的公開契約 import；不得猜測其他模組未聲明的符號。
{accumulated_ctx[:2500]}
只輸出純 Python 代碼，不含 Markdown。"""
                    code = self._generate(
                        role="coder", prompt=p, task_id=task_id,
                        action_desc=f"並行實作 {fn}", num_ctx=decided_ctx, preferred_model=model_name,
                        output_validator=self._is_valid_python_output,
                    )
                    clean_c = code.replace("```python", "").replace("```", "").strip()
                    (proj_dir / fn).write_text(clean_c, encoding="utf-8")
                    return fn, clean_c

                if not resuming_project:
                    pending_items = [
                        (index, item) for index, item in enumerate(file_list)
                        if item["filename"] not in generated_files
                    ]
                    with ThreadPoolExecutor(max_workers=max(1, len(pending_items))) as ex:
                        futs = [
                            ex.submit(_parallel_py_coder, i, item, swarm_models[i]["name"])
                            for i, item in pending_items
                        ]
                        for fut in as_completed(futs):
                            fn, clean_c = fut.result()
                            generated_files[fn] = clean_c
                            self.emit_log(task_id, f"✔ 並行模組產出完成：{fn}")
                            project_state = {
                                **project_state,
                                "phase": "spec_ready",
                                "spec": proj_spec,
                                "completed_modules": sorted(
                                    name for name in generated_files if name != "interfaces.py"
                                ),
                                "completed_quality_round": 1,
                            }
                            self._save_task_state(project_state_path, project_state)
                    project_state = {
                        **project_state,
                        "phase": "modules_ready",
                        "spec": proj_spec,
                        "completed_quality_round": 1,
                    }
                    self._save_task_state(project_state_path, project_state)

                # TDD 驗收
                self.checkpoint(task_id, "Stage 6.2: 沙盒 TDD 驗證與三級主管升級除錯", 88)
                test_file = proj_dir / "test_suite.py"
                if (
                    resuming_project
                    and checkpoint_phase in {
                        "tests_ready", "tdd_complete", "quality_round", "project_complete",
                    }
                    and test_file.is_file()
                    and self.sandbox.is_valid_python(test_file.read_text(encoding="utf-8"))
                ):
                    test_code = test_file.read_text(encoding="utf-8")
                else:
                    requirement_contracts = json.dumps(file_list, ensure_ascii=False)
                    t_prompt = f"""請依照原始使用者需求為以下專案撰寫 `test_suite.py`，測試必須驗證具體需求，不可只根據程式碼猜測測試內容。
原始使用者需求：{desc}
模組規格與跨模組公開契約：{requirement_contracts}
專案檔案：\n{json.dumps(generated_files, ensure_ascii=False)}
要求：第 1~3 行固定為 `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))`，用 assert 覆蓋原始需求、每個模組責任、模組間整合及重要邊界情況。只輸出純 Python 代碼。"""
                    t_raw = self._generate(
                        role="coder", prompt=t_prompt, task_id=task_id,
                        action_desc="建立 TDD 測試套件", num_ctx=decided_ctx,
                        output_validator=self._is_valid_python_output,
                    )
                    test_code = t_raw.replace("```python", "").replace("```", "").strip()
                    test_file.write_text(test_code, encoding="utf-8")
                    project_state = {
                        **project_state,
                        "phase": "tests_ready",
                        "spec": proj_spec,
                        "completed_quality_round": 1,
                    }
                    self._save_task_state(project_state_path, project_state)

                tdd_failures = list(project_state.get("tdd_failures", []))
                completed_tdd_tier = int(project_state.get("completed_tdd_tier", 0) or 0)
                tdd_passed = project_state.get("phase") in {
                    "tdd_complete", "quality_round", "project_complete",
                }
                for tier in range(completed_tdd_tier + 1, 4) if not tdd_passed else []:
                    self.checkpoint(task_id)
                    run_res = self._run_python_script(test_file)
                    if run_res["success"]:
                        self.emit_log(task_id, "✔ 專案代碼通過沙盒 TDD 驗證！")
                        tdd_passed = True
                        break
                    failure_text = run_res["stderr"] or run_res["stdout"] or "Unknown TDD failure."
                    tdd_failures.append(failure_text[:1200])
                    escalation_models = self.pool.rank_models_for_role("arbiter", num_ctx=decided_ctx)
                    if not escalation_models:
                        raise RuntimeError("No compatible arbiter model is available for TDD repair.")
                    fix_m = self._select_escalation_model(escalation_models, tier)["name"]
                    self.emit_log(task_id, f"▲ 測試未過，啟動 Tier {tier} 修復 ({fix_m}): {failure_text[:140]}")
                    allowed_files = set(generated_files) | {"test_suite.py"}
                    fix_p = f"""TDD 測試失敗。請依原始需求及模組契約提出最小必要的局部修補，絕對不可重寫整個檔案。
原始使用者需求：{desc[:3000]}
模組規格與跨模組公開契約：{json.dumps(file_list, ensure_ascii=False)}
當前模組：{json.dumps(generated_files, ensure_ascii=False)}
測試腳本：{test_code}
錯誤：{failure_text}
每個修補必須使用精確且唯一的舊文字，保留其餘所有程式碼：
若目標文字在檔案中重複，請在 <<<FIND>>> 區塊一併包含足以唯一識別位置的相鄰上下文行，勿只提供重複的單行。
<<<PATCH: 檔名.py>>>
<<<FIND>>>要精確匹配的原始程式碼<<<REPLACE>>>替換後程式碼<<<END_PATCH>>>
只允許修補以上專案檔或 test_suite.py，禁止輸出整檔程式碼。"""
                    fixed_raw = self._generate(
                        role="arbiter", prompt=fix_p, task_id=task_id,
                        action_desc=f"Tier {tier} 升級修補",
                        num_ctx=decided_ctx, preferred_model=fix_m,
                        output_validator=lambda output: self._are_valid_python_patches(
                            proj_dir, allowed_files, output
                        ),
                    )
                    for target_fn, patch_body in re.findall(
                        r"<<<PATCH:\s*([^>]+)>>>(.*?)(?=<<<PATCH:|$)",
                        fixed_raw,
                        re.DOTALL,
                    ):
                        target_fn = target_fn.strip()
                        if (
                            Path(target_fn).name != target_fn
                            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py", target_fn)
                            or target_fn not in allowed_files
                        ):
                            continue
                        target_path = proj_dir / target_fn
                        current_code = target_path.read_text(encoding="utf-8")
                        c_clean = self._apply_exact_patches(current_code, patch_body)
                        if c_clean is None:
                            self.emit_log(
                                task_id,
                                f"⚠ 已拒絕 {target_fn} 的補丁：舊片段不存在或不唯一；"
                                "請提供相鄰上下文行，避免修改錯誤位置。",
                            )
                            continue
                        if self.sandbox.is_valid_python(c_clean):
                            target_path.write_text(c_clean, encoding="utf-8")
                            if target_fn == "test_suite.py":
                                test_code = c_clean
                            else:
                                generated_files[target_fn] = c_clean
                    project_state = {
                        **project_state,
                        "phase": "tests_ready",
                        "spec": proj_spec,
                        "completed_tdd_tier": tier,
                        "tdd_failures": tdd_failures,
                        "completed_quality_round": 1,
                    }
                    self._save_task_state(project_state_path, project_state)

                if not tdd_passed:
                    final_tdd = self._run_python_script(test_file)
                    if final_tdd["success"]:
                        tdd_passed = True
                    else:
                        final_error = final_tdd.get("stderr") or final_tdd.get("stdout", "")
                        if tdd_failures:
                            try:
                                self.record_lesson(
                                    task_id,
                                    "Generated Python tests still fail after three escalating repairs. "
                                    "Check the test's actual requirements and verify each module follows "
                                    f"the shared symbol contract. Failure: {final_error[:700]}",
                                )
                            except (OSError, ValueError, TypeError) as lesson_error:
                                self.emit_log(task_id, f"⚠ 無法保存 TDD 避坑經驗：{lesson_error}")
                        raise RuntimeError(
                            f"Project did not pass TDD after three repair tiers: {final_error}"
                        )

                last_quality_feedback = ""
                if tdd_passed:
                    project_state = {
                        **project_state,
                        "phase": "tdd_complete",
                        "spec": proj_spec,
                        "completed_tdd_tier": 3,
                        "tdd_failures": tdd_failures,
                        "completed_quality_round": max(
                            1, int(project_state.get("completed_quality_round", 1) or 1)
                        ),
                    }
                    self._save_task_state(project_state_path, project_state)
                last_quality_feedback = str(project_state.get("quality_feedback", ""))
                completed_quality_round = max(
                    1, int(project_state.get("completed_quality_round", 1) or 1)
                )
                for evo in range(completed_quality_round + 1, iterations + 1):
                    self.checkpoint(
                        task_id,
                        f"軟體品質演化 ({evo}/{iterations})",
                        88 + int((evo - 1) / max(1, iterations - 1) * 5),
                    )
                    baseline_files = dict(generated_files)
                    baseline_code = {
                        filename: path.read_text(encoding="utf-8")
                        for filename, path in (
                            (name, proj_dir / name) for name in generated_files
                        )
                        if path.is_file()
                    }

                    def _refine_project_module(item: dict, model_name: str):
                        filename = item["filename"]
                        current_code = generated_files.get(filename, "")
                        repo_map = self._build_repo_map(generated_files)
                        prompt = f"""請改進 Python 模組 `{filename}`，實際修補程式以提升品質，不可只提供建議。
使用者需求：{desc[:2500]}
模組契約：{item["instruction"]}
目前完整專案公開符號 Repo Map（以此同步其他模組的最新介面）：\n{json.dumps(repo_map, ensure_ascii=False)}
目前完整模組：\n{current_code[:12000]}
不可重寫整個檔案，僅輸出精確唯一的局部補丁；若舊片段在檔案中重複，請在 <<<FIND>>> 加入足夠的相鄰上下文行，讓目標唯一；不得猜測要修改哪一處。無需修改時輸出空字串。
自動化測試：\n{test_code[:6000]}
本輪：{evo}/{iterations}。保留所有既有公開介面與行為，針對健壯性、錯誤處理、邊界情況、效率與可維護性做具體改善。
上一輪驗證結果：{last_quality_feedback[:1500]}
格式：
<<<PATCH: {filename}>>>
<<<FIND>>>精確舊文字<<<REPLACE>>>新文字<<<END_PATCH>>>
只輸出補丁標記，不要 Markdown，不要輸出完整檔案。"""
                        refined = self._generate(
                            role="coder", prompt=prompt, task_id=task_id,
                            action_desc=f"改善 {filename} ({evo}/{iterations})",
                            num_ctx=decided_ctx, preferred_model=model_name,
                            output_validator=lambda output: self._is_valid_python_patch(
                                filename, current_code, output
                            ),
                        )
                        return filename, refined

                    candidates = {}
                    with ThreadPoolExecutor(max_workers=len(file_list)) as executor:
                        futures = [
                            executor.submit(
                                _refine_project_module, item, swarm_models[index]["name"]
                            )
                            for index, item in enumerate(file_list)
                        ]
                        for future in as_completed(futures):
                            filename, code = future.result()
                            candidates[filename] = code

                    patched_candidates = dict(generated_files)
                    patch_errors = []
                    for filename, patch_response in candidates.items():
                        patched = self._apply_exact_patches(
                            generated_files.get(filename, ""), patch_response
                        )
                        if patched is not None:
                            patched_candidates[filename] = patched
                        elif patch_response.strip():
                            patch_errors.append(filename)
                    valid_candidates = (
                        len(candidates) == len(file_list)
                        and not patch_errors
                        and all(
                            self.sandbox.is_valid_python(patched_candidates.get(item["filename"], ""))
                            for item in file_list
                        )
                    )
                    if valid_candidates:
                        for filename, code in patched_candidates.items():
                            (proj_dir / filename).write_text(code, encoding="utf-8")
                        generated_files = patched_candidates
                        quality_test = self._run_python_script(test_file)
                    else:
                        quality_test = {
                            "success": False,
                            "stderr": (
                                "Ambiguous or unmatched patches for: "
                                + ", ".join(patch_errors)
                                if patch_errors else "Model output contained invalid Python."
                            ),
                        }

                    if not valid_candidates or not quality_test["success"]:
                        generated_files = baseline_files
                        for filename, code in baseline_code.items():
                            (proj_dir / filename).write_text(code, encoding="utf-8")
                        last_quality_feedback = quality_test.get("stderr", "Invalid Python output.")
                        self.emit_log(
                            task_id,
                            f"⚠ 第 {evo} 輪改善未通過測試，已回復上一個通過驗收的版本。"
                        )
                    else:
                        last_quality_feedback = quality_test.get("stdout", "All tests passed.")
                        self.emit_log(task_id, f"✔ 軟體第 {evo}/{iterations} 輪品質改善通過 TDD。")
                    project_state = {
                        **project_state,
                        "phase": "quality_round",
                        "spec": proj_spec,
                        "completed_quality_round": evo,
                        "quality_feedback": last_quality_feedback,
                        "tdd_failures": tdd_failures,
                    }
                    self._save_task_state(project_state_path, project_state)

                self.checkpoint(task_id, "Stage 6.3: 最終回歸測試與交付封存", 94)
                final_test = self._run_python_script(test_file)
                if not final_test["success"]:
                    if tdd_failures:
                        try:
                            self.record_lesson(
                                task_id,
                                "TDD repair did not stabilize the generated project. "
                                f"Review the failing assertion and shared module contracts first: "
                                f"{(final_test.get('stderr') or '')[:500]}",
                            )
                        except (OSError, ValueError, TypeError) as lesson_error:
                            self.emit_log(task_id, f"⚠ 無法保存 TDD 避坑經驗：{lesson_error}")
                    raise RuntimeError(f"Final project regression test failed: {final_test['stderr']}")
                if tdd_failures:
                    failure_summary = " | ".join(tdd_failures[-3:])
                    try:
                        self.record_lesson(
                            task_id,
                            "When generated Python project tests fail, inspect the failing assertion "
                            "and verify module imports match the architect's shared public-symbol "
                            f"contracts before changing implementation. Observed: {failure_summary[:900]}",
                        )
                    except (OSError, ValueError, TypeError) as lesson_error:
                        self.emit_log(task_id, f"⚠ 無法保存 TDD 修復經驗：{lesson_error}")
                project_state = {
                    **project_state,
                    "phase": "project_complete",
                    "spec": proj_spec,
                    "completed_quality_round": max(
                        1, int(project_state.get("completed_quality_round", 1) or 1)
                    ),
                    "tdd_failures": tdd_failures,
                }
                self._save_task_state(project_state_path, project_state)
                project_delivery_dir.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"{task_id}_project_", dir=str(project_delivery_dir.parent)
                ) as staging:
                    staged_project = Path(staging) / "project"
                    shutil.copytree(proj_dir, staged_project)
                    self._publish_staged_directory(staged_project, project_delivery_dir)

            # 7. 並行資安審查 + 數值平衡審查 + 綜合交付報告（3 模型同時並行）
            fb = self.checkpoint(task_id, "Stage 7: 多模型並行稽核與最終報告交付", 95)
            accumulated_ctx += fb
            swarm_end = self.pool.plan_greedy_swarm(num_ctx=decided_ctx)
            self.emit_log(task_id, "⚡ [多模型並行收尾] 同時執行『資安稽核』、『架構審查』與『完整交付報告』...")

            review_specs = [
                ("fast", f"請對本次產出進行資安與效能審查：\n{accumulated_ctx[:3000]}", "資安合規審查"),
                ("fast", f"請檢驗遊戲與系統規則完整性：\n{accumulated_ctx[:3000]}", "玩法與規則驗收"),
                ("architect", f"請撰寫完整專案交付與架構總結報告（Markdown）：\n{accumulated_ctx[:6000]}", "撰寫綜合交付報告"),
            ]
            additional_perspectives = [
                "需求覆蓋與遺漏",
                "邊界情況與穩定性",
                "可用性與無障礙",
                "模組介面與整合風險",
                "效能與資源使用",
            ]
            while len(review_specs) < len(swarm_end):
                perspective = additional_perspectives[
                    (len(review_specs) - 3) % len(additional_perspectives)
                ]
                review_specs.append((
                    "fast",
                    f"請獨立檢查本次需求與交付內容，著重【{perspective}】，指出具體缺漏及可驗證的改善建議：\n{accumulated_ctx[:4000]}",
                    f"{perspective}審查",
                ))

            review_results = {}
            with ThreadPoolExecutor(max_workers=min(len(review_specs), len(swarm_end))) as ex:
                futures = {
                    ex.submit(
                        self._generate,
                        role, prompt, "", task_id, action, None,
                        decided_ctx,
                        False, 0.2,
                        swarm_end[index % len(swarm_end)]["name"]
                    ): index
                    for index, (role, prompt, action) in enumerate(review_specs)
                }
                for future in as_completed(futures):
                    review_results[futures[future]] = future.result()

            sec_report = review_results[0]
            qa_report = review_results[1]
            final_report = review_results[2]
            additional_reports = "\n\n".join(
                f"### {review_specs[index][2]}\n{review_results[index]}"
                for index in range(3, len(review_specs))
            )
            review_context = (
                f"需求覆蓋與品質審查：\n{qa_report}\n\n"
                f"資安與效能審查：\n{sec_report}\n\n{additional_reports}"
            )
            for evo in range(2, iterations + 1):
                self.checkpoint(
                    task_id,
                    f"最終答案品質演化 ({evo}/{iterations})",
                    95 + int((evo - 1) / max(1, iterations - 1) * 4),
                )
                refined_report = self._generate(
                    role="architect",
                    prompt=f"""請依使用者原始需求及審查意見，直接修訂以下最終交付報告，使本輪內容更準確、完整、可執行。
本輪：{evo}/{iterations}；必須實際補足內容，不可只換句話說。
使用者需求：{accumulated_ctx[:5000]}
審查意見：\n{review_context[:7000]}
目前報告：\n{final_report[:10000]}
保留已驗證事實，標明限制與未驗證假設；以使用者使用的語言撰寫，不得捏造執行結果。""",
                    task_id=task_id,
                    action_desc=f"最終交付報告品質改善 ({evo}/{iterations})",
                    num_ctx=decided_ctx,
                )
                if refined_report.strip():
                    final_report = refined_report.strip()
                else:
                    self.emit_log(task_id, f"⚠ 報告第 {evo} 輪未產生有效內容，保留上一輪版本。")

            report_file = self._task_delivery_dir("reports", task_id) / "final_delivery.md"
            report_content = (
                f"{final_report}\n\n---\n### QA Verification\n{qa_report}"
                f"\n\n---\n### Security Audit\n{sec_report}"
            )
            if additional_reports:
                report_content += f"\n\n---\n{additional_reports}"
            report_content += f"\n\n<!-- TASK-COMPLETE: {task_id} -->\n"
            if not self._write_delivery_text(report_file, report_content):
                self.emit_log(task_id, "📦 偵測到既有最終報告，保留原檔，不覆寫交付。")

            self.update_task_status(task_id, "completed")
            self.current_task_progress_pct = 100
            self.current_task_stage = "Completed"
            self.emit_log(task_id, f"🎉 任務 [{task_id}] 全部完成！所有檔案已歸檔至 03_delivery。", event_type="completed")

        except TaskTerminatedException:
            self.update_task_status(task_id, "pending")
            self.current_task_stage = "Terminated (Pending)"
            self.emit_log(task_id, "⏹ 任務已由使用者強制終止，狀態已保留為 pending（隨時可按『開始』續跑）。")
        except Exception as e:
            if self.get_control_state().get("stop_requested"):
                self.update_task_status(task_id, "pending")
                self.current_task_stage = "Terminated (Pending)"
                self.emit_log(task_id, f"⏹ 停止期間中斷推理，任務已保留為 pending：{e}")
            elif self.get_control_state().get("pause_requested"):
                self.update_task_status(task_id, "paused")
                self.current_task_stage = "Paused (Checkpoint Preserved)"
                self.emit_log(task_id, f"⏸ 暫停期間已中斷推理，任務保留供續跑：{e}")
            else:
                self.update_task_status(task_id, "failed")
                self.current_task_stage = "Failed"
                try:
                    audit_path = self._archive_failure_audit(
                        task, e, traceback.format_exc()
                    )
                    self.emit_log(task_id, f"已保存失敗現場與程式碼快照：{audit_path}")
                except OSError as audit_error:
                    self.emit_log(task_id, f"⚠ 失敗現場歸檔失敗：{audit_error}")
                self.emit_log(task_id, f"❌ 任務異常中斷: {e}")
        finally:
            with self._task_runtime_lock:
                self._task_runtime.pop(task_id, None)
            with self._interrupt_lock:
                self._interrupt_events.pop(task_id, None)
            self.task_start_ts = None
            self.current_task_id = "-"

    def _refine_film_scenes(
        self, task_id: str, description: str, film_spec: Dict,
        models: List[Dict], evolution_round: int, max_rounds: int, num_ctx: int,
    ) -> List[Dict]:
        scenes = self.film_forge._normalize_scenes(film_spec.get("scenes", []))
        if not scenes:
            raise ValueError("A film needs at least one scene before refinement.")
        if not models:
            raise RuntimeError("No compatible Ollama models are available for film refinement.")
        worker_count = min(len(models), len(scenes))
        batches = self._partition_parallel_indices(len(scenes), worker_count)
        outline = [
            {"index": index + 1, "heading": scene["heading"]}
            for index, scene in enumerate(scenes)
        ]

        def refine_batch(worker_index: int, scene_indices: List[int]):
            assigned = [
                {"index": index + 1, **scenes[index]}
                for index in scene_indices
            ]
            prompt = f"""請在原創電影需求與完整分場脈絡下，改善指定場景的敘事、鏡頭、可視化動作及情緒連貫性。
這是第 {evolution_round}/{max_rounds} 輪；不可只改寫同義詞，要增加可拍攝的細節並保留故事方向。
片名：{film_spec["title"]}
概要：{film_spec.get("logline", "")}
需求：{description[:2500]}
完整分場：{json.dumps(outline, ensure_ascii=False)}
指定場景：{json.dumps(assigned, ensure_ascii=False)}
只輸出嚴格 JSON：{{"scenes":[{{"index":1,"heading":"...","visual":"可視化鏡頭描述","narration":"旁白或對白","mood":"hopeful","duration_sec":4}}]}}
必須恰好依序回傳這批場景的 index，保留故事連貫性及使用者要求。"""
            raw = self._generate(
                role="fast", prompt=prompt, task_id=task_id,
                action_desc=f"電影分鏡品質改善第 {evolution_round}/{max_rounds} 輪",
                format_json=True, num_ctx=num_ctx,
                preferred_model=models[worker_index]["name"]
            )
            result = json.loads(raw)
            refined = result.get("scenes", []) if isinstance(result, dict) else []
            expected = [index + 1 for index in scene_indices]
            returned = [
                item.get("index") for item in refined if isinstance(item, dict)
            ] if isinstance(refined, list) else []
            if returned != expected:
                raise ValueError("The scene refinement changed or omitted assigned scene indices.")
            normalized = self.film_forge._normalize_scenes(refined)
            if len(normalized) != len(scene_indices):
                raise ValueError("The scene refinement returned an invalid scene count.")
            return list(zip(scene_indices, normalized))

        refined_scenes = list(scenes)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(refine_batch, worker_index, indices)
                for worker_index, indices in enumerate(batches)
            ]
            for future in as_completed(futures):
                try:
                    for scene_index, refined_scene in future.result():
                        refined_scenes[scene_index] = refined_scene
                except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    self.emit_log(
                        task_id,
                        f"⚠ 電影第 {evolution_round} 輪場景改善失敗，保留上一輪內容：{exc}"
                    )
        return refined_scenes

    @staticmethod
    def _partition_parallel_indices(item_count: int, worker_count: int):
        if item_count < 0 or worker_count < 1:
            raise ValueError("Item count must be nonnegative and worker count must be positive.")
        active_workers = min(item_count, worker_count)
        return [
            list(range(worker_index, item_count, active_workers))
            for worker_index in range(active_workers)
        ]

    @staticmethod
    def _normalize_parallel_work_items(items, count: int, defaults, filenames: bool = False):
        if count < 1:
            raise ValueError("Parallel work-item count must be positive.")

        normalized = []
        seen = set()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            if filenames:
                name = Path(str(item.get("filename", ""))).name
                if (
                    not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.py", name)
                    or name == "test_suite.py"
                ):
                    continue
            else:
                name = str(item.get("title", "")).strip()[:100]
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]*", name):
                    continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append((name, str(item.get("instruction", "")).strip()[:4000]))
            if len(normalized) == count:
                break

        if filenames:
            return [
                {"filename": name, "instruction": instruction}
                for name, instruction in normalized
            ]

        fallback_items = [
            item for item in defaults
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
        extra_concerns = [
            "input validation and malformed data",
            "error handling and recovery",
            "edge cases and boundary conditions",
            "accessibility and responsive behavior",
            "integration contracts and compatibility",
            "automated checks and regression prevention",
            "performance and resource usage",
            "remaining user requirements not covered elsewhere",
        ]
        fallback_index = 0
        while len(normalized) < count:
            if fallback_index < len(fallback_items):
                name, instruction = fallback_items[fallback_index]
            else:
                concern_index = fallback_index - len(fallback_items)
                concern = extra_concerns[concern_index % len(extra_concerns)]
                suffix = fallback_index + 1
                name = f"additional_{suffix:02d}.py" if filenames else f"Additional_Requirement_{suffix:02d}"
                instruction = (
                    f"Implement an independent part of the user's requirements focusing on {concern}; "
                    "coordinate through clear interfaces and avoid duplicating other modules."
                )
            name = Path(str(name)).name if filenames else str(name)
            key = name.casefold()
            if key in seen:
                suffix = fallback_index + 1
                name = f"additional_{suffix:02d}.py" if filenames else f"Additional_Requirement_{suffix:02d}"
                key = name.casefold()
            seen.add(key)
            normalized.append((name, str(instruction)[:4000]))
            fallback_index += 1

        if filenames:
            return [{"filename": name, "instruction": instruction} for name, instruction in normalized]
        return normalized

    @staticmethod
    def _normalize_project_artifacts(artifacts) -> List[Dict[str, str]]:
        if artifacts is None:
            return []
        if not isinstance(artifacts, list) or len(artifacts) > 24:
            raise ValueError("Project additional_files must be a list of at most 24 files.")
        allowed_extensions = {
            ".css", ".html", ".js", ".json", ".md", ".mjs", ".cjs", ".sql",
            ".py", ".txt", ".toml", ".ts", ".tsx", ".jsx", ".yaml", ".yml",
        }
        normalized = []
        seen = set()
        total_size = 0
        for item in artifacts:
            if not isinstance(item, dict):
                raise ValueError("Every project support file must be an object.")
            name = item.get("path")
            content = item.get("content")
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError("Project support files require string path and content fields.")
            path = Path(name)
            is_env_example = bool(re.fullmatch(
                r"\.env(?:\.[A-Za-z0-9_-]+)+", path.name, re.IGNORECASE
            ))
            is_known_config = path.name.casefold() in {
                ".dockerignore", ".gitignore", ".npmrc", ".nvmrc",
                ".python-version", "dockerfile", "makefile",
            }
            if (
                not name
                or "\\" in name
                or ":" in name
                or path.is_absolute()
                or path.drive
                or ".." in path.parts
                or (
                    name.startswith(".")
                    and not is_env_example
                    and not is_known_config
                )
                or any(
                    not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                    or part.split(".", 1)[0].upper()
                    in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
                    for part in path.parts
                )
                or not (
                    path.suffix.casefold() in allowed_extensions
                    or is_env_example
                    or is_known_config
                )
            ):
                raise ValueError(f"Unsafe or unsupported project file path: {name!r}.")
            key = path.as_posix().casefold()
            encoded = content.encode("utf-8")
            total_size += len(encoded)
            if (
                key in seen
                or len(encoded) > 200_000
                or total_size > 1_000_000
            ):
                raise ValueError("Project support files must be unique and fit within the 1 MB limit.")
            seen.add(key)
            normalized.append({"path": path.as_posix(), "content": content})
        return normalized

    @staticmethod
    def _is_film_request(title: str, description: str) -> bool:
        text = f"{title} {description}".casefold()
        if any(term in text for term in [
            "電影", "短片", "音樂錄影帶", "預告片", "movie", "film",
            "cinematic", "trailer", "cinema", "motion picture"
        ]):
            return True
        media_terms = ["影片", "動畫", "video", "animation"]
        create_terms = ["製作", "創建", "生成", "做一", "拍攝", "create", "make", "generate", "produce", "render"]
        return any(term in text for term in media_terms) and any(term in text for term in create_terms)

    def _assemble_playable_game_html(self, title: str, assets: list, custom_js_bundle: str) -> str:
        """Assemble a game-neutral host for model-authored gameplay modules."""
        return assemble_game_html(title, custom_js_bundle, assets)

    # ================= 自適應佇列調度迴圈 =================
    def start_queue_loop(self, resume_paused: bool = False):
        with self.queue_condition:
            if self.daemon_active:
                if self.get_control_state().get("stop_requested"):
                    self._queue_restart_requested = True
                self.queue_condition.notify_all()
                return
            if resume_paused:
                with self.backlog_lock:
                    tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                    changed = False
                    for task in tasks:
                        if task.get("status") == "paused":
                            task["status"] = "pending"
                            changed = True
                    if changed:
                        self._save_task_state(BACKLOG_FILE, tasks)
            self.daemon_active = True
            self._queue_restart_requested = False
            if self.daemon_start_ts is None:
                self.daemon_start_ts = time.time()
                self.daemon_paused_accum = 0.0
        self.set_control_state("RUNNING", pause_req=False, stop_req=False)

        def _loop():
            try:
                active = {}
                submitted_ids = set()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    while True:
                        ctrl = self.get_control_state()
                        if ctrl.get("stop_requested"):
                            break
                        if ctrl.get("pause_requested"):
                            with self.queue_condition:
                                self.queue_condition.wait(timeout=1.0)
                            continue

                        self._refresh_dependency_states()
                        pending = self.peek_pending_tasks()

                        def priority(task):
                            try:
                                return int(task.get("priority", 0) or 0)
                            except (TypeError, ValueError):
                                return 0

                        pending.sort(key=priority, reverse=True)
                        active_ids = {
                            task.get("task_id") for task in active.values()
                        }
                        for task in pending:
                            if len(active) >= 1:
                                break
                            if task["task_id"] in submitted_ids or task["task_id"] in active_ids:
                                continue
                            future = executor.submit(self.execute_omnipotent_task, task)
                            active[future] = task
                            submitted_ids.add(task["task_id"])

                        if active:
                            completed, _ = wait(
                                active, timeout=1.0, return_when=FIRST_COMPLETED
                            )
                            for future in completed:
                                task = active.pop(future)
                                submitted_ids.discard(task["task_id"])
                                try:
                                    future.result()
                                except Exception as exc:
                                    self.update_task_status(task["task_id"], "failed")
                                    self.emit_log(
                                        task["task_id"],
                                        f"❌ 佇列工作執行緒異常中斷：{exc}",
                                    )
                            continue

                        with self.queue_condition:
                            self.queue_condition.wait(timeout=5.0)
            finally:
                with self.queue_condition:
                    restart = self._queue_restart_requested
                    self._queue_restart_requested = False
                    self.daemon_active = False
                    self.current_task_id = "-"
                    self.current_task_stage = "Idle"
                    self.set_control_state("IDLE", pause_req=False, stop_req=False)
                if restart:
                    self.start_queue_loop()

        threading.Thread(target=_loop, daemon=True).start()