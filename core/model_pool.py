import time
import json
import base64
import math
import threading
import os
import re
import subprocess
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from typing import List, Dict, Optional, Callable
from config import OLLAMA_API_URL, ROSTER_FILE, MEM_USAGE_FILE, STATUS_FILE, DEFAULT_MAX_MEMORY_GB, CLOUD_CONFIG_FILE


class ModelPoolError(RuntimeError):
    pass


class DynamicModelPool:
    MODEL_CONTEXT_OVERHEAD_GB = 1.8
    BASE_CONTEXT_TOKENS = 4096
    MODEL_REFRESH_INTERVAL_SEC = 30.0
    MODEL_REFRESH_RETRY_INTERVAL_SEC = 5.0
    MEMORY_POLL_TIMEOUT_SEC = 10.0
    MEMORY_POLL_ATTEMPTS = 3
    MEMORY_POLL_RETRY_BASE_SEC = 0.5
    MEMORY_SYNC_CACHE_SEC = 1.0
    SLOT_MEMORY_WAIT_TIMEOUT_SEC = 120.0
    FIRST_TOKEN_TIMEOUT_SEC = 90.0

    MODEL_PROFILES = {
        "nomic-embed-text:latest": {
            "supports_completion": False,
            "supports_embeddings": True,
            "embedding_dimensions": 768,
            "endpoint": "/api/embeddings",
        },
        "bge-m3:latest": {
            "supports_completion": False,
            "supports_embeddings": True,
            "embedding_dimensions": 1024,
            "endpoint": "/api/embeddings",
        },
        "minicpm-v:8b": { "requires_images": True },
        "llava:latest": { "requires_images": True },
        "clef:latest": { "supports_completion": False, "supports_document_qa": True, "endpoint": "CLI" },
        "phi4:14b": { "max_context": 8192 },
        "mistral-small:24b": { "timeout_sec": 120, "exclusive": True, "keep_alive": 0 },
        "gemma2:27b": {},
        "flint:latest": { "timeout_sec": 120 },
        "nemotron-3.5-bolt:latest": { "timeout_sec": 120, "include_thinking": True },
        "qwen3.8:27b": { "include_thinking": True },
        "gemma4:12b": { "stream": False },
        "nemotron-3.5-bolt:e4b": {},
        "qwen3.5:4b": { "include_thinking": True },
        "qwen2.5-coder:14b": {},
        "qwen2.5:7b": { "stream": False },
        "qwen2.5-coder:32b": { "timeout_sec": 180, "exclusive": True, "keep_alive": 0 },
        "deepseek-r1:8b": { "max_context": 8192, "include_thinking": True },
        "devstral:24b": { "exclusive": True, "keep_alive": 0 },
        "deepseek-r1:14b": {},
        "llama3.1:8b": { "stream": False, "exclusive": True, "keep_alive": 0 },
        "qwen2.5:32b": { "exclusive": True, "keep_alive": 0 },
        "deepseek-r1:32b": { "timeout_sec": 120, "max_context": 8192, "include_thinking": True },
    }

    @classmethod
    def model_profile(cls, model_name: str) -> Dict:
        return cls.MODEL_PROFILES.get(str(model_name).casefold(), {})

    def __init__(self):
        self.api_url = OLLAMA_API_URL.rstrip("/")
        self.cond = threading.Condition()
        self._models_lock = threading.Lock()
        self._memory_sync_lock = threading.Lock()
        self.available_models: List[Dict] = []
        self.last_error: Optional[str] = None
        self.last_memory_error: Optional[str] = None
        self._last_model_refresh = 0.0
        self._last_model_refresh_attempt = 0.0
        self._last_memory_sync_at = 0.0
        self._last_memory_sync: Optional[Dict] = None
        self.reserved_memory_gb: float = 0.0
        self.active_workers: Dict[str, Dict] = {}
        self.usage_counter: Dict[str, int] = {}
        self.model_token_rates: Dict[str, float] = {}
        self.ollama_version: Optional[str] = None
        self._gpu_info_cache = None
        self._gpu_info_checked_at = 0.0
        self.use_cloud = False
        self.google_api_key = ""
        self.rate_limit_url = ""
        self.refresh_models()

    @staticmethod
    def _model_context_length(model: Dict) -> int:
        if model.get("inference_endpoint") == "GOOGLE_API":
            return model.get("max_context", 1048576)
        model_info = model.get("model_info", {})
        if isinstance(model_info, dict):
            lengths = [
                value for key, value in model_info.items()
                if key.endswith(".context_length")
                and isinstance(value, (int, float))
                and value > 0
            ]
            if lengths:
                return int(max(lengths))
        configured = os.environ.get("OLLAMA_CONTEXT_LENGTH", "")
        if configured.isdigit() and int(configured) > 0:
            return int(configured)
        return 0

    def context_limit(self, model: Dict) -> int:
        if model.get("inference_endpoint") == "GOOGLE_API":
            return model.get("max_context", 1048576)
        limit = self._model_context_length(model)
        configured = os.environ.get("OLLAMA_CONTEXT_LENGTH", "")
        if configured.isdigit() and int(configured) > 0:
            limit = min(limit or int(configured), int(configured))
        available_for_context = max(
            0.0, self.get_hardware_budget() - float(model.get("est_vram_gb", 0.0))
        )
        memory_limit = int(
            available_for_context
            / self.MODEL_CONTEXT_OVERHEAD_GB
            * self.BASE_CONTEXT_TOKENS
        )
        upper = limit or memory_limit
        low, high = 0, upper
        while low < high:
            middle = (low + high + 1) // 2
            if self._context_cache_gb(model, middle) <= available_for_context:
                low = middle
            else:
                high = middle - 1
        return low

    def context_limit_source(self, model: Dict) -> str:
        if model.get("inference_endpoint") == "GOOGLE_API":
            return "Google Cloud Quota"
        model_info = model.get("model_info", {})
        if isinstance(model_info, dict) and any(
            key.endswith(".context_length") and isinstance(value, (int, float)) and value > 0
            for key, value in model_info.items()
        ):
            return "model metadata"
        if os.environ.get("OLLAMA_CONTEXT_LENGTH", "").isdigit():
            return "OLLAMA_CONTEXT_LENGTH"
        return "estimated from memory budget"

    def context_capacity(self, role: str = "architect") -> int:
        models = self.rank_models_for_role(role, num_ctx=self.BASE_CONTEXT_TOKENS)
        return max((self.context_limit(model) for model in models), default=0)

    @staticmethod
    def _estimate_output_tokens(role: str, prompt: str, action_desc: str) -> int:
        context = f"{role} {action_desc} {prompt[:2000]}".casefold()
        if any(term in context for term in ["json", "拆解", "規劃", "route", "分鏡", "自主判斷"]):
            estimate = 600
        elif any(term in context for term in [
            "並行實作", "implement", "write code", "module", "python code", "javascript"
        ]):
            estimate = 2400
        elif any(term in context for term in [
            "report", "審查", "稽核", "screenplay", "劇本", "綜合交付報告"
        ]):
            estimate = 2000
        else:
            estimate = 800
        return max(256, min(estimate, 4096))

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        ascii_chars = sum(char.isascii() for char in text)
        other_chars = len(text) - ascii_chars
        return max(1, int(ascii_chars / 4 + other_chars / 1.5))

    def get_active_worker_progress(self) -> List[Dict]:
        now = time.monotonic()
        with self.cond:
            workers = []
            for worker in self.active_workers.values():
                elapsed = max(0.0, now - worker["started_monotonic"])
                token_count = worker["tokens"]
                rate = token_count / elapsed if elapsed > 0 else 0.0
                if rate <= 0:
                    rate = self.model_token_rates.get(worker["model"], 20.0)
                expected = worker["estimated_output_tokens"]
                remaining = max(0, expected - token_count)
                progress = min(95, int(token_count / expected * 100)) if expected else 0
                if token_count >= expected:
                    remaining = max(1, int(rate * 3))
                workers.append({
                    key: value
                    for key, value in worker.items()
                    if key != "started_monotonic"
                } | {
                    "elapsed_sec": round(elapsed, 1),
                    "progress_pct": progress,
                    "eta_sec": round(remaining / rate) if rate > 0 else None,
                    "tokens_per_sec": round(rate, 1),
                })
            return workers

    def get_hardware_budgets(self) -> Dict[str, float]:
        if not ROSTER_FILE.exists():
            return {"gpu_gb": 6.0, "ram_gb": 16.0, "total_gb": 22.0}
        try:
            data = json.loads(ROSTER_FILE.read_text(encoding="utf-8"))
            gpu = float(data.get("gpu_budget_gb", 6.0))
            ram = float(data.get("ram_budget_gb", 16.0))
            total = float(data.get("hardware_budget_gb", gpu + ram))
            if not math.isfinite(total) or total <= 0:
                raise ValueError("the configured value must be positive and finite")
            return {"gpu_gb": gpu, "ram_gb": ram, "total_gb": total}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {"gpu_gb": 6.0, "ram_gb": 16.0, "total_gb": 22.0}

    def get_hardware_budget(self) -> float:
        return self.get_hardware_budgets()["total_gb"]

    def set_hardware_budgets(self, gpu_gb: float, ram_gb: float):
        if not math.isfinite(gpu_gb) or not math.isfinite(ram_gb) or (gpu_gb + ram_gb) <= 0:
            raise ValueError("Hardware memory budget must be valid.")
        with self.cond:
            ROSTER_FILE.write_text(json.dumps({
                "gpu_budget_gb": round(float(gpu_gb), 1),
                "ram_budget_gb": round(float(ram_gb), 1),
                "hardware_budget_gb": round(float(gpu_gb + ram_gb), 1)
            }, indent=2), encoding="utf-8")
            self.cond.notify_all()

    def get_gpu_info(self) -> Dict:
        if getattr(self, "use_cloud", False):
            return {
                "available": True, "name": "Google Cloud TPU/GPU",
                "total_gb": 999.0, "free_gb": 999.0, "used_gb": 0.0, "error": None
            }

        now = time.monotonic()
        cached = getattr(self, "_gpu_info_cache", None)
        if cached is not None and now - getattr(self, "_gpu_info_checked_at", 0.0) < 5:
            return dict(cached)

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
            devices = []
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",", 3)]
                if len(parts) != 4:
                    continue
                try:
                    total_mb, free_mb, used_mb = (float(value) for value in parts[1:])
                except ValueError:
                    continue
                devices.append({
                    "name": parts[0],
                    "total_gb": total_mb / 1024,
                    "free_gb": free_mb / 1024,
                    "used_gb": used_mb / 1024,
                })
            if not devices:
                raise ValueError("nvidia-smi returned no GPU memory information.")
            info = {
                "available": True,
                "name": ", ".join(device["name"] for device in devices),
                "total_gb": round(sum(device["total_gb"] for device in devices), 2),
                "free_gb": round(sum(device["free_gb"] for device in devices), 2),
                "used_gb": round(sum(device["used_gb"] for device in devices), 2),
                "error": None,
            }
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            info = {
                "available": False,
                "name": None,
                "total_gb": None,
                "free_gb": None,
                "used_gb": None,
                "error": str(exc),
            }
        self._gpu_info_cache = info
        self._gpu_info_checked_at = now
        return dict(info)

    def _discover_google_models(self) -> List[Dict]:
        """動態查詢 Google API，不再寫死！並轉化為系統可用的並行模型列表"""
        discovered = []
        if not self.google_api_key:
            return discovered

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={self.google_api_key}"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models_raw = data.get("models", [])
                    for m in models_raw:
                        # 只要有支援生成端點，就認定可用
                        if "generateContent" not in m.get("supportedGenerationMethods", []):
                            continue
                        
                        full_name = m.get("name", "")
                        model_id = full_name.replace("models/", "")
                        max_ctx = int(m.get("inputTokenLimit", 250000))
                        desc = m.get("description", "Google Cloud Model")

                        discovered.append({
                            "name": model_id,
                            "category": "Google Cloud API",
                            "rpm": "Dynamic",
                            "tpm": "Dynamic",
                            "rpd": "Dynamic",
                            "size_gb": 0.1,
                            "est_vram_gb": 0.1,
                            "capabilities": ["completion", "vision", "decision"],
                            "capability_error": "",
                            "supports_completion": True,
                            "supports_decision": True,
                            "supports_embeddings": "embed" in model_id.lower(),
                            "supports_document_qa": False,
                            "inference_endpoint": "GOOGLE_API",
                            "timeout_sec": 180,
                            "max_context": max_ctx,
                            "is_vision": True,
                            "requires_images": False,
                            "usage_guidance": desc,
                            "details": {},
                            "model_info": {},
                            "model_context_length": max_ctx
                        })
        except Exception as exc:
            self.last_error = f"查詢 Google 模型失敗: {exc}"
        
        return discovered

    def refresh_models(self) -> List[Dict]:
        """Discover installed models and infer supported tasks from Ollama or Google API."""
        with self._models_lock:
            now = time.monotonic()
            
            try:
                if CLOUD_CONFIG_FILE.exists():
                    c = json.loads(CLOUD_CONFIG_FILE.read_text(encoding="utf-8"))
                    self.use_cloud = c.get("use_cloud", False)
                    self.google_api_key = c.get("google_api_key", "").strip()
                    self.rate_limit_url = c.get("rate_limit_url", "").strip()
            except Exception:
                self.use_cloud = False
                self.google_api_key = ""
                self.rate_limit_url = ""

            if getattr(self, "use_cloud", False):
                self.available_models = self._discover_google_models()
                self.last_error = None if self.google_api_key else "雲端模式已開啟，但尚未設定 Google API 金鑰。"
                self._last_model_refresh = now
                return self.available_models

            last_refresh = getattr(self, "_last_model_refresh", 0.0)
            last_attempt = getattr(self, "_last_model_refresh_attempt", 0.0)
            if now - last_refresh < self.MODEL_REFRESH_INTERVAL_SEC:
                return self.available_models
            if now - last_attempt < self.MODEL_REFRESH_RETRY_INTERVAL_SEC:
                return self.available_models
            self._last_model_refresh_attempt = now
            try:
                with httpx.Client(timeout=15.0) as client:
                    response = client.get(f"{self.api_url}/api/tags")
                    response.raise_for_status()
                    payload = response.json()
                    try:
                        version_response = client.get(f"{self.api_url}/api/version")
                        version_response.raise_for_status()
                        version_payload = version_response.json()
                        self.ollama_version = (
                            str(version_payload.get("version"))
                            if isinstance(version_payload, dict) and version_payload.get("version")
                            else None
                        )
                    except (httpx.HTTPError, ValueError, TypeError):
                        self.ollama_version = None
                if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                    raise ValueError("Ollama returned an invalid model list.")

                raw_models = [
                    model for model in payload["models"]
                    if isinstance(model, dict) and model.get("name")
                ]

                def inspect_model(model: Dict) -> Dict:
                    name = model["name"]
                    capabilities = []
                    capability_error = ""
                    capabilities_reported = False
                    model_metadata = {}
                    model_context_length = 0
                    try:
                        with httpx.Client(timeout=5.0) as client:
                            response = client.post(
                                f"{self.api_url}/api/show",
                                json={"model": name}
                            )
                            response.raise_for_status()
                            details = response.json()
                        if not isinstance(details, dict):
                            raise ValueError("Ollama returned invalid model capability metadata.")
                        model_metadata = details.get("model_info", {})
                        if not isinstance(model_metadata, dict):
                            model_metadata = {}
                        context_lengths = [
                            value for key, value in model_metadata.items()
                            if key.endswith(".context_length")
                            and isinstance(value, (int, float))
                            and value > 0
                        ]
                        if context_lengths:
                            model_context_length = int(max(context_lengths))
                        raw_capabilities = details.get("capabilities")
                        capabilities_reported = isinstance(raw_capabilities, list)
                        if isinstance(raw_capabilities, list):
                            capabilities = [
                                capability for capability in raw_capabilities
                                if isinstance(capability, str)
                            ]
                    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                        capability_error = str(exc)

                    details = model.get("details")
                    if not isinstance(details, dict):
                        details = {}
                    size = model.get("size", 0)
                    if not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0:
                        size = 0
                    size_gb = round(size / (1024 ** 3), 2)
                    capabilities_set = {value.casefold() for value in capabilities}
                    profile = self.model_profile(name)
                    supports_completion = (
                        "completion" in capabilities_set
                        or (not capabilities_reported and not capabilities)
                        or bool(capability_error)
                    )
                    if "supports_completion" in profile:
                        supports_completion = profile["supports_completion"]
                    return {
                        "name": name,
                        "category": "Local Ollama",
                        "rpm": "Local",
                        "tpm": "Local",
                        "rpd": "Local",
                        "size_gb": size_gb,
                        "est_vram_gb": round(max(1.5, size_gb * 1.12), 2),
                        "capabilities": capabilities,
                        "capability_error": capability_error,
                        "supports_completion": supports_completion,
                        "supports_decision": "decision" in capabilities_set,
                        "supports_embeddings": (
                            "embedding" in capabilities_set
                            or profile.get("supports_embeddings", False)
                        ),
                        "embedding_dimensions": profile.get("embedding_dimensions"),
                        "supports_document_qa": profile.get("supports_document_qa", False),
                        "inference_endpoint": profile.get("endpoint", "/api/generate"),
                        "timeout_sec": profile.get("timeout_sec"),
                        "max_context": profile.get("max_context"),
                        "is_vision": (
                            "vision" in capabilities_set
                            or profile.get("requires_images", False)
                        ),
                        "requires_images": profile.get("requires_images", False),
                        "usage_guidance": profile.get("usage_guidance", ""),
                        "details": details,
                        "model_info": model_metadata,
                        "model_context_length": model_context_length,
                    }

                models = []
                with ThreadPoolExecutor(max_workers=min(8, len(raw_models) or 1)) as executor:
                    futures = [executor.submit(inspect_model, model) for model in raw_models]
                    for future in as_completed(futures):
                        models.append(future.result())
                models.sort(key=lambda model: (not model["supports_completion"], model["size_gb"]))
                self.available_models = models
                completion_models = [model for model in models if model["supports_completion"]]
                self.last_error = (
                    None if completion_models
                    else "Ollama has no models with completion capability installed."
                )
                self._last_model_refresh = time.monotonic()
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
                self.last_error = f"Could not load models from Ollama at {self.api_url}: {exc}"
            return self.available_models

    def sync_realtime_memory(self) -> Dict:
        if getattr(self, "use_cloud", False):
            # 雲端模式下，欺騙記憶體與正在運行的模型（為了支援系統排程不中斷）
            active_mocks = []
            if self.available_models:
                active_mocks = [m["name"] for m in self.available_models[:2]]
            res = {
                "used_gb": 0.1, "ps_gb": 0.1,
                "running_models": active_mocks,
                "model_vram_gb": {n: 0.1 for n in active_mocks},
                "available": True,
            }
            self._last_memory_sync = dict(res)
            return res

        sync_lock = getattr(self, "_memory_sync_lock", None)
        if sync_lock is None:
            sync_lock = threading.Lock()
            self._memory_sync_lock = sync_lock
        with sync_lock:
            now = time.monotonic()
            cached = getattr(self, "_last_memory_sync", None)
            if (
                cached is not None
                and cached.get("available")
                and now - getattr(self, "_last_memory_sync_at", 0.0)
                < self.MEMORY_SYNC_CACHE_SEC
            ):
                return dict(cached, running_models=list(cached["running_models"]),
                            model_vram_gb=dict(cached["model_vram_gb"]))

            last_error = None
            for attempt in range(self.MEMORY_POLL_ATTEMPTS):
                try:
                    with httpx.Client(timeout=self.MEMORY_POLL_TIMEOUT_SEC) as client:
                        res = client.get(f"{self.api_url}/api/ps")
                        res.raise_for_status()
                        payload = res.json()
                    if not isinstance(payload, dict) or not isinstance(payload.get("models", []), list):
                        raise ValueError("Ollama returned an invalid running-model list.")

                    used_gb = 0.0
                    running_models = []
                    model_vram_gb = {}
                    for model in payload.get("models", []):
                        if not isinstance(model, dict):
                            continue
                        size = model.get("size_vram", 0) or model.get("size", 0)
                        used_gb += size / (1024 ** 3)
                        if model.get("name"):
                            name = model["name"]
                            running_models.append(name)
                            model_vram_gb[name] = round(
                                float(model.get("size_vram", 0) or 0) / (1024 ** 3), 2
                            )
                    self.last_memory_error = None
                    total_effective = round(max(used_gb, self.reserved_memory_gb), 2)
                    result = {
                        "used_gb": total_effective,
                        "ps_gb": round(used_gb, 2),
                        "running_models": running_models,
                        "model_vram_gb": model_vram_gb,
                        "available": True,
                    }
                    try:
                        MEM_USAGE_FILE.write_text(json.dumps({
                            "used_memory_gb": total_effective,
                            "ollama_ps_gb": round(used_gb, 2),
                            "reserved_gb": round(self.reserved_memory_gb, 2),
                            "running_models": running_models,
                        }), encoding="utf-8")
                    except OSError as exc:
                        self.last_memory_error = f"Could not save memory usage to {MEM_USAGE_FILE}: {exc}"
                        result["available"] = False
                    self._last_memory_sync = dict(result)
                    self._last_memory_sync_at = time.monotonic()
                    return result
                except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                    last_error = exc
                    if attempt + 1 < self.MEMORY_POLL_ATTEMPTS:
                        time.sleep(self.MEMORY_POLL_RETRY_BASE_SEC * (2 ** attempt))

            self.last_memory_error = (
                f"Could not read Ollama memory usage at {self.api_url} "
                f"after {self.MEMORY_POLL_ATTEMPTS} attempts: {last_error}"
            )
            return {
                "used_gb": 0.0,
                "ps_gb": 0.0,
                "running_models": [],
                "model_vram_gb": {},
                "available": False,
            }

    def unload_all_models(self):
        if getattr(self, "use_cloud", False):
            return
        ps = self.sync_realtime_memory()
        for m_name in ps["running_models"]:
            try:
                with httpx.Client(timeout=3.0) as client:
                    client.post(f"{self.api_url}/api/generate", json={"model": m_name, "keep_alive": 0})
            except Exception:
                pass

    def plan_greedy_swarm(self, num_ctx: int = BASE_CONTEXT_TOKENS) -> List[Dict]:
        self.refresh_models()
        if getattr(self, "use_cloud", False):
            # 雲端模式下發放充裕的工作槽給系統並行使用 (選擇 API 中抓到最快的 flash / lite)
            best_lite = next((m["name"] for m in self.available_models if "flash" in m["name"].lower()), None)
            if not best_lite and self.available_models:
                best_lite = self.available_models[0]["name"]
            return [{"name": best_lite, "slot_memory_gb": 0.1} for _ in range(6)]

        budgets = self.get_hardware_budgets()
        total_budget = budgets["total_gb"]
        gpu_budget = budgets["gpu_gb"]
        
        available_budget = max(
            0.0, total_budget - float(getattr(self, "reserved_memory_gb", 0.0))
        )
        candidates = [
            model for model in self.available_models
            if model.get("supports_completion", True)
            and self._slot_memory_gb(model, num_ctx) <= available_budget
        ]
        if not candidates:
            if not self.available_models or not any(
                model.get("supports_completion", True) for model in self.available_models
            ):
                raise ModelPoolError(self.last_error or "No completion-capable Ollama models are available.")
            raise ModelPoolError(
                f"No installed Ollama model fits the available memory budget of {available_budget:.1f} GB."
            )

        costs = {model["name"]: self._slot_memory_gb(model, num_ctx) for model in candidates}
        target_count = 0
        minimum_memory = 0.0
        for cost in sorted(costs.values()):
            if minimum_memory + cost > available_budget:
                break
            minimum_memory += cost
            target_count += 1

        def swarm_priority(model: Dict) -> tuple:
            slot_mem = costs[model["name"]]
            fits_in_gpu = (slot_mem <= gpu_budget)
            tier = 0 if fits_in_gpu else 1
            return (
                tier,
                self.usage_counter.get(model["name"], 0),
                slot_mem,
                model["name"]
            )

        candidates_by_fairness = sorted(candidates, key=swarm_priority)
        swarm = []
        used_gb = 0.0
        for index, model in enumerate(candidates_by_fairness):
            if len(swarm) == target_count:
                break
            slot_cost = costs[model["name"]]
            remaining_slots = target_count - len(swarm) - 1
            cheapest_remaining = sorted(
                costs[other["name"]]
                for other in candidates_by_fairness[index + 1:]
            )[:remaining_slots]
            if len(cheapest_remaining) != remaining_slots:
                continue
            if used_gb + slot_cost + sum(cheapest_remaining) > available_budget:
                continue
            slot = dict(model)
            slot["slot_memory_gb"] = round(slot_cost, 2)
            swarm.append(slot)
            used_gb += slot_cost
        return swarm

    @classmethod
    def _context_cache_gb(cls, model: Dict, num_ctx: int) -> float:
        if model.get("inference_endpoint") == "GOOGLE_API":
            return 0.01
        model_info = model.get("model_info", {})
        if isinstance(model_info, dict):
            def metadata_value(suffix: str) -> Optional[float]:
                values = [
                    value for key, value in model_info.items()
                    if key.endswith(suffix)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                ]
                return float(max(values)) if values else None

            layers = metadata_value(".block_count")
            heads = metadata_value(".attention.head_count")
            kv_heads = metadata_value(".attention.head_count_kv") or heads
            embedding = metadata_value(".embedding_length")
            if layers and heads and kv_heads and embedding:
                head_dim = embedding / heads
                kv_cache_gb = 4 * layers * kv_heads * head_dim * num_ctx / (1024 ** 3)
            else:
                kv_cache_gb = 0.0
        else:
            kv_cache_gb = 0.0
        context_scale = max(1, num_ctx / cls.BASE_CONTEXT_TOKENS)
        if kv_cache_gb:
            context_gb = cls.MODEL_CONTEXT_OVERHEAD_GB * context_scale
        else:
            model_size = model.get("size_gb")
            model_vram = (
                float(model_size) * 1.12
                if isinstance(model_size, (int, float))
                and math.isfinite(model_size)
                and model_size > 0
                else cls.MODEL_CONTEXT_OVERHEAD_GB / 0.25
            )
            fallback_per_4k = min(
                cls.MODEL_CONTEXT_OVERHEAD_GB, max(0.25, model_vram * 0.25)
            )
            context_gb = fallback_per_4k * context_scale
        return max(context_gb, kv_cache_gb)

    @classmethod
    def _slot_memory_gb(cls, model: Dict, num_ctx: int = BASE_CONTEXT_TOKENS) -> float:
        if model.get("inference_endpoint") == "GOOGLE_API":
            return 0.1
        return round(float(model["est_vram_gb"]) + cls._context_cache_gb(model, num_ctx), 2)

    def rank_models_for_role(self, role: str, num_ctx: int = BASE_CONTEXT_TOKENS) -> List[Dict]:
        self.refresh_models()
        if getattr(self, "use_cloud", False):
            return self.available_models

        budgets = self.get_hardware_budgets()
        total_budget = budgets["total_gb"]
        gpu_budget = budgets["gpu_gb"]
        
        required_capability = "decision" if role == "decision" else "completion"
        valid = [
            model for model in self.available_models
            if model.get(
                "supports_decision" if required_capability == "decision" else "supports_completion",
                required_capability != "decision"
            )
            and (
                role == "vision"
                or not self.model_profile(model["name"]).get("requires_images", False)
            )
            and (
                not self.model_profile(model["name"]).get("max_context")
                or num_ctx <= self.model_profile(model["name"])["max_context"]
            )
            and self._slot_memory_gb(model, num_ctx) <= total_budget
            and (
                self._model_context_length(model) <= 0
                or self.context_limit(model) >= num_ctx
            )
        ]
        if not valid:
            if required_capability == "decision":
                if not any(model.get("supports_decision", False) for model in self.available_models):
                    raise ModelPoolError("No decision-capable Ollama models are available.")
                raise ModelPoolError(
                    f"No decision-capable Ollama model fits the configured memory budget of {total_budget:.1f} GB."
                )
            if not self.available_models or not any(
                model.get("supports_completion", True) for model in self.available_models
            ):
                capability = "decision-capable" if required_capability == "decision" else "completion-capable"
                raise ModelPoolError(
                    f"No {capability} Ollama models are available."
                )
            raise ModelPoolError(
                f"No installed Ollama model fits the configured memory budget of {total_budget:.1f} GB."
            )

        if role == "vision":
            vision_models = [model for model in valid if model.get("is_vision", False)]
            if not vision_models:
                raise ModelPoolError("No installed Ollama model reports vision capability.")
            valid = vision_models

        high_quality_roles = {"architect", "arbiter", "coder", "vision"}
        gpu_info = self.get_gpu_info()
        memory_info = self.sync_realtime_memory()
        resident_vram = memory_info.get("model_vram_gb", {}) if memory_info.get("available") else {}

        def priority_score(model: Dict) -> tuple:
            slot_mem = self._slot_memory_gb(model, num_ctx)
            fits_in_gpu = (slot_mem <= gpu_budget)
            tier = 0 if fits_in_gpu else 1
            quality_metric = -model["est_vram_gb"] if role in high_quality_roles else model["est_vram_gb"]
            
            real_gpu_penalty = False
            if gpu_info.get("available"):
                model_name = model["name"]
                resident_gb = float(resident_vram.get(model_name, 0.0) or 0.0)
                additional_gb = max(0.0, slot_mem - resident_gb)
                if additional_gb > gpu_info["free_gb"]:
                    real_gpu_penalty = True
                    
            return (
                tier,
                real_gpu_penalty,
                quality_metric,
                -(self._model_context_length(model) or 0) if role in high_quality_roles else 0,
                self.usage_counter.get(model["name"], 0),
                model["name"]
            )

        return sorted(valid, key=priority_score)

    def get_role_escalation_path(self, role: str, num_ctx: int) -> List[Dict]:
        if getattr(self, "use_cloud", False):
            return self.rank_models_for_role(role, num_ctx)

        self.refresh_models()
        budget = self.get_hardware_budget()
        ranked = self.rank_models_for_role(role, num_ctx=num_ctx)
        candidates = [m for m in ranked if self._slot_memory_gb(m, num_ctx) <= budget]

        def role_affinity_score(model: Dict) -> tuple:
            name = model["name"].lower()
            size = float(model.get("size_gb", 0.0))
            vram = float(model.get("est_vram_gb", 0.0))
            is_coder = "coder" in name or "devstral" in name
            is_reasoning = "deepseek-r1" in name or "nemotron" in name or "phi4" in name
            is_vision = model.get("is_vision", False) or "llava" in name or "minicpm" in name

            if role == "coder":
                priority = 2 if is_coder else (1 if not is_vision else 0)
                return (priority, vram, size)
            elif role in {"architect", "arbiter"}:
                priority = 2 if is_reasoning else (1 if not is_vision else 0)
                return (priority, vram, size)
            elif role == "vision":
                priority = 2 if is_vision else 0
                return (priority, vram, size)
            else:
                priority = 2 if not is_coder and not is_vision else 1
                return (priority, vram, size)

        sorted_path = sorted(candidates, key=role_affinity_score)
        return sorted_path

    def select_model_for_role(
        self,
        role: str,
        preferred_model: Optional[str] = None,
        num_ctx: int = BASE_CONTEXT_TOKENS,
    ) -> Dict:
        ranked = self.rank_models_for_role(role, num_ctx)
        if preferred_model:
            found = next((model for model in ranked if model["name"] == preferred_model), None)
            if found:
                return found
        return ranked[0]

    def classify_task_routes(self, title: str, description: str, attached_file: str = "") -> Dict[str, bool]:
        if getattr(self, "use_cloud", False):
            return {}

        self.refresh_models()
        if not any(model.get("supports_decision", False) for model in self.available_models):
            return {}
        model_info = self.select_model_for_role("decision")
        questions = {
            "vision": {
                "type": "noul",
                "instructions": "Does completing this request require analyzing an attached image?",
            },
            "data_cleaning": {
                "type": "noul",
                "instructions": "Does completing this request require cleaning or transforming attached tabular data?",
            },
            "game": {
                "type": "noul",
                "instructions": "Does the user request creating or changing a playable game?",
            },
            "film": {
                "type": "noul",
                "instructions": "Does the user request producing a film, movie, trailer, or animated short?",
            },
            "software": {
                "type": "noul",
                "instructions": "Does the user request creating or changing software, a website, or a program?",
            },
        }
        state = (
            f"Task title: {title}\nRequest: {description}\n"
            f"Attached file: {attached_file or 'none'}"
        )[:12000]
        worker_id = f"route-{threading.get_ident()}-{int(time.time() * 1000) % 10000}"
        self.acquire_slot(worker_id, model_info, "Independent task routing", 4096, task_id="Routing")
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.api_url}/v1/systemone",
                    json={"model": model_info["name"], "state": state, "questions": questions}
                )
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Ollama returned an invalid task-routing response.")
            answers = payload.get("answers", {})
            if not isinstance(answers, dict):
                raise ValueError("Ollama returned invalid task-routing answers.")
            routes = {}
            for name in questions:
                answer = answers.get(name, {})
                probability = answer.get("noul") if isinstance(answer, dict) else None
                if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                    raise ValueError(f"Ollama omitted a valid yes/no probability for route '{name}'.")
                routes[name] = probability >= 0.5
            return routes
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                f"Ollama decision model '{model_info['name']}' failed to classify task routes: {exc}"
            ) from exc
        finally:
            self.release_slot(worker_id)

    def decide_dynamic_ctx(self, prompt_text: str, complexity_tier: int = 1) -> int:
        length = len(prompt_text)
        if length > 18000 or complexity_tier >= 3:
            return 32768
        elif length > 7000 or complexity_tier == 2:
            return 16384
        elif length > 2500:
            return 8192
        return 4096

    def _save_worker_status(self):
        try:
            STATUS_FILE.write_text(json.dumps(self.active_workers, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def acquire_slot(
        self,
        worker_id: str,
        model_info: Dict,
        action_desc: str,
        ctx: int,
        estimated_output_tokens: int = 800,
        task_id: str = "",
    ):
        model_name = model_info["name"]
        est_gb = model_info["est_vram_gb"]
        exclusive = self.model_profile(model_name).get("exclusive", False)
        memory_error_started = None

        with self.cond:
            while True:
                if getattr(self, "use_cloud", False):
                    break

                exclusive_worker_active = any(
                    self.model_profile(worker["model"]).get("exclusive", False)
                    for worker in self.active_workers.values()
                )
                if self.active_workers and (exclusive or exclusive_worker_active):
                    self.cond.wait(timeout=1.0)
                    continue
                budget = self.get_hardware_budget()
                context_memory = self._context_cache_gb(model_info, ctx)
                if self._slot_memory_gb(model_info, ctx) > budget:
                    raise ModelPoolError(
                        f"Model '{model_name}' needs an estimated "
                        f"{self._slot_memory_gb(model_info, ctx):.1f} GB including context, "
                        f"above the configured {budget:.1f} GB budget."
                    )
                ps = self.sync_realtime_memory()
                if not ps.get("available", True):
                    if memory_error_started is None:
                        memory_error_started = time.monotonic()
                    elapsed = time.monotonic() - memory_error_started
                    if elapsed >= self.SLOT_MEMORY_WAIT_TIMEOUT_SEC:
                        raise ModelPoolError(
                            self.last_memory_error or "Cannot verify current Ollama memory usage."
                        )
                    self.cond.wait(timeout=min(2.0, self.SLOT_MEMORY_WAIT_TIMEOUT_SEC - elapsed))
                    continue
                memory_error_started = None
                active_models = [w["model"] for w in self.active_workers.values()]
                if exclusive:
                    for loaded_model in ps["running_models"]:
                        if loaded_model == model_name or loaded_model in active_models:
                            continue
                        with httpx.Client(timeout=3.0) as client:
                            unload_response = client.post(
                                f"{self.api_url}/api/generate",
                                json={"model": loaded_model, "keep_alive": 0},
                            )
                            unload_response.raise_for_status()
                    if any(
                        loaded_model != model_name
                        for loaded_model in ps["running_models"]
                    ):
                        self.cond.wait(timeout=1.0)
                        continue
                already_loaded = (model_name in active_models) or (model_name in ps["running_models"])
                needed_gb = context_memory + (0 if already_loaded else est_gb)
                effective_usage = max(self.reserved_memory_gb, ps["used_gb"])

                if effective_usage + needed_gb <= budget:
                    break

                for m_in_vram in ps["running_models"]:
                    if m_in_vram not in active_models:
                        try:
                            with httpx.Client(timeout=2.0) as client:
                                client.post(f"{self.api_url}/api/generate", json={"model": m_in_vram, "keep_alive": 0})
                        except Exception:
                            pass

                ps = self.sync_realtime_memory()
                if not ps.get("available", True):
                    if memory_error_started is None:
                        memory_error_started = time.monotonic()
                    elapsed = time.monotonic() - memory_error_started
                    if elapsed >= self.SLOT_MEMORY_WAIT_TIMEOUT_SEC:
                        raise ModelPoolError(
                            self.last_memory_error or "Cannot verify current Ollama memory usage."
                        )
                    self.cond.wait(timeout=min(2.0, self.SLOT_MEMORY_WAIT_TIMEOUT_SEC - elapsed))
                    continue
                effective_usage = max(self.reserved_memory_gb, ps["used_gb"])
                if effective_usage + needed_gb <= budget:
                    break

                self.cond.wait(timeout=1.0)

            self.reserved_memory_gb = round(self.reserved_memory_gb + (0.1 if getattr(self, "use_cloud", False) else needed_gb), 2)
            self.usage_counter[model_name] = self.usage_counter.get(model_name, 0) + 1
            self.active_workers[worker_id] = {
                "model": model_name,
                "action": action_desc,
                "ctx": ctx,
                "tokens": 0,
                "estimated_output_tokens": estimated_output_tokens,
                "progress_pct": 0,
                "elapsed_sec": 0.0,
                "eta_sec": None,
                "task_id": task_id or worker_id.rsplit("-", 2)[0],
                "started_monotonic": time.monotonic(),
                "reserved_gb": round(0.1 if getattr(self, "use_cloud", False) else needed_gb, 2)
            }
            self._save_worker_status()
            self.sync_realtime_memory()

    def release_slot(self, worker_id: str):
        with self.cond:
            info = self.active_workers.pop(worker_id, None)
            if info:
                self.reserved_memory_gb = round(max(0.0, self.reserved_memory_gb - info["reserved_gb"]), 2)
            self._save_worker_status()
            self.sync_realtime_memory()
            self.cond.notify_all()

    def embed(self, model_name: str, text: str) -> List[float]:
        if getattr(self, "use_cloud", False):
            return [0.1] * 768

        profile = self.model_profile(model_name)
        if not profile.get("supports_embeddings"):
            raise ModelPoolError(f"Model '{model_name}' is not configured for text embeddings.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be a non-empty text string.")

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.api_url}/api/embeddings",
                json={"model": model_name, "prompt": text},
            )
            response.raise_for_status()
            payload = response.json()
        embedding = payload.get("embedding") if isinstance(payload, dict) else None
        expected_dimensions = profile.get("embedding_dimensions")
        if (
            not isinstance(embedding, list)
            or (expected_dimensions and len(embedding) != expected_dimensions)
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in embedding
            )
        ):
            expected = f" with {expected_dimensions} dimensions" if expected_dimensions else ""
            raise ModelPoolError(
                f"Ollama returned an invalid embedding for '{model_name}'{expected}."
            )
        return [float(value) for value in embedding]

    def ask_document(
        self,
        document_text: str,
        question: str,
        model_name: str = "clef:latest",
        timeout: float = 120.0,
    ) -> str:
        if getattr(self, "use_cloud", False):
            return self.generate("architect", f"Document:\n{document_text}\n\nQuestion: {question}", task_id="DOC_QA")

        if not self.model_profile(model_name).get("supports_document_qa"):
            raise ModelPoolError(f"Model '{model_name}' is not configured for document Q&A.")
        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError("Document text must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The document question must be a non-empty string.")
        try:
            result = subprocess.run(
                ["clef", "--ask", question],
                input=document_text,
                capture_output=True,
                check=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("The 'clef' command is not installed or not on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"The clef document query exceeded {timeout:.0f} seconds.") from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or "").strip()
            raise RuntimeError(f"The clef document query failed: {details or exc}") from exc
        answer = result.stdout.strip()
        if not answer:
            raise RuntimeError("The clef document query returned no answer.")
        return answer

    def _call_google_api_with_retry(self, url: str, payload: dict, max_retries: int = 4) -> str:
        last_error = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=180.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise RuntimeError(f"Google API returned no candidates: {resp.text}")
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if not parts:
                            raise RuntimeError(f"Google API returned empty content: {resp.text}")
                        return parts[0].get("text", "")
                    
                    err_msg = resp.text
                    if resp.status_code in [429, 503, 500]:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Google API Error: {err_msg}")
            except (httpx.TimeoutException, httpx.NetworkError, ConnectionError) as net_err:
                last_error = net_err
                time.sleep(2.0 * (attempt + 1))
            except Exception as e:
                raise e
        raise RuntimeError(f"Google API request failed after {max_retries} retries. Error: {last_error}")

    def generate(
        self,
        role: str,
        prompt: str,
        system_prompt: str = "",
        task_id: str = "SYS",
        action_desc: str = "推理中",
        images: Optional[List[str]] = None,
        num_ctx: Optional[int] = None,
        format_json: bool = False,
        temperature: float = 0.2,
        preferred_model: Optional[str] = None,
        on_token: Optional[Callable[[str, int], None]] = None,
        output_validator: Optional[Callable[[str], bool]] = None,
        interrupt_event: Optional[threading.Event] = None,
    ) -> str:
        final_ctx = num_ctx if num_ctx else self.decide_dynamic_ctx(prompt)
        effective_role = "vision" if images else role

        escalation_ladder = self.get_role_escalation_path(effective_role, final_ctx)
        if not escalation_ladder:
            budget = self.get_hardware_budget()
            raise ModelPoolError(
                f"在目前 {budget} GB 預算下，找不到任何適用於 [{effective_role}] 角色的可用模型。"
            )

        m_info = None
        if preferred_model:
            m_info = next((m for m in escalation_ladder if m["name"] == preferred_model), None)

        if not m_info:
            if effective_role in {"architect", "arbiter"}:
                m_info = escalation_ladder[-1]
            else:
                m_info = escalation_ladder[0]

        model_name = m_info["name"]
        profile = self.model_profile(model_name)
        if profile.get("requires_images") and not images:
            raise ModelPoolError(f"Model '{model_name}' requires at least one image.")
        if images and not any(image for image in images):
            raise ValueError("At least one image file is required for a vision request.")
        final_ctx = min(final_ctx, profile.get("max_context", final_ctx))
        estimated_output_tokens = self._estimate_output_tokens(role, prompt, action_desc)

        attempted_models = set()
        model_failure_reasons = {}
        budget = self.get_hardware_budget()

        while True:
            worker_id = (
                f"{task_id}-{threading.get_ident()}-{int(time.time()*1000)%10000}"
                f"-ctx{final_ctx}"
            )
            self.acquire_slot(
                worker_id,
                m_info,
                f"[{task_id}] {action_desc} (Ctx {final_ctx} | {model_name})",
                final_ctx,
                estimated_output_tokens=estimated_output_tokens,
                task_id=task_id,
            )
            attempted_models.add(model_name)
            chunks = []
            done_reason = ""
            actual_output_tokens = 0
            has_eval_count = False
            prompt_eval_tokens = 0
            prompt_token_scale = 1.0
            estimated_prompt_tokens = self._estimate_text_tokens(system_prompt + prompt)
            attempt_started = time.monotonic()
            first_token_timeout = threading.Event()
            result = ""
            invalid_json = False

            try:
                if getattr(self, "use_cloud", False):
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.google_api_key}"
                    parts = []
                    if images:
                        for img_path in images:
                            if img_path:
                                with open(img_path, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode("utf-8")
                                    mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"
                                    parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                    parts.append({"text": prompt})
                    
                    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": temperature}}
                    if format_json:
                        payload["generationConfig"]["responseMimeType"] = "application/json"
                    if system_prompt:
                        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                        
                    with self.cond:
                        w = self.active_workers.get(worker_id)
                        if w:
                            w["progress_pct"] = 45
                    self._save_worker_status()
                    
                    result = self._call_google_api_with_retry(url, payload)
                        
                    with self.cond:
                        w = self.active_workers.get(worker_id)
                        if w:
                            w["progress_pct"] = 100
                            w["tokens"] = max(1, len(result) // 4)
                    self._save_worker_status()
                    
                    if on_token:
                        on_token(result, len(result) // 4)
                    done_reason = "stop"

                else:
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": profile.get("stream", True),
                        "keep_alive": profile.get("keep_alive", "15m"),
                        "options": {"temperature": temperature, "num_ctx": final_ctx}
                    }
                    if format_json:
                        payload["format"] = "json"
                    if images:
                        encoded_images = []
                        for image_path in images:
                            if image_path:
                                with open(image_path, "rb") as image_file:
                                    encoded_images.append(base64.b64encode(image_file.read()).decode("utf-8"))
                        payload["images"] = encoded_images

                    timeout_seconds = max(
                        self.FIRST_TOKEN_TIMEOUT_SEC,
                        profile.get("timeout_sec", self.FIRST_TOKEN_TIMEOUT_SEC),
                    )
                    timeout = httpx.Timeout(
                        connect=10.0,
                        read=timeout_seconds,
                        write=timeout_seconds,
                        pool=10.0,
                    )
                    with httpx.Client(timeout=timeout) as client:
                        with client.stream("POST", f"{self.api_url}/api/generate", json=payload) as resp:
                            resp.raise_for_status()
                            def close_after_no_first_token():
                                first_token_timeout.set()
                                close = getattr(resp, "close", None)
                                if callable(close):
                                    close()

                            first_token_timer = threading.Timer(
                                timeout_seconds,
                                close_after_no_first_token,
                            )
                            first_token_timer.daemon = True
                            first_token_timer.start()
                            interrupt_stop = threading.Event()
                            interrupt_thread = None
                            if interrupt_event is not None:
                                def interrupt_stream():
                                    while not interrupt_stop.wait(0.05):
                                        if interrupt_event.is_set():
                                            close = getattr(resp, "close", None)
                                            if callable(close):
                                                try:
                                                    close()
                                                except Exception:
                                                    pass
                                            return

                                interrupt_thread = threading.Thread(
                                    target=interrupt_stream, daemon=True
                                )
                                interrupt_thread.start()
                            try:
                                for line in resp.iter_lines():
                                    if interrupt_event is not None and interrupt_event.is_set():
                                        raise ModelPoolError("Ollama generation was interrupted.")
                                    if not line:
                                        continue
                                    data = json.loads(line)
                                    if data.get("error"):
                                        raise RuntimeError(data["error"])
                                    if "eval_count" in data:
                                        actual_output_tokens = max(0, int(data.get("eval_count") or 0))
                                        has_eval_count = True
                                    reported_prompt_tokens = int(data.get("prompt_eval_count") or 0)
                                    if reported_prompt_tokens > 0:
                                        prompt_eval_tokens = reported_prompt_tokens
                                        prompt_token_scale = reported_prompt_tokens / estimated_prompt_tokens
                                    tok = data.get("response", "")
                                    if profile.get("include_thinking"):
                                        thinking = data.get("thinking", "")
                                        if thinking:
                                            tok = f"{thinking}{tok}"
                                    if tok:
                                        first_token_timer.cancel()
                                        chunks.append(tok)
                                        tokens = max(
                                            1,
                                            round(
                                                self._estimate_text_tokens("".join(chunks))
                                                * prompt_token_scale
                                            ),
                                        )
                                        with self.cond:
                                            worker = self.active_workers.get(worker_id)
                                            if worker:
                                                if tokens >= worker["estimated_output_tokens"]:
                                                    worker["estimated_output_tokens"] = int(tokens * 1.25) + 128
                                                worker["tokens"] = tokens
                                                worker["progress_pct"] = min(
                                                    95, int(tokens / worker["estimated_output_tokens"] * 100)
                                                )
                                        self._save_worker_status()
                                        if on_token:
                                            on_token(tok, tokens)
                                    if data.get("done"):
                                        done_reason = str(data.get("done_reason") or "")
                                        prompt_eval_tokens = max(prompt_eval_tokens, reported_prompt_tokens)
                                        if has_eval_count:
                                            with self.cond:
                                                worker = self.active_workers.get(worker_id)
                                                if worker:
                                                    worker["tokens"] = actual_output_tokens
                                                    worker["estimated_output_tokens"] = actual_output_tokens
                                                    worker["progress_pct"] = 99
                                                    worker["eval_count"] = actual_output_tokens
                                        if prompt_eval_tokens:
                                            with self.cond:
                                                worker = self.active_workers.get(worker_id)
                                                if worker:
                                                    worker["prompt_eval_count"] = prompt_eval_tokens
                            finally:
                                first_token_timer.cancel()
                                interrupt_stop.set()
                                if interrupt_thread is not None:
                                    interrupt_thread.join(timeout=0.2)
                            if first_token_timeout.is_set():
                                raise RuntimeError(
                                    f"Model '{model_name}' produced no output tokens within "
                                    f"{timeout_seconds:.0f} seconds."
                                )
                            if interrupt_event is not None and interrupt_event.is_set():
                                raise ModelPoolError("Ollama generation was interrupted.")
                    result = "".join(chunks)

                if not result.strip() or (has_eval_count and actual_output_tokens == 0):
                    raise RuntimeError(
                        f"Model '{model_name}' finished without generating any tokens "
                        "or reported an output token count of zero."
                    )
                requested_ctx = re.search(r"\[\[REQUEST_MORE_CONTEXT(?::(\d+))?\]\]", result)
                if requested_ctx:
                    result = re.sub(r"\[\[REQUEST_MORE_CONTEXT(?::\d+)?\]\]", "", result).strip()

                if format_json:
                    clean_json_str = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
                    json_match = re.search(r"(\{.*\})", clean_json_str, re.DOTALL)
                    if json_match:
                        try:
                            json.loads(json_match.group(1))
                            result = json_match.group(1)
                        except Exception:
                            invalid_json = True
                    else:
                        try:
                            json.loads(clean_json_str)
                            result = clean_json_str
                        except Exception:
                            invalid_json = True
                invalid_output = output_validator is not None and not output_validator(result)
                validation_failed = invalid_json or invalid_output
                should_expand = (done_reason == "length" or requested_ctx is not None)

                if not validation_failed and not should_expand:
                    return result

                fail_reason = "長度截斷" if should_expand else ("JSON解析失敗" if invalid_json else "輸出檢驗未通過")
                model_failure_reasons[model_name] = fail_reason

                target_ctx = int(requested_ctx.group(1)) if requested_ctx and requested_ctx.group(1) else final_ctx * 2
                max_allowed_ctx = min(16384, self.context_limit(m_info) or 8192)
                target_ctx = min(max_allowed_ctx, max(final_ctx * 2, target_ctx))

                current_idx = next((i for i, m in enumerate(escalation_ladder) if m["name"] == model_name), -1)
                higher_candidates = escalation_ladder[current_idx + 1:] + escalation_ladder[:current_idx]
                next_model_choice = None

                for candidate in higher_candidates:
                    c_name = candidate["name"]
                    if c_name not in attempted_models:
                        if self._slot_memory_gb(candidate, target_ctx) <= budget:
                            next_model_choice = (candidate, target_ctx)
                            break
                        elif self._slot_memory_gb(candidate, final_ctx) <= budget:
                            next_model_choice = (candidate, final_ctx)
                            break

                if next_model_choice:
                    m_info, final_ctx = next_model_choice
                    model_name = m_info["name"]
                    profile = self.model_profile(model_name)
                    continue

                model_cap = self._model_context_length(m_info) or self.context_limit(m_info)
                context_cap = min(model_cap, self.context_limit(m_info))
                next_ctx = min(context_cap, target_ctx)
                if next_ctx > final_ctx and self._slot_memory_gb(m_info, next_ctx) <= budget:
                    final_ctx = next_ctx
                    continue

                reasons_str = "; ".join(f"{k}: {v}" for k, v in model_failure_reasons.items())
                raise ModelPoolError(
                    f"[{effective_role}] 任務失敗：已嘗試該職責專業梯隊內的所有適用模型。記錄：{reasons_str}"
                )

            except Exception as exc:
                if first_token_timeout.is_set():
                    exc = RuntimeError(
                        f"Model '{model_name}' produced no output tokens within "
                        f"{timeout_seconds:.0f} seconds."
                    )
                if interrupt_event is not None and interrupt_event.is_set():
                    raise ModelPoolError("Ollama generation was interrupted.") from exc
                if isinstance(exc, ModelPoolError):
                    raise

                model_failure_reasons[model_name] = str(exc)
                fallback_choice = next(
                    (
                        m for m in escalation_ladder
                        if m["name"] not in attempted_models
                        and self._slot_memory_gb(m, final_ctx) <= budget
                    ),
                    None
                )
                if fallback_choice is not None:
                    m_info = fallback_choice
                    model_name = m_info["name"]
                    profile = self.model_profile(model_name)
                    continue

                reasons_str = "; ".join(f"{k}: {v}" for k, v in model_failure_reasons.items())
                raise RuntimeError(
                    f"[{effective_role}] 推理失敗，專業梯隊模型皆無法完成。原因：{reasons_str}"
                ) from exc
            finally:
                elapsed = max(0.001, time.monotonic() - attempt_started)
                if actual_output_tokens:
                    measured_rate = actual_output_tokens / elapsed
                    with self.cond:
                        previous_rate = self.model_token_rates.get(model_name)
                        self.model_token_rates[model_name] = (
                            measured_rate if previous_rate is None
                            else previous_rate * 0.7 + measured_rate * 0.3
                        )
                self.release_slot(worker_id)
                