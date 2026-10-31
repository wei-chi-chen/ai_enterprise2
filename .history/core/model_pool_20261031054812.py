import time
import json
import base64
import math
import threading
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from typing import List, Dict, Optional, Callable
from config import OLLAMA_API_URL, ROSTER_FILE, MEM_USAGE_FILE, STATUS_FILE, DEFAULT_MAX_MEMORY_GB


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
            "usage_guidance": "應輸入單純文字字串至/api/embeddings端點的prompt欄位會輸出768維度的浮點數向量陣列必須走/api/embeddings端點且禁止當作文字生成模型調用才能成功",
        },
        "bge-m3:latest": {
            "supports_completion": False,
            "supports_embeddings": True,
            "embedding_dimensions": 1024,
            "endpoint": "/api/embeddings",
            "usage_guidance": "應輸入多語言文字字串至/api/embeddings端點的prompt欄位會輸出1024維度的浮點數語意向量陣列必須走/api/embeddings端點才能成功",
        },
        "minicpm-v:8b": {
            "requires_images": True,
            "usage_guidance": "應輸入包含文字prompt與base64編碼圖片字串陣列images的JSON至/api/generate或/api/chat端點會輸出針對該圖片的結構化描述與文字辨識分析結果必須包含至少一張合法圖片Payload才能成功",
        },
        "llava:latest": {
            "requires_images": True,
            "usage_guidance": "應輸入文字提示詞與base64編碼圖片陣列至/api/generate端點會輸出多模態視覺問答與物件場景描述文本必須傳入圖片資料才能成功",
        },
        "clef:latest": {
            "supports_completion": False,
            "supports_document_qa": True,
            "endpoint": "CLI",
            "usage_guidance": "應透過標準輸入stdin管道灌入欲分析的長文本內容並在命令列搭配--ask帶入問題參數會輸出針對該輸入文件內容的精確問答結論與資訊抽取結果必須透過管道提供文件本文搭配--ask使用且禁止走普通generate端點才能成功",
        },
        "phi4:14b": {
            "max_context": 8192,
            "usage_guidance": "應輸入自然語言提示詞至/api/generate端點會輸出結構化文本推理分析或解答必須控制上下文長度避免突發顯存溢出才能成功",
        },
        "mistral-small:24b": {
            "timeout_sec": 120,
            "exclusive": True,
            "keep_alive": 0,
            "usage_guidance": "應輸入通用任務提示詞或長篇指令至/api/generate端點會輸出邏輯嚴密的文本生成內容或總結分析必須維持獨立載入並建議設定keep_alive為0避免顯存堆疊才能成功",
        },
        "gemma2:27b": {
            "usage_guidance": "應輸入問答指令或語言轉換提示詞至/api/generate端點會輸出知識豐富的繁體中文或英文文本回答必須保留足夠主機RAM以進行層間快取調度才能成功",
        },
        "flint:latest": {
            "timeout_sec": 120,
            "usage_guidance": "應輸入多步驟任務指令至/api/generate端點會輸出對應的任務規劃或綜合推論結果因耗時較長需將請求逾時時間設定至120秒以上才能成功",
        },
        "nemotron-3.5-bolt:latest": {
            "timeout_sec": 120,
            "include_thinking": True,
            "usage_guidance": "應輸入複雜決策或長推導問題至/api/generate端點會輸出包含完整思考鏈與最終決策的長篇文本因輸出含長思考標籤需提高timeout秒數並容許解析思維文本才能成功",
        },
        "qwen3.8:27b": {
            "include_thinking": True,
            "usage_guidance": "應輸入深度對話或領域分析提示詞至/api/generate端點會輸出包含中間思考過程與最終觀點的完整文本必須在解析時相容thinking欄位才能成功",
        },
        "gemma4:12b": {
            "stream": False,
            "usage_guidance": "應輸入綜合任務提示詞至/api/generate端點會輸出結構化文本或初步思考推論答案必須設定非串流模式以便一次獲取完整字串才能成功",
        },
        "nemotron-3.5-bolt:e4b": {
            "usage_guidance": "應輸入輕量級任務提示詞至/api/generate端點會輸出低延遲且精煉的文字結論直接發送標準文字Payload即可成功",
        },
        "qwen3.5:4b": {
            "include_thinking": True,
            "usage_guidance": "應輸入數學邏輯或多輪問答提示詞至/api/generate端點會輸出含有ThinkingProcess結構思考歷程的文本解答必須同時抓取response與thinking內容才能成功",
        },
        "qwen2.5-coder:14b": {
            "usage_guidance": "應輸入程式碼生成重構或除錯指令至/api/generate端點會輸出包含完整語法高亮代碼區塊與技術說明的程式碼直接發送標準Prompt即可成功",
        },
        "qwen2.5:7b": {
            "stream": False,
            "usage_guidance": "應輸入日常對話或快速摘要文字至/api/generate端點會輸出低延遲且流暢的通用文字回應直接發送標準非串流JSON即可成功",
        },
        "qwen2.5-coder:32b": {
            "timeout_sec": 180,
            "exclusive": True,
            "keep_alive": 0,
            "usage_guidance": "應輸入大型專案架構代碼生成或複雜演算法任務至/api/generate端點會輸出高品質代碼實現與模組設計說明因單次運算耗時較長需放寬請求逾時至180秒以上且避免併發其他模型才能成功",
        },
        "deepseek-r1:8b": {
            "max_context": 8192,
            "include_thinking": True,
            "usage_guidance": "應輸入數理邏輯推理問題至/api/generate端點會輸出帶有think標籤的深度思維推導過程與最終答案必須限定num_ctx長度在合理範圍並容許解析思維標籤才能成功",
        },
        "devstral:24b": {
            "exclusive": True,
            "keep_alive": 0,
            "usage_guidance": "應輸入開發者任務系統指令或複雜上下文至/api/generate端點會輸出技術規格說明與結構化開發文件需在單獨載入且前一個模型完全釋放下執行以保證CUDAHostBuffer分配成功",
        },
        "deepseek-r1:14b": {
            "usage_guidance": "應輸入中等複雜度演算法或邏輯推導提示詞至/api/generate端點會輸出細緻的多步驟思考推導與推論答案必須預留計算圖動態重分配緩衝才能成功",
        },
        "llama3.1:8b": {
            "stream": False,
            "exclusive": True,
            "keep_alive": 0,
            "usage_guidance": "應輸入通用對話或文本任務指令至/api/generate端點會輸出穩定且符合指令要求的文本內容直接傳入非串流Prompt並確保單一模型獨佔顯存即可成功",
        },
        "qwen2.5:32b": {
            "exclusive": True,
            "keep_alive": 0,
            "usage_guidance": "應輸入跨領域知識問答或長篇內容撰寫指令至/api/generate端點會輸出全面且語意連貫的大規模生成文本因權重佔用接近顯存邊界必須在單獨載入且設定keep_alive為0釋放前置模型時才能成功",
        },
        "deepseek-r1:32b": {
            "timeout_sec": 120,
            "max_context": 8192,
            "include_thinking": True,
            "usage_guidance": "應輸入高難度數學競賽題或深層推理證明問題至/api/generate端點會輸出極其詳盡的思維鏈演繹與最終驗證答案必須設定逾時120秒以上且手動限制上下文長度避免爆顯存才能成功",
        },
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
        self.refresh_models()

    @staticmethod
    def _model_context_length(model: Dict) -> int:
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
        """Return the model/runtime context ceiling, further bounded by the memory budget."""
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
        if any(term in context for term in [
            "並行實作", "implement", "write code", "module", "python code", "javascript"
        ]):
            estimate = 2400
        elif any(term in context for term in [
            "report", "審查", "稽核", "screenplay", "劇本", "分析", "分析需求"
        ]):
            estimate = 1600
        elif any(term in context for term in ["json", "拆解", "規劃", "route", "分鏡"]):
            estimate = 1000
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

    def get_hardware_budget(self) -> float:
        if not ROSTER_FILE.exists():
            return DEFAULT_MAX_MEMORY_GB
        try:
            data = json.loads(ROSTER_FILE.read_text(encoding="utf-8"))
            budget = float(data.get("hardware_budget_gb", DEFAULT_MAX_MEMORY_GB))
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("the configured value must be positive and finite")
            return budget
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read the hardware memory budget from {ROSTER_FILE}: {exc}") from exc

    def set_hardware_budget(self, budget_gb: float):
        if not math.isfinite(budget_gb) or budget_gb <= 0:
            raise ValueError("Hardware memory budget must be a positive finite number.")
        with self.cond:
            ROSTER_FILE.write_text(json.dumps({"hardware_budget_gb": round(float(budget_gb), 1)}, indent=2), encoding="utf-8")
            self.cond.notify_all()

    def get_gpu_info(self) -> Dict:
        """Return NVIDIA GPU memory from nvidia-smi when available."""
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

    def refresh_models(self) -> List[Dict]:
        """Discover installed models and infer supported tasks from Ollama's model metadata."""
        with self._models_lock:
            now = time.monotonic()
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
        """Read Ollama's loaded-model memory with bounded retries and short success caching."""
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
        ps = self.sync_realtime_memory()
        for m_name in ps["running_models"]:
            try:
                with httpx.Client(timeout=3.0) as client:
                    client.post(f"{self.api_url}/api/generate", json={"model": m_name, "keep_alive": 0})
            except Exception:
                pass

    def plan_greedy_swarm(self, num_ctx: int = BASE_CONTEXT_TOKENS) -> List[Dict]:
        """Select as many distinct completion-capable models as fit the configured budget."""
        self.refresh_models()
        budget = self.get_hardware_budget()
        available_budget = max(
            0.0, budget - float(getattr(self, "reserved_memory_gb", 0.0))
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

        candidates_by_fairness = sorted(
            candidates,
            key=lambda model: (
                self.usage_counter.get(model["name"], 0),
                costs[model["name"]],
                model["name"]
            )
        )
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
        return round(float(model["est_vram_gb"]) + cls._context_cache_gb(model, num_ctx), 2)

    def rank_models_for_role(self, role: str, num_ctx: int = BASE_CONTEXT_TOKENS) -> List[Dict]:
        self.refresh_models()
        budget = self.get_hardware_budget()
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
            and self._slot_memory_gb(model, num_ctx) <= budget
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
                    f"No decision-capable Ollama model fits the configured memory budget of {budget:.1f} GB."
                )
            if not self.available_models or not any(
                model.get("supports_completion", True) for model in self.available_models
            ):
                capability = "decision-capable" if required_capability == "decision" else "completion-capable"
                raise ModelPoolError(
                    f"No {capability} Ollama models are available."
                )
            raise ModelPoolError(
                f"No installed Ollama model fits the configured memory budget of {budget:.1f} GB."
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

        def gpu_fit_penalty(model: Dict) -> bool:
            if not gpu_info.get("available"):
                return False
            model_name = model["name"]
            resident_gb = float(resident_vram.get(model_name, 0.0) or 0.0)
            additional_gb = max(
                0.0,
                self._slot_memory_gb(model, num_ctx) - resident_gb,
            )
            return additional_gb > gpu_info["free_gb"]

        return sorted(
            valid,
            key=lambda model: (
                gpu_fit_penalty(model),
                -model["est_vram_gb"] if role in high_quality_roles else model["est_vram_gb"],
                -(self._model_context_length(model) or 0) if role in high_quality_roles else 0,
                self.usage_counter.get(model["name"], 0),
                model["name"]
            )
        )

    def get_role_escalation_path(self, role: str, num_ctx: int) -> List[Dict]:
        """
        根據當前任務角色，篩選符合該領域專業的模型，並依照專業適配度與能力階梯排列，
        提供角色專屬的向上升級路徑，而非盲目遍歷無關模型。
        """
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
        """Use an installed decision-capable model to add independent task-routing signals."""
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
        """Reserve the model footprint and per-request context memory before inference."""
        model_name = model_info["name"]
        est_gb = model_info["est_vram_gb"]
        exclusive = self.model_profile(model_name).get("exclusive", False)
        memory_error_started = None

        with self.cond:
            while True:
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

            self.reserved_memory_gb = round(self.reserved_memory_gb + needed_gb, 2)
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
                "reserved_gb": round(needed_gb, 2)
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
        """Create a single text embedding using Ollama's embeddings endpoint."""
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
        """Ask the CLI model about document text supplied through stdin."""
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
            try:
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
                                            worker["tokens"] = tokens
                                            worker["progress_pct"] = min(
                                                95, int(tokens / estimated_output_tokens * 100)
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

                invalid_json = False
                if format_json:
                    # 去除 <think>...</think> 區塊，並從文本中提取標準 JSON 物件
                    clean_json_str = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
                    json_match = re.search(r"(\{.*\})", clean_json_str, re.DOTALL)
                    if json_match:
                        try:
                            json.loads(json_match.group(1))
                            result = json_match.group(1) # 將提取後的純 JSON 回填，避免外層 json.loads 再次報錯
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
                target_ctx = max(final_ctx * 2, target_ctx)

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