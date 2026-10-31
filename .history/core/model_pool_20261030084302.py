import time
import json
import base64
import threading
import httpx
from typing import List, Dict, Optional, Callable
from config import OLLAMA_API_URL, ROSTER_FILE, MEM_USAGE_FILE, STATUS_FILE, DEFAULT_MAX_MEMORY_GB

class DynamicModelPool:
    def __init__(self):
        self.api_url = OLLAMA_API_URL.rstrip("/")
        self.cond = threading.Condition()
        self.available_models: List[Dict] = []
        self.reserved_memory_gb: float = 0.0
        self.active_workers: Dict[str, Dict] = {}
        self.rr_counter: int = 0
        self.refresh_models()

    def get_hardware_budget(self) -> float:
        if ROSTER_FILE.exists():
            try:
                data = json.loads(ROSTER_FILE.read_text(encoding="utf-8"))
                return float(data.get("hardware_budget_gb", DEFAULT_MAX_MEMORY_GB))
            except Exception:
                pass
        return DEFAULT_MAX_MEMORY_GB

    def set_hardware_budget(self, budget_gb: float):
        with self.cond:
            ROSTER_FILE.write_text(json.dumps({"hardware_budget_gb": round(float(budget_gb), 1)}, indent=2), encoding="utf-8")
            self.cond.notify_all()

    def refresh_models(self) -> List[Dict]:
        """即時掃描本機所有已下載之 Ollama 模型"""
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.api_url}/api/tags")
                if res.status_code == 200:
                    raw = res.json().get("models", [])
                    models = []
                    for m in raw:
                        size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
                        est_vram = round(size_gb * 1.15, 2)
                        name = m.get("name", "")
                        models.append({
                            "name": name,
                            "size_gb": size_gb,
                            "est_vram_gb": est_vram,
                            "is_vision": any(k in name.lower() for k in ["llava", "vision", "minicpm", "bakllava", "vl"]),
                            "is_coder": any(k in name.lower() for k in ["coder", "code", "starcoder", "codellama", "deepseek"])
                        })
                    models.sort(key=lambda x: x["size_gb"], reverse=True)
                    self.available_models = models
        except Exception:
            pass
        return self.available_models

    def sync_realtime_memory(self) -> Dict:
        """同步 Ollama /api/ps 實時顯存與預留槽位總和"""
        used_gb = 0.0
        running_models = []
        try:
            with httpx.Client(timeout=2.5) as client:
                res = client.get(f"{self.api_url}/api/ps")
                if res.status_code == 200:
                    for m in res.json().get("models", []):
                        sz = m.get("size_vram", 0) or m.get("size", 0)
                        used_gb += sz / (1024 ** 3)
                        running_models.append(m.get("name"))
        except Exception:
            pass

        total_effective = round(max(used_gb, self.reserved_memory_gb), 2)
        try:
            MEM_USAGE_FILE.write_text(json.dumps({
                "used_memory_gb": total_effective,
                "ollama_ps_gb": round(used_gb, 2),
                "reserved_gb": round(self.reserved_memory_gb, 2),
                "running_models": running_models
            }), encoding="utf-8")
        except Exception:
            pass
        return {"used_gb": total_effective, "ps_gb": round(used_gb, 2), "running_models": running_models}

    def unload_idle_models(self, keep_models: List[str] = None):
        keep = set(keep_models or [])
        ps = self.sync_realtime_memory()
        for m_name in ps["running_models"]:
            if m_name not in keep:
                try:
                    with httpx.Client(timeout=3.0) as client:
                        client.post(f"{self.api_url}/api/generate", json={"model": m_name, "keep_alive": 0})
                except Exception:
                    pass

    def select_model_for_worker(self, role: str, worker_index: int = 0) -> Dict:
        """依據角色與並行編號，自動分流指派給本機不同的適當模型"""
        self.refresh_models()
        budget = self.get_hardware_budget()
        valid = [m for m in self.available_models if m["est_vram_gb"] <= budget]
        if not valid:
            fallback = self.available_models[-1] if self.available_models else {"name": "qwen2.5:7b", "size_gb": 5.0, "est_vram_gb": 5.5}
            return fallback

        if role == "vision":
            vis = [m for m in valid if m["is_vision"]]
            return vis[0] if vis else valid[0]

        if role in ["architect", "arbiter"]:
            # 總架構師與首席仲裁長：選用本機最強大的通用/推理模型 (如 32B)
            non_vis = [m for m in valid if not m["is_vision"]]
            return non_vis[0] if non_vis else valid[0]

        if role == "coder_lead":
            coders = [m for m in valid if m["is_coder"]]
            return coders[0] if coders else valid[0]

        if role == "coder_parallel":
            # 並行開發池：優先挑選中小型模型 (<= 16GB，如 14B/8B/7B)，以便在 40GB 下同時跑 2~4 個模型！
            parallel_candidates = [m for m in valid if not m["is_vision"] and m["est_vram_gb"] <= (budget * 0.55)]
            if not parallel_candidates:
                parallel_candidates = [m for m in valid if not m["is_vision"]]
            # 優先將 coder 排前面，並使用輪詢 (Round-Robin) 分散給不同模型同時執行
            coders_first = sorted(parallel_candidates, key=lambda x: (not x["is_coder"], x["size_gb"]))
            chosen = coders_first[worker_index % len(coders_first)]
            return chosen

        # fast / doc / sec
        light = [m for m in valid if not m["is_vision"] and m["size_gb"] <= 12.0]
        if light:
            return light[worker_index % len(light)]
        return valid[-1]

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

    def acquire_slot(self, worker_id: str, model_info: Dict, action_desc: str, ctx: int):
        """記憶體預算守門員：只要總預留 <= 上限，立即放行多個模型同時並行執行"""
        model_name = model_info["name"]
        est_gb = model_info["est_vram_gb"]

        with self.cond:
            while True:
                budget = self.get_hardware_budget()
                ps = self.sync_realtime_memory()
                active_models = [w["model"] for w in self.active_workers.values()]
                already_allocated = (model_name in active_models) or (model_name in ps["running_models"])
                needed_gb = 2.0 if already_allocated else est_gb

                if len(self.active_workers) == 0:
                    if ps["ps_gb"] + needed_gb > budget:
                        self.unload_idle_models(keep_models=[model_name])
                    break

                if (self.reserved_memory_gb + needed_gb) <= budget:
                    break

                # 清理沒有在 active_workers 裡面的閒置模型
                self.unload_idle_models(keep_models=active_models)
                if (self.reserved_memory_gb + needed_gb) <= budget:
                    break

                self.cond.wait(timeout=1.5)

            self.reserved_memory_gb = round(self.reserved_memory_gb + needed_gb, 2)
            self.active_workers[worker_id] = {
                "model": model_name,
                "action": action_desc,
                "ctx": ctx,
                "tokens": 0,
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
        worker_idx: Optional[int] = None,
        on_token: Optional[Callable[[str, int], None]] = None
    ) -> str:
        with self.cond:
            idx = worker_idx if worker_idx is not None else self.rr_counter
            self.rr_counter += 1

        m_info = self.select_model_for_worker("vision" if images else role, idx)
        model_name = m_info["name"]
        final_ctx = num_ctx if num_ctx else self.decide_dynamic_ctx(prompt)
        worker_id = f"{task_id}-{threading.get_ident()}-{int(time.time()*1000)%10000}"

        self.acquire_slot(worker_id, m_info, f"[{task_id}] {action_desc}", final_ctx)
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True,
                "keep_alive": "2m",
                "options": {"temperature": temperature, "num_ctx": final_ctx}
            }
            if format_json:
                payload["format"] = "json"
            if images:
                payload["images"] = [
                    base64.b64encode(open(p, "rb").read()).decode("utf-8")
                    for p in images if p
                ]

            chunks = []
            cnt = 0
            with httpx.Client(timeout=600.0) as client:
                with client.stream("POST", f"{self.api_url}/api/generate", json=payload) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        tok = data.get("response", "")
                        if tok:
                            chunks.append(tok)
                            cnt += 1
                            if cnt % 8 == 0:
                                if worker_id in self.active_workers:
                                    self.active_workers[worker_id]["tokens"] = cnt
                                    self._save_worker_status()
                                if on_token:
                                    on_token(tok, cnt)
            return "".join(chunks)
        except Exception as e:
            return f"Error: {e}"
        finally:
            self.release_slot(worker_id)