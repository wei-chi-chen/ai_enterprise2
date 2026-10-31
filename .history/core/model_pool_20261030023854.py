import time
import json
import base64
import threading
import httpx
from typing import List, Dict, Optional, Callable
from config import OLLAMA_API_URL, ROSTER_FILE, CONTROL_FILE, MEM_USAGE_FILE, STATUS_FILE, DEFAULT_MAX_MEMORY_GB

class DynamicModelPool:
    def __init__(self):
        self.api_url = OLLAMA_API_URL.rstrip("/")
        self.cond = threading.Condition()
        self.available_models: List[Dict] = []
        self.reserved_memory_gb: float = 0.0
        self.active_workers: Dict[str, Dict] = {}
        self.refresh_models()

    def get_hardware_budget(self) -> float:
        """即時讀取使用者設定之記憶體上限 (預設 40.0 GB)"""
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
        """即時掃描本機已下載的所有 Ollama 模型"""
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
                            "is_coder": any(k in name.lower() for k in ["coder", "code", "starcoder", "codellama", "deepseek-coder"])
                        })
                    models.sort(key=lambda x: x["size_gb"], reverse=True)
                    self.available_models = models
        except Exception:
            pass
        return self.available_models

    def sync_realtime_memory(self) -> Dict:
        """同步 Ollama /api/ps 實時記憶體佔用"""
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

    def select_models_by_tier(self, role: str) -> str:
        """依據當前記憶體上限與任務職能，動態挑選最適合的模型"""
        self.refresh_models()
        budget = self.get_hardware_budget()
        valid = [m for m in self.available_models if m["est_vram_gb"] <= budget]
        if not valid:
            return self.available_models[-1]["name"] if self.available_models else "qwen2.5:7b"

        if role == "vision":
            vis = [m for m in valid if m["is_vision"]]
            return vis[0]["name"] if vis else valid[0]["name"]
        elif role == "architect" or role == "arbiter":
            non_vis = [m for m in valid if not m["is_vision"]]
            return non_vis[0]["name"] if non_vis else valid[0]["name"]
        elif role == "coder_lead":
            coders = [m for m in valid if m["is_coder"]]
            return coders[0]["name"] if coders else valid[0]["name"]
        elif role == "coder_worker":
            # 優先選 14B 或 7B 等級的 Coder 以利於在 40GB 內多開並行
            mid_coders = [m for m in valid if m["is_coder"] and m["est_vram_gb"] <= (budget * 0.45)]
            if mid_coders:
                return mid_coders[0]["name"]
            coders = [m for m in valid if m["is_coder"]]
            return coders[-1]["name"] if coders else valid[-1]["name"]
        else:  # fast / router / doc / sec
            light = [m for m in valid if not m["is_vision"] and m["size_gb"] <= 10.0]
            return light[0]["name"] if light else valid[-1]["name"]

    def decide_dynamic_ctx(self, prompt_text: str, complexity_tier: int = 1) -> int:
        """根據提示詞長度與階段複雜度自主決定 Context Length"""
        length = len(prompt_text)
        if length > 16000 or complexity_tier >= 3:
            return 32768
        elif length > 6000 or complexity_tier == 2:
            return 16384
        elif length > 2000:
            return 8192
        return 4096

    def _save_worker_status(self):
        try:
            STATUS_FILE.write_text(json.dumps(self.active_workers, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def acquire_slot(self, worker_id: str, model_name: str, est_gb: float, action_desc: str, ctx: int):
        """只要不超過記憶體上限，允許多個執行緒同時取得槽位並行呼叫模型"""
        with self.cond:
            while True:
                budget = self.get_hardware_budget()
                ps = self.sync_realtime_memory()
                already_loaded = model_name in ps["running_models"]
                # 若同模型已在顯存中，並行呼叫僅需約 1.5GB KV Cache 增量
                needed_gb = 1.5 if already_loaded else est_gb

                # 若當前沒有任何活動工作執行緒，但預估單模型略大，放行單一模型執行
                if len(self.active_workers) == 0 and needed_gb <= budget:
                    if ps["ps_gb"] + needed_gb > budget:
                        self.unload_idle_models(keep_models=[model_name])
                    break

                # 若加總不超過預算，立即准入並行！
                current_committed = max(ps["ps_gb"], self.reserved_memory_gb)
                if (current_committed + needed_gb) <= budget:
                    break

                # 嘗試卸載未在使用的閒置模型以騰出空間
                active_model_names = [w["model"] for w in self.active_workers.values()]
                self.unload_idle_models(keep_models=active_model_names)
                ps_after = self.sync_realtime_memory()
                if (max(ps_after["ps_gb"], self.reserved_memory_gb) + needed_gb) <= budget:
                    break

                self.cond.wait(timeout=2.0)

            self.reserved_memory_gb += needed_gb
            self.active_workers[worker_id] = {
                "model": model_name,
                "action": action_desc,
                "ctx": ctx,
                "tokens": 0,
                "reserved_gb": needed_gb
            }
            self._save_worker_status()
            self.sync_realtime_memory()

    def release_slot(self, worker_id: str):
        with self.cond:
            info = self.active_workers.pop(worker_id, None)
            if info:
                self.reserved_memory_gb = max(0.0, self.reserved_memory_gb - info["reserved_gb"])
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
        on_token: Optional[Callable[[str, int], None]] = None
    ) -> str:
        model_name = self.select_models_by_tier("vision" if images else role)
        m_info = next((m for m in self.available_models if m["name"] == model_name), {"est_vram_gb": 6.5})
        final_ctx = num_ctx if num_ctx else self.decide_dynamic_ctx(prompt)
        worker_id = f"{task_id}-{threading.get_ident()}"

        self.acquire_slot(worker_id, model_name, m_info["est_vram_gb"], f"[{task_id}] {action_desc}", final_ctx)
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True,
                "keep_alive": "3m",
                "options": {"temperature": temperature, "num_ctx": final_ctx}
            }
            if format_json:
                payload["format"] = "json"
            if images:
                payload["images"] = [
                    base64.b64encode( open(p, "rb").read() ).decode("utf-8")
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