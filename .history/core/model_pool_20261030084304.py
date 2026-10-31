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
        self.usage_counter: Dict[str, int] = {}
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
        """即時掃描本機所有已下載之 Ollama 模型（排除純 embedding 模型外全部啟用）"""
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.api_url}/api/tags")
                if res.status_code == 200:
                    raw = res.json().get("models", [])
                    models = []
                    for m in raw:
                        name = m.get("name", "")
                        if "embed" in name.lower() or "nomic" in name.lower():
                            continue
                        size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
                        est_vram = round(max(1.5, size_gb * 1.12), 2)
                        models.append({
                            "name": name,
                            "size_gb": size_gb,
                            "est_vram_gb": est_vram,
                            "is_vision": any(k in name.lower() for k in ["llava", "vision", "minicpm", "bakllava", "vl", "moondream"]),
                            "is_coder": any(k in name.lower() for k in ["coder", "code", "starcoder", "codellama", "deepseek", "qwen"])
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

    def unload_all_models(self):
        ps = self.sync_realtime_memory()
        for m_name in ps["running_models"]:
            try:
                with httpx.Client(timeout=3.0) as client:
                    client.post(f"{self.api_url}/api/generate", json={"model": m_name, "keep_alive": 0})
            except Exception:
                pass

    def plan_greedy_swarm(self) -> List[Dict]:
        """
        【核心演算法：貪婪填滿記憶體集群規劃】
        從電腦現有的所有模型中，挑選能同時塞進記憶體上限 (例如 40GB) 的最大模型數量組合，
        確保不同模型全部都被調動，並把可用記憶體盡可能用滿！
        """
        self.refresh_models()
        budget = self.get_hardware_budget()
        text_models = [m for m in self.available_models if not m["is_vision"] and m["est_vram_gb"] <= budget]
        if not text_models:
            text_models = self.available_models

        if not text_models:
            return [{"name": "qwen2.5:7b", "size_gb": 5.0, "est_vram_gb": 5.5}]

        # 依「最少被呼叫次數」與「中大型優先」排序，確保所有下載的模型都能輪流上場並塞滿 40GB
        sorted_candidates = sorted(
            text_models,
            key=lambda m: (self.usage_counter.get(m["name"], 0), abs(m["est_vram_gb"] - (budget * 0.3)))
        )

        swarm = []
        accum_gb = 0.0

        # 第一輪：盡可能挑選「不同名稱」的模型塞入記憶體
        for m in sorted_candidates:
            if accum_gb + m["est_vram_gb"] <= budget:
                swarm.append(m)
                accum_gb += m["est_vram_gb"]

        # 第二輪：如果還有剩餘超過 2.0 GB 的記憶體空間，允許已載入的模型開啟額外並行執行緒 (僅需 ~1.8GB KV Cache)
        while (budget - accum_gb) >= 2.0 and len(swarm) < 8:
            for m in sorted_candidates:
                if (budget - accum_gb) >= 2.0 and len(swarm) < 8:
                    swarm.append(m)
                    accum_gb += 1.8
                else:
                    break

        return swarm if swarm else [text_models[0]]

    def select_model_for_role(self, role: str, preferred_model: Optional[str] = None) -> Dict:
        self.refresh_models()
        budget = self.get_hardware_budget()
        valid = [m for m in self.available_models if m["est_vram_gb"] <= budget]
        if not valid:
            return {"name": "qwen2.5:7b", "size_gb": 5.0, "est_vram_gb": 5.5}

        if preferred_model:
            found = next((m for m in valid if m["name"] == preferred_model), None)
            if found:
                return found

        if role == "vision":
            vis = [m for m in valid if m["is_vision"]]
            return vis[0] if vis else valid[0]

        if role in ["architect", "arbiter"]:
            non_vis = [m for m in valid if not m["is_vision"]]
            return non_vis[0] if non_vis else valid[0]

        # 預設挑選最少使用過的模型，確保所有模型皆能被調用
        non_vis = [m for m in valid if not m["is_vision"]] or valid
        non_vis.sort(key=lambda m: self.usage_counter.get(m["name"], 0))
        return non_vis[0]

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
        """只要預留記憶體 <= 預算上限，無上限放行多模型同時並行"""
        model_name = model_info["name"]
        est_gb = model_info["est_vram_gb"]

        with self.cond:
            while True:
                budget = self.get_hardware_budget()
                ps = self.sync_realtime_memory()
                active_models = [w["model"] for w in self.active_workers.values()]
                already_loaded = (model_name in active_models) or (model_name in ps["running_models"])
                needed_gb = 1.8 if already_loaded else est_gb

                if len(self.active_workers) == 0 and needed_gb <= budget:
                    break

                if (self.reserved_memory_gb + needed_gb) <= budget:
                    break

                # 若空間不足，卸載目前不在 active_workers 中的閒置模型
                for m_in_vram in ps["running_models"]:
                    if m_in_vram not in active_models:
                        try:
                            with httpx.Client(timeout=2.0) as client:
                                client.post(f"{self.api_url}/api/generate", json={"model": m_in_vram, "keep_alive": 0})
                        except Exception:
                            pass

                if (self.reserved_memory_gb + needed_gb) <= budget:
                    break

                self.cond.wait(timeout=1.0)

            self.reserved_memory_gb = round(self.reserved_memory_gb + needed_gb, 2)
            self.usage_counter[model_name] = self.usage_counter.get(model_name, 0) + 1
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
        preferred_model: Optional[str] = None,
        on_token: Optional[Callable[[str, int], None]] = None
    ) -> str:
        m_info = self.select_model_for_role("vision" if images else role, preferred_model=preferred_model)
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
                "keep_alive": "15m",  # 常駐 15 分鐘以保持記憶體滿載與零冷啟動延遲
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