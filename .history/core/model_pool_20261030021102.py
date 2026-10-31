import time
import base64
import threading
import httpx
from typing import List, Dict, Optional, Callable
from config import OLLAMA_API_URL, MAX_MEMORY_GB, EMPLOYEE_ZONE, TASK_ZONE
import json

class DynamicModelPool:
    def __init__(self):
        self.api_url = OLLAMA_API_URL.rstrip("/")
        self.mem_lock = threading.Lock()
        self.available_models: List[Dict] = []
        self.status_file = EMPLOYEE_ZONE / "live_status.json"
        self.refresh_models()

    def refresh_models(self) -> List[Dict]:
        """即時刷新本機所有已下載之 Ollama 模型與容量"""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.api_url}/api/tags")
                if res.status_code == 200:
                    raw = res.json().get("models", [])
                    models = []
                    for m in raw:
                        size_gb = round(m.get("size", 0) / (1024 ** 3), 2)
                        est_vram = round(size_gb * 1.2, 2)
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

    def get_running_memory_gb(self) -> Dict:
        """查詢 Ollama /api/ps 獲取當前駐留記憶體的模型與總 GB 數"""
        used_gb = 0.0
        running_names = []
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.api_url}/api/ps")
                if res.status_code == 200:
                    for m in res.json().get("models", []):
                        sz = m.get("size_vram", 0) or m.get("size", 0)
                        used_gb += sz / (1024 ** 3)
                        running_names.append(m.get("name"))
        except Exception:
            pass
        return {"used_gb": round(used_gb, 2), "running_models": running_names}

    def unload_model(self, model_name: str):
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"{self.api_url}/api/generate", json={"model": model_name, "keep_alive": 0})
        except Exception:
            pass

    def ensure_memory_budget(self, required_gb: float):
        """確保載入新模型後不超過 40GB 硬體預算"""
        with self.mem_lock:
            ps_info = self.get_running_memory_gb()
            if ps_info["used_gb"] + required_gb > MAX_MEMORY_GB:
                for m_name in ps_info["running_models"]:
                    self.unload_model(m_name)
                time.sleep(0.5)

    def select_best_model(self, role: str) -> str:
        """依任務需求自動從現有模型池挑選最適合模型 (architect / coder / vision / fast)"""
        self.refresh_models()
        valid = [m for m in self.available_models if m["est_vram_gb"] <= MAX_MEMORY_GB]
        if not valid:
            return "qwen2.5:7b"

        if role == "vision":
            vis = [m for m in valid if m["is_vision"]]
            return vis[0]["name"] if vis else valid[0]["name"]
        elif role == "coder":
            coders = [m for m in valid if m["is_coder"]]
            return coders[0]["name"] if coders else valid[0]["name"]
        elif role == "debugger":
            coders = [m for m in valid if m["is_coder"] and m["size_gb"] <= 15.0]
            return coders[0]["name"] if coders else valid[-1]["name"]
        elif role == "fast":
            return valid[-1]["name"]
        else:  # architect / general
            non_vis = [m for m in valid if not m["is_vision"]]
            return non_vis[0]["name"] if non_vis else valid[0]["name"]

    def decide_dynamic_ctx(self, prompt_text: str, complexity_tier: int = 1) -> int:
        """根據輸入長度與任務複雜度自主判斷並變動 Context Length"""
        char_len = len(prompt_text)
        base_ctx = 4096
        if char_len > 12000 or complexity_tier >= 3:
            base_ctx = 32768
        elif char_len > 4000 or complexity_tier == 2:
            base_ctx = 16384
        elif char_len > 1500:
            base_ctx = 8192
        return base_ctx

    def update_live_status(self, task_id: str, role: str, model: str, action: str, ctx: int, tokens: int):
        ps = self.get_running_memory_gb()
        state = {
            "timestamp": time.strftime("%H:%M:%S"),
            "task_id": task_id,
            "role": role,
            "model": model,
            "action": action,
            "num_ctx": ctx,
            "tokens_generated": tokens,
            "used_memory_gb": ps["used_gb"],
            "running_models": ps["running_models"]
        }
        try:
            self.status_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def generate(
        self,
        role: str,
        prompt: str,
        system_prompt: str = "",
        task_id: str = "SYS",
        action_desc: str = "思考中",
        images: Optional[List[str]] = None,
        num_ctx: Optional[int] = None,
        format_json: bool = False,
        temperature: float = 0.2,
        on_progress: Optional[Callable[[str, int], None]] = None
    ) -> str:
        model_name = self.select_best_model("vision" if images else role)
        model_info = next((m for m in self.available_models if m["name"] == model_name), {"est_vram_gb": 8.0})
        self.ensure_memory_budget(model_info["est_vram_gb"])

        final_ctx = num_ctx if num_ctx else self.decide_dynamic_ctx(prompt)
        self.update_live_status(task_id, role, model_name, action_desc, final_ctx, 0)

        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "keep_alive": "2m",
            "options": {
                "temperature": temperature,
                "num_ctx": final_ctx
            }
        }
        if format_json:
            payload["format"] = "json"

        if images:
            encoded_imgs = []
            for img_p in images:
                with open(img_p, "rb") as f:
                    encoded_imgs.append(base64.b64encode(f.read()).decode("utf-8"))
            payload["images"] = encoded_imgs

        full_resp = []
        token_cnt = 0
        try:
            with httpx.Client(timeout=600.0) as client:
                with client.stream("POST", f"{self.api_url}/api/generate", json=payload) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token_str = chunk.get("response", "")
                        if token_str:
                            full_resp.append(token_str)
                            token_cnt += 1
                            if token_cnt % 10 == 0:
                                self.update_live_status(task_id, role, model_name, action_desc, final_ctx, token_cnt)
                                if on_progress:
                                    on_progress("".join(full_resp[-20:]), token_cnt)
        except Exception as e:
            return f"Error: {e}"

        self.update_live_status(task_id, role, model_name, f"完成: {action_desc}", final_ctx, token_cnt)
        return "".join(full_resp)