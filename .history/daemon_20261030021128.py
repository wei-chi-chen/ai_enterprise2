import json
import time
import re
from pathlib import Path
from typing import Callable, Optional
from config import TASK_ZONE, DELIVERY_ZONE, HISTORY_ZONE, FEEDBACK_DIR
from core.model_pool import DynamicModelPool
from generators.asset_forge import AutonomousAssetForge
from execution.sandbox_runner import SandboxRunner
import pandas as pd

class OmniEnterpriseDaemon:
    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self.pool = DynamicModelPool()
        self.forge = AutonomousAssetForge()
        self.sandbox = SandboxRunner()
        self.backlog_file = TASK_ZONE / "task_backlog.json"
        self.lessons_file = HISTORY_ZONE / "lessons_learned.json"
        self.broadcast_fn = broadcast_fn
        self.is_running = False

    def emit(self, event_type: str, task_id: str, message: str, extra: dict = None):
        payload = {
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "task_id": task_id,
            "message": message,
            "extra": extra or {}
        }
        if self.broadcast_fn:
            self.broadcast_fn(payload)

    def consume_user_feedback(self, task_id: str) -> str:
        """在每個階段結束後讀取使用者隨時注入的新意見"""
        fb_file = FEEDBACK_DIR / f"{task_id}.json"
        if not fb_file.exists():
            return ""
        try:
            notes = json.loads(fb_file.read_text(encoding="utf-8"))
            if not notes:
                return ""
            fb_file.write_text("[]", encoding="utf-8")
            joined = "\n".join([f"- [{n['time']}] {n['text']}" for n in notes])
            self.emit("feedback_consumed", task_id, f"已讀取並套用使用者新意見：\n{joined}")
            return f"\n【使用者最新追加指導意見（最高優先級）】：\n{joined}\n"
        except Exception:
            return ""

    def get_lessons(self) -> str:
        try:
            data = json.loads(self.lessons_file.read_text(encoding="utf-8"))
            if not data:
                return ""
            return "【歷史自我迭代避坑守則】：\n" + "\n".join([f"- {x['rule']}" for x in data[-6:]])
        except Exception:
            return ""

    def record_lesson(self, task_id: str, rule: str):
        try:
            data = json.loads(self.lessons_file.read_text(encoding="utf-8"))
            data.append({"task_id": task_id, "rule": rule, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            self.lessons_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def process_task(self, task: dict):
        task_id = task["task_id"]
        title = task["title"]
        desc = task["description"]
        task_types = task.get("task_types", ["project_code"])
        attached_file = task.get("attached_file", "")
        iterations = int(task.get("iterations", 3))

        self.emit("stage", task_id, f"🚀 啟動複合任務 [{title}] | 類型: {task_types}")
        accumulated_context = f"初始任務目標：{title}\n詳細需求：{desc}\n{self.get_lessons()}\n"

        # --- 子流程 A：圖片分析 (Image Analysis) ---
        if "image_analysis" in task_types and attached_file and Path(attached_file).exists():
            self.emit("stage", task_id, "👁 [階段: 圖片分析] 調用視覺模型解析影像結構與特徵...")
            vis_res = self.pool.generate(
                role="vision",
                prompt="請詳細分析此圖片的結構、數據、視覺元素與文字細節，輸出結構化分析報告：",
                task_id=task_id,
                action_desc="解析上傳影像",
                images=[attached_file],
                on_progress=lambda chunk, cnt: self.emit("token", task_id, f"視覺模型輸出中 ({cnt} tokens)")
            )
            accumulated_context += f"\n【視覺模型影像分析結果】：\n{vis_res}\n"
            (DELIVERY_ZONE / "reports" / f"{task_id}_vision_report.md").write_text(vis_res, encoding="utf-8")
            self.emit("log", task_id, "✔ 圖片分析完成，報告已匯出並注入主上下文。")

        # 階段檢查點：讀取使用者新意見
        accumulated_context += self.consume_user_feedback(task_id)

        # --- 子流程 B：資料清理 (Data Cleaning) ---
        if "data_cleaning" in task_types:
            self.emit("stage", task_id, "🧹 [階段: 資料清理] 自動分析資料結構並生成清洗與驗證腳本...")
            data_preview = "無外部檔案，請自動生成模擬髒亂資料集並進行清洗"
            if attached_file and Path(attached_file).suffix.lower() in [".csv", ".xlsx", ".json"]:
                try:
                    df = pd.read_csv(attached_file) if attached_file.endswith(".csv") else pd.read_excel(attached_file)
                    data_preview = f"檔案路徑: {attached_file}\n欄位與前5筆:\n{df.head().to_string()}\n缺失值統計:\n{df.isnull().sum().to_string()}"
                except Exception as e:
                    data_preview = f"讀取錯誤: {e}"

            out_csv = DELIVERY_ZONE / "cleaned_data" / f"{task_id}_cleaned.csv"
            clean_prompt = f"""{accumulated_context}
資料概況：
{data_preview}

請撰寫一段獨立可執行的 Python 腳本，使用 pandas 執行：
1. 去除重複值、填補或剔除缺失值、標準化欄位型別與極端離群值。
2. 將清理後的結果儲存至 `{out_csv.as_posix()}`。
3. 附帶 assert 驗證確保輸出檔案存在且無空值。
只輸出純 Python 代碼，不含 Markdown 標籤。"""

            script_path = TASK_ZONE / f"{task_id}_cleaner.py"
            for attempt in range(1, 4):
                ctx_size = self.pool.decide_dynamic_ctx(clean_prompt, attempt)
                raw_code = self.pool.generate(
                    role="coder", prompt=clean_prompt, task_id=task_id,
                    action_desc=f"撰寫資料清理腳本 (嘗試 {attempt})", num_ctx=ctx_size
                )
                clean_code = raw_code.replace("```python", "").replace("```", "").strip()
                script_path.write_text(clean_code, encoding="utf-8")
                res = self.sandbox.run_python_script(script_path)
                if res["success"]:
                    self.emit("log", task_id, f"✔ 資料清理驗證通過！已產出：{out_csv.name}")
                    accumulated_context += f"\n【資料清理成果】：已產出乾淨資料集 `{out_csv.as_posix()}`\n"
                    break
                else:
                    clean_prompt += f"\n上次執行報錯：\n{res['stderr']}\n請修正錯誤。"

        accumulated_context += self.consume_user_feedback(task_id)

        # --- 子流程 C：遊戲製作與多輪視覺化演化 (Game Dev) ---
        if "game_dev" in task_types:
            game_dir = DELIVERY_ZONE / "games" / task_id
            assets_dir = game_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            # 1. 自主判斷並生成遊戲所需素材
            self.emit("stage", task_id, "🎨 [階段: 自主素材生成] 規劃並合成遊戲所需之圖片 Sprite 與 WAV 音效...")
            asset_prompt = f"""{accumulated_context}
請列出此遊戲需要的美術與音效素材清單，輸出嚴格 JSON 格式：
{{
  "assets": [
    {{"type": "sprite", "filename": "hero.png", "style": "hero", "color": "#38bdf8"}},
    {{"type": "sprite", "filename": "enemy.png", "style": "enemy", "color": "#ef4444"}},
    {{"type": "sprite", "filename": "coin.png", "style": "coin", "color": "#facc15"}},
    {{"type": "sfx", "filename": "shoot.wav", "style": "laser"}},
    {{"type": "sfx", "filename": "hit.wav", "style": "explosion"}}
  ]
}}"""
            asset_raw = self.pool.generate(role="fast", prompt=asset_prompt, task_id=task_id, action_desc="開立遊戲素材清單", format_json=True)
            try:
                manifest = json.loads(asset_raw).get("assets", [])
            except Exception:
                manifest = [
                    {"type": "sprite", "filename": "hero.png", "style": "hero", "color": "#38bdf8"},
                    {"type": "sprite", "filename": "enemy.png", "style": "enemy", "color": "#ef4444"},
                    {"type": "sfx", "filename": "shoot.wav", "style": "laser"}
                ]
            created_assets = self.forge.fulfill_asset_manifest(assets_dir, manifest)
            self.emit("game_assets", task_id, f"✔ 已自主合成 {len(created_assets)} 個遊戲素材: {created_assets}", {"assets": created_assets})

            # 2. 多輪自我演化製作遊戲 (即時更新預覽畫面)
            current_html = ""
            for evo in range(1, iterations + 1):
                accumulated_context += self.consume_user_feedback(task_id)
                self.emit("stage", task_id, f"🎮 [遊戲製作演化 Round {evo}/{iterations}] 正在構建與升級遊戲核心邏輯...")

                evo_goal = "建立基礎 Canvas 畫布、玩家鍵盤/滑鼠控制、載入 assets/ 素材與基礎碰撞" if evo == 1 else f"第 {evo} 輪進化：強化敵人波次 AI、粒子特效、分數排行榜、關卡難度曲線與音效觸發，並整合使用者最新意見"
                game_prompt = f"""{accumulated_context}
當前演化目標：{evo_goal}
已生成之本地素材（位於相對路徑 `assets/` 下）：{created_assets}
前次版本代碼：
{current_html[:12000] if current_html else "尚無（首次建立）"}

請輸出單一完整、可直接在瀏覽器運行試玩的 HTML5 Canvas 遊戲檔案 (`index.html`)，包含完整 HTML、CSS 與 JavaScript (`requestAnimationFrame` 主迴圈、鍵盤操作、計分 UI、重新開始按鈕)。
請只輸出純 HTML 代碼，以 `<!DOCTYPE html>` 開頭。"""

                dyn_ctx = self.pool.decide_dynamic_ctx(game_prompt, complexity_tier=evo)
                raw_html = self.pool.generate(
                    role="coder", prompt=game_prompt, task_id=task_id,
                    action_desc=f"編寫遊戲核心 (Round {evo}, Ctx:{dyn_ctx})", num_ctx=dyn_ctx,
                    on_progress=lambda chunk, cnt: self.emit("token", task_id, f"遊戲代碼演化中 ({cnt} tokens)")
                )
                clean_html = raw_html.replace("```html", "").replace("```", "").strip()
                if "<!DOCTYPE html>" in clean_html:
                    clean_html = clean_html[clean_html.find("<!DOCTYPE html>"):]

                html_file = game_dir / "index.html"
                html_file.write_text(clean_html, encoding="utf-8")

                # 自主驗證與除錯
                val_res = self.sandbox.validate_html_game(html_file)
                if not val_res["success"]:
                    self.emit("log", task_id, f"⚠ 遊戲靜態檢測發現問題: {val_res['stderr']}，啟動自動修復...")
                else:
                    current_html = clean_html
                    self.emit("game_preview", task_id, f"✔ 遊戲演化 Round {evo} 完成！右側預覽視窗已即時刷新。", {
                        "preview_url": f"/delivery/games/{task_id}/index.html?t={int(time.time())}",
                        "round": evo
                    })

        accumulated_context += self.consume_user_feedback(task_id)

        # --- 子流程 D：專案級程式碼開發與三級自動測試修復 (Project Code) ---
        if "project_code" in task_types:
            self.emit("stage", task_id, "💻 [階段: 專案級程式碼] 架構師規劃多模組結構與 TDD 驗收測試...")
            proj_dir = DELIVERY_ZONE / "projects" / task_id
            proj_dir.mkdir(parents=True, exist_ok=True)

            arch_prompt = f"""{accumulated_context}
請規劃一個高內聚低耦合的 Python 專案架構，包含主程式模組與 `test_suite.py` 自動化測試。
輸出嚴格 JSON 格式：
{{
  "suggested_num_ctx": 16384,
  "files": [
    {{"filename": "core_engine.py", "description": "核心類別與演算法實作"}},
    {{"filename": "main_app.py", "description": "CLI 入口與業務調用邏輯"}}
  ],
  "test_instruction": "撰寫 test_suite.py 嚴格驗證 core_engine.py 所有功能與邊界條件"
}}"""
            arch_raw = self.pool.generate(role="architect", prompt=arch_prompt, task_id=task_id, action_desc="制定專案架構合約", format_json=True)
            try:
                plan = json.loads(arch_raw)
            except Exception:
                plan = {
                    "suggested_num_ctx": 8192,
                    "files": [{"filename": "core_engine.py", "description": desc}],
                    "test_instruction": "驗證核心功能"
                }

            decided_ctx = min(max(int(plan.get("suggested_num_ctx", 8192)), 4096), 65536)
            generated_code_map = {}

            for f_spec in plan.get("files", []):
                accumulated_context += self.consume_user_feedback(task_id)
                fname = f_spec["filename"]
                f_prompt = f"""{accumulated_context}
已生成之專案模組：{json.dumps(generated_code_map, ensure_ascii=False)[:6000]}
請實作檔案 `{fname}`：{f_spec['description']}
只輸出純 Python 原始碼，不含 Markdown。"""
                code_str = self.pool.generate(role="coder", prompt=f_prompt, task_id=task_id, action_desc=f"編寫模組 {fname}", num_ctx=decided_ctx)
                clean_c = code_str.replace("```python", "").replace("```", "").strip()
                (proj_dir / fname).write_text(clean_c, encoding="utf-8")
                generated_code_map[fname] = clean_c
                self.emit("log", task_id, f"✔ 已生成專案檔案：{fname}")

            # 生成 TDD 測試並執行三級階梯除錯
            test_prompt = f"""請針對以下專案代碼撰寫獨立驗收測試 `test_suite.py`：
{json.dumps(generated_code_map, ensure_ascii=False)}
要求：開頭加入 `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))`，使用 `assert` 驗證核心邏輯。只輸出純 Python 原始碼。"""
            test_raw = self.pool.generate(role="coder", prompt=test_prompt, task_id=task_id, action_desc="生成單元測試 test_suite.py", num_ctx=decided_ctx)
            test_code = test_raw.replace("```python", "").replace("```", "").strip()
            test_path = proj_dir / "test_suite.py"
            test_path.write_text(test_code, encoding="utf-8")

            for tier in range(1, 4):
                run_res = self.sandbox.run_python_script(test_path)
                if run_res["success"]:
                    self.emit("log", task_id, f"✔ 專案通過沙盒 TDD 自動驗收測試 (Tier {tier})！")
                    self.record_lesson(task_id, f"任務 {title} 通過 TDD 驗證，模組介面保持一致。")
                    break
                else:
                    err_info = run_res["stderr"]
                    self.emit("log", task_id, f"⚠ 測試未通過 (Tier {tier} 自主糾錯中): {err_info[:180]}")
                    fix_role = "debugger" if tier == 1 else "architect"
                    fix_ctx = min(decided_ctx * tier, 65536)
                    fix_prompt = f"""專案測試執行失敗。
當前所有檔案：{json.dumps(generated_code_map, ensure_ascii=False)}
測試腳本：{test_code}
錯誤堆疊：{err_info}
請修復錯誤，使用以下標記格式輸出需要覆寫的檔案：
<<<FILE: 檔名.py>>>
完整Python代碼
<<<END_FILE>>>"""
                    fix_raw = self.pool.generate(role=fix_role, prompt=fix_prompt, task_id=task_id, action_desc=f"Tier {tier} 階梯修復", num_ctx=fix_ctx)
                    blocks = re.findall(r"<<<FILE:\s*(.*?)>>>(.*?)<<<END_FILE>>>", fix_raw, re.DOTALL)
                    for target_fn, new_code in blocks:
                        target_fn = target_fn.strip()
                        c_clean = new_code.replace("```python", "").replace("```", "").strip()
                        if self.sandbox.is_valid_python(c_clean):
                            (proj_dir / target_fn).write_text(c_clean, encoding="utf-8")
                            if target_fn != "test_suite.py":
                                generated_code_map[target_fn] = c_clean

        self.emit("completed", task_id, f"🎉 任務 [{task_id}] 全部流水線與自我迭代執行完畢！")