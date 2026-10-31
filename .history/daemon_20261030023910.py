import json
import time
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Dict, List

from config import TASK_ZONE, DELIVERY_ZONE, HISTORY_ZONE, FEEDBACK_DIR, CONTROL_FILE, BACKLOG_FILE, LESSONS_FILE
from core.model_pool import DynamicModelPool
from generators.asset_forge import AutonomousAssetForge
from execution.sandbox_runner import SandboxRunner
import pandas as pd

class TaskTerminatedException(Exception):
    pass

class OmniEnterpriseDaemon:
    def __init__(self, broadcast_fn: Optional[Callable] = None):
        self.pool = DynamicModelPool()
        self.forge = AutonomousAssetForge()
        self.sandbox = SandboxRunner()
        self.broadcast_fn = broadcast_fn
        self.backlog_lock = threading.Lock()

        # 計時與進度追蹤器
        self.daemon_active = False
        self.total_elapsed_sec = 0.0
        self.current_task_elapsed_sec = 0.0
        self.current_task_id = "-"
        self.current_task_stage = "Idle"
        self.current_task_progress_pct = 0
        self._timer_thread = threading.Thread(target=self._background_ticker, daemon=True)
        self._timer_thread.start()

    def _background_ticker(self):
        """每秒更新總耗時、當前任務耗時、記憶體與佇列總進度並推播至前端"""
        while True:
            time.sleep(1.0)
            ctrl = self.get_control_state()
            state = ctrl.get("state", "IDLE")
            if state == "RUNNING":
                self.total_elapsed_sec += 1.0
                if self.current_task_id != "-":
                    self.current_task_elapsed_sec += 1.0

            q_stats = self.get_queue_progress()
            mem_info = self.pool.sync_realtime_memory()
            budget = self.pool.get_hardware_budget()

            if self.broadcast_fn:
                self.broadcast_fn({
                    "type": "telemetry",
                    "state": state,
                    "total_elapsed_sec": int(self.total_elapsed_sec),
                    "task_elapsed_sec": int(self.current_task_elapsed_sec),
                    "current_task_id": self.current_task_id,
                    "current_stage": self.current_task_stage,
                    "task_progress_pct": self.current_task_progress_pct,
                    "queue_completed": q_stats["completed"],
                    "queue_total": q_stats["total"],
                    "queue_pct": q_stats["pct"],
                    "used_memory_gb": mem_info["used_gb"],
                    "budget_gb": budget,
                    "running_models": mem_info["running_models"],
                    "active_workers": list(self.pool.active_workers.values())
                })

    def get_control_state(self) -> Dict:
        if CONTROL_FILE.exists():
            try:
                return json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"state": "IDLE", "pause_requested": False, "stop_requested": False}

    def set_control_state(self, state: str, pause_req: bool = False, stop_req: bool = False):
        CONTROL_FILE.write_text(json.dumps({
            "state": state,
            "pause_requested": pause_req,
            "stop_requested": stop_req
        }, indent=2), encoding="utf-8")

    def get_queue_progress(self) -> Dict:
        with self.backlog_lock:
            if not BACKLOG_FILE.exists():
                return {"completed": 0, "total": 0, "pct": 0}
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                total = len(tasks)
                if total == 0:
                    return {"completed": 0, "total": 0, "pct": 0}
                comp = sum(1 for t in tasks if t.get("status") in ["completed", "failed"])
                return {"completed": comp, "total": total, "pct": int((comp / total) * 100)}
            except Exception:
                return {"completed": 0, "total": 0, "pct": 0}

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
        """原子檢查點：處理『終止』、『暫停與原地繼續』、以及讀取『使用者新意見』"""
        if stage_name:
            self.current_task_stage = stage_name
        if progress_pct is not None:
            self.current_task_progress_pct = progress_pct

        ctrl = self.get_control_state()
        if ctrl.get("stop_requested"):
            raise TaskTerminatedException(f"Task {task_id} terminated by user.")

        if ctrl.get("pause_requested"):
            self.set_control_state("PAUSED", pause_req=True, stop_req=False)
            self.update_task_status(task_id, "paused")
            self.emit_log(task_id, "⏸ 系統已進入暫停狀態（可隨時修改記憶體上限或注入意見，按『繼續』即刻接續執行）...")
            while True:
                time.sleep(0.5)
                c = self.get_control_state()
                if c.get("stop_requested"):
                    raise TaskTerminatedException(f"Task {task_id} terminated while paused.")
                if not c.get("pause_requested"):
                    self.set_control_state("RUNNING", pause_req=False, stop_req=False)
                    self.update_task_status(task_id, "processing")
                    new_budget = self.pool.get_hardware_budget()
                    self.emit_log(task_id, f"⏯ 系統已恢復運行！當前套用記憶體上限: {new_budget} GB")
                    break

        return self.consume_user_feedback(task_id)

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
            self.emit_log(task_id, f"💡 已讀取並整合使用者最新意見：\n{joined}")
            return f"\n【使用者最新追加指導意見（最高優先級）】：\n{joined}\n"
        except Exception:
            return ""

    def get_lessons(self) -> str:
        try:
            data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
            if not data:
                return ""
            return "【歷史避坑經驗】：\n" + "\n".join([f"- {x['rule']}" for x in data[-5:]])
        except Exception:
            return ""

    def record_lesson(self, task_id: str, rule: str):
        try:
            data = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
            data.append({"task_id": task_id, "rule": rule, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            LESSONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def update_task_status(self, task_id: str, status: str):
        with self.backlog_lock:
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                for t in tasks:
                    if t.get("task_id") == task_id:
                        t["status"] = status
                BACKLOG_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def peek_pending_tasks(self) -> List[Dict]:
        with self.backlog_lock:
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                return [t for t in tasks if t.get("status") in ["pending", "paused"]]
            except Exception:
                return []

    # ================= 全能任務執行主流程 =================
    def execute_omnipotent_task(self, task: dict):
        task_id = task["task_id"]
        title = task["title"]
        desc = task["description"]
        attached_file = task.get("attached_file", "")
        iterations = int(task.get("iterations", 3))

        self.current_task_id = task_id
        self.current_task_elapsed_sec = 0.0
        self.update_task_status(task_id, "processing")

        try:
            fb = self.checkpoint(task_id, "Stage 1: 全能意圖自主路由與規劃", 10)
            accumulated_ctx = f"任務標題：{title}\n使用者需求：{desc}\n附件檔案：{attached_file}\n{self.get_lessons()}\n{fb}"

            # 1. 模型自主判斷任務性質與所需流水線 (免使用者手動分類)
            self.emit_log(task_id, "🧠 [階段 1] 總架構師正在自主分析需求並決定執行管線與 Context 視窗...")
            router_prompt = f"""請分析以下使用者需求與附件，自主決定需要啟動哪些功能管線（可複選組合）：
{accumulated_ctx}

輸出嚴格 JSON 格式：
{{
  "suggested_num_ctx": 16384,
  "needs_vision_analysis": false,
  "needs_data_cleaning": false,
  "needs_game_development": false,
  "needs_software_project": true,
  "needs_direct_qa_report": true,
  "reasoning": "簡述為何如此規劃管線"
}}
判斷準則：
- 若需求涉及開發遊戲、互動畫布、彈幕、RPG等，設 `needs_game_development: true`。
- 若上傳了圖片或需解析視覺圖表，設 `needs_vision_analysis: true`。
- 若上傳了 CSV/Excel 或要求資料清洗/統計模擬處理，設 `needs_data_cleaning: true`。
- 若要求寫程式、演算法、工具、系統架構，設 `needs_software_project: true`。
- 若為學術問題、理論探討或需產出分析總結報告，設 `needs_direct_qa_report: true`。"""

            route_raw = self.pool.generate(
                role="architect", prompt=router_prompt, task_id=task_id,
                action_desc="自主判斷任務管線與 Context", format_json=True
            )
            try:
                plan_flags = json.loads(route_raw)
            except Exception:
                plan_flags = {
                    "suggested_num_ctx": 8192,
                    "needs_vision_analysis": bool(attached_file and Path(attached_file).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]),
                    "needs_data_cleaning": bool(attached_file and Path(attached_file).suffix.lower() in [".csv", ".xlsx"]),
                    "needs_game_development": any(w in (title + desc).lower() for w in ["遊戲", "game", "rpg", "彈幕", "canvas"]),
                    "needs_software_project": True,
                    "needs_direct_qa_report": True,
                    "reasoning": "預設全能複合管線"
                }

            decided_ctx = min(max(int(plan_flags.get("suggested_num_ctx", 8192)), 4096), 65536)
            self.emit_log(task_id, f"✔ 自主規劃完成 (動態 Ctx: {decided_ctx})：{plan_flags.get('reasoning', '')}")

            # 2. 視覺影像分析管線
            if plan_flags.get("needs_vision_analysis") and attached_file and Path(attached_file).exists():
                fb = self.checkpoint(task_id, "Stage 2: 多模態視覺感知解析", 25)
                accumulated_ctx += fb
                self.emit_log(task_id, "👁 [視覺管線] 啟動視覺專員模型解析影像...")
                vis_res = self.pool.generate(
                    role="vision",
                    prompt=f"請根據需求（{desc}）深度解析此圖片之數據、結構與細節：",
                    task_id=task_id, action_desc="視覺多模態解析", images=[attached_file], num_ctx=decided_ctx
                )
                accumulated_ctx += f"\n【視覺解析報告】：\n{vis_res}\n"
                (DELIVERY_ZONE / "reports" / f"{task_id}_vision_analysis.md").write_text(vis_res, encoding="utf-8")

            # 3. 資料清理與驗證管線
            if plan_flags.get("needs_data_cleaning"):
                fb = self.checkpoint(task_id, "Stage 3: 自動資料清理與沙盒驗證", 40)
                accumulated_ctx += fb
                self.emit_log(task_id, "🧹 [資料管線] 正在編寫自動化資料清理與統計驗證腳本...")
                out_csv = DELIVERY_ZONE / "cleaned_data" / f"{task_id}_cleaned.csv"
                data_info = f"目標檔案：{attached_file}" if attached_file else "請生成符合需求的模擬資料集並執行清理與統計檢驗"
                clean_prompt = f"""{accumulated_ctx}\n{data_info}\n請撰寫純 Python 腳本處理資料並輸出至 `{out_csv.as_posix()}`，末端包含 assert 驗證。只輸出純 Python 代碼。"""
                script_p = TASK_ZONE / f"{task_id}_data_job.py"
                for attempt in range(1, 4):
                    self.checkpoint(task_id)
                    c_raw = self.pool.generate(role="coder_worker", prompt=clean_prompt, task_id=task_id, action_desc=f"資料處理實現 (Try {attempt})", num_ctx=decided_ctx)
                    script_p.write_text(c_raw.replace("```python", "").replace("```", "").strip(), encoding="utf-8")
                    res = self.sandbox.run_python_script(script_p)
                    if res["success"]:
                        self.emit_log(task_id, f"✔ 資料清理與驗證通過！產出：{out_csv.name}")
                        break
                    clean_prompt += f"\n報錯修復：\n{res['stderr']}"

            # 4. 遊戲製作與自主素材合成管線
            if plan_flags.get("needs_game_development"):
                fb = self.checkpoint(task_id, "Stage 4: 遊戲自主素材合成與多輪演化", 55)
                accumulated_ctx += fb
                game_dir = DELIVERY_ZONE / "games" / task_id
                assets_dir = game_dir / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)

                self.emit_log(task_id, "🎨 [遊戲管線] 正在自主開立素材清單並合成 PNG 精靈圖與 WAV 音效...")
                asset_raw = self.pool.generate(
                    role="fast",
                    prompt=f"{accumulated_ctx}\n請以 JSON 輸出遊戲所需的圖片與音效清單：{{\"assets\": [{{\"type\":\"sprite\",\"filename\":\"hero.png\",\"style\":\"hero\",\"color\":\"#38bdf8\"}}, {{\"type\":\"sfx\",\"filename\":\"shoot.wav\",\"style\":\"laser\"}}]}}",
                    task_id=task_id, action_desc="生成遊戲素材規格", format_json=True
                )
                try:
                    manifest = json.loads(asset_raw).get("assets", [])
                except Exception:
                    manifest = [{"type": "sprite", "filename": "hero.png", "style": "hero", "color": "#38bdf8"}, {"type": "sfx", "filename": "shoot.wav", "style": "laser"}]

                created_assets = self.forge.fulfill_asset_manifest(assets_dir, manifest)
                self.emit_log(task_id, f"✔ 已自主生成素材：{created_assets}", event_type="game_assets", extra={"assets": created_assets})

                current_html = ""
                for evo in range(1, iterations + 1):
                    fb = self.checkpoint(task_id, f"Stage 4.{evo}: 遊戲演化迭代 ({evo}/{iterations})", 55 + int((evo / iterations) * 20))
                    accumulated_ctx += fb
                    evo_prompt = f"""{accumulated_ctx}
目前為第 {evo}/{iterations} 輪演化。可用本地素材 (`assets/` 目錄下)：{created_assets}
前次版本代碼：{current_html[:14000] if current_html else '無'}
請輸出完整單檔 HTML5 Canvas 遊戲 (`index.html`)，具備完整遊戲迴圈、鍵盤滑鼠控制、碰撞、音效播放與計分板。只輸出純 HTML。"""
                    raw_html = self.pool.generate(
                        role="coder_lead", prompt=evo_prompt, task_id=task_id,
                        action_desc=f"遊戲演化 Round {evo}/{iterations}", num_ctx=decided_ctx
                    )
                    clean_html = raw_html.replace("```html", "").replace("```", "").strip()
                    if "<!DOCTYPE html>" in clean_html:
                        clean_html = clean_html[clean_html.find("<!DOCTYPE html>"):]
                    html_path = game_dir / "index.html"
                    html_path.write_text(clean_html, encoding="utf-8")
                    current_html = clean_html
                    self.emit_log(
                        task_id, f"🎮 遊戲第 {evo} 輪演化完成！右側試玩畫面已同步更新。",
                        event_type="game_preview",
                        extra={"preview_url": f"/delivery/games/{task_id}/index.html?t={int(time.time())}"}
                    )

            # 5. 多模組專案開發（預算內最大化多模型並行 + 三級階梯 TDD 除錯）
            if plan_flags.get("needs_software_project"):
                fb = self.checkpoint(task_id, "Stage 5: 多模型並行專案開發與 TDD 驗收", 75)
                accumulated_ctx += fb
                proj_dir = DELIVERY_ZONE / "projects" / task_id
                proj_dir.mkdir(parents=True, exist_ok=True)

                spec_prompt = f"""{accumulated_ctx}
請將此軟體需求拆解為 2~4 個可獨立開發的 Python 模組檔案，以及一份嚴格的 `test_suite.py` 測試規格。
輸出嚴格 JSON：
{{
  "files": [
    {{"filename": "module_a.py", "instruction": "詳細類別、函式簽名與實作邏輯"}},
    {{"filename": "module_b.py", "instruction": "詳細類別、函式簽名與實作邏輯"}}
  ],
  "test_instruction": "測試所有核心函式與邊界條件"
}}"""
                spec_raw = self.pool.generate(role="architect", prompt=spec_prompt, task_id=task_id, action_desc="拆解多工並行模組契約", format_json=True, num_ctx=decided_ctx)
                try:
                    proj_spec = json.loads(spec_raw)
                    file_list = proj_spec.get("files", [])
                except Exception:
                    file_list = [{"filename": f"{task_id.lower()}_core.py", "instruction": desc}]

                # 【核心機制】：依據當前記憶體上限，使用 ThreadPoolExecutor 同時呼叫多個模型並行編寫不同檔案！
                self.emit_log(task_id, f"⚡ [多模型並行] 啟動並行工程師集群同時實作 {len(file_list)} 個模組 (受 {self.pool.get_hardware_budget()}GB 預算保護)...")
                generated_files = {}

                def _parallel_coder(f_item):
                    self.checkpoint(task_id)
                    fn = f_item["filename"]
                    p = f"""請實作 Python 模組 `{fn}`：\n規格要求：{f_item['instruction']}\n{accumulated_ctx}\n只輸出純 Python 代碼，不含 Markdown。"""
                    code = self.pool.generate(role="coder_worker", prompt=p, task_id=task_id, action_desc=f"並行實作 {fn}", num_ctx=decided_ctx)
                    clean_c = code.replace("```python", "").replace("```", "").strip()
                    (proj_dir / fn).write_text(clean_c, encoding="utf-8")
                    return fn, clean_c

                with ThreadPoolExecutor(max_workers=min(4, max(1, len(file_list)))) as ex:
                    futs = [ex.submit(_parallel_coder, item) for item in file_list]
                    for fut in as_completed(futs):
                        fn, clean_c = fut.result()
                        generated_files[fn] = clean_c
                        self.emit_log(task_id, f"✔ 並行模組產出完成：{fn}")

                # 產出單元測試並執行 Tier 1 ~ Tier 3 主管升級修復
                self.checkpoint(task_id, "Stage 5.2: 沙盒 TDD 驗證與三級主管升級除錯", 85)
                t_prompt = f"""請為以下已完成之專案檔案撰寫 `test_suite.py` 自動化測試：
{json.dumps(generated_files, ensure_ascii=False)}
要求：第 1~3 行固定為 `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))`，使用 `assert` 驗證。只輸出純 Python 代碼。"""
                t_raw = self.pool.generate(role="coder_worker", prompt=t_prompt, task_id=task_id, action_desc="建立 TDD 測試套件", num_ctx=decided_ctx)
                test_code = t_raw.replace("```python", "").replace("```", "").strip()
                test_file = proj_dir / "test_suite.py"
                test_file.write_text(test_code, encoding="utf-8")

                escalation_tiers = [
                    {"tier": 1, "name": "Tier 1: 審查員快速修復", "role": "coder_worker", "ctx": decided_ctx},
                    {"tier": 2, "name": "Tier 2: 研發主管重構", "role": "coder_lead", "ctx": min(decided_ctx * 2, 65536)},
                    {"tier": 3, "name": "Tier 3: 首席仲裁長終裁", "role": "arbiter", "ctx": 65536}
                ]
                for step in escalation_tiers:
                    self.checkpoint(task_id)
                    run_res = self.sandbox.run_python_script(test_file)
                    if run_res["success"]:
                        self.emit_log(task_id, "✔ 專案代碼通過沙盒 TDD 驗證！")
                        self.record_lesson(task_id, f"任務 {title} 多模組並行與 TDD 驗收通過。")
                        break
                    self.emit_log(task_id, f"▲ 測試未過，啟動 [{step['name']}] (Ctx: {step['ctx']}): {run_res['stderr'][:160]}")
                    fix_p = f"""TDD 測試失敗。
當前模組：{json.dumps(generated_files, ensure_ascii=False)}
測試腳本：{test_code}
錯誤堆疊：{run_res['stderr']}
請輸出需修正的檔案（若測試斷言不合理亦可修正 test_suite.py）：
<<<FILE: 檔名.py>>>
完整代碼
<<<END_FILE>>>"""
                    fixed_raw = self.pool.generate(role=step["role"], prompt=fix_p, task_id=task_id, action_desc=step["name"], num_ctx=step["ctx"])
                    for target_fn, c_body in re.findall(r"<<<FILE:\s*(.*?)>>>(.*?)<<<END_FILE>>>", fixed_raw, re.DOTALL):
                        target_fn = target_fn.strip()
                        c_clean = c_body.replace("```python", "").replace("```", "").strip()
                        if self.sandbox.is_valid_python(c_clean):
                            (proj_dir / target_fn).write_text(c_clean, encoding="utf-8")
                            if target_fn == "test_suite.py":
                                test_code = c_clean
                            else:
                                generated_files[target_fn] = c_clean

            # 6. 並行資安審查 + 綜合解答與交付報告生成
            fb = self.checkpoint(task_id, "Stage 6: 並行資安稽核與最終報告交付", 92)
            accumulated_ctx += fb
            self.emit_log(task_id, "⚡ [多模型並行] 同時執行『資安稽核專員』與『技術文檔/全能解答專家』...")

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_sec = ex.submit(
                    self.pool.generate, "fast",
                    f"請對本次任務產出進行安全性與合規性快速稽核：\n{accumulated_ctx[:4000]}",
                    "", task_id, "資安合規審查", None, 4096
                )
                fut_doc = ex.submit(
                    self.pool.generate, "architect",
                    f"請針對使用者需求提供完整、詳盡的最終解答與專案交付報告（Markdown 格式）：\n{accumulated_ctx}",
                    "", task_id, "撰寫綜合解答與交付報告", None, decided_ctx
                )
                sec_report = fut_sec.result()
                final_report = fut_doc.result()

            report_file = DELIVERY_ZONE / "reports" / f"{task_id}_final_delivery.md"
            report_file.write_text(f"{final_report}\n\n---\n### Security Audit\n{sec_report}", encoding="utf-8")

            self.update_task_status(task_id, "completed")
            self.current_task_progress_pct = 100
            self.current_task_stage = "Completed"
            self.emit_log(task_id, f"🎉 任務 [{task_id}] 完成！報告與成果已歸檔至 03_delivery。", event_type="completed", extra={"answer": final_report[:1500]})

        except TaskTerminatedException:
            self.update_task_status(task_id, "pending")
            self.current_task_stage = "Terminated"
            self.emit_log(task_id, "⏹ 任務已由使用者強制終止，狀態已重置回 pending。")
        except Exception as e:
            self.update_task_status(task_id, "failed")
            self.current_task_stage = "Failed"
            self.emit_log(task_id, f"❌ 任務異常中斷: {e}")

    # ================= 自適應多任務佇列調度主迴圈 =================
    def start_queue_loop(self):
        if self.daemon_active:
            return
        self.daemon_active = True
        self.set_control_state("RUNNING", pause_req=False, stop_req=False)

        def _loop():
            try:
                while True:
                    ctrl = self.get_control_state()
                    if ctrl.get("stop_requested"):
                        break

                    pending = self.peek_pending_tasks()
                    if not pending:
                        time.sleep(1.5)
                        continue

                    # 計算可並行任務槽位數
                    budget = self.pool.get_hardware_budget()
                    max_parallel_tasks = max(1, int(budget // 18.0))

                    with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
                        futures = []
                        for t in pending[:max_parallel_tasks]:
                            if self.get_control_state().get("stop_requested"):
                                break
                            futures.append(executor.submit(self.execute_omnipotent_task, t))
                        for f in as_completed(futures):
                            pass
            finally:
                self.daemon_active = False
                self.current_task_id = "-"
                self.current_task_stage = "Idle"
                self.set_control_state("IDLE", pause_req=False, stop_req=False)
                self.pool.unload_idle_models()

        threading.Thread(target=_loop, daemon=True).start()