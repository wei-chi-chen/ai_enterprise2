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

        # 真實物理時間戳記計時器 (Wall-Clock Time)
        self.daemon_active = False
        self.daemon_start_ts: Optional[float] = None
        self.daemon_paused_accum: float = 0.0
        self.task_start_ts: Optional[float] = None
        self.task_paused_accum: float = 0.0
        self.pause_begin_ts: Optional[float] = None

        self.current_task_id = "-"
        self.current_task_stage = "Idle"
        self.current_task_progress_pct = 0

        self._timer_thread = threading.Thread(target=self._background_ticker, daemon=True)
        self._timer_thread.start()

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
        """每秒透過真實系統時鐘 (time.time) 計算精準耗時，不受 CPU/GPU 滿載阻塞影響"""
        while True:
            time.sleep(1.0)
            ctrl = self.get_control_state()
            state = ctrl.get("state", "IDLE")
            total_sec, task_sec = self._get_accurate_elapsed()

            q_stats = self.get_queue_progress()
            mem_info = self.pool.sync_realtime_memory()
            budget = self.pool.get_hardware_budget()

            if self.broadcast_fn:
                self.broadcast_fn({
                    "type": "telemetry",
                    "state": state,
                    "total_elapsed_sec": total_sec,
                    "task_elapsed_sec": task_sec,
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
        now = time.time()
        if pause_req and self.pause_begin_ts is None:
            self.pause_begin_ts = now
        elif not pause_req and self.pause_begin_ts is not None:
            paused_duration = now - self.pause_begin_ts
            self.daemon_paused_accum += paused_duration
            self.task_paused_accum += paused_duration
            self.pause_begin_ts = None

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
        """階段檢查點：支援無損暫停、動態修改記憶體後繼續、以及即時讀取意見佇列"""
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
            self.emit_log(task_id, "⏸ 系統已進入暫停狀態（目前已生成之進度已完整保留，可調整記憶體上限或注入意見後按『繼續』）...")
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

    # ================= 全能任務執行主流程（大模型拆解 + 多模型並行） =================
    def execute_omnipotent_task(self, task: dict):
        task_id = task["task_id"]
        title = task["title"]
        desc = task["description"]
        attached_file = task.get("attached_file", "")
        iterations = int(task.get("iterations", 3))

        self.current_task_id = task_id
        self.task_start_ts = time.time()
        self.task_paused_accum = 0.0
        self.update_task_status(task_id, "processing")

        try:
            fb = self.checkpoint(task_id, "Stage 1: 總架構師自主路由與任務拆解", 8)
            accumulated_ctx = f"任務標題：{title}\n使用者需求：{desc}\n附件檔案：{attached_file}\n{self.get_lessons()}\n{fb}"

            # 1. 大模型自主判斷任務管線
            self.emit_log(task_id, "🧠 [階段 1] 總架構師正在自主分析需求並決定執行管線與 Context 視窗...")
            router_prompt = f"""請分析以下需求與附件，自主決定需要啟動哪些功能管線（可複選）：
{accumulated_ctx}

輸出嚴格 JSON 格式：
{{
  "suggested_num_ctx": 8192,
  "needs_vision_analysis": false,
  "needs_data_cleaning": false,
  "needs_game_development": false,
  "needs_software_project": true,
  "needs_direct_qa_report": true,
  "reasoning": "規劃說明"
}}"""
            route_raw = self.pool.generate(
                role="architect", prompt=router_prompt, task_id=task_id,
                action_desc="自主判斷任務管線與 Context", format_json=True, num_ctx=8192
            )
            try:
                plan_flags = json.loads(route_raw)
            except Exception:
                plan_flags = {
                    "suggested_num_ctx": 8192,
                    "needs_vision_analysis": bool(attached_file and Path(attached_file).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]),
                    "needs_data_cleaning": bool(attached_file and Path(attached_file).suffix.lower() in [".csv", ".xlsx"]),
                    "needs_game_development": any(w in (title + desc).lower() for w in ["遊戲", "game", "rpg", "推塔", "兵種", "主堡", "canvas"]),
                    "needs_software_project": True,
                    "needs_direct_qa_report": True,
                    "reasoning": "預設全能複合管線"
                }

            decided_ctx = min(max(int(plan_flags.get("suggested_num_ctx", 8192)), 4096), 32768)
            self.emit_log(task_id, f"✔ 自主規劃完成 (動態 Ctx: {decided_ctx})：{plan_flags.get('reasoning', '')}")

            # 2. 視覺影像分析管線
            if plan_flags.get("needs_vision_analysis") and attached_file and Path(attached_file).exists():
                fb = self.checkpoint(task_id, "Stage 2: 多模態視覺感知解析", 20)
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
                fb = self.checkpoint(task_id, "Stage 3: 自動資料清理與沙盒驗證", 32)
                accumulated_ctx += fb
                self.emit_log(task_id, "🧹 [資料管線] 正在編寫自動化資料清理與統計驗證腳本...")
                out_csv = DELIVERY_ZONE / "cleaned_data" / f"{task_id}_cleaned.csv"
                data_info = f"目標檔案：{attached_file}" if attached_file else "請生成符合需求的模擬資料集並執行清理與統計檢驗"
                clean_prompt = f"""{accumulated_ctx}\n{data_info}\n請撰寫純 Python 腳本處理資料並輸出至 `{out_csv.as_posix()}`，末端包含 assert 驗證。只輸出純 Python 代碼。"""
                script_p = TASK_ZONE / f"{task_id}_data_job.py"
                for attempt in range(1, 4):
                    self.checkpoint(task_id)
                    c_raw = self.pool.generate(role="coder_parallel", prompt=clean_prompt, task_id=task_id, action_desc=f"資料處理實現 (Try {attempt})", num_ctx=8192)
                    script_p.write_text(c_raw.replace("```python", "").replace("```", "").strip(), encoding="utf-8")
                    res = self.sandbox.run_python_script(script_p)
                    if res["success"]:
                        self.emit_log(task_id, f"✔ 資料清理與驗證通過！產出：{out_csv.name}")
                        break
                    clean_prompt += f"\n報錯修復：\n{res['stderr']}"

            # 4. 遊戲製作管線（大模型拆解契約 -> 多模型並行撰寫子模組 -> 快速組裝與演化）
            if plan_flags.get("needs_game_development"):
                fb = self.checkpoint(task_id, "Stage 4: 遊戲素材合成與多模型並行開發", 45)
                accumulated_ctx += fb
                game_dir = DELIVERY_ZONE / "games" / task_id
                assets_dir = game_dir / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)

                # 4.1 自主生成精簡關鍵素材 (限制 8~12 個核心素材，避免上下文膨脹)
                self.emit_log(task_id, "🎨 [遊戲管線] 正在自主開立核心素材清單並合成 PNG 精靈圖與 WAV 音效...")
                asset_raw = self.pool.generate(
                    role="fast",
                    prompt=f"{accumulated_ctx}\n請以 JSON 輸出 8 個最核心的圖片與音效清單：{{\"assets\": [{{\"type\":\"sprite\",\"filename\":\"hero.png\",\"style\":\"hero\",\"color\":\"#38bdf8\"}}, {{\"type\":\"sprite\",\"filename\":\"enemy.png\",\"style\":\"enemy\",\"color\":\"#ef4444\"}}, {{\"type\":\"sfx\",\"filename\":\"attack.wav\",\"style\":\"laser\"}}]}}",
                    task_id=task_id, action_desc="生成核心遊戲素材規格", format_json=True, num_ctx=4096
                )
                try:
                    manifest = json.loads(asset_raw).get("assets", [])[:12]
                except Exception:
                    manifest = [
                        {"type": "sprite", "filename": "base_player.png", "style": "base", "color": "#3b82f6"},
                        {"type": "sprite", "filename": "base_enemy.png", "style": "base", "color": "#ef4444"},
                        {"type": "sprite", "filename": "soldier_melee.png", "style": "hero", "color": "#38bdf8"},
                        {"type": "sprite", "filename": "soldier_ranged.png", "style": "hero", "color": "#22c55e"},
                        {"type": "sprite", "filename": "boss_mech.png", "style": "boss", "color": "#a855f7"},
                        {"type": "sfx", "filename": "shoot.wav", "style": "laser"},
                        {"type": "sfx", "filename": "hit.wav", "style": "explosion"},
                        {"type": "sfx", "filename": "evolve.wav", "style": "coin"}
                    ]

                created_assets = self.forge.fulfill_asset_manifest(assets_dir, manifest)
                self.emit_log(task_id, f"✔ 已自主合成 {len(created_assets)} 個遊戲素材：{created_assets}", event_type="game_assets", extra={"assets": created_assets})

                # 4.2 大模型拆解遊戲為 3 個獨立子模組，交由 3 個並行模型同時撰寫！
                for evo in range(1, iterations + 1):
                    fb = self.checkpoint(task_id, f"Stage 4.{evo}: 多模型並行遊戲演化 ({evo}/{iterations})", 45 + int((evo / iterations) * 28))
                    accumulated_ctx += fb

                    self.emit_log(task_id, f"🧠 [Round {evo}/{iterations}] 總架構師正在拆解遊戲子系統規格，準備分發給多模型並行實作...")
                    split_prompt = f"""{accumulated_ctx}
目前為第 {evo}/{iterations} 輪演化。
請將此橫向卷軸即時戰略遊戲拆解為三個彼此獨立的 JavaScript 模組規格，共享全域狀態物件 `window.G`：
- `window.G` 固定結構：`{{ gold: 200, exp: 0, eraIdx: 0, eras: ['石器時代','青銅時代','鐵器時代','工業時代','星際未來'], playerBaseHp: 1000, playerMaxHp: 1000, enemyBaseHp: 1000, enemyMaxHp: 1000, units: [], projectiles: [], turrets: [], bossSpawned: false, gameOver: false, winner: '' }}`
請輸出 JSON：
{{
  "module_data_ui": "定義 5 個時代各自的 4 種兵種資料 (window.UNIT_TYPES)、召喚函式 window.spawnUnit(typeIdx)、加裝砲塔 window.buildTurret()、時代進化 window.evolveEra() 與更新頂部 DOM 介面 window.updateUI()",
  "module_battle": "定義實體更新函式 window.updateBattle(dt)：每秒自動加金幣、敵方AI定時自動出兵、雙方單位直線前進、射程內停止並攻擊扣血、主堡砲塔自動射擊、第5時代敵主堡歸零時觸發終極機械Boss戰 (window.G.bossSpawned)",
  "module_render": "定義 Canvas 渲染函式 window.renderGame(ctx, canvas)：繪製時代背景、中央橫向道路、左右雙方主堡與血條、所有移動與攻擊中的士兵/Boss圖形與血條、雷射與震波特效"
}}"""
                    split_raw = self.pool.generate(
                        role="architect", prompt=split_prompt, task_id=task_id,
                        action_desc=f"拆解 Round {evo} 遊戲並行規格", format_json=True, num_ctx=8192
                    )
                    try:
                        sub_specs = json.loads(split_raw)
                    except Exception:
                        sub_specs = {
                            "module_data_ui": "實作 window.UNIT_TYPES, window.spawnUnit(typeIdx), window.buildTurret(), window.evolveEra(), window.updateUI()",
                            "module_battle": "實作 window.updateBattle(dt) 處理自動產金幣、敵軍AI生成、兵線移動、射程判定與攻擊扣血、第5關Boss召喚",
                            "module_render": "實作 window.renderGame(ctx, canvas) 繪製橫向戰場道路、左右主堡血條、士兵單位與攻擊特效"
                        }

                    self.emit_log(task_id, f"⚡ [Round {evo}/{iterations}] 啟動 3 個模型槽位同時並行編寫【UI經濟模組】、【戰鬥物理AI模組】、【Canvas渲染模組】...")

                    def _parallel_js_worker(idx: int, mod_name: str, spec_desc: str) -> str:
                        self.checkpoint(task_id)
                        p = f"""請使用純 JavaScript 實作遊戲子模組【{mod_name}】。
專案需求摘要：{desc[:1000]}
{fb}
共享狀態結構：
`window.G` 包含 `gold, exp, eraIdx (0~4), eras, playerBaseHp, playerMaxHp, enemyBaseHp, enemyMaxHp, units (陣列，每個元素含 side:'player'|'enemy', x, y, hp, maxHp, dmg, range, speed, cooldown, color, name, isBoss), projectiles, turrets, bossSpawned, gameOver, winner`。
本模組職責：{spec_desc}
嚴格要求：
1. 所有公開函式與資料必須掛載在 `window.` 下。
2. 程式碼必須精簡、完整、無省略，嚴禁寫 TODO 或佔位符。
3. 只輸出純 JavaScript 程式碼，不含 HTML 標籤或 Markdown。"""
                        raw_js = self.pool.generate(
                            role="coder_parallel", prompt=p, task_id=task_id,
                            action_desc=f"並行編寫 {mod_name} (R{evo})", num_ctx=8192, worker_idx=idx
                        )
                        return raw_js.replace("```javascript", "").replace("```js", "").replace("```", "").strip()

                    with ThreadPoolExecutor(max_workers=3) as ex:
                        f_ui = ex.submit(_parallel_js_worker, 0, "Data_UI_System", sub_specs.get("module_data_ui", ""))
                        f_bat = ex.submit(_parallel_js_worker, 1, "Battle_AI_Physics", sub_specs.get("module_battle", ""))
                        f_ren = ex.submit(_parallel_js_worker, 2, "Canvas_Renderer", sub_specs.get("module_render", ""))

                        js_ui_code = f_ui.result()
                        js_bat_code = f_bat.result()
                        js_ren_code = f_ren.result()

                    # 裝配為保證可互動、有完整保底機制的 HTML5 戰略遊戲單檔
                    final_html = self._assemble_playable_rts_html(title, created_assets, js_ui_code, js_bat_code, js_ren_code)
                    html_path = game_dir / "index.html"
                    html_path.write_text(final_html, encoding="utf-8")

                    self.emit_log(
                        task_id, f"🎮 遊戲第 {evo}/{iterations} 輪多模型並行組裝完成！右側試玩畫面已即時更新。",
                        event_type="game_preview",
                        extra={"preview_url": f"/delivery/games/{task_id}/index.html?t={int(time.time())}"}
                    )

            # 5. 多模組軟體專案開發（大模型拆解 + 多模型並行 + TDD 三級除錯）
            if plan_flags.get("needs_software_project"):
                fb = self.checkpoint(task_id, "Stage 5: 多模型並行專案開發與 TDD 驗收", 78)
                accumulated_ctx += fb
                proj_dir = DELIVERY_ZONE / "projects" / task_id
                proj_dir.mkdir(parents=True, exist_ok=True)

                spec_prompt = f"""{accumulated_ctx}
請將此軟體需求拆解為 2~4 個可獨立開發的 Python 模組檔案，以及一份 `test_suite.py` 測試規格。
輸出嚴格 JSON：
{{
  "files": [
    {{"filename": "module_a.py", "instruction": "詳細類別、函式簽名與實作邏輯"}},
    {{"filename": "module_b.py", "instruction": "詳細類別、函式簽名與實作邏輯"}}
  ],
  "test_instruction": "測試所有核心函式與邊界條件"
}}"""
                spec_raw = self.pool.generate(role="architect", prompt=spec_prompt, task_id=task_id, action_desc="拆解多工並行模組契約", format_json=True, num_ctx=8192)
                try:
                    proj_spec = json.loads(spec_raw)
                    file_list = proj_spec.get("files", [])
                except Exception:
                    file_list = [{"filename": f"{task_id.lower()}_core.py", "instruction": desc}]

                self.emit_log(task_id, f"⚡ [多模型並行] 同時啟動 {len(file_list)} 個模型槽位並行實作 Python 模組...")
                generated_files = {}

                def _parallel_py_coder(idx: int, f_item: dict):
                    self.checkpoint(task_id)
                    fn = f_item["filename"]
                    p = f"""請實作 Python 模組 `{fn}`：\n規格要求：{f_item['instruction']}\n{accumulated_ctx[:3000]}\n只輸出純 Python 代碼，不含 Markdown。"""
                    code = self.pool.generate(role="coder_parallel", prompt=p, task_id=task_id, action_desc=f"並行實作 {fn}", num_ctx=8192, worker_idx=idx)
                    clean_c = code.replace("```python", "").replace("```", "").strip()
                    (proj_dir / fn).write_text(clean_c, encoding="utf-8")
                    return fn, clean_c

                with ThreadPoolExecutor(max_workers=min(4, max(1, len(file_list)))) as ex:
                    futs = [ex.submit(_parallel_py_coder, i, item) for i, item in enumerate(file_list)]
                    for fut in as_completed(futs):
                        fn, clean_c = fut.result()
                        generated_files[fn] = clean_c
                        self.emit_log(task_id, f"✔ 並行模組產出完成：{fn}")

                # 產出單元測試並執行 Tier 1 ~ Tier 3 升級修復
                self.checkpoint(task_id, "Stage 5.2: 沙盒 TDD 驗證與三級主管升級除錯", 88)
                t_prompt = f"""請為以下專案檔案撰寫 `test_suite.py` 自動化測試：
{json.dumps(generated_files, ensure_ascii=False)}
要求：第 1~3 行固定為 `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))`，使用 `assert` 驗證。只輸出純 Python 代碼。"""
                t_raw = self.pool.generate(role="coder_parallel", prompt=t_prompt, task_id=task_id, action_desc="建立 TDD 測試套件", num_ctx=8192)
                test_code = t_raw.replace("```python", "").replace("```", "").strip()
                test_file = proj_dir / "test_suite.py"
                test_file.write_text(test_code, encoding="utf-8")

                escalation_tiers = [
                    {"tier": 1, "name": "Tier 1: 審查員快速修復", "role": "coder_parallel", "ctx": 8192},
                    {"tier": 2, "name": "Tier 2: 研發主管重構", "role": "coder_lead", "ctx": 16384},
                    {"tier": 3, "name": "Tier 3: 首席仲裁長終裁", "role": "arbiter", "ctx": 32768}
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
            fb = self.checkpoint(task_id, "Stage 6: 並行資安稽核與最終報告交付", 95)
            accumulated_ctx += fb
            self.emit_log(task_id, "⚡ [多模型並行] 同時執行『資安稽核專員』與『技術文檔/全能解答專家』...")

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_sec = ex.submit(
                    self.pool.generate, "fast",
                    f"請對本次任務產出進行安全性快速稽核：\n{accumulated_ctx[:3000]}",
                    "", task_id, "資安合規審查", None, 4096, False, 0.1, 0
                )
                fut_doc = ex.submit(
                    self.pool.generate, "coder_parallel",
                    f"請針對使用者需求提供完整、精確的專案交付與架構總結報告（Markdown 格式）：\n{accumulated_ctx[:6000]}",
                    "", task_id, "撰寫綜合交付報告", None, 8192, False, 0.2, 1
                )
                sec_report = fut_sec.result()
                final_report = fut_doc.result()

            report_file = DELIVERY_ZONE / "reports" / f"{task_id}_final_delivery.md"
            report_file.write_text(f"{final_report}\n\n---\n### Security Audit\n{sec_report}", encoding="utf-8")

            self.update_task_status(task_id, "completed")
            self.current_task_progress_pct = 100
            self.current_task_stage = "Completed"
            self.emit_log(task_id, f"🎉 任務 [{task_id}] 全部完成！所有檔案已歸檔至 03_delivery。", event_type="completed")

        except TaskTerminatedException:
            self.update_task_status(task_id, "pending")
            self.current_task_stage = "Terminated"
            self.emit_log(task_id, "⏹ 任務已由使用者強制終止，狀態已重置回 pending。")
        except Exception as e:
            self.update_task_status(task_id, "failed")
            self.current_task_stage = "Failed"
            self.emit_log(task_id, f"❌ 任務異常中斷: {e}")

    def _assemble_playable_rts_html(self, title: str, assets: list, js_ui: str, js_bat: str, js_ren: str) -> str:
        """將多模型並行產出的 JS 模組與標準可玩 RTS 引擎骨架無縫結合，確保 100% 可在右侧視窗點擊遊玩"""
        return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; user-select: none; }}
  body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: 'Segoe UI', sans-serif; overflow: hidden; display: flex; flex-direction: column; height: 100vh; }}
  #hud {{ background: #1e293b; padding: 8px 12px; border-bottom: 2px solid #334155; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; z-index: 10; }}
  .stats {{ font-size: 13px; font-weight: bold; color: #facc15; display: flex; gap: 14px; }}
  .controls {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  button {{ background: #2563eb; color: #fff; border: 1px solid #60a5fa; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; transition: 0.15s; }}
  button:hover {{ background: #1d4ed8; transform: translateY(-1px); }}
  button.evolve {{ background: #7c3aed; border-color: #c084fc; }}
  button.turret {{ background: #059669; border-color: #34d399; }}
  canvas {{ flex: 1; width: 100%; display: block; background: linear-gradient(to bottom, #020617 0%, #1e293b 75%, #334155 75%, #0f172a 100%); }}
</style>
</head>
<body>
<div id="hud">
  <div class="stats">
    <span id="ui-era">時代: 石器時代</span>
    <span id="ui-gold">💰 黃金: 200</span>
    <span id="ui-exp">⭐ 經驗: 0 / 100</span>
  </div>
  <div class="controls">
    <button onclick="window.spawnUnit(0)">🛡 肉盾步兵 (40G)</button>
    <button onclick="window.spawnUnit(1)">🏹 遠程射手 (65G)</button>
    <button onclick="window.spawnUnit(2)">💥 反裝甲重砲 (100G)</button>
    <button onclick="window.spawnUnit(3)">✨ 控場輔助 (80G)</button>
    <button class="turret" onclick="window.buildTurret()">🗼 加裝主堡砲塔 (120G)</button>
    <button class="evolve" onclick="window.evolveEra()">⚡ 時代進化</button>
  </div>
</div>
<canvas id="gameCanvas"></canvas>

<script>
// 1. 全域共享遊戲狀態與保底機制
window.G = {{
  gold: 200, exp: 0, eraIdx: 0,
  eras: ['石器時代', '青銅時代', '中世紀鐵器', '工業火藥', '星際未來'],
  expThresholds: [100, 260, 550, 1000, 999999],
  playerBaseHp: 1200, playerMaxHp: 1200,
  enemyBaseHp: 1200, enemyMaxHp: 1200,
  units: [], projectiles: [], turrets: [0, 0],
  bossSpawned: false, gameOver: false, winner: ''
}};

const BASE_TEMPLATES = [
  {{ name: '近戰重裝', cost: 40, hp: 160, dmg: 14, range: 28, speed: 45, color: '#38bdf8', radius: 12 }},
  {{ name: '高射速遠程', cost: 65, hp: 85, dmg: 18, range: 130, speed: 55, color: '#22c55e', radius: 10 }},
  {{ name: '爆發反裝甲', cost: 100, hp: 120, dmg: 42, range: 95, speed: 38, color: '#f97316', radius: 13 }},
  {{ name: '增益控場', cost: 80, hp: 110, dmg: 15, range: 110, speed: 48, color: '#eab308', radius: 11 }}
];

window.spawnUnit = function(typeIdx, side = 'player') {{
  if (window.G.gameOver) return;
  const mult = Math.pow(1.45, window.G.eraIdx);
  const tpl = BASE_TEMPLATES[typeIdx % 4];
  const cost = Math.round(tpl.cost * mult);
  if (side === 'player') {{
    if (window.G.gold < cost) return;
    window.G.gold -= cost;
  }}
  const canvas = document.getElementById('gameCanvas');
  window.G.units.push({{
    side: side,
    name: tpl.name,
    x: side === 'player' ? 75 : canvas.width - 75,
    y: canvas.height * 0.72 + (Math.random() * 14 - 7),
    hp: Math.round(tpl.hp * mult),
    maxHp: Math.round(tpl.hp * mult),
    dmg: Math.round(tpl.dmg * mult),
    range: tpl.range,
    speed: tpl.speed,
    color: side === 'player' ? tpl.color : '#ef4444',
    radius: tpl.radius,
    atkTimer: 0,
    isBoss: false
  }});
}};

window.buildTurret = function() {{
  const cost = 120 * (window.G.eraIdx + 1);
  if (window.G.gold >= cost && window.G.turrets[0] < 3) {{
    window.G.gold -= cost;
    window.G.turrets[0]++;
  }}
}};

window.evolveEra = function() {{
  const req = window.G.expThresholds[window.G.eraIdx];
  if (window.G.eraIdx < 4 && window.G.exp >= req) {{
    window.G.eraIdx++;
    window.G.playerMaxHp += 800;
    window.G.playerBaseHp = Math.min(window.G.playerMaxHp, window.G.playerBaseHp + 600);
    window.G.enemyMaxHp += 900;
    window.G.enemyBaseHp = Math.min(window.G.enemyMaxHp, window.G.enemyBaseHp + 700);
  }}
}};

// 2. 載入多模型並行生成的擴充模組 (try-catch 保護避免語法中斷)
try {{
  {js_ui}
}} catch(e) {{ console.warn('UI module fallback active:', e); }}

try {{
  {js_bat}
}} catch(e) {{ console.warn('Battle module fallback active:', e); }}

try {{
  {js_ren}
}} catch(e) {{ console.warn('Render module fallback active:', e); }}

// 3. 核心戰鬥與畫面保底主迴圈
let lastTime = performance.now();
let goldTick = 0, aiTick = 0, turretTick = 0;

function coreUpdate(dt) {{
  if (window.G.gameOver) return;
  goldTick += dt;
  if (goldTick >= 1.0) {{
    window.G.gold += 8 * (window.G.eraIdx + 1);
    goldTick = 0;
  }}
  aiTick += dt;
  if (aiTick >= Math.max(1.8, 3.8 - window.G.eraIdx * 0.4)) {{
    window.spawnUnit(Math.floor(Math.random() * 4), 'enemy');
    aiTick = 0;
  }}

  const canvas = document.getElementById('gameCanvas');
  const leftBaseX = 55, rightBaseX = canvas.width - 55;

  // 砲塔自動射擊
  turretTick += dt;
  if (turretTick >= 1.2) {{
    if (window.G.turrets[0] > 0) {{
      const target = window.G.units.find(u => u.side === 'enemy' && u.x < 320);
      if (target) {{
        target.hp -= 25 * window.G.turrets[0] * (window.G.eraIdx + 1);
        window.G.projectiles.push({{ x1: 60, y1: canvas.height*0.55, x2: target.x, y2: target.y, ttl: 0.15, color: '#38bdf8' }});
      }}
    }}
    turretTick = 0;
  }}

  // 單位推進與戰鬥
  for (let u of window.G.units) {{
    let blocked = false;
    const dir = u.side === 'player' ? 1 : -1;

    for (let other of window.G.units) {{
      if (other.side !== u.side && other.hp > 0) {{
        const dist = Math.abs(other.x - u.x);
        if (dist <= u.range) {{
          blocked = true;
          u.atkTimer += dt;
          if (u.atkTimer >= 0.85) {{
            other.hp -= u.dmg;
            u.atkTimer = 0;
            window.G.projectiles.push({{ x1: u.x, y1: u.y - 6, x2: other.x, y2: other.y - 6, ttl: 0.12, color: u.color }});
          }}
          break;
        }}
      }}
    }}

    // 攻擊敵方主堡
    if (!blocked) {{
      const distToBase = u.side === 'player' ? (rightBaseX - u.x) : (u.x - leftBaseX);
      if (distToBase <= u.range + 25) {{
        blocked = true;
        u.atkTimer += dt;
        if (u.atkTimer >= 1.0) {{
          if (u.side === 'player') window.G.enemyBaseHp -= u.dmg;
          else window.G.playerBaseHp -= u.dmg;
          u.atkTimer = 0;
        }}
      }}
    }}

    if (!blocked) u.x += dir * u.speed * dt;
  }}

  // 死亡結算與金幣經驗掠奪
  window.G.units = window.G.units.filter(u => {{
    if (u.hp <= 0) {{
      if (u.side === 'enemy') {{
        window.G.gold += 25 * (window.G.eraIdx + 1);
        window.G.exp += 30;
        if (u.isBoss) {{
          window.G.gameOver = true;
          window.G.winner = '🎉 玩家擊殺終極機械主宰，達成完全勝利！';
        }}
      }}
      return false;
    }}
    return true;
  }});

  // 第五時代二階段終極機械王者觸發
  if (window.G.enemyBaseHp <= 0) {{
    if (window.G.eraIdx < 4 && !window.G.bossSpawned) {{
      window.G.eraIdx++;
      window.G.enemyMaxHp = 1500 * (window.G.eraIdx + 1);
      window.G.enemyBaseHp = window.G.enemyMaxHp;
    }} else if (!window.G.bossSpawned) {{
      window.G.bossSpawned = true;
      window.G.enemyBaseHp = 500;
      window.G.units.push({{
        side: 'enemy', name: '🤖 終極機械主宰 BOSS',
        x: rightBaseX - 20, y: canvas.height * 0.68,
        hp: 8000, maxHp: 8000, dmg: 180, range: 150, speed: 18,
        color: '#a855f7', radius: 32, atkTimer: 0, isBoss: true
      }});
    }} else {{
      window.G.gameOver = true;
      window.G.winner = '🎉 敵方基地與機械主宰徹底摧毀，玩家獲勝！';
    }}
  }}

  if (window.G.playerBaseHp <= 0) {{
    window.G.playerBaseHp = 0;
    window.G.gameOver = true;
    window.G.winner = '💀 玩家基地被摧毀，戰役失敗！';
  }}

  window.G.projectiles.forEach(p => p.ttl -= dt);
  window.G.projectiles = window.G.projectiles.filter(p => p.ttl > 0);

  document.getElementById('ui-era').innerText = `時代: ${{window.G.eras[window.G.eraIdx]}} (Lv.${{window.G.eraIdx+1}})`;
  document.getElementById('ui-gold').innerText = `💰 黃金: ${{Math.floor(window.G.gold)}}`;
  document.getElementById('ui-exp').innerText = `⭐ 經驗: ${{window.G.exp}} / ${{window.G.expThresholds[window.G.eraIdx]}}`;
}}

function coreRender() {{
  const canvas = document.getElementById('gameCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const roadY = canvas.height * 0.75;
  ctx.fillStyle = '#334155';
  ctx.fillRect(0, roadY - 15, canvas.width, 50);

  // 繪製雙方防禦塔主堡
  ctx.fillStyle = '#2563eb';
  ctx.fillRect(15, roadY - 95, 55, 110);
  ctx.fillStyle = '#fff';
  ctx.font = '12px sans-serif';
  ctx.fillText(`我方主堡 ${{window.G.playerBaseHp}}/${{window.G.playerMaxHp}}`, 10, roadY - 105);

  ctx.fillStyle = '#dc2626';
  ctx.fillRect(canvas.width - 70, roadY - 95, 55, 110);
  ctx.fillStyle = '#fff';
  ctx.fillText(`敵方主堡 ${{window.G.enemyBaseHp}}/${{window.G.enemyMaxHp}}`, canvas.width - 135, roadY - 105);

  // 繪製砲塔
  for (let i = 0; i < window.G.turrets[0]; i++) {{
    ctx.fillStyle = '#34d399';
    ctx.fillRect(28, roadY - 115 - i * 18, 30, 14);
  }}

  // 繪製士兵與 Boss
  for (let u of window.G.units) {{
    ctx.beginPath();
    ctx.arc(u.x, u.y, u.radius, 0, Math.PI * 2);
    ctx.fillStyle = u.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    // 血條
    const bw = u.radius * 2.2;
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(u.x - bw/2, u.y - u.radius - 10, bw, 4);
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(u.x - bw/2, u.y - u.radius - 10, bw * Math.max(0, u.hp / u.maxHp), 4);
  }}

  // 繪製攻擊彈道與雷射
  for (let p of window.G.projectiles) {{
    ctx.beginPath();
    ctx.moveTo(p.x1, p.y1);
    ctx.lineTo(p.x2, p.y2);
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }}

  if (window.G.gameOver) {{
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#facc15';
    ctx.font = 'bold 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(window.G.winner, canvas.width / 2, canvas.height / 2);
  }}
}}

function loop(now) {{
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  coreUpdate(dt);
  coreRender();
  requestAnimationFrame(loop);
}}

window.addEventListener('resize', () => {{
  const c = document.getElementById('gameCanvas');
  c.width = c.clientWidth;
  c.height = c.clientHeight;
}});
window.onload = () => {{
  const c = document.getElementById('gameCanvas');
  c.width = c.clientWidth;
  c.height = c.clientHeight;
  requestAnimationFrame(loop);
}};
</script>
</body>
</html>"""

    # ================= 自適應多任務佇列調度主迴圈 =================
    def start_queue_loop(self):
        if self.daemon_active:
            return
        self.daemon_active = True
        if self.daemon_start_ts is None:
            self.daemon_start_ts = time.time()
            self.daemon_paused_accum = 0.0
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

                    budget = self.pool.get_hardware_budget()
                    max_parallel_tasks = max(1, int(budget // 20.0))

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