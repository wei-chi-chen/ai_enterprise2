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

        # 啟動時將上次中斷的 processing 任務自動恢復為 pending (確保未完成任務可立即被看到並續跑)
        self._recover_unfinished_tasks_on_boot()

        self.daemon_active = False
        self.daemon_start_ts: Optional[float] = None
        self.daemon_paused_accum: float = 0.0
        self.task_start_ts: Optional[float] = None
        self.task_paused_accum: float = 0.0
        self.pause_begin_ts: Optional[float] = None

        self.current_task_id = "-"
        self.current_task_stage = "Idle (Ready to Resume)"
        self.current_task_progress_pct = 0

        self._timer_thread = threading.Thread(target=self._background_ticker, daemon=True)
        self._timer_thread.start()

    def _recover_unfinished_tasks_on_boot(self):
        with self.backlog_lock:
            if not BACKLOG_FILE.exists():
                return
            try:
                tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8"))
                changed = False
                for t in tasks:
                    if t.get("status") in ["processing", "paused"]:
                        t["status"] = "pending"
                        changed = True
                if changed:
                    BACKLOG_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

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
                    "active_workers": list(self.pool.active_workers.values()),
                    "tasks": self.get_all_tasks()
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
        tasks = self.get_all_tasks()
        return [t for t in tasks if t.get("status") in ["pending", "paused"]]

    # ================= 全能任務執行主流程（貪婪塞滿記憶體 + 全模型協同） =================
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

            # 3. 資料清理管線
            if plan_flags.get("needs_data_cleaning"):
                fb = self.checkpoint(task_id, "Stage 3: 自動資料清理與沙盒驗證", 32)
                accumulated_ctx += fb
                out_csv = DELIVERY_ZONE / "cleaned_data" / f"{task_id}_cleaned.csv"
                clean_prompt = f"""{accumulated_ctx}\n請撰寫純 Python 腳本處理資料並輸出至 `{out_csv.as_posix()}`，末端包含 assert 驗證。只輸出純 Python 代碼。"""
                script_p = TASK_ZONE / f"{task_id}_data_job.py"
                for attempt in range(1, 4):
                    self.checkpoint(task_id)
                    c_raw = self.pool.generate(role="coder", prompt=clean_prompt, task_id=task_id, action_desc=f"資料處理實現 (Try {attempt})", num_ctx=8192)
                    script_p.write_text(c_raw.replace("```python", "").replace("```", "").strip(), encoding="utf-8")
                    res = self.sandbox.run_python_script(script_p)
                    if res["success"]:
                        self.emit_log(task_id, f"✔ 資料清理與驗證通過！產出：{out_csv.name}")
                        break
                    clean_prompt += f"\n報錯修復：\n{res['stderr']}"

            # 4. 遊戲製作管線（貪婪塞滿 40GB 記憶體，同時派出 N 個不同模型並行開發）
            if plan_flags.get("needs_game_development"):
                fb = self.checkpoint(task_id, "Stage 4: 遊戲素材合成與全模型並行開發", 45)
                accumulated_ctx += fb
                game_dir = DELIVERY_ZONE / "games" / task_id
                assets_dir = game_dir / "assets"
                assets_dir.mkdir(parents=True, exist_ok=True)

                existing_assets = [p.name for p in assets_dir.iterdir() if p.is_file()]
                if existing_assets:
                    created_assets = existing_assets
                    self.emit_log(task_id, f"📦 偵測到已存在之 {len(created_assets)} 個遊戲素材，直接載入免重算！", event_type="game_assets", extra={"assets": created_assets})
                else:
                    self.emit_log(task_id, "🎨 [遊戲管線] 正在合成核心 PNG 精靈圖與 WAV 音效...")
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
                    self.emit_log(task_id, f"✔ 已自主合成 {len(created_assets)} 個遊戲素材", event_type="game_assets", extra={"assets": created_assets})

                # 固定 5 大平行遊戲子模組，交由貪婪模型集群同時開發
                game_submodules = [
                    ("Unit_Stats_Era_System", "定義 5 個時代（石器/青銅/鐵器/工業/星際）各自的 4 種相剋兵種數值表 `window.ERA_UNITS` 與升級倍率"),
                    ("Economy_Turret_UI", "定義金幣每秒增長、經驗值門檻判斷、加裝主堡砲塔 `window.buildTurret()` 與時代晉升 `window.evolveEra()` 擴充特效"),
                    ("Battle_Physics_AI", "定義電腦敵軍自動出兵波次 AI `window.updateEnemyAI(dt)` 與單位技能（範圍增益、緩速控場、反裝甲暴擊）"),
                    ("Boss_Phase2_System", "定義第5關敵方主堡摧毀後觸發的『終極機械主宰 Boss』行為：震波踩踏、直線高能雷射光束、召喚自爆無人機群 `window.updateBossSkill(dt)`"),
                    ("Canvas_VFX_Renderer", "定義進階 Canvas 視覺特效渲染 `window.renderCustomVFX(ctx, canvas)`：時代背景切換、雷射光束、爆炸粒子與浮動傷害數字")
                ]

                for evo in range(1, iterations + 1):
                    fb = self.checkpoint(task_id, f"Stage 4.{evo}: 滿載多模型並行遊戲演化 ({evo}/{iterations})", 45 + int((evo / iterations) * 28))
                    accumulated_ctx += fb

                    # 規劃能把 40GB 塞滿的模型集群
                    swarm_models = self.pool.plan_greedy_swarm()
                    swarm_names = [m["name"] for m in swarm_models]
                    total_swarm_gb = round(sum(m["est_vram_gb"] for m in swarm_models), 1)
                    self.emit_log(
                        task_id,
                        f"🔥 [Round {evo}/{iterations} 記憶體滿載調度] 已規劃同時啟動 {len(swarm_models)} 個模型槽位 (預估總載入: {total_swarm_gb} GB / 上限 {self.pool.get_hardware_budget()} GB)：{swarm_names}"
                    )

                    def _run_game_submodule(idx: int, mod_title: str, mod_spec: str, target_model: str) -> str:
                        self.checkpoint(task_id)
                        p = f"""請使用純 JavaScript 實作遊戲子模組【{mod_title}】。
任務背景：{desc[:800]}
{fb}
共享全域狀態：`window.G` (含 gold, exp, eraIdx, eras, playerBaseHp, enemyBaseHp, units, projectiles, turrets, bossSpawned, gameOver)。
本模組負責：{mod_spec}
要求：所有函式掛載於 `window.` 下，直接輸出可執行的純 JavaScript 代碼，不加 HTML 或 Markdown。"""
                        res = self.pool.generate(
                            role="coder", prompt=p, task_id=task_id,
                            action_desc=f"並行開發 {mod_title} (R{evo})",
                            num_ctx=8192, preferred_model=target_model
                        )
                        return res.replace("```javascript", "").replace("```js", "").replace("```", "").strip()

                    js_parts = []
                    with ThreadPoolExecutor(max_workers=max(len(game_submodules), len(swarm_models))) as ex:
                        futures = []
                        for i, (m_title, m_spec) in enumerate(game_submodules):
                            assigned_m = swarm_models[i % len(swarm_models)]["name"]
                            futures.append(ex.submit(_run_game_submodule, i, m_title, m_spec, assigned_m))
                        for fut in as_completed(futures):
                            js_parts.append(fut.result())

                    final_html = self._assemble_playable_rts_html(title, created_assets, "\n\n".join(js_parts))
                    html_path = game_dir / "index.html"
                    html_path.write_text(final_html, encoding="utf-8")

                    self.emit_log(
                        task_id, f"🎮 遊戲第 {evo}/{iterations} 輪（{len(game_submodules)} 模型並行）組裝完成！右側試玩畫面已同步更新。",
                        event_type="game_preview",
                        extra={"preview_url": f"/delivery/games/{task_id}/index.html?t={int(time.time())}"}
                    )

            # 5. 多模組軟體專案開發（根據 40GB 預算動態拆成 N 個模組同時跑滿所有模型）
            if plan_flags.get("needs_software_project"):
                fb = self.checkpoint(task_id, "Stage 5: 滿載多模型並行專案開發與 TDD 驗收", 78)
                accumulated_ctx += fb
                proj_dir = DELIVERY_ZONE / "projects" / task_id
                proj_dir.mkdir(parents=True, exist_ok=True)

                swarm_models = self.pool.plan_greedy_swarm()
                num_parallel = max(3, min(6, len(swarm_models)))

                spec_prompt = f"""{accumulated_ctx}
請將此軟體系統拆解為 {num_parallel} 個可獨立並行開發的 Python 模組檔案，以發揮多模型並行算力。
輸出嚴格 JSON：
{{
  "files": [
    {{"filename": "module_1.py", "instruction": "詳細類別與函式邏輯"}},
    {{"filename": "module_2.py", "instruction": "詳細類別與函式邏輯"}}
  ]
}}"""
                spec_raw = self.pool.generate(role="architect", prompt=spec_prompt, task_id=task_id, action_desc=f"拆解 {num_parallel} 個並行模組契約", format_json=True, num_ctx=8192)
                try:
                    proj_spec = json.loads(spec_raw)
                    file_list = proj_spec.get("files", [])
                except Exception:
                    file_list = [
                        {"filename": "core_engine.py", "instruction": desc},
                        {"filename": "battle_simulator.py", "instruction": "模擬戰鬥與數值平衡驗證"},
                        {"filename": "data_models.py", "instruction": "資料結構與時代設定"}
                    ]

                self.emit_log(task_id, f"⚡ [滿載多模型並行] 同時調動 {len(swarm_models)} 個模型槽位並行實作 {len(file_list)} 個 Python 模組...")
                generated_files = {}

                def _parallel_py_coder(idx: int, f_item: dict, model_name: str):
                    self.checkpoint(task_id)
                    fn = f_item["filename"]
                    p = f"""請實作 Python 模組 `{fn}`：\n規格要求：{f_item['instruction']}\n{accumulated_ctx[:2500]}\n只輸出純 Python 代碼，不含 Markdown。"""
                    code = self.pool.generate(
                        role="coder", prompt=p, task_id=task_id,
                        action_desc=f"並行實作 {fn}", num_ctx=8192, preferred_model=model_name
                    )
                    clean_c = code.replace("```python", "").replace("```", "").strip()
                    (proj_dir / fn).write_text(clean_c, encoding="utf-8")
                    return fn, clean_c

                with ThreadPoolExecutor(max_workers=len(file_list)) as ex:
                    futs = [
                        ex.submit(_parallel_py_coder, i, item, swarm_models[i % len(swarm_models)]["name"])
                        for i, item in enumerate(file_list)
                    ]
                    for fut in as_completed(futs):
                        fn, clean_c = fut.result()
                        generated_files[fn] = clean_c
                        self.emit_log(task_id, f"✔ 並行模組產出完成：{fn}")

                # TDD 驗收
                self.checkpoint(task_id, "Stage 5.2: 沙盒 TDD 驗證與三級主管升級除錯", 88)
                t_prompt = f"""請為以下專案檔案撰寫 `test_suite.py` 自動化測試：
{json.dumps(generated_files, ensure_ascii=False)}
要求：第 1~3 行固定為 `import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))`，使用 `assert` 驗證。只輸出純 Python 代碼。"""
                t_raw = self.pool.generate(role="coder", prompt=t_prompt, task_id=task_id, action_desc="建立 TDD 測試套件", num_ctx=8192)
                test_code = t_raw.replace("```python", "").replace("```", "").strip()
                test_file = proj_dir / "test_suite.py"
                test_file.write_text(test_code, encoding="utf-8")

                for tier in range(1, 4):
                    self.checkpoint(task_id)
                    run_res = self.sandbox.run_python_script(test_file)
                    if run_res["success"]:
                        self.emit_log(task_id, "✔ 專案代碼通過沙盒 TDD 驗證！")
                        break
                    fix_m = swarm_models[tier % len(swarm_models)]["name"]
                    self.emit_log(task_id, f"▲ 測試未過，啟動 Tier {tier} 修復 ({fix_m}): {run_res['stderr'][:140]}")
                    fix_p = f"""TDD 測試失敗。\n當前模組：{json.dumps(generated_files, ensure_ascii=False)}\n測試腳本：{test_code}\n錯誤：{run_res['stderr']}\n請輸出修正檔案：\n<<<FILE: 檔名.py>>>\n完整代碼\n<<<END_FILE>>>"""
                    fixed_raw = self.pool.generate(role="arbiter", prompt=fix_p, task_id=task_id, action_desc=f"Tier {tier} 修復", num_ctx=16384, preferred_model=fix_m)
                    for target_fn, c_body in re.findall(r"<<<FILE:\s*(.*?)>>>(.*?)<<<END_FILE>>>", fixed_raw, re.DOTALL):
                        target_fn = target_fn.strip()
                        c_clean = c_body.replace("```python", "").replace("```", "").strip()
                        if self.sandbox.is_valid_python(c_clean):
                            (proj_dir / target_fn).write_text(c_clean, encoding="utf-8")
                            if target_fn == "test_suite.py":
                                test_code = c_clean
                            else:
                                generated_files[target_fn] = c_clean

            # 6. 並行資安審查 + 數值平衡審查 + 綜合交付報告（3 模型同時並行）
            fb = self.checkpoint(task_id, "Stage 6: 多模型並行稽核與最終報告交付", 95)
            accumulated_ctx += fb
            swarm_end = self.pool.plan_greedy_swarm()
            self.emit_log(task_id, "⚡ [多模型並行收尾] 同時執行『資安稽核』、『架構審查』與『完整交付報告』...")

            with ThreadPoolExecutor(max_workers=3) as ex:
                m0 = swarm_end[0]["name"]
                m1 = swarm_end[1 % len(swarm_end)]["name"]
                m2 = swarm_end[2 % len(swarm_end)]["name"]
                fut_sec = ex.submit(self.pool.generate, "fast", f"請對本次產出進行資安與效能審查：\n{accumulated_ctx[:3000]}", "", task_id, "資安合規審查", None, 4096, False, 0.1, m0)
                fut_qa = ex.submit(self.pool.generate, "fast", f"請檢驗遊戲與系統規則完整性：\n{accumulated_ctx[:3000]}", "", task_id, "玩法與規則驗收", None, 4096, False, 0.2, m1)
                fut_doc = ex.submit(self.pool.generate, "architect", f"請撰寫完整專案交付與架構總結報告（Markdown）：\n{accumulated_ctx[:6000]}", "", task_id, "撰寫綜合交付報告", None, 8192, False, 0.2, m2)
                sec_report = fut_sec.result()
                qa_report = fut_qa.result()
                final_report = fut_doc.result()

            report_file = DELIVERY_ZONE / "reports" / f"{task_id}_final_delivery.md"
            report_file.write_text(f"{final_report}\n\n---\n### QA Verification\n{qa_report}\n\n---\n### Security Audit\n{sec_report}", encoding="utf-8")

            self.update_task_status(task_id, "completed")
            self.current_task_progress_pct = 100
            self.current_task_stage = "Completed"
            self.emit_log(task_id, f"🎉 任務 [{task_id}] 全部完成！所有檔案已歸檔至 03_delivery。", event_type="completed")

        except TaskTerminatedException:
            self.update_task_status(task_id, "pending")
            self.current_task_stage = "Terminated (Pending)"
            self.emit_log(task_id, "⏹ 任務已由使用者強制終止，狀態已保留為 pending（隨時可按『開始』續跑）。")
        except Exception as e:
            self.update_task_status(task_id, "failed")
            self.current_task_stage = "Failed"
            self.emit_log(task_id, f"❌ 任務異常中斷: {e}")

    def _assemble_playable_rts_html(self, title: str, assets: list, custom_js_bundle: str) -> str:
        """組裝 100% 可互動試玩的雙向推塔即時戰略遊戲 (含五時代、四兵種、砲台與第五關二階段機械王者 Boss)"""
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
    <span id="ui-gold">💰 黃金: 250</span>
    <span id="ui-exp">⭐ 經驗: 0 / 100</span>
  </div>
  <div class="controls">
    <button onclick="window.spawnUnit(0)">🛡 重裝肉盾 (40G)</button>
    <button onclick="window.spawnUnit(1)">🏹 高速遠程 (65G)</button>
    <button onclick="window.spawnUnit(2)">💥 反裝甲爆發 (100G)</button>
    <button onclick="window.spawnUnit(3)">✨ 群體控場輔助 (80G)</button>
    <button class="turret" onclick="window.buildTurret()">🗼 加裝主堡砲塔 (120G)</button>
    <button class="evolve" onclick="window.evolveEra()">⚡ 時代進化</button>
  </div>
</div>
<canvas id="gameCanvas"></canvas>

<script>
window.G = {{
  gold: 250, exp: 0, eraIdx: 0,
  eras: ['石器時代', '青銅時代', '中世紀鐵器', '工業火藥', '星際未來'],
  expThresholds: [100, 260, 550, 1000, 999999],
  playerBaseHp: 1500, playerMaxHp: 1500,
  enemyBaseHp: 1500, enemyMaxHp: 1500,
  units: [], projectiles: [], turrets: [0, 0],
  bossSpawned: false, gameOver: false, winner: ''
}};

const BASE_TEMPLATES = [
  {{ name: '近戰重裝', cost: 40, hp: 190, dmg: 16, range: 30, speed: 45, color: '#38bdf8', radius: 13 }},
  {{ name: '高射速遠程', cost: 65, hp: 95, dmg: 20, range: 140, speed: 55, color: '#22c55e', radius: 10 }},
  {{ name: '爆發反裝甲', cost: 100, hp: 135, dmg: 48, range: 100, speed: 38, color: '#f97316', radius: 14 }},
  {{ name: '增益控場', cost: 80, hp: 120, dmg: 18, range: 115, speed: 48, color: '#eab308', radius: 11 }}
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
    side: side, name: tpl.name,
    x: side === 'player' ? 75 : canvas.width - 75,
    y: canvas.height * 0.72 + (Math.random() * 14 - 7),
    hp: Math.round(tpl.hp * mult), maxHp: Math.round(tpl.hp * mult),
    dmg: Math.round(tpl.dmg * mult), range: tpl.range, speed: tpl.speed,
    color: side === 'player' ? tpl.color : '#ef4444',
    radius: tpl.radius, atkTimer: 0, isBoss: false
  }});
}};

window.buildTurret = function() {{
  const cost = 120 * (window.G.eraIdx + 1);
  if (window.G.gold >= cost && window.G.turrets[0] < 4) {{
    window.G.gold -= cost;
    window.G.turrets[0]++;
  }}
}};

window.evolveEra = function() {{
  const req = window.G.expThresholds[window.G.eraIdx];
  if (window.G.eraIdx < 4 && window.G.exp >= req) {{
    window.G.eraIdx++;
    window.G.playerMaxHp += 900;
    window.G.playerBaseHp = Math.min(window.G.playerMaxHp, window.G.playerBaseHp + 700);
    window.G.enemyMaxHp += 1000;
    window.G.enemyBaseHp = Math.min(window.G.enemyMaxHp, window.G.enemyBaseHp + 800);
  }}
}};

// 載入多模型並行生成的模組代碼
try {{
  {custom_js_bundle}
}} catch(e) {{ console.warn('Custom parallel bundle fallback:', e); }}

let lastTime = performance.now();
let goldTick = 0, aiTick = 0, turretTick = 0;

function coreUpdate(dt) {{
  if (window.G.gameOver) return;
  goldTick += dt;
  if (goldTick >= 1.0) {{
    window.G.gold += 10 * (window.G.eraIdx + 1);
    goldTick = 0;
  }}
  aiTick += dt;
  if (aiTick >= Math.max(1.6, 3.6 - window.G.eraIdx * 0.4)) {{
    window.spawnUnit(Math.floor(Math.random() * 4), 'enemy');
    aiTick = 0;
  }}

  const canvas = document.getElementById('gameCanvas');
  const leftBaseX = 55, rightBaseX = canvas.width - 55;

  turretTick += dt;
  if (turretTick >= 1.0) {{
    if (window.G.turrets[0] > 0) {{
      const target = window.G.units.find(u => u.side === 'enemy' && u.x < 350);
      if (target) {{
        target.hp -= 30 * window.G.turrets[0] * (window.G.eraIdx + 1);
        window.G.projectiles.push({{ x1: 60, y1: canvas.height*0.55, x2: target.x, y2: target.y, ttl: 0.15, color: '#38bdf8' }});
      }}
    }}
    turretTick = 0;
  }}

  for (let u of window.G.units) {{
    let blocked = false;
    const dir = u.side === 'player' ? 1 : -1;

    for (let other of window.G.units) {{
      if (other.side !== u.side && other.hp > 0) {{
        const dist = Math.abs(other.x - u.x);
        if (dist <= u.range) {{
          blocked = true;
          u.atkTimer += dt;
          if (u.atkTimer >= 0.8) {{
            other.hp -= u.dmg;
            u.atkTimer = 0;
            window.G.projectiles.push({{ x1: u.x, y1: u.y - 6, x2: other.x, y2: other.y - 6, ttl: 0.12, color: u.color }});
          }}
          break;
        }}
      }}
    }}

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

  window.G.units = window.G.units.filter(u => {{
    if (u.hp <= 0) {{
      if (u.side === 'enemy') {{
        window.G.gold += 28 * (window.G.eraIdx + 1);
        window.G.exp += 35;
        if (u.isBoss) {{
          window.G.gameOver = true;
          window.G.winner = '🎉 終極機械主宰已被徹底擊殺！達成完全勝利！';
        }}
      }}
      return false;
    }}
    return true;
  }});

  if (window.G.enemyBaseHp <= 0) {{
    if (window.G.eraIdx < 4 && !window.G.bossSpawned) {{
      window.G.eraIdx++;
      window.G.enemyMaxHp = 1600 * (window.G.eraIdx + 1);
      window.G.enemyBaseHp = window.G.enemyMaxHp;
    }} else if (!window.G.bossSpawned) {{
      window.G.bossSpawned = true;
      window.G.enemyBaseHp = 600;
      window.G.units.push({{
        side: 'enemy', name: '🤖 終極機械主宰 BOSS',
        x: rightBaseX - 20, y: canvas.height * 0.68,
        hp: 9000, maxHp: 9000, dmg: 200, range: 160, speed: 18,
        color: '#a855f7', radius: 34, atkTimer: 0, isBoss: true
      }});
    }} else {{
      window.G.gameOver = true;
      window.G.winner = '🎉 敵方主堡與終極機械王者徹底摧毀，完全勝利！';
    }}
  }}

  if (window.G.playerBaseHp <= 0) {{
    window.G.playerBaseHp = 0;
    window.G.gameOver = true;
    window.G.winner = '💀 我方防禦主堡被摧毀，戰役失敗！';
  }}

  window.G.projectiles.forEach(p => p.ttl -= dt);
  window.G.projectiles = window.G.projectiles.filter(p => p.ttl > 0);

  document.getElementById('ui-era').innerText = `時代: ${{window.G.eras[window.G.eraIdx]}} (關卡 ${{window.G.eraIdx+1}}/5)`;
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

  ctx.fillStyle = '#2563eb';
  ctx.fillRect(15, roadY - 95, 55, 110);
  ctx.fillStyle = '#fff';
  ctx.font = '12px sans-serif';
  ctx.fillText(`我方主堡 ${{window.G.playerBaseHp}}/${{window.G.playerMaxHp}}`, 10, roadY - 105);

  ctx.fillStyle = '#dc2626';
  ctx.fillRect(canvas.width - 70, roadY - 95, 55, 110);
  ctx.fillStyle = '#fff';
  ctx.fillText(`敵方主堡 ${{window.G.enemyBaseHp}}/${{window.G.enemyMaxHp}}`, canvas.width - 135, roadY - 105);

  for (let i = 0; i < window.G.turrets[0]; i++) {{
    ctx.fillStyle = '#34d399';
    ctx.fillRect(28, roadY - 115 - i * 18, 30, 14);
  }}

  for (let u of window.G.units) {{
    ctx.beginPath();
    ctx.arc(u.x, u.y, u.radius, 0, Math.PI * 2);
    ctx.fillStyle = u.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    const bw = u.radius * 2.2;
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(u.x - bw/2, u.y - u.radius - 10, bw, 4);
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(u.x - bw/2, u.y - u.radius - 10, bw * Math.max(0, u.hp / u.maxHp), 4);
  }}

  for (let p of window.G.projectiles) {{
    ctx.beginPath();
    ctx.moveTo(p.x1, p.y1);
    ctx.lineTo(p.x2, p.y2);
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }}

  if (typeof window.renderCustomVFX === 'function') {{
    try {{ window.renderCustomVFX(ctx, canvas); }} catch(e) {{}}
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

    # ================= 自適應佇列調度迴圈 =================
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
                        time.sleep(1.0)
                        continue

                    for t in pending:
                        if self.get_control_state().get("stop_requested"):
                            break
                        self.execute_omnipotent_task(t)
            finally:
                self.daemon_active = False
                self.current_task_id = "-"
                self.current_task_stage = "Idle"
                self.set_control_state("IDLE", pause_req=False, stop_req=False)

        threading.Thread(target=_loop, daemon=True).start()