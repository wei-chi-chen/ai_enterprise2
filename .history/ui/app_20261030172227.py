import json
import os
import time
import re
import asyncio
import tempfile
from pathlib import Path
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from config import DELIVERY_ZONE, FEEDBACK_DIR, UPLOAD_DIR, BACKLOG_FILE
from daemon import OmniEnterpriseDaemon

app = FastAPI(title="Omni-Autonomous AI Enterprise v4.0")
app.mount("/delivery", StaticFiles(directory=str(DELIVERY_ZONE)), name="delivery")

ws_clients = set()
main_loop = None

async def _send_ws_client(ws, message):
    try:
        await ws.send_text(message)
    except Exception:
        ws_clients.discard(ws)

def sync_broadcast(payload: dict):
    if main_loop and ws_clients:
        msg = json.dumps(payload, ensure_ascii=False)
        for ws in list(ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(_send_ws_client(ws, msg), main_loop)
            except RuntimeError:
                ws_clients.discard(ws)

daemon = OmniEnterpriseDaemon(broadcast_fn=sync_broadcast)

@app.middleware("http")
async def allow_delivery_asset_fetch(request, call_next):
    if request.url.path.startswith("/delivery/") and request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )
    response = await call_next(request)
    if request.url.path.startswith("/delivery/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def _next_task_id(tasks):
    task_ids = {
        task.get("task_id", "")
        for task in tasks
        if isinstance(task, dict) and re.fullmatch(r"TASK-\d{3,}", str(task.get("task_id", "")))
    }
    for area in DELIVERY_ZONE.iterdir():
        if area.is_dir():
            task_ids.update(
                child.name
                for child in area.iterdir()
                if child.is_dir() and re.fullmatch(r"TASK-\d{3,}", child.name)
            )
    task_ids.update(
        match.group(1)
        for path in UPLOAD_DIR.iterdir()
        if (match := re.match(r"(TASK-\d{3,})_", path.name))
    )
    task_ids.update(
        path.stem
        for path in FEEDBACK_DIR.iterdir()
        if re.fullmatch(r"TASK-\d{3,}", path.stem)
    )
    next_number = max((int(task_id[5:]) for task_id in task_ids), default=0) + 1
    return f"TASK-{next_number:03d}"


@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

@app.get("/api/init")
async def get_init_state():
    models = daemon.pool.refresh_models()
    for model in models:
        model["context_limit"] = daemon.pool.context_limit(model)
        model["context_limit_source"] = daemon.pool.context_limit_source(model)
    mem = daemon.pool.sync_realtime_memory()
    budget = daemon.pool.get_hardware_budget()
    ctrl = daemon.get_control_state()
    q_prog = daemon.get_queue_progress()
    tasks = daemon.get_all_tasks()

    # 自動檢查最後一個任務是否已有遊戲預覽與素材
    existing_preview = ""
    existing_preview_error = ""
    existing_assets = []
    existing_film = ""
    if tasks:
        last_tid = tasks[-1]["task_id"]
        g_dir = DELIVERY_ZONE / "games" / last_tid
        if (g_dir / "index.html").exists():
            validation = daemon.sandbox.validate_html_game(g_dir / "index.html")
            if validation["success"]:
                existing_preview = f"/delivery/games/{last_tid}/index.html?t={int(time.time())}"
            else:
                existing_preview_error = validation["stderr"]
        if (g_dir / "assets").exists():
            existing_assets = [p.name for p in (g_dir / "assets").iterdir() if p.is_file()]
        film_viewer = DELIVERY_ZONE / "films" / last_tid / "index.html"
        if film_viewer.exists():
            existing_film = f"/delivery/films/{last_tid}/index.html?t={int(time.time())}"

    return {
        "models": models,
        "ollama_version": daemon.pool.ollama_version,
        "context_capacity": daemon.pool.context_capacity("architect"),
        "memory": mem,
        "budget_gb": budget,
        "model_error": daemon.pool.last_error,
        "memory_error": daemon.pool.last_memory_error,
        "control": ctrl,
        "queue": q_prog,
        "tasks": tasks,
        "existing_preview": existing_preview,
        "existing_preview_error": existing_preview_error,
        "existing_film": existing_film,
        "existing_assets": existing_assets
    }

@app.get("/api/tasks/{task_id}/deliveries")
async def get_task_deliveries(
    task_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    search: str = Query("", max_length=500),
):
    return await asyncio.to_thread(
        _get_task_deliveries_sync, task_id, offset, limit, search
    )


def _get_task_deliveries_sync(
    task_id: str, offset: int, limit: int, search: str
):
    if not re.fullmatch(r"TASK-\d{3,}", task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID.")
    categories = ("reports", "cleaned_data", "projects", "games", "films", "illustrations")
    needle = search.casefold().strip()
    files = []
    for category in categories:
        root = DELIVERY_ZONE / category / task_id
        if not root.is_dir():
            continue
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for name in names:
                path = Path(current) / name
                if path.is_symlink() or name.startswith("."):
                    continue
                relative = path.relative_to(root).as_posix()
                if needle and needle not in relative.casefold() and needle not in category.casefold():
                    continue
                files.append({
                    "category": category,
                    "path": relative,
                    "url": f"/delivery/{category}/{task_id}/{quote(relative, safe='/')}",
                    "size": path.stat().st_size,
                })
    files.sort(key=lambda item: (categories.index(item["category"]), item["path"].casefold()))
    total = len(files)
    page = files[offset:offset + limit]
    grouped = []
    for category in categories:
        category_files = [item for item in page if item["category"] == category]
        if category_files:
            grouped.append({
                "category": category,
                "files": [{key: value for key, value in item.items() if key != "category"} for item in category_files],
            })
    return {
        "task_id": task_id,
        "categories": grouped,
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(page) if offset + len(page) < total else None,
    }


@app.post("/api/config/memory")
async def update_memory_budget(budget_gb: float = Form(..., gt=0, le=512)):
    daemon.pool.set_hardware_budget(budget_gb)
    try:
        swarm = daemon.pool.plan_greedy_swarm()
        models = [m["name"] for m in swarm]
        message = f"⚡ 記憶體上限已更新為 {budget_gb} GB！可調度 {len(swarm)} 個模型槽位：{models}"
    except RuntimeError as exc:
        models = []
        message = f"⚠ 記憶體上限已更新為 {budget_gb} GB，但目前無可調度模型：{exc}"
    daemon.emit_log("SYS", message)
    return {"status": "ok", "budget_gb": budget_gb, "models": models, "model_error": daemon.pool.last_error}

@app.post("/api/daemon/control")
async def control_daemon(action: str = Form(...)):
    if action == "start":
        daemon.start_queue_loop(resume_paused=True)
        return {"status": "RUNNING"}
    elif action == "pause":
        daemon.set_control_state("PAUSED", pause_req=True, stop_req=False)
        return {"status": "PAUSED"}
    elif action == "resume":
        daemon.set_control_state("RUNNING", pause_req=False, stop_req=False)
        if not daemon.daemon_active:
            daemon.start_queue_loop(resume_paused=True)
        return {"status": "RUNNING"}
    elif action == "stop":
        daemon.set_control_state("STOPPED", pause_req=False, stop_req=True)
        daemon.pool.unload_all_models()
        return {"status": "STOPPED"}
    return {"status": "UNKNOWN"}

@app.post("/api/tasks/create")
async def create_task(
    title: str = Form(..., min_length=1, max_length=200),
    description: str = Form(..., min_length=1, max_length=20000),
    iterations: int = Form(3, ge=1, le=20),
    priority: int = Form(0, ge=-100, le=100),
    file: UploadFile = File(None)
):
    upload_name = ""
    upload_data = b""
    if file and file.filename:
        upload_name = Path(file.filename.replace("\\", "/")).name
        if not upload_name:
            raise HTTPException(status_code=400, detail="Invalid upload filename.")
        upload_data = await file.read(25 * 1024 * 1024 + 1)
        if len(upload_data) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Attachments must be 25 MB or smaller.")

    with daemon.backlog_lock:
        tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8")) if BACKLOG_FILE.exists() else []
        task_id = _next_task_id(tasks)

        saved_file = ""
        if upload_name:
            dest = UPLOAD_DIR / f"{task_id}_{upload_name}"
            dest.write_bytes(upload_data)
            saved_file = str(dest)

        new_task = {
            "task_id": task_id,
            "title": title,
            "description": description,
            "iterations": iterations,
            "priority": priority,
            "attached_file": saved_file,
            "status": "pending"
        }
        tasks.append(new_task)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=BACKLOG_FILE.parent,
                prefix=f".{BACKLOG_FILE.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(tasks, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, BACKLOG_FILE)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    daemon.emit_log(task_id, f"📥 新全能任務已加入佇列：{title}")
    daemon.start_queue_loop()
    return {"status": "queued", "task_id": task_id}

@app.post("/api/tasks/feedback")
async def submit_feedback(task_id: str = Form(...), feedback: str = Form(...)):
    if not re.fullmatch(r"TASK-\d{3,}", task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID.")
    fb_file = FEEDBACK_DIR / f"{task_id}.json"
    items = json.loads(fb_file.read_text(encoding="utf-8")) if fb_file.exists() else []
    items.append({"time": time.strftime("%H:%M:%S"), "text": feedback})
    fb_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    daemon.emit_log(task_id, f"📨 收到使用者新意見（已排入佇列，共 {len(items)} 則待下一階段讀取）")
    return {"status": "queued", "count": len(items)}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)

HTML_UI = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title id="page-title" data-i18n="pageTitle">Omni-Autonomous AI Enterprise v4.0</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #090d16; color: #f1f5f9; font-family: 'Segoe UI', monospace; display: flex; flex-direction: column; height: 100vh; }
  header { background: #1e293b; padding: 10px 18px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; gap: 12px; flex-wrap: wrap; }
  .ctrl-group { display: flex; align-items: center; gap: 8px; }
  .btn { padding: 7px 13px; border-radius: 6px; border: none; font-weight: bold; cursor: pointer; color: #fff; font-size: 13px; }
  .btn-start { background: #16a34a; }
  .btn-pause { background: #d97706; }
  .btn-resume { background: #2563eb; }
  .btn-stop { background: #dc2626; }
  .btn-lang { background: #475569; }
  .kpi-bar { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; padding: 10px 18px; background: #0f172a; border-bottom: 1px solid #1e293b; }
  .kpi-card { background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 8px 12px; }
  .kpi-title { font-size: 12px; color: #94a3b8; }
  .kpi-val { font-size: 18px; font-weight: bold; color: #38bdf8; margin-top: 4px; }
  .prog-bg { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; margin-top: 6px; overflow: hidden; }
  .prog-fill { height: 100%; background: #22c55e; width: 0%; transition: width 0.3s; }
  .container { display: grid; grid-template-columns: 350px 1fr 500px; flex: 1; overflow: hidden; }
  .col { padding: 12px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; overflow-y: auto; gap: 10px; }
  .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; }
  h3 { margin: 0 0 8px 0; font-size: 14px; color: #38bdf8; border-bottom: 1px solid #1e293b; padding-bottom: 5px; }
  input[type="text"], textarea, input[type="number"] { width: 100%; background: #020617; border: 1px solid #334155; color: #fff; padding: 7px; border-radius: 5px; margin-bottom: 6px; font-size: 13px; }
  #log-stream { flex: 1; background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; overflow-y: auto; font-size: 13px; white-space: pre-wrap; }
  .log-item { margin-bottom: 8px; border-bottom: 1px solid #0f172a; padding-bottom: 4px; }
  .model-progress-card { background:#0f172a; border:1px solid #334155; border-radius:6px; padding:8px; margin-bottom:8px; }
  .model-progress-meta { display:flex; justify-content:space-between; gap:8px; font-size:11px; color:#cbd5e1; margin-top:5px; }
  .model-progress-track { height:7px; background:#1e293b; border-radius:999px; overflow:hidden; margin-top:6px; }
  .model-progress-fill { height:100%; width:0; background:linear-gradient(90deg,#0ea5e9,#22c55e); transition:width .5s ease; }
  .preview-placeholder { color: #94a3b8; font-size: 13px; }
  .game-stage[hidden] { display: none; }
  .game-stage { position: fixed; inset: 0; z-index: 1000; display: flex; flex-direction: column; background: #020617; }
  .game-stage-toolbar { min-height: 48px; display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 6px 12px; background: #0f172a; border-bottom: 1px solid #334155; }
  .game-stage-actions { display: flex; gap: 8px; }
  .game-stage iframe { flex: 1; width: 100%; height: 0; min-height: 0; background: #000; border: 0; }
  .worker-pill { font-size: 12px; background: #1e1b4b; border: 1px solid #6366f1; padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; }
  .task-row { font-size: 12px; padding: 6px 8px; border-radius: 4px; background: #020617; border-left: 4px solid #facc15; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center; }
  .task-row.completed { border-left-color: #22c55e; }
  .task-row.processing { border-left-color: #38bdf8; background: #0f172a; }
  .task-row.failed { border-left-color: #ef4444; }
  @media (max-width: 1150px) {
    .kpi-bar { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .container { grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.5fr); overflow-y: auto; }
    .container > .col:last-child { grid-column: 1 / -1; }
  }
  @media (max-width: 700px) {
    body { height: auto; min-height: 100vh; overflow: auto; }
    .kpi-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 10px; }
    .container { display: flex; flex-direction: column; overflow: visible; }
    .col { overflow: visible; border-right: 0; border-bottom: 1px solid #1e293b; }
    header { align-items: flex-start; }
    .ctrl-group { flex-wrap: wrap; }
  }
</style>
</head>
<body>
<header>
  <div><strong id="ui-app-title" data-i18n="appTitle">🌐 本地全能自主 AI 企業系統 v4.0</strong> <span id="ui-state-badge" style="color:#22c55e;font-weight:bold;">[IDLE]</span></div>
  <div class="ctrl-group">
    <span id="ui-mem-label" style="font-size:13px;" data-i18n="memLabel">記憶體上限 (GB):</span>
    <input type="number" id="mem-input" value="40" min="4" max="256" step="1" style="width:70px;margin:0;" />
    <button class="btn btn-resume" onclick="applyMemory()" id="ui-btn-apply" data-i18n="apply">套用</button>
    <span id="mem-live" style="font-size:13px;color:#38bdf8;margin:0 8px;">VRAM/RAM: 0.0 / 40.0 GB</span>
    <span id="context-live" style="font-size:12px;color:#a78bfa;margin:0 8px;"></span>
    <button class="btn btn-start" onclick="ctrlDaemon('start')" id="ui-btn-start" data-i18n="start">▶ 開始 / 續跑未完成任務</button>
    <button class="btn btn-pause" onclick="ctrlDaemon('pause')" id="ui-btn-pause" data-i18n="pause">⏸ 暫停</button>
    <button class="btn btn-resume" onclick="ctrlDaemon('resume')" id="ui-btn-resume" data-i18n="resume">⏯ 繼續</button>
    <button class="btn btn-stop" onclick="ctrlDaemon('stop')" id="ui-btn-stop" data-i18n="stop">⏹ 終止</button>
    <button class="btn btn-lang" onclick="toggleLang()">🌐 中文 / EN</button>
  </div>
</header>

<div class="kpi-bar">
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-total-time" data-i18n="totalTime">⏱ 總執行耗時 (真實物理時鐘)</div>
    <div class="kpi-val" id="val-total-time">00:00:00</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-task-time" data-i18n="taskTime">⏳ 當前任務耗時</div>
    <div class="kpi-val" id="val-task-time">00:00:00</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-estimated-total" data-i18n="estimatedTotalTime">⏱ 預估總完成時間</div>
    <div class="kpi-val" id="val-estimated-total">計算中</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-task-prog" data-i18n="taskProgress">🎯 當前任務進度</div>
    <div class="kpi-val" id="val-task-prog">0% (<span data-i18n="idle">閒置</span>)</div>
    <div class="prog-bg"><div class="prog-fill" id="bar-task-prog"></div></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-queue-prog" data-i18n="queueProgress">📊 總任務佇列進度</div>
    <div class="kpi-val" id="val-queue-prog">0 / 0 (0%)</div>
    <div class="prog-bg"><div class="prog-fill" id="bar-queue-prog"></div></div>
  </div>
</div>

<div class="container">
  <div class="col">
    <!-- 新增：隨時可見的未完成與待辦任務佇列看板 -->
    <div class="card">
      <h3 id="ui-backlog-title" data-i18n="backlogTitle">📋 任務佇列看板 (含先前未完成任務)</h3>
      <input type="text" id="backlog-search" data-i18n-placeholder="taskSearch" placeholder="搜尋任務..." oninput="filterTaskBacklog()" />
      <select id="backlog-status" onchange="filterTaskBacklog()" style="width:100%;margin-bottom:6px;">
        <option value="all" data-i18n="allStatuses">所有狀態</option>
        <option value="pending" data-i18n="pending">⏳ 未完成 (待續跑)</option>
        <option value="processing" data-i18n="processing">🔥 執行中</option>
        <option value="completed" data-i18n="completed">✅ 完成</option>
        <option value="failed" data-i18n="failed">❌ 失敗</option>
        <option value="paused" data-i18n="paused">⏸ 已暫停</option>
      </select>
      <div id="backlog-box" style="max-height:130px;overflow-y:auto;" data-i18n="loading">載入中...</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:5px;font-size:11px;">
        <button type="button" onclick="changeTaskPage(-1)" data-i18n="previousPage">上一頁</button>
        <span id="backlog-page-label"></span>
        <button type="button" onclick="changeTaskPage(1)" data-i18n="nextPage">下一頁</button>
      </div>
    </div>

    <div class="card">
      <h3 id="ui-new-task-title" data-i18n="newTaskTitle">🚀 自主拆解任務與預算內多模型並行</h3>
      <input type="text" id="t-title" placeholder="輸入任務標題或問題摘要..." data-i18n-placeholder="titlePlaceholder" />
      <textarea id="t-desc" rows="3" placeholder="輸入任何需求，系統將自動調動所有可用模型塞滿記憶體並行執行..." data-i18n-placeholder="descriptionPlaceholder"></textarea>
      <label style="font-size:12px;color:#94a3b8;" id="ui-iter-label" data-i18n="iterationsLabel">最大自主演化輪數：</label>
      <input type="number" id="t-iter" value="3" min="1" max="20" />
      <label style="font-size:12px;color:#94a3b8;" id="ui-priority-label" data-i18n="priorityLabel">任務優先級 (-100–100)：</label>
      <input type="number" id="t-priority" value="0" min="-100" max="100" step="1" />
      <label style="font-size:12px;color:#94a3b8;" id="ui-file-label" data-i18n="fileLabel">附加檔案 (圖片 / CSV / Excel / 代碼)：</label>
      <input type="file" id="t-file" style="margin-bottom:6px;font-size:12px;" />
      <button class="btn btn-start" style="width:100%;" onclick="submitTask()" id="ui-btn-submit" data-i18n="submitTask">加入佇列並自主執行</button>
    </div>

    <div class="card">
      <h3 id="ui-fb-title" data-i18n="feedbackTitle">💡 隨時注入新意見 (階段完成後讀取)</h3>
      <input type="text" id="fb-task" value="TASK-001" />
      <textarea id="fb-text" rows="2" placeholder="執行中或暫停時皆可輸入新要求..." data-i18n-placeholder="feedbackPlaceholder"></textarea>
      <button class="btn btn-resume" style="width:100%;" onclick="sendFeedback()" id="ui-btn-fb" data-i18n="sendFeedback">送出意見</button>
      <div id="fb-status" style="font-size:12px;color:#22c55e;margin-top:4px;"></div>
    </div>

    <div class="card">
      <h3 id="ui-models-title" data-i18n="modelsTitle">🧠 本機 Ollama 模型 (依模型能力自動派工)</h3>
      <div id="model-status" role="status" style="font-size:12px;color:#fca5a5;"></div>
      <div id="model-list" style="font-size:12px;"></div>
    </div>
  </div>

  <div class="col">
    <h3 id="ui-workers-title" data-i18n="workersTitle">⚡ 預算內可用的並行模型工作槽</h3>
    <div id="active-workers-box"><div style="font-size:12px;color:#64748b;" data-i18n="workerIdle">閒置 - 請點擊右上角「▶ 開始 / 續跑未完成任務」</div></div>
    <h3 id="ui-logs-title" data-i18n="logsTitle">📡 系統決策、多模型分治與演化日誌</h3>
    <div id="log-stream">
      <div id="model-progress-stream" aria-live="polite"></div>
      <div id="log-messages"></div>
    </div>
  </div>

  <div class="col">
    <h3 id="ui-preview-title" data-i18n="previewTitle">🎮 遊戲即時試玩預覽 / 交付預覽</h3>
    <div class="preview-placeholder">
      <p id="preview-empty" data-i18n="previewEmpty">尚未載入遊戲。遊戲完成後會自動以全螢幕版面開啟。</p>
      <button class="btn btn-resume" id="game-open-btn" onclick="openGamePreview()" hidden data-i18n="openGame">開啟遊戲</button>
    </div>
    <div class="card">
      <h3 id="ui-assets-title" data-i18n="assetsTitle">🎨 自主合成素材庫 (Sprite & WAV)</h3>
      <div id="asset-box" style="font-size:12px;color:#94a3b8;max-height:180px;overflow-y:auto;" data-i18n="emptyAssets">-</div>
    </div>
    <div class="card">
      <h3 data-i18n="deliveryBrowser">📁 任務交付瀏覽器</h3>
      <select id="delivery-task-select" aria-label="Task deliveries" onchange="loadTaskDeliveries(this.value)" style="width:100%;margin-bottom:8px;"></select>
      <input type="text" id="delivery-search" data-i18n-placeholder="deliverySearch" placeholder="搜尋交付檔案..." oninput="searchTaskDeliveries()" />
      <div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0;font-size:11px;">
        <button type="button" id="delivery-prev" onclick="changeDeliveryPage(-1)" data-i18n="previousPage">上一頁</button>
        <span id="delivery-page-label"></span>
        <button type="button" id="delivery-next" onclick="changeDeliveryPage(1)" data-i18n="nextPage">下一頁</button>
      </div>
      <div id="delivery-tree" style="max-height:190px;overflow:auto;font-size:12px;"></div>
      <div id="delivery-preview" style="max-height:360px;overflow:auto;margin-top:8px;"></div>
    </div>
  </div>
</div>

<section id="game-stage" class="game-stage" hidden aria-label="遊戲預覽" data-i18n-aria-label="previewTitle">
  <div class="game-stage-toolbar">
    <strong data-i18n="previewTitle">🎮 遊戲即時試玩預覽 / 交付預覽</strong>
    <div class="game-stage-actions">
      <button class="btn btn-resume" id="ui-btn-fullscreen" onclick="toggleGameFullscreen()" aria-label="全螢幕"><span id="ui-btn-fullscreen-label" data-i18n="fullscreen">全螢幕</span></button>
      <button class="btn btn-stop" onclick="closeGamePreview()" data-i18n="closeGame">關閉遊戲</button>
    </div>
  </div>
  <iframe id="game-frame" src="about:blank" title="遊戲預覽" data-i18n-title="previewTitle" sandbox="allow-scripts allow-downloads" allow="fullscreen"></iframe>
</section>

<script>
  let lang = localStorage.getItem('omni-language') === 'en' ? 'en' : 'zh';
  let gamePreviewUrl = '';
  let backlogPage = 0;
  let deliveryOffset = 0;
  let deliveryRequestId = 0;
  let deliverySearchTimer;
  const taskPageSize = 25;
  const deliveryPageSize = 100;
  const translations = {
    zh: {
      pageTitle: '本地全能自主 AI 企業系統 v4.0',
      appTitle: '🌐 本地全能自主 AI 企業系統 v4.0', memLabel: '記憶體上限 (GB):', apply: '套用',
      start: '▶ 開始 / 續跑未完成任務', pause: '⏸ 暫停', resume: '⏯ 繼續', stop: '⏹ 終止',
      totalTime: '⏱ 總執行耗時 (真實物理時鐘)', taskTime: '⏳ 當前任務耗時',
      estimatedTotalTime: '⏱ 預估總完成時間 (含已耗時間)', modelElapsed: '已執行', modelEta: '預估剩餘',
      modelProgress: '進度', estimating: '計算中', tokenRate: '估算速度',
      taskProgress: '🎯 當前任務進度', queueProgress: '📊 總任務佇列進度',
      backlogTitle: '📋 任務佇列看板 (含先前未完成任務)', loading: '載入中...',
      taskSearch: '搜尋任務...', allStatuses: '所有狀態', noMatchingTasks: '沒有符合條件的任務',
      previousPage: '上一頁', nextPage: '下一頁', pageOf: '第 {page} / {pages} 頁',
      deliverySearch: '搜尋交付檔案...',
      priorityLabel: '任務優先級 (-100–100)：',
      noTasks: '目前無任務', completed: '✅ 完成', processing: '🔥 執行中',
      failed: '❌ 失敗', paused: '⏸ 已暫停',
      pending: '⏳ 未完成 (待續跑)', newTaskTitle: '🚀 自主拆解任務與預算內多模型並行',
      titlePlaceholder: '輸入任務標題或問題摘要...',       descriptionPlaceholder: '描述任何問題或需求（例如研究分析、爬蟲、資料清理、圖片、遊戲、電影或軟體），系統會自動選擇處理流程...',
      iterationsLabel: '最大自主演化輪數（增加可提升成果品質）：', fileLabel: '附加檔案 (圖片 / CSV / Excel / 代碼)：',
      submitTask: '加入佇列並自主執行', feedbackTitle: '💡 隨時注入新意見 (階段完成後讀取)',
      feedbackPlaceholder: '執行中或暫停時皆可輸入新要求...', sendFeedback: '送出意見',
      modelsTitle: '🧠 本機 Ollama 模型 (依模型能力自動派工)',
      capabilitiesUnavailable: '能力資訊不可用',
      modelsUnavailable: '無法使用 Ollama 模型：',
      memoryUnavailable: '無法讀取 Ollama 記憶體用量：',
      contextCapacity: '可用 Context 上限',
      ollamaVersion: 'Ollama',
      workersTitle: '⚡ 預算內可用的並行模型工作槽',
      workerIdle: '閒置 - 請點擊右上角「▶ 開始 / 續跑未完成任務」',
      workerIdleShort: '閒置 (點擊右上角「▶ 開始」立即調動多模型執行佇列中之未完成任務)',
      logsTitle: '📡 系統決策、多模型分治與演化日誌',
      previewTitle: '🎬 電影 / 🎮 遊戲成果預覽',
      previewEmpty: '尚未載入成果。電影或遊戲完成後會自動以全視窗版面開啟。',
      deliveryBrowser: '📁 任務交付瀏覽器',
      selectDeliveryTask: '選擇任務以瀏覽交付成果',
      noDeliveryFiles: '此任務尚無可瀏覽的交付檔案。',
      reports: '研究與報告', cleaned_data: '清理資料', projects: '軟體專案',
      games: '遊戲', films: '電影', illustrations: '圖片',
      previewLoadError: '無法載入交付內容：',
      previewTooLarge: '檔案超過線上預覽大小限制，請下載檢視。',
      previewInvalid: '既有遊戲檔案無效，請重新執行遊戲任務：',
      openGame: '開啟成果', fullscreen: '全螢幕', exitFullscreen: '退出全螢幕',
      openImage: '開啟圖片成果',
      closeGame: '關閉遊戲', assetsTitle: '🎨 自主合成素材庫 (Sprite & WAV)',
      emptyAssets: '-', idle: '閒置', enterDescription: '請輸入需求描述',
      feedbackSent: '✔ 已成功送出意見至 {taskId} 佇列！'
    },
    en: {
      pageTitle: 'Local Omnipotent Autonomous AI Enterprise v4.0',
      appTitle: '🌐 Local Omnipotent Autonomous AI Enterprise v4.0', memLabel: 'Memory budget (GB):', apply: 'Apply',
      start: '▶ Start / Resume unfinished tasks', pause: '⏸ Pause', resume: '⏯ Resume', stop: '⏹ Stop',
      totalTime: '⏱ Total elapsed time (wall clock)', taskTime: '⏳ Current task elapsed time',
      estimatedTotalTime: '⏱ Estimated total completion time (elapsed included)',
      modelElapsed: 'Elapsed', modelEta: 'Est. remaining', modelProgress: 'Progress',
      estimating: 'Estimating...', tokenRate: 'Est. speed',
      taskProgress: '🎯 Current task progress', queueProgress: '📊 Overall queue progress',
      backlogTitle: '📋 Task backlog (including unfinished tasks)', loading: 'Loading...',
      taskSearch: 'Search tasks...', allStatuses: 'All statuses', noMatchingTasks: 'No matching tasks',
      previousPage: 'Previous', nextPage: 'Next', pageOf: 'Page {page} of {pages}',
      deliverySearch: 'Search delivery files...',
      priorityLabel: 'Task priority (-100–100):',
      noTasks: 'No tasks', completed: '✅ Completed', processing: '🔥 Running',
      failed: '❌ Failed', paused: '⏸ Paused',
      pending: '⏳ Pending / resumable', newTaskTitle: '🚀 Decompose tasks and parallelize within budget',
      titlePlaceholder: 'Enter a task title or brief summary...', descriptionPlaceholder: 'Describe any problem or request (e.g. research, crawler, data cleaning, image, game, film, or software)...',
      iterationsLabel: 'Maximum quality evolution rounds:', fileLabel: 'Attachment (image / CSV / Excel / code):',
      submitTask: 'Add to queue and run', feedbackTitle: '💡 Send feedback at any time (read after each stage)',
      feedbackPlaceholder: 'Add requirements while running or paused...', sendFeedback: 'Send feedback',
      modelsTitle: '🧠 Local Ollama models (dispatched by capabilities)',
      workersTitle: '⚡ Parallel model workers within the memory budget',
      capabilitiesUnavailable: 'capabilities unavailable',
      modelsUnavailable: 'Ollama models unavailable:',
      memoryUnavailable: 'Ollama memory usage unavailable:',
      contextCapacity: 'Available Context ceiling',
      ollamaVersion: 'Ollama',
      workerIdle: 'Idle - Click “▶ Start / Resume unfinished tasks” in the upper-right',
      workerIdleShort: 'Idle (click “▶ Start” to resume queued tasks with available models)',
      logsTitle: '📡 System decisions, parallel work, and progress log',
      previewTitle: '🎬 Film / 🎮 Game preview',
      previewEmpty: 'No result loaded. Completed films and games open automatically in a full-window view.',
      deliveryBrowser: '📁 Task delivery browser',
      selectDeliveryTask: 'Select a task to browse its deliverables',
      noDeliveryFiles: 'This task has no browsable delivery files.',
      reports: 'Research & reports', cleaned_data: 'Cleaned data', projects: 'Software projects',
      games: 'Games', films: 'Films', illustrations: 'Images',
      previewLoadError: 'Could not load delivery content: ',
      previewTooLarge: 'File exceeds the online preview size limit. Download it to inspect.',
      openImage: 'Open generated image',
      previewInvalid: 'The existing game file is invalid. Run the game task again:',
      openGame: 'Open result', fullscreen: 'Fullscreen', exitFullscreen: 'Exit fullscreen',
      closeGame: 'Close game', assetsTitle: '🎨 Generated assets (Sprite & WAV)',
      emptyAssets: '-', idle: 'Idle', enterDescription: 'Please enter a task description',
      feedbackSent: '✔ Feedback sent to the {taskId} queue!'
    }
  };
  function tr(key) {
    return translations[lang][key] || translations.zh[key] || key;
  }
  function applyTranslations() {
    document.documentElement.lang = lang === 'zh' ? 'zh-TW' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = tr(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = tr(el.dataset.i18nPlaceholder);
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
      el.setAttribute('aria-label', tr(el.dataset.i18nAriaLabel));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
      el.title = tr(el.dataset.i18nTitle);
    });
    document.getElementById('ui-btn-fullscreen')?.setAttribute('aria-label', tr('fullscreen'));
    renderTaskBacklog(window.currentTasks || []);
    renderWorkers(window.currentWorkers || []);
    document.getElementById('ui-btn-fullscreen-label')?.replaceChildren(tr(document.fullscreenElement ? 'exitFullscreen' : 'fullscreen'));
    syncPreviewLanguage();
  }
  function fmtSec(s) {
    if (s === null || s === undefined || s === '' || !Number.isFinite(Number(s)) || Number(s) < 0) {
      return tr('estimating');
    }
    s = Math.floor(Number(s));
    const h = String(Math.floor(s / 3600)).padStart(2, '0');
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const sec = String(s % 60).padStart(2, '0');
    return `${h}:${m}:${sec}`;
  }

  function renderTaskBacklog(tasks) {
    window.currentTasks = tasks || [];
    renderDeliveryTaskOptions(window.currentTasks);
    const box = document.getElementById('backlog-box');
    const allTasks = window.currentTasks;
    const search = document.getElementById('backlog-search').value.trim().toLocaleLowerCase();
    const statusFilter = document.getElementById('backlog-status').value;
    const filtered = allTasks.filter(task => {
      const taskStatus = task.status || 'pending';
      const matchesStatus = statusFilter === 'all' || taskStatus === statusFilter;
      const text = `${task.task_id || ''} ${task.title || ''}`.toLocaleLowerCase();
      return matchesStatus && (!search || text.includes(search));
    });
    const pageCount = Math.max(1, Math.ceil(filtered.length / taskPageSize));
    backlogPage = Math.min(backlogPage, pageCount - 1);
    const visibleTasks = filtered.slice(backlogPage * taskPageSize, (backlogPage + 1) * taskPageSize);
    document.getElementById('backlog-page-label').textContent =
      tr('pageOf').replace('{page}', String(backlogPage + 1)).replace('{pages}', String(pageCount));
    document.querySelector('[onclick="changeTaskPage(-1)"]').disabled = backlogPage === 0;
    document.querySelector('[onclick="changeTaskPage(1)"]').disabled = backlogPage >= pageCount - 1;
    if (!allTasks.length || !filtered.length) {
      const empty = document.createElement('div');
      empty.style.cssText = 'font-size:12px;color:#64748b;';
      empty.textContent = allTasks.length ? tr('noMatchingTasks') : tr('noTasks');
      box.replaceChildren(empty);
      return;
    }
    const rows = visibleTasks.map(t => {
      const st = t.status || 'pending';
      const badge = tr(['completed', 'processing', 'failed', 'paused'].includes(st) ? st : 'pending');
      const row = document.createElement('div');
      row.className = `task-row ${['completed', 'processing', 'failed'].includes(st) ? st : ''}`;
      const name = document.createElement('div');
      const id = document.createElement('strong');
      id.textContent = `[${t.task_id}]`;
      name.append(id, document.createTextNode(` ${t.title || ''}`));
      const status = document.createElement('div');
      status.style.cssText = 'color:#38bdf8;font-weight:bold;';
      status.textContent = badge;
      row.append(name, status);
      return row;
    });
    box.replaceChildren(...rows);
  }

  function filterTaskBacklog() {
    backlogPage = 0;
    renderTaskBacklog(window.currentTasks || []);
  }

  function changeTaskPage(direction) {
    backlogPage = Math.max(0, backlogPage + direction);
    renderTaskBacklog(window.currentTasks || []);
  }

  function renderDeliveryTaskOptions(tasks) {
    const select = document.getElementById('delivery-task-select');
    const previous = select.value || window.deliveryTaskId || '';
    const recentTasks = (tasks || []).slice(-100);
    if (previous && !recentTasks.some(task => task.task_id === previous)) {
      const previousTask = (tasks || []).find(task => task.task_id === previous);
      if (previousTask) recentTasks.unshift(previousTask);
    }
    const options = recentTasks.map(task => {
      const option = document.createElement('option');
      option.value = task.task_id;
      option.textContent = `[${task.task_id}] ${task.title || ''}`;
      return option;
    });
    const currentIds = Array.from(select.options || []).map(option => option.value);
    const nextIds = options.map(option => option.value);
    if (JSON.stringify(currentIds) === JSON.stringify(nextIds)) return;
    select.replaceChildren(...options);
    const chosen = nextIds.includes(previous) ? previous : nextIds.at(-1);
    if (!chosen) {
      window.deliveryTaskId = '';
      return;
    }
    select.value = chosen;
    loadTaskDeliveries(chosen);
  }

  function escapePreviewText(value) {
    return String(value).replace(/[&<>"]/g, char => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
    })[char]);
  }

  function renderMarkdownPreview(markdown) {
    const lines = String(markdown).replace(/\\r\\n/g, '\\n').split('\\n');
    const blocks = [];
    let paragraph = [];
    let codeLines = null;
    const flushParagraph = () => {
      if (paragraph.length) {
        blocks.push(`<p>${paragraph.map(escapePreviewText).join('<br>')}</p>`);
        paragraph = [];
      }
    };
    for (const line of lines) {
      if (line.trim().startsWith('```')) {
        if (codeLines) {
          blocks.push(`<pre><code>${codeLines.map(escapePreviewText).join('\\n')}</code></pre>`);
          codeLines = null;
        } else {
          flushParagraph();
          codeLines = [];
        }
      } else if (codeLines) {
        codeLines.push(line);
      } else if (/^#{1,6}[ ]/.test(line)) {
        flushParagraph();
        const depth = line.match(/^#+/)[0].length;
        blocks.push(`<h${depth}>${escapePreviewText(line.replace(/^#{1,6}[ ]+/, ''))}</h${depth}>`);
      } else if (!line.trim()) {
        flushParagraph();
      } else {
        paragraph.push(line);
      }
    }
    if (codeLines) blocks.push(`<pre><code>${codeLines.map(escapePreviewText).join('\\n')}</code></pre>`);
    flushParagraph();
    return blocks.join('');
  }

  function parseCsvPreview(text) {
    const rows = [];
    let row = [], cell = '', quoted = false;
    for (let index = 0; index < text.length && rows.length < 101; index++) {
      const char = text[index];
      if (char === '"' && quoted && text[index + 1] === '"') {
        cell += '"'; index++;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === ',' && !quoted) {
        row.push(cell); cell = '';
      } else if ((char === '\\n' || char === '\\r') && !quoted) {
        if (char === '\\r' && text[index + 1] === '\\n') index++;
        row.push(cell); rows.push(row); row = []; cell = '';
      } else {
        cell += char;
      }
    }
    if (cell || row.length) {
      row.push(cell);
      rows.push(row);
    }
    return rows;
  }

  function renderCsvPreview(text) {
    const rows = parseCsvPreview(text);
    const table = document.createElement('table');
    table.style.cssText = 'border-collapse:collapse;font-size:11px;white-space:nowrap;';
    rows.slice(0, 101).forEach((row, rowIndex) => {
      const tr = document.createElement('tr');
      row.slice(0, 30).forEach(value => {
        const cell = document.createElement(rowIndex === 0 ? 'th' : 'td');
        cell.textContent = value;
        cell.style.cssText = 'border:1px solid #334155;padding:4px 6px;max-width:220px;overflow:hidden;text-overflow:ellipsis;';
        tr.appendChild(cell);
      });
      table.appendChild(tr);
    });
    return table;
  }

  async function loadTaskDeliveries(taskId) {
    const changedTask = taskId !== window.deliveryTaskId;
    window.deliveryTaskId = taskId;
    deliveryOffset = 0;
    if (changedTask) document.getElementById('delivery-search').value = '';
    return fetchTaskDeliveries(taskId, deliveryOffset);
  }

  async function fetchTaskDeliveries(taskId, offset) {
    const tree = document.getElementById('delivery-tree');
    const preview = document.getElementById('delivery-preview');
    const requestId = ++deliveryRequestId;
    tree.replaceChildren();
    preview.replaceChildren();
    if (!taskId) {
      document.getElementById('delivery-page-label').textContent = '';
      document.getElementById('delivery-prev').disabled = true;
      document.getElementById('delivery-next').disabled = true;
      return;
    }
    try {
      const search = document.getElementById('delivery-search').value.trim();
      const params = new URLSearchParams({ offset: String(offset), limit: String(deliveryPageSize), search });
      const response = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/deliveries?${params}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || response.statusText);
      if (requestId !== deliveryRequestId || taskId !== window.deliveryTaskId) return;
      const pageCount = Math.max(1, Math.ceil(payload.total / deliveryPageSize));
      const page = Math.floor(payload.offset / deliveryPageSize) + 1;
      document.getElementById('delivery-page-label').textContent =
        `${payload.total} · ${tr('pageOf').replace('{page}', String(page)).replace('{pages}', String(pageCount))}`;
      document.getElementById('delivery-prev').disabled = payload.offset === 0;
      document.getElementById('delivery-next').disabled = payload.next_offset === null;
      if (!payload.categories.length) {
        tree.textContent = tr('noDeliveryFiles');
        return;
      }
      payload.categories.forEach(group => {
        const category = document.createElement('details');
        const heading = document.createElement('summary');
        heading.textContent = `${tr(group.category) || group.category} (${group.files.length})`;
        heading.style.cssText = 'margin:7px 0 3px;color:#7dd3fc;cursor:pointer;';
        category.appendChild(heading);
        group.files.forEach(file => {
          const button = document.createElement('button');
          button.type = 'button';
          button.textContent = file.path;
          button.style.cssText = 'display:block;width:100%;text-align:left;padding:4px 7px;margin:2px 0;font-size:11px;overflow-wrap:anywhere;';
          button.addEventListener('click', () => previewDeliveryFile(file));
          category.appendChild(button);
        });
        tree.appendChild(category);
      });
    } catch (error) {
      if (requestId === deliveryRequestId) preview.textContent = `${tr('previewLoadError')}${error.message}`;
    }
  }

  function searchTaskDeliveries() {
    clearTimeout(deliverySearchTimer);
    deliverySearchTimer = setTimeout(() => fetchTaskDeliveries(window.deliveryTaskId, 0), 250);
  }

  function changeDeliveryPage(direction) {
    const nextOffset = Math.max(0, deliveryOffset + direction * deliveryPageSize);
    if (nextOffset === deliveryOffset) return;
    deliveryOffset = nextOffset;
    fetchTaskDeliveries(window.deliveryTaskId, deliveryOffset);
  }

  async function previewDeliveryFile(file) {
    const preview = document.getElementById('delivery-preview');
    preview.replaceChildren();
    const link = document.createElement('a');
    link.href = file.url;
    link.textContent = file.path;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    preview.appendChild(link);
    const extension = file.path.split('.').pop().toLowerCase();
    if (extension === 'html') {
      openGamePreview(file.url);
      return;
    }
    if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(extension)) {
      const image = document.createElement('img');
      image.src = file.url;
      image.alt = file.path;
      image.style.cssText = 'display:block;max-width:100%;max-height:320px;margin-top:8px;object-fit:contain;';
      preview.appendChild(image);
      return;
    }
    if (extension === 'mp4') {
      const video = document.createElement('video');
      video.src = file.url;
      video.controls = true;
      video.style.cssText = 'display:block;width:100%;max-height:320px;margin-top:8px;';
      preview.appendChild(video);
      return;
    }
    if (file.size > 1500000) {
      const note = document.createElement('p');
      note.textContent = tr('previewTooLarge');
      preview.appendChild(note);
      return;
    }
    try {
      const response = await fetch(file.url);
      if (!response.ok) throw new Error(response.statusText);
      const text = await response.text();
      if (extension === 'csv') {
        preview.appendChild(renderCsvPreview(text));
      } else if (extension === 'md' || extension === 'markdown') {
        const article = document.createElement('article');
        article.style.cssText = 'font-size:13px;overflow-wrap:anywhere;';
        article.innerHTML = renderMarkdownPreview(text);
        preview.appendChild(article);
      } else {
        const pre = document.createElement('pre');
        pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;';
        pre.textContent = extension === 'json' ? JSON.stringify(JSON.parse(text), null, 2) : text;
        preview.appendChild(pre);
      }
    } catch (error) {
      preview.textContent = `${tr('previewLoadError')}${error.message}`;
    }
  }

  function renderWorkers(workers) {
    window.currentWorkers = workers || [];
    const wBox = document.getElementById('active-workers-box');
    if (workers && workers.length > 0) {
      const pills = workers.map(w => {
        const pill = document.createElement('div');
        pill.className = 'worker-pill';
        pill.textContent = `🔥 ${w.model} (Ctx:${w.ctx} | ~${w.reserved_gb}GB) - ${w.action} [${w.tokens} tokens]`;
        return pill;
      });
      wBox.replaceChildren(...pills);
    } else {
      const idle = document.createElement('div');
      idle.style.cssText = 'font-size:12px;color:#64748b;';
      idle.textContent = tr('workerIdleShort');
      wBox.replaceChildren(idle);
    }
  }

  function renderModelProgress(workers) {
    const box = document.getElementById('model-progress-stream');
    const active = (workers || []).filter(w => w.model);
    if (!active.length) {
      box.replaceChildren();
      return;
    }
    const cards = active.map(worker => {
      const card = document.createElement('div');
      card.className = 'model-progress-card';
      const heading = document.createElement('div');
      heading.style.cssText = 'display:flex;justify-content:space-between;gap:8px;font-size:12px;';
      const identity = document.createElement('strong');
      identity.textContent = `${worker.model} · ${worker.task_id || ''}`;
      const percent = document.createElement('span');
      percent.textContent = `${worker.progress_pct || 0}%`;
      heading.append(identity, percent);

      const action = document.createElement('div');
      action.style.cssText = 'font-size:11px;color:#94a3b8;margin-top:3px;overflow-wrap:anywhere;';
      action.textContent = worker.action || '';

      const track = document.createElement('div');
      track.className = 'model-progress-track';
      track.setAttribute('role', 'progressbar');
      track.setAttribute('aria-valuemin', '0');
      track.setAttribute('aria-valuemax', '100');
      track.setAttribute('aria-valuenow', String(worker.progress_pct || 0));
      const fill = document.createElement('div');
      fill.className = 'model-progress-fill';
      fill.style.width = `${Math.max(0, Math.min(99, worker.progress_pct || 0))}%`;
      track.append(fill);

      const meta = document.createElement('div');
      meta.className = 'model-progress-meta';
      const elapsed = document.createElement('span');
      elapsed.textContent = `${tr('modelElapsed')} ${fmtSec(worker.elapsed_sec)}`;
      const eta = document.createElement('span');
      eta.textContent = `${tr('modelEta')} ${fmtSec(worker.eta_sec)}`;
      meta.append(elapsed, eta);

      const rate = document.createElement('div');
      rate.style.cssText = 'font-size:10px;color:#64748b;margin-top:3px;';
      rate.textContent = `${tr('modelProgress')}: ${worker.tokens || 0} / ~${worker.estimated_output_tokens || 0} tokens · ${tr('tokenRate')}: ${worker.tokens_per_sec || 0} tok/s`;
      card.append(heading, action, track, meta, rate);
      return card;
    });
    box.replaceChildren(...cards);
  }

  function openGamePreview(url) {
    if (url) gamePreviewUrl = url;
    if (!gamePreviewUrl) return;
    const frame = document.getElementById('game-frame');
    frame.onload = syncPreviewLanguage;
    if (frame.src !== new URL(gamePreviewUrl, location.href).href) frame.src = gamePreviewUrl;
    document.getElementById('game-stage').hidden = false;
    document.getElementById('game-open-btn').hidden = false;
  }
  function syncPreviewLanguage() {
    document.getElementById('game-frame').contentWindow?.postMessage(
      { type: 'set-language', language: lang },
      '*'
    );
  }
  function closeGamePreview() {
    const stage = document.getElementById('game-stage');
    if (document.fullscreenElement) document.exitFullscreen();
    stage.hidden = true;
  }
  async function toggleGameFullscreen() {
    const stage = document.getElementById('game-stage');
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (stage.requestFullscreen) await stage.requestFullscreen();
  }
  document.addEventListener('fullscreenchange', () => {
    const label = document.getElementById('ui-btn-fullscreen-label');
    if (label) label.textContent = tr(document.fullscreenElement ? 'exitFullscreen' : 'fullscreen');
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !document.getElementById('game-stage').hidden) closeGamePreview();
  });

  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
  const logBox = document.getElementById('log-messages');
  const logContainer = document.getElementById('log-stream');

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'telemetry') {
      document.getElementById('ui-state-badge').innerText = `[${data.state}]`;
      if (data.model_error !== undefined) {
        const modelStatus = document.getElementById('model-status');
        modelStatus.textContent = [
          data.model_error && `${tr('modelsUnavailable')} ${data.model_error}`,
          data.memory_error && `${tr('memoryUnavailable')} ${data.memory_error}`
        ].filter(Boolean).join(' ');
      }
      document.getElementById('val-total-time').innerText = fmtSec(data.total_elapsed_sec);
      document.getElementById('val-task-time').innerText = fmtSec(data.task_elapsed_sec);
      document.getElementById('val-estimated-total').innerText = fmtSec(data.estimated_total_sec);
      document.getElementById('val-task-prog').innerText = `${data.task_progress_pct}% (${data.current_stage === 'Idle' ? tr('idle') : data.current_stage})`;
      document.getElementById('bar-task-prog').style.width = `${data.task_progress_pct}%`;
      document.getElementById('val-queue-prog').innerText = `${data.queue_completed} / ${data.queue_total} (${data.queue_pct}%)`;
      document.getElementById('bar-queue-prog').style.width = `${data.queue_pct}%`;
      document.getElementById('mem-live').innerText = `VRAM/RAM: ${data.used_memory_gb} / ${data.budget_gb} GB`;

      if (data.tasks) renderTaskBacklog(data.tasks);

      renderWorkers(data.active_workers);
      renderModelProgress(data.active_workers);
      return;
    }

    const div = document.createElement('div');
    div.className = 'log-item';
    const timeLabel = document.createElement('span');
    timeLabel.style.color = '#64748b';
    timeLabel.textContent = `[${data.time}] `;
    const taskLabel = document.createElement('strong');
    taskLabel.style.color = '#facc15';
    taskLabel.textContent = `[${data.task_id}] `;
    div.append(timeLabel, taskLabel, document.createTextNode(data.message || ''));
    logBox.appendChild(div);
    if (data.type === 'image_preview' && data.extra?.image_url) {
      const link = document.createElement('a');
      link.href = data.extra.image_url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = ` ${tr('openImage')}`;
      div.appendChild(link);
    }
    logContainer.scrollTop = logContainer.scrollHeight;

    if (data.type === 'game_preview' && data.extra.preview_url) {
      openGamePreview(data.extra.preview_url);
    }
    if (data.type === 'film_preview' && data.extra.preview_url) {
      openGamePreview(data.extra.preview_url);
    }
    if (data.type === 'game_assets' && data.extra.assets) {
      document.getElementById('asset-box').replaceChildren(...data.extra.assets.map(a => {
        const item = document.createElement('div');
        item.textContent = `✔ ${a}`;
        return item;
      }));
    }
  };

  async function initLoad() {
    const res = await fetch('/api/init');
    const data = await res.json();
    document.getElementById('mem-input').value = data.budget_gb;
    document.getElementById('context-live').textContent =
      `${tr('contextCapacity')}: ${data.context_capacity || 0} tokens · ${tr('ollamaVersion')} ${data.ollama_version || 'unknown'}`;
    renderTaskBacklog(data.tasks);
    const modelStatus = document.getElementById('model-status');
    modelStatus.textContent = [
      data.model_error && `${tr('modelsUnavailable')} ${data.model_error}`,
      data.memory_error && `${tr('memoryUnavailable')} ${data.memory_error}`
    ].filter(Boolean).join(' ');
    document.getElementById('model-list').replaceChildren(...data.models.map(m => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b;';
      const name = document.createElement('span');
      name.textContent = `${m.supports_completion || m.supports_decision ? '✔' : 'ℹ'} ${m.name}`;
      const details = document.createElement('span');
      details.style.color = '#38bdf8';
      const capabilities = Array.isArray(m.capabilities) && m.capabilities.length
        ? m.capabilities.join(', ')
        : tr('capabilitiesUnavailable');
      details.textContent = `${m.size_gb}GB (VRAM ~${m.est_vram_gb}G) · Ctx ≤ ${m.context_limit || 0} (${m.context_limit_source || 'unknown'}) · ${capabilities}`;
      row.append(name, details);
      return row;
    }));

    if (data.existing_preview) {
      gamePreviewUrl = data.existing_preview;
      document.getElementById('game-open-btn').hidden = false;
      openGamePreview(data.existing_preview);
    }
    if (data.existing_film) {
      openGamePreview(data.existing_film);
    }
    if (data.existing_preview_error) {
      document.getElementById('preview-empty').textContent =
        `${tr('previewInvalid')} ${data.existing_preview_error}`;
    }
    if (data.existing_assets && data.existing_assets.length > 0) {
      document.getElementById('asset-box').replaceChildren(...data.existing_assets.map(a => {
        const item = document.createElement('div');
        item.textContent = `✔ ${a}`;
        return item;
      }));
    }
  }

  async function applyMemory() {
    const val = document.getElementById('mem-input').value;
    const fd = new FormData();
    fd.append('budget_gb', val);
    await fetch('/api/config/memory', { method: 'POST', body: fd });
    await initLoad();
  }

  async function ctrlDaemon(action) {
    const fd = new FormData();
    fd.append('action', action);
    await fetch('/api/daemon/control', { method: 'POST', body: fd });
  }

  async function submitTask() {
    const title = document.getElementById('t-title').value.trim() || 'Autonomous Task';
    const desc = document.getElementById('t-desc').value.trim();
    if (!desc) return alert(tr('enterDescription'));
    const iter = document.getElementById('t-iter').value;
    const fInput = document.getElementById('t-file');

    const fd = new FormData();
    fd.append('title', title);
    fd.append('description', desc);
    fd.append('iterations', iter);
    fd.append('priority', document.getElementById('t-priority').value || '0');
    if (fInput.files[0]) fd.append('file', fInput.files[0]);

    const res = await fetch('/api/tasks/create', { method: 'POST', body: fd });
    const out = await res.json();
    document.getElementById('fb-task').value = out.task_id;
    document.getElementById('t-title').value = '';
    document.getElementById('t-desc').value = '';
  }

  async function sendFeedback() {
    const tid = document.getElementById('fb-task').value.trim();
    const txt = document.getElementById('fb-text').value.trim();
    const statusEl = document.getElementById('fb-status');
    if (!txt) return;
    const fd = new FormData();
    fd.append('task_id', tid);
    fd.append('feedback', txt);
    const res = await fetch('/api/tasks/feedback', { method: 'POST', body: fd });
    if (res.ok) {
      document.getElementById('fb-text').value = '';
      statusEl.innerText = tr('feedbackSent').replace('{taskId}', tid);
      setTimeout(() => { statusEl.innerText = ''; }, 4000);
    }
  }

  function toggleLang() {
    lang = lang === 'zh' ? 'en' : 'zh';
    localStorage.setItem('omni-language', lang);
    applyTranslations();
  }

  applyTranslations();
  initLoad();
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_UI)