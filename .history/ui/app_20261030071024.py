import json
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from config import DELIVERY_ZONE, FEEDBACK_DIR, UPLOAD_DIR, BACKLOG_FILE
from daemon import OmniEnterpriseDaemon

app = FastAPI(title="Omni-Autonomous AI Enterprise v4.0")
app.mount("/delivery", StaticFiles(directory=str(DELIVERY_ZONE)), name="delivery")

ws_clients = set()
main_loop = None

def sync_broadcast(payload: dict):
    if main_loop and ws_clients:
        msg = json.dumps(payload, ensure_ascii=False)
        for ws in list(ws_clients):
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), main_loop)

daemon = OmniEnterpriseDaemon(broadcast_fn=sync_broadcast)

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

@app.get("/api/init")
async def get_init_state():
    models = daemon.pool.refresh_models()
    mem = daemon.pool.sync_realtime_memory()
    budget = daemon.pool.get_hardware_budget()
    ctrl = daemon.get_control_state()
    q_prog = daemon.get_queue_progress()
    tasks = daemon.get_all_tasks()

    # 自動檢查最後一個任務是否已有遊戲預覽與素材
    existing_preview = ""
    existing_assets = []
    if tasks:
        last_tid = tasks[-1]["task_id"]
        g_dir = DELIVERY_ZONE / "games" / last_tid
        if (g_dir / "index.html").exists():
            existing_preview = f"/delivery/games/{last_tid}/index.html?t={int(time.time())}"
        if (g_dir / "assets").exists():
            existing_assets = [p.name for p in (g_dir / "assets").iterdir() if p.is_file()]

    return {
        "models": models,
        "memory": mem,
        "budget_gb": budget,
        "control": ctrl,
        "queue": q_prog,
        "tasks": tasks,
        "existing_preview": existing_preview,
        "existing_assets": existing_assets
    }

@app.post("/api/config/memory")
async def update_memory_budget(budget_gb: float = Form(...)):
    daemon.pool.set_hardware_budget(budget_gb)
    swarm = daemon.pool.plan_greedy_swarm()
    daemon.emit_log("SYS", f"⚡ 記憶體上限已更新為 {budget_gb} GB！貪婪調度器將同時啟動 {len(swarm)} 個模型槽位：{[m['name'] for m in swarm]}")
    return {"status": "ok", "budget_gb": budget_gb}

@app.post("/api/daemon/control")
async def control_daemon(action: str = Form(...)):
    if action == "start":
        daemon.start_queue_loop()
        return {"status": "RUNNING"}
    elif action == "pause":
        daemon.set_control_state("PAUSED", pause_req=True, stop_req=False)
        return {"status": "PAUSED"}
    elif action == "resume":
        daemon.set_control_state("RUNNING", pause_req=False, stop_req=False)
        if not daemon.daemon_active:
            daemon.start_queue_loop()
        return {"status": "RUNNING"}
    elif action == "stop":
        daemon.set_control_state("STOPPED", pause_req=False, stop_req=True)
        daemon.pool.unload_all_models()
        return {"status": "STOPPED"}
    return {"status": "UNKNOWN"}

@app.post("/api/tasks/create")
async def create_task(
    title: str = Form(...),
    description: str = Form(...),
    iterations: int = Form(3),
    file: UploadFile = File(None)
):
    with daemon.backlog_lock:
        tasks = json.loads(BACKLOG_FILE.read_text(encoding="utf-8")) if BACKLOG_FILE.exists() else []
        task_id = f"TASK-{len(tasks) + 1:03d}"

        saved_file = ""
        if file and file.filename:
            dest = UPLOAD_DIR / f"{task_id}_{file.filename}"
            dest.write_bytes(await file.read())
            saved_file = str(dest)

        new_task = {
            "task_id": task_id,
            "title": title,
            "description": description,
            "iterations": iterations,
            "attached_file": saved_file,
            "status": "pending"
        }
        tasks.append(new_task)
        BACKLOG_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    daemon.emit_log(task_id, f"📥 新全能任務已加入佇列：{title}")
    daemon.start_queue_loop()
    return {"status": "queued", "task_id": task_id}

@app.post("/api/tasks/feedback")
async def submit_feedback(task_id: str = Form(...), feedback: str = Form(...)):
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
    except Exception:
        ws_clients.remove(ws)

HTML_UI = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>Omni-Autonomous AI Enterprise v4.0</title>
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
  .kpi-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 10px 18px; background: #0f172a; border-bottom: 1px solid #1e293b; }
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
  iframe { width: 100%; height: 380px; background: #000; border: 1px solid #38bdf8; border-radius: 6px; }
  .worker-pill { font-size: 12px; background: #1e1b4b; border: 1px solid #6366f1; padding: 6px 8px; border-radius: 4px; margin-bottom: 4px; }
  .task-row { font-size: 12px; padding: 6px 8px; border-radius: 4px; background: #020617; border-left: 4px solid #facc15; margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center; }
  .task-row.completed { border-left-color: #22c55e; }
  .task-row.processing { border-left-color: #38bdf8; background: #0f172a; }
</style>
</head>
<body>
<header>
  <div><strong id="ui-app-title">🌐 本地全能自主 AI 企業系統 v4.0</strong> <span id="ui-state-badge" style="color:#22c55e;font-weight:bold;">[IDLE]</span></div>
  <div class="ctrl-group">
    <span id="ui-mem-label" style="font-size:13px;">記憶體上限 (GB):</span>
    <input type="number" id="mem-input" value="40" min="4" max="256" step="1" style="width:70px;margin:0;" />
    <button class="btn btn-resume" onclick="applyMemory()" id="ui-btn-apply">套用</button>
    <span id="mem-live" style="font-size:13px;color:#38bdf8;margin:0 8px;">VRAM/RAM: 0.0 / 40.0 GB</span>
    <button class="btn btn-start" onclick="ctrlDaemon('start')" id="ui-btn-start">▶ 開始 / 續跑未完成任務</button>
    <button class="btn btn-pause" onclick="ctrlDaemon('pause')" id="ui-btn-pause">⏸ 暫停</button>
    <button class="btn btn-resume" onclick="ctrlDaemon('resume')" id="ui-btn-resume">⏯ 繼續</button>
    <button class="btn btn-stop" onclick="ctrlDaemon('stop')" id="ui-btn-stop">⏹ 終止</button>
    <button class="btn btn-lang" onclick="toggleLang()">🌐 中文 / EN</button>
  </div>
</header>

<div class="kpi-bar">
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-total-time">⏱ 總執行耗時 (真實物理時鐘)</div>
    <div class="kpi-val" id="val-total-time">00:00:00</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-task-time">⏳ 當前任務耗時</div>
    <div class="kpi-val" id="val-task-time">00:00:00</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-task-prog">🎯 當前任務進度</div>
    <div class="kpi-val" id="val-task-prog">0% (Idle)</div>
    <div class="prog-bg"><div class="prog-fill" id="bar-task-prog"></div></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title" id="ui-kpi-queue-prog">📊 總任務佇列進度</div>
    <div class="kpi-val" id="val-queue-prog">0 / 0 (0%)</div>
    <div class="prog-bg"><div class="prog-fill" id="bar-queue-prog"></div></div>
  </div>
</div>

<div class="container">
  <div class="col">
    <!-- 新增：隨時可見的未完成與待辦任務佇列看板 -->
    <div class="card">
      <h3 id="ui-backlog-title">📋 任務佇列看板 (含先前未完成任務)</h3>
      <div id="backlog-box" style="max-height:130px;overflow-y:auto;">載入中...</div>
    </div>

    <div class="card">
      <h3 id="ui-new-task-title">🚀 派發全能任務 (貪婪塞滿記憶體並行)</h3>
      <input type="text" id="t-title" placeholder="輸入任務標題或問題摘要..." />
      <textarea id="t-desc" rows="3" placeholder="輸入任何需求，系統將自動調動所有可用模型塞滿記憶體並行執行..."></textarea>
      <label style="font-size:12px;color:#94a3b8;" id="ui-iter-label">最大自主演化輪數：</label>
      <input type="number" id="t-iter" value="3" min="1" max="20" />
      <label style="font-size:12px;color:#94a3b8;" id="ui-file-label">附加檔案 (圖片 / CSV / Excel / 代碼)：</label>
      <input type="file" id="t-file" style="margin-bottom:6px;font-size:12px;" />
      <button class="btn btn-start" style="width:100%;" onclick="submitTask()" id="ui-btn-submit">加入佇列並自主執行</button>
    </div>

    <div class="card">
      <h3 id="ui-fb-title">💡 隨時注入新意見 (階段完成後讀取)</h3>
      <input type="text" id="fb-task" value="TASK-001" />
      <textarea id="fb-text" rows="2" placeholder="執行中或暫停時皆可輸入新要求..."></textarea>
      <button class="btn btn-resume" style="width:100%;" onclick="sendFeedback()" id="ui-btn-fb">送出意見</button>
      <div id="fb-status" style="font-size:12px;color:#22c55e;margin-top:4px;"></div>
    </div>

    <div class="card">
      <h3 id="ui-models-title">🧠 本機全可用 Ollama 模型池 (全部參與調度)</h3>
      <div id="model-list" style="font-size:12px;"></div>
    </div>
  </div>

  <div class="col">
    <h3 id="ui-workers-title">⚡ 當前並行執行中之模型槽位 (Greedy Memory Packing Slots)</h3>
    <div id="active-workers-box"><div style="font-size:12px;color:#64748b;">Idle - 請點擊右上角「▶ 開始 / 續跑未完成任務」</div></div>
    <h3 id="ui-logs-title">📡 系統決策、多模型分治與演化日誌</h3>
    <div id="log-stream"></div>
  </div>

  <div class="col">
    <h3 id="ui-preview-title">🎮 遊戲即時試玩預覽 / 交付預覽</h3>
    <iframe id="game-frame" src="about:blank"></iframe>
    <div class="card">
      <h3 id="ui-assets-title">🎨 自主合成素材庫 (Sprite & WAV)</h3>
      <div id="asset-box" style="font-size:12px;color:#94a3b8;max-height:180px;overflow-y:auto;">-</div>
    </div>
  </div>
</div>

<script>
  let lang = 'zh';
  function fmtSec(s) {
    const h = String(Math.floor(s / 3600)).padStart(2, '0');
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const sec = String(s % 60).padStart(2, '0');
    return `${h}:${m}:${sec}`;
  }

  function renderTaskBacklog(tasks) {
    const box = document.getElementById('backlog-box');
    if (!tasks || tasks.length === 0) {
      box.innerHTML = '<div style="font-size:12px;color:#64748b;">目前無任務</div>';
      return;
    }
    box.innerHTML = tasks.map(t => {
      const st = t.status || 'pending';
      const badge = st === 'completed' ? '✅ 完成' : (st === 'processing' ? '🔥 執行中' : '⏳ 未完成(待續跑)');
      return `<div class="task-row ${st}">
        <div><strong>[${t.task_id}]</strong> ${t.title}</div>
        <div style="color:#38bdf8;font-weight:bold;">${badge}</div>
      </div>`;
    }).join('');
  }

  const ws = new WebSocket(`ws://${location.host}/ws`);
  const logBox = document.getElementById('log-stream');

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'telemetry') {
      document.getElementById('ui-state-badge').innerText = `[${data.state}]`;
      document.getElementById('val-total-time').innerText = fmtSec(data.total_elapsed_sec);
      document.getElementById('val-task-time').innerText = fmtSec(data.task_elapsed_sec);
      document.getElementById('val-task-prog').innerText = `${data.task_progress_pct}% (${data.current_stage})`;
      document.getElementById('bar-task-prog').style.width = `${data.task_progress_pct}%`;
      document.getElementById('val-queue-prog').innerText = `${data.queue_completed} / ${data.queue_total} (${data.queue_pct}%)`;
      document.getElementById('bar-queue-prog').style.width = `${data.queue_pct}%`;
      document.getElementById('mem-live').innerText = `VRAM/RAM: ${data.used_memory_gb} / ${data.budget_gb} GB`;

      if (data.tasks) renderTaskBacklog(data.tasks);

      const wBox = document.getElementById('active-workers-box');
      if (data.active_workers && data.active_workers.length > 0) {
        wBox.innerHTML = data.active_workers.map(w =>
          `<div class="worker-pill">🔥 <strong>${w.model}</strong> (Ctx:${w.ctx} | ~${w.reserved_gb}GB) - ${w.action} [${w.tokens} tokens]</div>`
        ).join('');
      } else {
        wBox.innerHTML = `<div style="font-size:12px;color:#64748b;">Idle (點擊右上角「▶ 開始」立即調動多模型執行佇列中之未完成任務)</div>`;
      }
      return;
    }

    const div = document.createElement('div');
    div.className = 'log-item';
    div.innerHTML = `<span style="color:#64748b">[${data.time}]</span> <strong style="color:#facc15">[${data.task_id}]</strong> ${data.message}`;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;

    if (data.type === 'game_preview' && data.extra.preview_url) {
      document.getElementById('game-frame').src = data.extra.preview_url;
    }
    if (data.type === 'game_assets' && data.extra.assets) {
      document.getElementById('asset-box').innerHTML = data.extra.assets.map(a => `<div>✔ ${a}</div>`).join('');
    }
  };

  async function initLoad() {
    const res = await fetch('/api/init');
    const data = await res.json();
    document.getElementById('mem-input').value = data.budget_gb;
    renderTaskBacklog(data.tasks);
    document.getElementById('model-list').innerHTML = data.models.map(m =>
      `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b;">
        <span>✔ ${m.name}</span><span style="color:#38bdf8">${m.size_gb}GB (VRAM ~${m.est_vram_gb}G)</span>
      </div>`
    ).join('');

    if (data.existing_preview) {
      document.getElementById('game-frame').src = data.existing_preview;
    }
    if (data.existing_assets && data.existing_assets.length > 0) {
      document.getElementById('asset-box').innerHTML = data.existing_assets.map(a => `<div>✔ ${a}</div>`).join('');
    }
  }

  async function applyMemory() {
    const val = document.getElementById('mem-input').value;
    const fd = new FormData();
    fd.append('budget_gb', val);
    await fetch('/api/config/memory', { method: 'POST', body: fd });
  }

  async function ctrlDaemon(action) {
    const fd = new FormData();
    fd.append('action', action);
    await fetch('/api/daemon/control', { method: 'POST', body: fd });
  }

  async function submitTask() {
    const title = document.getElementById('t-title').value.trim() || 'Autonomous Task';
    const desc = document.getElementById('t-desc').value.trim();
    if (!desc) return alert('請輸入需求描述');
    const iter = document.getElementById('t-iter').value;
    const fInput = document.getElementById('t-file');

    const fd = new FormData();
    fd.append('title', title);
    fd.append('description', desc);
    fd.append('iterations', iter);
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
      statusEl.innerText = `✔ 已成功送出意見至 ${tid} 佇列！`;
      setTimeout(() => { statusEl.innerText = ''; }, 4000);
    }
  }

  function toggleLang() {
    lang = lang === 'zh' ? 'en' : 'zh';
    document.getElementById('ui-app-title').innerText = lang === 'zh' ? '🌐 本地全能自主 AI 企業系統 v4.0' : '🌐 Local Omnipotent Autonomous AI Enterprise v4.0';
    document.getElementById('ui-backlog-title').innerText = lang === 'zh' ? '📋 任務佇列看板 (含先前未完成任務)' : '📋 Task Backlog (Unfinished & Pending)';
  }

  initLoad();
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_UI)