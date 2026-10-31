import json
import time
import asyncio
import threading
from pathlib import Path
from typing import List
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from config import WORKSPACE_DIR, TASK_ZONE, DELIVERY_ZONE, FEEDBACK_DIR, UPLOAD_DIR
from core.model_pool import DynamicModelPool
from daemon import OmniEnterpriseDaemon

app = FastAPI(title="Omni-Autonomous AI Enterprise v2")
app.mount("/delivery", StaticFiles(directory=str(DELIVERY_ZONE)), name="delivery")

pool = DynamicModelPool()
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

@app.get("/api/models")
async def get_models():
    models = pool.refresh_models()
    mem = pool.get_running_memory_gb()
    live_file = WORKSPACE_DIR / "01_staff" / "live_status.json"
    live_status = json.loads(live_file.read_text(encoding="utf-8")) if live_file.exists() else {}
    return {"models": models, "memory": mem, "live": live_status}

@app.post("/api/tasks/create")
async def create_task(
    title: str = Form(...),
    description: str = Form(...),
    task_types: str = Form(...),
    iterations: int = Form(3),
    file: UploadFile = File(None)
):
    backlog_path = TASK_ZONE / "task_backlog.json"
    tasks = json.loads(backlog_path.read_text(encoding="utf-8")) if backlog_path.exists() else []
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
        "task_types": json.loads(task_types),
        "iterations": iterations,
        "attached_file": saved_file,
        "status": "processing"
    }
    tasks.append(new_task)
    backlog_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 背景執行緒啟動長時任務
    threading.Thread(target=daemon.process_task, args=(new_task,), daemon=True).start()
    return {"status": "started", "task_id": task_id}

@app.post("/api/tasks/feedback")
async def submit_feedback(task_id: str = Form(...), feedback: str = Form(...)):
    fb_file = FEEDBACK_DIR / f"{task_id}.json"
    items = json.loads(fb_file.read_text(encoding="utf-8")) if fb_file.exists() else []
    items.append({"time": time.strftime("%H:%M:%S"), "text": feedback})
    fb_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "queued", "message": "意見已排入佇列，將於當前階段完成後立即讀取！"}

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
<title>Omni-Autonomous AI Enterprise v2</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #090d16; color: #f1f5f9; font-family: 'Segoe UI', monospace; display: flex; flex-direction: column; height: 100vh; }
  header { background: #1e293b; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
  .status-pill { background: #0f172a; padding: 6px 14px; border-radius: 6px; border: 1px solid #38bdf8; color: #38bdf8; font-size: 13px; }
  .container { display: grid; grid-template-columns: 340px 1fr 480px; flex: 1; overflow: hidden; }
  .col { padding: 14px; border-right: 1px solid #1e293b; display: flex; flex-direction: column; overflow-y: auto; gap: 12px; }
  .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }
  h3 { margin: 0 0 10px 0; font-size: 15px; color: #38bdf8; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
  input[type="text"], textarea, input[type="number"] { width: 100%; background: #020617; border: 1px solid #334155; color: #fff; padding: 8px; border-radius: 5px; margin-bottom: 8px; }
  .checkbox-group label { display: block; margin-bottom: 6px; cursor: pointer; font-size: 13px; }
  button { background: #2563eb; color: #fff; border: none; padding: 9px 14px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; }
  button:hover { background: #1d4ed8; }
  #log-stream { flex: 1; background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; overflow-y: auto; font-size: 13px; white-space: pre-wrap; }
  .log-item { margin-bottom: 8px; border-bottom: 1px solid #0f172a; padding-bottom: 4px; }
  .tag { color: #facc15; font-weight: bold; }
  iframe { width: 100%; height: 360px; background: #000; border: 1px solid #38bdf8; border-radius: 6px; }
  .model-tag { font-size: 12px; padding: 4px 6px; background: #1e293b; border-radius: 4px; margin-bottom: 4px; display: flex; justify-content: space-between; }
</style>
</head>
<body>
<header>
  <div><strong>🌐 本地全自主 AI 研發與遊戲工廠 v2.0</strong> (40GB 動態顯存保護 / 自動 Context 調節)</div>
  <div id="live-bar" class="status-pill">正在掃描本機 Ollama 模型...</div>
</header>
<div class="container">
  <!-- 左欄：任務派發、意見隨時注入、模型即時清單 -->
  <div class="col">
    <div class="card">
      <h3>🚀 派發複合型自主任務</h3>
      <input type="text" id="t-title" placeholder="任務標題 (如: 太空彈幕倖存者遊戲)" />
      <textarea id="t-desc" rows="3" placeholder="輸入初始想法，其餘架構、素材、除錯由 AI 全權自主判斷..."></textarea>
      <div class="checkbox-group">
        <label><input type="checkbox" value="project_code" class="t-type" checked> 💻 專案級程式碼 (Multi-File TDD)</label>
        <label><input type="checkbox" value="game_dev" class="t-type"> 🎮 遊戲製作 (自主生成素材+即時預覽)</label>
        <label><input type="checkbox" value="image_analysis" class="t-type"> 👁 圖片分析 (多模態視覺萃取)</label>
        <label><input type="checkbox" value="data_cleaning" class="t-type"> 🧹 資料清理 (自動清洗與斷言驗證)</label>
      </div>
      <label style="font-size:12px;color:#94a3b8;">自我演化迭代輪數：</label>
      <input type="number" id="t-iter" value="3" min="1" max="20" />
      <label style="font-size:12px;color:#94a3b8;">附件上傳 (圖片 / CSV / Excel)：</label>
      <input type="file" id="t-file" style="margin-bottom:10px;font-size:12px;" />
      <button onclick="startTask()">啟動全自主執行</button>
    </div>

    <div class="card">
      <h3>💡 隨時提供新意見 (階段完成後讀取)</h3>
      <input type="text" id="fb-task" placeholder="目標任務編號 (預設 TASK-001)" value="TASK-001" />
      <textarea id="fb-text" rows="2" placeholder="隨時輸入修改意見（如：把主角改成雙發雷射、加入Boss戰）..."></textarea>
      <button style="background:#059669;" onclick="sendFeedback()">注入新意見至佇列</button>
    </div>

    <div class="card">
      <h3>🧠 本機已下載 Ollama 模型 (<a href="#" onclick="loadModels()" style="color:#38bdf8;">即時刷新</a>)</h3>
      <div id="model-list">載入中...</div>
    </div>
  </div>

  <!-- 中欄：模型目前動作與即時思考日誌流 -->
  <div class="col">
    <h3>📡 模型即時工作狀態與自我糾錯日誌</h3>
    <div id="current-action" class="card" style="color:#a855f7;font-weight:bold;">目前狀態：待命中</div>
    <div id="log-stream"></div>
  </div>

  <!-- 右欄：遊戲製作過程即時預覽與自主合成素材展示 -->
  <div class="col">
    <h3>🎮 遊戲製作即時畫面預覽 & 自主素材庫</h3>
    <iframe id="game-frame" src="about:blank"></iframe>
    <div class="card">
      <h3>🎨 模型自主生成之素材 (Sprite & SFX)</h3>
      <div id="asset-box" style="font-size:12px;color:#94a3b8;">尚無生成素材</div>
    </div>
  </div>
</div>

<script>
  const ws = new WebSocket(`ws://${location.host}/ws`);
  const logBox = document.getElementById('log-stream');

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'token') {
      document.getElementById('current-action').innerText = `[${data.task_id}] ${data.message}`;
      return;
    }
    const div = document.createElement('div');
    div.className = 'log-item';
    div.innerHTML = `<span style="color:#64748b">[${data.time}]</span> <span class="tag">[${data.task_id}]</span> ${data.message}`;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;

    if (data.type === 'game_preview' && data.extra.preview_url) {
      document.getElementById('game-frame').src = data.extra.preview_url;
    }
    if (data.type === 'game_assets' && data.extra.assets) {
      document.getElementById('asset-box').innerHTML = data.extra.assets.map(a => `<div>✔ ${a}</div>`).join('');
    }
  };

  async function loadModels() {
    const res = await fetch('/api/models');
    const data = await res.json();
    const mList = document.getElementById('model-list');
    mList.innerHTML = data.models.map(m =>
      `<div class="model-tag"><span>${m.name}</span><span>${m.size_gb} GB (預估VRAM ${m.est_vram_gb}G)</span></div>`
    ).join('');
    const live = data.live || {};
    document.getElementById('live-bar').innerText =
      `已載入顯存: ${data.memory.used_gb} / 40.0 GB | 當前模型: ${live.model || '無'} | 動態 Ctx: ${live.num_ctx || '-'} | 動作: ${live.action || '閒置'}`;
  }

  async function startTask() {
    const title = document.getElementById('t-title').value.trim() || '未命名自主任務';
    const desc = document.getElementById('t-desc').value.trim();
    const types = Array.from(document.querySelectorAll('.t-type:checked')).map(cb => cb.value);
    const iter = document.getElementById('t-iter').value;
    const fileInput = document.getElementById('t-file');

    if (!types.length) return alert('請至少勾選一種任務類型');

    const fd = new FormData();
    fd.append('title', title);
    fd.append('description', desc);
    fd.append('task_types', JSON.stringify(types));
    fd.append('iterations', iter);
    if (fileInput.files[0]) fd.append('file', fileInput.files[0]);

    const res = await fetch('/api/tasks/create', { method: 'POST', body: fd });
    const out = await res.json();
    document.getElementById('fb-task').value = out.task_id;
  }

  async function sendFeedback() {
    const tid = document.getElementById('fb-task').value.trim();
    const text = document.getElementById('fb-text').value.trim();
    if (!text) return;
    const fd = new FormData();
    fd.append('task_id', tid);
    fd.append('feedback', text);
    await fetch('/api/tasks/feedback', { method: 'POST', body: fd });
    document.getElementById('fb-text').value = '';
  }

  setInterval(loadModels, 3000);
  loadModels();
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_UI)