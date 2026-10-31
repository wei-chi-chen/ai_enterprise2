# 🌐 本地全能企業 (Local Omnipotent Autonomous AI Enterprise)
<img width="1917" height="867" alt="螢幕擷取畫面 2026-10-31 120256" src="https://github.com/user-attachments/assets/d1d26b86-53a3-4522-be8d-acfa0d2a9861" />
> **Documentation Language / 文件語言**
>
> This documentation provides complete, mirrored guides in both **Traditional Chinese (繁體中文)** and **English**.
>
> 本說明文件提供繁體中文與英文雙語對照，詳盡記錄系統定位、核心架構、動態模型池規格約束、多模型並行裝箱演算法、沙盒隔離與完整操作指南。

## 繁體中文說明 (Traditional Chinese)

### 一、 系統定位與核心理念

**本地全能企業 (Local Omnipotent Autonomous AI Enterprise)** 是一套以本機 Ollama 叢集為驅動核心的自主研發、全端工程與內容生產系統。系統具備嚴謹的「動態硬體記憶體預算排程」**與**「職責導向多模型並行分治 (Role-Driven Swarm Escalation)」機制，所有模型推理、程式碼驗證、插畫繪製、短片合成與資料清理皆在本地端離線閉環執行。

傳統 AI 工具往往依賴單一模型或外部固定 API，面臨模型上下文溢出、幻覺、單點崩潰及難以勝任複雜跨領域任務等問題。本系統致力於將消費級顯卡（如 RTX 3060 6GB）至工作站級硬體發揮至極致，構建「本地端的全自主 AI 研發企業」：

1. **全能自主意圖路由 (Omnipotent Intent Router)**：

   * 系統接收使用者自然語言需求與附件（圖檔、CSV、Excel、原始碼等）。

   * Stage 1 總架構師自主決策啟動哪些管線：一般知識報告、多模態視覺感知、大數據資料清洗、HTML5 Canvas 遊戲開發、短片製作、或是多模態全端軟體開發。

   * 嚴格杜絕「問答任務被誤判為軟體工程」或「純文字分析被強啟遊戲管線」之誤派問題。

2. **角色專業梯隊升級 (Role-Specific Tier Escalation)**：

   * 模型失敗或截斷時，**不再盲目遍歷無關模型**。

   * 系統將模型依專業劃分：

     * **程式設計師 (Coder)**：優先在 `qwen2.5-coder`、`devstral` 等程式特化模型階梯中升級。

     * **總架構師與仲裁者 (Architect / Arbiter)**：優先分派具備深度思維鏈（Thinking Process）之推理模型（如 `deepseek-r1`、`nemotron`、`phi4`）。

     * **視覺專家 (Vision)**：嚴格限制具備圖片感知能力之模型（如 `minicpm-v:8b`、`llava`）。

     * **快速審查 (Fast)**：調度低延遲、高流暢度通用模型（如 `qwen2.5:7b`、`llama3.1:8b`）。

   * 內建思維標籤（`<think>`）與 Markdown 雜訊濾除器，確保結構化 JSON 解析 100% 穩定，終結死循環。

3. **貪婪記憶體裝箱調度 (Greedy Memory Swarm Planner)**：

   * 透過 `/api/tags`、`/api/show` 動態偵測模型層數、注意力頭數與嵌入維度，精確計算 KV Cache 記憶體佔用。

   * 結合 `nvidia-smi` 實時 GPU 顯存監控與 Ollama `/api/ps` 負載狀態，在設定的記憶體預算內（例如 6 GB 或 40 GB），同時調動最多不重複之可用模型槽位並行處理（Map-Reduce）。

4. **雙重程式碼執行沙盒 (Dual-Mode Execution Sandbox)**：

   * **Docker 隔離模式**：預設在唯讀掛載、網路阻斷（`--network=none`）、CPU 與記憶體嚴格受限的容器內執行 Python/Node 程式與測試。

   * **本機直通模式**：針對無 Docker 或本機快速除錯環境，提供環境變數直通運行機制（`OMNI_UNSAFE_LOCAL_PYTHON=1`、`OMNI_UNSAFE_LOCAL_NODE=1`）。

5. **多輪自主演化與 TDD 三級升級除錯**：

   * 支援 1–20 輪自主演化（Iterations）。

   * 軟體專案先建立跨模組共享介面存根（`interfaces.py`），平行撰寫各模組後，由自動生成的測試套件（`test_suite.py`）進行嚴格驗收。未通過測試時，由高階仲裁模型提出最小局部差異補丁（Exact Context Patch），確保不會破壞既有程式。

6. **程序化多媒體鍛造引擎 (Forge Engines)**：

   * **遊戲鍛造**：依任務動態規劃素材清單，程序化產製 2D 精靈圖（PNG）與音效（WAV），以無狀態通用 Canvas 執行殼與模組介面註冊，交付前由 Node.js 執行首幀與語法驗證。

   * **電影短片鍛造**：撰寫多場景劇本與分鏡，利用本機 `imageio-ffmpeg` 渲染 H.264 MP4 短片，並自動透過 Windows 原生語音模組（`System.Speech`）合成對齊旁白。

   * **高解析度向量插畫**：生成包含漸層、自訂幾何路徑與發光層之向量規格，最高輸出 2048×2048 PNG。

### 二、 系統架構與專區目錄

專案遵循四大實體專區規範（位於 `enterprise_workspace/` 下）：

```
Project_Root/
├── config.py                       # 全域路徑、硬體預算與狀態檔初始化
├── main.py                         # 終端 CLI 啟動選單與工作區重置工具
├── daemon.py                       # 核心排程守護程序 (自主路由、DAG 依賴解析、演化循環)
├── requirements.txt                # Python 依賴清單 (pip 安裝項目)
├── core/
│   └── model_pool.py               # 動態模型池 (模型約束、KV Cache 計算、角色梯隊升級、Ollama API 通訊)
├── generators/
│   ├── asset_forge.py              # 遊戲素材程序化鍛造 (PNG 精靈、WAV 音效)
│   ├── film_forge.py               # 電影短片鍛造 (分鏡繪製、FFmpeg MP4 編碼、System.Speech 語音合成)
│   └── image_forge.py              # 高解析向量圖繪製 (多層幾何、漸層與發光濾鏡)
├── execution/
│   ├── sandbox_runner.py           # Docker/本機隔離執行器、Python/Node 驗證、SQLite 靜態審查
│   └── game_shell.py               # 現代化通用 HTML5 Canvas 遊戲宿主
├── ui/
│   └── app.py                      # FastAPI + WebSocket 響應式監控台 (即時遙測、成果預覽)
└── enterprise_workspace/           # 四大企業實體專區
    ├── 01_staff/                   # 員工與硬體配置 (staff_roster.json, employee_status.json)
    ├── 02_tasks/                   # 任務佇列與控制 (task_backlog.json, daemon_control.json, uploads/)
    ├── 03_delivery/                # 正式交付成果 (projects/, games/, films/, cleaned_data/, reports/, illustrations/)
    └── 04_history/                 # 歷史記憶與審計 (lessons_learned.json, failure_audits/)

```

### 三、 模型規格與約束清單 (`MODEL_PROFILES`)

系統於 `core/model_pool.py` 中明確定義各模型的硬體邊界、端點規範與提示詞約束：

| 模型名稱 | 支援模式 | 端點規範 | 逾時/約束 | 專屬限制與調用策略 | 
| ----- | ----- | ----- | ----- | ----- | 
| `nomic-embed-text:latest` | 向量 | `/api/embeddings` | \- | 輸入單純文字至 `prompt`，輸出 768 維浮點向量；**嚴禁當作生成模型**。 | 
| `bge-m3:latest` | 向量 | `/api/embeddings` | \- | 輸入多語言文字至 `prompt`，輸出 1024 維語意向量；**必須走嵌入端點**。 | 
| `minicpm-v:8b` | 視覺 | `/api/generate` | \- | Payload 必須包含文字與 base64 圖片陣列（`images`）。 | 
| `llava:latest` | 視覺 | `/api/generate` | \- | 必須傳入圖片資料，輸出多模態視覺場景描述。 | 
| `clef:latest` | 長文本問答 | CLI (stdin) | 120s | 需由 stdin 管道灌入全文並搭配 `--ask` 參數；**禁止走一般 generate 端點**。 | 
| `phi4:14b` | 推理/文本 | `/api/generate` | Context ≤ 8192 | 必須限制 Context 長度以避免顯存突發溢出。 | 
| `mistral-small:24b` | 邏輯生成 | `/api/generate` | 120s / 獨佔 | 需獨佔載入，建議設定 `keep_alive: 0` 避免顯存堆疊。 | 
| `gemma2:27b` | 知識問答 | `/api/generate` | \- | 需預留足夠的主機 RAM 以利層間快取調度。 | 
| `flint:latest` | 多步驟規劃 | `/api/generate` | ≥ 120s | 長任務推論，請求逾時需設為 120 秒以上。 | 
| `nemotron-3.5-bolt:latest` | 決策推理 | `/api/generate` | 120s / 思維 | 輸出含完整思考鏈標籤，需提高逾時並支援思考文本擷取。 | 
| `qwen3.8:27b` | 深度分析 | `/api/generate` | 思維 | 文本包含思考過程，解析時必須相容 `thinking` 欄位。 | 
| `gemma4:12b` | 綜合推論 | `/api/generate` | 非串流 | 必須設定非串流模式（`stream: false`）一次取得完整字串。 | 
| `nemotron-3.5-bolt:e4b` | 輕量生成 | `/api/generate` | \- | 低延遲、精煉結論，直接傳入標準文字 Payload。 | 
| `qwen3.5:4b` | 數理邏輯 | `/api/generate` | 思維 | 含 ThinkingProcess 思考歷程，需同時讀取 `response` 與 `thinking`。 | 
| `qwen2.5-coder:14b` | 程式開發 | `/api/generate` | \- | 專門用於程式碼生成、重構與除錯，發送標準 Prompt。 | 
| `qwen2.5:7b` | 日常摘要 | `/api/generate` | 非串流 | 發送標準非串流 JSON，低延遲通用文字回應。 | 
| `qwen2.5-coder:32b` | 大型專案架構 | `/api/generate` | 180s / 獨佔 | 逾時設為 180 秒以上，**避免併發其他模型**並設 `keep_alive: 0`。 | 
| `deepseek-r1:8b` | 數理推理 | `/api/generate` | Context ≤ 8192 | 輸出帶有 `<think>` 標籤，需控制 Context 並支援思考解析。 | 
| `devstral:24b` | 系統規格 | `/api/generate` | 獨佔 | 需在單獨載入且前一個模型完全釋放下執行，保證 CUDAHostBuffer 成功。 | 
| `deepseek-r1:14b` | 演算法推導 | `/api/generate` | \- | 細緻多步驟推論，需預留計算圖動態重分配緩衝。 | 
| `llama3.1:8b` | 通用文本 | `/api/generate` | 獨佔 | 非串流傳入，確保單一模型獨佔顯存。 | 
| `qwen2.5:32b` | 大規模撰寫 | `/api/generate` | 獨佔 | 權重接近顯存邊界，必須單獨載入且設定 `keep_alive: 0`。 | 
| `deepseek-r1:32b` | 高難度競賽證明 | `/api/generate` | 120s / Context ≤ 8192 | 超長思維鏈，逾時至少 120 秒並手動限制 Context 避免爆顯存。 | 

### 四、 安裝與環境準備

#### 1. Python 依賴套件安裝

在專案根目錄下執行：

```
pip install -r requirements.txt

```

> **注意**：作業系統層級之執行檔（Docker、Node.js、Ollama、NVIDIA 驅動）**不能且不應**寫入 `requirements.txt`，請依下方說明配置。

#### 2. 必要外部環境依賴

* **Ollama 本地服務**：

  * 確保本機已安裝 [Ollama](https://ollama.com/) 並啟動服務（預設監聽 `http://localhost:11434`）。

  * 下載至少一個支援生成的模型（例如 `ollama pull qwen2.5:7b`）。

  * 若硬體支援多模型並行，可於啟動服務時設定：

    ```
    $env:OLLAMA_MAX_LOADED_MODELS = "4"
    $env:OLLAMA_NUM_PARALLEL = "4"
    ollama serve
    
    ```

* **Node.js**：

  * 需安裝 Node.js（v18+，建議 LTS v22+），用於 HTML5 遊戲首幀初始化與 JavaScript 語法驗證。

* **程式碼沙盒（二選一）**：

  * **方案 A（Docker 隔離模式 - 推薦）**：
    啟動 Docker Desktop，並於專案目錄建立 Python 沙盒映像檔：

    ```
    docker pull node:22-alpine
    docker build -f execution/Dockerfile.python-sandbox -t omni-python-sandbox:latest .
    
    ```

  * **方案 B（本機直通模式 - 免 Docker）**：
    若僅個人本地研發，可在 PowerShell 設定環境變數直接使用主機環境：

    ```
    [System.Environment]::SetEnvironmentVariable("OMNI_UNSAFE_LOCAL_PYTHON", "1", "User")
    [System.Environment]::SetEnvironmentVariable("OMNI_UNSAFE_LOCAL_NODE", "1", "User")
    
    ```

### 五、 快速啟動與操作指南

1. **啟動主程式**：

   ```
   python main.py
   
   ```

2. **選單說明**：

   * 輸入 `1`：**啟動 Web 控制台**（預設位址：`http://127.0.0.1:8000`）。

   * 輸入 `2`：**重置工作區與佇列**（清空所有交付檔案，重置任務編號為 `TASK-001`）。

   * 輸入 `3`：退出程式。

3. **Web 控制台操作**：

   * **調整記憶體預算**：於右上角設定硬體預算（例如 RTX 3060 建議設定為 `5.5` \~ `6.0` GB，若搭配主機 RAM 混合運算可設為 `12.0` \~ `16.0` GB），點選「套用」。

   * **新增任務**：於左側表單輸入標題、需求說明、設定自主演化輪數（預設 3），可上傳附件圖檔或資料集，點擊「僅加入佇列」。

   * **開始執行**：點選右上角「▶ 開始 / 續跑」，排程守護程序將自動調度最適模型展開流水線作業。

   * **即時預覽**：遊戲或短片產出後，右側預覽視窗會自動開啟，支援全螢幕操作與成果檢驗。

## English Guide

### I. System Vision and Core Concepts

**Local Omnipotent Autonomous AI Enterprise** is a fully autonomous research, software engineering, and multimedia production cluster driven entirely by local Ollama models. Featuring **dynamic memory-budget bin-packing** and **role-specific swarm escalation**, the system runs all inference, code testing, illustration rendering, movie generation, and data cleaning completely offline.

Unlike simple wrapper scripts or single-turn chat interfaces, this enterprise architecture solves token limits, model hallucinations, and system crashes through specialized modular orchestration:

1. **Omnipotent Autonomous Intent Router**:

   * Analyzes user instructions and uploaded files (images, tabular datasets, source code).

   * In Stage 1, the Chief Architect routes tasks to tailored execution pipelines: Direct Q&A Report, Vision Analysis, Data Cleaning, HTML5 Canvas Game Development, Animated Short Film Production, or Full-Stack Python Software Development.

   * Prevents general knowledge inquiries from being misrouted into complex software builds.

2. **Role-Specific Tier Escalation**:

   * Eliminates blind model iterations. When a model truncates or fails schema/test checks, it ascends strictly within its **dedicated domain hierarchy**:

     * **Coder**: Escalates through code-specialized models (`qwen2.5-coder`, `devstral`).

     * **Architect / Arbiter**: Routes to chain-of-thought and reasoning engines (`deepseek-r1`, `nemotron`, `phi4`).

     * **Vision**: Strictly limited to vision-capable models (`minicpm-v:8b`, `llava`).

     * **Fast / Review**: Utilizes lightweight, fluent conversational models (`qwen2.5:7b`, `llama3.1:8b`).

   * Integrated markdown and `<think>` filter guarantees 100% reliable JSON parsing, preventing infinite fallback loops.

3. **Greedy Swarm Memory Planner**:

   * Analyzes transformer architecture parameters (layers, attention heads, embedding length) to compute exact KV Cache memory footprints.

   * Leverages `nvidia-smi` GPU VRAM tracking and Ollama `/api/ps` to saturate the specified memory budget (e.g., 6 GB or 40 GB) with maximal distinct parallel model slots.

4. **Dual-Mode Sandbox Runner**:

   * **Docker Sandbox (Default)**: Enforces network isolation (`--network=none`), read-only mounts, and strict memory/CPU limits for script verification.

   * **Host Pass-Through Mode**: Allows rapid local execution without Docker via environment variables (`OMNI_UNSAFE_LOCAL_PYTHON=1`, `OMNI_UNSAFE_LOCAL_NODE=1`).

5. **Multi-Round Evolution & TDD Verification**:

   * Supports 1 to 20 autonomous quality evolution rounds.

   * Multi-module software builds enforce public symbol contracts via `interfaces.py`, followed by automated regression verification (`test_suite.py`). Failed assertions prompt minimal context-anchored patches rather than destructive whole-file rewrites.

6. **Procedural Media Forge Engines**:

   * **Game Forge**: Synthesizes 2D sprites (PNG) and sound effects (WAV), binds them to a canvas host shell, and verifies execution via Node.js before delivery.

   * **Film Forge**: Authors multi-scene screenplays, renders H.264 MP4 videos via `imageio-ffmpeg`, and synthesizes synchronized voice-over audio via Windows `System.Speech`.

   * **Image Forge**: Procedurally generates multi-layered vector illustrations with gradients and glow shaders up to 2048×2048 PNG resolution.

### II. Directory and Zone Architecture

The workspace strictly maintains an enterprise four-zone layout under `enterprise_workspace/`:

```
Project_Root/
├── config.py                       # Global paths, budget defaults, and state files
├── main.py                         # CLI launcher and workspace reset menu
├── daemon.py                       # Autonomous daemon (routing, DAG orchestration, TDD loops)
├── requirements.txt                # Python package dependencies
├── core/
│   └── model_pool.py               # Dynamic model pool (KV Cache math, role ladders, Ollama APIs)
├── generators/
│   ├── asset_forge.py              # Procedural game assets (PNG sprites, WAV SFX)
│   ├── film_forge.py               # Short film production (storyboards, FFmpeg encoding, TTS)
│   └── image_forge.py              # Vector illustration rendering (gradients, glow filters)
├── execution/
│   ├── sandbox_runner.py           # Docker/host execution sandbox, Node/Python validators
│   └── game_shell.py               # Canvas game runner and module host
├── ui/
│   └── app.py                      # FastAPI + WebSocket bilingual dashboard
└── enterprise_workspace/           # Four Enterprise Zones
    ├── 01_staff/                   # Hardware rosters (staff_roster.json, employee_status.json)
    ├── 02_tasks/                   # Queue & controls (task_backlog.json, daemon_control.json, uploads/)
    ├── 03_delivery/                # Final deliverables (projects/, games/, films/, reports/, etc.)
    └── 04_history/                 # Audits and logs (lessons_learned.json, failure_audits/)

```

### III. Model Constraints Profile (`MODEL_PROFILES`)

Configured in `core/model_pool.py`, defining exact operational boundaries:

| Model | Type | Endpoint | Timeout / Flags | Constraints & Calling Strategy | 
| ----- | ----- | ----- | ----- | ----- | 
| `nomic-embed-text:latest` | Embedding | `/api/embeddings` | \- | Outputs 768-dim float vectors from `prompt`. **Never invoke as a completion model**. | 
| `bge-m3:latest` | Embedding | `/api/embeddings` | \- | Multilingual semantic embedding; outputs 1024-dim vectors via embeddings endpoint. | 
| `minicpm-v:8b` | Vision | `/api/generate` | \- | Requires valid base64 image strings in the `images` payload array. | 
| `llava:latest` | Vision | `/api/generate` | \- | Must provide image data for multimodal visual Q&A. | 
| `clef:latest` | Document QA | CLI (stdin) | 120s | Pipe document body via stdin with `--ask`. **Bypasses HTTP generate endpoints**. | 
| `phi4:14b` | Reasoning | `/api/generate` | Context ≤ 8192 | Context ceiling enforced to prevent sudden VRAM spills. | 
| `mistral-small:24b` | Structured Gen | `/api/generate` | 120s / Exclusive | Must run exclusively; set `keep_alive: 0` to release VRAM immediately after completion. | 
| `gemma2:27b` | Knowledge QA | `/api/generate` | \- | Preserves host RAM buffers for inter-layer cache scheduling. | 
| `flint:latest` | Planning | `/api/generate` | ≥ 120s | Extended multi-step synthesis; timeout set to 120s minimum. | 
| `nemotron-3.5-bolt:latest` | Complex Decision | `/api/generate` | 120s / Thinking | Emits full chain-of-thought tokens; requires thinking field extraction. | 
| `qwen3.8:27b` | Deep Dialogue | `/api/generate` | Thinking | Response parsing must support intermediate `thinking` attributes. | 
| `gemma4:12b` | Task Synthesis | `/api/generate` | Non-streaming | Must use `stream: false` to retrieve the complete string at once. | 
| `nemotron-3.5-bolt:e4b` | Lightweight | `/api/generate` | \- | Low-latency concise generation via standard text payloads. | 
| `qwen3.5:4b` | Logic & Math | `/api/generate` | Thinking | Captures both `response` and `thinking` tokens. | 
| `qwen2.5-coder:14b` | Coding | `/api/generate` | \- | Specialized in syntax highlighting, refactoring, and code debugging. | 
| `qwen2.5:7b` | Summary / Chat | `/api/generate` | Non-streaming | Standard non-streaming JSON; fast general text responses. | 
| `qwen2.5-coder:32b` | Large Architecture | `/api/generate` | 180s / Exclusive | Set timeout ≥ 180s; **run alone** with `keep_alive: 0`. | 
| `deepseek-r1:8b` | Deep Reasoning | `/api/generate` | Context ≤ 8192 | Bounded context; parses internal `<think>` reasoning blocks. | 
| `devstral:24b` | Developer Spec | `/api/generate` | Exclusive | Must unload preceding models to secure contiguous CUDA Host Buffers. | 
| `deepseek-r1:14b` | Algorithm | `/api/generate` | \- | Requires computation graph dynamic reallocation buffers. | 
| `llama3.1:8b` | Standard Text | `/api/generate` | Exclusive | Non-streaming prompt; requires isolated VRAM footprint. | 
| `qwen2.5:32b` | Long Writing | `/api/generate` | Exclusive | Near VRAM limits; runs isolated with `keep_alive: 0`. | 
| `deepseek-r1:32b` | Olympiad Proof | `/api/generate` | 120s / Context ≤ 8192 | Long chain-of-thought; requires ≥ 120s timeout and context ceilings. | 

### IV. Installation & Prerequisites

#### 1. Python Environment

Install library dependencies via `pip`:

```
pip install -r requirements.txt

```

> **Note**: Operating-system level executables (Docker, Node.js, Ollama, GPU drivers) cannot and must not be installed via `requirements.txt`. Configure them as detailed below.

#### 2. External System Dependencies

* **Ollama Service**:

  * Install and launch [Ollama](https://ollama.com/) (listening at `http://localhost:11434`).

  * Pull at least one general completion model (e.g., `ollama pull qwen2.5:7b`).

  * For parallel multi-model execution, launch Ollama with concurrency settings:

    ```
    $env:OLLAMA_MAX_LOADED_MODELS = "4"
    $env:OLLAMA_NUM_PARALLEL = "4"
    ollama serve
    
    ```

* **Node.js**:

  * Ensure Node.js (v18+, recommended LTS v22+) is available in your PATH for HTML5 game validation.

* **Code Execution Sandbox (Choose One)**:

  * **Option A (Docker Sandbox - Recommended)**:
    Start Docker Desktop and build the sandbox image from the repository root:

    ```
    docker pull node:22-alpine
    docker build -f execution/Dockerfile.python-sandbox -t omni-python-sandbox:latest .
    
    ```

  * **Option B (Host Pass-Through Mode - No Docker Required)**:
    For trusted local environments, bypass Docker by setting permanent user environment variables in PowerShell:

    ```
    [System.Environment]::SetEnvironmentVariable("OMNI_UNSAFE_LOCAL_PYTHON", "1", "User")
    [System.Environment]::SetEnvironmentVariable("OMNI_UNSAFE_LOCAL_NODE", "1", "User")
    
    ```

### V. Quick Start & Operation

1. **Launch Enterprise Console**:

   ```
   python main.py
   
   ```

2. **CLI Menu**:

   * `1`: **Launch Web Console** (Default URL: `http://127.0.0.1:8000`).

   * `2`: **Reset Tasks & Deliverables** (Wipes old deliveries, resets backlog to `TASK-001`).

   * `3`: Exit.

3. **Web Console Workflow**:

   * **Hardware Budget**: Set your VRAM/RAM limit at the top-right (e.g., `5.5`–`6.0` GB for RTX 3060; `12.0`–`16.0` GB when leveraging system RAM) and click "Apply".

   * **Create Task**: Fill in the title, description, iterations, and optional file attachments on the left panel, then click "Add to queue only".

   * **Start Processing**: Click "▶ Start / Resume" to trigger autonomous routing and parallel swarm execution.

   * **Review Deliverables**: Monitor real-time logs in the center panel; games, movies, and reports will render in the right-hand preview panel automatically upon completion.
