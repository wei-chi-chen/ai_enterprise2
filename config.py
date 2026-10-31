import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "enterprise_workspace"

# 四大核心專區
EMPLOYEE_ZONE = WORKSPACE_DIR / "01_staff"
TASK_ZONE = WORKSPACE_DIR / "02_tasks"
DELIVERY_ZONE = WORKSPACE_DIR / "03_delivery"
HISTORY_ZONE = WORKSPACE_DIR / "04_history"

FEEDBACK_DIR = TASK_ZONE / "feedback_queue"
UPLOAD_DIR = TASK_ZONE / "uploads"

ROSTER_FILE = EMPLOYEE_ZONE / "staff_roster.json"
STATUS_FILE = EMPLOYEE_ZONE / "employee_status.json"
CONTROL_FILE = TASK_ZONE / "daemon_control.json"
MEM_USAGE_FILE = TASK_ZONE / "memory_usage.json"
BACKLOG_FILE = TASK_ZONE / "task_backlog.json"
LESSONS_FILE = HISTORY_ZONE / "lessons_learned.json"

DEFAULT_GPU_BUDGET_GB = 6.0
DEFAULT_RAM_BUDGET_GB = 16.0
DEFAULT_MAX_MEMORY_GB = DEFAULT_GPU_BUDGET_GB + DEFAULT_RAM_BUDGET_GB
OLLAMA_API_URL = "http://localhost:11434"

# 雲端模型切換設定檔
CLOUD_CONFIG_FILE = EMPLOYEE_ZONE / ".cloud_config.json"

for p in [
    EMPLOYEE_ZONE, TASK_ZONE, DELIVERY_ZONE, HISTORY_ZONE,
    FEEDBACK_DIR, UPLOAD_DIR,
    DELIVERY_ZONE / "source_code",
    DELIVERY_ZONE / "projects",
    DELIVERY_ZONE / "reports",
    DELIVERY_ZONE / "cleaned_data",
    DELIVERY_ZONE / "illustrations",
    DELIVERY_ZONE / "games",
    DELIVERY_ZONE / "films",
    HISTORY_ZONE / "failure_audits",
    HISTORY_ZONE / "timeline_logs"
]:
    p.mkdir(parents=True, exist_ok=True)

if not ROSTER_FILE.exists():
    ROSTER_FILE.write_text(json.dumps({"gpu_budget_gb": DEFAULT_GPU_BUDGET_GB,
        "ram_budget_gb": DEFAULT_RAM_BUDGET_GB,
        "hardware_budget_gb": DEFAULT_MAX_MEMORY_GB}, indent=2), encoding="utf-8")

if not CONTROL_FILE.exists():
    CONTROL_FILE.write_text(json.dumps({
        "state": "IDLE",
        "pause_requested": False,
        "stop_requested": False
    }, indent=2), encoding="utf-8")

for f in [BACKLOG_FILE, LESSONS_FILE]:
    if not f.exists():
        f.write_text("[]", encoding="utf-8")

if not CLOUD_CONFIG_FILE.exists():
    CLOUD_CONFIG_FILE.write_text(json.dumps({
        "use_cloud": False,
        "google_api_key": "",
        "rate_limit_url": ""
    }, indent=2), encoding="utf-8")