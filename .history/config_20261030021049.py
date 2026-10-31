from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR / "enterprise_workspace"

EMPLOYEE_ZONE = WORKSPACE_DIR / "01_staff"
TASK_ZONE = WORKSPACE_DIR / "02_tasks"
DELIVERY_ZONE = WORKSPACE_DIR / "03_delivery"
HISTORY_ZONE = WORKSPACE_DIR / "04_history"

FEEDBACK_DIR = TASK_ZONE / "feedback_queue"
UPLOAD_DIR = TASK_ZONE / "uploads"

MAX_MEMORY_GB = 40.0
OLLAMA_API_URL = "http://localhost:11434"

for p in [
    EMPLOYEE_ZONE, TASK_ZONE, DELIVERY_ZONE, HISTORY_ZONE,
    FEEDBACK_DIR, UPLOAD_DIR,
    DELIVERY_ZONE / "source_code",
    DELIVERY_ZONE / "projects",
    DELIVERY_ZONE / "reports",
    DELIVERY_ZONE / "cleaned_data",
    DELIVERY_ZONE / "games",
    HISTORY_ZONE / "failure_audits",
    HISTORY_ZONE / "timeline_logs"
]:
    p.mkdir(parents=True, exist_ok=True)