import json
import shutil
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from config import DELIVERY_ZONE, BACKLOG_FILE, CONTROL_FILE, FEEDBACK_DIR

console = Console()

def launch_dashboard():
    console.print("[bold green]Starting Omni-Autonomous AI Enterprise Web Console v3.5...[/bold green]")
    console.print("[cyan]URL: [http://127.0.0.1:8000](http://127.0.0.1:8000)[/cyan]\n")
    uvicorn.run("ui.app:app", host="127.0.0.1", port=8000, reload=False)

def reset_workspace():
    BACKLOG_FILE.write_text("[]", encoding="utf-8")
    CONTROL_FILE.write_text(json.dumps({"state": "IDLE", "pause_requested": False, "stop_requested": False}, indent=2), encoding="utf-8")
    for f in FEEDBACK_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
    for sub in ["source_code", "projects", "reports", "cleaned_data", "games"]:
        p = DELIVERY_ZONE / sub
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    console.print("[bold green]✔ 已清空所有任務與交付區成果，任務序號重置回 TASK-001！[/bold green]\n")

if __name__ == "__main__":
    while True:
        console.clear()
        menu = """[bold cyan]Omni-Autonomous AI Enterprise v3.5 (大模型拆解 + 多模型並行全能版)[/bold cyan]
[1] Launch Web Console (啟動全能 Web 控制台 - [http://127.0.0.1:8000](http://127.0.0.1:8000))
[2] Reset Tasks & Deliverables (清空任務佇列與交付成果，重置為 TASK-001)
[3] Exit (退出)"""
        console.print(Panel(menu, border_style="cyan"))
        c = Prompt.ask("Select option", choices=["1", "2", "3"], default="1")
        if c == "1":
            launch_dashboard()
            break
        elif c == "2":
            reset_workspace()
            Prompt.ask("Press Enter to continue...")
        else:
            break