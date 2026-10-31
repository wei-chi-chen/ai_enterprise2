import ast
import subprocess
import sys
from pathlib import Path

class SandboxRunner:
    @staticmethod
    def is_valid_python(code_str: str) -> bool:
        if not code_str or len(code_str.strip()) < 10:
            return False
        try:
            ast.parse(code_str)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def run_python_script(script_path: Path, timeout: float = 30.0) -> dict:
        try:
            res = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": res.returncode == 0,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Execution Timeout (Infinite loop or hang detected)"}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}

    @staticmethod
    def validate_html_game(html_path: Path) -> dict:
        if not html_path.exists():
            return {"success": False, "stderr": "找不到 index.html 遊戲入口檔"}
        content = html_path.read_text(encoding="utf-8")
        if "<canvas" not in content.lower():
            return {"success": False, "stderr": "HTML 遊戲缺少 <canvas> 標籤"}
        if "<script" not in content.lower() or "requestAnimationFrame" not in content:
            return {"success": False, "stderr": "遊戲缺少 requestAnimationFrame 主迴圈"}
        if "spawn" not in content.lower() and "button" not in content.lower():
            return {"success": False, "stderr": "遊戲缺少互動召喚按鈕與產兵綁定函式"}
        return {"success": True, "stdout": "HTML5 Canvas 遊戲結構檢驗通過"}