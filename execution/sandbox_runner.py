import ast
import json
import os
import re
import shutil
import subprocess
import sys
import sqlite3
from pathlib import Path


_GAME_RUNTIME_PROBE = r"""
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const ids = new Set(input.ids);
const elements = new Map();
const frameCallbacks = [];
const eventHandlers = new Map();
function makeElement(id = '') {
  const listeners = new Map();
  const context = new Proxy({
    setTransform() {}, resetTransform() {}, clearRect() {}, fillRect() {}, strokeRect() {},
    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, arc() {}, arcTo() {},
    ellipse() {}, roundRect() {}, rect() {}, quadraticCurveTo() {}, bezierCurveTo() {},
    clip() {}, isPointInPath() { return false; }, isPointInStroke() { return false; },
    fill() {}, stroke() {}, save() {}, restore() {}, translate() {},
    rotate() {}, scale() {}, drawImage() {}, fillText() {}, strokeText() {},
    setLineDash() {}, putImageData() {},
    createLinearGradient() { return { addColorStop() {} }; },
    createRadialGradient() { return { addColorStop() {} }; },
    createConicGradient() { return { addColorStop() {} }; },
    createPattern() { return {}; },
    createImageData(width, height) {
      return { width, height, data: new Uint8ClampedArray(width * height * 4) };
    },
    getImageData(_x, _y, width, height) {
      return { width, height, data: new Uint8ClampedArray(width * height * 4) };
    },
    measureText(text) { return { width: String(text).length * 8 }; },
  }, { get(target, key) { return key in target ? target[key] : () => {}; } });
  const element = {
    id, tagName: 'CANVAS', textContent: '', innerText: '', value: '', width: 800, height: 600,
    clientWidth: 800, clientHeight: 600, style: {}, dataset: {}, complete: true,
    naturalWidth: 64, naturalHeight: 64, currentTime: 0, duration: 1, paused: true,
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(type, fn) { listeners.set(type, fn); },
    removeEventListener(type) { listeners.delete(type); },
    dispatchEvent(event) { listeners.get(event.type)?.(event); },
    appendChild() {}, append() {}, replaceChildren() {}, setAttribute() {},
    getAttribute() { return null; }, getBoundingClientRect() {
      return { x: 0, y: 0, left: 0, top: 0, width: 800, height: 600 };
    },
    getContext() { return context; }, focus() {}, click() {},
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; }, load() {},
  };
  Object.defineProperty(element, 'src', {
    get() { return this._src || ''; },
    set(value) {
      this._src = String(value);
      this.onload?.();
      listeners.get('load')?.({ type: 'load', target: this });
    },
  });
  return element;
}
class StubPath2D {
  addPath() {} closePath() {} moveTo() {} lineTo() {} arc() {} arcTo() {}
  ellipse() {} rect() {} roundRect() {} quadraticCurveTo() {} bezierCurveTo() {}
}
class StubOffscreenCanvas {
  constructor(width, height) {
    const canvas = makeElement();
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
}
class StubAudio {
  constructor(src = '') {
    const audio = makeElement();
    audio.tagName = 'AUDIO';
    audio.src = src;
    return audio;
  }
}
class StubImage {
  constructor(width = 64, height = 64) {
    const image = makeElement();
    image.tagName = 'IMG';
    image.width = width;
    image.height = height;
    return image;
  }
}
class StubHTMLElement {}
for (const id of ids) elements.set(id, makeElement(id));
const document = {
  documentElement: makeElement('documentElement'),
  body: makeElement('body'),
  getElementById(id) { return elements.get(id) || null; },
  createElement(tagName = '') {
    const element = makeElement();
    element.tagName = String(tagName).toUpperCase();
    return element;
  },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  addEventListener(type, fn) {
    const list = eventHandlers.get(type) || [];
    list.push(fn);
    eventHandlers.set(type, list);
  },
};
const sandbox = {
  document, console: { log() {}, warn() {}, error() {} },
  Image: StubImage, Audio: StubAudio, Path2D: StubPath2D, HTMLElement: StubHTMLElement,
  OffscreenCanvas: StubOffscreenCanvas,
  customElements: {
    registry: new Map(),
    define(name, constructor) { this.registry.set(name, constructor); },
    get(name) { return this.registry.get(name); },
    whenDefined() { return Promise.resolve(); },
  },
  performance: { now() { return 16; } },
  requestAnimationFrame(fn) { frameCallbacks.push(fn); return frameCallbacks.length; },
  setTimeout() { return 1; }, clearTimeout() {}, setInterval() { return 1; },
  clearInterval() {}, localStorage: { getItem() { return null; }, setItem() {} },
  location: { href: 'http://localhost/game.html', search: '' },
  navigator: { userAgent: 'Node game validation' },
  devicePixelRatio: 1, innerWidth: 800, innerHeight: 600,
  addEventListener(type, fn) {
    const list = eventHandlers.get(type) || [];
    list.push(fn);
    eventHandlers.set(type, list);
  },
};
sandbox.window = sandbox;
sandbox.parent = sandbox;
vm.createContext(sandbox, { codeGeneration: { strings: false, wasm: false } });
for (const script of input.scripts) {
  vm.runInContext(script, sandbox, { timeout: 3000, filename: 'game-inline.js' });
}
for (const handler of eventHandlers.get('DOMContentLoaded') || []) {
  handler({ type: 'DOMContentLoaded' });
}
const firstFrame = frameCallbacks.shift();
if (typeof firstFrame === 'function') firstFrame(16);
process.stdout.write('runtime-ok');
"""


_DOCKER_INFRASTRUCTURE_ERRORS = re.compile(
    r"cannot connect to the docker daemon|"
    r"error during connect|"
    r"is the docker daemon running|"
    r"dockerdesktoplinuxengine|"
    r"docker_engine.*(?:pipe|not found)|"
    r"failed to connect to the docker api|"
    r"error response from daemon|"
    r"no such image|"
    r"pull access denied|"
    r"invalid mount config|"
    r"mounts denied",
    re.IGNORECASE,
)


def _docker_mount_source(path: Path) -> str:
    source = os.fspath(path)
    if not re.match(r"^[A-Za-z]:[\\/]", source):
        source = str(Path(path).resolve())
    return source.replace("\\", "/") if re.match(r"^[A-Za-z]:[\\/]", source) else source


def _execution_result(
    success: bool,
    stdout: str = "",
    stderr: str = "",
    *,
    execution_mode: str,
    error_type: str | None = None,
    warning: str = "",
) -> dict:
    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "execution_mode": execution_mode,
        "sandboxed": execution_mode == "docker",
        "error_type": error_type,
        "infrastructure_error": error_type in {"infrastructure", "dependency"},
        "warning": warning,
    }


def _docker_result(result, *, python: bool = False) -> dict:
    output = result.stdout or ""
    error = result.stderr or ""
    combined = f"{error}\n{output}"
    if python and re.search(r"\b(?:ModuleNotFoundError|No module named)\b", combined):
        image = os.environ.get("OMNI_PYTHON_SANDBOX_IMAGE", "omni-python-sandbox:latest")
        return _execution_result(
            False,
            output,
            (
                f"{error.rstrip()}\nPython sandbox image {image!r} is missing a project "
                "dependency. Use a prebuilt image with the required packages via "
                "OMNI_PYTHON_SANDBOX_IMAGE; package installation is disabled in the "
                "network-isolated sandbox."
            ).strip(),
            execution_mode="docker",
            error_type="dependency",
        )
    if result.returncode:
        infrastructure_error = (
            result.returncode in {125, 126, 127}
            or bool(_DOCKER_INFRASTRUCTURE_ERRORS.search(combined))
        )
        if infrastructure_error:
            error = (
                "Docker infrastructure failure (not a test failure; the container did not "
                "start or complete successfully): "
                f"{error.strip()}"
            ).strip()
            if re.search(r"no such image|pull access denied", combined, re.IGNORECASE):
                error += (
                    "\nBuild the default Python sandbox image from the project root with: "
                    "docker build -f execution/Dockerfile.python-sandbox "
                    "-t omni-python-sandbox:latest . "
                    "Host execution is intentionally not enabled as an automatic fallback."
                )
        return _execution_result(
            False,
            output,
            error,
            execution_mode="docker",
            error_type="infrastructure" if infrastructure_error else "script",
        )
    return _execution_result(True, output, error, execution_mode="docker")


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
    def is_valid_javascript(code_str: str) -> bool:
        node = shutil.which("node")
        if not node or not code_str.strip():
            return False
        try:
            result = subprocess.run(
                [node, "--check"],
                input=code_str,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0

    @staticmethod
    def run_python_script(script_path: Path, timeout: float = 120.0) -> dict:
        script_path = Path(script_path)
        if script_path.is_symlink() or not script_path.is_file():
            return {"success": False, "stdout": "", "stderr": "Python script must be a regular file."}
        script_path = script_path.resolve()
        docker = shutil.which("docker")
        unsafe_local = os.environ.get("OMNI_UNSAFE_LOCAL_PYTHON") == "1"
        if unsafe_local:
            warning = (
                "WARNING: Running Python on the host without sandbox isolation because "
                "OMNI_UNSAFE_LOCAL_PYTHON=1."
            )
            try:
                res = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(script_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return _execution_result(
                    res.returncode == 0,
                    res.stdout,
                    res.stderr,
                    execution_mode="host-unsandboxed",
                    error_type=None if res.returncode == 0 else "script",
                    warning=warning,
                )
            except subprocess.TimeoutExpired:
                return _execution_result(
                    False,
                    stderr=f"Execution exceeded the {timeout:g}s timeout.",
                    execution_mode="host-unsandboxed",
                    error_type="timeout",
                    warning=warning,
                )
            except OSError as exc:
                return _execution_result(
                    False,
                    stderr=str(exc),
                    execution_mode="host-unsandboxed",
                    error_type="infrastructure",
                    warning=warning,
                )

        if docker:
            image = os.environ.get("OMNI_PYTHON_SANDBOX_IMAGE", "omni-python-sandbox:latest")
            command = [
                docker, "run", "--rm", "--pull=never",
                "--network=none", "--read-only",
                "--memory=2g", "--cpus=1", "--pids-limit=64",
                "--security-opt=no-new-privileges", "--cap-drop=ALL",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
                "--mount", (
                    f"type=bind,source={_docker_mount_source(script_path.parent)},"
                    "target=/workspace,readonly"
                ),
                "--workdir", "/workspace",
                image, "python", script_path.name,
            ]
            try:
                res = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return _docker_result(res, python=True)
            except subprocess.TimeoutExpired:
                return _execution_result(
                    False,
                    stderr=f"Execution exceeded the {timeout:g}s Docker sandbox timeout.",
                    execution_mode="docker",
                    error_type="timeout",
                )
            except OSError as exc:
                return _execution_result(
                    False,
                    stderr=f"Docker sandbox infrastructure failure: {exc}",
                    execution_mode="docker",
                    error_type="infrastructure",
                )

        return _execution_result(
            False,
            stderr=(
                "Docker CLI is unavailable; refusing to execute generated Python without "
                "explicit host-execution consent. To allow host execution without sandbox "
                "isolation, set "
                "OMNI_UNSAFE_LOCAL_PYTHON=1."
            ),
            execution_mode="none",
            error_type="infrastructure",
        )

    @staticmethod
    def run_node_script(script_path: Path, timeout: float = 120.0) -> dict:
        script_path = Path(script_path)
        if script_path.is_symlink() or not script_path.is_file():
            return {"success": False, "stdout": "", "stderr": "Node.js script must be a regular file."}
        script_path = script_path.resolve()
        docker = shutil.which("docker")
        unsafe_local = os.environ.get("OMNI_UNSAFE_LOCAL_NODE") == "1"
        if unsafe_local:
            node = shutil.which("node")
            if not node:
                return _execution_result(
                    False,
                    stderr="Node.js is not installed.",
                    execution_mode="host-unsandboxed",
                    error_type="infrastructure",
                    warning=(
                        "Host execution was explicitly requested, but no Node.js runtime "
                        "is installed."
                    ),
                )
            warning = (
                "WARNING: Running Node.js on the host without sandbox isolation because "
                "OMNI_UNSAFE_LOCAL_NODE=1."
            )
            command = [node, str(script_path)]
            execution_mode = "host-unsandboxed"
        elif docker:
            image = os.environ.get("OMNI_NODE_SANDBOX_IMAGE", "node:22-alpine")
            command = [
                docker, "run", "--rm", "--pull=never",
                "--network=none", "--read-only",
                "--memory=1g", "--cpus=1", "--pids-limit=64",
                "--security-opt=no-new-privileges", "--cap-drop=ALL",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
                "--mount", (
                    f"type=bind,source={_docker_mount_source(script_path.parent)},"
                    "target=/workspace,readonly"
                ),
                "--workdir", "/workspace",
                image, "node", script_path.name,
            ]
            execution_mode = "docker"
            warning = ""
        else:
            return _execution_result(
                False,
                stderr=(
                    "Docker CLI is unavailable; refusing to execute generated JavaScript "
                    "without explicit host-execution consent. To allow host execution "
                    "without sandbox isolation, set "
                    "OMNI_UNSAFE_LOCAL_NODE=1."
                ),
                execution_mode="none",
                error_type="infrastructure",
            )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if execution_mode == "docker":
                return _docker_result(result)
            return _execution_result(
                result.returncode == 0,
                result.stdout,
                result.stderr,
                execution_mode=execution_mode,
                error_type=None if result.returncode == 0 else "script",
                warning=warning,
            )
        except subprocess.TimeoutExpired:
            return _execution_result(
                False,
                stderr=f"Node.js execution exceeded the {timeout:g}s {execution_mode} timeout.",
                execution_mode=execution_mode,
                error_type="timeout",
                warning=warning,
            )
        except OSError as exc:
            return _execution_result(
                False,
                stderr=f"Node.js execution infrastructure failure: {exc}",
                execution_mode=execution_mode,
                error_type="infrastructure",
                warning=warning,
            )

    @staticmethod
    def run_sql_script(script_path: Path) -> dict:
        try:
            content = Path(script_path).read_text(encoding="utf-8")
            if re.search(
                r"\b(?:ATTACH|DETACH)\b|\bVACUUM\b[^;]*\bINTO\b|\bPRAGMA\b",
                content,
                re.IGNORECASE,
            ):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "SQL contains a file-access or stateful SQLite command.",
                }
            connection = sqlite3.connect(":memory:")
            try:
                connection.executescript(content)
            finally:
                connection.close()
            return {"success": True, "stdout": "SQLite migration executed in memory.", "stderr": ""}
        except (OSError, UnicodeError, sqlite3.Error) as exc:
            return {"success": False, "stdout": "", "stderr": str(exc)}

    @staticmethod
    def validate_project_files(project_dir: Path, relative_paths: list[str]) -> dict:
        project_root = Path(project_dir).resolve()
        typescript_syntax_fallback = False
        for relative_path in relative_paths:
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts or "\\" in relative_path:
                return {"success": False, "stderr": f"Unsafe project path: {relative_path}"}
            file_path = project_root / path
            try:
                file_path.resolve().relative_to(project_root)
            except ValueError:
                return {"success": False, "stderr": f"Project file escapes its root: {relative_path}"}
            if file_path.is_symlink() or not file_path.is_file():
                return {"success": False, "stderr": f"Missing project file: {relative_path}"}
            suffix = file_path.suffix.casefold()
            try:
                content = file_path.read_text(encoding="utf-8")
                if suffix in {".js", ".mjs", ".cjs"}:
                    node = shutil.which("node")
                    if not node:
                        return {"success": False, "stderr": "Node.js is required to validate JavaScript files."}
                    result = subprocess.run(
                        [node, "--check", str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if result.returncode:
                        return {"success": False, "stderr": result.stderr.strip()}
                elif suffix in {".ts", ".tsx", ".jsx"}:
                    tsc = shutil.which("tsc")
                    if not tsc:
                        node = shutil.which("node")
                        if suffix == ".ts" and node:
                            result = subprocess.run(
                                [
                                    node,
                                    "--experimental-vm-modules",
                                    "-e",
                                    (
                                        "const fs=require('node:fs');"
                                        "const {stripTypeScriptTypes}=require('node:module');"
                                        "const vm=require('node:vm');"
                                        "if(typeof stripTypeScriptTypes!=='function')process.exit(2);"
                                        "new vm.SourceTextModule("
                                        "stripTypeScriptTypes(fs.readFileSync(process.argv[1],'utf8')));"
                                    ),
                                    str(file_path),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            if result.returncode:
                                details = result.stderr.strip() or result.stdout.strip()
                                return {
                                    "success": False,
                                    "stderr": (
                                        f"TypeScript validation fallback failed for {relative_path}: "
                                        f"{details}"
                                    ),
                                }
                            typescript_syntax_fallback = True
                            continue
                        return {
                            "success": False,
                            "stderr": (
                                "TypeScript compiler (tsc) is unavailable. Plain .ts files can "
                                "use Node.js 22.13+ as a limited syntax-only fallback; .tsx "
                                "and .jsx require tsc."
                            ),
                        }
                    compiler_args = [
                        "--noEmit", "--skipLibCheck", "--target", "ES2020",
                        "--jsx", "preserve",
                    ]
                    if suffix == ".jsx":
                        compiler_args.extend(["--allowJs", "--checkJs", "false"])
                    result = subprocess.run(
                        [tsc, *compiler_args, str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode:
                        return {"success": False, "stderr": result.stderr.strip()}
                elif suffix == ".sql":
                    result = SandboxRunner.run_sql_script(file_path)
                    if not result["success"]:
                        return {"success": False, "stderr": f"{relative_path}: {result['stderr']}"}
                elif suffix == ".json":
                    json.loads(content)
                elif file_path.name.casefold() == "dockerfile" or file_path.name.casefold().startswith("dockerfile."):
                    if not re.search(r"^\s*FROM\s+\S+", content, re.IGNORECASE | re.MULTILINE):
                        return {"success": False, "stderr": f"{relative_path} must contain a FROM instruction."}
            except (OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error) as exc:
                return {"success": False, "stderr": f"Invalid {relative_path}: {exc}"}
            except subprocess.TimeoutExpired:
                return {"success": False, "stderr": f"Validation timed out for {relative_path}."}
        stdout = "Project support files passed static validation."
        if typescript_syntax_fallback:
            stdout = (
                "Project support files passed available static validation; plain .ts syntax "
                "was checked by Node.js without TypeScript type checking."
            )
        return {
            "success": True,
            "stdout": stdout,
            "stderr": "",
        }

    @staticmethod
    def validate_html_game(html_path: Path) -> dict:
        if not html_path.exists():
            return {"success": False, "stderr": "找不到 index.html 遊戲入口檔"}
        try:
            content = html_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return {"success": False, "stderr": f"無法讀取遊戲入口檔：{exc}"}
        if content.lstrip().startswith("Error:") or "</html>" not in content.lower():
            return {"success": False, "stderr": "遊戲入口檔不是完整 HTML 文件"}
        if not re.search(r"<canvas\b", content, re.IGNORECASE):
            return {"success": False, "stderr": "HTML 遊戲缺少 <canvas> 標籤"}
        script_tags = re.findall(
            r"<script\b([^>]*)>(.*?)</script\s*>", content, re.IGNORECASE | re.DOTALL
        )
        if not script_tags:
            return {"success": False, "stderr": "HTML 遊戲沒有可驗證的 JavaScript"}

        node = shutil.which("node")
        if not node:
            return {"success": False, "stderr": "無法執行遊戲驗證：找不到 Node.js"}
        scripts = []
        for attributes, inline_code in script_tags:
            if re.search(r"\btype\s*=\s*['\"]?module\b", attributes, re.IGNORECASE):
                return {"success": False, "stderr": "目前沙盒無法驗證 JavaScript module 腳本"}
            source_match = re.search(r"\bsrc\s*=\s*['\"]([^'\"]+)['\"]", attributes, re.IGNORECASE)
            if source_match:
                source = source_match.group(1)
                if re.match(r"^(?:https?:)?//", source, re.IGNORECASE):
                    return {"success": False, "stderr": "無法離線驗證遊戲引用的外部 JavaScript"}
                script_path = (html_path.parent / source).resolve()
                try:
                    script_path.relative_to(html_path.parent.resolve())
                    inline_code = script_path.read_text(encoding="utf-8")
                except (OSError, ValueError, UnicodeError) as exc:
                    return {"success": False, "stderr": f"無法讀取遊戲 JavaScript {source}: {exc}"}
            if inline_code.strip():
                syntax = subprocess.run(
                    [node, "--check"], input=inline_code, capture_output=True,
                    text=True, timeout=10,
                )
                if syntax.returncode:
                    return {"success": False, "stderr": f"JavaScript 語法錯誤：{syntax.stderr.strip()}"}
                scripts.append(inline_code)

        if not scripts:
            return {"success": False, "stderr": "HTML 遊戲沒有可執行的 JavaScript"}
        ids = re.findall(r"\bid\s*=\s*['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
        runtime_input = json.dumps({
            "scripts": scripts,
            "ids": ids,
        }, ensure_ascii=False)
        try:
            runtime = subprocess.run(
                [node, "-e", _GAME_RUNTIME_PROBE],
                input=runtime_input,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "stderr": "遊戲初始化或首幀執行超時"}
        if runtime.returncode:
            return {
                "success": False,
                "stderr": f"JavaScript 執行期錯誤：{runtime.stderr.strip() or runtime.stdout.strip()}",
            }
        return {"success": True, "stdout": "遊戲 JavaScript 語法與初始化首幀執行檢驗通過"}