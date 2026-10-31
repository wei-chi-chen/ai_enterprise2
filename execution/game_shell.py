import html
import json


def assemble_game_html(title: str, custom_js_bundle: str, assets=None) -> str:
    safe_title = html.escape(title, quote=True)
    asset_urls = {
        str(name): f"assets/{name}"
        for name in (assets or [])
        if isinstance(name, str) and name and "/" not in name and "\\" not in name
    }
    asset_urls_json = json.dumps(asset_urls, ensure_ascii=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{safe_title}</title>
<style>
*{{box-sizing:border-box}}html,body{{width:100%;height:100%;margin:0}}
body{{display:flex;flex-direction:column;background:#101522;color:#f5f7fb;font:16px/1.45 system-ui,sans-serif}}
header{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 18px;background:#171e2d}}
h1{{font-size:clamp(1rem,3vw,1.5rem);margin:0}}#game-status{{color:#b8c5d8;font-size:.9rem}}
#game-controls{{display:flex;flex-wrap:wrap;gap:8px;padding:8px 18px}}
button{{border:1px solid #52627a;border-radius:8px;padding:8px 14px;background:#29364c;color:inherit;font:inherit;cursor:pointer}}
button:hover,button:focus-visible{{background:#3a4b67;outline:2px solid #8cc8ff}}
#game-root{{position:relative;display:flex;flex:1;min-height:0;overflow:hidden}}
canvas{{display:block;width:100%;height:100%;touch-action:none}}
@media(max-width:600px){{header{{padding:8px 10px}}#game-controls{{padding:6px 10px}}button{{padding:7px 10px}}}}
</style>
</head>
<body>
<header><h1>{safe_title}</h1><div id="game-status" role="status" aria-live="polite"></div></header>
<nav id="game-controls" aria-label="Game controls"></nav>
<main id="game-root"><canvas id="game-canvas" aria-label="Game area"></canvas></main>
<script>
window.G = {{}};
window.GameModules = window.GameModules || {{}};
window.GameAssetUrls = {asset_urls_json};
{custom_js_bundle}
</script>
<script>
(() => {{
  const canvas = document.getElementById('game-canvas');
  const context = canvas.getContext('2d');
  const status = document.getElementById('game-status');
  const controls = document.getElementById('game-controls');
  const modules = Object.keys(window.GameModules).sort().map(key => ({{ key, module: window.GameModules[key] }}));
  const assetUrls = window.GameAssetUrls || {{}};
  const normalizeAssetName = name => String(name).split(/[\\/\\\\]/).pop()
    .replace(/\\.[^.]*$/, '').toLowerCase().replace(/[^a-z0-9]/g, '');
  function resolveAssetName(name) {{
    if (typeof name !== 'string') return null;
    if (Object.prototype.hasOwnProperty.call(assetUrls, name)) return name;
    const requested = normalizeAssetName(name);
    if (!requested) return null;
    const names = Object.keys(assetUrls);
    const exact = names.filter(candidate => normalizeAssetName(candidate) === requested);
    if (exact.length === 1) return exact[0];
    const fuzzy = names.filter(candidate => {{
      const normalized = normalizeAssetName(candidate);
      return normalized.startsWith(requested) || requested.startsWith(normalized);
    }});
    return fuzzy.length === 1 ? fuzzy[0] : null;
  }}
  function prepareAudioFallback(audio) {{
    if (audio.__gameFallbackBound) return audio;
    audio.__gameFallbackBound = true;
    const nativePlay = audio.play.bind(audio);
    audio.play = (...args) => Promise.resolve(nativePlay(...args)).catch(error => {{
      status.textContent = error?.name === 'NotAllowedError'
        ? 'Audio playback requires a user gesture.'
        : 'Audio playback is unavailable.';
      return false;
    }});
    audio.addEventListener('error', () => {{
      audio.removeAttribute('src');
      try {{ audio.load(); }} catch (error) {{}}
    }}, {{ once: true }});
    return audio;
  }}
  function makeFallbackAsset(name) {{
    if (/\\.(?:wav|mp3|ogg|m4a|aac)$/i.test(String(name))) {{
      const audio = prepareAudioFallback(new Audio());
      audio.preload = 'none';
      audio.play = () => Promise.resolve();
      assets[String(name)] = audio;
      return audio;
    }}
    const canvasFallback = document.createElement('canvas');
    canvasFallback.width = 64;
    canvasFallback.height = 64;
    const fallbackContext = canvasFallback.getContext('2d');
    if (fallbackContext) {{
      fallbackContext.fillStyle = '#29364c';
      fallbackContext.fillRect(0, 0, 64, 64);
      fallbackContext.strokeStyle = '#b8c5d8';
      fallbackContext.lineWidth = 3;
      fallbackContext.beginPath();
      fallbackContext.moveTo(16, 46);
      fallbackContext.lineTo(28, 32);
      fallbackContext.lineTo(36, 41);
      fallbackContext.lineTo(42, 34);
      fallbackContext.lineTo(52, 46);
      fallbackContext.stroke();
    }}
    assets[String(name)] = canvasFallback;
    return canvasFallback;
  }}
  const assets = Object.fromEntries(Object.entries(assetUrls).map(([name, url]) => {{
    const audio = /\\.(?:wav|mp3|ogg|m4a|aac)$/i.test(url);
    if (audio) {{
      const sound = prepareAudioFallback(new Audio());
      sound.preload = 'none';
      sound.src = url;
      return [name, sound];
    }}
    const image = new Image();
    image.src = url;
    return [name, image];
  }}));
  const api = {{
    canvas,
    context,
    state: window.G,
    assets,
    assetUrls,
    get language() {{ return window.GameLanguage; }},
    loadAsset(name) {{
      const resolvedName = resolveAssetName(name);
      const url = resolvedName && assetUrls[resolvedName];
      if (!url) return Promise.resolve(makeFallbackAsset(name));
      if (/\\.(?:wav|mp3|ogg|m4a|aac)$/i.test(url)) {{
        const audio = prepareAudioFallback(assets[resolvedName] || new Audio());
        if (!audio.src) audio.src = url;
        audio.preload = 'auto';
        try {{
          audio.load();
        }} catch (error) {{
          audio.removeAttribute('src');
        }}
        assets[resolvedName] = audio;
        return Promise.resolve(audio);
      }}
      return new Promise((resolve, reject) => {{
        const fallback = () => {{
          const image = new Image();
          image.onload = () => resolve(image);
          image.onerror = () => reject(new Error(`Could not load game asset: ${{name}}`));
          image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64"><rect width="64" height="64" rx="8" fill="#29364c"/><path d="M16 46l12-14 8 9 6-7 10 12" fill="none" stroke="#b8c5d8" stroke-width="3"/></svg>'
          );
          assets[resolvedName] = image;
        }};
        const cachedImage = assets[resolvedName];
        const image = cachedImage || new Image();
        if (cachedImage?.complete) {{
          if (image.naturalWidth > 0) resolve(image);
          else fallback();
          return;
        }}
        image.onload = () => {{ assets[resolvedName] = image; resolve(image); }};
        image.onerror = fallback;
        if (!cachedImage) image.src = url;
      }});
    }},
    playAsset(name) {{
      return api.loadAsset(name).then(audio => {{
        if (typeof audio?.play !== 'function') return false;
        return Promise.resolve().then(() => audio.play()).then(
          result => result !== false,
          error => {{
            api.setStatus(error?.name === 'NotAllowedError'
              ? 'Audio playback requires a user gesture.'
              : 'Audio playback is unavailable.');
            return false;
          }}
        );
      }});
    }},
    setStatus(message) {{ status.textContent = String(message ?? ''); }},
    addControl(label, callback) {{
      if (typeof callback !== 'function') throw new TypeError('Game control callback must be a function.');
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = String(label);
      button.addEventListener('click', callback);
      controls.appendChild(button);
      return button;
    }},
    reset() {{
      for (const {{ module: gameModule }} of modules) gameModule.reset?.(api);
    }},
  }};
  let previousTime = 0;
  let animationStarted = false;
  function resize() {{
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(bounds.width * ratio));
    canvas.height = Math.max(1, Math.round(bounds.height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    for (const {{ module: gameModule }} of modules) gameModule.resize?.(api);
  }}
  function frame(now) {{
    const elapsed = previousTime ? Math.min(0.05, (now - previousTime) / 1000) : 0;
    previousTime = now;
    for (const {{ module: gameModule }} of modules) gameModule.update?.(elapsed, api);
    context.clearRect(0, 0, canvas.width, canvas.height);
    for (const {{ module: gameModule }} of modules) gameModule.render?.(api);
    window.requestAnimationFrame(frame);
  }}
  function start() {{
    if (animationStarted) return;
    if (!modules.length) throw new Error('No game modules registered.');
    animationStarted = true;
    try {{
      resize();
      for (const {{ key, module: gameModule }} of modules) {{
        if (typeof gameModule.init !== 'function') throw new TypeError(`Game module ${{key}} must define init(api).`);
        gameModule.init(api);
      }}
      notifyLanguageChange();
      window.addEventListener('resize', resize);
      window.addEventListener('keydown', event => {{
        for (const {{ module: gameModule }} of modules) gameModule.onKeyDown?.(event, api);
      }});
      canvas.addEventListener('pointerdown', event => {{
        for (const {{ module: gameModule }} of modules) gameModule.onPointerDown?.(event, api);
      }});
      window.requestAnimationFrame(frame);
    }} catch (error) {{
      api.setStatus(`Game startup failed: ${{error.message}}`);
      throw error;
    }}
  }}
  window.addEventListener('DOMContentLoaded', start, {{ once: true }});
  const applyLanguage = language => {{
    const english = String(language || '').toLowerCase().startsWith('en');
    document.documentElement.lang = english ? 'en' : 'zh-Hant';
    controls.setAttribute('aria-label', english ? 'Game controls' : '遊戲控制');
    canvas.setAttribute('aria-label', english ? 'Game area' : '遊戲區域');
    window.GameLanguage = english ? 'en' : 'zh';
    if (animationStarted) notifyLanguageChange();
  }};
  function notifyLanguageChange() {{
    for (const {{ module: gameModule }} of modules) {{
      gameModule.onLanguageChange?.(window.GameLanguage, api);
    }}
  }}
  try {{
    applyLanguage(localStorage.getItem('omni-language') || 'zh');
  }} catch (error) {{
    if (error?.name !== 'SecurityError') throw error;
    applyLanguage('zh');
  }}
  window.addEventListener('message', event => {{
    if (event.source === window.parent && event.data?.type === 'set-language') {{
      applyLanguage(event.data.language);
    }}
  }});
}})();
</script>
</body>
</html>"""
