const $ = (id) => document.getElementById(id);

function originPattern(url) {
  const u = new URL(url);
  return `${u.protocol}//${u.host}/*`;
}

// Runs in the page (via activeTab) to list where its large images come from.
function scanImageOrigins() {
  const out = new Set();
  for (const img of document.images) {
    if (img.naturalWidth >= 300 && img.naturalHeight >= 400) {
      const src = img.currentSrc || img.src;
      if (/^https?:/.test(src)) out.add(new URL(src).origin);
    }
  }
  const canvases = Array.from(document.querySelectorAll("canvas")).filter((c) => c.width >= 600 && c.height >= 800).length;
  return { origins: Array.from(out), canvases, injected: !!window.__mltInjected };
}

async function main() {
  const health = await chrome.runtime.sendMessage({ type: "health" });
  if (health?.ok) {
    const model = health.llm_loaded ? `LLM: ${health.llm_loaded}` : `LLM: ${health.llm_model} not loaded yet`;
    $("status").innerHTML = `<span class="good">Service OK.</span> <span class="muted"></span>`;
    $("status").querySelector(".muted").textContent = model + (health.llm_error ? ` (${health.llm_error})` : "");
  } else {
    $("status").innerHTML = `<span class="bad">Service: </span>`;
    $("status").append(health?.error || "unknown error");
  }

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url || !/^https?:/.test(tab.url)) {
    $("site").textContent = "Open a manga page (http/https) to use this.";
    return;
  }
  let info = { origins: [], canvases: 0, injected: false };
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: scanImageOrigins });
    info = res.result;
  } catch (e) {
    $("site").textContent = `Can't access this tab: ${e.message}`;
    return;
  }
  const pageOrigin = originPattern(tab.url);
  const wanted = [pageOrigin, ...info.origins.map((o) => o + "/*")].filter((v, i, a) => a.indexOf(v) === i);
  const granted = await chrome.permissions.contains({ origins: wanted });

  const site = $("site");
  site.innerHTML = "";
  const p = document.createElement("p");
  p.className = "muted";
  p.textContent = info.origins.length
    ? `Large images on this page come from:`
    : `No large <img> elements found yet.` + (info.canvases ? ` ${info.canvases} large <canvas> found: canvas/tiled readers aren't supported.` : "");
  site.append(p);
  if (info.origins.length) {
    const ul = document.createElement("ul");
    for (const o of info.origins) { const li = document.createElement("li"); li.textContent = o; ul.append(li); }
    site.append(ul);
  }

  if (!granted) {
    $("enable").hidden = false;
    $("enable").onclick = async () => {
      // Must run directly in the click handler (user gesture) for the permission prompt.
      const ok = await chrome.permissions.request({ origins: wanted });
      if (!ok) return;
      await chrome.runtime.sendMessage({ type: "enableSite", pageOrigin, tabId: tab.id });
      window.close();
    };
  } else {
    await chrome.runtime.sendMessage({ type: "enableSite", pageOrigin, tabId: info.injected ? null : tab.id });
    $("all").hidden = false;
    $("disable").hidden = false;
  }
  if (!info.injected && !granted) $("inject").hidden = false;
  $("inject").onclick = async () => {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    window.close();
  };
  $("all").onclick = async () => {
    await chrome.tabs.sendMessage(tab.id, { type: "translateAll" });
    window.close();
  };
  $("disable").onclick = async () => {
    await chrome.runtime.sendMessage({ type: "disableSite", pageOrigin });
    await chrome.permissions.remove({ origins: [pageOrigin] }).catch(() => {});
    window.close();
  };
}

main();
