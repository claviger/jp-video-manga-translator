// Service worker. The only component that knows the service token or talks to the local service.
// Content scripts ask it (via runtime messages) to translate an image URL; web pages can't message it
// (no externally_connectable) and can't read chrome.storage.

const SERVICE = "http://127.0.0.1:8765";
const REFERER_RULE_ID = 1;

async function getToken() {
  const { token } = await chrome.storage.local.get("token");
  return token || "";
}

function originPattern(url) {
  const u = new URL(url);
  return `${u.protocol}//${u.host}/*`;
}

async function hasHostAccess(url) {
  try {
    return await chrome.permissions.contains({ origins: [originPattern(url)] });
  } catch {
    return false;
  }
}

// Many image CDNs check Referer. Requests from the service worker carry none, so for the one image we're about
// to fetch, set Referer to the page that showed it. Session rule, scoped to requests this extension initiates.
async function setReferer(imgUrl, pageUrl) {
  await chrome.declarativeNetRequest.updateSessionRules({
    removeRuleIds: [REFERER_RULE_ID],
    addRules: [{
      id: REFERER_RULE_ID,
      priority: 1,
      action: { type: "modifyHeaders", requestHeaders: [{ header: "Referer", operation: "set", value: pageUrl }] },
      condition: {
        urlFilter: "|" + imgUrl + "|",
        initiatorDomains: [chrome.runtime.id],
        resourceTypes: ["xmlhttprequest"],
      },
    }],
  });
}

async function fetchImage(imgUrl, pageUrl) {
  const u = new URL(imgUrl);
  if (u.protocol !== "https:" && u.protocol !== "http:") throw new Error("unsupported URL scheme");
  if (!(await hasHostAccess(imgUrl))) {
    const e = new Error(`No access to ${u.host}. Open the extension popup and click "Enable on this site".`);
    e.code = "no-permission";
    e.origin = originPattern(imgUrl);
    throw e;
  }
  try { await setReferer(imgUrl, pageUrl); } catch (e) { console.warn("referer rule failed", e); }
  const r = await fetch(imgUrl, { credentials: "include", cache: "force-cache" });
  if (!r.ok) throw new Error(`image fetch failed: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function bytesToB64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function sha256Hex(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

// imageKey (URL, or "bytes:<sha>") -> sha256 of the image bytes, so the next page can send the previous page's
// hash as context. Session storage: cleared when the browser closes.
async function rememberHash(key, hash) {
  if (!key) return;
  const { hashes = {} } = await chrome.storage.session.get("hashes");
  hashes[key] = hash;
  const keys = Object.keys(hashes);
  if (keys.length > 2000) for (const k of keys.slice(0, keys.length - 2000)) delete hashes[k];
  await chrome.storage.session.set({ hashes });
}

async function lookupHash(key) {
  if (!key) return "";
  const { hashes = {} } = await chrome.storage.session.get("hashes");
  return hashes[key] || "";
}

// Fetch/hash/issue one page at a time, in order (keeps the Referer rule unambiguous and the service's page order).
let chain = Promise.resolve();
function serial(fn) {
  const p = chain.then(fn, fn);
  chain = p.catch(() => {});
  return p;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function translate(msg, sender) {
  const token = await getToken();
  if (!token) throw new Error("No service token set. Open the extension options and paste service/token.txt.");
  const pageUrl = sender.tab?.url || sender.url;
  // In order: fetch the image, hash it, remember the hash (the next page's context key), and start the service
  // request. The service overlaps pages and waits for the previous page's translation itself, as long as requests
  // arrive in page order, hence the small gap. Waiting for the response happens outside this section.
  const { bytes, key, pending } = await serial(async () => {
    const bytes = msg.bytesB64 ? b64ToBytes(msg.bytesB64) : await fetchImage(msg.imgUrl, pageUrl);
    const hash = await sha256Hex(bytes);
    const key = msg.imgUrl || "bytes:" + hash;
    const prevHash = await lookupHash(msg.prevKey);
    await rememberHash(key, hash);
    const headers = { Authorization: `Bearer ${token}`, "Content-Type": "application/octet-stream" };
    if (prevHash) headers["X-Prev-Hash"] = prevHash;
    if (msg.noCache) headers["X-No-Cache"] = "1";
    const pending = fetch(`${SERVICE}/v1/translate`, { method: "POST", headers, body: bytes });
    pending.catch(() => {}); // handled below; avoid an unhandled-rejection warning while we sleep
    await sleep(300);
    return { bytes, key, pending };
  });
  let r;
  try {
    r = await pending;
  } catch (e) {
    throw new Error("Local service not reachable on 127.0.0.1:8765. Is it running?");
  }
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).error; } catch {}
    throw new Error(`service: HTTP ${r.status} ${detail || ""}`.trim());
  }
  const out = new Uint8Array(await r.arrayBuffer());
  return {
    ok: true,
    key,
    dataUrl: "data:image/png;base64," + bytesToB64(out),
    cache: r.headers.get("X-Cache"),
    regions: r.headers.get("X-Regions"),
  };
}

async function health() {
  const token = await getToken();
  if (!token) return { ok: false, error: "no token set" };
  try {
    const r = await fetch(`${SERVICE}/v1/health`, { headers: { Authorization: `Bearer ${token}` } });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
    return await r.json();
  } catch {
    return { ok: false, error: "service not reachable" };
  }
}

async function enableSite(pageOrigin, tabId) {
  // Called from the popup after it obtained the host permissions. Register the content script for this site so
  // buttons appear automatically next time, and inject it into the current tab now.
  const id = "mlt-" + pageOrigin.replace(/[^a-z0-9]/gi, "_");
  const existing = await chrome.scripting.getRegisteredContentScripts({ ids: [id] });
  if (!existing.length) {
    await chrome.scripting.registerContentScripts([{
      id, matches: [pageOrigin], js: ["content.js"], runAt: "document_idle", persistAcrossSessions: true,
    }]);
  }
  if (tabId != null) await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
}

async function disableSite(pageOrigin) {
  const id = "mlt-" + pageOrigin.replace(/[^a-z0-9]/gi, "_");
  try { await chrome.scripting.unregisterContentScripts({ ids: [id] }); } catch {}
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  const fromPopupOrOptions = !sender.tab && sender.url?.startsWith(`chrome-extension://${chrome.runtime.id}/`);
  let p;
  switch (msg?.type) {
    case "translate":
      if (!sender.tab) return false; // only content scripts translate
      p = translate(msg, sender);
      break;
    case "health":
      p = health();
      break;
    case "enableSite":
      if (!fromPopupOrOptions) return false;
      p = enableSite(msg.pageOrigin, msg.tabId).then(() => ({ ok: true }));
      break;
    case "disableSite":
      if (!fromPopupOrOptions) return false;
      p = disableSite(msg.pageOrigin).then(() => ({ ok: true }));
      break;
    default:
      return false;
  }
  p.then(sendResponse, (e) => sendResponse({ ok: false, error: e.message, code: e.code, origin: e.origin }));
  return true;
});
