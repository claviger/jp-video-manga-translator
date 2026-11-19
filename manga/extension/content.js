// Content script: adds a "translate" button over large <img> elements and swaps in the translated page.
// It never sees the service token; it only sends image URLs to the background worker.
(() => {
  if (window.__mltInjected) return;
  window.__mltInjected = true;

  const MIN_NATURAL_W = 300, MIN_NATURAL_H = 400, MIN_SHOWN = 180;
  const state = new WeakMap(); // img -> { btn, original: {src, srcset}, translated, key, busy }
  const tracked = new Set();

  // Buttons live in a closed shadow root so page CSS/JS can't restyle or read them easily.
  const host = document.createElement("div");
  host.style.cssText = "position:absolute;top:0;left:0;width:0;height:0;z-index:2147483647;";
  const root = host.attachShadow({ mode: "closed" });
  root.innerHTML = `<style>
    button{position:absolute;font:600 12px/1 system-ui,sans-serif;padding:6px 9px;border-radius:6px;border:1px solid #0003;
      background:#1d4ed8;color:#fff;cursor:pointer;opacity:.82;box-shadow:0 1px 4px #0005;white-space:nowrap}
    button:hover{opacity:1} button.busy{background:#6b7280;cursor:progress} button.done{background:#15803d}
    button.err{background:#b91c1c}
    .toast{position:fixed;right:16px;bottom:16px;max-width:420px;background:#111827;color:#f9fafb;padding:10px 12px;
      border-radius:8px;font:13px/1.4 system-ui,sans-serif;box-shadow:0 2px 10px #0006}
  </style>`;
  document.documentElement.appendChild(host);

  function toast(text, ms = 6000) {
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = text;
    root.appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  function isCandidate(img) {
    if (!img.complete || !img.naturalWidth) return false;
    if (img.naturalWidth < MIN_NATURAL_W || img.naturalHeight < MIN_NATURAL_H) return false;
    const r = img.getBoundingClientRect();
    return r.width >= MIN_SHOWN && r.height >= MIN_SHOWN;
  }

  function candidates() {
    return Array.from(document.images).filter((img) => tracked.has(img) || isCandidate(img));
  }

  function imgKey(img) {
    const s = state.get(img);
    if (s?.key) return s.key;
    const src = s?.original?.src || img.currentSrc || img.src;
    return src.startsWith("data:") || src.startsWith("blob:") ? "" : src;
  }

  function place(img) {
    const s = state.get(img);
    if (!s) return;
    const r = img.getBoundingClientRect();
    const visible = r.width > 0 && r.bottom > 0 && r.top < innerHeight;
    s.btn.style.display = visible ? "" : "none";
    s.btn.style.top = `${r.top + scrollY + 8}px`;
    s.btn.style.left = `${r.right + scrollX - s.btn.offsetWidth - 8}px`;
  }

  function setLabel(s, text, cls) {
    s.btn.textContent = text;
    s.btn.className = cls || "";
  }

  function track(img) {
    if (tracked.has(img)) return;
    tracked.add(img);
    const btn = document.createElement("button");
    const s = { btn, original: null, translated: null, key: "", busy: false, showing: "original" };
    state.set(img, s);
    setLabel(s, "Translate");
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (!ev.isTrusted) return; // ignore synthetic clicks from page scripts
      onClick(img, ev.shiftKey);
    });
    root.appendChild(btn);
    place(img);
  }

  async function readBytesInPage(src) {
    // data: and blob: URLs can't be fetched by the background worker; read them here.
    const r = await fetch(src);
    const buf = new Uint8Array(await r.arrayBuffer());
    let bin = "";
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(bin);
  }

  function prevKeyFor(img) {
    const list = candidates();
    const i = list.indexOf(img);
    return i > 0 ? imgKey(list[i - 1]) : "";
  }

  async function translateImg(img, noCache = false) {
    const s = state.get(img);
    if (!s || s.busy) return false;
    if (s.translated && !noCache) {
      showVersion(img, s.showing === "translated" ? "original" : "translated");
      return true;
    }
    s.busy = true;
    setLabel(s, "Translating…", "busy");
    place(img);
    try {
      if (!s.original) s.original = { src: img.currentSrc || img.src, srcset: img.getAttribute("srcset") };
      const src = s.original.src;
      const msg = { type: "translate", prevKey: prevKeyFor(img), noCache };
      if (src.startsWith("data:") || src.startsWith("blob:")) msg.bytesB64 = await readBytesInPage(src);
      else msg.imgUrl = src;
      const res = await chrome.runtime.sendMessage(msg);
      if (!res?.ok) throw Object.assign(new Error(res?.error || "translation failed"), { code: res?.code });
      s.key = res.key;
      s.translated = res.dataUrl;
      showVersion(img, "translated");
      return true;
    } catch (e) {
      setLabel(s, "Retry", "err");
      toast(`Manga translator: ${e.message}`, e.code === "no-permission" ? 10000 : 6000);
      return false;
    } finally {
      s.busy = false;
      place(img);
    }
  }

  function showVersion(img, which) {
    const s = state.get(img);
    if (which === "translated") {
      if (img.hasAttribute("srcset")) img.removeAttribute("srcset");
      img.src = s.translated;
      s.showing = "translated";
      setLabel(s, "Original", "done");
    } else {
      if (s.original.srcset != null) img.setAttribute("srcset", s.original.srcset);
      img.src = s.original.src;
      s.showing = "original";
      setLabel(s, "English", "");
    }
    place(img);
  }

  function onClick(img, shift) {
    // Shift+click re-translates, bypassing the service cache.
    translateImg(img, shift);
  }

  // Up to 3 pages in flight, issued in DOM order: the service overlaps one page's detection/OCR with another's
  // LLM call and still gives each page the previous page's translations as context.
  async function translateAll(concurrency = 3) {
    const list = candidates().filter((img) => !state.get(img)?.translated);
    const running = new Set();
    let stop = false;
    for (const img of list) {
      if (stop) break;
      track(img);
      while (running.size >= concurrency) await Promise.race(running);
      const p = translateImg(img).then((ok) => { if (!ok) stop = true; }).finally(() => running.delete(p));
      running.add(p);
    }
    await Promise.all(running);
  }

  // Readers that draw pages on <canvas> (often as shuffled tiles) can't be handled: there's no image URL, and
  // reading the canvas back is blocked when it's tainted, and the tiles are scrambled anyway.
  function canvasReport() {
    const big = Array.from(document.querySelectorAll("canvas")).filter((c) => c.width >= 600 && c.height >= 800);
    return big.length;
  }

  function scan() {
    for (const img of document.images) if (isCandidate(img)) track(img);
    for (const img of tracked) {
      if (!img.isConnected) {
        state.get(img)?.btn.remove();
        tracked.delete(img);
      } else {
        place(img);
      }
    }
  }

  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; scan(); }); };
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  document.addEventListener("load", (e) => { if (e.target instanceof HTMLImageElement) schedule(); }, true);
  new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true });
  scan();

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (sender.id !== chrome.runtime.id || sender.tab) return false;
    if (msg?.type === "translateAll") {
      translateAll();
      sendResponse({ ok: true });
    } else if (msg?.type === "pageInfo") {
      sendResponse({ images: candidates().length, canvases: canvasReport() });
    }
    return false;
  });

  if (!candidates().length && canvasReport()) {
    toast("Manga translator: this page draws on <canvas>, not <img>; it can't be translated (see README).", 8000);
  }
})();
