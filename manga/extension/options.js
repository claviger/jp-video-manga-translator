const tokenEl = document.getElementById("token");
const msg = document.getElementById("msg");

chrome.storage.local.get("token").then(({ token }) => { tokenEl.value = token || ""; });

document.getElementById("save").onclick = async () => {
  await chrome.storage.local.set({ token: tokenEl.value.trim() });
  const h = await chrome.runtime.sendMessage({ type: "health" });
  msg.textContent = h?.ok ? "Saved. Service reachable." : `Saved, but: ${h?.error || "service check failed"}`;
};
