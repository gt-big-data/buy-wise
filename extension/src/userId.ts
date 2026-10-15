// Each install gets its own random ID, created the first time it's needed and
// kept in chrome.storage.local. It identifies the install, not a person, and
// goes away if the extension is uninstalled.

const KEY = "buywiseUserId";

export async function getUserId(): Promise<string> {
  const stored = await chrome.storage.local.get(KEY);
  if (typeof stored[KEY] === "string") return stored[KEY];
  const id = crypto.randomUUID();
  await chrome.storage.local.set({ [KEY]: id });
  return id;
}
