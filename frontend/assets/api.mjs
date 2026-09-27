/** Small typed-by-convention client for the PFC Assistant API. */

async function readError(response) {
  try {
    const body = await response.json();
    if (body && typeof body.error === "string") {
      return body.error;
    }
  } catch {
    /* Response body is not the public error shape. */
  }
  return "request_failed";
}

async function requestJson(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("unavailable");
  }
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json();
}

export function askQuestion(baseUrl, question) {
  const root = String(baseUrl || "").replace(/\/$/, "");
  return requestJson(`${root}/api/v1/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export function listDocuments(baseUrl) {
  const root = String(baseUrl || "").replace(/\/$/, "");
  return requestJson(`${root}/api/v1/documents`);
}

export function getDocument(baseUrl, pfcId) {
  const root = String(baseUrl || "").replace(/\/$/, "");
  return requestJson(`${root}/api/v1/documents/${encodeURIComponent(pfcId)}`);
}

export { documentFileUrl } from "./logic.mjs";
