import { apiBaseUrl } from "./config.mjs";
import { askQuestion } from "./api.mjs";
import {
  LOADING_MESSAGES,
  answerParts,
  createAssistantController,
  documentFileUrl,
} from "./logic.mjs";

const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-question");
const statusNode = document.querySelector("#status");
const resultNode = document.querySelector("#result");
const askedNode = document.querySelector("#asked");
const answerNode = document.querySelector("#answer");
const limitNode = document.querySelector("#limit-note");
const sourcesNode = document.querySelector("#sources");
const errorNode = document.querySelector("#error");
const retryButton = document.querySelector("#retry");

const controller = createAssistantController({
  ask: (question) => askQuestion(apiBaseUrl, question),
});

let messageTimer = 0;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void run(questionInput.value);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

retryButton.addEventListener("click", () => {
  void run(questionInput.value);
});

for (const suggestion of document.querySelectorAll("[data-suggestion]")) {
  suggestion.addEventListener("click", () => {
    questionInput.value = suggestion.textContent.trim();
    questionInput.focus();
  });
}

async function run(question) {
  if (controller.loading) {
    return;
  }
  if (!question || !question.trim()) {
    questionInput.setAttribute("aria-invalid", "true");
    questionInput.focus();
    return;
  }
  questionInput.removeAttribute("aria-invalid");
  showLoading(question.trim());
  const result = await controller.submit(question);
  stopLoading();
  if (result.status === "error") {
    showError();
    return;
  }
  if (result.status === "answered") {
    showAnswer(result.result);
  }
}

function showLoading(question) {
  errorNode.hidden = true;
  resultNode.hidden = false;
  askedNode.textContent = question;
  answerNode.replaceChildren();
  sourcesNode.replaceChildren();
  limitNode.hidden = true;
  submitButton.disabled = true;
  questionInput.setAttribute("aria-invalid", "false");
  form.setAttribute("aria-busy", "true");
  let index = 0;
  statusNode.hidden = false;
  statusNode.textContent = LOADING_MESSAGES[0];
  window.clearInterval(messageTimer);
  messageTimer = window.setInterval(() => {
    index = (index + 1) % LOADING_MESSAGES.length;
    statusNode.textContent = LOADING_MESSAGES[index];
  }, 12000);
}

function stopLoading() {
  window.clearInterval(messageTimer);
  statusNode.hidden = true;
  submitButton.disabled = false;
  form.removeAttribute("aria-busy");
}

function showError() {
  resultNode.hidden = true;
  errorNode.hidden = false;
}

function showAnswer(payload) {
  errorNode.hidden = true;
  resultNode.hidden = false;
  askedNode.textContent = payload.question || controller.lastQuestion;
  renderAnswer(payload.answer || "");
  limitNode.hidden = !payload.diagnostics?.insufficient_evidence;
  renderSources(payload.sources || []);
}

function renderAnswer(answer) {
  const fragment = document.createDocumentFragment();
  for (const part of answerParts(answer)) {
    if (part.type === "text") {
      fragment.append(document.createTextNode(part.text));
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation";
    button.textContent = `[${part.id}]`;
    button.addEventListener("click", () => focusSource(part.id));
    fragment.append(button);
  }
  answerNode.replaceChildren(fragment);
}

function renderSources(sources) {
  sourcesNode.replaceChildren();
  if (sources.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Nenhuma fonte foi associada a esta resposta.";
    sourcesNode.append(empty);
    return;
  }
  for (const source of sources) {
    sourcesNode.append(sourceCard(source));
  }
}

function sourceCard(source) {
  const card = document.createElement("article");
  card.className = "source-card";
  card.id = `source-${source.citation_id}`;
  card.tabIndex = -1;

  const citation = document.createElement("p");
  citation.className = "citation-id";
  citation.textContent = `[${source.citation_id}]`;

  const title = document.createElement("h3");
  title.textContent = source.title || "PFC sem título";

  const meta = document.createElement("p");
  meta.className = "meta";
  const authors = (source.authors || []).join(", ");
  const bits = [authors, source.year].filter((bit) => bit !== null && bit !== undefined && bit !== "");
  meta.textContent = bits.join(" · ");

  const page = document.createElement("p");
  page.textContent = source.page ? `Página ${source.page}` : "Página não indicada";

  card.append(citation, title, meta, page);
  if (source.pfc_id) {
    const link = document.createElement("a");
    link.href = documentFileUrl(apiBaseUrl, source.pfc_id, source.page);
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "Ver PFC";
    card.append(link);
  }
  return card;
}

function focusSource(citationId) {
  const card = document.getElementById(`source-${citationId}`);
  if (!card) {
    return;
  }
  for (const node of sourcesNode.querySelectorAll(".is-highlighted")) {
    node.classList.remove("is-highlighted");
  }
  card.classList.add("is-highlighted");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.focus();
}
