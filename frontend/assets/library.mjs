import { apiBaseUrl } from "./config.mjs";
import { getDocument, listDocuments } from "./api.mjs";
import { detailFields, distinctYears, documentFileUrl, searchDocuments } from "./logic.mjs";

document.querySelector(".filters").addEventListener("submit", (event) => {
  event.preventDefault();
});

const searchInput = document.querySelector("#catalogue-search");
const yearSelect = document.querySelector("#year-filter");
const listNode = document.querySelector("#catalogue");
const emptyNode = document.querySelector("#catalogue-empty");
const errorNode = document.querySelector("#catalogue-error");
const detailNode = document.querySelector("#detail");
const detailTitle = document.querySelector("#detail-title");
const detailAuthors = document.querySelector("#detail-authors");
const detailFacts = document.querySelector("#detail-facts");
const openLink = document.querySelector("#open-pfc");

let documents = [];
let selectedId = "";

searchInput.addEventListener("input", renderList);
yearSelect.addEventListener("change", renderList);

load();

async function load() {
  try {
    documents = await listDocuments(apiBaseUrl);
    fillYears();
    renderList();
  } catch {
    errorNode.hidden = false;
  }
}

function fillYears() {
  const current = yearSelect.value;
  for (const year of distinctYears(documents)) {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    yearSelect.append(option);
  }
  yearSelect.value = current;
}

function renderList() {
  const matches = searchDocuments(documents, searchInput.value, yearSelect.value);
  listNode.replaceChildren();
  emptyNode.hidden = matches.length !== 0;
  for (const documentRecord of matches) {
    listNode.append(card(documentRecord));
  }
}

function card(documentRecord) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "catalogue-card";
  if (documentRecord.id === selectedId) {
    button.classList.add("is-selected");
  }
  const title = document.createElement("span");
  title.className = "card-title";
  title.textContent = documentRecord.title || "PFC sem título";
  const meta = document.createElement("span");
  meta.className = "meta";
  const authors = (documentRecord.authors || []).join(", ");
  meta.textContent = [authors, documentRecord.year, documentRecord.course]
    .filter((part) => part !== null && part !== undefined && part !== "")
    .join(" · ");
  button.append(title, meta);
  button.addEventListener("click", () => {
    void showDetail(documentRecord.id);
  });
  item.append(button);
  return item;
}

async function showDetail(pfcId) {
  selectedId = pfcId;
  renderList();
  try {
    const documentRecord = await getDocument(apiBaseUrl, pfcId);
    detailNode.hidden = false;
    detailTitle.textContent = documentRecord.title || "PFC sem título";
    const authors = (documentRecord.authors || []).join(", ");
    detailAuthors.textContent = [authors, documentRecord.year]
      .filter((part) => part !== null && part !== undefined && part !== "")
      .join(" · ");
    detailFacts.replaceChildren();
    for (const field of detailFields(documentRecord)) {
      const label = document.createElement("dt");
      label.textContent = field.label;
      const value = document.createElement("dd");
      value.textContent = field.value;
      detailFacts.append(label, value);
    }
    openLink.href = documentFileUrl(apiBaseUrl, pfcId);
    detailNode.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch {
    errorNode.hidden = false;
  }
}
