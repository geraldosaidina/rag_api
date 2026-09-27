import assert from "node:assert/strict";
import test from "node:test";

import {
  LOADING_MESSAGES,
  answerParts,
  createAssistantController,
  detailFields,
  documentFileUrl,
  indexedDocuments,
  searchDocuments,
} from "./logic.mjs";

const catalogue = [
  {
    id: "belso",
    title: "Sistema de videovigilância IP",
    authors: ["Belso Bento Langa"],
    year: 2016,
    course: "Licenciatura em Engenharia Informática e de Telecomunicação",
    status: "indexed",
  },
  {
    id: "edson",
    title: "Sistema RFID no ensino",
    authors: ["Edson Samuel Langa"],
    year: 2018,
    course: "Engenharia Informática",
    status: "indexed",
  },
  {
    id: "jaime",
    title: null,
    authors: [],
    year: null,
    course: null,
    status: "failed",
  },
];

test("initial loading copy does not pretend to know pipeline stages", () => {
  const joined = LOADING_MESSAGES.join(" ");
  assert.match(joined, /Projectos Finais de Curso/);
  assert.doesNotMatch(joined, /Rerank|Generating|%|\d+%/i);
});

test("blank questions are rejected and do not call the API", async () => {
  let calls = 0;
  const controller = createAssistantController({
    ask: async () => {
      calls += 1;
      return {};
    },
  });
  const result = await controller.submit("   ");
  assert.equal(result.status, "blank");
  assert.equal(calls, 0);
});

test("submit calls the API once and blocks a duplicate while loading", async () => {
  let release;
  const pending = new Promise((resolve) => {
    release = resolve;
  });
  const seen = [];
  const controller = createAssistantController({
    ask: (question) => {
      seen.push(question);
      return pending;
    },
  });
  const first = controller.submit("  Como funciona o RFID?  ");
  assert.equal(controller.loading, true);
  const second = await controller.submit("outra pergunta");
  assert.equal(second.status, "blocked");
  assert.deepEqual(seen, ["Como funciona o RFID?"]);
  release({
    question: "Como funciona o RFID?",
    answer: "Usa RFID [S1].",
    sources: [{ citation_id: "S1", title: "RFID", authors: ["Edson"], year: 2018, page: 74 }],
    diagnostics: { insufficient_evidence: false },
  });
  const done = await first;
  assert.equal(done.status, "answered");
  assert.equal(done.result.sources[0].title, "RFID");
  assert.equal(controller.loading, false);
});

test("API failure keeps the question and is not an answer", async () => {
  const controller = createAssistantController({
    ask: async () => {
      throw new Error("down");
    },
  });
  const result = await controller.submit("Pergunta que falha");
  assert.equal(result.status, "error");
  assert.equal(result.question, "Pergunta que falha");
  assert.equal(controller.lastQuestion, "Pergunta que falha");
});

test("insufficient evidence stays an answered result", async () => {
  const controller = createAssistantController({
    ask: async () => ({
      answer: "Não há evidência suficiente.",
      diagnostics: { insufficient_evidence: true },
      sources: [],
    }),
  });
  const result = await controller.submit("Pergunta sem suporte");
  assert.equal(result.status, "answered");
  assert.equal(result.result.diagnostics.insufficient_evidence, true);
});

test("answer parts keep citation markers separate from text", () => {
  const parts = answerParts("Texto [S1] e depois [s2].");
  assert.deepEqual(
    parts.map((part) => part.type === "citation" ? part.id : part.text),
    ["Texto ", "S1", " e depois ", "S2", "."],
  );
});

test("file URL uses pfc id and page fragment", () => {
  assert.equal(
    documentFileUrl("http://127.0.0.1:8000", "id com espaço", 74),
    "http://127.0.0.1:8000/api/v1/documents/id%20com%20espa%C3%A7o/file#page=74",
  );
  assert.equal(documentFileUrl("", "abc", null), "/api/v1/documents/abc/file");
});

test("library hides failed PFCs and searches title, author, course and year", () => {
  assert.deepEqual(indexedDocuments(catalogue).map((item) => item.id), ["belso", "edson"]);
  assert.deepEqual(searchDocuments(catalogue, "videovigilancia", "").map((item) => item.id), ["belso"]);
  assert.deepEqual(searchDocuments(catalogue, "belso bento", "").map((item) => item.id), ["belso"]);
  assert.deepEqual(searchDocuments(catalogue, "telecomunicacao", "").map((item) => item.id), ["belso"]);
  assert.deepEqual(searchDocuments(catalogue, "2018", "").map((item) => item.id), ["edson"]);
  assert.deepEqual(searchDocuments(catalogue, "jaime", "").map((item) => item.id), []);
  assert.deepEqual(searchDocuments(catalogue, "", "2016").map((item) => item.id), ["belso"]);
});

test("detail omits empty fields and internal catalogue fields", () => {
  const rows = detailFields({
    title: "Título",
    course: "Curso A",
    institution: "",
    department: null,
    supervisor: "Eng. Ana",
    abstract: "Resumo curto.",
    keywords: [],
    language: "pt",
    error: "No usable text",
    status: "failed",
    original_filename: "segredo.pdf",
    content_hash: "abc",
    storage_path: "data/corpus/x.pdf",
  });
  assert.deepEqual(rows.map((row) => row.label), ["Curso", "Supervisor", "Língua", "Resumo"]);
  assert.equal(rows.some((row) => String(row.value).includes("segredo")), false);
  assert.equal(rows.some((row) => String(row.value).includes("No usable text")), false);
});
