/** Pure helpers for the student UI. No network and no RAG logic. */

export const LOADING_MESSAGES = [
  "A pesquisar nos Projectos Finais de Curso...",
  "A analisar as fontes relevantes...",
  "Esta resposta pode demorar algum tempo.",
];

const DETAIL_FIELDS = [
  ["course", "Curso"],
  ["institution", "Instituição"],
  ["department", "Departamento"],
  ["supervisor", "Supervisor"],
  ["language", "Língua"],
  ["keywords", "Palavras-chave"],
  ["abstract", "Resumo"],
];

export function isBlank(value) {
  return !value || !String(value).trim();
}

export function normalizeSearch(value) {
  return String(value)
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase();
}

export function indexedDocuments(documents) {
  return (documents || []).filter((document) => document.status === "indexed");
}

export function searchDocuments(documents, query, year) {
  const needle = normalizeSearch(query || "").trim();
  const yearFilter = year ? String(year) : "";
  return indexedDocuments(documents).filter((document) => {
    if (yearFilter && String(document.year ?? "") !== yearFilter) {
      return false;
    }
    if (!needle) {
      return true;
    }
    const haystack = [
      document.title,
      ...(document.authors || []),
      document.course,
      document.year,
    ]
      .filter((part) => part !== null && part !== undefined && part !== "")
      .join(" ");
    return normalizeSearch(haystack).includes(needle);
  });
}

export function distinctYears(documents) {
  const years = new Set();
  for (const document of indexedDocuments(documents)) {
    if (document.year !== null && document.year !== undefined && document.year !== "") {
      years.add(String(document.year));
    }
  }
  return [...years].sort((left, right) => Number(right) - Number(left));
}

export function detailFields(document) {
  const rows = [];
  for (const [key, label] of DETAIL_FIELDS) {
    const value = document[key];
    if (Array.isArray(value)) {
      if (value.length === 0) {
        continue;
      }
      rows.push({ label, value: value.join(", ") });
      continue;
    }
    if (value === null || value === undefined || String(value).trim() === "") {
      continue;
    }
    rows.push({ label, value: String(value) });
  }
  return rows;
}

export function answerParts(answer) {
  const text = String(answer || "");
  const parts = [];
  const pattern = /\[(S\d+)\]/gi;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) {
      parts.push({ type: "text", text: text.slice(cursor, match.index) });
    }
    parts.push({ type: "citation", id: `S${match[1].slice(1)}` });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    parts.push({ type: "text", text: text.slice(cursor) });
  }
  return parts;
}

export function documentFileUrl(baseUrl, pfcId, page) {
  const root = String(baseUrl || "").replace(/\/$/, "");
  const url = `${root}/api/v1/documents/${encodeURIComponent(pfcId)}/file`;
  const pageNumber = Number(page);
  if (Number.isInteger(pageNumber) && pageNumber > 0) {
    return `${url}#page=${pageNumber}`;
  }
  return url;
}

export function createAssistantController({ ask }) {
  let loading = false;
  let lastQuestion = "";

  return {
    get loading() {
      return loading;
    },
    get lastQuestion() {
      return lastQuestion;
    },
    async submit(question) {
      if (loading) {
        return { status: "blocked" };
      }
      if (isBlank(question)) {
        return { status: "blank" };
      }
      const submitted = String(question).trim();
      lastQuestion = submitted;
      loading = true;
      try {
        const result = await ask(submitted);
        return { status: "answered", question: submitted, result };
      } catch {
        return { status: "error", question: submitted };
      } finally {
        loading = false;
      }
    },
  };
}
