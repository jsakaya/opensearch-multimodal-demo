const state = {
  mode: "hybrid",
  hits: [],
  activeId: null,
  modality: "",
  source: "",
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function compact(value) {
  const text = String(value ?? "").trim();
  return text || "unknown";
}

function firstThumb(hit) {
  const assets = hit.assets || [];
  const asset = assets.find((item) => item.thumbnail_url || item.url);
  return asset ? asset.thumbnail_url || asset.url : "";
}

function score(value) {
  const n = Number(value || 0);
  return n >= 10 ? n.toFixed(2) : n.toFixed(4);
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2200);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

async function loadStatus() {
  try {
    const payload = await api("/api/status");
    $("indexName").textContent = payload.index || "openlens_multimodal";
    const os = payload.opensearch || {};
    $("statusLine").textContent = os.available ? `${os.doc_count || 0} docs on OpenSearch` : "local fallback";
  } catch {
    $("statusLine").textContent = "status unavailable";
  }
}

async function loadExamples() {
  try {
    const payload = await api("/api/examples");
    if (payload.queries && payload.queries.length && !$("queryInput").value) {
      $("queryInput").value = payload.queries[0];
    }
  } catch {
    return null;
  }
  return null;
}

async function runSearch() {
  const q = $("queryInput").value.trim();
  if (!q) return;
  const params = new URLSearchParams({
    q,
    mode: state.mode,
    top_k: "12",
    candidate_k: "80",
  });
  if (state.modality) params.set("modality", state.modality);
  if (state.source) params.set("source", state.source);
  try {
    const payload = await api(`/api/search?${params}`);
    renderSearch(payload);
  } catch (error) {
    showToast("Search failed");
    console.error(error);
  }
}

function renderSearch(payload) {
  state.hits = payload.hits || [];
  $("retrieverName").textContent = compact(payload.retriever);
  $("latencyValue").textContent = `${Number(payload.latency_ms || 0).toFixed(1)} ms`;
  $("hitCount").textContent = String(payload.total || state.hits.length);
  renderFacets("modalityFacets", payload.facets?.modality || {}, "modality");
  renderFacets("sourceFacets", payload.facets?.source || {}, "source");
  renderResults();
}

function renderFacets(id, facets, key) {
  const root = $(id);
  const active = state[key];
  const rows = Object.entries(facets).sort((a, b) => b[1] - a[1]);
  root.innerHTML = "";
  const all = document.createElement("button");
  all.type = "button";
  all.className = `facet-button ${active ? "" : "is-active"}`;
  all.innerHTML = `<span>All</span><small>${rows.reduce((sum, row) => sum + row[1], 0)}</small>`;
  all.addEventListener("click", () => {
    state[key] = "";
    runSearch();
  });
  root.appendChild(all);
  rows.forEach(([name, count]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `facet-button ${active === name ? "is-active" : ""}`;
    button.innerHTML = `<span>${esc(name)}</span><small>${count}</small>`;
    button.addEventListener("click", () => {
      state[key] = active === name ? "" : name;
      runSearch();
    });
    root.appendChild(button);
  });
}

function renderResults() {
  const root = $("resultList");
  root.innerHTML = "";
  if (!state.hits.length) {
    root.innerHTML = `<article class="result-card"><div class="modality-rail"></div><div class="result-body"><h2 class="result-title">No hits</h2><p class="result-summary">Build or ingest a corpus, then search again.</p></div></article>`;
    $("detailPane").innerHTML = `<div class="empty-detail">No document selected.</div>`;
    return;
  }
  state.hits.forEach((hit) => {
    const card = document.createElement("article");
    card.className = `result-card ${state.activeId === hit.doc_id ? "is-active" : ""}`;
    card.dataset.modality = hit.modality || "document";
    const thumb = firstThumb(hit);
    card.innerHTML = `
      <div class="modality-rail"></div>
      <div class="result-body">
        <h2 class="result-title">${esc(hit.title)}</h2>
        <div class="result-meta">${esc(hit.modality)} &middot; ${esc(hit.source)}</div>
        <div class="score-line">${esc(hit.method)} ${score(hit.score)}</div>
        <p class="result-summary">${esc(hit.excerpt || hit.summary)}</p>
      </div>
      ${
        thumb
          ? `<img class="thumb" src="${esc(thumb)}" alt="" loading="lazy" referrerpolicy="no-referrer" />`
          : `<div class="rank-chip">#${hit.rank}</div>`
      }
    `;
    card.addEventListener("click", () => {
      state.activeId = hit.doc_id;
      renderResults();
      renderDetail(hit);
    });
    root.appendChild(card);
  });
  renderDetail(state.hits.find((hit) => hit.doc_id === state.activeId) || state.hits[0]);
}

function renderDetail(hit) {
  const pane = $("detailPane");
  const thumb = firstThumb(hit);
  const table = hit.table || {};
  const facets = hit.facets || {};
  const rows = Object.entries({ ...facets, ...table })
    .filter(([, value]) => value !== null && value !== undefined && String(value) !== "")
    .slice(0, 28)
    .map(([key, value]) => `<div class="kv-row"><div class="kv-key">${esc(key)}</div><div class="kv-value">${esc(value)}</div></div>`)
    .join("");
  const tags = (hit.tags || []).slice(0, 10).map((tag) => `<span class="tag">${esc(tag)}</span>`).join("");
  pane.innerHTML = `
    ${thumb ? `<img class="detail-media" src="${esc(thumb)}" alt="" referrerpolicy="no-referrer" />` : ""}
    <h2>${esc(hit.title)}</h2>
    <div class="detail-meta">${esc(hit.modality)} &middot; ${esc(hit.source)} &middot; ${esc(hit.license)}</div>
    <p class="detail-summary">${esc(hit.summary || hit.excerpt || "")}</p>
    <div class="tag-row">${tags}</div>
    ${
      hit.source_url
        ? `<a href="${esc(hit.source_url)}" target="_blank" rel="noreferrer">Open source record</a>`
        : ""
    }
    ${rows ? `<div class="kv-list">${rows}</div>` : ""}
  `;
}

async function ingest(event) {
  event.preventDefault();
  const title = $("ingestTitle").value.trim();
  const body = $("ingestBody").value.trim();
  if (!title || !body) {
    showToast("Title and text required");
    return;
  }
  try {
    const payload = await api("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        body,
        modality: $("ingestModality").value,
        source: "Live ingest",
        tags: ["inline"],
      }),
    });
    $("queryInput").value = title;
    $("ingestTitle").value = "";
    $("ingestBody").value = "";
    showToast(`Indexed to ${payload.indexed_to}`);
    await loadStatus();
    await runSearch();
  } catch (error) {
    showToast("Ingest failed");
    console.error(error);
  }
}

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    state.mode = button.dataset.mode || "hybrid";
    runSearch();
  });
});

$("searchForm").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});

$("ingestForm").addEventListener("submit", ingest);

loadStatus();
loadExamples().then(runSearch);
