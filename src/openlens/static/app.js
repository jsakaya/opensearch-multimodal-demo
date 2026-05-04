const state = {
  mode: "hybrid",
  hits: [],
  activeId: null,
  modality: "",
  source: "",
  query: "",
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
  const asset = assets.find(
    (item) => item.thumbnail_url || String(item.kind).includes("image") || String(item.mime_type).includes("image")
  );
  if (!asset) return "";
  return asset.thumbnail_url || asset.url || "";
}

function playableAsset(hit, kind) {
  const assets = hit.assets || [];
  const mediaKind = String(kind || hit.modality || "");
  if (mediaKind.startsWith("audio") || hit.modality === "audio") {
    return assets.find((item) => String(item.kind).includes("audio") || String(item.mime_type).includes("audio"));
  }
  if (mediaKind.startsWith("video") || hit.modality === "video") {
    return assets.find((item) => String(item.kind).includes("video") || String(item.mime_type).includes("video"));
  }
  return null;
}

function mediaSrc(value) {
  const text = String(value || "").trim();
  return text ? `/api/media?url=${encodeURIComponent(text)}` : "";
}

function score(value) {
  const n = Number(value || 0);
  return n >= 10 ? n.toFixed(2) : n.toFixed(4);
}

function regexEsc(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightTerms(text, terms) {
  let safe = esc(text);
  const unique = [...new Set((terms || []).filter((term) => String(term).length > 2))].sort((a, b) => b.length - a.length);
  unique.forEach((term) => {
    safe = safe.replace(new RegExp(`(${regexEsc(esc(term))})`, "gi"), "<mark>$1</mark>");
  });
  return safe;
}

function evidenceFor(hit) {
  const evidence = hit.evidence || [];
  if (evidence.length) return evidence;
  return (hit.patches || []).slice(0, 8).map((patch) => ({
    kind: patch.kind || "patch",
    label: patch.kind || "patch",
    loc: patch.start_s !== null && patch.start_s !== undefined
      ? `${Math.round(Number(patch.start_s))}-${Math.round(Number(patch.end_s || 0))}s`
      : patch.page ? `p${patch.page}` : `#${Number(patch.ordinal || 0) + 1}`,
    text: patch.text || "",
    matched_terms: [],
    page: patch.page,
    start_s: patch.start_s,
    end_s: patch.end_s,
    asset_url: patch.asset_url || patch.source_file || "",
  }));
}

function segmentFor(hit, item) {
  const kind = String(item.kind || hit.modality || "");
  const isAudio = kind.startsWith("audio") || hit.modality === "audio";
  const isVideo = kind.startsWith("video") || hit.modality === "video";
  const start = Number(item.start_s);
  const end = Number(item.end_s);
  if ((!isAudio && !isVideo) || !Number.isFinite(start)) return null;
  const asset = playableAsset(hit, kind);
  const source = item.asset_url || asset?.url || "";
  if (!source) return null;
  return {
    type: isVideo ? "video" : "audio",
    src: mediaSrc(source),
    start,
    end: Number.isFinite(end) && end > start ? end : null,
    label: item.loc || `${Math.round(start)}s`,
  };
}

function formatSegment(segment) {
  if (!segment) return "";
  return segment.end === null
    ? `${Math.round(segment.start)}s`
    : `${Math.round(segment.start)}-${Math.round(segment.end)}s`;
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
    $("statusLine").textContent = os.available
      ? `${os.doc_count || 0} docs · ${embeddingLabel(payload)} · OpenSearch`
      : "OpenSearch unavailable";
  } catch {
    $("statusLine").textContent = "status unavailable";
  }
}

function embeddingLabel(payload) {
  const backend = payload.embedding_backend || "feature-hash";
  const dim = payload.vector_dim ? `${payload.vector_dim}d` : "";
  if (backend === "qwen") {
    const runtime = payload.qwen_runtime || {};
    const device = runtime.cuda_available ? runtime.device_name || "CUDA" : runtime.device || "no GPU";
    return `${payload.qwen_model || "qwen"} ${dim} · ${device}`;
  }
  if (backend === "colpali") {
    const runtime = payload.colpali_runtime || {};
    const device = runtime.cuda_available ? runtime.device_name || "CUDA" : runtime.device || "no GPU";
    return `${payload.colpali_model || "ColPali"} ${dim} · ${device}`;
  }
  if (
    backend === "mlx" ||
    backend === "mlx-text" ||
    backend === "mlx-qwen-vl" ||
    backend === "mlx-vl" ||
    backend === "mlx-colqwen" ||
    backend === "mlx-colpali"
  ) {
    const runtime = payload.mlx_runtime || {};
    return `${payload.embedding_model || payload.mlx_text_model || "MLX"} ${dim} · ${runtime.default_device || "Apple GPU"}`;
  }
  return `${backend} ${dim}`.trim();
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
  state.query = payload.query || "";
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
    const evidence = evidenceFor(hit)[0];
    const patchLabel = hit.patch_vector_count && hit.patch_vector_count > hit.patch_count
      ? `${hit.patch_count} patches · ${hit.patch_vector_count} vectors`
      : hit.patch_count ? `${hit.patch_count} patches` : "single vector";
    card.innerHTML = `
      <div class="modality-rail"></div>
      <div class="result-body">
        <h2 class="result-title">${esc(hit.title)}</h2>
        <div class="result-meta">${esc(hit.modality)} &middot; ${esc(hit.source)} &middot; ${esc(patchLabel)}</div>
        <div class="score-line">${esc(hit.method)} ${score(hit.score)}</div>
        <p class="result-summary">${highlightTerms((evidence && evidence.text) || hit.excerpt || hit.summary, evidence?.matched_terms || [])}</p>
      </div>
      ${
        thumb
          ? `<img class="thumb" src="${esc(mediaSrc(thumb))}" alt="" loading="lazy" referrerpolicy="no-referrer" />`
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
  const evidenceItems = evidenceFor(hit).slice(0, 10);
  const firstSegment = evidenceItems.map((item) => segmentFor(hit, item)).find(Boolean);
  const evidence = evidenceItems
    .map((item, index) => {
      const terms = item.matched_terms || [];
      const termBadges = terms.map((term) => `<span>${esc(term)}</span>`).join("");
      const segment = segmentFor(hit, item);
      return `
        <div class="evidence-row" data-kind="${esc(item.kind || "patch")}">
          <div class="evidence-loc">${esc(item.loc || "")}</div>
          <div class="evidence-copy">
            <div class="evidence-head">
              <strong>${esc(item.kind || "patch")}</strong>
              ${
                segment
                  ? `<button class="segment-button" type="button" data-segment-index="${index}">Play ${esc(formatSegment(segment))}</button>`
                  : ""
              }
            </div>
            <p>${highlightTerms(item.text || "", terms)}</p>
            ${termBadges ? `<div class="evidence-terms">${termBadges}</div>` : ""}
          </div>
        </div>
      `;
    })
    .join("");
  pane.innerHTML = `
    ${
      thumb || firstSegment
        ? `<div class="detail-media-frame ${thumb ? "" : "is-empty"}" id="detailPreview">${
            thumb
              ? `<img class="detail-media" src="${esc(mediaSrc(thumb))}" alt="" referrerpolicy="no-referrer" />`
              : `<div class="detail-media-placeholder">${esc(firstSegment.type)} segment preview</div>`
          }</div>`
        : ""
    }
    <h2>${esc(hit.title)}</h2>
    <div class="detail-meta">${esc(hit.modality)} &middot; ${esc(hit.source)} &middot; ${esc(hit.license)}</div>
    <div class="embedding-strip">
      <span>${esc(hit.patch_count || 0)} patches</span>
      <span>${esc(hit.patch_vector_count || 0)} vectors</span>
      <span>${esc(hit.embedding_backend || "feature-hash")}</span>
      <span>${esc(hit.embedding_model || "feature-hash")}</span>
    </div>
    <p class="detail-summary">${esc(hit.summary || hit.excerpt || "")}</p>
    <div class="tag-row">${tags}</div>
    ${
      hit.source_url
        ? `<a href="${esc(hit.source_url)}" target="_blank" rel="noreferrer">Open source record</a>`
        : ""
    }
    ${evidence ? `<div class="evidence-title">Evidence trail</div><div class="evidence-list">${evidence}</div>` : ""}
    ${rows ? `<div class="kv-list">${rows}</div>` : ""}
  `;
  pane.querySelectorAll(".segment-button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const index = Number(button.dataset.segmentIndex);
      playSegment(hit, evidenceItems[index]);
    });
  });
}

function playSegment(hit, item) {
  const segment = segmentFor(hit, item || {});
  const preview = $("detailPreview");
  if (!segment || !preview) {
    showToast("No playable segment");
    return;
  }
  const tag = segment.type === "video" ? "video" : "audio";
  preview.classList.remove("is-empty");
  preview.classList.add("is-playing");
  preview.innerHTML = `
    <div class="segment-player-label">${esc(segment.type)} segment ${esc(formatSegment(segment))}</div>
    <${tag} class="detail-segment-media" controls playsinline preload="metadata" src="${esc(segment.src)}"></${tag}>
  `;
  const media = preview.querySelector(".detail-segment-media");
  const stopAtEnd = () => {
    if (segment.end !== null && media.currentTime >= segment.end) {
      media.pause();
      media.removeEventListener("timeupdate", stopAtEnd);
    }
  };
  const seekAndPlay = () => {
    media.currentTime = segment.start;
    media.play().catch(() => showToast("Media needs a click to play"));
  };
  media.addEventListener("loadedmetadata", seekAndPlay, { once: true });
  media.addEventListener("timeupdate", stopAtEnd);
  media.addEventListener("error", () => showToast("Media could not be loaded"), { once: true });
  media.load();
}

async function ingest(event) {
  event.preventDefault();
  const title = $("ingestTitle").value.trim();
  const body = $("ingestBody").value.trim();
  const assetUrl = $("ingestAssetUrl").value.trim();
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
        asset_url: assetUrl,
        tags: ["inline"],
      }),
    });
    $("queryInput").value = title;
    $("ingestTitle").value = "";
    $("ingestAssetUrl").value = "";
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
