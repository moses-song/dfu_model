const form = document.querySelector("#analyzeForm");
const fileInput = document.querySelector("#fileInput");
const fileLabel = document.querySelector("#fileLabel");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const submitButton = document.querySelector("#submitButton");
const apiStatus = document.querySelector("#apiStatus");
const nextAction = document.querySelector("#nextAction");
const pipelineSummary = document.querySelector("#pipelineSummary");
const comparisonResults = document.querySelector("#comparisonResults");
const disclaimer = document.querySelector("#disclaimer");
const modelButtons = document.querySelector("#modelButtons");

let modelCatalog = [];

function setStatus(text, state) {
  apiStatus.textContent = text;
  apiStatus.className = `status ${state || ""}`.trim();
}

function percent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function ensureFile() {
  const file = fileInput.files?.[0];
  if (!file) {
    throw new Error("Select an image first.");
  }
  return file;
}

function renderModelButtons() {
  modelButtons.innerHTML = modelCatalog
    .map(
      (model) => `
        <button class="modelButton" type="button" data-model-id="${model.id}">
          <strong>${model.title}</strong>
          <span>${model.summary}</span>
          <small>${model.weights_found ? "weights found" : "weights missing"}</small>
        </button>
      `,
    )
    .join("");

  [...modelButtons.querySelectorAll(".modelButton")].forEach((button) => {
    button.addEventListener("click", async () => {
      const modelId = button.dataset.modelId;
      await runModel(modelId, button);
    });
  });
}

function renderArtifacts(artifacts) {
  if (!artifacts?.length) return "";
  return `
    <div class="artifactGrid">
      ${artifacts
        .map(
          (artifact) => `
            <figure>
              <img src="${artifact.data_url}" alt="${artifact.label}" />
              <figcaption>${artifact.label}</figcaption>
            </figure>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderMetrics(metrics) {
  if (!metrics?.length) return "";
  return `
    <div class="metricList">
      ${metrics
        .map(
          (metric) => `
            <div class="metric">
              <span>${metric.label}</span>
              <strong>${metric.value}</strong>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderEvalMetrics(metrics) {
  if (!metrics?.length) return "";
  return `
    <section class="componentBox">
      <div class="componentHeader">
        <h3>Evaluation Metrics</h3>
        <span>inference mode</span>
      </div>
      <div class="componentGrid">
        ${metrics
          .map(
            (item) => `
              <article class="componentItem">
                <strong>${item.label}</strong>
                <span>${item.available ? item.display_value || item.value : "N/A"}</span>
                <small>${item.note || ""}</small>
              </article>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderComponentScores(scores) {
  if (!scores?.length) return "";
  const numeric = scores.filter((item) => typeof item.score === "number").map((item) => item.score);
  const total = numeric.length ? numeric.reduce((sum, value) => sum + value, 0) : null;
  return `
    <section class="componentBox">
      <div class="componentHeader">
        <h3>SINBAD Components</h3>
        <span>${total === null ? "pending" : `sum ${total}`}</span>
      </div>
      <div class="componentGrid">
        ${scores
          .map(
            (item) => `
              <article class="componentItem">
                <strong>${item.label}</strong>
                <span>${item.score === null ? "-" : item.score}</span>
                <small>${item.note || ""}</small>
              </article>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderDetections(detections) {
  if (!detections?.length) return "";
  return `
    <section class="componentBox">
      <div class="componentHeader">
        <h3>Detection Output</h3>
        <span>${detections.length} item(s)</span>
      </div>
      <div class="detectionList">
        ${detections
          .map(
            (item) => `
              <article class="detectionItem">
                ${Object.entries(item)
                  .map(
                    ([key, value]) => `
                      <div class="metric">
                        <span>${key}</span>
                        <strong>${value}</strong>
                      </div>
                    `,
                  )
                  .join("")}
              </article>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderRawOutputs(rawOutputs) {
  const entries = Object.entries(rawOutputs || {});
  if (!entries.length) return "";
  return `
    <section class="componentBox">
      <div class="componentHeader">
        <h3>Raw Output</h3>
        <span>debug</span>
      </div>
      <div class="detectionItem">
        ${entries
          .map(
            ([key, value]) => `
              <div class="metric">
                <span>${key}</span>
                <strong>${value}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function upsertResultCard(result) {
  const existing = document.querySelector(`[data-result-id="${result.model.id}"]`);
  const note = result.note ? `<p class="note">${result.note}</p>` : "";
  const scoreLine = typeof result.score === "number" ? `<p class="scoreLine">Confidence ${percent(result.score)}</p>` : "";
  const html = `
    <article class="resultCard" data-result-id="${result.model.id}">
      <header class="resultHeader">
        <div>
          <p class="eyebrow">${result.model.kind}</p>
          <h2>${result.model.title}</h2>
          <p>${result.model.summary}</p>
        </div>
        <span class="badge">${result.model.backend}</span>
      </header>
      <div class="resultBody">
        <div class="summaryBlock">
          <div class="metricHero">
            <span>Primary output</span>
            <strong>${result.primary_label || result.status}</strong>
          </div>
          ${scoreLine}
          <p class="scoreLine">Timing ${result.timing_ms} ms, FPS ${result.fps}</p>
          <p class="scoreLine">Shared feature ${result.feature_backend}, cache ${result.feature_cache_hit ? "hit" : "miss"}</p>
          ${note}
        </div>
        ${renderMetrics(result.metrics)}
        ${renderEvalMetrics(result.eval_metrics)}
        ${renderDetections(result.detections)}
        ${renderComponentScores(result.component_scores)}
        ${renderRawOutputs(result.raw_outputs)}
        ${renderArtifacts(result.artifacts)}
      </div>
    </article>
  `;

  if (existing) {
    existing.outerHTML = html;
  } else {
    comparisonResults.insertAdjacentHTML("afterbegin", html);
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    setStatus(response.ok ? "API online" : "API error", response.ok ? "ok" : "error");
  } catch {
    setStatus("API offline", "error");
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Failed to load models");
  }
  modelCatalog = data.models || [];
  renderModelButtons();
}

async function runModel(modelId, button) {
  const file = ensureFile();
  const formData = new FormData();
  formData.append("file", file);
  button.disabled = true;
  nextAction.textContent = `Running ${modelId}...`;

  try {
    const response = await fetch(`/api/models/${modelId}/run`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Model run failed");
    }
    upsertResultCard(data);
    nextAction.textContent = `${data.model.title} finished.`;
  } catch (error) {
    nextAction.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  fileLabel.textContent = file.name;
  previewImage.src = URL.createObjectURL(file);
  emptyPreview.hidden = true;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = ensureFile();
  const formData = new FormData(form);
  formData.set("file", file);
  submitButton.disabled = true;
  submitButton.textContent = "Running pipeline...";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Pipeline failed");
    }
    pipelineSummary.textContent = data.next_action || "Completed.";
    disclaimer.textContent = data.disclaimer || "";
    nextAction.textContent = "Legacy pipeline completed.";
  } catch (error) {
    pipelineSummary.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Run full pipeline";
  }
});

Promise.all([checkHealth(), loadModels()]).catch((error) => {
  nextAction.textContent = error.message;
});
