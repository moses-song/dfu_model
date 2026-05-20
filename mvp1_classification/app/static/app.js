const form = document.querySelector("#analyzeForm");
const fileInput = document.querySelector("#fileInput");
const fileLabel = document.querySelector("#fileLabel");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const submitButton = document.querySelector("#submitButton");
const apiStatus = document.querySelector("#apiStatus");
const nextAction = document.querySelector("#nextAction");
const originalImage = document.querySelector("#originalImage");
const overlayImage = document.querySelector("#overlayImage");
const maskImage = document.querySelector("#maskImage");
const stageList = document.querySelector("#stageList");
const disclaimer = document.querySelector("#disclaimer");

function setStatus(text, state) {
  apiStatus.textContent = text;
  apiStatus.className = `status ${state || ""}`.trim();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    setStatus(response.ok ? "API 연결됨" : "API 오류", response.ok ? "ok" : "error");
  } catch {
    setStatus("API 연결 실패", "error");
  }
}

function percent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function stageCard(title, result) {
  if (!result) {
    return `
      <article class="stage mutedStage">
        <header>
          <h3>${title}</h3>
          <span class="badge">skipped</span>
        </header>
        <div class="metric"><span>Result</span><strong>-</strong></div>
      </article>
    `;
  }

  const note = result.note ? `<p class="note">${result.note}</p>` : "";
  const weights = result.weights_found ? "found" : "missing";
  return `
    <article class="stage">
      <header>
        <h3>${title}</h3>
        <span class="badge">${result.backend}</span>
      </header>
      <div class="metric"><span>Result</span><strong>${result.class_label}</strong></div>
      <div class="metric"><span>Score</span><strong>${percent(result.score)}</strong></div>
      <div class="metric"><span>Weights</span><strong>${weights}</strong></div>
      ${note}
    </article>
  `;
}

function segmentationCard(result) {
  if (!result) return "";
  const note = result.note ? `<p class="note">${result.note}</p>` : "";
  const wound = result.wound_present ? "present" : "not detected";
  const weights = result.weights_found ? "found" : "missing";
  return `
    <article class="stage">
      <header>
        <h3>Segmentation</h3>
        <span class="badge">${result.backend}</span>
      </header>
      <div class="metric"><span>Wound</span><strong>${wound}</strong></div>
      <div class="metric"><span>Area</span><strong>${percent(result.area_ratio)}</strong></div>
      <div class="metric"><span>Weights</span><strong>${weights}</strong></div>
      ${note}
    </article>
  `;
}

function renderResult(data) {
  nextAction.textContent = data.next_action || "분석 결과를 확인하세요.";
  disclaimer.textContent = data.disclaimer || "";

  originalImage.src = data.segmentation?.original?.data_url || "";
  overlayImage.src = data.segmentation?.overlay?.data_url || "";
  maskImage.src = data.segmentation?.mask?.data_url || "";

  stageList.innerHTML = [
    stageCard("Foot Check", data.foot),
    segmentationCard(data.segmentation),
    stageCard("DFU Check", data.dfu),
    stageCard("Wagner Grade", data.wagner),
    stageCard("SINBAD Risk", data.sinbad),
  ].join("");
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
  const formData = new FormData(form);
  submitButton.disabled = true;
  submitButton.textContent = "분석 중";
  nextAction.textContent = "이미지를 분석하고 있습니다.";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "analysis failed");
    }
    renderResult(data);
  } catch (error) {
    nextAction.textContent = `분석 실패: ${error.message}`;
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m5 12 4 4L19 6" />
      </svg>
      분석 시작
    `;
  }
});

checkHealth();
