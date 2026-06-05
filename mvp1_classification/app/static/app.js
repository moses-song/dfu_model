const fileInput = document.querySelector("#fileInput");
const previewImage = document.querySelector("#previewImage");
const previewViewport = document.querySelector("#previewViewport");
const previewDropzone = document.querySelector("#previewDropzone");
const emptyPreview = document.querySelector("#emptyPreview");
const imageState = document.querySelector("#imageState");
const apiStatus = document.querySelector("#apiStatus");
const cameraStatus = document.querySelector("#cameraStatus");
const screenMessage = document.querySelector("#screenMessage");
const cameraVideo = document.querySelector("#cameraVideo");
const liveOverlay = document.querySelector("#liveOverlay");
const startCameraButton = document.querySelector("#startCameraButton");
const captureButton = document.querySelector("#captureButton");
const taskHome = document.querySelector("#taskHome");
const taskDetail = document.querySelector("#taskDetail");
const backButton = document.querySelector("#backButton");
const detailEyebrow = document.querySelector("#detailEyebrow");
const detailTitle = document.querySelector("#detailTitle");
const detailDescription = document.querySelector("#detailDescription");
const detailAlert = document.querySelector("#detailAlert");
const taskModelList = document.querySelector("#taskModelList");
const taskGuide = document.querySelector("#taskGuide");
const nextAction = document.querySelector("#nextAction");
const comparisonResults = document.querySelector("#comparisonResults");

const MODEL_COPY = {
  dinov3_linear_foot: {
    title: "DINOv3 기반 foot / non-foot 분류",
    summary: "Backbone feature 위에 연결된 foot 분류 헤드 상태를 확인합니다.",
  },
  dinov3_backbone_pca: {
    title: "PCA / Cosine 기반 feature 반응 해석",
    summary: "DINOv3가 이미지에서 어디를 구분 단서로 보는지 PCA와 cosine map으로 확인합니다.",
  },
  dinov3_fastinst_d3_segmentation: {
    title: "DINOv3 기반 상처 세그멘테이션",
    summary: "상처 마스크, overlay, 감지 여부를 확인합니다.",
  },
  dinov3_linear_dfu: {
    title: "DINOv3 기반 DFU / 기타 상처 분류",
    summary: "상처가 DFU인지 다른 상처인지 구분합니다.",
  },
  dinov3_linear_wagner: {
    title: "DINOv3 기반 Wagner 분류",
    summary: "상처 severity를 Wagner grade 형태로 분류합니다.",
  },
  dinov3_linear_sinbad: {
    title: "DINOv3 기반 SINBAD 분류",
    summary: "상처 상태를 SINBAD score 관점으로 확인합니다.",
  },
};

const TASKS = [
  {
    id: "backbone_explain",
    title: "기본",
    eyebrow: "기본",
    description: "DINOv3 backbone feature를 PCA와 cosine map으로 시각화해, 모델이 이미지의 어느 부분을 구분 단서로 보는지 먼저 확인합니다.",
    homeSummary: "분류 전에 backbone feature의 반응 위치와 변화 강도를 해석합니다.",
    status: "active",
    models: ["dinov3_backbone_pca"],
    guide:
      "PCA map은 patch token들의 큰 변화 축을 요약해 주고, cosine map은 기준 patch와 비슷한 영역을 보여줍니다. 즉 '모델이 어디를 다르게 보고 있는지'를 먼저 점검하는 해석용 task입니다.",
    alertTitle: "왜 먼저 보나",
    alertBody:
      "이 시각화는 최종 진단 결과를 내기 위한 task가 아니라, backbone feature가 발 전체 형상, 상처 경계, 피부 결 변화 같은 구분 단서를 실제로 잡고 있는지 확인하기 위한 기본 점검 단계입니다.",
  },
  {
    id: "foot_check",
    title: "발인지 아닌지 분류",
    eyebrow: "Task 01",
    description: "입력 이미지가 발인지 아닌지를 가장 먼저 확인합니다.",
    homeSummary: "Foot 분류 헤드의 현재 연결 상태를 먼저 점검합니다.",
    status: "active",
    models: ["dinov3_linear_foot"],
    guide:
      "현재 저장소에는 학습된 foot classifier weight가 연결되어 있지 않을 수 있습니다. 따라서 foot linear head 결과는 구조 검증용일 가능성이 높고, DINOv3 feature 시각화와 함께 해석해야 합니다.",
    alertTitle: "중요",
    alertBody:
      "학습하지 않은 linear head는 의미 있는 분류기를 만들지 못합니다. 이 화면은 '지금 바로 foot / non-foot가 구분되는지'를 검증하는 실험용 UI이며, 실제로는 최소한 선형 프로브 학습이 필요합니다.",
  },
  {
    id: "wound_presence",
    title: "발에 상처가 있는지 분류",
    eyebrow: "Task 02",
    description: "상처가 있는지 없는지를 빠르게 확인합니다.",
    homeSummary: "현재는 segmentation 결과의 wound detected 여부를 우선 활용합니다.",
    status: "partial",
    models: ["dinov3_fastinst_d3_segmentation", "dinov3_linear_dfu"],
    guide:
      "현재 전용 wound / non-wound 분류 헤드는 연결되어 있지 않습니다. 세그멘테이션 결과의 wound detected 여부를 임시 분류 신호로 사용합니다.",
    alertTitle: "현재 상태",
    alertBody:
      "전용 상처 유무 분류 헤드는 아직 별도 task로 연결되어 있지 않습니다. 지금은 segmentation 결과를 기반으로 상처 유무를 확인합니다.",
  },
  {
    id: "grade0_vs_normal",
    title: "상처 없음: DFU Grade 0 vs Normal Foot",
    eyebrow: "Task 03",
    description: "상처가 없을 때 Grade 0인지 정상 발인지 추가 분기합니다.",
    homeSummary: "UI 구조만 먼저 준비하고, 전용 모델 연결은 다음 단계입니다.",
    status: "pending",
    models: [],
    guide: "이 task는 아직 전용 모델과 파이프라인이 연결되지 않았습니다.",
    alertTitle: "준비 중",
    alertBody: "전용 분류 모델과 backend 파이프라인이 아직 없습니다.",
  },
  {
    id: "segmentation",
    title: "상처 세그멘테이션",
    eyebrow: "Task 04",
    description: "발 이미지라면 backbone feature를 캐싱한 뒤 상처 segmentation을 수행합니다.",
    homeSummary: "세그멘테이션 결과를 모션처럼 드러나게 표시합니다.",
    status: "active",
    models: ["dinov3_fastinst_d3_segmentation"],
    guide:
      "세그멘테이션 결과는 공통 DINOv3 feature cache 상태와 함께 보여줍니다. overlay와 mask가 순차적으로 나타나도록 애니메이션 처리됩니다.",
    alertTitle: "현재 상태",
    alertBody:
      "세그멘테이션 backend는 fine-tuned weight가 없으면 demo backend로 자동 폴백할 수 있습니다. 결과 카드에서 backend와 weights 상태를 꼭 확인하세요.",
  },
  {
    id: "grading",
    title: "상처 bbox crop 후 Wagner / SINBAD 분류",
    eyebrow: "Task 05",
    description: "상처 bbox를 기준으로 crop한 뒤 Wagner grade와 SINBAD score를 판단합니다.",
    homeSummary: "현재는 분류 UI를 먼저 제공하고, bbox crop 재분류 파이프라인은 다음 단계입니다.",
    status: "partial",
    models: ["dinov3_linear_wagner", "dinov3_linear_sinbad"],
    guide:
      "현재 backend는 bbox crop 후 재분류를 아직 자동 수행하지 않습니다. 우선 분류 task를 분리해 놓고, 다음 단계에서 crop 기반 재추론으로 연결하는 것이 맞습니다.",
    alertTitle: "현재 상태",
    alertBody:
      "Wagner / SINBAD 페이지는 먼저 진입 구조와 결과 카드 형태를 제공합니다. bbox crop 재분류는 backend 추가 구현이 필요합니다.",
  },
];

const state = {
  modelCatalog: [],
  currentTaskId: null,
  cameraStream: null,
  previewScale: 1,
};

function setStatus(text, stateName) {
  apiStatus.textContent = text;
  apiStatus.className = `status ${stateName || ""}`.trim();
}

function percent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

function ensureFile() {
  const file = fileInput.files?.[0];
  if (!file && !previewImage.src) {
    throw new Error("먼저 이미지를 업로드하세요.");
  }
  return file;
}

function setCameraStatus(text) {
  if (cameraStatus) {
    cameraStatus.textContent = text;
  }
}

function applyPreviewScale() {
  if (!previewViewport) return;
  previewViewport.style.transform = `scale(${state.previewScale})`;
}

function setPreviewScale(nextScale) {
  const clamped = Math.max(0.5, Math.min(3, nextScale));
  state.previewScale = clamped;
  applyPreviewScale();
}

function resetPreviewScale() {
  state.previewScale = 1;
  applyPreviewScale();
}

function byId(id) {
  return state.modelCatalog.find((model) => model.id === id);
}

function displayTitle(model) {
  return MODEL_COPY[model.id]?.title || model.title;
}

function displaySummary(model) {
  return MODEL_COPY[model.id]?.summary || model.summary;
}

function kindLabel(kind) {
  if (kind === "classification") return "분류";
  if (kind === "segmentation") return "세그멘테이션";
  if (kind === "visualization") return "시각화";
  return kind;
}

function taskStatusText(status) {
  if (status === "active") return "실행 가능";
  if (status === "partial") return "부분 연결";
  return "준비 중";
}

function renderTaskHome() {
  taskHome.innerHTML = TASKS.map((task) => {
    const runnableCount = task.models.filter((modelId) => byId(modelId)).length;
    return `
      <button class="taskCard" type="button" data-task-id="${task.id}">
        <div class="taskCardHeader">
          <div>
            <p class="sectionEyebrow">${task.eyebrow}</p>
            <h3>${task.title}</h3>
            <p class="taskMeta">${task.homeSummary}</p>
          </div>
          <span class="taskBadge ${task.status === "active" ? "" : task.status}">${taskStatusText(task.status)}</span>
        </div>
        <div class="taskFooter">
          <span>${runnableCount}개 모델 연결</span>
          <strong>열기 →</strong>
        </div>
      </button>
    `;
  }).join("");

  [...taskHome.querySelectorAll("[data-task-id]")].forEach((button) => {
    button.addEventListener("click", () => openTask(button.dataset.taskId));
  });
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus("카메라 미지원");
    nextAction.textContent = "이 브라우저는 실시간 카메라 스트림을 지원하지 않습니다.";
    return;
  }

  try {
    if (!state.cameraStream) {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
    }

    cameraVideo.srcObject = state.cameraStream;
    cameraVideo.classList.remove("isHidden");
    liveOverlay.classList.remove("isHidden");
    previewImage.src = "";
    emptyPreview.hidden = true;
    resetPreviewScale();
    imageState.textContent = "실시간 프리뷰";
    setCameraStatus("카메라 연결됨");
    nextAction.textContent = "실시간 카메라 프리뷰가 연결되었습니다. 현재 화면을 캡처해 task 실행에 사용할 수 있습니다.";
  } catch (error) {
    setCameraStatus("카메라 거부됨");
    nextAction.textContent = `카메라 연결 실패: ${error.message}`;
  }
}

function captureCurrentFrame() {
  if (!state.cameraStream || cameraVideo.classList.contains("isHidden")) {
    nextAction.textContent = "먼저 카메라를 켜세요.";
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth || 720;
  canvas.height = cameraVideo.videoHeight || 960;
  const context = canvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  previewImage.src = canvas.toDataURL("image/png");
  cameraVideo.classList.add("isHidden");
  liveOverlay.classList.add("isHidden");
  emptyPreview.hidden = true;
  resetPreviewScale();
  imageState.textContent = `${canvas.width} x ${canvas.height}`;
  setCameraStatus("캡처 이미지 사용");
  nextAction.textContent = "카메라 프레임을 캡처했습니다. 이제 task를 선택해 실행할 수 있습니다.";
}

function handlePreviewWheel(event) {
  if (!event.ctrlKey) return;
  event.preventDefault();
  const delta = event.deltaY < 0 ? 0.12 : -0.12;
  setPreviewScale(state.previewScale + delta);
}

function loadSelectedFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  imageState.textContent = `${Math.round(file.size / 1024)} KB`;
  previewImage.src = URL.createObjectURL(file);
  cameraVideo.classList.add("isHidden");
  liveOverlay.classList.add("isHidden");
  emptyPreview.hidden = true;
  resetPreviewScale();
  setCameraStatus("업로드 이미지 사용");
}

function renderTaskModels(task) {
  const models = task.models.map((modelId) => byId(modelId)).filter(Boolean);
  if (!models.length) {
    taskModelList.innerHTML = `<div class="emptyState">이 task에 연결된 모델이 아직 없습니다.</div>`;
    return;
  }

  taskModelList.innerHTML = models
    .map((model) => {
      const stateText = model.weights_found ? "가중치 연결됨" : "가중치 없음";
      return `
        <article class="modelTile">
          <div class="modelTileHeader">
            <div>
              <strong>${displayTitle(model)}</strong>
              <p class="taskMeta">${displaySummary(model)}</p>
            </div>
            <span class="modelState ${model.weights_found ? "" : "missing"}">${stateText}</span>
          </div>
          <p class="taskMeta">backend: ${model.backend} · task: ${model.task}</p>
          <button class="modelRunButton" type="button" data-model-id="${model.id}">
            이 모델 실행
          </button>
        </article>
      `;
    })
    .join("");

  [...taskModelList.querySelectorAll("[data-model-id]")].forEach((button) => {
    button.addEventListener("click", async () => {
      await runModel(button.dataset.modelId, button);
    });
  });
}

function openTask(taskId) {
  const task = TASKS.find((item) => item.id === taskId);
  if (!task) return;

  state.currentTaskId = taskId;
  detailEyebrow.textContent = task.eyebrow;
  detailTitle.textContent = task.title;
  detailDescription.textContent = task.description;
  detailAlert.innerHTML = `<strong>${task.alertTitle}</strong><span>${task.alertBody}</span>`;
  taskGuide.textContent = task.guide;
  comparisonResults.innerHTML = "";
  nextAction.textContent = "실행할 모델을 선택하세요.";
  renderTaskModels(task);

  taskHome.classList.add("isHidden");
  taskDetail.classList.remove("isHidden");
  screenMessage.textContent = `${task.title} 화면입니다. 이미지 업로드 후 모델을 실행해 보세요.`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function closeTask() {
  state.currentTaskId = null;
  taskDetail.classList.add("isHidden");
  taskHome.classList.remove("isHidden");
  comparisonResults.innerHTML = "";
  nextAction.textContent = "이미지를 업로드한 뒤 task를 선택하세요.";
  screenMessage.textContent = "아래 task 중 하나를 선택하면 새로운 화면으로 들어가 해당 단계에 맞는 모델 비교를 수행할 수 있습니다.";
}

function renderMetrics(metrics) {
  if (!metrics?.length) return "";
  return `
    <div class="metricGrid">
      ${metrics
        .map(
          (metric) => `
            <article class="metricCard">
              <span>${metric.label}</span>
              <strong>${metric.value}</strong>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderRawOutputs(rawOutputs) {
  const entries = Object.entries(rawOutputs || {});
  if (!entries.length) return "";
  return `
    <section class="rawBox">
      <h3>Raw Output</h3>
      <div class="rawGrid">
        ${entries
          .map(
            ([key, value]) => `
              <div class="rawRow">
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

function segmentationMotionHtml(result) {
  const original = result.artifacts?.find((artifact) => artifact.label === "original");
  const overlay = result.artifacts?.find((artifact) => artifact.label === "overlay");
  const mask = result.artifacts?.find((artifact) => artifact.label === "mask");
  if (!original || !overlay || !mask) return "";

  return `
    <section class="resultStack">
      <div class="motionStage">
        <img class="motionBase" src="${original.data_url}" alt="원본 이미지" />
        <img class="motionOverlay" src="${overlay.data_url}" alt="세그멘테이션 오버레이" />
        <img class="motionMask" src="${mask.data_url}" alt="세그멘테이션 마스크" />
      </div>
      <div class="artifactGrid">
        ${[original, overlay, mask]
          .map(
            (artifact) => `
              <figure class="artifactThumb">
                <img src="${artifact.data_url}" alt="${artifact.label}" />
                <figcaption>${artifact.label}</figcaption>
              </figure>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function standardArtifactsHtml(artifacts) {
  if (!artifacts?.length) return "";
  return `
    <div class="artifactGrid">
      ${artifacts
        .map(
          (artifact) => `
            <figure class="artifactThumb">
              <img src="${artifact.data_url}" alt="${artifact.label}" />
              <figcaption>${artifact.label}</figcaption>
            </figure>
          `,
        )
        .join("")}
    </div>
  `;
}

function taskSpecificMessage(result) {
  if (result.model.id === "dinov3_linear_foot" && !result.model.weights_found) {
    return "현재 foot 분류 head 가중치가 연결되지 않아 이 결과는 구조 검증용입니다. 의미 있는 foot / non-foot 판별을 하려면 선형 프로브 학습이 필요합니다.";
  }
  if (result.model.id === "dinov3_backbone_pca") {
    return "PCA는 patch token 변화가 큰 축을 요약하고, cosine map은 기준 patch와 닮은 영역을 보여줍니다. 두 결과를 같이 보면 backbone이 실제로 어떤 시각 단서를 구분에 쓰는지 해석할 수 있습니다.";
  }
  if (result.model.id === "dinov3_fastinst_d3_segmentation") {
    return "세그멘테이션 결과는 DINO feature cache를 사용하며, overlay가 순차적으로 드러나는 모션형 뷰로 표시됩니다.";
  }
  return result.note || "";
}

function upsertResultCard(result) {
  const existing = document.querySelector(`[data-result-id="${result.model.id}"]`);
  const note = taskSpecificMessage(result);
  const artifactsHtml =
    result.model.kind === "segmentation" || state.currentTaskId === "segmentation"
      ? segmentationMotionHtml(result) || standardArtifactsHtml(result.artifacts)
      : standardArtifactsHtml(result.artifacts);

  const html = `
    <article class="resultCard" data-result-id="${result.model.id}">
      <header class="resultHeader">
        <div>
          <p class="sectionEyebrow">${kindLabel(result.model.kind)}</p>
          <h3>${displayTitle(result.model)}</h3>
          <p class="metricMeta">${displaySummary(result.model)}</p>
        </div>
        <span class="resultBadge">${result.model.backend}</span>
      </header>

      <div class="resultBody">
        <div class="heroMetric">
          <span>주요 결과</span>
          <strong>${result.primary_label || result.status}</strong>
        </div>
        <p class="metricMeta">신뢰도 ${typeof result.score === "number" ? percent(result.score) : "-"} · ${result.timing_ms} ms · FPS ${result.fps}</p>
        <p class="metricMeta">Feature backend ${result.feature_backend || "-"} · cache ${result.feature_cache_hit ? "hit" : "miss"}</p>
        ${note ? `<p class="resultNote">${note}</p>` : ""}
        ${renderMetrics(result.metrics)}
        ${renderRawOutputs(result.raw_outputs)}
        ${artifactsHtml}
        ${
          result.model.id === "dinov3_backbone_pca"
            ? `<section class="explainBox">
                <h3>PCA와 Cosine을 보는 이유</h3>
                <p>PCA map은 patch feature가 크게 갈리는 위치를 보여줍니다. 발 윤곽, 상처 주변 조직, 피부 결 변화처럼 backbone이 서로 다르게 인식하는 구간을 확인할 수 있습니다.</p>
                <p>Cosine map은 기준이 되는 중심 patch와 비슷한 feature를 가진 영역을 보여줍니다. 즉 특정 patch와 같은 성격의 영역이 어디까지 퍼져 있는지 확인하는 용도입니다.</p>
                <p>이 두 시각화를 함께 보면, 분류 head를 붙이기 전에 backbone 자체가 발과 상처를 구분할 단서를 충분히 갖고 있는지 먼저 판단할 수 있습니다.</p>
              </section>`
            : ""
        }
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
    setStatus(response.ok ? "API 연결됨" : "API 오류", response.ok ? "ok" : "error");
  } catch {
    setStatus("API 오프라인", "error");
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "모델 목록을 불러오지 못했습니다.");
  }
  state.modelCatalog = data.models || [];
  renderTaskHome();
}

async function runModel(modelId, button) {
  const file = ensureFile();
  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  } else if (previewImage.src.startsWith("data:image/")) {
    const blob = await (await fetch(previewImage.src)).blob();
    formData.append("file", blob, "camera-capture.png");
  } else {
    throw new Error("실행 가능한 이미지 입력이 없습니다.");
  }
  button.disabled = true;
  nextAction.textContent = `${button.closest(".modelTile")?.querySelector("strong")?.textContent || modelId} 실행 중...`;

  try {
    const response = await fetch(`/api/models/${modelId}/run`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "모델 실행에 실패했습니다.");
    }
    upsertResultCard(data);
    nextAction.textContent = `${data.model.title} 실행이 완료되었습니다.`;
  } catch (error) {
    nextAction.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  loadSelectedFile(fileInput.files?.[0]);
});

backButton.addEventListener("click", closeTask);
startCameraButton?.addEventListener("click", startCamera);
captureButton?.addEventListener("click", captureCurrentFrame);
previewDropzone?.addEventListener("wheel", handlePreviewWheel, { passive: false });
previewDropzone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  previewDropzone.classList.add("dragOver");
});
previewDropzone?.addEventListener("dragleave", () => {
  previewDropzone.classList.remove("dragOver");
});
previewDropzone?.addEventListener("drop", (event) => {
  event.preventDefault();
  previewDropzone.classList.remove("dragOver");
  loadSelectedFile(event.dataTransfer?.files?.[0]);
});

Promise.all([checkHealth(), loadModels()]).catch((error) => {
  nextAction.textContent = error.message;
});
