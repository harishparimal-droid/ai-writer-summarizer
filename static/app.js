const MODE_CLASSES = {
  active: "border-teal-400/70 bg-teal-400/10 text-teal-100",
  idle: "border-zinc-800 bg-zinc-900 text-zinc-300 hover:border-zinc-600 hover:text-zinc-100",
};

const MODE_LABELS = {
  tldr: "TL;DR",
  standard: "Standard",
  detailed: "In-Depth",
};

const MAX_INPUT_CHARS = 12000;
const REQUEST_TIMEOUT_MS = 130000;

const sourceText = document.getElementById("source-text");
const charCount = document.getElementById("char-count");
const wordCount = document.getElementById("word-count");
const clearBtn = document.getElementById("clear-btn");
const summarizeBtn = document.getElementById("summarize-btn");
const summarizeLabel = document.getElementById("summarize-label");
const spinner = document.getElementById("spinner");
const alertBox = document.getElementById("alert-box");
const outputCard = document.getElementById("output-card");
const summaryText = document.getElementById("summary-text");
const statsBadge = document.getElementById("stats-badge");
const copyBtn = document.getElementById("copy-btn");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const modeChips = document.querySelectorAll(".mode-chip");

let selectedMode = "standard";
let copyResetTimer = null;

function countWords(value) {
  const trimmed = value.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

function updateCounters() {
  const value = sourceText.value;
  const chars = value.length;
  const words = countWords(value);
  charCount.textContent = `${chars.toLocaleString()} / ${MAX_INPUT_CHARS.toLocaleString()} characters`;
  wordCount.textContent = `${words.toLocaleString()} word${words === 1 ? "" : "s"}`;
  autoGrow();
}

function autoGrow() {
  sourceText.style.height = "auto";
  sourceText.style.height = `${Math.max(180, sourceText.scrollHeight)}px`;
}

function setMode(mode) {
  selectedMode = mode;
  modeChips.forEach((chip) => {
    const isActive = chip.dataset.mode === mode;
    chip.className = `mode-chip rounded-xl border px-4 py-3 text-left transition ${
      isActive ? MODE_CLASSES.active : MODE_CLASSES.idle
    }`;
    chip.setAttribute("aria-pressed", String(isActive));
  });
}

function showAlert(message) {
  alertBox.textContent = message;
  alertBox.classList.remove("hidden");
}

function hideAlert() {
  alertBox.classList.add("hidden");
  alertBox.textContent = "";
}

function setLoading(isLoading) {
  summarizeBtn.disabled = isLoading;
  spinner.classList.toggle("hidden", !isLoading);
  summarizeLabel.textContent = isLoading ? "Generating…" : "Generate Summary";
}

function setStatus(state, label) {
  statusLabel.textContent = label;
  const colors = {
    ready: "bg-teal-400 shadow-[0_0_10px_rgba(45,212,191,0.8)]",
    warning: "bg-amber-400",
    error: "bg-rose-500",
    idle: "bg-zinc-500",
  };
  statusDot.className = `h-2 w-2 rounded-full ${colors[state] || colors.idle}`;
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health", { headers: { Accept: "application/json" } });
    const data = await response.json();
    if (data.ok) {
      setStatus("ready", "Ready");
      return;
    }
    setStatus("error", "Missing HF_TOKEN");
  } catch {
    setStatus("warning", "Unreachable");
  }
}

function handleApiError(status, data) {
  const message = data.error || "Failed to summarize.";

  if (status === 400) {
    showAlert(message);
    return;
  }
  if (status === 401) {
    showAlert(message);
    setStatus("error", "Invalid token");
    return;
  }
  if (status === 500 && /HF_TOKEN/i.test(message)) {
    showAlert(message);
    setStatus("error", "Missing HF_TOKEN");
    return;
  }
  if (status === 503) {
    showAlert(message);
    setStatus("warning", "Model warming up");
    return;
  }
  if (status === 502) {
    showAlert(message);
    setStatus("warning", "HF unreachable");
    return;
  }
  showAlert(message);
}

async function generateSummary() {
  hideAlert();
  const text = sourceText.value.trim();
  if (!text) {
    showAlert("Please enter text to summarize.");
    return;
  }

  setLoading(true);
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ text, mode: selectedMode }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }

    if (!response.ok) {
      handleApiError(response.status, data);
      return;
    }

    const modeLabel = MODE_LABELS[data.mode] || data.mode || MODE_LABELS[selectedMode];
    summaryText.textContent = data.summary;
    statsBadge.textContent = `${modeLabel} · ${data.reduction_pct}% shorter · ${data.original_words} → ${data.summary_words} words`;
    outputCard.classList.remove("hidden");
    copyBtn.textContent = "Copy to Clipboard";
    setStatus("ready", "Ready");
  } catch (error) {
    if (error.name === "AbortError") {
      showAlert("The request timed out waiting for Hugging Face. Please retry in 10 seconds.");
    } else {
      showAlert("Could not reach Hugging Face. Please check your internet connection.");
    }
  } finally {
    setLoading(false);
  }
}

async function copySummary() {
  const value = summaryText.textContent || "";
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const helper = document.createElement("textarea");
    helper.value = value;
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
  }
  copyBtn.textContent = "Copied!";
  if (copyResetTimer) clearTimeout(copyResetTimer);
  copyResetTimer = setTimeout(() => {
    copyBtn.textContent = "Copy to Clipboard";
  }, 1800);
}

sourceText.addEventListener("input", updateCounters);
clearBtn.addEventListener("click", () => {
  sourceText.value = "";
  updateCounters();
  hideAlert();
  outputCard.classList.add("hidden");
  sourceText.focus();
});
summarizeBtn.addEventListener("click", generateSummary);
copyBtn.addEventListener("click", copySummary);
modeChips.forEach((chip) => {
  chip.addEventListener("click", () => setMode(chip.dataset.mode));
});

sourceText.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    generateSummary();
  }
});

setMode("standard");
updateCounters();
refreshHealth();
setInterval(refreshHealth, 45000);
