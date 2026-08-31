const MODE_CLASSES = {
  active: "border-teal-400/70 bg-teal-400/10 text-teal-200",
  idle: "border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-200",
};

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
  charCount.textContent = `${chars.toLocaleString()} character${chars === 1 ? "" : "s"}`;
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
    chip.className = `mode-chip rounded-full border px-4 py-2 text-sm font-medium transition ${
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

async function generateSummary() {
  hideAlert();
  const text = sourceText.value.trim();
  if (!text) {
    showAlert("Paste some text first. Empty input cannot be summarized.");
    return;
  }

  setLoading(true);
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 95000);
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
      if (response.status === 503) {
        showAlert(data.error || "The Hugging Face model is warming up. Wait a few seconds and try again.");
        setStatus("warning", "Model warming up");
        return;
      }
      if (response.status === 500 && /HF_TOKEN/i.test(data.error || "")) {
        showAlert(data.error);
        setStatus("error", "Missing HF_TOKEN");
        return;
      }
      showAlert(data.error || "Summarization failed. Check your connection and try again.");
      return;
    }

    summaryText.textContent = data.summary;
    statsBadge.textContent = `${data.reduction_pct}% shorter • ${data.original_words} → ${data.summary_words} words`;
    outputCard.classList.remove("hidden");
    copyBtn.textContent = "Copy to Clipboard";
    setStatus("ready", "Ready");
  } catch (error) {
    if (error.name === "AbortError") {
      showAlert("The request timed out waiting for Hugging Face. Please retry.");
    } else {
      showAlert("Network error. Confirm the server is running and try again.");
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
