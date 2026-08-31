from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import requests

load_dotenv()

app = Flask(__name__)

MODEL_ID = "facebook/bart-large-cnn"

# Endpoints with fallback routes
ENDPOINTS = [
    f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}",
    f"https://api.huggingface.co/models/{MODEL_ID}",
    f"https://api-inference.huggingface.co/models/{MODEL_ID}",
]

MAX_INPUT_CHARS = 12000

# Strict parameter profiles to prevent mode overlap
LENGTH_PROFILES = {
    "tldr": {
        "min_length": 15,
        "max_length": 45,
        "length_penalty": 0.6,  # Favors concise 1-sentence takeaways
        "num_beams": 3,
    },
    "standard": {
        "min_length": 65,
        "max_length": 150,
        "length_penalty": 1.6,  # Enforces a complete 3-4 sentence paragraph
        "num_beams": 4,
    },
    "detailed": {
        "min_length": 120,
        "max_length": 300,
        "length_penalty": 2.2,  # Encourages deep, multi-sentence extraction
        "num_beams": 4,
    },
}


def _word_count(text: str) -> int:
  return len(text.split())


def _reduction_pct(original_words: int, summary_words: int) -> float:
  if original_words <= 0:
    return 0.0
  pct = (1 - (summary_words / original_words)) * 100
  return round(max(pct, 0.0), 1)


def query_huggingface(
    text: str, profile: dict[str, Any], token: str
) -> tuple[str | None, int, str]:
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }

  payload = {
      "inputs": text,
      "parameters": {
          "min_length": profile["min_length"],
          "max_length": profile["max_length"],
          "length_penalty": profile["length_penalty"],
          "num_beams": profile["num_beams"],
          "no_repeat_ngram_size": 3,
          "early_stopping": True,
          "do_sample": False,
      },
      "options": {"wait_for_model": True},
  }

  last_error = ""

  for url in ENDPOINTS:
    try:
      response = requests.post(url, headers=headers, json=payload, timeout=40)

      if response.status_code == 503:
        return (
            None,
            503,
            "The model is warming up on Hugging Face. Please retry in 10"
            " seconds.",
        )

      if response.status_code == 401:
        return (
            None,
            401,
            "Invalid Hugging Face token. Please verify HF_TOKEN in your .env"
            " file.",
        )

      data = response.json()

      if isinstance(data, list) and len(data) > 0:
        summary = data[0].get("summary_text", "")
        if summary:
          return summary.strip(), 200, ""

      if isinstance(data, dict):
        if "summary_text" in data:
          return data["summary_text"].strip(), 200, ""
        if "error" in data:
          err_msg = str(data["error"])
          if "loading" in err_msg.lower():
            return (
                None,
                503,
                "Model is currently loading. Please retry in 10 seconds.",
            )
          last_error = err_msg

    except requests.exceptions.RequestException as exc:
      last_error = f"Network resolution issue: {str(exc)}"
      continue

  return (
      None,
      502,
      last_error
      or "Could not reach Hugging Face. Please check your internet connection.",
  )


@app.route("/")
def index():
  return render_template("index.html")


@app.route("/api/health")
def health():
  token_present = bool(os.getenv("HF_TOKEN", "").strip())
  return jsonify({
      "ok": token_present,
      "model": MODEL_ID,
      "status": "ready" if token_present else "missing_token",
  }), (200 if token_present else 500)


@app.route("/api/summarize", methods=["POST"])
def summarize():
  token = os.getenv("HF_TOKEN", "").strip()
  if not token:
    return (
        jsonify({
            "error": "HF_TOKEN is missing. Add it to your .env file or Render"
            " Environment Variables."
        }),
        500,
    )

  payload = request.get_json(silent=True) or {}
  text = payload.get("text", "")
  if not isinstance(text, str) or not text.strip():
    return jsonify({"error": "Please enter text to summarize."}), 400

  text = text.strip()
  if len(text) > MAX_INPUT_CHARS:
    text = text[:MAX_INPUT_CHARS]

  mode = str(payload.get("mode") or "standard").lower()
  profile = dict(LENGTH_PROFILES.get(mode, LENGTH_PROFILES["standard"]))

  original_words = _word_count(text)

  # Prevent min_length from exceeding the word count of smaller input texts
  if original_words < profile["min_length"]:
    profile["min_length"] = max(10, original_words // 2)
    profile["max_length"] = max(profile["min_length"] + 15, original_words)

  summary, status_code, error_msg = query_huggingface(text, profile, token)

  if error_msg or not summary:
    return jsonify({"error": error_msg or "Failed to summarize."}), status_code

  summary_words = _word_count(summary)

  return jsonify({
      "summary": summary,
      "original_words": original_words,
      "summary_words": summary_words,
      "reduction_pct": _reduction_pct(original_words, summary_words),
      "mode": mode,
      "model": MODEL_ID,
  })


if __name__ == "__main__":
  port = int(os.getenv("PORT", "5000"))
  app.run(host="0.0.0.0", port=port, debug=True)