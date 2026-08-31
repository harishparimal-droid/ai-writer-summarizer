# Nimbus Summarizer

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/Inference-facebook%2Fbart--large--cnn-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/facebook/bart-large-cnn)
[![Render](https://img.shields.io/badge/Deploy-Render%20Free%20Tier-46E3B7?logo=render&logoColor=black)](https://render.com/)
[![RAM](https://img.shields.io/badge/Runtime-under%2050MB%20RAM-0D9488)](#architecture)

Production-grade, cloud-native AI text summarizer. The web process is a thin Flask API; **all model inference runs on Hugging Face Serverless Inference**. There is no local `transformers`, PyTorch, torchvision, or spaCy stack — which is what makes this deployable on Render’s free tier.

## Architecture

```
Browser  ──JSON──►  Flask (gunicorn)  ──HTTPS──►  Hugging Face Inference API
                         │                              │
                    templates/                    facebook/bart-large-cnn
                    static/                       (loaded in HF cloud)
                         │
                    Render free web service
                    (~small RAM footprint)
```

| Approach | What happens | RAM on your server |
| --- | --- | --- |
| **This app (serverless inference)** | `huggingface_hub.InferenceClient` / REST `POST` to `api-inference.huggingface.co` | Flask + gunicorn only (typically well under 50MB) |
| Local transformers | Download BART weights, run `torch` forward pass in-process | Multiple GB; will OOM on Render free tier |

The backend maps UI modes to generation length profiles:

| Mode | `min_length` | `max_length` |
| --- | ---: | ---: |
| `tldr` | 20 | 60 |
| `standard` | 40 | 130 |
| `detailed` | 80 | 250 |

If Hugging Face is still loading the model, the API returns **503** with a retry-friendly message instead of a generic failure.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dark-mode summarizer UI |
| `GET` | `/api/health` | Token/config status for the header indicator |
| `POST` | `/api/summarize` | `{ "text": str, "mode": "tldr" \| "standard" \| "detailed" }` |

Successful summarize response:

```json
{
  "summary": "...",
  "original_words": 240,
  "summary_words": 84,
  "reduction_pct": 65.0,
  "mode": "standard",
  "model": "facebook/bart-large-cnn"
}
```

Error contracts:

- **400** — empty text or invalid JSON/mode
- **500** — `HF_TOKEN` missing
- **503** — model warming up; client should retry

## Local setup

1. Create and activate a virtualenv (optional but recommended).
2. Install **only** the pinned cloud-native stack:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add a Hugging Face token with Inference permission:

```bash
HF_TOKEN=hf_your_token_here
```

Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

4. Run the app:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Production-style locally:

```bash
gunicorn app:app
```

On Windows, use `python app.py` if gunicorn’s worker model is unavailable.

## Deploy on Render (free tier)

1. Push this repository to GitHub.
2. In Render, **New → Web Service** and connect the repo.
3. Settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app` (from `Procfile`)
4. Add environment variable **`HF_TOKEN`** (same value as `.env`; never commit the real token).
5. Deploy. The first summarize call may 503 while `facebook/bart-large-cnn` loads on Hugging Face; retry after a few seconds.

`runtime.txt` pins CPython 3.12 for reproducible builds.

## Project layout

```
ai-text-summarizer/
├── app.py                 # Flask app + HF Inference client
├── requirements.txt       # Flask, requests, huggingface_hub, gunicorn, python-dotenv
├── Procfile               # web: gunicorn app:app
├── runtime.txt            # Python 3.12 on Render
├── .env.example           # HF_TOKEN template
├── .gitignore
├── templates/index.html   # Tailwind dark UI
└── static/app.js          # counters, modes, copy, toasts
```

## Security notes

- `.env` is gitignored. Use Render’s dashboard (or your host’s secret store) in production.
- The token is never sent to the browser; the Flask process calls Hugging Face server-side.

## License

MIT — use this as a portfolio or production starter.
