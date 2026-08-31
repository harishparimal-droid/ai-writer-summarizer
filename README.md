# Nimbus Summarizer

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/Inference-facebook%2Fbart--large--cnn-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/facebook/bart-large-cnn)
[![Render](https://img.shields.io/badge/Deploy-Render%20Free%20Tier-46E3B7?logo=render&logoColor=black)](https://render.com/)
[![RAM](https://img.shields.io/badge/Runtime-under%2050MB%20RAM-0D9488)](#architecture)

Cloud-native AI text summarizer. The web process is a thin Flask app that calls **Hugging Face Serverless Inference** with `requests` only. There is no `huggingface_hub` client, no local `transformers`, PyTorch, torchvision, or spaCy — so it fits Render’s free tier.

## Architecture

```
Browser  ──JSON──►  Flask (gunicorn)  ──HTTPS──►  Hugging Face (fallback chain)
                         │                              │
                    templates/              1. router.huggingface.co/hf-inference
                    static/                 2. api.huggingface.co/models
                         │                  3. api-inference.huggingface.co
                    Render free web         facebook/bart-large-cnn
```

| Approach | What happens | RAM on your server |
| --- | --- | --- |
| **This app** | `requests.post` to HF REST, `wait_for_model: true`, 40s timeout per endpoint | Flask + gunicorn only (typically well under 50MB) |
| Local transformers | Download BART weights, run `torch` in-process | Multiple GB; OOM on Render free tier |

Unknown modes fall back to **standard**. If the source has fewer words than `min_length`, Flask lowers `min_length` / `max_length` so short paste still works. Input is truncated at **12,000** characters.

### Length profiles (`LENGTH_PROFILES` in `app.py`)

| Mode | `min_length` | `max_length` | `length_penalty` | `num_beams` | Intent |
| --- | ---: | ---: | ---: | ---: | --- |
| `tldr` | 15 | 45 | 0.6 | 3 | Concise 1-sentence takeaway |
| `standard` | 65 | 150 | 1.6 | 4 | Complete 3–4 sentence paragraph |
| `detailed` | 120 | 300 | 2.2 | 4 | Deep multi-sentence extract |

Shared generation flags: `no_repeat_ngram_size=3`, `early_stopping=True`, `do_sample=False`.

If Hugging Face returns **503** or a “loading” error, the API asks the client to retry in **10 seconds**.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Dark-mode summarizer UI |
| `GET` | `/api/health` | Token presence for the header indicator |
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

Error contracts (messages come from `app.py`):

- **400** — empty text (`Please enter text to summarize.`)
- **401** — invalid `HF_TOKEN`
- **500** — `HF_TOKEN` missing
- **502** — all HF endpoints failed / network
- **503** — model warming up; retry in ~10s

## Local setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set a Hugging Face token with Inference permission:

```bash
HF_TOKEN=hf_your_token_here
```

Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

3. Run:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Production-style (Linux/macOS):

```bash
gunicorn app:app
```

On Windows, prefer `python app.py`.

## Deploy on Render (free tier)

1. Push this repository to GitHub.
2. **New → Web Service** and connect the repo.
3. Settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app` (from `Procfile`)
4. Set environment variable **`HF_TOKEN`**.
5. Deploy. First inference can 503 while `facebook/bart-large-cnn` loads; wait 10 seconds and retry.

`runtime.txt` pins CPython 3.12.

## Project layout

```
ai-text-summarizer/
├── app.py                 # Flask + requests fallback chain
├── requirements.txt       # Flask, requests, gunicorn, python-dotenv
├── Procfile               # web: gunicorn app:app
├── runtime.txt
├── .env.example
├── .gitignore
├── templates/index.html
└── static/app.js
```

## Security notes

- `.env` is gitignored. Use Render’s dashboard in production.
- The token never leaves the server; the browser only talks to Flask.

## License

MIT — use this as a portfolio or production starter.
