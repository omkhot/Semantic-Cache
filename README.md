# SemanticCache

**A semantic caching proxy for LLM calls.** It sits in front of an LLM (Gemini), recognises when a new question means the same as one it has already answered, and returns the stored answer in milliseconds instead of paying for another LLM call. A verification stage stops the classic failure of semantic caches: serving the answer to *"capital of France"* when someone asked about *"capital of Germany"*.

```
 client ──► FastAPI ──► normalise ──► exact match? ──yes──► cached answer   (~0.3 ms)
                           │ no
                           ▼
                  embed query (MiniLM) ──► vector search (FAISS)  ── nearest cached questions
                           │
                similarity ≥ threshold ?  ──no──┐
                           │ yes                │
                           ▼                    │
              verifier: SAME question?  ──no────┤   (entities, numbers, negation,
                           │ yes                │    antonyms, optional cross-encoder)
                           ▼                    ▼
                  cached answer (~10 ms)    call Gemini ──► store answer ──► return
```

**Highlights**
- Three cache modes you can compare: `exact`, `semantic`, `semantic+verify`
- Verifier: rule-based checks, plus an optional cross-encoder trained on Quora duplicate questions
- FAISS vector index (automatic NumPy fallback), TTL expiry, LRU eviction, per-namespace isolation
- System prompt is part of the cache key, so different personas never share answers
- Request coalescing: 5 identical simultaneous questions trigger **one** Gemini call
- Disk persistence, live metrics (hit rate, p50/p95 latency, tokens and notional cost saved)
- Evaluation suite: threshold sweep, false-hit analysis, Quora Question Pairs benchmark, workload simulation
- 38 automated tests that run offline (no API key needed)
- Demo web dashboard at `/`, Swagger docs at `/docs`

---

## 1. Quick start (Windows + Git Bash; also works on Linux/macOS)

You need **Python 3.10 or newer**. Check with `python --version`.

### Step 1: Get a free Gemini API key
Open <https://aistudio.google.com/apikey>, sign in, click **Create API key**, copy it.

### Step 2: Open a terminal in this folder
```bash
cd semantic-cache
```

### Step 3: Create an isolated Python environment
```bash
python -m venv .venv
```
*Creates a private folder `.venv` so this project's packages do not mix with other projects.*

Activate it:
```bash
source .venv/Scripts/activate      # Git Bash on Windows
.venv\Scripts\activate             # cmd / PowerShell on Windows
source .venv/bin/activate          # Linux / macOS
```
*Your prompt now starts with `(.venv)`.*

### Step 4: Install the packages
```bash
pip install -r requirements.txt
pip install -r requirements-faiss.txt     # optional; if it errors, skip it (NumPy is used instead)
pip install -r requirements-dev.txt       # only needed for tests and evaluation plots
```
*`requirements.txt` installs the web framework and the embedding library (this includes PyTorch, so allow a few minutes).*

### Step 5: Add your key
```bash
cp .env.example .env
```
Open `.env` in any editor and replace `paste-your-key-here` with your key.

### Step 6: Check the key and model name work
```bash
python -m scripts.check_gemini
```
*Lists the models your key can use and sends one test prompt. If the model in `.env` is rejected, copy a name from the printed list into `GEMINI_MODEL`.*

### Step 7: Start the server
```bash
python -m uvicorn semcache.app:app --port 8000
```
*`uvicorn` runs the FastAPI app. The first start downloads the MiniLM model (~90 MB, one time).*
You should see a line like `Ready | llm=gemini embedder=sentence-transformers/all-MiniLM-L6-v2 index=faiss ...`.

> **Shortcut:** `bash start.sh` (Git Bash) or `start.bat` (cmd) does steps 3-7 for you.

### Step 8: Try it
- **Dashboard:** open <http://localhost:8000>. Click the sample questions and watch HIT / MISS badges.
- **Scripted demo** (new terminal, venv activated): `python -m scripts.demo --clear`
- **curl:**
  ```bash
  curl -s -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" \
       -d '{"query": "What is the capital of France?"}'
  ```
  Ask again in different words (`"Which city is France's capital?"`) and look at `"cached": true`.

**No API key yet?** Set `LLM_PROVIDER=mock` in `.env` to run everything with a fake LLM.

---

## 2. API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat` | Ask a question (cache first, LLM on miss) |
| GET | `/v1/metrics` | Hit rate, latency percentiles, tokens saved |
| GET | `/v1/cache/entries` | Most recently used cached questions |
| DELETE | `/v1/cache` | Clear the cache |
| POST | `/v1/cache/save` | Save the cache to disk now (also saved on shutdown) |
| GET | `/v1/config` | Active settings (API key hidden) |
| GET | `/health` | Liveness check |

`POST /v1/chat` body: `query` (required), `namespace` (default `"default"`), `system` (optional system prompt, part of the cache key), `temperature`, `bypass_cache`.

Response: `answer`, `cached`, `cache` (`kind` = exact / semantic / coalesced / miss / bypass, `similarity`, `matched_query`, `verifier_rejections`, `reason`), `latency_ms` (total / embed / search / verify / llm), `tokens`, `model`.

---

## 3. Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | - | Your key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model name (run `scripts.check_gemini` to list) |
| `EMBEDDER` | `minilm` | `minilm` (local), `gemini` (API), `hash` (offline toy, tests only) |
| `CACHE_MODE` | `semantic+verify` | `exact`, `semantic`, `semantic+verify` |
| `SIMILARITY_THRESHOLD` | `0.85` | Minimum cosine similarity to consider a match |
| `VERIFIER` | `rules` | `none`, `rules`, `cross`, `rules+cross` |
| `TTL_SECONDS` | `86400` | Entries expire after this long (0 = never) |
| `MAX_ENTRIES` | `10000` | LRU eviction above this size |
| `PERSIST_DIR` | `data/cache_store` | Where the cache is saved (empty = do not persist) |

---

## 4. Evaluation (this is what makes it a strong project)

All commands run from the project folder with the venv active.

```bash
# A. Real benchmark: Quora Question Pairs (human-labelled duplicate / not duplicate)
python -m evaluation.run_eval --dataset qqp --embedder minilm --verifier rules

# B. Same, with the cross-encoder verifier (downloads one more small model)
python -m evaluation.run_eval --dataset qqp --embedder minilm --verifier rules+cross

# C. Controlled stress test with built-in hard cases (France vs Germany, enable vs disable, 5 vs 50)
python -m evaluation.run_eval --dataset synthetic --embedder minilm --verifier rules

# D. Workload simulation: 1000 paraphrased requests through all three cache modes
python -m evaluation.simulate --embedder minilm --threshold 0.85 --requests 1000
```

Each run writes to `results/`: a CSV of the full threshold sweep, `summary.json`, a ready-to-paste markdown table, and a PNG plot.

**What the numbers mean** (every test item is a pair: a cached question and a new question, with a human label):
- **Hit rate**: share of true duplicates the cache answered (higher = more LLM calls saved)
- **False-hit rate**: share of *different* questions the cache wrongly answered (lower = fewer wrong answers)
- **Precision**: of all cache hits, the share that were correct

The threshold is a trade-off: raise it and you get fewer wrong answers but fewer hits. The sweep shows that curve, and `summary.json` reports the threshold that gives the highest hit rate while keeping false hits at or below 1%. Put that value in `.env` as `SIMILARITY_THRESHOLD`.

**Honest scope notes**
- Use the **QQP** results for claims about accuracy. The synthetic set contains exactly the failure types the rules were built for, so the verifier looks better there than it will on real traffic.
- The workload simulation charges cache misses a **simulated** LLM latency (default 1200 ms; set `--llm-latency-ms` to what you measured). Cache-side latency (embedding, search, verification) is measured for real.
- Cost figures use placeholder prices from `.env`. On the Gemini free tier your real cost is zero.
- Run the evaluation on your machine and use only the numbers it prints.

---

## 5. Tests

```bash
python -m pytest -q
```
*Runs 38 tests in about 2 seconds, fully offline: verifier rules, vector index (FAISS and NumPy), cache behaviour (TTL, LRU, namespaces, persistence), the Gemini client (retries, errors, using a mock HTTP transport), and the API (including concurrent-request coalescing).*

---

## 6. Docker (optional)

```bash
docker compose up --build
```
Then open <http://localhost:8000>. The image uses CPU-only PyTorch and bakes in the MiniLM model. Your `.env` is read at runtime and never copied into the image. The cache persists in `./data`.

---

## 7. Project structure

```
semcache/
  app.py        FastAPI service, request flow, coalescing
  cache.py      SemanticCache: lookup pipeline, TTL, LRU, namespaces, persistence
  embedders.py  MiniLM / Gemini / offline hash embedders
  index.py      FAISS index with NumPy fallback
  verifier.py   rule checks + cross-encoder verifier
  llm.py        Gemini REST client (retries, error handling) + mock LLM
  metrics.py    counters and latency percentiles
  config.py     settings from environment / .env
  static/       demo dashboard
evaluation/     dataset builders, threshold sweep, workload simulation
scripts/        check_gemini.py, demo.py
tests/          38 offline tests
docs/           ARCHITECTURE.md, INTERVIEW.md
```

---

## 8. Design decisions and known limitations

- **Why a verifier?** Embeddings capture topic similarity, not whether details match. Tightening the threshold alone removes real paraphrases too. The verifier keeps the threshold lower and rejects only near-misses that differ in a detail.
- **Rules are conservative.** A false rejection costs one extra LLM call; a false acceptance serves a wrong answer. Rules cannot catch lowercase topic swaps (for example "quicksort" vs "merge sort"). That is what the optional cross-encoder is for.
- **Time-sensitive questions** ("today's weather", "latest news") should not be cached for 24 hours. Lower `TTL_SECONDS` or use `bypass_cache` for them.
- **Single process.** The cache lives in memory (persisted to disk), so run one worker. A shared store such as Redis would be the next step for multiple servers.
- **Multi-turn chat** is out of scope: each request is treated as a standalone question.
- **Gemini model names change.** If you see a 404, run `python -m scripts.check_gemini` and pick a listed model.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is missing` | Create `.env` from `.env.example` and paste your key |
| HTTP 404 from Gemini | Wrong model name: run `python -m scripts.check_gemini`, copy a listed name |
| HTTP 429 from Gemini | Free-tier rate limit. Wait a minute; the client already retries with backoff |
| `Could not load the sentence-transformers model` | Run `pip install -r requirements.txt`; stay online for the first run |
| `pip install faiss-cpu` fails | Skip it. The project falls back to NumPy automatically |
| Port 8000 busy | Use `--port 8001` and open that port instead |
| Old cached answers after changing embedder | Safe: the saved cache is ignored when the embedder changes. Or `curl -X DELETE localhost:8000/v1/cache` |

---

## 10. Resume

See `docs/INTERVIEW.md` for resume bullets (with blanks for YOUR measured numbers) and likely interview questions with answers.
