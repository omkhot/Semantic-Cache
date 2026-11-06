# Resume bullets and interview preparation

## Resume entry (fill the brackets ONLY with numbers from your own `results/` files)

**SemanticCache: LLM response caching layer** | Python, FastAPI, FAISS, sentence-transformers, Gemini API
- Built a semantic caching proxy for LLM calls using sentence embeddings and vector search; on a [N]-request simulated workload it cut LLM calls by [X]% and served cache hits in [Y] ms (p50) versus [Z] ms for a model call
- Tuned the similarity threshold with a precision/recall sweep on Quora Question Pairs ([N] pairs) and added a verification stage (rule checks + cross-encoder) that reduced false cache hits from [A]% to [B]% at comparable hit rate
- Added TTL/LRU eviction, namespace isolation keyed on system prompt, request coalescing, disk persistence, and live metrics; covered by 38 automated tests

Where to find each number:
- `[X]`, `[Y]`, `[Z]`: `results/workload.md` (`python -m evaluation.simulate`). Miss latency there is simulated, so say "simulated workload", or replace the simulated figure with your measured Gemini latency (`latency_ms.llm` in a real `/v1/chat` response)
- `[A]`, `[B]`, `[N]`: `results/comparison_qqp.md` (`python -m evaluation.run_eval --dataset qqp`)

If a number does not look impressive, report it honestly with the trade-off. "Hit rate X% at 1% false hits" is a good sentence.

## 60-second pitch
"LLM calls are slow and cost money, and many users ask the same thing in different words. I built a caching proxy that embeds each question, searches for similar cached ones, and reuses the answer. The hard part is that embeddings say 'capital of France' and 'capital of Germany' are almost identical. So I measured the trade-off on a labelled dataset, then added a verification stage that checks the details, such as entities, numbers and negations, before serving a cached answer. I report hit rate against false-hit rate instead of just hit rate."

## Questions you will probably get

**Why not just use a high threshold?** A high threshold removes false hits but also removes real paraphrases. The sweep plot shows it: to get false hits near zero by threshold alone you lose most hits. The verifier lets you keep a lower threshold.

**How did you choose the threshold?** Swept 0.50 to 0.99 on labelled pairs and picked the point with the highest hit rate under a 1% false-hit rate. The right point depends on how costly a wrong answer is in the application.

**How is this different from GPTCache / Redis semantic cache?** Same core idea. My focus was on measuring correctness (false hits), verification, and safe keys (system prompt in the namespace). I did not claim to invent semantic caching.

**What does it get wrong?** Rules cannot see lowercase topic swaps (quicksort vs merge sort) or subtle meaning changes; the cross-encoder helps but is not perfect. Time-sensitive questions need a short TTL. Multi-turn context is not handled.

**Why a cross-encoder for verification and a bi-encoder for retrieval?** A bi-encoder embeds each question once and compares by dot product, which is fast and indexable but coarse. A cross-encoder reads both questions together, which is more accurate but cannot be pre-indexed, so it only checks the few candidates that survive the first stage.

**Why FAISS `IndexFlatIP`?** Exact search is sub-millisecond at this scale and removes recall error from the analysis. At millions of entries I would switch to HNSW or IVF and accept approximate recall.

**What is request coalescing?** If five identical questions arrive while the first is still being answered, they all wait for that one LLM call, so you pay once. It is the same idea as single-flight in Go.

**How would you scale it?** Move the cache to a shared store (Redis with vector search, or Qdrant), run several API workers, and keep the verifier as a stateless step.

**How do you know the benchmark is meaningful?** QQP is human-labelled real questions. My synthetic set is only a stress test of known failure types, so I report it separately and do not use it for headline accuracy.

**What would you do next?** Per-category thresholds, caching with conversation context, an online feedback signal ("this answer was wrong") that evicts bad entries, and a Redis backend.
