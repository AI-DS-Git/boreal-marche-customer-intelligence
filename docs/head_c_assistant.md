# Head C - assistant design and evaluation

## What it does
A bilingual (EN/FR) customer assistant that answers policy questions with citations, suggests products for a
basket, describes a product from its code, and redirects retention questions to the Prediction page. Three
interchangeable providers sit behind one `Provider` interface (`app/services/assistant.py`):

| Provider | What it is | When it is available |
|---|---|---|
| `rule` | deterministic baseline: typed tools chosen by keyword rules, retrieval with a similarity floor | always |
| `openai` | hosted model with tool calling, token streaming (`app/services/llm.py`) | when `OPENAI_API_KEY` is set |
| `ollama` | local model with tool calling, token streaming | when `OLLAMA_BASE_URL` is set |

## Retrieval
Policies are embedded once per language and a question is matched by cosine similarity of the best chunk
(`app/services/retrieval.py`), so a French question can match the English text of a policy. Back-ends:
`openai` (text-embedding-3-small), `ollama` (bge-m3), `hashing` (offline fallback, the deployed default) and
`lexical` (the earlier baseline). The offline `hashing` embedder is built from hashed words and character
n-grams: deterministic and dependency-free, but **lexical, not semantic**. The real multilingual embedders
have not been measured yet in this repository (see Limitations).

## Guardrails (deterministic, around the model)
1. **Before:** privacy (another customer's data) and out-of-scope messages are refused without calling the model.
2. **During:** four typed tools only (`policy_search`, `recommendations`, `product_lookup`, `retention_info`);
   at most three rounds; the last round has no tools.
3. **After:** `verify_answer` rejects an answer containing a product code that no tool returned, or a percentage or
   decimal that is not verbatim in a tool result. A model that skipped the tools is not trusted either. On
   rejection the deterministic rule answer replaces it and a `verifier_fallback` entry is added to `tool_calls`.
4. `policy_search` returns `NO_EVIDENCE` below the similarity floor; the assistant then refuses and hands the
   question to staff instead of guessing.

## Streaming
`POST /chat/stream` returns server-sent events: `tool`, `token`, `replace`, then one `final`. For the LLM
providers tokens are forwarded as the model produces them (tested: the first token arrives hundreds of
milliseconds before the answer completes). `replace` tells the client to discard the text so far when a
guardrail or the verifier overrides the model. The rule provider has its whole answer up front, so its tokens are
chunked words, not model tokens.

## Golden set and evaluation
`tests/golden/assistant_golden.json` holds 40 questions: 20 policy (10 EN, 10 FR, including paraphrases),
3 recommendation, 3 product lookup, 3 retention, 5 out-of-scope, 3 privacy, 3 no-evidence. Each has an expected
tool, expected refusal reason and (for policy questions) expected policy.

* **In CI** (`tests/test_assistant_golden.py`, offline, no key): the rule provider with both retrieval
  back-ends must score at least 95% on tool choice, 95% on refusal behaviour and 90% on citations. The LLM
  provider code is exercised end to end over real sockets against a local fake OpenAI/Ollama server
  (`tests/test_llm_providers.py`): tool loop, citations, verifier, guardrails, unreachable back-end, streaming.
* **Offline results, rule provider** (both retrieval back-ends): tool choice 97.5%, refusal 97.5%, citation 95%.
  Misses: q19 "customer service hours" retrieves the price-adjustment policy; q38 / q40 (student discount,
  cash gift-card refund) sit at the evidence floor and one of them slips through, depending on the back-end.
* **Caveat:** the hashing floor (0.25) was chosen while looking at these same 40 questions, and the scores of the
  weakest correct hit (0.260) and the strongest no-evidence question (0.269) overlap. Treat the figures as
  in-sample.
* **Real models** are measured with `python scripts/eval_assistant.py --provider openai|ollama [--save]`. This
  is not run in CI because it needs a key or a running Ollama, so **the tool-choice behaviour of real LLMs
  is not yet measured** in this repository.

## Limitations
* The deployed service has no LLM key: it serves the `rule` provider with offline retrieval. Setting
  `OPENAI_API_KEY` (and `EMBEDDING_PROVIDER=openai`) enables the hosted provider; the Ollama provider is
  available locally with `docker compose --profile llm up`.
* The default evidence floors for `openai` (0.30) and `ollama` (0.40) are placeholders until
  `scripts/eval_assistant.py --calibrate` is run against the real embedders.
* Policies are prototype placeholders; the assistant is not a legal or financial advisor.
