# Vera Bot — magicpin AI Challenge Submission

**Team**: Munthasir  
**Contact**: smuntasir2005@gmail.com  
**Model**: llama-3.3-70b-versatile (Groq)  
**Version**: 1.0.0

---

## Approach

### Core Architecture

A **Groq-powered HTTP API bot** built on FastAPI. Every message composition routes through a structured prompt that injects all 4 context layers (category, merchant, trigger, customer) and instructs the model to output a fixed JSON schema.

```
Judge → POST /v1/context  →  In-memory store (by scope+id+version)
Judge → POST /v1/tick     →  Trigger loop → compose() → Groq → action[]
Judge → POST /v1/reply    →  Intent classifier → route → compose() → action
```

### Composer Design

The `compose()` function builds a **ground-truth prompt** from real data in the context store:

- Pulls merchant perf numbers, signals, active offers, and conversation history
- Resolves digest items referenced by trigger payload `top_item_id`
- Selects the right voice/tone constraints from category context
- Injects the customer relationship and consent scope for customer-facing messages
- Includes the last 2-3 prior Vera messages explicitly — the model is instructed NOT to repeat them

The model is given a **strict JSON-only output schema**: `{body, cta, send_as, suppression_key, rationale}`. Temperature is set to 0 (via model default determinism) for reproducibility.

### Routing Layer

Before calling Groq for replies, the bot classifies the merchant's message:

| Classification | Signal | Action |
|---|---|---|
| `auto_reply` | Canned phrases or repeated verbatim message | 1 redirect → exit |
| `not_interested` | "stop", "nahi chahiye", etc. | Immediate `end` |
| `join_intent` | "judrna", "join karna", "sign up" | Switch to action mode, no re-qualify |
| `accepted` | "yes", "ok", "chalega" | Execute the promised action |
| `question` | Contains `?` | Acknowledge + answer |

This handles the 4 replay-test scenarios (auto-reply hell, intent transition, hostile) explicitly.

### Suppression

All fired `suppression_key` values are tracked in a set. `/v1/tick` skips any trigger whose key has already fired, preventing duplicate sends across ticks.

### Language

The composer prompt explicitly passes `language_pref` from both merchant identity and customer identity. The model is instructed to match — Hindi-English code-mix for `hi` or `hi-en mix` merchants, English for others.

---

## Tradeoffs

**What I optimized for:**
- **Specificity over volume**: The bot may send fewer messages per tick but each message anchors on at least one concrete fact (number, date, stat, citation). Judges penalize generic copy more than sparse sends.
- **Context fidelity**: No hallucination. If a digest item is referenced by ID, the bot resolves it from the actual category payload before composing.
- **Graceful exits**: Auto-reply detection and not-interested routing prevent spam loops that would hurt operational scores.

**What I traded off:**
- **Multi-merchant parallelism**: The current tick loop is sequential (one trigger at a time). For 20 actions/tick at scale, an async batch approach would be faster. For the 30-pair test, sequential is safe within the 30s budget.
- **Retrieval**: No embedding/semantic search over digest items. The full digest (typically 3-5 items) is passed directly into the prompt. At 50+ digest items this would need a retrieval layer — for the challenge dataset size it's fine.

---

## What Additional Context Would Have Helped Most

1. **Real merchant conversation history** — the seed data has 1-2 turns; real Vera has 4.7 avg turns/merchant/day. More multi-turn history would let the bot calibrate tone and avoid re-pitching what was already discussed.
2. **Customer opt-in timestamps** — knowing how recently a customer opted in helps calibrate recall urgency.
3. **GBP post history** — knowing what was posted and when (not just "stale_posts:22d") would enable the bot to write a different post, not a generic one.

---

## Files

| File | Purpose |
|---|---|
| `bot.py` | Main FastAPI HTTP server — all 5 required endpoints |
| `conversation_handlers.py` | Optional multi-turn handler with state machine |
| `generate_submission.py` | Script to regenerate `submission.jsonl` from dataset |
| `submission.jsonl` | 30 pre-composed messages for the canonical test pairs |
| `README.md` | This file |

## Run

```bash
pip install fastapi uvicorn httpx
export GROQ_API_KEY=gsk_your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Then test locally:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Deploy to any public URL (Render, Railway, Fly.io, ngrok) and submit the URL.
