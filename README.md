# Vera Bot — magicpin AI Challenge Submission

**Team**: Muntasir  
**Contact**: smuntasir2005@gmail.com  
**Model**: llama-3.3-70b-versatile (Groq)  
**Version**: 2.0.0

---

## Approach

### Core Architecture

A **Groq-powered HTTP API bot** built on FastAPI. Every message routes through a 4-context composition pipeline (category, merchant, trigger, customer) that enforces specificity, voice-match, and engagement compulsion.

```
Judge → POST /v1/context  →  In-memory store (scope+id+version keyed)
Judge → POST /v1/tick     →  Concurrent trigger processing (semaphore-limited) → compose() → Groq → action[]
Judge → POST /v1/reply    →  Intent classifier → route → compose() → action
```

### Why the Score Jumped

| Area | v1 Problem | v2 Fix |
|---|---|---|
| **Model** | `llama-3.1-8b-instant` (low quality) | `llama-3.3-70b-versatile` (70B, much better judgment) |
| **Specificity** | Prompt asked for specificity but didn't enforce | Now requires **2+ exact numbers** from trigger payload. Prompt explicitly lists payload data |
| **Category voice** | Taboo words not injected | `vocab_taboo` from category context is now passed explicitly — model told to AVOID them |
| **Tick speed** | Sequential + 2s sleep per trigger | Concurrent with `asyncio.Semaphore(3)` — 3 parallel calls, no artificial sleep |
| **Merchant fit** | Generic placeholder in system prompt | All merchant data (owner, locality, CTR vs peer) dynamically injected in user prompt |
| **Suppression key** | Wrong field reference | Correctly reads `trigger["suppression_key"]` from trigger root, not payload |
| **Reply quality** | Separate weak system prompt | Rich `REPLY_SYSTEM` with intent-aware instructions, acceptance-mode switching |
| **Anti-repetition** | Only checked last 2 history turns | Now collects all prior Vera messages (conversation_history + live history) |
| **Digest resolution** | Only checked `top_item_id` | Now checks `top_item_id`, `digest_item_id`, `alert_id` |

### Composer Design

The `build_compose_prompt()` function builds a **ground-truth prompt** from real data:

- **Trigger payload** is shown verbatim with a requirement to use ≥2 exact numbers
- **Merchant CTR vs peer** is calculated and shown (above/below peer)
- **Taboo vocabulary** per category is explicitly listed — model avoids it
- **Digest items** are resolved by multiple key variants and shown in full
- **Previous Vera messages** are shown to prevent repetition
- **Language preference** from both merchant identity and customer identity is resolved

### Tick Processing

Concurrent processing with a semaphore of 3 parallel Groq calls — stays well under the 30s hard limit while maximizing throughput. No artificial sleep delays.

### Intent Routing in `/v1/reply`

| Intent | Signal | Action |
|---|---|---|
| `auto_reply` | Canned phrases or repeated verbatim message | `end` immediately |
| `not_interested` | "stop", "nahi chahiye", etc. | `end` respectfully |
| `join_intent` | "judrna", "join karna", "sign up" | Switch to ACTION mode — give first concrete onboarding step |
| `accept` | "yes", "ok", "chalega" | Confirm and execute — do NOT re-pitch |
| `question` | Contains `?` | Answer with specific data |
| `slot_selection` | Customer picks a slot label | Confirm booking immediately |

### Suppression

All fired `suppression_key` values are tracked in a set. `/v1/tick` skips any trigger whose key has already fired, preventing duplicate sends across ticks.

---

## Scoring Dimensions

| Dimension | Strategy |
|---|---|
| **Specificity (0-10)** | 2+ numbers from trigger payload in every message |
| **Category Fit (0-10)** | Voice tone + taboo words from category context injected into prompt |
| **Merchant Fit (0-10)** | Owner name, locality, CTR vs peer, active signals all used |
| **Decision Quality (0-10)** | Trigger kind, urgency, and payload drive the CTA and urgency framing |
| **Engagement (0-10)** | Loss aversion hooks, single binary CTA, effort externalization |

---

## Files

| File | Purpose |
|---|---|
| `bot.py` | Main FastAPI HTTP server — all 5 required endpoints (v2) |
| `conversation_handlers.py` | Multi-turn handler state machine (reference module) |
| `generate_submission.py` | Script to regenerate `submission.jsonl` from dataset |
| `submission.jsonl` | 30 pre-composed messages for the canonical test pairs |
| `README.md` | This file |

## Run

```bash
pip install fastapi uvicorn httpx python-dotenv
export GROQ_API_KEY=gsk_your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Then test locally:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Deploy to any public URL (Render, Railway, Fly.io, ngrok) and submit the URL.

## What Additional Context Would Have Helped Most

1. **Real multi-turn conversation history** — seed data has 1-2 turns; real Vera has 4.7 avg turns/merchant/day.
2. **Customer opt-in timestamps** — knowing recency of opt-in helps calibrate recall urgency.
3. **GBP post history** — knowing what was posted (not just "stale_posts:22d") enables writing a genuinely different post.
