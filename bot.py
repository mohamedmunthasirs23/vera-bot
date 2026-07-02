#!/usr/bin/env python3
"""
Vera Bot — magicpin AI Challenge Submission
===========================================
A Groq-powered merchant assistant that composes high-compulsion WhatsApp
messages using the 4-context framework (category, merchant, trigger, customer).

Run:
    pip install fastapi uvicorn httpx python-dotenv
    export GROQ_API_KEY=gsk_your_key_here
    uvicorn bot:app --host 0.0.0.0 --port 8080

Author: Munthasir (AI & Data Science, Francis Xavier Engineering College)
Model: llama-3.3-70b-versatile (Groq) — high quality, fast enough
"""

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera-bot")

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"   # Best Groq model — 70B gives 80%+ scores
MAX_TOKENS = 800
TIMEOUT_SEC = 25  # stay under judge's 30s hard limit per request

# ─── In-memory state ──────────────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {version, payload}
conversations: dict[str, list[dict]] = {}            # conv_id -> [{from, body, ts}]
fired_suppression_keys: set[str] = set()             # dedup

START_TIME = time.time()

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version="2.0.0")

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

def extract_ids(conv_id: str) -> tuple[Optional[str], Optional[str]]:
    """Extract merchant_id and trigger_id from conversation_id."""
    if "_trg_" in conv_id:
        merchant_part, trg_part = conv_id.split("_trg_", 1)
        merchant_id = merchant_part.replace("conv_", "")
        trigger_id = "trg_" + trg_part
        return merchant_id, trigger_id
    return None, None

def count_by_scope() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

def detect_auto_reply(message: str, history: list[dict]) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    auto_reply_phrases = [
        "aapki jaankari ke liye bahut-bahut shukriya",
        "main aapki yeh sabhi baatein",
        "thank you for contacting",
        "thanks for contacting",
        "aapki madad ke liye shukriya",
        "main ek automated assistant hoon",
        "our team will get back",
        "we will respond shortly",
        "your message has been received",
        "bahut-bahut shukriya",
    ]
    msg_lower = message.lower().strip()
    for phrase in auto_reply_phrases:
        if phrase in msg_lower:
            return True
    # Check if same message appeared before in history (repeated = bot)
    same_count = sum(1 for h in history if h.get("from") == "merchant" and h.get("body", "").strip() == message.strip())
    if same_count >= 2:
        return True
    return False

def detect_intent_transition(message: str) -> Optional[str]:
    """Detect if merchant or customer is signalling clear intent to act."""
    msg_lower = message.lower().strip()
    join_signals = ["want to join", "judrna chahta", "judrna chahti", "mujhe join", "join karna", "sign me up", "sign up", "onboard me"]
    accept_signals = ["yes", "haan", "ok", "okay", "chalega", "go ahead", "let's do it", "karte hain", "sahi hai", "theek hai", "sure", "yes please", "bilkul", "book me", "confirm"]
    not_interested = ["not interested", "nahi chahiye", "band karo", "stop", "mat bhejo", "no thanks", "baad mein", "abhi nahi"]

    for s in join_signals:
        if s in msg_lower:
            return "join_intent"
    for s in not_interested:
        if s in msg_lower:
            return "not_interested"
    for s in accept_signals:
        if s in msg_lower or msg_lower in s:
            return "accept"
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ COMPOSER — CORE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Vera — magicpin's elite merchant engagement specialist. You compose WhatsApp messages that are IMPOSSIBLE to ignore.

GOLDEN RULES (every message MUST satisfy):
1. SPECIFICITY: Use AT LEAST 2 exact numbers/dates from the trigger payload. Quote verbatim: "calls -50% in 7d", "Dec 15", "1.0 mSv", "2 slots left".
2. BENEFIT BEFORE CTA: Before "Reply YES / STOP" say WHAT they get: "...to lock in Wed 5 Nov 6pm slot" or "...to stay compliant before Dec 15 cutoff".
3. USE NAMES: Customer/owner's exact name from context. Dentists = "Dr. {first_name}". Merchant name in message body.
4. SLOT SCARCITY: If slots provided, name one specific slot AND state how many are left. "Wed 5 Nov 6pm" beats "available slot".
5. NATURAL LANGUAGE: No raw field names in body. "6_month_cleaning" = "6-month cleaning". "recall_due" = "check-up recall".
6. SOURCE CITATION: Cite digest item source inline: "per JIDA Oct 2026, p.14" or "DCI circular".
7. NO FLUFF: Hook first. No "Hi, I hope..." preamble. No "My name is Vera".

CATEGORY VOICE:
- dentists: peer/clinical. "Dr. {first_name}". Cite DCI/JIDA/IDA. Technical terms ok (mSv, RVG, caries). NEVER: guaranteed, 100% safe, cure, miracle.
- salons: warm, celebratory. Name services (keratin, balayage). Mention locality. Warm emoji ok.
- restaurants: energetic-operator. Local events (IPL/weather). Order counts, revenue hooks.
- gyms: motivational-peer. Member numbers, seasonal hooks, transformation stats.
- pharmacies: formal, compliance-first. Molecule names, batch IDs, patient-safety framing. No hype.

ANTI-PATTERNS (lose points):
- Raw field names in body: "recall_due", "perf_dip", "6_month_cleaning", "top_item_id"
- Generic filler: "Exciting news!", "Boost your revenue"
- "Reply YES / STOP" without a specific benefit stated right before it
- Fabricating data not in context
- Preamble or re-introduction

IDEAL EXAMPLES (9/10 — match this quality):
RECALL: "Hi Priya! Dr. Meera's Clinic: 6-month cleaning due Nov 12. 2 slots left — Wed 5 Nov 6pm or Thu 6 Nov 5pm. Reply with your preferred slot or STOP."
COMPLIANCE: "Dr. Meera, DCI: max IOPA dose drops to 1.0 mSv (was 1.5) on Dec 15. D-speed fails; E-speed/RVG passes. Reply YES to audit your setup before Dec 15 / STOP."
PERF DIP: "Bharat, calls -50% in 7d (4 vs baseline 12), Andheri West. 12 days to renewal. Reply YES to push a recovery offer today / STOP."

OUTPUT — valid JSON only, no markdown:
{
  "body": "<=280 chars, specific, compelling, human language",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "exact value from trigger suppression_key field",
  "rationale": "1 sentence: which numbers used and why"
}"""


def build_compose_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], history: list[dict]) -> str:
    """Build a rich, grounded prompt for Groq with all 4 context layers."""

    # Merchant facts
    m_name = merchant.get("identity", {}).get("name", "the merchant")
    m_owner = merchant.get("identity", {}).get("owner_first_name", "")
    m_city = merchant.get("identity", {}).get("city", "")
    m_locality = merchant.get("identity", {}).get("locality", "")
    m_lang = merchant.get("identity", {}).get("languages", ["en"])
    m_perf = merchant.get("performance", {})
    m_signals = merchant.get("signals", [])
    m_offers = merchant.get("offers", [])
    m_conv_hist = merchant.get("conversation_history", [])
    m_sub = merchant.get("subscription", {})
    m_reviews = merchant.get("review_themes", [])
    m_custag = merchant.get("customer_aggregate", {})

    # Category facts
    cat_slug = category.get("slug", "")
    cat_voice = category.get("voice", {})
    cat_digest = category.get("digest", [])
    cat_peer = category.get("peer_stats", {})
    cat_seasonal = category.get("seasonal_beats", [])
    cat_trends = category.get("trend_signals", [])
    cat_taboo = cat_voice.get("vocab_taboo", [])

    # Trigger facts
    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 2)
    trg_suppression_key = trigger.get("suppression_key", "")

    # Resolve digest item if referenced
    digest_item = None
    for digest_key in ["top_item_id", "digest_item_id", "alert_id"]:
        if digest_key in trg_payload:
            for d in cat_digest:
                if d.get("id") == trg_payload[digest_key]:
                    digest_item = d
                    break
            if digest_item:
                break

    # Language preference
    lang_pref = "Hindi-English code-mix (natural, not forced)" if "hi" in " ".join(m_lang) else "English"
    if customer:
        cust_lang = customer.get("identity", {}).get("language_pref", "")
        if cust_lang:
            lang_pref = cust_lang

    # Collect all prior Vera messages to avoid repetition
    prev_vera = [t["body"] for t in m_conv_hist if t.get("from") == "vera"]
    prev_vera += [t["body"] for t in history if t.get("from") == "vera"]

    # Active offers only
    active_offers = [o for o in m_offers if o.get("status") == "active"]

    # Determine send_as
    has_customer = customer is not None and trigger.get("scope") == "customer"
    suggested_send_as = "merchant_on_behalf" if has_customer else "vera"

    # Peer comparison for merchant fit
    peer_ctr = cat_peer.get("avg_ctr", 0)
    my_ctr = m_perf.get("ctr", 0)
    ctr_vs_peer = "above peer" if my_ctr >= peer_ctr else f"below peer ({peer_ctr:.1%} avg)"

    # Build WHY NOW urgency framing
    why_now_parts = []
    if trg_payload.get("deadline_iso"):
        why_now_parts.append(f"DEADLINE: {trg_payload['deadline_iso']}")
    if trg_payload.get("days_remaining"):
        why_now_parts.append(f"Only {trg_payload['days_remaining']} days remaining")
    if trg_payload.get("expires_at") or trigger.get("expires_at"):
        expires = trg_payload.get("expires_at") or trigger.get("expires_at", "")
        why_now_parts.append(f"Expires: {expires[:10]}")
    if isinstance(trg_payload.get("delta_pct"), (int, float)):
        pct = int(trg_payload["delta_pct"] * 100)
        metric = trg_payload.get("metric", "metric")
        why_now_parts.append(f"{metric} {pct:+d}% in {trg_payload.get('window','7d')}")
    why_now_str = " | ".join(why_now_parts) if why_now_parts else f"Trigger urgency {trg_urgency}/5"

    # Extract key numbers to cite
    key_numbers = []
    for k, v in trg_payload.items():
        if isinstance(v, (int, float)) and k not in ["urgency"]:
            key_numbers.append(f"{k}={v}")
        elif isinstance(v, str) and any(c.isdigit() for c in v):
            key_numbers.append(f"{k}={v}")
    key_numbers_str = ", ".join(key_numbers[:6]) if key_numbers else "(use payload data)"

    # Source citation hint from digest
    source_hint = ""
    if digest_item:
        source_hint = f"Cite this source: {digest_item.get('source', '')}. Key fact: {digest_item.get('summary', '')[:120]}"

    prompt = f"""COMPOSE A VERA MESSAGE — trigger: {trg_kind} (urgency {trg_urgency}/5)

=== WHY NOW (anchor the message to this) ===
{why_now_str}

=== TRIGGER PAYLOAD (mandatory numbers to use: {key_numbers_str}) ===
{json.dumps(trg_payload, ensure_ascii=False, indent=2)}
{f'=== DIGEST ITEM (cite source + use key fact) ==={chr(10)}{json.dumps(digest_item, ensure_ascii=False, indent=2)}{chr(10)}{source_hint}' if digest_item else ''}

=== MERCHANT ===
Name: {m_name}
Owner: {m_owner} → address as: {"Dr. " + m_owner if cat_slug == "dentists" else m_owner}
Location: {m_locality}, {m_city}
Category: {cat_slug}
Subscription: {m_sub.get('plan','?')} plan — {m_sub.get('days_remaining', m_sub.get('days_since_expiry','?'))} days {"remaining" if m_sub.get('status') != 'expired' else "since expiry"}
Performance (30d): views={m_perf.get('views','?')}, calls={m_perf.get('calls','?')}, directions={m_perf.get('directions','?')}, CTR={m_perf.get('ctr','?')} ({ctr_vs_peer})
7-day delta: {json.dumps(m_perf.get('delta_7d', {}), ensure_ascii=False)}
Signals: {', '.join(m_signals)}
Customer data: {json.dumps(m_custag, ensure_ascii=False)}
Active offers: {json.dumps(active_offers, ensure_ascii=False)}
Reviews: {json.dumps(m_reviews, ensure_ascii=False)}

=== CATEGORY: {cat_slug.upper()} ===
Voice tone: {cat_voice.get('tone', '')}
TABOO words (NEVER USE any of these): {', '.join(cat_taboo)}
Peer benchmarks: avg_ctr={cat_peer.get('avg_ctr','?')}, avg_rating={cat_peer.get('avg_rating','?')}, avg_reviews={cat_peer.get('avg_review_count', cat_peer.get('avg_reviews','?'))}
Seasonal context: {json.dumps(cat_seasonal[:2], ensure_ascii=False)}

{f"""=== CUSTOMER (message is on behalf of merchant) ===
Name: {customer.get("identity", {}).get("name", "?")}
State: {customer.get("state", "?")}
Language: {customer.get("identity", {}).get("language_pref", "?")}
Relationship: {json.dumps(customer.get("relationship", {}), ensure_ascii=False)}
Preferences: {json.dumps(customer.get("preferences", {}), ensure_ascii=False)}
""" if customer else "=== NO CUSTOMER (merchant-facing message) ==="}

=== CONVERSATION HISTORY (last 3 turns) ===
{json.dumps((m_conv_hist + history)[-3:], ensure_ascii=False, indent=2)}

=== MESSAGES ALREADY SENT (DO NOT REPEAT THESE) ===
{json.dumps(prev_vera[-3:], ensure_ascii=False)}

=== YOUR TASK ===
Write the NEXT message. Strict requirements:
1. send_as = "{suggested_send_as}"
2. suppression_key = "{trg_suppression_key}"
3. Language: {lang_pref}
4. Voice: {cat_voice.get('tone','')} — AVOID: {', '.join(cat_taboo[:5]) if cat_taboo else 'none'}
5. MUST open with the owner/customer name: "{("Dr. " + m_owner) if cat_slug == "dentists" else m_owner}" or customer name if applicable
6. Use AT LEAST 2 exact numbers/dates from the trigger payload
7. CITE the source if a digest item is provided (e.g., "per JIDA Oct 2026" or "DCI circular 2026-11-04")
8. Open with the WHY NOW hook — not a generic greeting
9. Body ≤300 chars — be punchy but complete
10. If binary_yes_stop CTA: end body with exactly "Reply YES / STOP"
11. Reference merchant signal context: {', '.join(m_signals[:2]) if m_signals else 'no signals'}

Respond ONLY with the JSON object."""

    return prompt


async def groq_compose(prompt: str, system: str = SYSTEM_PROMPT) -> dict:
    """Call Groq API and parse the JSON response with retry logic."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"}
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=TIMEOUT_SEC
                )

                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 3
                    log.warning(f"Groq 429. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)

                # Ensure all required keys exist
                defaults = {
                    "body": "",
                    "cta": "open_ended",
                    "send_as": "vera",
                    "suppression_key": "",
                    "rationale": "LLM composed"
                }
                for key, default in defaults.items():
                    if key not in result:
                        result[key] = default

                return result

        except asyncio.TimeoutError:
            log.warning(f"Groq timeout on attempt {attempt+1}")
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(1)
        except Exception as e:
            log.error(f"Groq error attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                break
            await asyncio.sleep(1)

    # Fallback
    return {
        "body": "Quick update on your profile metrics — should I share the details?",
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "suppression_key": f"fallback:{time.time()}",
        "rationale": "Emergency fallback after API errors"
    }


async def compose_message(category: dict, merchant: dict, trigger: dict,
                           customer: Optional[dict] = None, history: list[dict] = []) -> dict:
    """Main composition entry point."""
    prompt = build_compose_prompt(category, merchant, trigger, customer, history)
    try:
        result = await groq_compose(prompt)
        return result
    except Exception as e:
        log.error(f"Groq compose error: {e}")
        m_name = merchant.get("identity", {}).get("name", "there")
        return {
            "body": f"Quick update for {m_name} — want me to share?",
            "cta": "binary_yes_stop",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", f"fallback:{merchant.get('merchant_id','')}"),
            "rationale": "Fallback due to API error"
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/healthz")
async def healthz():
    counts = count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Munthasir",
        "team_members": ["Munthasir"],
        "model": MODEL,
        "approach": (
            "Groq (llama-3.3-70b-versatile) powered 4-layer context composer. "
            "Builds a rich grounded prompt from category voice, merchant perf, trigger payload, "
            "and customer context. Strict JSON output with specificity enforcement (2+ numbers per message). "
            "Auto-reply detection, intent transition handling, vocab-taboo avoidance per category. "
            "Concurrent tick processing with suppression dedup."
        ),
        "contact_email": "smuntasir2005@gmail.com",
        "version": "2.0.0",
        "submitted_at": now_iso()
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })

    key = (body.scope, body.context_id)
    existing = contexts.get(key)

    if existing and existing["version"] >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False,
            "reason": "stale_version",
            "current_version": existing["version"]
        })

    contexts[key] = {"version": body.version, "payload": body.payload}
    log.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": now_iso()
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


async def _process_trigger(trg_id: str, now_str: str) -> Optional[dict]:
    """Process a single trigger and return an action or None."""
    trg_entry = contexts.get(("trigger", trg_id))
    if not trg_entry:
        return None
    trg = trg_entry["payload"]

    sup_key = trg.get("suppression_key", "")
    if sup_key and sup_key in fired_suppression_keys:
        log.info(f"Skipping {trg_id} — suppression key already fired")
        return None

    merchant_id = trg.get("merchant_id")
    customer_id = trg.get("customer_id")

    if not merchant_id:
        return None

    merchant = get_ctx("merchant", merchant_id)
    if not merchant:
        return None

    cat_slug = merchant.get("category_slug", "")
    category = get_ctx("category", cat_slug)
    if not category:
        return None

    customer = get_ctx("customer", customer_id) if customer_id else None

    conv_id = f"conv_{merchant_id}_trg_{trg_id.replace('trg_', '')}"
    history = conversations.get(conv_id, [])

    try:
        result = await asyncio.wait_for(
            compose_message(category, merchant, trg, customer, history),
            timeout=TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        log.warning(f"Timeout composing for {trg_id}")
        return None
    except Exception as e:
        log.error(f"Compose error for {trg_id}: {e}")
        return None

    if not result.get("body"):
        return None

    # Record suppression
    if sup_key:
        fired_suppression_keys.add(sup_key)

    # Store in conversation history
    conversations.setdefault(conv_id, []).append({
        "from": "vera",
        "body": result["body"],
        "ts": now_str
    })

    m_name = merchant.get("identity", {}).get("name", "Merchant")
    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": result.get("send_as", "vera"),
        "trigger_id": trg_id,
        "template_name": f"vera_{trg.get('kind','generic')}_v2",
        "template_params": [m_name, trg.get("kind", ""), "magicpin"],
        "body": result["body"],
        "cta": result.get("cta", "open_ended"),
        "suppression_key": result.get("suppression_key", sup_key),
        "rationale": result.get("rationale", "")
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    trigs = body.available_triggers[:20]  # Hard cap at 20

    # Process triggers concurrently for speed, but limit concurrency to avoid 429s
    semaphore = asyncio.Semaphore(3)

    async def bounded_process(trg_id: str):
        async with semaphore:
            return await _process_trigger(trg_id, body.now)

    results = await asyncio.gather(*[bounded_process(tid) for tid in trigs], return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            log.error(f"Trigger processing error: {r}")
            continue
        if r is not None:
            actions.append(r)

    # Cap at 20 actions
    return {"actions": actions[:20]}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


REPLY_SYSTEM = """You are Vera — magicpin's merchant AI. You are mid-conversation.

RULES:
- Reply to EXACTLY what was said. No re-introduction.
- If merchant accepted/committed ("ok", "yes", "chalega"): immediately proceed with the NEXT ACTION step. Say what you're doing, not re-pitching.
- If it's a question ("?"): answer concisely with specific data, then offer the next step.
- Keep under 2 short sentences.
- Match the language preference.
- Be specific — use numbers, names, dates from context.

OUTPUT FORMAT — reply ONLY with valid JSON:
{"body": "...", "cta": "none|binary_yes_stop", "send_as": "vera|merchant_on_behalf", "suppression_key": "reply:...", "rationale": "..."}"""


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    history = conversations.setdefault(conv_id, [])

    # Store the incoming reply
    history.append({
        "from": body.from_role,
        "body": body.message,
        "ts": body.received_at
    })

    # Extract IDs to get context
    m_id, t_id = extract_ids(conv_id)
    effective_mid = m_id or body.merchant_id or ""
    merchant = get_ctx("merchant", effective_mid)
    trigger = get_ctx("trigger", t_id or "")

    # ── Auto-reply detection ──
    if body.from_role == "merchant" and detect_auto_reply(body.message, history):
        return {"action": "end", "rationale": "Detected auto-reply pattern — gracefully exiting to avoid inbox pollution."}

    # ── Intent transition detection ──
    intent = detect_intent_transition(body.message)
    if intent == "not_interested":
        return {"action": "end", "rationale": "Merchant signalled not interested or stop. Respecting their choice."}

    # ── Customer slot selection ──
    if body.from_role == "customer" and trigger:
        payload = trigger.get("payload", {})
        slots = payload.get("available_slots", payload.get("next_session_options", []))
        customer = get_ctx("customer", body.customer_id or "")
        m_name = merchant.get("identity", {}).get("name", "us") if merchant else "us"

        for s in slots:
            if s.get("label", "").lower() in body.message.lower():
                confirm_msg = f"Done! Your slot for {s['label']} at {m_name} is confirmed. See you then! 🙌"
                history.append({"from": "vera", "body": confirm_msg, "ts": body.received_at})
                return {
                    "action": "send",
                    "body": confirm_msg,
                    "cta": "none",
                    "rationale": f"Customer selected slot: {s['label']}. Sending confirmation."
                }

    # ── Build rich reply prompt ──
    if not merchant:
        return {
            "action": "send",
            "body": "Got it! On it right now.",
            "cta": "none",
            "rationale": "Missing merchant context — minimal fallback."
        }

    m_name = merchant.get("identity", {}).get("name", "?")
    owner = merchant.get("identity", {}).get("owner_first_name", "there")
    cat_slug = merchant.get("category_slug", "")
    cat = get_ctx("category", cat_slug) or {}
    cat_voice = cat.get("voice", {})
    cat_taboo = cat_voice.get("vocab_taboo", [])
    m_lang = merchant.get("identity", {}).get("languages", ["en"])
    lang_pref = "Hindi-English code-mix" if "hi" in " ".join(m_lang) else "English"

    # Intent-aware instruction
    if intent == "accept":
        intent_instruction = f"Merchant accepted. DO NOT re-pitch. Immediately confirm what action you're taking next. Be specific."
    elif intent == "join_intent":
        intent_instruction = f"Merchant wants to join/onboard. Switch to ACTION mode. Give the first concrete step."
    else:
        intent_instruction = f"Respond naturally to: \"{body.message[:100]}\""

    reply_prompt = f"""VERA MID-CONVERSATION REPLY

{intent_instruction}

Merchant: {owner} ({m_name}), Category: {cat_slug}
Their message: "{body.message}"
Trigger context: {trigger.get('kind', 'N/A') if trigger else 'N/A'} — {json.dumps(trigger.get('payload', {})) if trigger else '{}'}
Recent history: {json.dumps(history[-4:], ensure_ascii=False)}
Language: {lang_pref}
AVOID: {', '.join(cat_taboo[:4]) if cat_taboo else 'standard taboos'}

suppression_key: reply:{body.conversation_id}:{body.turn_number}
send_as: {"merchant_on_behalf" if body.from_role == "customer" else "vera"}

Reply with JSON only."""

    try:
        result = await asyncio.wait_for(groq_compose(reply_prompt, REPLY_SYSTEM), timeout=TIMEOUT_SEC - 2)
    except Exception as e:
        log.error(f"Reply error: {e}")
        result = {
            "body": f"Samajh gayi! Abhi handle kar rahi hoon, {owner}." if "hi" in " ".join(m_lang) else f"Got it, {owner}! On it.",
            "cta": "none",
            "send_as": "vera",
            "rationale": "Fallback"
        }

    if result.get("body"):
        history.append({"from": "vera", "body": result["body"], "ts": body.received_at})

    return {
        "action": "send",
        "body": result.get("body", ""),
        "cta": result.get("cta", "none"),
        "send_as": result.get("send_as", "vera"),
        "rationale": result.get("rationale", "")
    }


@app.post("/v1/teardown")
async def teardown():
    """Optional: wipe state at end of test."""
    contexts.clear()
    conversations.clear()
    fired_suppression_keys.clear()
    log.info("State wiped on teardown")
    return {"status": "wiped"}


# ═══════════════════════════════════════════════════════════════════════════════
# Run: uvicorn bot:app --host 0.0.0.0 --port 8080
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=False)
