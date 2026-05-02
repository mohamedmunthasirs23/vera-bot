#!/usr/bin/env python3
"""
Vera Bot — magicpin AI Challenge Submission
===========================================
A Groq-powered merchant assistant that composes high-compulsion WhatsApp
messages using the 4-context framework (category, merchant, trigger, customer).

Run:
    pip install fastapi uvicorn httpx
    export GROQ_API_KEY=gsk_your_key_here
    uvicorn bot:app --host 0.0.0.0 --port 8080

Author: Munthasir (AI & Data Science, Francis Xavier Engineering College)
Model: llama-3.3-70b-versatile (Groq)
"""

import os
import time
import json
import uuid
import httpx
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera-bot")

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000
TIMEOUT_SEC = 28  # stay under judge's 30s hard limit

# ─── In-memory state ──────────────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {version, payload}
conversations: dict[str, list[dict]] = {}            # conv_id -> [{from, body, ts}]
fired_suppression_keys: set[str] = set()             # dedup

START_TIME = time.time()

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Vera Bot", version="1.0.0")

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

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
    # Check keyword match
    for phrase in auto_reply_phrases:
        if phrase in msg_lower:
            return True
    # Check if same message appeared before in history
    same_count = sum(1 for h in history if h.get("from") == "merchant" and h.get("body", "").strip() == message.strip())
    if same_count >= 2:
        return True
    return False

def detect_intent_transition(message: str) -> Optional[str]:
    """Detect if merchant is signalling clear intent to act."""
    msg_lower = message.lower().strip()
    join_signals = ["want to join", "judrna chahta", "judrna chahti", "mujhe join", "join karna", "sign me up", "sign up", "onboard me"]
    accept_signals = ["yes", "haan", "ok", "okay", "chalega", "go ahead", "let's do it", "karte hain", "sahi hai", "theek hai", "sure", "yes please", "bilkul"]
    not_interested = ["not interested", "nahi chahiye", "band karo", "stop", "mat bhejo", "no thanks", "baad mein", "abhi nahi"]
    
    for s in join_signals:
        if s in msg_lower:
            return "join_intent"
    for s in accept_signals:
        if msg_lower in s or s in msg_lower:
            return "accept"
    for s in not_interested:
        if s in msg_lower:
            return "not_interested"
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ COMPOSER — CORE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Vera — magicpin's merchant AI assistant. You compose WhatsApp messages for Indian merchants.

CORE MISSION: Every message must make a real merchant want to reply. Use these levers:
1. Specificity — concrete numbers, dates, source citations (not generic "increase sales")
2. Loss aversion — "you're missing X", "before this closes"
3. Social proof — "3 dentists in your locality did Y this month"
4. Effort externalization — "I've drafted it — just say go"
5. Curiosity — "want to see who?", "want the full list?"
6. Single binary CTA — Reply YES / STOP (never multiple choice except booking slots)

VOICE RULES by category:
- dentists: peer/clinical tone, cite sources (JIDA, DCI), technical vocab OK, NEVER "guaranteed/cure/100% safe"
- salons: warm, trend-aware, celebrate wins, local specificity
- restaurants: energetic but not loud, local moment hooks (IPL, weather), food specifics
- gyms: motivational-peer, transformation-proof, seasonal hooks
- pharmacies: informational, compliance-forward, patient-safety first

ANTI-PATTERNS (judge penalizes these):
- Long preamble ("I hope you're doing well...")  
- Re-introducing yourself after turn 1
- Multiple CTAs in one message
- Generic "Flat 30% off" when service+price is available
- Hallucinating data not in the context
- Promotional tone for clinical categories
- Sending same message as before in same conversation

OUTPUT FORMAT: Reply ONLY with valid JSON. No preamble, no markdown.
{
  "body": "the WhatsApp message text",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "string",
  "rationale": "1-2 sentence explanation of why this message, what compulsion lever used"
}

For binary_yes_stop CTAs, the message body must end with "Reply YES / STOP"
For merchant_on_behalf: message comes from the merchant's WhatsApp, not Vera's
Keep body concise — WhatsApp friendly. Hindi-English code-mix encouraged where language_pref says hi or hi-en mix."""

def build_compose_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], history: list[dict]) -> str:
    """Build a rich, grounded prompt for Groq."""
    
    # Pull key merchant facts
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
    
    # Pull category facts
    cat_slug = category.get("slug", "")
    cat_voice = category.get("voice", {})
    cat_digest = category.get("digest", [])
    cat_offers = category.get("offer_catalog", [])
    cat_peer = category.get("peer_stats", {})
    cat_seasonal = category.get("seasonal_beats", [])
    cat_trends = category.get("trend_signals", [])
    
    # Pull trigger facts
    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 2)
    trg_scope = trigger.get("scope", "merchant")
    
    # Find relevant digest item if trigger references one
    digest_item = None
    if "top_item_id" in trg_payload:
        for d in cat_digest:
            if d.get("id") == trg_payload["top_item_id"]:
                digest_item = d
                break
    
    # Language preference
    lang_pref = "hi-en mix" if "hi" in " ".join(m_lang) else "english"
    if customer:
        lang_pref = customer.get("identity", {}).get("language_pref", lang_pref)
    
    # Previous bot messages for anti-repetition check
    prev_vera_messages = [t["body"] for t in m_conv_hist if t.get("from") == "vera"]
    prev_vera_messages += [t["body"] for t in history if t.get("from") == "vera"]
    
    # Active offers
    active_offers = [o for o in m_offers if o.get("status") == "active"]
    
    prompt = f"""COMPOSE A VERA MESSAGE

=== TRIGGER (WHY NOW) ===
kind: {trg_kind}
urgency: {trg_urgency}/5
scope: {trg_scope}
payload: {json.dumps(trg_payload, ensure_ascii=False)}
{f'digest_item: {json.dumps(digest_item, ensure_ascii=False)}' if digest_item else ''}

=== MERCHANT CONTEXT ===
name: {m_name}
owner: {m_owner}
location: {m_locality}, {m_city}
category: {cat_slug}
language_pref: {lang_pref}
subscription: {m_sub.get('plan','?')} plan, {m_sub.get('days_remaining','?')} days remaining, status={m_sub.get('status','?')}
performance_30d: views={m_perf.get('views','?')}, calls={m_perf.get('calls','?')}, directions={m_perf.get('directions','?')}, ctr={m_perf.get('ctr','?')}
delta_7d: {json.dumps(m_perf.get('delta_7d', {}), ensure_ascii=False)}
signals: {', '.join(m_signals)}
active_offers: {json.dumps(active_offers, ensure_ascii=False)}
customer_aggregate: {json.dumps(m_custag, ensure_ascii=False)}
review_themes: {json.dumps(m_reviews, ensure_ascii=False)}

=== CATEGORY CONTEXT ===
voice_tone: {cat_voice.get('tone','')}
vocab_taboo: {', '.join(cat_voice.get('vocab_taboo', cat_voice.get('vocab_taboo', [])))}
peer_stats: avg_ctr={cat_peer.get('avg_ctr','?')}, avg_rating={cat_peer.get('avg_rating','?')}, avg_reviews={cat_peer.get('avg_review_count', cat_peer.get('avg_reviews','?'))}
seasonal_beats: {json.dumps(cat_seasonal, ensure_ascii=False)}
trend_signals: {json.dumps(cat_trends, ensure_ascii=False)}
catalog_offers: {json.dumps(cat_offers[:4], ensure_ascii=False)}

{f'''=== CUSTOMER CONTEXT ===
name: {customer.get("identity", {}).get("name", "?")}
state: {customer.get("state", "?")}
language_pref: {customer.get("identity", {}).get("language_pref", "?")}
relationship: {json.dumps(customer.get("relationship", {}), ensure_ascii=False)}
preferences: {json.dumps(customer.get("preferences", {}), ensure_ascii=False)}
consent_scope: {", ".join(customer.get("consent", {}).get("scope", []))}
trigger_payload: {json.dumps(trg_payload, ensure_ascii=False)}
''' if customer else '(no customer context — this is a merchant-facing message)'}

=== CONVERSATION HISTORY (recent turns) ===
{json.dumps(m_conv_hist[-3:] + history[-2:], ensure_ascii=False, indent=2) if (m_conv_hist or history) else 'No prior conversation'}

IMPORTANT — these messages were already sent to this merchant. DO NOT repeat them:
{json.dumps(prev_vera_messages[-3:], ensure_ascii=False) if prev_vera_messages else '[]'}

=== YOUR TASK ===
Compose the next message. 
- send_as = "merchant_on_behalf" ONLY if customer context is present AND the trigger targets a customer
- send_as = "vera" for all merchant-facing messages
- suppression_key = use the trigger's suppression_key if available: {trigger.get('suppression_key', '')}
- anchor on at least ONE concrete fact from the context (number, date, stat, source)
- match language_pref: {lang_pref}
- use the right voice for category: {cat_slug} ({cat_voice.get('tone','')})
- choose the right compulsion lever for this trigger kind

Respond ONLY with the JSON object. No extra text."""
    
    return prompt

async def groq_compose(prompt: str) -> dict:
    """Call Groq API (OpenAI-compatible) and parse the JSON response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
        resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    text = data["choices"][0]["message"]["content"].strip()
    # Strip any accidental markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    result = json.loads(text)
    # Ensure required keys
    for key in ["body", "cta", "send_as", "suppression_key", "rationale"]:
        if key not in result:
            result[key] = ""
    return result

async def compose_message(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None, history: list[dict] = []) -> dict:
    """Main composition entry point."""
    prompt = build_compose_prompt(category, merchant, trigger, customer, history)
    try:
        result = await groq_compose(prompt)
        return result
    except Exception as e:
        log.error(f"Groq compose error: {e}")
        # Fallback minimal response
        m_name = merchant.get("identity", {}).get("name", "there")
        return {
            "body": f"Hi {m_name}, quick update on your magicpin profile — want me to share?",
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
            "Groq (llama-3.3-70b-versatile) powered single-prompt composer with trigger-kind dispatch. "
            "Builds a rich grounded context prompt from all 4 layers, calls Groq with "
            "strict JSON output schema. Handles auto-reply detection, intent transitions, "
            "and graceful conversation exits. Anti-repetition via conversation history injection."
        ),
        "contact_email": "smuntasir2005@gmail.com",
        "version": "1.0.0",
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

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    for trg_id in body.available_triggers:
        # Skip already fired suppression keys
        trg_entry = contexts.get(("trigger", trg_id))
        if not trg_entry:
            continue
        trg = trg_entry["payload"]
        
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in fired_suppression_keys:
            continue
        
        # Check expiry
        expires_at = trg.get("expires_at", "")
        if expires_at and expires_at < body.now:
            continue
        
        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")
        
        if not merchant_id:
            continue
        
        merchant = get_ctx("merchant", merchant_id)
        if not merchant:
            continue
        
        cat_slug = merchant.get("category_slug", "")
        category = get_ctx("category", cat_slug)
        if not category:
            continue
        
        customer = get_ctx("customer", customer_id) if customer_id else None
        
        conv_id = f"conv_{merchant_id}_{trg_id}"
        history = conversations.get(conv_id, [])
        
        try:
            result = await asyncio.wait_for(
                compose_message(category, merchant, trg, customer, history),
                timeout=25
            )
        except asyncio.TimeoutError:
            log.warning(f"Timeout composing for {trg_id}")
            continue
        except Exception as e:
            log.error(f"Compose error for {trg_id}: {e}")
            continue
        
        if not result.get("body"):
            continue
        
        # Record suppression
        if sup_key:
            fired_suppression_keys.add(sup_key)
        
        # Store in conversation history
        conversations.setdefault(conv_id, []).append({
            "from": "vera",
            "body": result["body"],
            "ts": body.now
        })
        
        # Build template params (first 3 words of body as mock params)
        m_name = merchant.get("identity", {}).get("name", "Merchant")
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind','generic')}_v1",
            "template_params": [m_name, trg.get("kind", ""), "magicpin"],
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", "")
        })
        
        # Max 20 actions per tick
        if len(actions) >= 20:
            break
    
    return {"actions": actions}

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

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
    
    # ── Auto-reply detection ──
    if body.from_role == "merchant" and detect_auto_reply(body.message, history):
        # Try once more with a direct question, then exit on next auto-reply
        auto_reply_count = sum(
            1 for h in history
            if h.get("from") == "merchant" and detect_auto_reply(h.get("body", ""), [])
        )
        if auto_reply_count >= 2:
            # Graceful exit
            return {
                "action": "end",
                "rationale": "Detected WhatsApp Business auto-reply (2+ identical canned responses). Exiting gracefully to avoid spamming."
            }
        else:
            # One gentle redirect
            merchant = get_ctx("merchant", body.merchant_id) if body.merchant_id else None
            m_name = merchant.get("identity", {}).get("owner_first_name", "aap") if merchant else "aap"
            bypass_msg = f"Lagta hai yeh auto-reply hai 😊 Agar {m_name} khud dekh rahe hain — main 2 min mein batata/batati hoon kya update hua. Chalega?"
            history.append({"from": "vera", "body": bypass_msg, "ts": body.received_at})
            return {
                "action": "send",
                "body": bypass_msg,
                "cta": "binary_yes_stop",
                "rationale": "Detected likely auto-reply; sending one direct human-targeted message before exit."
            }
    
    # ── Intent transition detection ──
    intent = detect_intent_transition(body.message)
    
    if intent == "not_interested":
        return {
            "action": "end",
            "rationale": "Merchant signalled not interested. Gracefully exiting conversation."
        }
    
    if intent == "join_intent":
        merchant = get_ctx("merchant", body.merchant_id) if body.merchant_id else None
        m_name = merchant.get("identity", {}).get("owner_first_name", "aap") if merchant else "aap"
        join_msg = (
            f"Bilkul! {m_name} ko magicpin se join karna bahut easy hai — "
            f"sirf 3 steps mein: profile setup, offers add karo, aur pehle customers aane shuru. "
            f"Main abhi onboarding shuru kar sakti hoon. Reply YES to proceed."
        )
        history.append({"from": "vera", "body": join_msg, "ts": body.received_at})
        return {
            "action": "send",
            "body": join_msg,
            "cta": "binary_yes_stop",
            "rationale": "Merchant expressed explicit join intent. Immediately switching from pitch to onboarding action mode."
        }
    
    # ── Normal reply handling — compose contextually ──
    merchant = get_ctx("merchant", body.merchant_id) if body.merchant_id else None
    customer = get_ctx("customer", body.customer_id) if body.customer_id else None
    
    if not merchant:
        return {
            "action": "send",
            "body": "Got it! Main aapke liye abhi isko handle karti hoon. 2 minute mein update bhejti hoon.",
            "cta": "none",
            "rationale": "No merchant context available; sending generic acknowledgment."
        }
    
    cat_slug = merchant.get("category_slug", "")
    category = get_ctx("category", cat_slug) or {}
    
    # Build a reply prompt incorporating what was just said
    reply_prompt = f"""You are Vera mid-conversation. The merchant just replied:

MERCHANT SAID: "{body.message}"
TURN NUMBER: {body.turn_number}

CONVERSATION SO FAR:
{json.dumps(history[-5:], ensure_ascii=False, indent=2)}

MERCHANT: {merchant.get('identity', {}).get('name', '?')}
CATEGORY: {cat_slug}
ACTIVE OFFERS: {json.dumps([o for o in merchant.get('offers',[]) if o.get('status')=='active'], ensure_ascii=False)}
LANGUAGE PREF: {merchant.get('identity', {}).get('languages', ['en'])}

Compose the NEXT Vera reply. Rules:
- If merchant accepted/said YES: move forward, do the thing they agreed to, don't re-pitch
- If merchant asked a question: answer specifically from the context, no hallucination
- If merchant seems confused: clarify concisely
- Keep it short for WhatsApp — 2-3 sentences max
- No re-introduction
- Language match: {'Hindi-English mix preferred' if 'hi' in str(merchant.get('identity',{}).get('languages',[])) else 'English'}

Output JSON only:
{{"body": "...", "cta": "open_ended|binary_yes_stop|none", "send_as": "vera", "suppression_key": "reply:{body.conversation_id}:t{body.turn_number}", "rationale": "..."}}"""
    
    try:
        result = await asyncio.wait_for(groq_compose(reply_prompt), timeout=25)
    except Exception as e:
        log.error(f"Reply compose error: {e}")
        result = {
            "body": "Samajh gayi! Main isko abhi handle karti hoon aur 5 minute mein update bhejti hoon.",
            "cta": "none",
            "send_as": "vera",
            "suppression_key": f"reply:{conv_id}:t{body.turn_number}",
            "rationale": "Fallback acknowledgment"
        }
    
    if result.get("body"):
        history.append({"from": "vera", "body": result["body"], "ts": body.received_at})
    
    return {
        "action": "send",
        "body": result.get("body", ""),
        "cta": result.get("cta", "none"),
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
