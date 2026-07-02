#!/usr/bin/env python3
"""
generate_submission.py
======================
Generates submission.jsonl with composed messages for all 30 test pairs.
Uses the same Groq composer logic as bot.py.

Run:
    pip install httpx
    export GROQ_API_KEY=gsk_your_key_here
    python generate_submission.py
    
Output: submission.jsonl (30 lines)
"""

import os
import json
import asyncio
import httpx
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path("../expanded_dataset")  # adjust if needed
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000
TIMEOUT_SEC = 28

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
- Long preamble, re-introducing yourself, multiple CTAs, generic "Flat 30% off"
- Hallucinating data not in the context, promotional tone for clinical categories

OUTPUT FORMAT: Reply ONLY with valid JSON. No preamble, no markdown.
{
  "body": "the WhatsApp message text",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "string",
  "rationale": "1-2 sentence explanation"
}"""

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def find_file(directory: Path, prefix: str) -> Path | None:
    for f in directory.glob(f"{prefix}*.json"):
        return f
    return None

def build_prompt(category, merchant, trigger, customer=None):
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
    
    cat_slug = category.get("slug", "")
    cat_voice = category.get("voice", {})
    cat_digest = category.get("digest", [])
    cat_offers = category.get("offer_catalog", [])
    cat_peer = category.get("peer_stats", {})
    cat_seasonal = category.get("seasonal_beats", [])
    cat_trends = category.get("trend_signals", [])
    
    trg_kind = trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})
    trg_urgency = trigger.get("urgency", 2)
    trg_scope = trigger.get("scope", "merchant")
    
    digest_item = None
    if "top_item_id" in trg_payload:
        for d in cat_digest:
            if d.get("id") == trg_payload["top_item_id"]:
                digest_item = d
                break
    
    lang_pref = "hi-en mix" if "hi" in " ".join(m_lang) else "english"
    if customer:
        lang_pref = customer.get("identity", {}).get("language_pref", lang_pref)
    
    active_offers = [o for o in m_offers if o.get("status") == "active"]
    
    return f"""COMPOSE A VERA MESSAGE

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

=== CONVERSATION HISTORY ===
{json.dumps(m_conv_hist[-2:], ensure_ascii=False, indent=2) if m_conv_hist else 'No prior conversation'}

IMPORTANT RULES:
- send_as = "merchant_on_behalf" ONLY if customer context present AND trigger targets a customer
- send_as = "vera" for all merchant-facing messages
- suppression_key = use trigger's suppression_key: {trigger.get('suppression_key', '')}
- anchor on at least ONE concrete fact (number, date, stat, source)
- match language_pref: {lang_pref}
- right voice for {cat_slug}: {cat_voice.get('tone','')}

Respond ONLY with JSON object. No extra text."""

async def compose_one(test_pair: dict) -> dict:
    test_id = test_pair["test_id"]
    merchant_id = test_pair["merchant_id"]
    trigger_id = test_pair["trigger_id"]
    customer_id = test_pair.get("customer_id")
    
    # Load files
    merchant_file = find_file(DATASET_DIR / "merchants", merchant_id)
    trigger_file = find_file(DATASET_DIR / "triggers", trigger_id)
    
    if not merchant_file or not trigger_file:
        print(f"  [WARN] Missing file for {test_id}: merchant={merchant_file}, trigger={trigger_file}")
        return {"test_id": test_id, "body": "Error: missing context", "cta": "none", "send_as": "vera", "suppression_key": "", "rationale": "Missing context files"}
    
    merchant = load_json(merchant_file)
    trigger = load_json(trigger_file)
    cat_slug = merchant.get("category_slug", "")
    
    category_file = DATASET_DIR / "categories" / f"{cat_slug}.json"
    if not category_file.exists():
        print(f"  [WARN] Category file not found: {cat_slug}")
        return {"test_id": test_id, "body": "Error: missing category", "cta": "none", "send_as": "vera", "suppression_key": "", "rationale": "Missing category"}
    
    category = load_json(category_file)
    
    customer = None
    if customer_id:
        customer_file = find_file(DATASET_DIR / "customers", customer_id)
        if customer_file:
            customer = load_json(customer_file)
    
    prompt = build_prompt(category, merchant, trigger, customer)
    
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
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    
    result = json.loads(text)
    result["test_id"] = test_id
    return result

async def main():
    test_pairs_file = DATASET_DIR / "test_pairs.json"
    test_pairs = load_json(test_pairs_file)["pairs"]
    print(f"Generating submissions for {len(test_pairs)} test pairs...\n")
    
    results = []
    for i, pair in enumerate(test_pairs):
        print(f"  [{i+1:02d}/30] {pair['test_id']} — {pair['merchant_id']} + {pair['trigger_id']}")
        try:
            result = await compose_one(pair)
            print(f"         ✓ {result.get('cta','?')} | {result.get('send_as','?')}")
            print(f"           {result.get('body','')[:80]}...")
        except Exception as e:
            print(f"         ✗ ERROR: {e}")
            result = {
                "test_id": pair["test_id"],
                "body": f"Error composing: {str(e)[:50]}",
                "cta": "none", "send_as": "vera",
                "suppression_key": "",
                "rationale": "Generation error"
            }
        results.append(result)
        await asyncio.sleep(0.5)  # gentle rate limiting
    
    # Write submission.jsonl
    output_path = Path("../submission.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Written to {output_path}")
    print(f"   Total: {len(results)} lines")

if __name__ == "__main__":
    asyncio.run(main())
