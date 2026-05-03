#!/usr/bin/env python3
"""
conversation_handlers.py — Multi-Turn Conversation Handler for Vera Bot
=========================================================================
Optional module demonstrating multi-turn capability.

Usage:
    from conversation_handlers import respond, ConversationState
    
    state = ConversationState(
        conversation_id="conv_001",
        merchant_id="m_001_drmeera_dentist_delhi",
        customer_id=None,
        turns=[]
    )
    result = respond(state, "Yes, send me the abstract")
"""

import json
import httpx
from dataclasses import dataclass, field
from typing import Optional

# ─── State ────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    from_role: str          # "vera" | "merchant" | "customer"
    body: str
    timestamp: str = ""
    engagement_tag: str = ""  # "accepted" | "auto_reply" | "not_interested" | "question" | ""

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    turns: list[Turn] = field(default_factory=list)
    phase: str = "active"   # "active" | "action_mode" | "winding_down" | "ended"
    auto_reply_count: int = 0
    last_intent: str = ""   # "accepted" | "not_interested" | "join_intent" | "question"

# ─── Intent & State Classifiers ───────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "aapki jaankari ke liye",
    "thank you for contacting",
    "thanks for contacting",
    "main ek automated",
    "bahut-bahut shukriya",
    "our team will get back",
    "we will respond shortly",
    "your message has been received",
    "automated assistant",
]

ACCEPT_PHRASES = ["yes", "haan", "ok", "okay", "sure", "chalega", "go ahead",
                   "let's do it", "karte hain", "bilkul", "sahi hai", "theek hai"]

NOT_INTERESTED_PHRASES = ["not interested", "nahi chahiye", "band karo", "stop",
                           "mat bhejo", "no thanks", "abhi nahi", "baad mein dekha jayega"]

JOIN_PHRASES = ["join karna", "join chahta", "judrna", "sign up", "onboard"]

def classify_message(message: str, state: ConversationState) -> str:
    """Classify merchant message into intent category."""
    msg_lower = message.lower().strip()
    
    # Auto-reply detection
    for phrase in AUTO_REPLY_PHRASES:
        if phrase in msg_lower:
            return "auto_reply"
    # Repeated message = auto-reply
    if len(state.turns) >= 2:
        last_merchant_msgs = [t.body.strip() for t in state.turns if t.from_role == "merchant"]
        if last_merchant_msgs.count(message.strip()) >= 1:
            return "auto_reply"
    
    for phrase in NOT_INTERESTED_PHRASES:
        if phrase in msg_lower:
            return "not_interested"
    
    for phrase in JOIN_PHRASES:
        if phrase in msg_lower:
            return "join_intent"
    
    for phrase in ACCEPT_PHRASES:
        if msg_lower == phrase or msg_lower.startswith(phrase + " ") or msg_lower.endswith(" " + phrase):
            return "accepted"
    
    if "?" in message:
        return "question"
    
    return "neutral"

# ─── Response Templates by Intent ─────────────────────────────────────────────

def respond_to_auto_reply(state: ConversationState, merchant_name: str) -> dict:
    """Handle auto-reply detection."""
    state.auto_reply_count += 1
    
    if state.auto_reply_count == 1:
        # First auto-reply: one gentle redirect
        body = (
            f"Lagta hai yeh auto-reply hai 😊 "
            f"Agar {merchant_name} khud dekh rahe hain — "
            f"main 2 min mein ek quick update share karti hoon. "
            f"Chalega? Reply YES / STOP"
        )
        state.turns.append(Turn(from_role="vera", body=body, engagement_tag="auto_reply_redirect"))
        return {
            "action": "send",
            "body": body,
            "cta": "binary_yes_stop",
            "rationale": "First auto-reply detected. Sending one human-targeted bypass message before potential exit."
        }
    else:
        # Second auto-reply: graceful exit
        state.phase = "ended"
        return {
            "action": "end",
            "rationale": f"Auto-reply detected {state.auto_reply_count} times. Gracefully exiting to avoid polluting merchant's inbox."
        }

def respond_to_not_interested(state: ConversationState) -> dict:
    """Handle clear rejection."""
    state.phase = "ended"
    return {
        "action": "end",
        "rationale": "Merchant signalled not interested or stop. Ending conversation respectfully."
    }

def respond_to_join_intent(state: ConversationState, merchant_name: str) -> dict:
    """Handle explicit join intent — switch immediately to action mode."""
    state.phase = "action_mode"
    state.last_intent = "join_intent"
    
    body = (
        f"Bilkul! Joining is 3 quick steps:\n"
        f"1️⃣ Profile verify\n"
        f"2️⃣ First offer set up\n"
        f"3️⃣ Go live on magicpin 🎉\n\n"
        f"Main abhi Step 1 shuru karti hoon. Ready? Reply YES / STOP"
    )
    state.turns.append(Turn(from_role="vera", body=body, engagement_tag="action_mode"))
    return {
        "action": "send",
        "body": body,
        "cta": "binary_yes_stop",
        "rationale": "Merchant expressed explicit join intent. Immediately switching to onboarding action mode, no re-qualification."
    }

def respond_to_accept(state: ConversationState, merchant_context: dict) -> dict:
    """Handle acceptance — move forward with the promised action."""
    state.phase = "action_mode"
    
    # Figure out what was promised in the last vera turn
    last_vera = next(
        (t.body for t in reversed(state.turns) if t.from_role == "vera"),
        ""
    )
    
    m_name = merchant_context.get("identity", {}).get("owner_first_name", "aap")
    
    # Compose a contextual follow-through
    active_offers = [o for o in merchant_context.get("offers", []) if o.get("status") == "active"]
    offer_title = active_offers[0]["title"] if active_offers else "your offer"
    
    body = (
        f"Done! Main abhi isko process kar rahi hoon. "
        f"Thodi der mein update aayega, {m_name}. "
        f"Koi aur cheez chahiye? Reply kar sakte hain."
    )
    state.turns.append(Turn(from_role="vera", body=body, engagement_tag="accepted"))
    return {
        "action": "send",
        "body": body,
        "cta": "none",
        "rationale": "Merchant accepted. Acknowledging and executing the promised action without re-pitching."
    }

# ─── Main respond() function ──────────────────────────────────────────────────

def respond(state: ConversationState, merchant_message: str,
            merchant_context: Optional[dict] = None) -> dict:
    """
    Given conversation state + merchant's latest message, produce the next reply.
    
    Args:
        state: ConversationState with history
        merchant_message: The latest message from merchant/customer
        merchant_context: Optional merchant dict for personalization
    
    Returns:
        dict with keys: action ("send"|"wait"|"end"), body (if send), cta, rationale
    """
    
    # Record incoming message
    state.turns.append(Turn(from_role="merchant", body=merchant_message))
    
    # Already ended
    if state.phase == "ended":
        return {
            "action": "end",
            "rationale": "Conversation already ended."
        }
    
    # Classify intent
    intent = classify_message(merchant_message, state)
    state.last_intent = intent
    
    m_name = "aap"
    if merchant_context:
        m_name = merchant_context.get("identity", {}).get("owner_first_name", "aap")
    
    # Route by intent
    if intent == "auto_reply":
        return respond_to_auto_reply(state, m_name)
    
    if intent == "not_interested":
        return respond_to_not_interested(state)
    
    if intent == "join_intent":
        return respond_to_join_intent(state, m_name)
    
    if intent == "accepted":
        return respond_to_accept(state, merchant_context or {})
    
    if intent == "question":
        # Pass back to Groq via bot.py for contextual answer
        # Here we provide a simple fallback template
        body = (
            f"Achha sawaal! Main check karke 2 minute mein batati hoon, {m_name}. "
            f"Tab tak — koi aur cheez?"
        )
        state.turns.append(Turn(from_role="vera", body=body, engagement_tag="question_deflect"))
        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "Merchant asked a question. Acknowledging and promising a specific answer."
        }
    
    # Neutral / default
    # Check if we've had too many unanswered turns
    vera_turns_without_merchant_engage = 0
    for t in reversed(state.turns):
        if t.from_role == "vera":
            vera_turns_without_merchant_engage += 1
        else:
            break
    
    if vera_turns_without_merchant_engage >= 3:
        # Graceful wrap-up
        state.phase = "winding_down"
        body = f"Koi baat nahi, {m_name}! Jab bhi zaroorat ho, main yahan hoon. Good luck! 🙂"
        state.turns.append(Turn(from_role="vera", body=body))
        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "3+ unanswered Vera turns. Gracefully winding down conversation."
        }
    
    # Generic follow-up
    body = f"Samajh gayi! Kuch aur help chahiye? Batao, {m_name}."
    state.turns.append(Turn(from_role="vera", body=body))
    return {
        "action": "send",
        "body": body,
        "cta": "none",
        "rationale": "Neutral merchant response; keeping conversation open with a soft follow-up."
    }

# ─── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing conversation_handlers.py...\n")
    
    state = ConversationState(
        conversation_id="test_conv_001",
        merchant_id="m_001",
        customer_id=None
    )
    
    test_messages = [
        "Yes, send me the abstract",
        "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein team tak pahuncha deti hoon.",
        "Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein team tak pahuncha deti hoon.",
    ]
    
    for msg in test_messages:
        print(f"MERCHANT: {msg}")
        result = respond(state, msg)
        print(f"VERA ({result['action']}): {result.get('body', 'N/A')}")
        print(f"RATIONALE: {result['rationale']}\n")
        if result["action"] == "end":
            break
