# Vera Bot — magicpin AI Challenge Submission

**Team:** Munthasir  
**Model:** `llama-3.3-70b-versatile` (Groq)  
**Live URL:** https://vera-bot-by62.onrender.com

---

## What is Vera?

Vera is magicpin's AI-powered merchant engagement assistant. It composes highly personalised WhatsApp messages for merchants using a 4-layer context framework:

1. **Category context** — voice/tone, taboo words, peer benchmarks
2. **Merchant context** — performance, offers, signals, reviews
3. **Trigger context** — what happened and why to act now
4. **Customer context** — name, language, relationship, preferences

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/v1/healthz` | Health check |
| GET | `/v1/metadata` | Team & model info |
| POST | `/v1/context` | Push category/merchant/customer/trigger context |
| POST | `/v1/tick` | Generate proactive messages for available triggers |
| POST | `/v1/reply` | Handle inbound replies (auto-reply, intent, slot booking) |

---

## Key Features

- **Specificity** — uses ≥2 exact numbers/dates from trigger payload every message
- **Category voice** — dentist (peer-clinical), salon (warm), restaurant (energetic), gym (motivational), pharmacy (formal)
- **Smart behaviours** — auto-reply detection, intent transition (yes/ok → action mode), hostile message handling
- **Source citation** — cites JIDA/DCI/IDA digest items inline
- **Slot scarcity** — names specific slots and remaining count
- **Suppression dedup** — never sends the same message twice
- **Concurrent processing** — handles up to 20 triggers in parallel

---

## Dataset

```
dataset/
├── categories/     # 5 category configs (dentists, salons, restaurants, gyms, pharmacies)
├── merchants/      # 50 merchant profiles
├── customers/      # 200 customer profiles
├── triggers/       # 100 trigger definitions
└── test_pairs.json # 30 evaluation test pairs (T01–T30)
```

---

## Running Locally

```bash
pip install -r requirements.txt
# Set GROQ_API_KEY in .env
uvicorn bot:app --host 0.0.0.0 --port 8080
```

---

## Deployment

Deployed on **Render** (Free Web Service) via `render.yaml`.  
Environment variable required: `GROQ_API_KEY`
