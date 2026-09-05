"""
PharmaCopilot — qa_pipeline.py  (v2 — judge-proof edition)
Step 5+6: Q&A pipeline + honest refusal path.

Pipeline for every question:
  1. Python: validate product/store name exists in data (trap fuzzy names early)
  2. Gemini: parse question → structured intent JSON
  3. Python: compute ALL numbers deterministically
  4. Gemini: phrase a short factual answer + empathetic tone, using ONLY those numbers
  5. Every answer ends with data source + date range

Gemini's ONLY jobs:
  (a) Understand plain language → intent JSON
  (b) Phrase the answer using Python-computed facts — never invent a single number

Out-of-scope categories with specific honest messages:
  - Financial (profit/margin)  → we have cost+price but no P&L tracking
  - Supplier/logistics          → not in our data
  - Future prediction           → we show trends, not forecasts
  - External (competitor/weather) → internal data only
"""

import json
import traceback
from datetime import date, timedelta

from google import genai
import data_loader
import analytics

TODAY      = date.today()
TODAY_STR  = TODAY.isoformat()
DATA_START = (TODAY - timedelta(days=60)).isoformat()   # 2026-07-07
DATA_FOOTER = f"📊 Data: {DATA_START} to {TODAY_STR} | 3 MedPlus branches (Downtown, Mall Branch, Suburb)"

# ── Out-of-scope keyword sets ─────────────────────────────────────────────────
_OOS_FINANCIAL  = {"profit","margin","revenue","income","loss","earning","p&l","roi","markup"}
_OOS_SUPPLIER   = {"supplier","vendor","delivery","shipment","lead time","order from","restock from","distributor"}
_OOS_PREDICTION = {"next month","next week","next year","forecast","predict","will sell","future","projection"}
_OOS_EXTERNAL   = {"competitor","weather","market","industry","trend outside","news","inflation","economy"}


# ── 1. Product + Store fuzzy-match validation ─────────────────────────────────

def _find_product(query_name: str) -> dict | None:
    """
    Returns matching product dict if found, else None.
    Matches on product_id (exact) or product_name (case-insensitive substring).
    """
    if not query_name:
        return None
    q = query_name.strip().lower()
    for p in data_loader.get_products():
        if q == p["product_id"].lower():
            return p
        if q in p["product_name"].lower() or p["product_name"].lower() in q:
            return p
    return None


def _suggest_products(query_name: str) -> list[str]:
    """Return up to 3 products whose words partially overlap with the query."""
    q_words = set(query_name.lower().split())
    scored  = []
    for p in data_loader.get_products():
        p_words = set(p["product_name"].lower().split())
        overlap = len(q_words & p_words)
        if overlap > 0:
            scored.append((overlap, p["product_name"]))
    scored.sort(reverse=True)
    return [name for _, name in scored[:3]]


def _find_store(query_name: str) -> dict | None:
    """Returns matching store dict or None."""
    if not query_name:
        return None
    q = query_name.strip().lower()
    for s in data_loader.get_stores():
        if q == s["store_id"].lower():
            return s
        if q in s["store_name"].lower() or s["store_name"].lower() in q:
            return s
        # common aliases
        if "mall" in q and "mall" in s["store_name"].lower():
            return s
        if "downtown" in q and "downtown" in s["store_name"].lower():
            return s
        if ("suburb" in q or "suburban" in q) and "suburb" in s["store_name"].lower():
            return s
    return None


def _detect_oos(question: str) -> str | None:
    """Return out-of-scope category name if question is outside available data, else None."""
    q = question.lower()
    if any(kw in q for kw in _OOS_FINANCIAL):
        return "financial"
    if any(kw in q for kw in _OOS_SUPPLIER):
        return "supplier"
    if any(kw in q for kw in _OOS_PREDICTION):
        return "prediction"
    if any(kw in q for kw in _OOS_EXTERNAL):
        return "external"
    return None


# ── 2. Gemini intent parser ───────────────────────────────────────────────────

INTENT_PROMPT = """You are the intent parser for PharmaCopilot, a pharmacy inventory assistant.

Available data:
  Stores : S001=MedPlus Downtown, S002=MedPlus Mall Branch, S003=MedPlus Suburb
  Products (id → name):
    P001=Paracetamol 500mg Strip, P002=Ibuprofen 400mg Strip, P003=Cetirizine 10mg Strip,
    P004=Antacid Syrup 200ml, P005=Cough Syrup 100ml, P006=Vitamin C 500mg Strip,
    P007=ORS Sachets Pack of 10, P008=Metformin 500mg Strip, P009=Atorvastatin 10mg Strip,
    P010=Amlodipine 5mg Strip, P011=Azithromycin 500mg Strip, P012=Pantoprazole 40mg Strip,
    P013=Losartan 50mg Strip, P014=Hand Sanitizer 500ml, P015=Sunscreen SPF50 50g,
    P016=Moisturizing Lotion 200ml, P017=Antiseptic Cream 25g, P018=Lip Balm SPF15,
    P019=Baby Diaper Rash Cream 50g, P020=Baby Fever Syrup 60ml, P021=Baby Vitamin D Drops 15ml,
    P022=Baby Wet Wipes Pack of 72, P023=Digital Thermometer, P024=Blood Pressure Monitor,
    P025=Glucometer Test Strips Pack of 25
  Sales history: {start} to {today}

Intents available:
  stock_level        — current stock of a product
  sales_summary      — how a product sold in a time period
  low_stock          — what products are at stockout risk
  expiry_risk        — what is expiring soon
  dead_stock         — what is overstocked / not moving
  sales_spike_drop   — unusual sales trends
  store_comparison   — which store is performing best/worst
  attention_summary  — general "what needs attention today"
  unknown            — question cannot be mapped to available data

Return ONLY a JSON object, no markdown, no explanation:
{{
  "intent": "<one of above>",
  "store_id": "<S001/S002/S003 or null>",
  "product_id": "<P001...P025 or null — use null if product mentioned does not exist in list above>",
  "product_name_mentioned": "<exact product name the manager typed, or null>",
  "days": <integer, default 30 for 'this month', 7 for recent>,
  "confidence": "high" or "low"
}}

Manager's question: "{question}"
"""


def parse_intent(question: str, gemini_client, chat_model: str) -> dict:
    prompt = INTENT_PROMPT.format(
        start=DATA_START, today=TODAY_STR, question=question
    )
    try:
        resp = gemini_client.models.generate_content(model=chat_model, contents=prompt)
        raw  = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        return {"intent": "unknown", "store_id": None, "product_id": None,
                "product_name_mentioned": None, "days": 7,
                "confidence": "low", "_parse_error": str(exc)}


# ── 3. Python fact fetchers ───────────────────────────────────────────────────

def _time_window(days: int) -> str:
    return (TODAY - timedelta(days=days)).isoformat()


def fetch_facts(intent: dict) -> dict:
    """
    All computation in Python — no Gemini involved.
    Returns structured facts (or found=False with honest reason).
    """
    itype      = intent.get("intent", "unknown")
    store_id   = intent.get("store_id")
    product_id = intent.get("product_id")
    mentioned  = intent.get("product_name_mentioned")
    days       = max(1, int(intent.get("days") or 7))

    pmap = data_loader.get_product_map()
    smap = data_loader.get_store_map()

    def pname(pid): return pmap.get(pid, {}).get("product_name", pid) if pid else None
    def sname(sid): return smap.get(sid, {}).get("store_name", sid)   if sid else "all stores"

    # ── Product existence check (trap question handler) ───────────────────────
    if mentioned and not product_id:
        # Gemini couldn't map it → product doesn't exist in our data
        suggestions = _suggest_products(mentioned)
        return {
            "found": False,
            "reason": "product_not_found",
            "searched_for": mentioned,
            "suggestions": suggestions,
            "data_has": f"{len(data_loader.get_products())} products listed in our system",
        }

    # ── Intent handlers ───────────────────────────────────────────────────────

    if itype == "stock_level":
        if not product_id:
            return {"found": False, "reason": "No specific product mentioned in your question."}
        rows = data_loader.get_stock(store_id=store_id, product_id=product_id)
        if not rows:
            return {"found": False, "reason": "No stock record found.",
                    "product": pname(product_id), "store": sname(store_id)}
        total = sum(r["current_stock"] for r in rows)
        breakdown = [{"store": sname(r["store_id"]), "stock": r["current_stock"],
                      "expiry": r.get("expiry_date","N/A")} for r in rows]
        rate  = data_loader.avg_daily_sales(product_id, store_id=store_id, days=7)
        return {
            "found": True, "intent": itype,
            "product": pname(product_id), "store": sname(store_id),
            "total_stock": total, "breakdown": breakdown,
            "avg_daily_rate_7d": rate,
            "days_left": round(total / rate, 1) if rate > 0 else "N/A (no recent sales)",
        }

    elif itype == "sales_summary":
        if not product_id:
            return {"found": False, "reason": "Please mention a specific product name."}
        since = _time_window(days)
        rows  = data_loader.get_sales(store_id=store_id, product_id=product_id, since_date=since)
        total = sum(r["units_sold"] for r in rows)
        avg   = round(total / days, 2)
        # Revenue estimate using unit_price
        price = float(pmap.get(product_id, {}).get("unit_price", 0))
        return {
            "found": True, "intent": itype,
            "product": pname(product_id), "store": sname(store_id),
            "period_days": days, "period_from": since, "period_to": TODAY_STR,
            "total_units_sold": total, "avg_units_per_day": avg,
            "estimated_revenue_inr": round(total * price, 2),
        }

    elif itype == "low_stock":
        flags = analytics.check_stockout_risk()
        if store_id:
            flags = [f for f in flags if f["store_id"] == store_id]
        return {
            "found": True, "intent": itype,
            "store": sname(store_id),
            "count": len(flags),
            "items": flags[:6],
            "note": "Sorted by urgency (fewest days remaining first).",
        }

    elif itype == "expiry_risk":
        flags = analytics.check_expiry_risk()
        if store_id:
            flags = [f for f in flags if f["store_id"] == store_id]
        if product_id:
            flags = [f for f in flags if f["product_id"] == product_id]
        return {
            "found": True, "intent": itype,
            "store": sname(store_id),
            "count": len(flags),
            "items": flags,
            "urgency_note": "Items sorted by expiry date — most urgent first.",
        }

    elif itype == "dead_stock":
        flags = analytics.check_dead_stock()
        return {
            "found": True, "intent": itype,
            "count": len(flags),
            "items": flags,
            "criteria": "Less than 0.5 units/day sold over the last 21 days with 30+ units in stock.",
        }

    elif itype == "sales_spike_drop":
        all_flags = analytics.check_sales_spikes_and_drops()
        spikes = [f for f in all_flags if f["flag"] == "sales_spike"]
        drops  = [f for f in all_flags if f["flag"] == "sales_drop"]
        if product_id:
            spikes = [f for f in spikes if f["product_id"] == product_id]
            drops  = [f for f in drops  if f["product_id"] == product_id]
        return {
            "found": True, "intent": itype,
            "spikes": spikes, "drops": drops,
            "method": "Compares last 7-day avg vs previous 7-day avg.",
        }

    elif itype == "store_comparison":
        since = _time_window(days)
        results = []
        for s in data_loader.get_stores():
            sid   = s["store_id"]
            rows  = data_loader.get_sales(store_id=sid, since_date=since)
            total = sum(r["units_sold"] for r in rows)
            # Rough revenue: units × unit_price per product
            rev = 0.0
            for r in rows:
                price = float(pmap.get(r["product_id"], {}).get("unit_price", 0))
                rev  += r["units_sold"] * price
            results.append({
                "store_id": sid,
                "store_name": s["store_name"],
                "total_units_sold": total,
                "estimated_revenue_inr": round(rev, 2),
                "unique_products_sold": len({r["product_id"] for r in rows}),
                "selling_days": len({r["date"] for r in rows}),
            })
        results.sort(key=lambda x: x["total_units_sold"])
        return {
            "found": True, "intent": itype,
            "period_days": days, "period_from": since, "period_to": TODAY_STR,
            "ranked_stores": results,  # index 0 = worst, -1 = best
            "note": "Ranked by total units sold. Revenue is estimated from unit_price × units_sold.",
        }

    elif itype == "attention_summary":
        report = analytics.run_all_checks()
        return {
            "found": True, "intent": itype,
            "summary": report["summary"],
            "top_stockouts": report["stockout_risk"][:3],
            "top_expiry":    report["expiry_risk"][:3],
            "top_dead":      report["dead_stock"][:2],
        }

    else:
        return {
            "found": False,
            "reason": "This question falls outside the data available to PharmaCopilot.",
            "what_i_have": (
                "Sales history (Jul 7–Sep 5, 2026), current stock levels, expiry dates, "
                "product catalogue (25 items), 3 branch locations."
            ),
        }


# ── 4. Gemini answer phraser ──────────────────────────────────────────────────

ANSWER_PROMPT = """You are PharmaCopilot — an honest, empathetic pharmacy inventory assistant.

STRICT RULES (violating any = wrong answer):
1. Use ONLY the numbers in the "Python-computed facts" block below. Never invent or estimate.
2. Start with the direct factual answer — lead with the key number(s).
3. Keep response to 3-5 sentences. Be warm but precise.
4. If facts contain "found": false and reason "product_not_found":
   - Say the product was not found in the system.
   - Mention the suggestions if any.
   - Do NOT make up stock/sales data for it.
5. If facts contain "found": false (other reasons):
   - Apologise briefly and explain honestly what data you do and don't have.
   - Never guess.
6. For "out_of_scope" facts: give the specific honest message from the facts block.
7. Always end your response with exactly this line (replace nothing):
   {footer}

Manager's question: "{question}"

Python-computed facts (your ONLY source of truth):
{facts}

Your answer:"""


def phrase_answer(question: str, facts: dict, gemini_client, chat_model: str) -> str:
    prompt = ANSWER_PROMPT.format(
        question=question,
        facts=json.dumps(facts, indent=2),
        footer=DATA_FOOTER
    )
    try:
        resp = gemini_client.models.generate_content(model=chat_model, contents=prompt)
        return resp.text.strip()
    except Exception as exc:
        return (
            f"I'm sorry, I ran into a technical issue phrasing the answer. "
            f"Error: {str(exc)}\n{DATA_FOOTER}"
        )


# ── Out-of-scope specific messages ────────────────────────────────────────────

OOS_MESSAGES = {
    "financial": {
        "found": False,
        "reason": "out_of_scope",
        "category": "financial",
        "honest_message": (
            "I have unit cost and unit price data for all 25 products, but I don't track "
            "profit & loss, margins, or overall revenue totals — that's outside my scope. "
            "For profitability analysis, please check your accounting software. "
            "I can tell you units sold and estimated revenue (units × price) if that helps."
        ),
    },
    "supplier": {
        "found": False,
        "reason": "out_of_scope",
        "category": "supplier",
        "honest_message": (
            "I don't have supplier, vendor, or delivery data — my knowledge is limited to "
            "in-store sales history, current stock levels, and expiry dates. "
            "Please check your procurement system for supplier lead times."
        ),
    },
    "prediction": {
        "found": False,
        "reason": "out_of_scope",
        "category": "prediction",
        "honest_message": (
            "I can't predict or forecast future sales — I only work with historical data "
            f"from {DATA_START} to {TODAY_STR}. "
            "What I can do is show you recent trends (last 7 vs prior 7 days) so you can "
            "make an informed judgment yourself."
        ),
    },
    "external": {
        "found": False,
        "reason": "out_of_scope",
        "category": "external",
        "honest_message": (
            "I only have access to internal data — sales, stock, and expiry records for "
            "your 3 MedPlus branches. I don't have competitor data, market trends, "
            "or any external information."
        ),
    },
}


# ── Master pipeline ───────────────────────────────────────────────────────────

def answer_question(question: str, gemini_client, chat_model: str) -> dict:
    """
    Full Q&A pipeline:
      1. Detect out-of-scope questions early (specific honest message per category)
      2. Gemini parses intent
      3. Python fetches facts (includes product-not-found trap handling)
      4. Gemini phrases answer using ONLY those facts
    """
    if not question or not question.strip():
        return {
            "question": question,
            "intent":   "empty",
            "facts":    {},
            "answer":   f"Please ask a question about your pharmacy's sales or inventory.\n{DATA_FOOTER}",
        }

    # Step 0 — out-of-scope early exit
    oos = _detect_oos(question)
    if oos:
        facts  = OOS_MESSAGES[oos]
        answer = phrase_answer(question, facts, gemini_client, chat_model)
        return {"question": question, "intent": f"out_of_scope_{oos}",
                "facts": facts, "answer": answer}

    try:
        # Step 1 — Gemini: parse intent
        intent = parse_intent(question, gemini_client, chat_model)

        # Step 2 — Python: fetch facts
        facts  = fetch_facts(intent)

        # Step 3 — Gemini: phrase answer
        answer = phrase_answer(question, facts, gemini_client, chat_model)

        return {
            "question": question,
            "intent":   intent.get("intent", "unknown"),
            "facts":    facts,
            "answer":   answer,
        }

    except Exception as exc:
        return {
            "question": question,
            "intent":   "error",
            "facts":    {},
            "answer":   (
                f"Sorry, something went wrong on my end. Please try again.\n{DATA_FOOTER}"
            ),
            "error":    str(exc),
            "trace":    traceback.format_exc(),
        }
