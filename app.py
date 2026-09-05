"""
PharmaCopilot — app.py
Step 2: Flask server + Gemini (google-genai SDK) + CSV data layer.

Run:  py -3.12 app.py
Test: http://localhost:8000/api/health
      http://localhost:8000/api/gemini-test
      http://localhost:8000/api/data/summary
      http://localhost:8000/api/data/stores
      http://localhost:8000/api/data/products
      http://localhost:8000/api/data/stock?store_id=S001
"""

import os
import traceback

from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from google import genai

import data_loader
import analytics
import qa_pipeline

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Create a .env file with: GEMINI_API_KEY=your_key_here"
    )

# ── Gemini client (new google-genai SDK) ──────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Model names — single place to update ──────────────────────────────────────
CHAT_MODEL      = "gemini-2.5-flash"      # free tier chat/generation model
EMBEDDING_MODEL = "gemini-embedding-001"  # for RAG/semantic search (Step 4)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — General
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the PharmaCopilot frontend."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "port": 8000})


@app.route("/api/gemini-test")
def gemini_test():
    """Live Gemini call — language only, no math, no invented numbers."""
    try:
        prompt = (
            "You are PharmaCopilot, an AI assistant for a pharmacy chain. "
            "Greet the pharmacy manager in one short sentence and mention "
            "that you are ready to help with sales and inventory questions."
        )
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        return jsonify({
            "status":       "ok",
            "model":        CHAT_MODEL,
            "gemini_reply": response.text.strip()
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error":  str(exc),
            "trace":  traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Data layer (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/data/summary")
def data_summary():
    """Snapshot of loaded data — judges can verify CSV data is live."""
    try:
        stores   = data_loader.get_stores()
        products = data_loader.get_products()
        sales    = data_loader.get_sales()
        stock    = data_loader.get_stock()

        dates     = [r["date"] for r in sales]
        date_from = min(dates) if dates else "n/a"
        date_to   = max(dates) if dates else "n/a"

        return jsonify({
            "status": "ok",
            "loaded": {
                "stores":      len(stores),
                "products":    len(products),
                "sales_rows":  len(sales),
                "stock_rows":  len(stock),
                "sales_from":  date_from,
                "sales_to":    date_to,
            }
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/data/stores")
def api_stores():
    """List all stores."""
    try:
        return jsonify({"status": "ok", "stores": data_loader.get_stores()})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/data/products")
def api_products():
    """
    List products. Optional: ?category=OTC+Medicine
    """
    try:
        products = data_loader.get_products()
        category = request.args.get("category")
        if category:
            products = [p for p in products if p["category"] == category]
        return jsonify({
            "status":   "ok",
            "count":    len(products),
            "products": products
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/data/stock")
def api_stock():
    """
    Stock snapshot. Optional: ?store_id=S001 or ?product_id=P004
    """
    try:
        store_id   = request.args.get("store_id")
        product_id = request.args.get("product_id")
        rows       = data_loader.get_stock(store_id=store_id,
                                           product_id=product_id)
        pmap = data_loader.get_product_map()
        smap = data_loader.get_store_map()
        enriched = [{
            **r,
            "product_name": pmap.get(r["product_id"], {}).get("product_name", ""),
            "store_name":   smap.get(r["store_id"],   {}).get("store_name",   ""),
        } for r in rows]

        return jsonify({"status": "ok", "count": len(enriched), "stock": enriched})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Analytics / Attention Panel (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/attention")
def attention_panel():
    """
    Today's Attention panel — runs all 4 deterministic checks.
    Pure Python math — Gemini is NOT called here.
    Each flag includes the numbers, assumption, and recommended action.
    """
    try:
        report = analytics.run_all_checks()
        return jsonify({"status": "ok", "report": report})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc),
                        "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Q&A Pipeline (Step 5)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Manager asks a plain-language question.
    Pipeline: Gemini parses intent → Python fetches numbers → Gemini phrases answer.
    Gemini NEVER invents numbers.
    Body: { "question": "what's running low at the mall branch?" }
    """
    try:
        body     = request.get_json(force=True, silent=True) or {}
        question = (body.get("question") or "").strip()

        if not question:
            return jsonify({
                "status": "error",
                "answer": "Please provide a question in the request body: {\"question\": \"...\"}"
            }), 400

        result = qa_pipeline.answer_question(question, client, CHAT_MODEL)
        return jsonify({"status": "ok", **result})

    except Exception as exc:
        return jsonify({
            "status": "error",
            "answer": "Something went wrong. Please try again.",
            "error":  str(exc),
            "trace":  traceback.format_exc()
        }), 500


@app.route("/api/chat/test-refusal")
def chat_refusal_test():
    """
    Step 6 — tests all 7 judge-style questions including deliberate traps.
    Hit this endpoint to verify every scenario works before demo.
    """
    test_questions = [
        "Which products are likely to run out of stock soon?",
        "How did Paracetamol 500mg sell this month at the Mall Branch?",
        "What's the stock status of Ibuprofen Gel 500mg?",          # trap — doesn't exist
        "Which store is performing worst this month and why?",
        "What's my profit margin on cold and flu medicines?",        # out of scope
        "Is anything close to expiry that I should worry about?",
        "What's overstocked?",
    ]
    results = []
    for q in test_questions:
        result = qa_pipeline.answer_question(q, client, CHAT_MODEL)
        results.append({
            "question": q,
            "intent":   result.get("intent"),
            "answer":   result.get("answer"),
        })
    return jsonify({"status": "ok", "tests": results})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading CSV data...")
    data_loader.load_all()

    print("=" * 55)
    print("  PharmaCopilot — NexusTiq24 (TRACK PS03)")
    print("  Server : http://localhost:8000")
    print("  Data   : http://localhost:8000/api/data/summary")
    print("=" * 55)

    app.run(host="0.0.0.0", port=8000, debug=False)
