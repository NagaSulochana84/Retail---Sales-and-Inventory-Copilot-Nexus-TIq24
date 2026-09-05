"""
PharmaCopilot — app.py
Step 1: Flask server on port 8000 + Gemini hello-world end-to-end check.

Run:  python app.py
Test: http://localhost:8000/
      http://localhost:8000/api/health
      http://localhost:8000/api/gemini-test

Note: Uses Python built-in csv module — no pandas/polars dependency.
"""

import os
import traceback

from flask import Flask, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# ── Load environment variables from .env (GEMINI_API_KEY lives here) ──────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Create a .env file with: GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

# ── Model names — change here if needed, never scattered in code ───────────────
CHAT_MODEL      = "gemini-2.5-flash"   # free tier chat/generation model
EMBEDDING_MODEL = "gemini-embedding-001"  # for RAG/semantic search (Step 4)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Root — quick confirmation the server is alive."""
    return jsonify({
        "app":    "PharmaCopilot",
        "track":  "PS03 — NexusTiq24",
        "status": "running",
        "endpoints": {
            "health":      "/api/health",
            "gemini_test": "/api/gemini-test",
        }
    })


@app.route("/api/health")
def health():
    """Health-check — judges can hit this to confirm the server is up."""
    return jsonify({"status": "ok", "port": 8000})


@app.route("/api/gemini-test")
def gemini_test():
    """
    Live Gemini call — proves the API key works end-to-end.
    Gemini is ONLY used for language — no math, no invented numbers.
    """
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        prompt = (
            "You are PharmaCopilot, an AI assistant for a pharmacy chain. "
            "Greet the pharmacy manager in one short sentence and mention "
            "that you are ready to help with sales and inventory questions."
        )
        response = model.generate_content(prompt)
        gemini_reply = response.text.strip()

        return jsonify({
            "status":       "ok",
            "model":        CHAT_MODEL,
            "gemini_reply": gemini_reply
        })

    except Exception as exc:
        # Graceful error — never crash, always return JSON
        return jsonify({
            "status": "error",
            "error":  str(exc),
            "trace":  traceback.format_exc()
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PharmaCopilot — NexusTiq24 (TRACK PS03)")
    print("  Server: http://localhost:8000")
    print("  Gemini test: http://localhost:8000/api/gemini-test")
    print("=" * 55)
    app.run(host="0.0.0.0", port=8000, debug=False)
