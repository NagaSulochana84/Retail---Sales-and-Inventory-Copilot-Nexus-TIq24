# Retail---Sales-and-Inventory-Copilot-Nexus-TIq24
PharmaCopilot is a Retail-Sales and Inventory Copilot designed for small pharmacies. I narrowed the general retail problem to pharmacy to make it more accurate to a real-world scenario and enable a realistic demo. It identifies low stock, stock-outs, overstock, slow-moving and near-expiry medicines.

## Run

From the project folder:

```text
python -m pip install -r requirements.txt
python app.py
```

Then open http://localhost:8000.

For live Gemini chat, create a `.env` file in this folder with a Gemini API key:

```text
GEMINI_API_KEY=your_gemini_api_key_here
# Optional: choose a model available to your API project
GEMINI_CHAT_MODEL=gemini-2.5-flash
# Use true for a keyless demo when Gemini quota is unavailable
GEMINI_OFFLINE_MODE=false
```

The app loads this file relative to `app.py`, so the commands work from the project folder without requiring a specific Python executable version or a separate working directory.

Gemini API keys use the quota of their Google Cloud project. Creating another key in the same project does not create another free-tier quota. If the app reports `429 RESOURCE_EXHAUSTED`, use a key from a different project, enable billing for the project, or wait for the quota reset.

For a keyless demo when Gemini quota is unavailable, set `GEMINI_OFFLINE_MODE=true` and remove the `GEMINI_API_KEY` line. The dashboard and chat analytics then use the local CSV data without making API calls. The app also automatically uses this local mode when no API key is present, so the standard `python app.py` command still starts successfully.
