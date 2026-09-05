"""
PharmaCopilot — data_loader.py
Step 2: Loads all 4 CSV files into memory once at startup.
Provides clean query functions used by analytics (Step 3) and Q&A (Step 4).
No pandas — pure Python csv module.
"""

import csv
import os
from datetime import date, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── In-memory tables (loaded once at startup) ─────────────────────────────────
_stores   = []
_products = []
_sales    = []
_stock    = []


def _load_csv(filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all():
    """Call once at app startup to load all CSVs into memory."""
    global _stores, _products, _sales, _stock

    _stores   = _load_csv("stores.csv")
    _products = _load_csv("products.csv")

    raw_sales = _load_csv("sales.csv")
    for row in raw_sales:
        row["units_sold"] = int(row["units_sold"])
    _sales = raw_sales

    raw_stock = _load_csv("stocks.csv")
    for row in raw_stock:
        row["current_stock"] = int(row["current_stock"])
    _stock = raw_stock

    print(f"  [data] Loaded {len(_stores)} stores, {len(_products)} products, "
          f"{len(_sales)} sales rows, {len(_stock)} stock rows.")


# ── Accessors ──────────────────────────────────────────────────────────────────

def get_stores() -> list[dict]:
    return _stores


def get_products() -> list[dict]:
    return _products


def get_store_map() -> dict:
    """Dict keyed by store_id for O(1) lookup."""
    return {s["store_id"]: s for s in _stores}


def get_product_map() -> dict:
    """Dict keyed by product_id for O(1) lookup."""
    return {p["product_id"]: p for p in _products}


def get_sales(
    store_id: str = None,
    product_id: str = None,
    since_date: str = None,   # "YYYY-MM-DD"
    until_date: str = None,   # "YYYY-MM-DD"
) -> list[dict]:
    """Return filtered sales rows."""
    rows = _sales
    if store_id:
        rows = [r for r in rows if r["store_id"] == store_id]
    if product_id:
        rows = [r for r in rows if r["product_id"] == product_id]
    if since_date:
        rows = [r for r in rows if r["date"] >= since_date]
    if until_date:
        rows = [r for r in rows if r["date"] <= until_date]
    return rows


def get_stock(
    store_id: str = None,
    product_id: str = None,
) -> list[dict]:
    """Return filtered stock rows."""
    rows = _stock
    if store_id:
        rows = [r for r in rows if r["store_id"] == store_id]
    if product_id:
        rows = [r for r in rows if r["product_id"] == product_id]
    return rows


# ── Helper calculations used by analytics ─────────────────────────────────────

def avg_daily_sales(
    product_id: str,
    store_id: str = None,
    days: int = 7,
    offset_days: int = 0,
) -> float:
    """
    Average units sold per day for a product over a window.
    offset_days=0  → last `days` days (recent window)
    offset_days=7  → 7–14 days ago (prior window, for spike/drop comparison)
    """
    today_str = date.today().isoformat()
    end   = (date.today() - timedelta(days=offset_days)).isoformat()
    start = (date.today() - timedelta(days=offset_days + days)).isoformat()

    rows = get_sales(store_id=store_id, product_id=product_id,
                     since_date=start, until_date=end)
    total = sum(r["units_sold"] for r in rows)
    return round(total / days, 2)


def total_stock(product_id: str, store_id: str = None) -> int:
    """Total current stock across all stores (or one store)."""
    rows = get_stock(store_id=store_id, product_id=product_id)
    return sum(r["current_stock"] for r in rows)
