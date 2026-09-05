"""
PharmaCopilot — analytics.py
Step 3: Deterministic analysis engine.
ALL math is done here in Python — Gemini never touches numbers.

Flags generated:
  1. stockout_risk    — low stock + steady recent sales → runs out within 7 days
  2. dead_stock       — high stock + near-zero recent sales (last 21 days)
  3. sales_spike      — recent 7-day avg > 2x prior 7-day avg
  4. sales_drop       — recent 7-day avg < 0.4x prior 7-day avg
  5. expiry_risk      — expires within 30 days AND meaningful stock remaining
"""

from datetime import date, timedelta
import data_loader

TODAY      = date.today()
TODAY_STR  = TODAY.isoformat()

# ── Thresholds (all in one place — easy to tweak) ─────────────────────────────
STOCKOUT_DAYS_LEFT      = 7      # flag if days-of-stock remaining < this
DEAD_STOCK_WINDOW_DAYS  = 21     # look-back window for "near zero" check
DEAD_STOCK_MAX_RATE     = 0.5    # avg units/day below this = near zero
DEAD_STOCK_MIN_UNITS    = 30     # only flag if stock is meaningfully large
SPIKE_FACTOR            = 2.0    # recent avg must be > SPIKE_FACTOR × prior avg
DROP_FACTOR             = 0.4    # recent avg must be < DROP_FACTOR × prior avg
SPIKE_MIN_RECENT_SALES  = 3      # ignore spikes if recent avg is trivially small
EXPIRY_DAYS             = 30     # flag if expiry within this many days
EXPIRY_MIN_STOCK        = 10     # only flag if meaningful quantity remains


# ── Internal helpers ──────────────────────────────────────────────────────────

def _avg_daily(product_id: str, days: int, offset: int = 0) -> float:
    """
    Avg units/day for a product across ALL stores.
    offset=0  → most recent `days` days
    offset=7  → 7–14 days ago (prior window for spike/drop)
    """
    end_date   = (TODAY - timedelta(days=offset)).isoformat()
    start_date = (TODAY - timedelta(days=offset + days)).isoformat()
    rows = data_loader.get_sales(
        product_id=product_id,
        since_date=start_date,
        until_date=end_date
    )
    total = sum(r["units_sold"] for r in rows)
    return round(total / days, 2) if days > 0 else 0.0


def _avg_daily_by_store(product_id: str, store_id: str, days: int, offset: int = 0) -> float:
    """Avg units/day for a specific store+product."""
    end_date   = (TODAY - timedelta(days=offset)).isoformat()
    start_date = (TODAY - timedelta(days=offset + days)).isoformat()
    rows = data_loader.get_sales(
        product_id=product_id,
        store_id=store_id,
        since_date=start_date,
        until_date=end_date
    )
    total = sum(r["units_sold"] for r in rows)
    return round(total / days, 2) if days > 0 else 0.0


# ── Flag generators ───────────────────────────────────────────────────────────

def check_stockout_risk() -> list[dict]:
    """
    Per (store, product): if current_stock / recent_daily_rate < STOCKOUT_DAYS_LEFT → flag.
    Assumption: sales rate of the last 7 days continues unchanged.
    """
    flags = []
    pmap  = data_loader.get_product_map()
    smap  = data_loader.get_store_map()

    for row in data_loader.get_stock():
        sid = row["store_id"]
        pid = row["product_id"]
        stock = row["current_stock"]

        rate = _avg_daily_by_store(pid, sid, days=7)
        if rate <= 0:
            continue  # no recent sales → not a stockout risk

        days_left = round(stock / rate, 1)
        if days_left < STOCKOUT_DAYS_LEFT:
            flags.append({
                "flag":          "stockout_risk",
                "store_id":      sid,
                "store_name":    smap.get(sid, {}).get("store_name", sid),
                "product_id":    pid,
                "product_name":  pmap.get(pid, {}).get("product_name", pid),
                "current_stock": stock,
                "daily_rate":    rate,
                "days_left":     days_left,
                "assumption":    f"Sales continue at {rate} units/day (last 7-day avg)",
                "action":        f"Reorder immediately — stock lasts ~{days_left} day(s).",
            })

    return sorted(flags, key=lambda x: x["days_left"])


def check_dead_stock() -> list[dict]:
    """
    Per product (across all stores): if recent DEAD_STOCK_WINDOW_DAYS avg < threshold
    AND total stock is large → dead/overstock.
    Assumption: if no one has bought it in 3 weeks, demand has stalled.
    """
    flags = []
    pmap  = data_loader.get_product_map()

    seen = set()
    for row in data_loader.get_stock():
        pid = row["product_id"]
        if pid in seen:
            continue
        seen.add(pid)

        total_stk = sum(
            r["current_stock"] for r in data_loader.get_stock(product_id=pid)
        )
        if total_stk < DEAD_STOCK_MIN_UNITS:
            continue

        recent_rate = _avg_daily(pid, days=DEAD_STOCK_WINDOW_DAYS)
        if recent_rate < DEAD_STOCK_MAX_RATE:
            flags.append({
                "flag":          "dead_stock",
                "product_id":    pid,
                "product_name":  pmap.get(pid, {}).get("product_name", pid),
                "category":      pmap.get(pid, {}).get("category", ""),
                "total_stock":   total_stk,
                "daily_rate":    recent_rate,
                "window_days":   DEAD_STOCK_WINDOW_DAYS,
                "assumption":    f"Avg sales < {DEAD_STOCK_MAX_RATE} units/day over last {DEAD_STOCK_WINDOW_DAYS} days",
                "action":        "Consider promotion, return to supplier, or inter-branch transfer.",
            })

    return sorted(flags, key=lambda x: x["total_stock"], reverse=True)


def check_sales_spikes_and_drops() -> list[dict]:
    """
    Per product (all stores): compare recent 7-day avg vs prior 7-day avg.
    Spike : recent > SPIKE_FACTOR  × prior
    Drop  : recent < DROP_FACTOR   × prior
    """
    flags = []
    pmap  = data_loader.get_product_map()
    seen  = set()

    for row in data_loader.get_stock():
        pid = row["product_id"]
        if pid in seen:
            continue
        seen.add(pid)

        recent = _avg_daily(pid, days=7, offset=0)
        prior  = _avg_daily(pid, days=7, offset=7)

        # Spike
        if prior > 0 and recent >= SPIKE_MIN_RECENT_SALES and recent > SPIKE_FACTOR * prior:
            factor = round(recent / prior, 1)
            flags.append({
                "flag":         "sales_spike",
                "product_id":   pid,
                "product_name": pmap.get(pid, {}).get("product_name", pid),
                "category":     pmap.get(pid, {}).get("category", ""),
                "recent_avg":   recent,
                "prior_avg":    prior,
                "spike_factor": factor,
                "assumption":   "Comparing last 7 days vs previous 7 days (all stores combined)",
                "action":       f"Sales up {factor}x — check stock levels and consider emergency reorder.",
            })

        # Drop
        elif prior > 0 and recent < DROP_FACTOR * prior and prior >= 2:
            factor = round(recent / prior, 2)
            flags.append({
                "flag":         "sales_drop",
                "product_id":   pid,
                "product_name": pmap.get(pid, {}).get("product_name", pid),
                "category":     pmap.get(pid, {}).get("category", ""),
                "recent_avg":   recent,
                "prior_avg":    prior,
                "drop_factor":  factor,
                "assumption":   "Comparing last 7 days vs previous 7 days (all stores combined)",
                "action":       "Sales dropped sharply — check for supply issues, expiry, or competition.",
            })

    return flags


def check_expiry_risk() -> list[dict]:
    """
    Per (store, product): if expiry_date within EXPIRY_DAYS AND stock >= EXPIRY_MIN_STOCK → flag.
    """
    flags = []
    pmap  = data_loader.get_product_map()
    smap  = data_loader.get_store_map()

    for row in data_loader.get_stock():
        exp_str = row.get("expiry_date", "")
        if not exp_str:
            continue
        try:
            exp_date = date.fromisoformat(exp_str)
        except ValueError:
            continue

        days_to_expiry = (exp_date - TODAY).days
        stock = row["current_stock"]

        if 0 < days_to_expiry <= EXPIRY_DAYS and stock >= EXPIRY_MIN_STOCK:
            pid = row["product_id"]
            sid = row["store_id"]
            flags.append({
                "flag":           "expiry_risk",
                "store_id":       sid,
                "store_name":     smap.get(sid, {}).get("store_name", sid),
                "product_id":     pid,
                "product_name":   pmap.get(pid, {}).get("product_name", pid),
                "current_stock":  stock,
                "expiry_date":    exp_str,
                "days_to_expiry": days_to_expiry,
                "assumption":     f"Stock expires in {days_to_expiry} days with {stock} units remaining",
                "action":         "Prioritise selling this batch — consider discount or inter-branch move.",
            })

    return sorted(flags, key=lambda x: x["days_to_expiry"])


# ── Master function ────────────────────────────────────────────────────────────

def run_all_checks() -> dict:
    """
    Run all 4 checks and return a structured attention report.
    This is what the frontend 'Today's Attention' panel displays.
    """
    stockouts = check_stockout_risk()
    dead      = check_dead_stock()
    spikes    = [f for f in check_sales_spikes_and_drops() if f["flag"] == "sales_spike"]
    drops     = [f for f in check_sales_spikes_and_drops() if f["flag"] == "sales_drop"]
    expiry    = check_expiry_risk()

    return {
        "generated_on":   TODAY_STR,
        "stockout_risk":  stockouts,
        "dead_stock":     dead,
        "sales_spikes":   spikes,
        "sales_drops":    drops,
        "expiry_risk":    expiry,
        "summary": {
            "stockout_count": len(stockouts),
            "dead_stock_count": len(dead),
            "spike_count":    len(spikes),
            "drop_count":     len(drops),
            "expiry_count":   len(expiry),
        }
    }
