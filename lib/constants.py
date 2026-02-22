from __future__ import annotations

import hashlib
from typing import Dict, List

CANONICAL_COLUMNS: List[str] = [
    "Scenario",
    "Site ID",
    "Product Group",
    "Brand",
    "Supplier",
    "Assigned Carrier",
    "Home Terminal",
    "New Terminal",
    "Home TCN",
    "New TCN",
    "Latitude",
    "Longitude",
    "Daily Volume (gal)",
    "Monthly Volume (gal 30 day)",
    "Daily Volume (bbl)",
    "Monthly Volume (bbl 30 day)",
    "Freight Cost (30 days)",
    "Product Cost (30 days)",
    "Total Cost (30 days)",
    "Primary Freight Rate",
    "New Freight Rate",
    "Primary Supply Rate",
    "New Supply Rate",
]

REQUIRED_COLUMNS = [
    "Scenario",
    "Site ID",
    "Product Group",
    "Home Terminal",
    "New Terminal",
    "New TCN",
    "Total Cost (30 days)",
    "Freight Cost (30 days)",
    "Product Cost (30 days)",
]

EXPLORER_COLUMNS = [
    "Scenario",
    "Site ID",
    "Brand",
    "Product Group",
    "Home Terminal",
    "New Terminal",
    "Home TCN",
    "New TCN",
    "Primary Freight Rate",
    "New Freight Rate",
    "Primary Supply Rate",
    "New Supply Rate",
    "Freight Cost (30 days)",
    "Product Cost (30 days)",
    "Total Cost (30 days)",
]


PRODUCT_GROUPS = ["Regular", "Premium", "Diesel"]
ANNUALIZATION_FACTOR = 365 / 30
GALLONS_PER_BBL = 42

# Normalized key -> canonical column
COLUMN_VARIANTS: Dict[str, str] = {
    "scenario": "Scenario",
    "site id": "Site ID",
    "product group": "Product Group",
    "brand": "Brand",
    "supplier": "Supplier",
    "assigned carrier": "Assigned Carrier",
    "carrier": "Assigned Carrier",
    "home terminal": "Home Terminal",
    "home terminal ": "Home Terminal",
    "new terminal": "New Terminal",
    "terminal": "New Terminal",
    "home tcn": "Home TCN",
    "new tcn": "New TCN",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "long": "Longitude",
    "daily volume (gal)": "Daily Volume (gal)",
    "daily volume gal": "Daily Volume (gal)",
    "monthly volume (gal 30 day)": "Monthly Volume (gal 30 day)",
    "monthly volume (gal)": "Monthly Volume (gal 30 day)",
    "daily volume(b)": "Daily Volume (bbl)",
    "daily volume (bbl)": "Daily Volume (bbl)",
    "daily volume (b)": "Daily Volume (bbl)",
    "monthyl volume (30 day b)": "Monthly Volume (bbl 30 day)",
    "monthly volume (30 day b)": "Monthly Volume (bbl 30 day)",
    "monthly volume (bbl 30 day)": "Monthly Volume (bbl 30 day)",
    "freight cost (30 days)": "Freight Cost (30 days)",
    "product cost (30 days)": "Product Cost (30 days)",
    "total cost (30 days)": "Total Cost (30 days)",
    "primary freight rate": "Primary Freight Rate",
    "new freight rate": "New Freight Rate",
    "primary supply rate": "Primary Supply Rate",
    "new supply rate": "New Supply Rate",
}

COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def terminal_color(terminal: str) -> str:
    key = (terminal or "Unassigned").strip().lower().encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    idx = int(digest[:8], 16) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]
