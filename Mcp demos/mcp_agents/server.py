"""
Demo MCP server — order pricing.

Two tools, deliberately. One looks up data the model cannot possibly know,
the other does the arithmetic. The model has to work out that it needs them
in that order, which is the whole point of the demo.

Run standalone (for Claude Desktop):
    python server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Order Pricing")

# Stand-in for a real database. The model has no way to guess these.
CATALOGUE = {
    "laptop stand": 450.0,
    "mechanical keyboard": 3299.0,
    "usb-c hub": 1850.0,
    "monitor arm": 2750.0,
    "webcam": 1499.0,
}

GST_RATES = {
    "laptop stand": 18.0,
    "mechanical keyboard": 18.0,
    "usb-c hub": 18.0,
    "monitor arm": 18.0,
    "webcam": 18.0,
}


@mcp.tool()
def lookup_item(item_name: str) -> dict:
    """Look up the unit price in rupees and the GST rate for a catalogue item.

    Use this before calculating any total, because prices are not public and
    cannot be guessed. Item names are matched loosely, so "keyboard" will
    find "mechanical keyboard".

    Returns the matched item name, its unit price, and its GST percentage.
    """
    query = item_name.strip().lower()

    if query in CATALOGUE:
        match = query
    else:
        match = next((name for name in CATALOGUE if query in name), None)

    if match is None:
        return {
            "found": False,
            "error": f"No catalogue item matches {item_name!r}.",
            "available_items": sorted(CATALOGUE),
        }

    return {
        "found": True,
        "item": match,
        "unit_price": CATALOGUE[match],
        "gst_percent": GST_RATES[match],
    }


@mcp.tool()
def calculate_total(quantity: int, unit_price: float, gst_percent: float) -> dict:
    """Calculate subtotal, GST amount and final total for an order line.

    Call lookup_item first to get the unit price and GST rate. Quantity must
    be a positive whole number.

    Returns subtotal, gst_amount and total, all rounded to two decimals.
    """
    if quantity <= 0:
        return {"error": "Quantity must be greater than zero."}
    if unit_price < 0:
        return {"error": "Unit price cannot be negative."}
    if not 0 <= gst_percent <= 100:
        return {"error": "GST percent must be between 0 and 100."}

    subtotal = quantity * unit_price
    gst_amount = subtotal * gst_percent / 100

    return {
        "quantity": quantity,
        "unit_price": round(unit_price, 2),
        "subtotal": round(subtotal, 2),
        "gst_percent": gst_percent,
        "gst_amount": round(gst_amount, 2),
        "total": round(subtotal + gst_amount, 2),
        "currency": "INR",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
