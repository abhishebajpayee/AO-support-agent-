"""
Mocked tools the agent can call via function/tool calling.
Swap the mock data for a real DB or API call when you extend this.
"""
import random

# Fake order database
_MOCK_ORDERS = {
    "ORD1001": {"status": "delivered", "item": "Wireless Mouse", "amount": 799, "payment": "prepaid"},
    "ORD1002": {"status": "in_transit", "item": "Bluetooth Speaker", "amount": 1999, "payment": "cod"},
    "ORD1003": {"status": "delivered", "item": "USB-C Hub", "amount": 1299, "payment": "prepaid"},
}


def order_lookup(order_id: str) -> dict:
    """Look up an order by ID. Raises if not found (used to test the
    agent's recovery/escalation flow)."""
    order = _MOCK_ORDERS.get(order_id.upper())
    if not order:
        raise ValueError(f"Order {order_id} not found")
    return order


def refund_calculator(order_id: str) -> dict:
    """Calculate refund amount based on order and return/refund policy."""
    order = order_lookup(order_id)  # will raise if order doesn't exist
    amount = order["amount"]
    if order["payment"] == "prepaid":
        return {"refund_amount": amount, "method": "original payment method"}
    else:
        return {"refund_amount": amount, "method": "store credit"}


def create_ticket(issue: str, order_id: str | None = None) -> dict:
    """Create a support ticket for issues the agent can't resolve itself."""
    ticket_id = f"TCK{random.randint(10000, 99999)}"
    return {
        "ticket_id": ticket_id,
        "issue": issue,
        "order_id": order_id,
        "status": "open",
    }


# Tool schema in Anthropic's tool-use format, so the LLM knows what it can call
TOOL_SCHEMAS = [
    {
        "name": "order_lookup",
        "description": "Look up the status, item, amount and payment method of an order by its order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. ORD1001"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "refund_calculator",
        "description": "Calculate the refund amount and method for a given order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Create a support ticket when the issue cannot be resolved automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "issue": {"type": "string", "description": "Short description of the unresolved issue"},
                "order_id": {"type": "string", "description": "Optional related order ID"},
            },
            "required": ["issue"],
        },
    },
]

TOOL_FUNCTIONS = {
    "order_lookup": order_lookup,
    "refund_calculator": refund_calculator,
    "create_ticket": create_ticket,
}
