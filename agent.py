"""
The core agent, built as a LangGraph state machine rather than a linear
chain. This is what makes it "agentic": it can branch to RAG, branch to a
tool call, retry a failed tool call, and escalate to a human node if it
still can't resolve the issue.
"""
import os
import json
from typing import TypedDict, Optional
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from src.rag import KnowledgeBase
from src.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-4-6"
MAX_TOOL_RETRIES = 2

client = Anthropic()  # reads ANTHROPIC_API_KEY from env
kb = KnowledgeBase()

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce company.
Use the knowledge base context you're given to answer policy questions.
Use tools to look up real order data, calculate refunds, or create a ticket.
If a tool fails or you don't have enough information to help the customer,
create a support ticket rather than guessing.
Be concise and helpful."""


class AgentState(TypedDict):
    query: str
    kb_context: Optional[str]
    messages: list
    tool_retry_count: int
    escalated: bool
    final_answer: Optional[str]


def retrieve_node(state: AgentState) -> AgentState:
    """Pull relevant knowledge base chunks for the query."""
    chunks = kb.retrieve(state["query"], k=3)
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    state["kb_context"] = context
    return state


def agent_node(state: AgentState) -> AgentState:
    """Call the LLM with KB context + tools available. It decides whether
    to answer directly or call a tool."""
    if not state["messages"]:
        user_content = f"Knowledge base context:\n{state['kb_context']}\n\nCustomer question: {state['query']}"
        state["messages"] = [{"role": "user", "content": user_content}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=state["messages"],
    )

    # Store assistant turn (tool_use blocks and/or text)
    state["messages"].append({"role": "assistant", "content": response.content})

    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    if tool_use_blocks:
        state["_pending_tool_calls"] = tool_use_blocks
    else:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        state["final_answer"] = "\n".join(text_blocks)

    return state


def tool_node(state: AgentState) -> AgentState:
    """Execute pending tool calls. On failure, increment retry count so the
    router can decide whether to retry or escalate."""
    tool_results = []
    any_failure = False

    for block in state.get("_pending_tool_calls", []):
        fn = TOOL_FUNCTIONS.get(block.name)
        try:
            result = fn(**block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        except Exception as e:
            any_failure = True
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {str(e)}",
                "is_error": True,
            })

    state["messages"].append({"role": "user", "content": tool_results})

    if any_failure:
        state["tool_retry_count"] = state.get("tool_retry_count", 0) + 1
    else:
        state["tool_retry_count"] = 0  # reset on success

    state.pop("_pending_tool_calls", None)
    return state


def escalate_node(state: AgentState) -> AgentState:
    """Recovery flow: after repeated tool failures, create a ticket instead
    of letting the agent hallucinate or loop forever."""
    from src.tools import create_ticket
    ticket = create_ticket(issue=f"Unresolved after {MAX_TOOL_RETRIES} tool failures: {state['query']}")
    state["escalated"] = True
    state["final_answer"] = (
        f"I wasn't able to resolve this automatically, so I've created a "
        f"support ticket ({ticket['ticket_id']}). Our team will follow up shortly."
    )
    return state


def route_after_agent(state: AgentState) -> str:
    if state.get("_pending_tool_calls"):
        return "tool"
    return "end"


def route_after_tool(state: AgentState) -> str:
    if state.get("tool_retry_count", 0) >= MAX_TOOL_RETRIES:
        return "escalate"
    return "agent"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tool": "tool", "end": END})
    graph.add_conditional_edges("tool", route_after_tool, {"agent": "agent", "escalate": "escalate"})
    graph.add_edge("escalate", END)

    return graph.compile()


def run_agent(query: str) -> str:
    app = build_graph()
    initial_state: AgentState = {
        "query": query,
        "kb_context": None,
        "messages": [],
        "tool_retry_count": 0,
        "escalated": False,
        "final_answer": None,
    }
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]
