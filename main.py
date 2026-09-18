"""
CLI chat loop for the Agentic Support Assistant.
Run: python main.py
Requires: export ANTHROPIC_API_KEY="your-key"
"""
from dotenv import load_dotenv
load_dotenv()

from src.agent import run_agent


def main():
    print("Agentic Support Assistant (type 'quit' to exit)")
    print("Try asking about shipping, returns, or an order like ORD1001 / ORD1002 / ORD1003\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        answer = run_agent(query)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
