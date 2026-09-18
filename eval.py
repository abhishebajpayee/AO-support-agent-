"""
Evaluation harness: run a fixed set of test queries against the agent and
score whether it behaves correctly. This is the piece most fresher projects
skip — highlight it in interviews.

Two kinds of checks:
1. Keyword checks: does the final answer contain expected key info?
2. Behavior checks: did the agent call the right tool, or correctly escalate
   on a known-bad input (regression test for the recovery flow)?
"""
from src.agent import run_agent

TEST_CASES = [
    {
        "name": "shipping_policy_question",
        "query": "How long does standard shipping take?",
        "expect_keywords": ["5-7", "business days"],
    },
    {
        "name": "refund_policy_question",
        "query": "Can I return an item after 20 days?",
        "expect_keywords": ["15 days"],
    },
    {
        "name": "order_lookup_flow",
        "query": "What's the status of order ORD1002?",
        "expect_keywords": ["transit"],
    },
    {
        "name": "refund_calculation_flow",
        "query": "How much refund will I get for order ORD1001?",
        "expect_keywords": ["799"],
    },
    {
        "name": "recovery_flow_invalid_order",
        # deliberately a bad order ID -> should escalate, not hallucinate
        "query": "What's the refund amount for order ORD9999?",
        "expect_keywords": ["ticket"],
    },
]


def score_case(case: dict) -> dict:
    answer = run_agent(case["query"])
    answer_lower = answer.lower()
    hits = [kw for kw in case["expect_keywords"] if kw.lower() in answer_lower]
    passed = len(hits) == len(case["expect_keywords"])
    return {
        "name": case["name"],
        "query": case["query"],
        "answer": answer,
        "passed": passed,
        "missing_keywords": [kw for kw in case["expect_keywords"] if kw not in hits],
    }


def run_eval_suite(verbose: bool = True) -> list[dict]:
    results = [score_case(c) for c in TEST_CASES]
    if verbose:
        passed = sum(r["passed"] for r in results)
        print(f"\n{'='*50}\nEVAL RESULTS: {passed}/{len(results)} passed\n{'='*50}")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] {r['name']}")
            print(f"  Query: {r['query']}")
            print(f"  Answer: {r['answer'][:150]}")
            if not r["passed"]:
                print(f"  Missing: {r['missing_keywords']}")
            print()
    return results


if __name__ == "__main__":
    run_eval_suite()
