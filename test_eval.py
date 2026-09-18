"""Run with: python -m tests.test_eval"""
from src.eval import run_eval_suite

if __name__ == "__main__":
    results = run_eval_suite()
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n{len(failed)} test(s) failed.")
    else:
        print("\nAll tests passed.")
