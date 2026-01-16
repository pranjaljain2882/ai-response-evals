import yaml
from chatbot import get_chatbot_response
from rubric_judge import rubric_judge


NUM_TRIALS = 3
MIN_PASS_RATIO = 0.66


def load_testcase(path: str) -> dict:
    """Load a YAML test case definition."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_case():
    """
    Run the same prompt multiple times to account for LLM non-determinism.
    Test passes if the model succeeds in a majority of runs.
    """
    testcase = load_testcase("testcases/payment_issue.yaml")

    successful_runs = 0
    evaluations = []

    print(f"\nRunning robustness test ({NUM_TRIALS} trials)")

    for run_number in range(1, NUM_TRIALS + 1):
        evaluation = run_single_evaluation(testcase)
        evaluations.append(evaluation)

        log_evaluation(run_number, evaluation)

        if evaluation["pass"]:
            successful_runs += 1

    pass_ratio = successful_runs / NUM_TRIALS
    log_summary(successful_runs, NUM_TRIALS, pass_ratio)

    assert pass_ratio >= MIN_PASS_RATIO, (
        f"Flaky behavior detected: passed {successful_runs}/{NUM_TRIALS} runs"
    )


def run_single_evaluation(testcase: dict) -> dict:
    """Execute one model call and judge its response."""
    response = get_chatbot_response(testcase["prompt"])

    return rubric_judge(
        prompt=testcase["prompt"],
        response=response,
        expected=testcase["expected_behavior"],
        rubric=testcase["rubric"],
        threshold=testcase["pass_threshold"],
    )


def log_evaluation(run_number: int, result: dict) -> None:
    """Print detailed evaluation results for a single run."""
    print(f"\nRun {run_number}")
    print(f"Reasoning: {result['reasoning']}")
    print("Scores:")
    for rubric_name, score in result["scores"].items():
        print(f"  {rubric_name}: {score}")
    print(f"Final Score: {result['final_score']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Pass: {result['pass']}")
    print(f"Verdict: {result['verdict']}")


def log_summary(passes: int, total: int, ratio: float) -> None:
    """Print aggregated test summary."""
    print("\n================ SUMMARY ================")
    print(f"Final Pass Rate: {ratio:.2%} ({passes}/{total})")
    print("========================================")
