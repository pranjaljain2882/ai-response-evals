from json_utils import extract_json
from openrouter_client import call_llm
from typing import List, Dict, Any

def rubric_judge(
    prompt: str,
    response: str,
    expected: str,
    rubric: List[Dict[str, str]],
    threshold: float,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Evaluate an AI assistant's response against a rubric using an LLM as a judge.

    Args:
        prompt (str): The user's prompt.
        response (str): The assistant's response.
        expected (str): The expected behavior/answer.
        rubric (List[Dict]): List of rubric criteria with 'name' and 'description'.
        threshold (float): Minimum normalized score to pass the test.
        max_retries (int): Number of times to retry LLM call if JSON parsing fails.

    Returns:
        Dict[str, Any]: Evaluation including scores, final_score, pass/fail, reasoning, and verdict.
    """

    # Build rubric text for the LLM
    rubric_text = "\n".join(f"- {r['name']}: {r['description']}" for r in rubric)

    # LLM evaluation prompt
    judge_prompt = f"""
You are an expert AI evaluator.

Evaluate the assistant response according to the rubric below.

User Prompt:
{prompt}

Assistant Response:
{response}

Expected Behavior:
{expected}

Rubric:
{rubric_text}

Return ONLY valid JSON with this structure:
{{
  "reasoning": "...",
  "scores": {{
    "<rubric_name>": <int>
  }},
  "verdict": "Clearly explain summary why the response passed or failed."
}}
No markdown, no extra text outside JSON.
"""

    last_error = None

    # Retry mechanism to handle occasional LLM misformatting
    for attempt in range(max_retries + 1):
        output = call_llm(judge_prompt)

        try:
            data = extract_json(output)
            break  # Successfully parsed JSON
        except Exception as e:
            last_error = e
    else:
        # LLM failed to return valid JSON even after retries
        return {
            "scores": {r["name"]: 0 for r in rubric},
            "final_score": 0.0,
            "pass": False,
            "threshold": threshold,
            "verdict": "Judge failed to return valid JSON",
            "reasoning": str(last_error)
        }

    # Normalize scores (assuming each rubric category max = 10)
    scores = data["scores"]
    max_score = 10 * len(scores)
    total_score = sum(scores.values())
    final_score = total_score / max_score

    passed = final_score >= threshold

    return {
        "scores": scores,
        "final_score": round(final_score, 3),
        "pass": passed,
        "threshold": threshold,
        "verdict": data.get("verdict", "No verdict provided"),
        "reasoning": data.get("reasoning", "No reasoning provided")
    }
