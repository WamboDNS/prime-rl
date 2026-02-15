"""Scoring functions for SDFT completions.

Ported from SDPO: verl/utils/reward_score/feedback/
"""

import json
import re
from collections import Counter


def score_completion(completion: str, answer: str, kind: str | None = None) -> dict:
    """Score a completion against a ground truth answer.

    Returns:
        Dict with keys: score (float), pred (str), feedback (str | None).
    """
    if kind == "mcq":
        return _score_mcq(completion, answer)
    elif kind == "tooluse":
        return _score_tooluse(completion, answer)
    elif kind == "code":
        return _score_code_format_only(completion, answer)
    else:
        return _score_exact_match(completion, answer)


def _extract_xml_answer(text: str) -> str | None:
    """Extract answer from <answer>X</answer> tags."""
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return None
    return last_part.split("</answer>")[0].strip()


def _score_mcq(completion: str, answer: str) -> dict:
    """Score MCQ completion by extracting answer tag and comparing."""
    pred = _extract_xml_answer(completion)
    if pred is None:
        # Fallback: look for last standalone letter A-D
        match = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", completion)
        pred = match.group(1) if match else ""

    score = 1.0 if pred.upper() == answer.upper() else 0.0
    return {"score": score, "pred": pred, "feedback": None}


def _score_tooluse(completion: str, answer: str) -> dict:
    """Score tooluse completion by comparing actions and inputs."""
    # Extract predicted actions
    pred_actions = re.findall(r"Action:\s*(\w+)", completion)
    pred_inputs_raw = re.findall(r"Action Input:\s*(\{.*?\})", completion, re.DOTALL)
    pred_inputs = {}
    for raw in pred_inputs_raw:
        try:
            pred_inputs.update(json.loads(raw))
        except json.JSONDecodeError:
            pass

    # Parse ground truth
    try:
        gt_list = json.loads(answer) if isinstance(answer, str) else answer
    except json.JSONDecodeError:
        return {"score": 0.0, "pred": str(pred_actions), "feedback": f"Could not parse ground truth: {answer}"}

    gt_actions = []
    gt_inputs = {}
    for item in gt_list:
        gt_actions.append(item.get("Action", ""))
        try:
            inp = item.get("Action_Input", item.get("Action Input", "{}"))
            if isinstance(inp, str):
                inp = json.loads(inp)
            gt_inputs.update(inp)
        except (json.JSONDecodeError, TypeError):
            pass

    actions_match = Counter(pred_actions) == Counter(gt_actions)
    inputs_match = pred_inputs == gt_inputs
    score = 1.0 if actions_match and inputs_match else 0.0

    feedback = None
    if not actions_match:
        feedback = f"Expected actions {gt_actions}, got {pred_actions}"
    elif not inputs_match:
        feedback = f"Expected inputs {gt_inputs}, got {pred_inputs}"

    return {"score": score, "pred": f"Actions: {pred_actions}, Inputs: {pred_inputs}", "feedback": feedback}


def _score_code_format_only(completion: str, answer: str) -> dict:
    """Score code completion - format check only (no execution).

    For actual code execution scoring, use an external evaluator and
    pass feedback via the reprompt config.
    """
    code_blocks = re.findall(r"```\w*\n(.*?)```", completion, re.DOTALL)
    has_code = len(code_blocks) > 0
    return {
        "score": 0.0,  # Can't verify without execution
        "pred": code_blocks[0][:500] if has_code else "",
        "feedback": "Code extraction only - no execution available" if has_code else "No code block found",
    }


def _score_exact_match(completion: str, answer: str) -> dict:
    """Fallback: check if the answer appears in the completion."""
    # Try XML answer extraction first
    pred = _extract_xml_answer(completion)
    if pred is not None:
        score = 1.0 if pred.strip() == answer.strip() else 0.0
        return {"score": score, "pred": pred, "feedback": None}

    # Check if answer appears in completion
    score = 1.0 if answer.strip() in completion else 0.0
    return {"score": score, "pred": completion[-200:], "feedback": None}
