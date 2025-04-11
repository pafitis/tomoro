import numpy as np
import pandas as pd

from re import finditer


def find_answer(input_string: str) -> str:
    """Finds the answer in a provided LLM output
    Expects answer to be either enclosed in 
        -\boxed
        -\x08 (fix; not necessary but implemented for more robustness)

    Args:
        text (str): LLM output

    Returns:
        str: Answer string
    """

    # Finds the last occurrence of \boxed{} or \08oxed{}
    pattern = r".*(?:\\boxed\{|\\x08oxed\{)(.*)\}"

    input_string = (
        input_string
        .replace(' ', '')
        .replace('\n', '')

    )

    matches = list(finditer(pattern, input_string))
    if not matches:
        return ""
    last_match = matches[-1]
    # Extract content from the matched group (either \boxed{} or backticks)
    ans = (
        last_match.group(1)
        if last_match.group(1) is not None
        else last_match.group(2)
    )

    # Clean up in case of LateX remnants
    ans = (
        ans
        .replace('\\', '')
        .replace('{', '')
        .replace('}', '')
        .replace('text', '')
        .replace('left', '')
        .replace('right', '')
    )

    return ans


def cleanup_answer(input_string: str) -> str:
    """Removes
        bad Latex formatting,
        % signs,
        whitespaces,
        newline chars

    Args:
        input (str): Answer to be cleaned

    Returns:
        (str): Cleaned answer
    """

    output = (
        str(input_string)
        .replace("%", "")
        .replace("\\", "")
        .replace(' ', '')
        .replace('\n', '')
    )

    return output


def evaluate_maths(input_string: str) -> float | int:
    """Math operation evaluator
    Uses internal python eval method to check if
    answer is executable, based on the 5 predefined functions

    Args:
        input_string (str): Answer string

    Returns:
        float|int: evaluated answer
    """

    input_string = cleanup_answer(input_string)

    # Matches the context given
    # will only accept these functions
    def add(a, b):
        return a + b

    def subtract(a, b):
        return a - b

    def multiply(a, b):
        return a * b

    def divide(a, b):
        return a / b

    def power(a, b):
        return a ** b

    # Restrict the eval to use only the allowed functions
    maths_functions = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
        "power": power
    }

    return eval(input_string, {"__builtins__": {}}, maths_functions)


def compare_answers(
        prediction: float | int, ground_truth: str
) -> tuple[float, float]:
    """Compares evaluated LLM answer versus ground truth
    Also has fix for cases where there is a mismatch of % vs absolute form
    between the format of the ground truth and prediction

    Args:
        prediction (float | int): LLM evaluated answer
        ground_truth (str): ground truth

    Returns:
        tuple[float, float]: true difference, difference after %-mismatch fix
    """

    ground_truth = cleanup_answer(ground_truth)
    pred_float = float(prediction)
    gt_float = float(ground_truth)

    # Original difference
    difference = np.abs(pred_float - gt_float)

    diff_percentage_fix = difference  # Initialize with original difference
    # Case 1: prediction is given as decimal and ground truth as percentage
    if 0.0 <= pred_float <= 1.0 and gt_float >= 1:
        adjusted_pred = pred_float * 100
        diff_pct = np.abs(adjusted_pred - gt_float)
        diff_percentage_fix = min(diff_pct, diff_percentage_fix)
    # Case 2: prediction is given as percentage and ground truth as decimal
    if 0.0 <= gt_float <= 1.0 and pred_float >= 1:
        adjusted_gt = gt_float * 100
        diff_pct = np.abs(adjusted_gt - pred_float)
        diff_percentage_fix = min(diff_pct, diff_percentage_fix)

    return difference, diff_percentage_fix


def evaluate_experiment(experiment_results, delta=0.05):
    metrics = []
    correct_counter = 0
    correct_counter_fix = 0
    total_preds = 0

    if isinstance(experiment_results, str):
        experiment_results = pd.read_json(
            experiment_results, typ='series').to_dict()

    if not isinstance(experiment_results, dict):
        experiment_results = pd.Series(experiment_results).to_dict()

    for x, entry in experiment_results.items():
        entry_metrics = []

        for interim_step in entry:
            complete_response = interim_step.get('complete_response')
            model_pred = interim_step.get("model_response")
            annot_answer = interim_step.get("annotator_answer")

            # This has now been resolved with the usage of tenacity
            # Retries are done at execution
            if 'Error encountered' in complete_response:
                continue

            # Is answer parsable?
            try:
                evaluated_pred = evaluate_maths(model_pred)

                _parsable = True
                _delta, _delta_percentage_fix = compare_answers(
                    evaluated_pred, annot_answer
                )

            except:
                evaluated_pred = None
                _parsable = False
                _delta, _delta_percentage_fix = np.inf, np.inf

            correct_counter += 1 if _delta <= delta else 0
            correct_counter_fix += 1 if _delta_percentage_fix <= delta else 0

            _close_match = 1 if _delta <= 0.05 else 0
            _close_match_perc_fix = 1 if _delta_percentage_fix <= delta else 0

            entry_metrics.append(
                {
                    "question": interim_step.get("question"),
                    "complete_response": complete_response,
                    "model_pred": model_pred,
                    "eval_ans": evaluated_pred,
                    "annot_ans": annot_answer,
                    "parsable": _parsable,
                    "close_match": _close_match,
                    "delta": _delta,
                    "close_match_perc_fix": _close_match_perc_fix,
                    "delta_perc_fix": _delta_percentage_fix,
                }
            )

            total_preds += 1

        metrics.append(entry_metrics)

    print(
        f"""Metrics:
            % correct: {np.round(correct_counter/total_preds, 3)}
            % correct (percentage fix): {
            np.round(correct_counter_fix/total_preds, 3)
        }
        """
    )
    return metrics


if __name__ == "__main__":

    # mock_answer = ['0.01', '10', '10%']
    # mock_preds = [1, 0.1, 10]

    # for x, y in zip(mock_preds, mock_answer):
    #     compare_answers(x, y)

    evaluate_experiment('results/exp3.json')
