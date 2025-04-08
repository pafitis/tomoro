import numpy as np
import pandas as pd

from re import finditer


def find_answer(input_string: str) -> str:
    """Finds the answer in a provided LLM output
    Expects answer to be either enclosed in \boxed or backticks (`)

    Args:
        text (str): LLM output

    Returns:
        str: Answer string
    """

    pattern = r"\\boxed\{(.*?)\}|```(.*?)```|`(.*?)`"
    input_string = input_string.replace(' ', '').replace('\n', '')

    matches = list(finditer(pattern, input_string))
    if not matches:
        return ""
    last_match = matches[-1]
    # Extract content from the matched group (either \boxed{} or backticks)
    return (
        last_match.group(1)
        if last_match.group(1) is not None
        else last_match.group(2)
    )


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

    _ground_truth = cleanup_answer(ground_truth)

    difference = np.abs(float(prediction) - float(_ground_truth))
    diff_percentage_fix = difference

    if "%" in str(ground_truth) and str(prediction).startswith("0."):
        prediction = float(prediction) * 100
        diff_percentage_fix = np.abs(float(prediction) - float(_ground_truth))

    return difference, diff_percentage_fix


def evaluate_experiment(experiment_results, delta=0.05):
    metrics = []
    correct_counter = 0
    correct_counter_fix = 0
    total_preds = 0

    if isinstance(experiment_results, pd.DataFrame):
        experiment_results = experiment_results.to_dict()

    for x, entry in experiment_results.items():
        entry_metrics = []
        for interim_step in entry:

            total_preds += 1
            model_pred = interim_step.get("model_response")
            annot_answer = interim_step.get("annotator_answer")

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
                    "complete_response": interim_step.get("complete_response"),
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

        metrics.append(entry_metrics)

    print(
        f"""Experiment:
            % correct: {np.round(correct_counter/total_preds, 3)}
            % correct (percentage fix): {
            np.round(correct_counter_fix/total_preds, 3)
        }
        """
    )
    return metrics
