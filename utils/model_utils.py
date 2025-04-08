import asyncio
import pandas as pd
from openai import AsyncOpenAI

from utils.eval_utils import find_answer


def add_past_responses(history: list[str]) -> list[dict]:
    """Function to iteratively update the LLM history
    with its own responses

    Needed in order to be able to have a conversation back-and-forth
    as well as to use intermediate steps  

    Args:
        history (list[str]): _description_

    Returns:
        list[dict]: _description_
    """
    output = []
    if len(history):
        for entry in history:
            output += [
                {"role": "user", "content": entry.get("question")},
                {"role": "assistant", "content": entry.get(
                    "complete_response")},
            ]
    return output


async def async_converse_llm(
    processed_data_entry: pd.Series,
    client: AsyncOpenAI,
    model_name: str = "deepseek-chat",
    stream: bool = False,
    use_short_context: bool = False,
    question_to_use: str = "step_by_step_questions",
    answers_to_use: str = "step_by_step_answers",
    example_shots: str = "",
    sys_prompt: str = "",
):

    # Chooses which context to use
    _context = (
        processed_data_entry.get("short_context") if use_short_context
        else processed_data_entry.get('full_context')
    )

    # Adds the example shots at the start of the context
    # TODO: consider adding this as a mod to the inputs below
    _context = example_shots + _context

    list_of_questions = processed_data_entry.get(question_to_use)
    list_of_answers = processed_data_entry.get(answers_to_use)

    # When the question/ answer is a string it must be converted
    # to a list of strings; this is to match formats,
    # when we are assessing a single question/answer vs multiple
    if isinstance(list_of_questions, str):
        list_of_questions = [list_of_questions]
    if isinstance(list_of_answers, str):
        list_of_answers = [list_of_answers]

    assert len(list_of_answers) == len(list_of_questions)

    history = []
    for current_question, current_answer in zip(list_of_questions, list_of_answers):
        inputs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _context},
            *add_past_responses(history),
            {"role": "user", "content": current_question},
        ]

        # Using temp=0.0
        # per DeepSeek's docs; low temp -> deterministic
        # since math questions, we want low variation
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=inputs,
                stream=stream,
                temperature=0.0,
                max_tokens=1024,  # restrict, as verbose
            )

            response = response.choices[0].message.content
        except Exception as e:
            response = f"Error encountered: {str(e)}"

        history.append(
            {
                "question": current_question,
                "complete_response": response,
                "model_response": find_answer(response),
                "annotator_answer": current_answer,
            }
        )

    return history
