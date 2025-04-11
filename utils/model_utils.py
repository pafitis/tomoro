import asyncio
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
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
                {"role": "assistant", "content": entry.get("model_response")},
            ]
    return output


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def tenacious_model_completions(
    client: AsyncOpenAI, model_name: str, messages: list,
    temperature: float = 0.0, max_token: int = 1024,
    use_structured_outputs=False,
) -> str:
    """Wrapper function that incorporates retries attempts

    As sometimes there are connection errors due to my VPN/ provider

    Args:
        client (AsyncOpenAI): OpenAI client
        model_name (str): model name
        messages (list): input prompt
        temperature (float): defaults to 0.0
        max_token (int): defaults to 1024

    Returns:
        str: LLM output
    """

    return await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_token,
        response_format=(
            {'type': 'json_object'} if use_structured_outputs else None)
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def async_converse_llm(
    processed_data_entry: pd.Series,
    client: AsyncOpenAI,
    model_name: str,
    use_short_context: bool = False,
    use_gold_inds: bool = False,
    question_to_use: str = "step_by_step_questions",
    answers_to_use: str = "step_by_step_answers",
    example_shots: str = "",
    sys_prompt: str = "",
    use_structured_outputs: bool = False,
):

    if use_short_context and use_gold_inds:
        raise ValueError(
            'use_short_context and use_gold_inds cannot be both True')

    # Chooses which context to use
    _context = (
        processed_data_entry.get('gold_inds') if use_gold_inds
        else processed_data_entry.get("short_context") if use_short_context
        else processed_data_entry.get('full_context')
    )

    # Adds the example shots at the start of the context
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
            {"role": "assistant", "content": 'Context acknowledged, awaiting for question.'},
            *add_past_responses(history),
            {"role": "user", "content": current_question},
        ]

        # Using temp=0.0
        # per DeepSeek's docs; low temp -> deterministic
        # since math questions, we want low variation
        try:
            response = await tenacious_model_completions(
                client=client, model_name=model_name, messages=inputs,
                temperature=0.0, max_token=1024,
                use_structured_outputs=use_structured_outputs)
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


def model_completions(
    client: OpenAI, model_name: str, messages: list,
    temperature: float = 0.0, max_token: int = 1024,
    use_structured_outputs: bool = False,
) -> str:
    """Wrapper function for model completions
    Args:
        client (AsyncOpenAI): OpenAI client
        model_name (str): model name
        messages (list): input prompt
        temperature (float): defaults to 0.0
        max_token (int): defaults to 1024

    Returns:
        str: LLM output
    """

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_token,
        response_format=(
            {'type': 'json_object'} if use_structured_outputs else None)
    )


def converse_llm(
    processed_data_entry: pd.Series,
    client: OpenAI,
    model_name: str,
    use_short_context: bool = False,
    use_gold_inds: bool = False,
    question_to_use: str = "step_by_step_questions",
    answers_to_use: str = "step_by_step_answers",
    example_shots: str = "",
    sys_prompt: str = "",
    use_structured_outputs: bool = False,
):

    if use_short_context and use_gold_inds:
        raise ValueError(
            'use_short_context and use_gold_inds cannot be both True')

    # Chooses which context to use
    _context = (
        processed_data_entry.get('gold_inds') if use_gold_inds
        else processed_data_entry.get("short_context") if use_short_context
        else processed_data_entry.get('full_context')
    )

    # Adds the example shots at the start of the context
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
    for i, (current_question, current_answer) in enumerate(zip(list_of_questions, list_of_answers)):

        inputs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _context},
            {"role": "assistant", "content": 'Context acknowledged, awaiting for question.'},
            *add_past_responses(history),
            {"role": "user", "content": current_question},
        ]

        # Using temp=0.0
        # per DeepSeek's docs; low temp -> deterministic
        # since math questions, we want low variation
        try:
            response = model_completions(
                client=client, model_name=model_name, messages=inputs,
                temperature=0.0, max_token=1024,
                use_structured_outputs=use_structured_outputs)
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


if __name__ == '__main__':

    from data_utils import process_data_table
    from prompts import SYSTEM_PROMPT_V3

    raw_data = pd.read_json('data/train.json')
    all_prompts = process_data_table(raw_data)

    import os
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API'),
        base_url="https://api.deepseek.com",
    )

    test = converse_llm(
        processed_data_entry=all_prompts[0],
        client=client,
        model_name='deepseek-reasoner',
        use_gold_inds=True,
        question_to_use='step_by_step_questions',
        answers_to_use='step_by_step_answers',
        example_shots='',
        sys_prompt=SYSTEM_PROMPT_V3,
        use_structured_outputs=False,
    )
