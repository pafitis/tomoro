import pandas as pd
import re


def table_to_sentences(context_table: list[list[str]]) -> str:
    """ Converts a table into sentences for better parsing into prompts

    Assumes table is given as a list of rows. 
        Each row is a list of values
    Assumes that the first row is 
        [column name 1, column name 2, ..., column name n] 

    Args:
        context_table (list[list[str]]): Table in list form. 
            Expects each row to appear as a list of values

    Returns:
        str: String forming the content of the table
            [Row i]: Col_1: Val_1; Col_2: Val_2; ...
    """
    cols = context_table[0]

    sentences = []
    # Iterate over the rows and create a sentence for each row
    for ix, row_vals in enumerate(context_table[1:]):
        contents = [f'{col}: {val}' for col, val in zip(cols, row_vals)]
        sentence = f"[Row {ix}] " + "; ".join(contents)

        sentences.append(sentence)

    return "\n".join(sentences)


def text_cleaner(text_list: list[str]) -> str:
    """Used to clean the pre- and post- context
    Removes incorrect whitespace
    Removes artifact from message transmission

    Args:
        text_list (list[str]): input text, as a list of strings

    Returns:
        str: cleans and flattens the contents
    """

    def _text_cleaner(_text: str) -> str:
        _text = _text.replace(" , ", ", ")
        _text = _text.replace(" .", ".")
        _text = _text.replace(" ; ", "; ")

        # There are entries with %%transmsg which appears to be
        # an artifact from how this text was sent/sourced
        # These always appear at the end of the context;
        # remove this
        if r"%%transmsg***" in _text:
            _text = _text.split(r"%%transmsg")[0]

        return _text

    # It appears that each entry in the list of text
    # ends with a ' .' instead of '.'
    text_list = [_text_cleaner(text)
                 for text in text_list if text.endswith(" .")]

    return " ".join(text_list)


def get_context(entry: pd.Series, return_short=True) -> str:
    """Extracts context from a data entry
    Can either add the pre- and post- text onto the table
    or choose to only provide the table

    Args:
        entry (pd.Series): Single entry from the training data
        return_short (bool, optional): Determines if the pre- and post- texts 
            are added onto the table. Defaults to True.

    Returns:
        str: Flattened string with the provided context
    """

    pre_context = entry.pre_text
    post_context = entry.post_text

    clean_pre_text = text_cleaner(pre_context)
    clean_post_text = text_cleaner(post_context)

    table = entry.table
    table_sentences = table_to_sentences(table)

    full_context = (
        f"Context:\n\n{clean_pre_text}\n\n"
        f"{table_sentences}\n\n{clean_post_text}\n\nAnswer:\n\n"
    )

    if return_short:
        short_context = f'Context:\n\n{table_sentences}\n\nAnswer:\n\n'
        return short_context, full_context
    else:
        return full_context


def process_data_entry(entry: pd.Series) -> dict:
    """Wrapper to process training/test data into an easily digestible format
    Creates a dictionary for each entry with specific useful information

    Args:
        entry (pd.Series): Single entry from the data provided

    Returns:
        dict: Output dictionary with information needed for modelling
    """
    if isinstance(entry["qa"], dict):
        question_type = "simple"
        question = entry.qa.get("question")
        answer = entry.qa.get("answer")
    else:
        question_type = "hybrid"
        question = [entry.qa_0.get("question"), entry.qa_1.get("question")]
        answer = [entry.qa_0.get("answer"), entry.qa_1.get("answer")]

    short_context, full_context = get_context(entry, return_short=True)

    return {
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "short_context": short_context,
        "full_context": full_context,
        "step_by_step_questions": entry.annotation.get("dialogue_break"),
        "step_by_step_answers": entry.annotation.get("exe_ans_list"),
        "step_by_step_split": entry.annotation.get("qa_split"),
    }


def process_data_table(table: pd.DataFrame) -> pd.Series:
    """Wrapper to process all entries

    Args:
        table (pd.DataFrame): Dataframe with all entries to be processed

    Returns:
        pd.Series: Collection of dictionaries
    """
    return table.apply(process_data_entry, axis=1)
