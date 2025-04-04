import pandas


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
