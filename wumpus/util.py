def string_to_int(input: str):
    """
    Parses a string to an integer list that has 128 entries
    TODO: Are 128 chars enough? Too much?
    :param input: initial string
    :return: the integer list
    """
    if len(input) < 128:
        raw_list = input + ''.join([' ' for x in range(128 - len(input))])
    elif len(input) > 128:
        # hopefully this doesn't happen
        raw_list = input[:len(input)-128]
    return [(ord(x) / 256) for x in raw_list]
