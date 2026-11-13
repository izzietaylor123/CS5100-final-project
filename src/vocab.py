"""
Every image in PrIMuS has a .semantic file that lists the
symbols in that image, in left-to-right order, for example:

    clef-G2    timeSignature-2/4    note-A4_quarter    barline

Each one of these symbols is called a "token". NNs can only work
with numbers, so this file gives every unique token its own number:

    <blank>          -> 0    (reserved for CTC, never used for a real symbol)
    barline          -> 1
    clef-G2          -> 2
    note-A4_quarter  -> 3
    ...

Two lookups are built:
    token_to_id : symbol -> number   (used by dataset.py to convert labels)
    id_to_token : number -> symbol   (used after training, to turn the model's
                                      number output back into symbols)

read_tokens() reads the symbols from one label file
build_vocab() scans many label files and builds the two lookups
"""

from pathlib import Path


def read_tokens(label_path):
    """Reads one .semantic file and return its symbols as a list"""
    text = Path(label_path).read_text()
    text = text.strip()
    tokens = text.split("\t")
    return tokens


def build_vocab(label_paths):
    """Give every unique symbol a number. Number 0 is saved for the CTC blank."""

    # collect every unique symbol
    all_tokens = set()
    for path in label_paths:
        tokens = read_tokens(path)
        for token in tokens:
            all_tokens.add(token)

    # sort them so the numbers are the same every run
    sorted_tokens = sorted(all_tokens)

    # assign numbers to each symbol starting at 1
    token_to_id = {}
    number = 1
    for token in sorted_tokens:
        token_to_id[token] = number
        number = number + 1

    # build the reverse lookup (e.g., number to symbol)
    id_to_token = {0: "<blank>"}
    for token in token_to_id:
        number = token_to_id[token]
        id_to_token[number] = token

    return token_to_id, id_to_token
