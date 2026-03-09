CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_mappings(charset: str):
    idx2char = {i: ch for i, ch in enumerate(charset)}
    char2idx = {ch: i for i, ch in idx2char.items()}
    return char2idx, idx2char
