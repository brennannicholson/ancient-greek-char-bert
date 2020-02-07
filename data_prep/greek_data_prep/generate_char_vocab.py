"""Creates a vocab file in the format expected by BERT. This is needed even with a custom tokenizer if we want to manipulate the special tokens (which are not individual characters."""
SPECIAL_CHARS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def create_vocab():
    data_path = "char_BERT_dataset.txt"
    output_file = "greek_char_vocab.txt"

    print("Generating character vocab...")

    with open(data_path, "r") as fp:
        data = fp.read()

    chars = sorted(list(set(data)))

    with open(output_file, "w") as fp:
        for s in SPECIAL_CHARS:
            fp.write(s + "\n")
        fp.write("_\n")
        for c in chars:
            if c not in [" ", "\n"]:
                fp.write(c + "\n")


if __name__ == "__main__":
    create_vocab()
