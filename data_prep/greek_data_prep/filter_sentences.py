"""Filters certain out of the Ancient_Greek_ML dataset. The first step in prepare the Ancient_Greek_ML dataset for use with the Ancient Greek character-level BERT."""
from greek_data_prep.clean_data import write_to_file


def filter_sentences():
    """Removes sentences with square brackets an obeli marks from the dataset."""
    filtered_data = []
    square_brackets = []
    obeli = []

    print("Filtering sentences...")
    sent_tokenized_data = "Ancient_Greek_ML.txt"
    with open(sent_tokenized_data, "r") as fp:
        data = fp.read().splitlines()
    for line in data:
        if "†" in line:
            obeli.append(line)
        elif ("[" in line) or ("]" in line):
            square_brackets.append(line)
        else:
            filtered_data.append(line)
    write_to_file("char_BERT_dataset.txt", filtered_data)
    # write_to_file('eval_square_bracket.txt', square_brackets)
    # write_to_file('eval_obeli.txt', obeli)


if __name__ == "__main__":
    filter_sentences()
