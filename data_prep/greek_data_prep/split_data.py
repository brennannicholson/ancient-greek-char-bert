"""Splits the sentence tokenized data into train, dev and test sets. This script shuffles the sentences. As a result the dataset created this way can't be used for next sentence prediction."""
from greek_data_prep.utils import write_to_file
import random as rn
import math


def get_data(filename):
    with open(filename, "r") as f:
        data = f.read().splitlines()
    return data


def split_data(data, train_split, val_split, test_split):
    assert train_split + val_split + test_split == 1.0
    train = []
    val = []
    test = []

    # we want to be able to reproduce this split
    rn.seed(42)
    rn.shuffle(data)

    nb_of_texts = len(data)
    total_len = sum([len(text) for text in data])
    print("Total number of chars: " + str(total_len))
    train_target_len = math.floor(total_len * train_split)
    val_target_len = math.floor(total_len * (train_split + val_split))

    current_len = 0
    train_end = 0
    val_end = 0
    for i, text in enumerate(data):
        current_len += len(text)
        if current_len < train_target_len:
            # keep updating the train end index until current len >= train_target_len
            train_end = i
        if current_len > val_target_len:
            val_end = i
            # now that we're finished, correct the train_end index
            train_end += 1
            break
    train = data[0 : train_end + 1]
    val = data[train_end + 1 : val_end + 1]
    test = data[val_end + 1 :]
    assert len(train) + len(val) + len(test) == len(data)

    train_len = sum([len(text) for text in train])
    val_len = sum([len(text) for text in val])
    test_len = sum([len(text) for text in test])
    print(f"Train length: {train_len}")
    print(f"Val length: {val_len}")
    print(f"Test length: {test_len}")
    return train, val, test


def ninty_eight_one_one_spilt():
    filename = "char_BERT_dataset.txt"

    print("Splitting data...")
    data = get_data(filename)
    train, val, test = split_data(data, 0.98, 0.01, 0.01)
    write_to_file("train.txt", train)
    write_to_file("dev.txt", val)
    write_to_file("test.txt", test)


if __name__ == "__main__":
    ninty_eight_one_one_spilt()
