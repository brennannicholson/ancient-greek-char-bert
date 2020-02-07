import os
import random

from farm.data_handler.utils import _download_extract_downstream_data, logger
from tqdm import tqdm


def read_docs_from_txt(filename, delimiter="", encoding="utf-8"):
    """Note: this is a slightly modified version of farm/data_handler/utils.py which removes a minor bug. - BN"""
    """Original documentation: Reads a text file with one sentence per line and a delimiter between docs (default: empty lines) ."""
    if not (os.path.exists(filename)):
        _download_extract_downstream_data(filename)
    all_docs = []
    doc = []
    corpus_lines = 0
    with open(filename, "r", encoding=encoding) as f:
        for line_num, line in enumerate(
            tqdm(f, desc="Loading Dataset", total=corpus_lines)
        ):
            line = line.strip()
            if line == delimiter:
                if len(doc) > 0:
                    all_docs.append({"doc": doc})
                    doc = []
                else:
                    logger.warning(
                        f"Found empty document in file (line {line_num}). "
                        f"Make sure that you comply with the format: "
                        f"One sentence per line and exactly *one* empty line between docs. "
                        f"You might have multiple subsequent empty lines."
                    )
            else:
                doc.append(line)

        # the following line has been modified to remove a check which prevented single document datasets.
        if len(doc) > 0:
            all_docs.append({"doc": doc})
    return all_docs


def char_mlm_mask_random_words(tokens, max_predictions_per_seq=38, masked_lm_prob=0.20):
    """
    Masks tokens using a algorithm designed for character level masking. The idea is to ensure groups and dense clusters of characters as masked so that the task won't be too easy.

    :param tokens: tokenized sentence.
    :type tokens: [str]
    :param max_predictions_per_seq: maximum number of masked tokens
    :type max_predictions_per_seq: int
    :param masked_lm_prob: probability of masking a token
    :type masked_lm_prob: float
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """

    # we don't want to mask the special tokens
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        else:
            cand_indices.append(i)

    # save the original tokens:
    output_label = tokens.copy()

    num_to_mask = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

    num_masked = 0
    start_index = random.randint(0, len(cand_indices) - 1)
    first_pass = True

    # 2. Mask tokens; keep looping until enough have been masked
    while num_masked < num_to_mask:
        last_token_masked = False

        for index in cand_indices:
            if num_masked >= num_to_mask:
                break
            # wait until the start index is reached
            if first_pass:
                if index < start_index:
                    continue
            original_token = tokens[index]
            # skip tokens that are already masked.
            if original_token == "[MASK]":
                last_token_masked = True
                continue
            prob = random.random()
            # if the prior token is masked, the chance to mask the next one is 70%
            # last token masked, 70% chance to mask
            if last_token_masked:
                if prob <= 0.70:
                    tokens[index] = "[MASK]"
                    num_masked += 1
                else:
                    last_token_masked = False
            # last token not masked, 5% chance to mask
            elif prob <= 0.05:
                tokens[index] = "[MASK]"
                num_masked += 1
                last_token_masked = True
        first_pass = False
    return tokens, output_label


def get_sentence_pair_with_placeholder(doc, idx):
    """
    Simply returns a placeholder in place of the second sentence (which would usually be randomly selected for the next sentence prediction task.
    """
    sent_1 = doc[idx]
    sent_2 = "_"

    label = False

    assert len(sent_1) > 0
    assert len(sent_2) > 0
    return sent_1, sent_2, label
