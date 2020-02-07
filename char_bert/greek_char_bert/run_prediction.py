"""Uses a specified model to predicit missing characters from a file containing texts. """

from greek_char_bert.predict import (
    MLMPredicter,
    replace_square_brackets,
    sentences_to_dicts,
)
from greek_char_bert.run_eval import convert_masking
from greek_data_prep.clean_data import clean_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE
from cltk.corpus.utils.formatter import cltk_normalize
import re
import argparse


def predict_from_file(path, model, use_sequential_decoding, align, step_len):
    """Runs prediction using the model on the texts located in the file given in path."""
    max_seq_len = model.processor.max_seq_len - 2
    with open(path, "r") as fp:
        texts = fp.read().splitlines()
    # prepare texts
    texts = clean_texts(texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE)
    texts = [cltk_normalize(replace_square_brackets(t)) for t in texts]
    texts = [t.replace(" ", "_") for t in texts]
    results = []
    # break up long texts
    for t in texts:
        sequences = []
        if len(t) >= max_seq_len:
            if not (step_len and step_len < max_seq_len):
                step_len = round(max_seq_len / 2)
            # for i in range(0, len(t) - step_len, step_len):
            for i in range(0, len(t), step_len):
                seq = t[i : i + max_seq_len]
                sequences.append(seq)
        else:
            sequences.append(t)
        sequences = convert_masking(sequences)
        dicts = sentences_to_dicts(sequences)
        if use_sequential_decoding:
            result = model.predict_sequentially(dicts=dicts)
        else:
            result = model.predict(dicts=dicts)
        results.append(result)
    # output results
    for result in results:
        nb_of_masks = 0  # needed to proper alignment
        for i, res in enumerate(result):
            prediced_text = res["predictions"]["text_with_preds"].replace("_", " ")
            masked_text = res["predictions"]["masked_text"].replace("_", " ")
            if align:
                if not step_len:
                    step_len = round(max_seq_len / 2)
                # an approximate alignment is calculated by shifting each line by step_len + 2 * the number of masks in the overlaping portion of the previous prediction (to take into account the square brackets which are added around each prediction)
                print(" " * (step_len * i + (2 * nb_of_masks)) + prediced_text)
                nb_of_masks += len(re.findall(r"#+", masked_text[:step_len]))
            else:
                print(res["predictions"]["text_with_preds"].replace("_", " "))


if __name__ == "__main__":
    model_path = "../../models/greek_char_BERT"
    file = "../../data/prediction_test.txt"

    parser = argparse.ArgumentParser(
        description="Run prediction on a file containing one or more texts with missing characters."
    )
    parser.add_argument(
        "-f",
        "--file",
        default=file,
        help="The file with the texts. Missing characters are indicated by enclosing the number of full stops corresponding to the number of missing characters with square brackets, e.g.: μῆνιν ἄ[...]ε θεὰ Πηληϊάδεω Ἀχ[...]ος",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        default=model_path,
        help="The path to the saved model to use for prediction.",
    )
    parser.add_argument(
        "-s",
        "--sequential_decoding",
        default=False,
        action="store_true",
        help="Use sequential decoding (warning: very slow, especially without a GPU).",
    )
    parser.add_argument(
        "-a",
        "--align",
        default=False,
        action="store_true",
        help="Align output from long texts which are broken up into multiple parts.",
    )
    parser.add_argument(
        "--step_len",
        type=int,
        help="The step length to use when handling texts longer than the model's maximum input length.",
    )
    args = parser.parse_args()

    file = args.file
    model_path = args.model_path
    use_sequential_decoding = args.sequential_decoding
    align = args.align
    step_len = args.step_len
    model = MLMPredicter.load(model_path, batch_size=32)

    predict_from_file(file, model, use_sequential_decoding, align, step_len)
