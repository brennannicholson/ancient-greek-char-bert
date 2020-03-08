"""Uses a specified model to predicit missing characters from a file containing texts. """

from char_bert.greek_char_bert.predict import (
    MLMPredicter,
    replace_square_brackets,
    sentences_to_dicts,
)
from char_bert.greek_char_bert.run_eval import convert_masking
from data_prep.greek_data_prep.clean_data import clean_texts, CHARS_TO_REMOVE, CHARS_TO_REPLACE
from cltk.corpus.utils.formatter import cltk_normalize
import re
import argparse


def predict_from_file(path, model, decoding, align, step_len, beam_width=3):
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
        if decoding == "sequential":
            result = model.predict_sequentially(dicts=dicts)
        elif decoding == "beam":
            result = model.beam_search_predictions(dicts=dicts,beam_width=beam_width)
        else:
            result = model.predict(dicts=dicts)
        results.append(result)
    # output results
    for result in results:
        if decoding != "beam":
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
        else:
            for texts,p_vals in result:
                for i,t in enumerate(texts):
                    print("Candidate %i with log prob: %.5f || Text: %s" %(i,p_vals[i],t.replace("_", " ")))
                # TODO bring together text that was splitted apart
                # this seems quite complex.
                # Hacky solution: compare the return p vals per splitted text
                # when the masked symbols are not fully covered by a splitted text passage, take the predictions where it is complete

if __name__ == "__main__":
    model_path = "models/greek_char_BERT"
    file = "data/prediction_test.txt"
    #file = "data/PH2334_masked.txt"
    decoding = "beam" #"beam" "sequential"
    align = False
    step_len = 500
    beam_width = 10
    model = MLMPredicter.load(model_path, batch_size=32)
    model.processor.max_seq_len = 512
    predict_from_file(file, model, decoding, align, step_len, beam_width)
