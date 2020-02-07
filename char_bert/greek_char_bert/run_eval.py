"""Evaluate a model using several different datasets and produce a simple accuracy report. Accuracy is reported per mask length and both the per character accuracy and the per sequence (or per mask) accuracy are reported. Two types of evaluation data are supported, tsv files with two fields (masked sentences, answers) and with three fields (maksed sentences, original sentences, answers.). The files is currently set up to evaluate on two dataset, char-gaps, brackets and pythia. Note that the decoding style has to be changed manually."""
from greek_char_bert.predict import MLMPredicter, sentences_to_dicts
import numpy as np
import re
import random as rn


def load_data(path):
    """Loads data, skipping lines with fewer than two fields."""
    with open(path, "r") as fp:
        data = fp.read().splitlines()
        sents_and_answers = []
        for sent in data:
            sent = sent.replace(" ", "_")
            sent = sent.split("\t")
            if len(sent) < 2:
                continue
            sents_and_answers.append(sent)
    data = rn.sample(sents_and_answers, min(10, len(sents_and_answers)))
    return np.array(data)


def convert_masking(sentences):
    """Converts hash masking to internal [MASK] tokens"""
    converted_sentences = []
    for sent in sentences:
        sent = list(sent)
        for i, c in enumerate(sent):
            if c == "#":
                sent[i] = "[MASK]"
        converted_sentences.append("".join(sent))
    return converted_sentences


def evaluate_predictions_per_char(predictions, original_sentences, answers):
    """Evaluates predictions per char, returning the accuracy and lists of correct and incorrect sentences."""
    predicted_chars = []
    sentences_with_preds = []
    errors = set()
    correct_sentences = []
    total = 0
    correct = 0
    for pred in predictions:
        predicted_chars.append(pred["predictions"]["predictions"])
        sentences_with_preds.append(pred["predictions"]["text_with_preds"])
    for pred_chars, sent_with_pred, ans, orig_sent in zip(
        predicted_chars, sentences_with_preds, answers, original_sentences
    ):
        sent_correct = True
        for p, a in zip(pred_chars, ans):
            total += 1
            if p != a:
                errors.add(f"{sent_with_pred}\t{orig_sent}\t{ans}")
                sent_correct = False
            else:
                correct += 1
        if sent_correct:
            correct_sentences.append(sent_with_pred)
    acc = correct / total
    return acc, errors, correct_sentences


def prepare_data(data):
    """Prepares data with three fields."""
    sentences = data[:, 0]
    original_sents = data[:, 1]
    answers = data[:, 2]
    sentences = convert_masking(sentences)
    dicts = sentences_to_dicts(sentences)
    return dicts, sentences, original_sents, answers


def eval_char_gaps(model):
    """Iterates through the char-gap files and uses them to evaluate the model."""
    accuracies = []
    errors = []
    all_correct_sentences = []
    for i in range(1, 11):
        eval_data_path = f"../../data/eval/eval_{i}_char_gaps.tsv"
        data = load_data(eval_data_path)
        dicts, sentences, original_sents, answers = prepare_data(data)
        predictions = model.predict(dicts=dicts)
        # predictions = model.predict_sequentially(dicts=dicts)
        acc_per_char, errs, correct_sentences = evaluate_predictions_per_char(
            predictions, original_sents, answers
        )
        acc_per_seq = 1.0 - len(errs) / len(original_sents)
        acc_string = [acc_per_char, acc_per_seq]
        accuracies.append(acc_string)
        errors.append(errs)
        all_correct_sentences.append(correct_sentences)
    return accuracies, errors, all_correct_sentences


def evaluate_predictions_per_mask(predictions, original_sents, answers):
    """Evaluates per mask, returning dicts of accuracies per mask length, correct and incorrect sentences per mask length."""
    predicted_chars = []
    sentences_with_preds = []
    all_errors = {}
    correct_masks = {}
    acc_per_mask_length = {}
    for pred in predictions:
        predicted_chars.append(pred["predictions"]["predictions"])
        sentences_with_preds.append(pred["predictions"]["text_with_preds"])
    for preds, ans, sent_with_preds, orig_sent in zip(
        predicted_chars, answers, sentences_with_preds, original_sents
    ):
        chars_checked = 0
        # iterate over every mask in the sentence with predictions
        try:
            for match in re.finditer(r"\[[^[]*\]", sent_with_preds):
                mask_len = (
                    match.end() - match.start() - 2
                )  # length without the brackets
                mask_correct = True
                # initialize dict if needed
                if mask_len not in acc_per_mask_length:
                    acc_per_mask_length[mask_len] = {
                        "total_chars": 0,
                        "total_masks": 0,
                        "char_errors": 0,
                        "correct_masks": 0,
                    }
                acc_per_mask_length[mask_len]["total_masks"] += 1
                # check every predicted char against the original chars
                for i in range(chars_checked, mask_len + chars_checked):
                    chars_checked += 1
                    acc_per_mask_length[mask_len]["total_chars"] += 1
                    if preds[i] != ans[i]:
                        mask_correct = False
                        # initialize if needed
                        if mask_len not in all_errors:
                            all_errors[mask_len] = set()
                        all_errors[mask_len].add(f"{sent_with_preds}\t{orig_sent}")
                        acc_per_mask_length[mask_len]["char_errors"] += 1
                if mask_correct:
                    acc_per_mask_length[mask_len]["correct_masks"] += 1
                    # initialize if needed
                    if mask_len not in correct_masks:
                        correct_masks[mask_len] = set()
                    correct_masks[mask_len].add(f"{sent_with_preds}")
        except IndexError:
            # skip malformed samples in the pythia datasest
            continue
    return acc_per_mask_length, all_errors, correct_masks


def eval_sentences_with_brackets(model):
    """Runs evaluation on the brackets dataset."""
    eval_data_path = "../../data/eval/eval_masked_square_brackets.tsv"
    data = load_data(eval_data_path)
    dicts, sentences, original_sents, answers = prepare_data(data)
    predictions = model.predict(dicts=dicts)
    # predictions = model.predict_sequentially(dicts=dicts)
    (
        acc_per_mask,
        errs_per_mask,
        correct_sentences_per_mask,
    ) = evaluate_predictions_per_mask(predictions, original_sents, answers)
    acc_per_char, errs, _ = evaluate_predictions_per_char(
        predictions, original_sents, answers
    )
    return acc_per_char, errs, acc_per_mask, errs_per_mask, correct_sentences_per_mask


def prepare_pythia_data(data):
    """Prepares data with two fields (which so far has only been the pythia test dataset)."""
    masked_sequences = data[:, 0]
    answers = data[:, 1]
    original_sentences = []
    for masked_seq, ans in zip(masked_sequences, answers):
        # recreate the original sentence by inserting the answers into the masked sentences
        for c in ans:
            masked_seq = masked_seq.replace("#", f"{c}", 1)
        original_sentences.append(masked_seq)
    masked_sequences = convert_masking(masked_sequences)
    dicts = sentences_to_dicts(masked_sequences)
    return dicts, masked_sequences, np.array(original_sentences), answers


def eval_pythia_sequences(model):
    """Evaluate on the pythia test dataset."""
    eval_data_path = "/../../data/pythia/test.txt"
    data = load_data(eval_data_path)
    dicts, sentences, original_sents, answers = prepare_pythia_data(data)
    predictions = model.predict(dicts=dicts)
    # predictions = model.predict_sequentially(dicts=dicts)
    (
        acc_per_mask,
        errs_per_mask,
        correct_sentences_per_mask,
    ) = evaluate_predictions_per_mask(predictions, original_sents, answers)
    acc_per_char, errs, _ = evaluate_predictions_per_char(
        predictions, original_sents, answers
    )
    return acc_per_char, errs, acc_per_mask, errs_per_mask, correct_sentences_per_mask


def generate_char_gap_report(fp, char_gap_accuracies):
    """Generate a accuracy report for the char-gap set. Outputs to file fp."""
    fp.write("==== Character gaps ====\n")
    all_pc_accs = []
    all_ps_accs = []
    for i, acc in enumerate(char_gap_accuracies):
        pc_acc = acc[0] * 100
        ps_acc = acc[1] * 100
        all_pc_accs.append(pc_acc)
        all_ps_accs.append(ps_acc)
        fp.write(
            "%d char gaps: accuracy per character %.2f%%, accuracy per sequence %.2f%%\n"
            % (i + 1, pc_acc, ps_acc)
        )
    fp.write(
        "Character gaps average per character accuracy: %.2f%%\n"
        % (sum(all_pc_accs) / float(len(all_pc_accs)))
    )
    fp.write(
        "Character gaps average per mask accuracy: %.2f%%\n"
        % ((sum(all_ps_accs) / float(len(all_ps_accs))))
    )


def generate_per_mask_report(fp, title, desc, acc_per_mask_len, acc):
    """Uses per mask data to generate a report which break down the accuracy per mask length."""
    fp.write(f"==== {title} ====\n")
    all_correct_masks = 0
    all_total_masks = 0
    for length in sorted(acc_per_mask_len.keys()):
        pc_acc = (
            1.0
            - acc_per_mask_len[length]["char_errors"]
            / acc_per_mask_len[length]["total_chars"]
        )
        correct_masks = acc_per_mask_len[length]["correct_masks"]
        total_masks = acc_per_mask_len[length]["total_masks"]
        pm_acc = correct_masks / total_masks
        all_correct_masks += correct_masks
        all_total_masks += total_masks
        fp.write(
            "Mask length %d: per character accuracy: %.2f%%, per mask accuracy %.2f%% (%d/%d)\n"
            % (length, pc_acc * 100, pm_acc * 100, correct_masks, total_masks)
        )
    fp.write(f"{desc} average per character accuracy: %.2f%%\n" % (acc * 100))
    fp.write(
        f"{desc} average per mask accuracy: %.2f%%\n"
        % ((all_correct_masks / all_total_masks) * 100)
    )


def print_char_gap_specimens(fp, desc, specimens):
    """Prints correct and incorrect sentences from the char-gap, broken down by gap length."""
    for i, spec in enumerate(specimens):
        fp.write(f"==== {i + 1} {desc} ====\n")
        for s in spec:
            s = s.split("\t")
            fp.write(s[0] + "\n")
            try:
                fp.write(s[1] + "\n")
            except IndexError:
                pass
            fp.write("--------\n")


def print_pred_specimens(fp, desc, specimens):
    """Writes correct and incorrect sentences to fp, broken down by mask length."""
    for length in sorted(specimens.keys()):
        fp.write(f"==== {desc} for masks of the length {length} ====\n")
        for s in specimens[length]:
            s = s.split("\t")
            fp.write(s[0] + "\n")
            try:
                fp.write(s[1] + "\n")
            except IndexError:
                pass
            fp.write("--------\n")


if __name__ == "__main__":
    rn.seed(42)
    model_name = "greek_char_BERT"
    save_dir = f"save/{model_name}"
    accuracy_report_dir = f"../../data/eval/bert_acc_report_{model_name}.txt"
    errors_path = f"../../data/eval/bert_errors_{model_name}.txt"
    correct_sentences_path = (
        f"../../data/eval/bert_correct_sentences_{model_name}.txt"
    )
    model = MLMPredicter.load(save_dir, batch_size=32)
    (
        pythia_acc,
        _,
        pythia_acc_per_mask_len,
        pythia_errors,
        pythia_correct_sentences,
    ) = eval_pythia_sequences(model)
    (
        brackets_acc,
        _,
        brackets_acc_per_mask_len,
        brackets_errors,
        brackets_correct_sentences,
    ) = eval_sentences_with_brackets(model)
    char_gap_accuracies, char_gap_errors, char_gap_correct_sentences = eval_char_gaps(
        model
    )
    with open(accuracy_report_dir, "w") as fp:
        generate_char_gap_report(fp, char_gap_accuracies)
        generate_per_mask_report(
            fp,
            "Brackets",
            "Bracketed sentences",
            brackets_acc_per_mask_len,
            brackets_acc,
        )
        generate_per_mask_report(
            fp, "Pythia", "Pythia sentences", pythia_acc_per_mask_len, pythia_acc
        )
    with open(errors_path, "w") as fp:
        print_char_gap_specimens(fp, "character gap errors", char_gap_errors)
        print_pred_specimens(fp, "Bracketed sentence errors", brackets_errors)
        print_pred_specimens(fp, "Pythia sentence errors", pythia_errors)
    with open(correct_sentences_path, "w") as fp:
        print_char_gap_specimens(
            fp, "character gap correct sentences", char_gap_correct_sentences
        )
        print_pred_specimens(
            fp, "Bracketed correct sentences", brackets_correct_sentences
        )
        print_pred_specimens(fp, "Pythia correct sentences", pythia_correct_sentences)
