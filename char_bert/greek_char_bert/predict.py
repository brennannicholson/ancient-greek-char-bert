"""The CharMLMPredicter class plus various functions needed to predict missing chars."""
from greek_char_bert.infer import CharMLMInferencer
from greek_char_bert.data_handler.processor import CharMLMPredProcessor
from farm.data_handler.dataloader import NamedDataLoader
from torch.utils.data.sampler import SequentialSampler
import torch
import re
import copy


class MLMPredicter(CharMLMInferencer):
    def predict(self, dicts):
        """
        This function is a simple modification of the MLMInferencer/Inferencer's run_inference method (located at farm/infer.py) except that it uses a custom processor which does not mask the input (which is already masked when running prediction).
        :param dicts: Masked samples to run prediction on provided as a list of dicts. One dict per sample.
        :type dicst: [dict]
        :return: dict of predictions

        """
        pred_processor = CharMLMPredProcessor(
            tokenizer=self.processor.tokenizer,
            max_seq_len=self.processor.max_seq_len,
            data_dir=self.processor.data_dir,
        )
        dataset, tensor_names = pred_processor.dataset_from_dicts(dicts)
        samples = []
        for dict in dicts:
            samples.extend(pred_processor._dict_to_samples(dict, dicts))

        data_loader = NamedDataLoader(
            dataset=dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            tensor_names=tensor_names,
        )

        preds_all = []
        for i, batch in enumerate(data_loader):
            batch = {key: batch[key].to(self.device) for key in batch}
            batch_samples = samples[i * self.batch_size : (i + 1) * self.batch_size]
            with torch.no_grad():
                logits = self.model.forward(**batch)
                preds = self.model.formatted_preds(
                    logits=logits,
                    label_maps=pred_processor.label_maps,
                    samples=batch_samples,
                    tokenizer=pred_processor.tokenizer,
                    **batch,
                )
                preds_all.append(preds)
        # flatten list
        preds_all = [
            p for outer_list in preds_all for pred_dict in outer_list for p in pred_dict
        ]
        return preds_all

    def predict_sequentially(self, dicts):
        """An experimental sequential decoder, with recursively decoders one character at a time. A very slow implementation best thought of as a proof of concept."""
        nb_of_sequences = len(dicts)
        nb_finished = 0
        predictions = self.predict(dicts=dicts)
        final_predictions = copy.deepcopy(predictions)
        while nb_finished < nb_of_sequences:
            nb_finished = 0
            new_masked_sequences = []
            for pred in predictions:
                masked_seq = pred["predictions"]["masked_text"]
                if masked_seq.find("#") == -1:
                    nb_finished += 1
                else:
                    first_predicted_char = pred["predictions"]["predictions"][0]
                    masked_seq = masked_seq.replace("#", first_predicted_char, 1)
                masked_seq = convert_masks(masked_seq)
                new_masked_sequences.append(masked_seq)
            dicts = sentences_to_dicts(new_masked_sequences)
            predictions = self.predict(dicts=dicts)
        # ensure that we output a well formated list of prediction dicts
        for i, (pred_dict, predicted_seq) in enumerate(
            zip(final_predictions, predictions)
        ):
            masked_seq = pred_dict["predictions"]["masked_text"]
            predicted_seq = predicted_seq["predictions"]["text_with_preds"]
            predicted_chars = [
                c1 for c1, c2 in zip(predicted_seq, masked_seq) if c1 != c2
            ]
            # insert the predictions into the masked text
            for p in predicted_chars:
                masked_seq = masked_seq.replace("#", f"[{p}]", 1)
            final_predictions[i]["predictions"]["text_with_preds"] = masked_seq.replace(
                "][", ""
            )
            final_predictions[i]["predictions"]["predictions"] = predicted_chars
        return final_predictions


def replace_square_brackets(sent):
    """Converts sentences where missing characters are indicated by full stops enclosed by square brackets to sentences where the missing characaters are indicated by hash symbols."""
    masked_sent = list(sent)
    matches = re.finditer(r"\[\.+\]", sent)
    for m in matches:
        # replace the dots
        for i in range(m.start() + 1, m.end() - 1):  # just replace the dots
            masked_sent[i] = "#"
        # remove the brackets
        masked_sent[m.start()] = ""
        masked_sent[m.end() - 1] = ""
    return "".join(masked_sent)


def convert_masks(seq):
    """Converts # masking to the [MASK] symbol used by BERT."""
    seq = list(seq)
    for i, c in enumerate(seq):
        if c == "#":
            seq[i] = "[MASK]"
    return "".join(seq)


def sentences_to_dicts(sentences):
    "Packs sentences into dicts for prediction."
    dicts = []
    for sent in sentences:
        d = {"doc": [sent, "_"]}
        dicts.append(d)
    return dicts
