import logging
from farm.modeling.prediction_head import PredictionHead, BertLMHead

logger = logging.getLogger(__name__)


class CharMLMHead(BertLMHead):
    """A prediction head for CharMLM. It handels transforming the raw model output logits into human-readable predictions."""

    def logits_to_preds(self, logits, label_map, input_ids, **kwargs):
        """Converts the raw output logits into a list of predicted characters.

        This method is a modified version of BertLMHead.logits_to_loss from farm/modeling/prediction_head.py. It extracts the logits for the masked tokens by checking the indices rather than using lm_label_ids.
        """
        logits = logits.cpu().numpy()
        input_ids = input_ids.cpu().numpy()
        lm_preds_ids = logits.argmax(2)
        # apply mask to get rid of predictions for non-masked tokens
        assert lm_preds_ids.shape == input_ids.shape
        char_indices = dict([(v, k) for k, v in label_map.items()])
        mask_index = char_indices["[MASK]"]
        lm_preds_ids[input_ids != mask_index] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append(
                [label_map[int(x)] for x in pred_ids_for_sequence if int(x) != -1]
            )
        return preds

    def prepare_labels(self, label_map, lm_label_ids, **kwargs):
        """Returns a list of the ids of characters which were originally masked. Based on BertLMHead.prepare_labels() (located at farm/modeling/prediction_head.py)."""
        label_ids = lm_label_ids.cpu().numpy().tolist()
        input_ids = kwargs["input_ids"]
        input_ids = list(input_ids.cpu().numpy())
        assert len(label_ids) == len(input_ids)
        assert len(label_ids[0]) == len(input_ids[0])
        labels = []
        char_indices = dict([(v, k) for k, v in label_map.items()])
        mask_index = char_indices["[MASK]"]
        # we have a batch of sequences here. we need to convert for each token in each sequence.
        for label_ids_for_sequence, input_ids_for_sequence in zip(label_ids, input_ids):
            labels.append(
                [
                    label_map[int(x)]
                    for x, y in zip(label_ids_for_sequence, input_ids_for_sequence)
                    if int(y) == mask_index
                ]
            )
        return labels

    def replace_unkown_tokens(self, masked_seq, original_text):
        """Swaps the [UNK] tokens out for the characters which originally stood there."""
        masked_seq = list(masked_seq)
        # the [MASK] tokens need to be replaces so that there are an equivalent number of characters in the original and masked sentences
        original_text = list(original_text.replace("[MASK]", "#"))
        for i, (co, cm) in enumerate(zip(original_text, masked_seq)):
            if cm == "&":
                masked_seq[i] = co
        return "".join(masked_seq)

    def tokens_as_text(
        self, original_text, input_ids, lm_label_ids, padding_mask, label_map
    ):
        """Devectorizes the input to reconstruct the masked sentence. Ideally the masked sentence should be saved along side the sample."""
        text = ""
        has_unknowns = False
        for input_id_tensor, padding_mask_tensor, lm_label_ids_tensor in zip(
            input_ids, padding_mask, lm_label_ids
        ):
            input_id = input_id_tensor.item()
            padding_mask_value = padding_mask_tensor.item()
            token = label_map[input_id]
            if token == "[CLS]" or token == "[SEP]" or padding_mask_value == 0:
                continue
            elif token == "[MASK]":
                text += "#"
            elif token == "[UNK]":
                text += "&"
                has_unknowns = True
            else:
                text += token
        if has_unknowns:
            text = self.replace_unkown_tokens(text, original_text)
        return text

    def insert_preds_into_text(self, text, preds):
        """Inserts the predicted characters into the original text, enclosing them in square brackets."""
        for p in preds:
            # replace the first masked position with the next prediction (not an efficient implementation)
            text = text.replace("#", f'[{p.strip("#")}]', 1)
        return text.replace("][", "")

    def formatted_preds(self, logits, label_map, samples, **kwargs):
        """Take the raw logits and produce json output containing the original text, the text with predictions, the masked text and the predicted characters."""
        input_ids = kwargs["input_ids"]
        lm_label_ids = kwargs["lm_label_ids"]
        padding_mask = kwargs["padding_mask"]
        preds = self.logits_to_preds(logits, label_map, input_ids)
        res = []
        for (
            sample,
            sample_preds,
            sample_input_ids,
            sample_lm_labels,
            sample_padding_mask,
        ) in zip(samples, preds, input_ids, lm_label_ids, padding_mask):
            original_text = "".join(sample.clear_text["doc"])
            masked_text = self.tokens_as_text(
                original_text,
                sample_input_ids,
                sample_lm_labels,
                sample_padding_mask,
                label_map,
            )
            text_with_preds = self.insert_preds_into_text(masked_text, sample_preds)
            res.append(
                {
                    "task": "mlm",
                    "predictions": {
                        "original_text": original_text,
                        "text_with_preds": text_with_preds,
                        "masked_text": masked_text,
                        "predictions": sample_preds,
                    },
                }
            )
        return res

    @classmethod
    def load(cls, config_file):
        """
        Loads a Prediction Head. Directly calls PredictionHead.load() as we don't want to use the superclass's load method.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        return PredictionHead.load(config_file)
