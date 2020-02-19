from farm.data_handler.samples import Sample
from char_bert.greek_char_bert.data_handler.utils import get_sentence_pair_with_placeholder
from char_bert.greek_char_bert.data_handler.tokenization import tokenize_with_metadata
from tqdm import tqdm


def create_char_mlm_prediction_samples_sentence_pairs(baskets, tokenizer, max_seq_len):
    """A modified version of create_samples_sentence_pairs from farm/data_handlers/samples.py which simply assigns the first text as text_a and the second text as text_b. This only works becauses the docs contain a sentence to be predicted and a placeholder as the second text."""
    for basket in tqdm(baskets):
        doc = basket.raw["doc"]
        basket.samples = []
        id = "%s" % (basket.id)
        text_a = doc[0]
        text_b = doc[1]
        is_next_label = 1
        sample_in_clear_text = {
            "text_a": text_a,
            "text_b": text_b,
            "is_next_label": is_next_label,
        }
        tokenized = {}
        tokenized["text_a"] = tokenize_with_metadata(text_a, tokenizer, max_seq_len)
        tokenized["text_b"] = tokenize_with_metadata(text_b, tokenizer, max_seq_len)
        basket.samples.append(
            Sample(id=id, clear_text=sample_in_clear_text, tokenized=tokenized)
        )
    return baskets


def create_samples_sentence_pairs_using_placeholder(baskets, tokenizer, max_seq_len):
    """A modified version of create_samples_sentence_pairs from farm/data_handlers/samples.py which calls a modified version of get_sentence_pair which just fetches a placeholder for the second sentence."""
    # TODO why not just use create_char_mlm_prediction_samples_sentence_pairs? Check if it makes a difference.
    for basket in tqdm(baskets):
        doc = basket.raw["doc"]
        basket.samples = []
        for idx in range(len(doc) - 1):
            id = "%s-%s" % (basket.id, idx)
            text_a, text_b, is_next_label = get_sentence_pair_with_placeholder(doc, idx)
            sample_in_clear_text = {
                "text_a": text_a,
                "text_b": text_b,
                "is_next_label": is_next_label,
            }
            tokenized = {}
            tokenized["text_a"] = tokenize_with_metadata(text_a, tokenizer, max_seq_len)
            tokenized["text_b"] = tokenize_with_metadata(text_b, tokenizer, max_seq_len)
            basket.samples.append(
                Sample(id=id, clear_text=sample_in_clear_text, tokenized=tokenized)
            )
    return baskets
