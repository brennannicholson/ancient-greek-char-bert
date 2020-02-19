"""Modified utility functions for use with the CharMLM."""
from farm.data_handler.utils import truncate_seq_pair
from char_bert.greek_char_bert.data_handler.utils import char_mlm_mask_random_words


def remove_unknown_chars(tokens, tokenizer):
    """Replaces unknown characters with a special [UNK] token."""
    for i, t in enumerate(tokens):
        try:
            tokenizer.vocab[t]
        except KeyError:
            tokens[i] = "[UNK]"
    return tokens


# TODO these three samples_to_features variants can probably be combined and a parameter passed to select what functionality is needed

# TODO remove the second seq entirely instead of using a placeholder. We only want to run prediction on individual sequences, not pairs. Alternately, ensure the code doesn't assume a placeholder is used so that the next sequence prediction task can be used with future character based models.


def samples_to_features_bert_char_mlm(sample, max_seq_len, tokenizer):
    """
    This method is a copy of samples_to_features_bert_lm from farm/data_handler/input_features.py. It has been modified to use a custom masking algorithm. -BN

    Original docs:

    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = sample.tokenized["text_a"]["tokens"]
    tokens_b = sample.tokenized["text_b"]["tokens"]
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

    tokens_a, t1_label = char_mlm_mask_random_words(tokens_a)
    # TODO don't make a function call here, tokens_b should always just be '_'
    tokens_b, t2_label = char_mlm_mask_random_words(tokens_b)
    # convert lm labels to ids
    t1_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t1_label]
    t2_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t2_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label_ids + [-1] + t2_label_ids + [-1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        padding_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
    if sample.clear_text["is_next_label"]:
        is_next_label_id = [0]
    else:
        is_next_label_id = [1]

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
        "label_ids": is_next_label_id,
    }

    return [feature_dict]


def premasked_samples_to_features_bert_char_mlm(sample, max_seq_len, tokenizer):
    """
    This method is a copy of samples_to_features_bert_lm from farm/data_handler/input_features.py. It has been modified to not mask the samples. -BN

    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = sample.tokenized["text_a"]["tokens"]
    tokens_b = sample.tokenized["text_b"]["tokens"]
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

    # change unknown tokens to the [UNK] token
    tokens_a = remove_unknown_chars(tokens_a, tokenizer)
    tokens_b = remove_unknown_chars(tokens_b, tokenizer)

    # usually t1_label and t2_label would contain the original unmasked tokens, as the tokens are already masked when running prediction, they labels here are simple copies of the original tokens.
    t1_label = tokens_a.copy()
    t2_label = tokens_b.copy()

    # convert lm labels to ids
    t1_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t1_label]
    t2_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t2_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label_ids + [-1] + t2_label_ids + [-1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        padding_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
    if sample.clear_text["is_next_label"]:
        is_next_label_id = [0]
    else:
        is_next_label_id = [1]

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
        "label_ids": is_next_label_id,
    }

    return [feature_dict]


def premasked_samples_with_answers_to_features_bert_char_mlm(
    sample, max_seq_len, tokenizer
):
    """
    This method is a copy of samples_to_features_bert_lm from farm/data_handler/input_features.py. It has been modified to not mask the samples, but simply convert the existing masking. It expects the sample texts to consist of the text then the answers separated by a tab. This is only used when the text has been masked by an external masking algorithm. -BN

    Original documentation:

    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :param max_seq_len: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = sample.tokenized["text_a"]["tokens"]
    tokens_b = sample.tokenized["text_b"]["tokens"]

    seq_and_ans = "".join(tokens_a).split("\t")
    tokens_a = seq_and_ans[0]
    ans = seq_and_ans[1]

    # usually t1_label and t2_label would contain the original unmasked tokens, here to have to construct t1 from the answers, t2 is just a copy of the placeholder
    t1_label = tokens_a
    t2_label = tokens_b.copy()

    # construct t1_label
    for c in ans:
        t1_label = t1_label.replace("#", c, 1)

    # here we're effectively retokenizing...
    tokens_a = list(tokens_a)
    t1_label = list(t1_label)

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)

    # convert masking
    conversions = 0
    for i, t in enumerate(tokens_a):
        if t == "#":
            tokens_a[i] = "[MASK]"
            conversions += 1
    assert conversions == len(ans)

    # remove unknown tokens
    tokens_a = remove_unknown_chars(tokens_a, tokenizer)
    t1_label = remove_unknown_chars(t1_label, tokenizer)

    # convert lm labels to ids
    t1_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t1_label]
    t2_label_ids = [-1 if tok == "" else tokenizer.vocab[tok] for tok in t2_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label_ids + [-1] + t2_label_ids + [-1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        padding_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
    if sample.clear_text["is_next_label"]:
        is_next_label_id = [0]
    else:
        is_next_label_id = [1]

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(lm_label_ids) == max_seq_len

    feature_dict = {
        "input_ids": input_ids,
        "padding_mask": padding_mask,
        "segment_ids": segment_ids,
        "lm_label_ids": lm_label_ids,
        "label_ids": is_next_label_id,
    }

    return [feature_dict]
