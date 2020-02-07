# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from farm.modeling.tokenization import BertTokenizer, BasicTokenizer, _words_to_tokens
from pytorch_transformers.tokenization_bert import whitespace_tokenize


class CharMLMBasicTokenizer(BasicTokenizer):
    """This is identical to the superclass (located at farm/modeling/tokenization.py). It simply overrides tokenize from the superclass's method (itself inherited) to to stop it from stripping accents and lower-casing the tokens"""

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        never_split = self.never_split + (
            never_split if never_split is not None else []
        )
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class CharMLMTokenizer(BertTokenizer):
    """This simply replaces the BasicTokenizer with the MLMBasicTokenizer and changes the tokenize method to simply return a list of chars, otherwise it is identical to the superclass (located at farm/modeling/tokenization.py)"""

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        """Init the tokenizer with the MLMBasicTokenizer."""
        super().__init__(
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
        )
        if do_basic_tokenize:
            self.basic_tokenizer = CharMLMBasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
            )

    def _tokenize(self, text):
        """Simply tokenize by return all the characters in a list. The simplest possible tokenization."""
        split_tokens = list(text)
        return split_tokens


def tokenize_with_metadata(text, tokenizer, max_seq_len):
    """This is a very slightly modified copy of tokenize_with_metadata from farm/modeling/tokenization.py. - BN"""
    # split text into "words" (here: simple whitespace tokenizer)
    words = text.split(" ")
    word_offsets = []
    # the following line has been changed to set the default to 1 instead of 0 - BN
    cumulated = 1
    for idx, word in enumerate(words):
        word_offsets.append(cumulated)
        cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

    # split "words"into "subword tokens"
    tokens, offsets, start_of_word = _words_to_tokens(
        words, word_offsets, tokenizer, max_seq_len
    )

    tokenized = {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}
    return tokenized
