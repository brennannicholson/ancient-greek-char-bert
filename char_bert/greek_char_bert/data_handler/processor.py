"""The three data processors used for CharMLMs. CharMLMProcessor is for training, it masks the input; CharMLMPredProcessor is for prediction, where the input is already masked. PremaskedCharMLMProcessor is for training on a premasked dataset where the answers are supplied alongside the data."""
from farm.data_handler.processor import BertStyleLMProcessor, TOKENIZER_MAP
import random
import logging

from farm.data_handler.samples import Sample
from farm.modeling.tokenization import tokenize_with_metadata
from char_bert.greek_char_bert.data_handler.input_features import (
    samples_to_features_bert_char_mlm,
    premasked_samples_to_features_bert_char_mlm,
    premasked_samples_with_answers_to_features_bert_char_mlm,
)
from char_bert.greek_char_bert.data_handler.samples import (
    create_char_mlm_prediction_samples_sentence_pairs,
    create_samples_sentence_pairs_using_placeholder,
)
from char_bert.greek_char_bert.data_handler.utils import read_docs_from_txt
from char_bert.greek_char_bert.data_handler.tokenization import CharMLMTokenizer

logger = logging.getLogger(__name__)

# add the MLMTokenizer to the map
TOKENIZER_MAP["CharMLMTokenizer"] = CharMLMTokenizer


class CharMLMProcessor(BertStyleLMProcessor):
    """Prepares data for a CharMLM."""

    def _log_samples(self, n_samples):
        """This is a modified version of Processor._log_samples from farm/data_handler/processor.py. It works around a bug where some baskets are not initialized correctly. -BN"""
        # TODO check whether this bug still occurs.
        logger.info("*** Show {} random examples ***".format(n_samples))
        if self.baskets:
            for i in range(n_samples):
                while True:
                    random_basket = random.choice(self.baskets)
                    if random_basket.samples:
                        break
                random_sample = random.choice(random_basket.samples)
                logger.info(random_sample)

    def _init_samples_in_baskets(self):
        """This function is a copy of Processor._init_samples_in_baskets from farm/data_handler/processor.py except that it calls a modified version of create_sample_sentence_pairs - BN"""
        self.baskets = create_samples_sentence_pairs_using_placeholder(
            self.baskets, self.tokenizer, self.max_seq_len
        )

    @classmethod
    def _dict_to_samples(cls, dict, all_dicts=None):
        """
        Converts a dict with a document to a sample (which will subsequently be featurized). It is used during prediction.
        
        This is a modified version of BertStyleLMProcessor._dict_to_samples from farm/data_handler/processor.py. It has been modified to create samples with just a single text, rather than two, as is the case for a normal BERT model.
        """
        doc = dict["doc"]
        samples = []
        for idx in range(len(doc) - 1):
            tokenized = {}
            tokenized["text_a"] = tokenize_with_metadata(
                doc[idx], cls.tokenizer, cls.max_seq_len
            )
            samples.append(
                Sample(id=None, clear_text={"doc": doc[idx]}, tokenized=tokenized)
            )
        return samples

    def _file_to_dicts(self, file: str) -> list:
        """This function is a copy of BertStyleLMProcessor._file_to_dicts from farm/data_handler/processor.py except that it calls a modified version of read_docs_from_txt - BN"""
        dicts = read_docs_from_txt(filename=file, delimiter=self.delimiter)
        return dicts

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        """This function is a copy of the function of the same name in farm/data_handler/input_features.py. It has been modifed to call a modified featurization function. -BN"""
        features = samples_to_features_bert_char_mlm(
            sample=sample, max_seq_len=cls.max_seq_len, tokenizer=cls.tokenizer
        )
        return features


class CharMLMPredProcessor(CharMLMProcessor):
    """A modified processor for predictions. It modifies _sample_to_features to not mask the input sequences."""

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        """This function is a copy of the function of the same name in farm/data_handler/input_features.py. It has been modifed to call a modified featurization function. -BN"""
        features = premasked_samples_to_features_bert_char_mlm(
            sample=sample, max_seq_len=cls.max_seq_len, tokenizer=cls.tokenizer
        )
        return features

    def _init_samples_in_baskets(self):
        """This function is a copy of Processor._init_samples_in_baskets from farm/data_handler/processor.py except that it calls a modified version of create_sample_sentence_pairs which does not randomly assign a second sentence - BN"""
        self.baskets = create_char_mlm_prediction_samples_sentence_pairs(
            self.baskets, self.tokenizer, self.max_seq_len
        )


class PremaskedCharMLMProcessor(CharMLMProcessor):
    def _init_samples_in_baskets(self):
        """This function is a copy of Processor._init_samples_in_baskets from farm/data_handler/processor.py except that it calls a modified version of create_sample_sentence_pairs - BN"""
        # TODO what are the advantages of this over create_mlm_prediction_samples_sentence_pairs
        self.baskets = create_samples_sentence_pairs_using_placeholder(
            self.baskets, self.tokenizer, self.max_seq_len
        )

    @classmethod
    def _sample_to_features(cls, sample) -> dict:
        """This function is a copy of the function of the same name in farm/data_handler/input_features.py. It has been modifed to call a modified featurization function. -BN"""
        features = premasked_samples_with_answers_to_features_bert_char_mlm(
            sample=sample, max_seq_len=cls.max_seq_len, tokenizer=cls.tokenizer
        )
        return features
