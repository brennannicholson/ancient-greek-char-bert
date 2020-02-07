"""Creates the Ancient_Greek_ML dataset and then prepares the train, dev and test sets for the character-level BERT."""
from greek_data_prep.download_data import get_data
from greek_data_prep.clean_data import clean_data
from greek_data_prep.sentence_tokenization import sentence_tokenize_corpus
from greek_data_prep.filter_sentences import filter_sentences
from greek_data_prep.generate_char_vocab import create_vocab
from greek_data_prep.split_data import ninty_eight_one_one_spilt
from greek_data_prep.convert_data_to_bert_format import convert_data
import os

if __name__ == "__main__":
    # set up
    data_path = "../../data"
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    os.chdir(data_path)
    # create the Ancient_Greek_ML dataset
    get_data()
    clean_data()
    sentence_tokenize_corpus()
    # create the specific train, dev and test sets used to train the Ancient Greek character-level BERT
    filter_sentences()
    create_vocab()
    ninty_eight_one_one_spilt()
    convert_data()
    # clean up intermediate files
    for f in ["char_BERT_dataset.txt", "First1KGreek-1.1.4529.zip"]:
        os.remove(f)
