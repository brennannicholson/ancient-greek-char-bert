#! /bin/bash

curl https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/research/finetuned_pythia_greek_char_BERT.zip > finetuned_pythia_greek_char_BERT.zip
curl https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/research/greek_char_BERT.zip > greek_char_BERT.zip
unzip finetuned_pythia_greek_char_BERT.zip -d ./models/finetuned_pythia_greek_char_BERT/
unzip greek_char_BERT.zip -d ./models/greek_char_BERT/
rm finetuned_pythia_greek_char_BERT.zip greek_char_BERT.zip
