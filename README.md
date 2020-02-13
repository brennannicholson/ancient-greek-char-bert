# Ancient-Greek-Char-Bert

This repository contains two character-level BERT models trained to predict missing characters in Ancient Greek texts, as well as the scripts need to use them and the dataset and code used to train them. 

The motivation behind the project was to create models which can help experts restore damaged Ancient Greek texts.

The code was developed as part of a Bachelor thesis at the University of Leipzig and is based on [FARM](https://github.com/deepset-ai/FARM).

Note that while a GPU isn't require to use the models (or train new ones) everything will run very slowly if a GPU isn't available. 

The models mentioned above are not directly included in the repo (they are too large) but are being hosted by [Deepset](https://deepset.ai/) and are downloaded as part of the setup process detailed below.

## Structure

The repository has three main parts:

1. `data_prep`: this folder contains the scripts used to create the dataset used to train the models
2. `char_bert`: this folder has all the components need to train, finetune, use and evaluate new or existing models
3. `models`: the two pretrained models are located here: 
    * `greek_char_BERT` was trained on the datasest based on [Perseus](https://github.com/PerseusDL/canonical-greekLit) and [First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek) (which the files in `data_prep` create)
    * `finetuned_pythia_greek_char_BERT` is a version of the previous model which has been finetuned on the dataset used to train [Pythia](https://github.com/sommerschield/ancient-text-restoration/)

Once the dataset has been generated it resides in the `data` folder.

## Installation 

The following has been tested on Ubuntu 18.04, it should work on similar systems.

Clone this repo:
```
git clone https://git.informatik.uni-leipzig.de/bn53lody/ancient-greek-char-bert.git
```

Go to the directory and set up a virtualenv:

```
cd ancient-greek-char-bert
virtualenv .venv
source .venv/bin/activate
```

Install the data preparation code:

```
cd data_prep
pip3 install -e ./
pip3 install -r requirements.txt
```

Generate the dataset (this might take a while):
```
cd greek_data_prep
python3 prepare_dataset.py
```

Once that's done or while it going on (don't forget to activate the virtualenv if you open a new terminal window) setup the FARM repo:
```
cd ../..
./setup_farm.sh
```

The install the `char_bert` code:
```
cd char_bert
pip3 install -e ./
```

Download the models:
```
cd ..
./download_models.sh
```

Now everything should be ready.

## Use

### Prediction

To use one of the existing models to predict missing characters, first the text with missing characters has to be in the correct format. The text should be in a separate file, with one text per line (if there are more than one). Missing characters should be indicated by full stops (one per character) and enclosed by square brackets. For example:

```
μῆνιν ἄ[...]ε θεὰ Πηληϊάδεω Ἀχ[...]ος
```

One the texts are in the correct format, the script `run_prediction.py` can be used to predict the missing characters. The script has various options which can be passed to it. Invoke it with `-h` to see the options:

```
python3 run_prediction.py -h
```

Note that sequential decoding `-s` can be very slow, especially without a GPU. Alignment `-a` is best used with text wrapping off.

If you'd like to, for instance, use the `greek_char_BERT` model to predict missing characters in a text located in `data/prediction_test.txt` using sequential decoding, this can be done with (if you are in the `greek_char_bert` folder):

```
python3 run_prediction.py -f ../../data/prediction_test.txt -m ../../models/greek_char_BERT -s
``` 

The output with appear directly on the command line.

### Training

If you'd like to train a new model from scratch (perhaps for another language) you'll first need to set the parameters in the `train.py` script. One everything there has been set up, it's simply a matter of invoking `train.py`.

```
python3 train.py
```

If you'd like to finetune an existing model, the `load_dir` variable within the script will need to be set to the model you'd like to finetune and then the `data_dir` variable changed to point to the new dataset. Then call the script with the `-f` flag.

```
python3 train.py -f
```

## Evaluation

An evaluation script `run_eval.py` is provided but the evaluation datasets (which are quite large) have not been supplied. However, the report and the examples of correct and incorrect sentences which the script generates have been included for each of the models within their folders. Note that, should you want to use the script, the model and decoder have to be set directly within the script.
