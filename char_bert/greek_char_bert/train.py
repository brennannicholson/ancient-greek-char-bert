"""Trains a CharMLM from scratch or finetunes an existing model. Can also be used to resume training on the same datasest."""
from char_bert.greek_char_bert.data_handler.tokenization import CharMLMTokenizer
from char_bert.greek_char_bert.data_handler.processor import CharMLMProcessor
from farm.data_handler.data_silo import DataSilo
from farm.modeling.language_model import BertModel
from farm.eval import Evaluator
from char_bert.greek_char_bert.modelling.language_model import PretrainingBERT
from char_bert.greek_char_bert.modelling.prediction_head import CharMLMHead
from char_bert.greek_char_bert.modelling.adaptive_model import CharMLMAdaptiveModel
from farm.experiment import initialize_optimizer
from farm.train import Trainer
from pytorch_transformers.modeling_bert import BertConfig
from farm.utils import set_all_seeds, initialize_device_settings
from datetime import datetime
import logging
from shutil import copyfile
import pathlib
import argparse


def setup_evaluator(dataset_name, data_silo, device):
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader(dataset_name),
        label_maps=data_silo.processor.label_maps,
        device=device,
        metrics=data_silo.processor.metrics,
        classification_report=False,
    )
    return evaluator


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a CharMLM from scatch or finetune an existing model."
    )
    parser.add_argument(
        "-f",
        "--finetune",
        default=False,
        action="store_true",
        help="Load an existing model (specified in the load_dir variable within the script).",
    )
    args = parser.parse_args()

    finetune = args.finetune

    model_name = "test_ancient_greek_char_MLM"
    load_dir = "../../models/greek_char_BERT"

    device, n_gpu = initialize_device_settings(use_cuda=True)
    print("Devices available: {}".format(device))

    # logging setup:

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # set seeds
    set_all_seeds(seed=42)

    # save a copy of source code files before training

    save_dir = f"save/{model_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    for file in ["train"]:
        copyfile(f"{file}.py", f"{save_dir}/{file}.py")

    # tokenizer setup

    tokenizer = CharMLMTokenizer(
        vocab_file="../../data/greek_char_vocab.txt", do_lower_case=False
    )

    # data handling setup

    data_dir = "../../data"

    if finetune:
        # load existing processor
        processor = CharMLMProcessor.load_from_dir(load_dir)
    else:
        # init new processor
        processor = CharMLMProcessor(
            tokenizer=tokenizer, max_seq_len=192, data_dir=data_dir
        )

    batch_size = 32

    data_silo = DataSilo(processor=processor, batch_size=batch_size)

    # model setup

    config = BertConfig(
        vocab_size_or_config_json_file=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    prediction_head = CharMLMHead(hidden_size=768, vocab_size=tokenizer.vocab_size)

    if finetune:
        # load an existing model
        model = CharMLMAdaptiveModel.load(load_dir, device)
    else:
        # initialize a new model
        internal_model = BertModel(config=config)
        language_model = PretrainingBERT(internal_model)
        language_model.language = "ancient-greek"

        embeds_dropout_prob = 0.1

        model = CharMLMAdaptiveModel(
            language_model=language_model,
            prediction_heads=[prediction_head],
            embeds_dropout_prob=embeds_dropout_prob,
            lm_output_types=["per_token"],
            device=device,
        )

    # evaluators setup

    evaluator_dev = setup_evaluator("dev", data_silo, device)
    evaluator_test = setup_evaluator("test", data_silo, device)

    # training

    learning_rate = 1e-4
    warmup_proportion = 0.1
    n_epochs = 1

    optimizer, warmup_linear = initialize_optimizer(
        model=model,
        learning_rate=learning_rate,
        warmup_proportion=warmup_proportion,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
    )

    trainer = Trainer(
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        warmup_linear=warmup_linear,
        device=device,
        evaluate_every=6000,
        evaluator_dev=evaluator_dev,
        evaluator_test=evaluator_test,
        grad_acc_steps=1,
    )

    model = trainer.train(model)

    # Save model

    model.save(save_dir)
    processor.save(save_dir)
